// Reference run on the vendored libllama (b11669): greedy decode of a token-id prompt.
//
//   ref_llama <model.gguf> <ids.txt> <n_predict> [--ctx N] [--dump-dir D --dump-call K] [--logits-bin F]
//
// ids.txt holds whitespace-separated token ids. Prints one JSON object on stdout:
//   {"tokens":[...],"top":[[id1,logit1,id2,logit2],...],"prefill_ms":..,"decode_ms":..}
// --dump-dir writes the last row of every "l_out-N", "result_norm" and "result_output" tensor of llama_decode
// call number K (0 = the prefill) as raw f32 files D/<name>.f32 (the tensors llama.cpp names with cb()).
// Build: see engine/tools/build_ref.sh. Always run it under the GPU lock.
#include "llama.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char * g_dump_dir = NULL;
static int g_call = 0, g_dump_call = 0;

static bool want(const char * n) {
    return strncmp(n, "l_out-", 6) == 0 || strncmp(n, "attn_norm-", 10) == 0 || strncmp(n, "ffn_inp-", 8) == 0
        || strncmp(n, "sa_out-", 7) == 0 || strncmp(n, "kqv_out-", 8) == 0 || strncmp(n, "Qcur-", 5) == 0 || strncmp(n, "Kcur-", 5) == 0
        || strcmp(n, "result_norm") == 0 || strcmp(n, "result_output") == 0 || strcmp(n, "embd") == 0 || strcmp(n, "inp_scaled") == 0;
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    if (!g_dump_dir || g_call != g_dump_call || !want(t->name)) return ask ? false : true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32) return true;
    size_t row = (size_t) t->ne[0] * (t->ne[1] > 0 ? 1 : 1);
    // keep the last "token" row: tensors are [width, n_rows(, n_head)]; dump the whole last slice of dim 1
    int64_t n_rows = t->ne[t->ne[2] > 1 ? 2 : 1];
    size_t slice = (size_t) ggml_nbytes(t) / (size_t) n_rows;
    (void) row;
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.f32", g_dump_dir, t->name);
    void * buf = malloc(slice);
    ggml_backend_tensor_get(t, buf, ggml_nbytes(t) - slice, slice);
    FILE * f = fopen(path, "wb");
    fwrite(buf, 1, slice, f);
    fclose(f);
    free(buf);
    return true;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: ref_llama model ids.txt n_predict [--ctx N] [--dump-dir D --dump-call K] [--logits-bin F]\n"); return 64; }
    const char * model_path = argv[1];
    int n_predict = atoi(argv[3]);
    int n_ctx = 4096;
    const char * logits_bin = NULL;
    for (int i = 4; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--ctx")) n_ctx = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--dump-dir")) g_dump_dir = argv[i + 1];
        if (!strcmp(argv[i], "--dump-call")) g_dump_call = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--logits-bin")) logits_bin = argv[i + 1];
    }
    FILE * f = fopen(argv[2], "r");
    if (!f) { perror("ids"); return 1; }
    llama_token * prompt = malloc(sizeof(llama_token) * 1000000);
    int n_prompt = 0, v;
    while (fscanf(f, "%d", &v) == 1) prompt[n_prompt++] = v;
    fclose(f);

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    struct llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.no_perf = true;
    if (g_dump_dir) { cp.cb_eval = eval_cb; cp.cb_eval_user_data = NULL; }
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 3; }
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    printf("{\"tokens\":[");
    double t0 = now_ms();
    // prefill in n_ubatch chunks, logits for the last prompt token only
    struct llama_batch b = llama_batch_init(512, 0, 1);
    int pos = 0;
    double prefill_ms = 0;
    float * logits = NULL;
    while (pos < n_prompt) {
        int n = n_prompt - pos < 512 ? n_prompt - pos : 512;
        b.n_tokens = n;
        for (int i = 0; i < n; i++) {
            b.token[i] = prompt[pos + i]; b.pos[i] = pos + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = (pos + i == n_prompt - 1);
        }
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 4; }
        g_call++;
        pos += n;
    }
    prefill_ms = now_ms() - t0;
    logits = llama_get_logits_ith(ctx, -1);
    static char top[1 << 20]; int tl = 0;
    double t1 = now_ms();
    llama_token cur = 0;
    for (int s = 0; s < n_predict; s++) {
        int a = 0, bb = -1;
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > logits[a]) { bb = a; a = i; } else if (bb < 0 || logits[i] > logits[bb]) bb = i;
        }
        if (bb < 0) bb = 1;
        tl += snprintf(top + tl, sizeof top - tl, "%s[%d,%.6f,%d,%.6f]", s ? "," : "", a, logits[a], bb, logits[bb]);
        if (logits_bin && s == 0) { FILE * lf = fopen(logits_bin, "wb"); fwrite(logits, 4, n_vocab, lf); fclose(lf); }
        cur = a;
        printf("%s%d", s ? "," : "", cur);
        if (s + 1 == n_predict) break;
        b.n_tokens = 1; b.token[0] = cur; b.pos[0] = n_prompt + s; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = true;
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 5; }
        g_call++;
        logits = llama_get_logits_ith(ctx, -1);
    }
    double decode_ms = now_ms() - t1;
    printf("],\"top\":[%s],\"prefill_ms\":%.3f,\"decode_ms\":%.3f,\"n_prompt\":%d}\n", top, prefill_ms, decode_ms, n_prompt);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
