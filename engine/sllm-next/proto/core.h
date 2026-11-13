// sllm-next: what the bench driver (next.m) and the engine library (sn.m) share: the GGUF reader, the weight conversions,
// the kernels' argument structs, the step encoder and the matmul dispatch.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

// ---- GGUF ---------------------------------------------------------------------------------------------------------
typedef struct { char name[128]; int nd; int64_t ne[4]; int type; uint64_t off; } tensor_t;
typedef struct {
    const uint8_t * base; size_t size; size_t data;
    tensor_t * t; int nt;
    char arch[32];
    int n_layer, n_embd, n_ff, n_head, n_kv, hd;
    int n_nextn;  // guess-head blocks after the model's own (nextn_predict_layers): blk.n_layer .., never part of the main pass
    float eps, rope_base;
    // Gemma 3 / 4
    int hd_swa, window, shared_kv, ple, ffa[128], swa[128], kva[128], n_ffa, n_swa, n_kva;
    float rope_base_swa, softcap, rope_scale;
    float rope_attn; int ctx_orig;  // Phi-3: rope.scaling.attn_factor, rope.scaling.original_context_length
    int ssm_conv, ssm_state, ssm_groups, ssm_rank, ssm_inner, fa_interval, n_rot;  // Qwen3.5: ssm.*, full_attention_interval, rope.dimension_count
    int n_expert, n_expert_used, n_ff_exp, n_ff_shexp;  // mixture of experts
    float yarn_bf, yarn_bs;  // gpt-oss: rope.scaling.yarn_beta_fast / slow
} gguf_t;

static const uint8_t * P;
static uint64_t rd_u64(void) { uint64_t v; memcpy(&v, P, 8); P += 8; return v; }
static uint32_t rd_u32(void) { uint32_t v; memcpy(&v, P, 4); P += 4; return v; }
static void rd_str(char * out, size_t cap) {
    uint64_t n = rd_u64();
    if (out) { size_t m = n < cap - 1 ? n : cap - 1; memcpy(out, P, m); out[m] = 0; }
    P += n;
}
static const size_t TSZ[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

// a number of GGUF type ty (8 / 16-bit ones and the rest apart), P past it
static double rd_num_narrow(uint32_t ty) {
    switch (ty) {
        case 0: return *(uint8_t *) P;   case 1: return *(int8_t *) P;
        case 2: return *(uint16_t *) P;  case 3: return *(int16_t *) P;
        case 7: return *(uint8_t *) P;
    }
    return 0;
}
static double rd_num_wide(uint32_t ty) {
    switch (ty) {
        case 4: return *(uint32_t *) P;  case 5: return *(int32_t *) P;
        case 6: return *(float *) P;
        case 10: return (double) *(uint64_t *) P; case 11: return (double) *(int64_t *) P;
        case 12: return *(double *) P;
    }
    return 0;
}
static double rd_num(uint32_t ty) {
    const double v = ty <= 3 || ty == 7 ? rd_num_narrow(ty) : rd_num_wide(ty);
    P += TSZ[ty];
    return v;
}
static void skip_val(uint32_t ty) {
    if (ty == 8) { rd_str(NULL, 0); return; }
    if (ty == 9) {
        uint32_t at = rd_u32(); uint64_t n = rd_u64();
        if (at == 8) { for (uint64_t i = 0; i < n; ++i) rd_str(NULL, 0); }
        else P += n * TSZ[at];
        return;
    }
    P += TSZ[ty];
}

// the model's scalar keys (after "<arch>."): where each goes in gguf_t, an int or (real) a float
typedef struct { const char * key; size_t off; bool real; } gguf_key_t;
static const gguf_key_t GGUF_KEYS[] = {
    {"block_count", offsetof(gguf_t, n_layer), false},
    {"embedding_length", offsetof(gguf_t, n_embd), false},
    {"feed_forward_length", offsetof(gguf_t, n_ff), false},
    {"attention.head_count", offsetof(gguf_t, n_head), false},
    {"attention.head_count_kv", offsetof(gguf_t, n_kv), false},
    {"attention.key_length", offsetof(gguf_t, hd), false},
    {"attention.layer_norm_rms_epsilon", offsetof(gguf_t, eps), true},
    {"rope.freq_base", offsetof(gguf_t, rope_base), true},
    {"rope.freq_base_swa", offsetof(gguf_t, rope_base_swa), true},
    {"attention.key_length_swa", offsetof(gguf_t, hd_swa), false},
    {"attention.sliding_window", offsetof(gguf_t, window), false},
    {"attention.shared_kv_layers", offsetof(gguf_t, shared_kv), false},
    {"embedding_length_per_layer_input", offsetof(gguf_t, ple), false},
    {"final_logit_softcapping", offsetof(gguf_t, softcap), true},
    {"rope.scaling.factor", offsetof(gguf_t, rope_scale), true},
    {"rope.scaling.attn_factor", offsetof(gguf_t, rope_attn), true},
    {"rope.scaling.original_context_length", offsetof(gguf_t, ctx_orig), false},
    {"ssm.conv_kernel", offsetof(gguf_t, ssm_conv), false},
    {"ssm.state_size", offsetof(gguf_t, ssm_state), false},
    {"ssm.group_count", offsetof(gguf_t, ssm_groups), false},
    {"ssm.time_step_rank", offsetof(gguf_t, ssm_rank), false},
    {"ssm.inner_size", offsetof(gguf_t, ssm_inner), false},
    {"full_attention_interval", offsetof(gguf_t, fa_interval), false},
    {"rope.dimension_count", offsetof(gguf_t, n_rot), false},
    {"expert_count", offsetof(gguf_t, n_expert), false},
    {"expert_used_count", offsetof(gguf_t, n_expert_used), false},
    {"expert_feed_forward_length", offsetof(gguf_t, n_ff_exp), false},
    {"expert_shared_feed_forward_length", offsetof(gguf_t, n_ff_shexp), false},
    {"rope.scaling.yarn_beta_fast", offsetof(gguf_t, yarn_bf), true},
    {"rope.scaling.yarn_beta_slow", offsetof(gguf_t, yarn_bs), true},
    {"nextn_predict_layers", offsetof(gguf_t, n_nextn), false},
};

// scalar key k (its "<arch>." prefix off) of value v into g; a key not in GGUF_KEYS is dropped
static void gguf_scalar(gguf_t * g, const char * k, double v) {
    for (size_t i = 0; i < sizeof GGUF_KEYS / sizeof GGUF_KEYS[0]; ++i) {
        if (strcmp(k, GGUF_KEYS[i].key)) continue;
        if (GGUF_KEYS[i].real) *(float *) ((char *) g + GGUF_KEYS[i].off) = (float) v;
        else *(int *) ((char *) g + GGUF_KEYS[i].off) = (int) v;
        return;
    }
}

// a per-layer array (Gemma 4: feed_forward_length, attention.sliding_window_pattern, attention.head_count_kv) into its
// table and count; false, P where it was, for any other array
static bool gguf_layer_array(gguf_t * g, const char * k) {
    const uint8_t * save = P;
    uint32_t at = rd_u32(); uint64_t n = rd_u64();
    int * dst = !strcmp(k, "feed_forward_length") ? g->ffa : !strcmp(k, "attention.sliding_window_pattern") ? g->swa
              : !strcmp(k, "attention.head_count_kv") ? g->kva : NULL;
    if (dst && at != 8 && n <= 128) {
        for (uint64_t j = 0; j < n; ++j) dst[j] = (int) rd_num(at);
        *(dst == g->ffa ? &g->n_ffa : dst == g->swa ? &g->n_swa : &g->n_kva) = (int) n;
        return true;
    }
    P = save;
    return false;
}

// general.architecture (a string) and general.alignment: true when key is one of them (its value read)
static bool gguf_general(gguf_t * g, const char * key, uint32_t ty, uint32_t * align) {
    if (!strcmp(key, "general.architecture") && ty == 8) { rd_str(g->arch, sizeof g->arch); return true; }
    if (!strcmp(key, "general.alignment")) { *align = (uint32_t) rd_num(ty); return true; }
    return false;
}

// one key of the header: the general ones, the model's per-layer arrays and scalars ("<arch>.*"); any other skipped
static void gguf_key(gguf_t * g, uint32_t * align) {
    char key[256], pre[64];
    rd_str(key, sizeof key);
    uint32_t ty = rd_u32();
    if (gguf_general(g, key, ty, align)) return;
    snprintf(pre, sizeof pre, "%s.", g->arch);
    snprintf(pre, sizeof pre, "%s.", g->arch);
    const bool mine = g->arch[0] && !strncmp(key, pre, strlen(pre));
    if (mine && ty == 9 && gguf_layer_array(g, key + strlen(pre))) return;  // per-layer arrays (Gemma 4)
    if (mine && ty != 8 && ty != 9) { gguf_scalar(g, key + strlen(pre), rd_num(ty)); return; }
    skip_val(ty);
}

// the tensor table (nt entries) at P
static void gguf_tensors(gguf_t * g, uint64_t nt) {
    g->nt = (int) nt;
    g->t = calloc(nt, sizeof(tensor_t));
    for (uint64_t i = 0; i < nt; ++i) {
        tensor_t * t = &g->t[i];
        rd_str(t->name, sizeof t->name);
        t->nd = (int) rd_u32();
        for (int d = 0; d < 4; ++d) t->ne[d] = 1;
        for (int d = 0; d < t->nd; ++d) t->ne[d] = (int64_t) rd_u64();
        t->type = (int) rd_u32();
        t->off = rd_u64();
    }
}

static void gguf_open(const char * path, gguf_t * g) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st);
    size_t page = getpagesize(), len = (st.st_size + page - 1) / page * page;
    g->base = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
    g->size = len;
    P = g->base;
    if (memcmp(P, "GGUF", 4)) { fprintf(stderr, "not a GGUF\n"); exit(1); }
    P += 4; rd_u32();
    uint64_t nt = rd_u64(), nk = rd_u64();
    uint32_t align = 32;
    for (uint64_t i = 0; i < nk; ++i) gguf_key(g, &align);
    // block_count counts the guess-head blocks too: the model's own layers are the ones before them
    if (g->n_nextn > 0 && g->n_nextn < g->n_layer) g->n_layer -= g->n_nextn; else g->n_nextn = 0;
    gguf_tensors(g, nt);
    size_t pos = P - g->base;
    g->data = (pos + align - 1) / align * align;
}

static tensor_t * find(gguf_t * g, const char * name) {
    for (int i = 0; i < g->nt; ++i) if (!strcmp(g->t[i].name, name)) return &g->t[i];
    fprintf(stderr, "missing tensor %s\n", name);
    exit(1);
}

// fp16 bits as a float
static float h2f(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float) x; }

// ---- kernel argument structs (same layout as kernels.metal) ---------------------------------------------------------
typedef struct { int K, M, R, ldx, ldy, acc; } mm_args;
typedef struct { int n; } n_args;
typedef struct { int n; float eps; int ldx, ldy; } norm_args;
typedef struct { int n_head, n_kv, ctx, n_seq; float eps, base; int ldq, ldk, mode; float mscale; } qkr_args;  // mode: kernels.metal's (0: Qwen3)
typedef struct { int n_head, n_kv, ctx; float scale; int ldq, ldo; } attn_args;
typedef struct { int n_head, n_kv, ctx; float scale; int ldq, ldo, maxc, n_fast, maxch, msp; } attn2_args;  // n_fast, maxch: fa1's rows (0: none)
typedef struct { int n_head, n_kv, ctx; float scale; int ldq, ldo, swap; } attnp_args;  // swap: pre2 tiles fastest in the grid
typedef struct { int x, y, z, w; } int4;  // a prompt query block, as kernels.metal reads it
#define AC 64  // attention chunk (cells), as kernels.metal

typedef struct { uint64_t q, k, v, o, gate, up, down, attn_norm, ffn_norm, q_norm, k_norm, qb, kb, vb; int down_fmt, fmt, v_fmt; } layer_t;  // qb..: Qwen2's biases  // fmt: q..up

static id<MTLDevice> dev;
static id<MTLBuffer> W;  // the whole GGUF file, no copy
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> * pso;
static id<MTLLibrary> g_lib;  // set: pipelines are built from it on first use (sn.m); nil: pso holds every pipeline (next.m)

static id<MTLComputePipelineState> K(NSString * n) {
    id<MTLComputePipelineState> p = pso[n];
    if (!p && g_lib) {
        id<MTLFunction> f = [g_lib newFunctionWithName:n];
        p = f ? [dev newComputePipelineStateWithFunction:f error:nil] : nil;
        if (p) pso[n] = p;
    }
    return p;
}

// ---- the step's encoder; in profile mode every kernel gets its own encoder with start/end GPU timestamps ----------
static id<MTLCommandBuffer> g_cb;
static id<MTLComputeCommandEncoder> g_e;
static int g_prof = 0, g_np = 0;
static id<MTLCounterSampleBuffer> g_sb;
static char g_cat[8192][24];
// concurrent mode: kernels may overlap, with a barrier before each one unless the caller says it does not depend on the
// kernel before (q, k, v of a layer read the same input; so do gate and up)
typedef struct { int K, M, R, ldx, ldy, acc, nb01; } mvm_args;  // kernels.metal's mvm_q4_0
static int g_mvm_nsg = 0, g_mvm_nt = 0;
static int g_mvm = 1;  // a wide check's Q4_0 matmuls on mvm_q4_0 (ggml's few-row MMA design); SN_MVM=0: the block matmuls
// SN_SMALL=n (n <= 8, default 8; 0: off): a step of at most n background write rows takes mvm_q4_0 for them (any prompt
// rows beside them the block matmuls): a row's bits are the same at 1-8 rows but not the block matmuls', so an agent's tokens
// are the same whenever at most n write (whatever prompts are read beside them) and whenever more than n write, but may differ
// between the two (ggml's rule, at 16). 8 agents +12..+20% on the Q4_0 models (qb). g_nsw: the step's background write rows
static int g_small = 8, g_small_step = 0, g_nsw = 0;
static int g_xhb_stale = 0;  // an mvm call came after the last f2h: g_xhb is not its input (no xh_same reuse)
static int g_conc = 0, g_conc_env = 0, g_nobar = 0;  // overlap measured -4.5% at 1 row, +3% at 128: off by default (SN_CONC: every step)
// SN_LATE (a step with interactive and background rows; default 11 = 1 + 2 + 8): 1 the interactive rows' mat-vec of a matrix
// right AFTER the background rows' block matmul; 2 the fused q/k/v and gate/up mat-vecs give way to one mat-vec per matrix
// there (Q4_0 / Q4_1, whose mm() mat-vec has the fused kernels' sums); 8 such a step (write rows only) on a concurrent
// encoder, so each mat-vec runs beside its block matmul with no barrier between them (the block matmuls leave about half
// the memory speed unused). Same kernels, same output rows each: the bits do not change. Chat + 31 background step: Gemma 3
// 4B 23.9 -> 21.4 ms, Qwen3-4B 23.9 -> 21.0, E2B 16.2 -> 13.7, Phi (+15) 32.9 -> 29.0, Qwen3.5-4B 41.1 -> 38.0; MoE models
// level. Tried, slower: the output layer in 16 MB row chunks (each chunk's mat-vec from the cache), the mat-vec dispatched
// just before its block matmul, k / v beside q in small background steps (no gain)
static int g_late = 11;
// SN_CHATBATCH (default 1; Ben: on by default): in a step with background write rows the interactive rows' matmuls join theirs
// (the block matmuls, or mvm in a small step), so the weights are read once: chat + 31 background +5..+32% (Llama 8B 20 -> 26
// tok/s, Mistral +25%, Gemma 3 4B +15%). A chat's tokens may then differ at near-ties with and without background writers
// (ggml's behavior); alone it keeps the one-row kernels. Its attention stays on the one-row kernels. 0: the chat exact (SN_LATE)
static int g_chatbatch = 1;
// SN_QMMS=n: block matmuls of n rows or more on Q8_0 / Q6_K / Q5_K weights (half input) take qmms: two 32-token tiles share
// each unpacked weight tile in threadgroup memory (qmm_ks's simdgroups, sums and slice order: the same bits, kq_bench). At 512
// rows these formats are bound by re-reading the weights (cost tracks bytes a weight): q8_0 0.72x, q6_K 0.78x, q5_K 0.9x
static int g_qmms = 64;  // (default: prompt steps of 64+ rows; 0: off). Mistral doc read +2% (its Q6_K down projections)
// SN_CONCSMALL (default 1; 0: off): a write step of at most 16 background rows (or a wide check) on a concurrent encoder:
// k / v beside q, and gate beside up with the gated unit's epilogue in its own pass (glu_comb, the same expression): small
// steps leave the GPU half idle (ggml's concurrent graph is worth ~3% there). Same kernels and sums: the same bits. 8 agents
// +2..+5% (Mistral 290 -> 303), wide checks +2..+10%
static int g_concsmall = 1, g_conc_step = 0;
static int g_pre_attn = 2;  // prompt rows: 2 the matrix-unit prompt attention, 1 the simdgroup one, 0 the write attention (debugging)
static int g_nt_min = 16;            // the smallest token tile (8 or 16)
static int g_bg = 0;  // SN_BG: the block matmuls' K slices (tuning; 0: by shape)
static int g_nt64 = 1 << 30;         // 64-token tiles from this many rows (same bits, 40% slower reads: off)
// block matmuls read their input as half, converted once (kernel f2h); g_xh_same: the input is the one the last block matmul
// converted (k / v after q, up after gate). 0: float inputs as before (SN_XF32=1)
static int g_xh = 1, g_xh_same = 0;
static id<MTLBuffer> g_xhb;
// a gated unit's output in half for the next block matmul: mm_glu asks the up projection's block rows to write it into g_xhn
// (g_xhn_want) instead of y; the next mm() whose input is that y, K = its M, takes g_xhn as its half input (no f2h)
static id<MTLBuffer> g_xhn;
static __unsafe_unretained id<MTLBuffer> g_xhn_src;
static int g_xhn_want = 0, g_xhn_ok = 0, g_xhn_k = 0;
static void enc(const char * cat) {
    if (!g_prof) {
        if (!g_e) g_e = g_conc ? [g_cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [g_cb computeCommandEncoder];
        else if (g_conc && !g_nobar) [g_e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        g_nobar = 0;
        return;
    }
    g_nobar = 0;
    if (g_e) [g_e endEncoding];
    MTLComputePassDescriptor * pd = [MTLComputePassDescriptor computePassDescriptor];
    pd.sampleBufferAttachments[0].sampleBuffer = g_sb;
    pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * g_np;
    pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * g_np + 1;
    snprintf(g_cat[g_np], sizeof g_cat[0], "%s", cat);
    ++g_np;
    g_e = [g_cb computeCommandEncoderWithDescriptor:pd];
}
// buffers b0, b1, b2 (from byte offsets o0, o1, o2) at indices i, i + 1, i + 2
static inline void set3(id<MTLComputeCommandEncoder> c, int i, id<MTLBuffer> b0, size_t o0, id<MTLBuffer> b1, size_t o1, id<MTLBuffer> b2, size_t o2) {
    [c setBuffer:b0 offset:o0 atIndex:i];
    [c setBuffer:b1 offset:o1 atIndex:i + 1];
    [c setBuffer:b2 offset:o2 atIndex:i + 2];
}

// a kernel's encoder: the step's next one (enc; cat names it in profile mode) with kernel `name` and its arguments at 0
static inline id<MTLComputeCommandEncoder> kernel_begin(const char * cat, NSString * name, const void * args, size_t len) {
    enc(cat);
    id<MTLComputeCommandEncoder> c = g_e;
    [c setComputePipelineState:K(name)];
    [c setBytes:args length:len atIndex:0];
    return c;
}
typedef struct { int K, M, N, nb01, ldx, ldd, ksplit, flags, estride, ldh; } qmm_args;  // estride: grouped (flags 16) experts' matrices apart; ldh: flags 32's half rows
typedef struct { int K, M, R, ldx, ldy, acc, nb01; float eps; } mv_args;
// the interactive rows' input for the next mm(): mode 1 the residual normed on the way in (aux = the norm weight), 2 the
// gate with silu(gate) * up (aux = up), 0 the mm's own input; reset by mm()
static int g_amode = 0;
static int g_hout = 1;  // SN_HOUT=0: gated units' block rows in float (f2h before the down projection)
static int g_q8k = 1;  // Q8_0 interactive rows on mv_q8_0k, K split over 4 simdgroups (35B / gpt-oss steps -1.5%; SN_Q8K=0: mv_q8_0)
// outputs a threadgroup of the K-quant mat-vecs (2 simdgroups of 2 rows; Q4_K / Q5_K rows of under 4 blocks: kernels.metal's
// row groups, 4 / blocks times as many)
static int mv_per(int fmt, int K_) { const int nb = K_ / 256; return (fmt == 6 || fmt == 8) && nb < 4 ? 4 * (4 / nb) : 4; }
static int g_fold = 0;  // fold the norm / swiglu into the interactive mat-vecs: measured 8% slower at 1 row, off
static id<MTLBuffer> g_skpart;  // split-K partials of the small f32 / bf16 matmuls
static int g_tmm = 1;  // the small f32 / bf16 matrices' batch rows on the matrix units (0: simdgroup MMA)
static int g_spec = 0;  // most draft tokens per interactive agent and step (prompt lookup), 0: off
static long g_spec_drafted = 0, g_spec_kept = 0;
static id<MTLBuffer> g_ax, g_aux;
static uint64_t g_auxo;
static float g_eps;
static int M_E_ROWS;  // the residual's row stride (n_embd)
static int g_nA = 0;   // the step's first g_nA rows belong to interactive agents: the one-row kernels (mv_*)
static int g_wide = 0;  // this step is a wide draft check (sn_step n_fast < 0): its block matmuls may take another K split
static int g_nAtt = 0;  // the rows whose attention takes the one-row kernels: g_nA's, or a wide draft check's (sn_step n_fast < 0)
static int g_fast = 0; // the first g_fast sequences are interactive
static int g_mvf1 = 1;   // one-row f32 mat-vecs without mvf_ks's 32-row accumulators (SN_MVF1=0: off)
static int g_topk1 = 1;  // the experts' top k in one simdgroup a row, no barriers (SN_TOPK1=0: moe_topk)
static int g_join = 11;  // SN_JOIN: kernels of one step joined into one dispatch, the same bits: 1 conv_save into conv8, 2 one-row f32
                          // pairs (alpha + beta, router + shared gate), 8 one-row Q8_0 groups (qkv + z, q + k + v); off by default (measured
                          // slower on Qwen3.6-35B-A3B): 4 moe_out + the next rmsnorm (+0.16 ms a token), 16 expert gate + up (+0.04 ms)
static int g_conv8 = 1;  // q35_conv as threadgroups of 8 heads (SN_CONV8=0: one threadgroup a row)
static int g_skip = 0;       // timing only: 1 attention, 2 block matmuls, 4 output layer, 8 small ops (garbage tokens)
static int g_f32 = 0;        // 1: the simple f32 kernels (mm_*), 0: the matrix-unit kernels (qmm_*)
static uint64_t g_head = 0;  // token_embd (Q6_K) in W: the output layer in qmm mode

#include "core_mm.inc.h"

// ==== Gemma 3 / 4 =======================================================================================================
// The same row design as Qwen3: every token in flight is a row, each kernel computes a row the same way whatever else is in
// the step. Sliding-window layers and full layers, each with its own head size and K/V heads (Gemma 4: head 256 / 512, the
// full layers with frequency factors; Gemma 3: head 256 or 128 everywhere, the full layers' rope scaled), GELU, post-norms.
// Gemma 4 E2B also keeps notes only in the first layers (the rest reuse the last window / full layer's) and has per-layer
// token embeddings, whose table (1.6 GB) stays off the GPU: the CPU copies the step's rows.

// Q4_K's 6-bit scale and min of group j (of 8) from a block's 12 scale bytes sc
static void q4k_scale_min(const uint8_t * sc, int j, int * s_, int * m_) {
    if (j < 4) { *s_ = sc[j] & 63; *m_ = sc[j + 4] & 63; }
    else { *s_ = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); *m_ = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
}

// one Q4_K block (256 weights) as 8 q4_1 blocks of 32 (20 bytes each) at dst
static void q4k_block_to_q41(const uint8_t * blk, uint8_t * dst) {
    const float d = h2f(*(const uint16_t *) blk), dmin = h2f(*(const uint16_t *) (blk + 2));
    const uint8_t * sc = blk + 4, * q = blk + 16;
    for (int g = 0; g < 8; ++g) {
        int s_, m_;
        q4k_scale_min(sc, g, &s_, &m_);
        uint8_t * o = dst + g * 20;
        __fp16 dd = (__fp16) (d * (float) s_), mm_ = (__fp16) (-(dmin * (float) m_));
        memcpy(o, &dd, 2); memcpy(o + 2, &mm_, 2);
        const uint8_t * qq = q + (g / 2) * 32;
        for (int l = 0; l < 16; ++l) {
            const int lo = (g % 2) ? (qq[l] >> 4) : (qq[l] & 0xF), hi = (g % 2) ? (qq[l + 16] >> 4) : (qq[l + 16] & 0xF);
            o[4 + l] = (uint8_t) (lo | (hi << 4));
        }
    }
}

// Q4_K -> q4_1 (fp16 scale d*sc and min -dmin*m per 32 weights): the output layer and the embeddings, for the q4_1 kernels
static void q4k_to_q41(const uint8_t * src, int64_t rows, int64_t K, uint8_t * dst) {
    const int64_t nb = K / 256;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t b = 0; b < nb; ++b) q4k_block_to_q41(src + (r * nb + b) * 144, dst + (r * nb + b) * 8 * 20);
}

typedef struct { int n; float scale; int ld; int ple_n; int n_layer; float eps; } g4_args;
typedef struct { int n; float eps; int ldx, ldy, has_s; } na_args;
typedef struct { int n, ldg, ldu, rows; } gm_args;
typedef struct { int n_head, n_kv, ctx; float eps, base, fscale; int ldq, ldk, has_kv, has_ff, vnorm, oss; float ylo, yhi, ymscale; } g4q_args;
typedef struct { int n_head, n_kv, ctx; float scale; int ldq, ldo, maxc, window, n_fast, maxch, hg, sinks, msp; } g4a_args;
typedef struct { int n_head, n_kv, ctx; float scale; int ldq, ldo, window, sinks; } g4p_args;
typedef struct { int C, nk, nv, ld_qkv, ld_ab, nlin; float eps; int lin; } q35_args;               // kernels.metal's Qwen3.5 ones
typedef struct { int n_head, n_kv, ctx, ldqf, ldq, ldk; float eps, base; int rot; } q35q_args;
typedef struct { int ne, k_used, r0, npairs, nt, ldl, mode; } moe_args;                                       // kernels.metal's MoE ones
typedef struct { mv_args m; int k_used, xpair, estride; } mvid_args;
// an MoE layer's experts (sn.m): the router (f32) and its bias (gpt-oss), the experts' gate / up / down (s_*: bytes from one
// expert's matrix to the next) and their biases (gpt-oss), the shared expert and its gate (f32, Qwen3.5-MoE)
typedef struct { uint64_t router, router_b, eg, eu, ed, bg, bu, bd, sg, su, sd, sgate; int f_eg, f_eu, f_ed, f_sg, f_sd, s_eg, s_eu, s_ed; } moe_t;
// a layer: its matrices (fmt: q .. up, as mm's), head size D, K/V heads nkv (G = n_head / nkv query heads each), prompt
// tiles of 32 / G tokens (list: which of the step's two tile lists); gpt-oss: q / k / v / output biases, sinks, experts
typedef struct { uint64_t q, k, v, o, gate, up, down, attn_norm, ffn_norm, q_norm, k_norm, post_attn, post_ffw, inp_gate, proj, post_norm, out_scale;
                 int fmt, v_fmt, down_fmt, swa, D, nkv, ff, kv_src, has_kv, has_v, list, pool; size_t kc_off, koff;
                 uint64_t qb, kb, vb, ob, sinks; moe_t m; int pi; } g4layer_t;  // koff, pi: sn.m's

