// sllm-next engine library, a part of sn.m (one translation unit; sn.m includes its parts in order): sn_open's first stages: the family, sizes and SN_* switches; Qwen3.5's and the dense models' weights, cache and buffers

// the model's family and traits, from its architecture and its blocks' tensor types
static void arch_flags(sn_engine * e) {
    gguf_t * g = &e->g;
    e->oss = !strcmp(g->arch, "gpt-oss");
    e->gemma = !strcmp(g->arch, "gemma4") || !strcmp(g->arch, "gemma3") || e->oss;
    e->q35 = !strcmp(g->arch, "qwen35") || !strcmp(g->arch, "qwen35moe");
    e->moe = !strcmp(g->arch, "qwen35moe") || e->oss;
    for (int i = 0; i < g->nt; ++i) if (g->t[i].type == 2 && !strncmp(g->t[i].name, "blk.", 4)) e->q40 = 1;
}

// the window, slots, cache pages and step rows sn_open was asked for, and the model's widths
static void engine_sizes(sn_engine * e, int slots, int ctx, int max_rows, int pages) {
    const gguf_t * g = &e->g;
    e->ctx = (ctx + 63) / 64 * 64;
    e->slots = slots;
    e->pl[0].n = pages > 0 ? pages : slots * (e->ctx / 64);  // 0: every sequence its whole window
    e->rmax = MAX(max_rows, slots + 4);
    e->maxc = (e->ctx + 63) / 64;
    e->E = g->n_embd; e->L = g->n_layer; e->NH = g->n_head;
}

// the SN_* switches (core.h's g_*), read at every sn_open in this order: how each sets its variable when it is set
enum { ENV_OFF, ENV_INT, ENV_ON, ENV_MAX8 };  // 0; atoi; 1, and 0 when it is not set; atoi, at most 8
typedef struct { const char * name; int * var; int how; } env_switch;
static const env_switch ENV_SWITCHES[] = {
    {"SN_NOFUSE", &g_fuse, ENV_OFF}, {"SN_XF32", &g_xh, ENV_OFF}, {"SN_TIME", &g_time, ENV_ON}, {"SN_NOFA1", &g_fa1, ENV_OFF},
    {"SN_NOFA2", &g_fa2, ENV_OFF}, {"SN_NOFA2R", &g_fa2r, ENV_OFF}, {"SN_PRE2_SWAP", &g_pre2swap, ENV_INT},
    {"SN_KVSKIP", &g_kvskip, ENV_INT}, {"SN_MVM", &g_mvm, ENV_INT}, {"SN_SMALL", &g_small, ENV_MAX8},
    {"SN_MVM_NSG", &g_mvm_nsg, ENV_INT}, {"SN_MVM_NT", &g_mvm_nt, ENV_INT}, {"SN_NONAN", &g_nan, ENV_OFF},
    {"SN_SKIP", &g_skip, ENV_INT}, {"SN_CONV8", &g_conv8, ENV_INT}, {"SN_MVF1", &g_mvf1, ENV_INT}, {"SN_TOPK1", &g_topk1, ENV_INT},
    {"SN_JOIN", &g_join, ENV_INT}, {"SN_CONC", &g_conc_env, ENV_INT}, {"SN_LATE", &g_late, ENV_INT},
    {"SN_CHATBATCH", &g_chatbatch, ENV_INT}, {"SN_QMMS", &g_qmms, ENV_INT}, {"SN_CONCSMALL", &g_concsmall, ENV_INT},
    {"SN_Q8K", &g_q8k, ENV_INT}, {"SN_HOUT", &g_hout, ENV_INT}, {"SN_NTMIN", &g_nt_min, ENV_INT}, {"SN_NT64", &g_nt64, ENV_INT},
    {"SN_BG", &g_bg, ENV_INT}, {"SN_GRT", &g_grt, ENV_INT}, {"SN_GG", &g_gg, ENV_INT}, {"SN_GNT", &g_gnt, ENV_INT},
    {"SN_MSP", &g_msp, ENV_INT},
};

static void env_switches(void) {
    for (size_t i = 0; i < sizeof ENV_SWITCHES / sizeof ENV_SWITCHES[0]; ++i) {
        const env_switch * w = &ENV_SWITCHES[i];
        const char * v = getenv(w->name);
        if (w->how == ENV_ON) *w->var = v != NULL;
        else if (v) *w->var = w->how == ENV_OFF ? 0 : w->how == ENV_MAX8 ? MIN(atoi(v), 8) : atoi(v);
    }
}

// SN_PROF: a timestamp sample buffer for the kernels' times, unless one was made (nerr: sn_open's, which the library's
// compile reports into after)
static void prof_init(NSError ** nerr) {
    if (getenv("SN_PROF") && !g_sb) {
        MTLCounterSampleBufferDescriptor * sd = [MTLCounterSampleBufferDescriptor new];
        for (id<MTLCounterSet> cs in dev.counterSets) if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) sd.counterSet = cs;
        sd.sampleCount = 4096;  // the most a sample buffer takes: 2,048 kernels a step
        sd.storageMode = MTLStorageModeShared;
        g_sb = [dev newCounterSampleBufferWithDescriptor:sd error:nerr];
        g_prof_r = g_sb ? atoi(getenv("SN_PROF")) : 0;
        g_prof_sum = [NSMutableDictionary new];
    }  // timing only (core.h g_skip): the tokens are garbage
}

#define Q35T(field, suffix) do { snprintf(nm, sizeof nm, "blk.%d." suffix, i); l->field = g->data + find(g, nm)->off; } while (0)
#define Q35F(field, suffix) do { snprintf(nm, sizeof nm, "blk.%d." suffix, i); l->field = fmt_of(find(g, nm)->type); } while (0)

// Qwen3.5's blocks: linear-attention or attention layers (every fa_interval-th), each with a MoE or a dense FFN
static void q35_load_layers(sn_engine * e) {
    gguf_t * g = &e->g;
    const int L = e->L;
    char nm[128];
    e->ql = calloc(L, sizeof(q35l));
    for (int i = 0; i < L; ++i) {
        q35l * l = &e->ql[i];
        l->lin = (i + 1) % g->fa_interval != 0;
        l->li = l->lin ? e->nlin++ : e->natt++;
        Q35T(attn_norm, "attn_norm.weight"); Q35T(post_norm, "post_attention_norm.weight");
        if (e->moe) {
            Q35T(m.sgate, "ffn_gate_inp_shexp.weight");
            Q35T(m.sg, "ffn_gate_shexp.weight"); Q35T(m.su, "ffn_up_shexp.weight"); Q35T(m.sd, "ffn_down_shexp.weight");
            Q35F(m.f_sg, "ffn_gate_shexp.weight"); Q35F(m.f_sd, "ffn_down_shexp.weight");
            experts(g, i, &l->m);
        } else {
            Q35T(gate, "ffn_gate.weight"); Q35T(up, "ffn_up.weight"); Q35T(down, "ffn_down.weight");
            Q35F(fmt, "ffn_gate.weight"); Q35F(down_fmt, "ffn_down.weight");
        }
        if (l->lin) {
            Q35T(qkv, "attn_qkv.weight"); Q35T(z, "attn_gate.weight"); Q35T(alpha, "ssm_alpha.weight"); Q35T(beta, "ssm_beta.weight");
            Q35T(a, "ssm_a"); Q35T(conv, "ssm_conv1d.weight"); Q35T(dt, "ssm_dt.bias"); Q35T(norm, "ssm_norm.weight"); Q35T(out, "ssm_out.weight");
            Q35F(f_qkv, "attn_qkv.weight"); Q35F(f_z, "attn_gate.weight"); Q35F(f_ab, "ssm_alpha.weight"); Q35F(f_out, "ssm_out.weight");
            snprintf(nm, sizeof nm, "blk.%d.ssm_alpha.weight", i);
            if (find(g, nm)->type == 0) l->f_ab = 3;  // f32 (Qwen3.6)
        } else {
            Q35T(q, "attn_q.weight"); Q35T(k, "attn_k.weight"); Q35T(v, "attn_v.weight"); Q35T(o, "attn_output.weight");
            Q35T(q_norm, "attn_q_norm.weight"); Q35T(k_norm, "attn_k_norm.weight");
            Q35F(f_q, "attn_q.weight"); Q35F(f_v, "attn_v.weight"); Q35F(f_o, "attn_output.weight");
        }
    }
}

// the guess head's side file: SN_GUESS_HEAD, or the model's path with .gguf replaced by .guess-head.gguf ("": none)
static void q35_side_path(const char * path, char * sp, size_t cap) {
    if (getenv("SN_GUESS_HEAD")) snprintf(sp, cap, "%s", getenv("SN_GUESS_HEAD"));
    else if (strlen(path) > 5 && strlen(path) < 1100 && !strcmp(path + strlen(path) - 5, ".gguf")) snprintf(sp, cap, "%.*s.guess-head.gguf", (int) strlen(path) - 5, path);
}

// the side file in e->g2 holds this model's guess-head block L (gguf_open took the head's blocks off its count: the side
// file's own count is the model's plus them)
static bool q35_side_ok(sn_engine * e, int L) {
    return e->g2.base && e->g2.n_nextn == 1 && e->g2.n_layer == L && !strcmp(e->g2.arch, e->g.arch) && mtp_ok(&e->g2, L);
}

// The GGUF that holds the guess head's block: the side file beside the model (e->g2, mapped for the GPU as wb[3] at
// SIDE_BASE) when it has the block, else the model's own
static gguf_t * q35_side_head(sn_engine * e, const char * path) {
    gguf_t * g = &e->g;
    char sp[1200] = "";
    q35_side_path(path, sp, sizeof sp);
    if (!sp[0] || access(sp, R_OK) != 0) return g;
    gguf_open(sp, &e->g2);
    if (!q35_side_ok(e, e->L)) return g;
    const size_t page = getpagesize();
    e->wb[3] = [dev newBufferWithBytesNoCopy:(void *) e->g2.base length:(e->g2.size + page - 1) / page * page options:MTLResourceStorageModeShared deallocator:nil];
    if (!e->wb[3]) return g;
    e->rb[3][0] = SIDE_BASE; e->rb[3][1] = SIDE_BASE + e->g2.size;
    e->g2.data += SIDE_BASE;  // its tensors' handles: SIDE_BASE + their place in the side file
    return &e->g2;
}

// the guess head (e->mtp): block L of g (the model's GGUF or the side file), its norms after the model's output norm when
// it has none of its own
static void q35_head_load(sn_engine * e, gguf_t * g) {
    const int i = e->L;
    char nm[128];
    q35l * l = e->mtp = calloc(1, sizeof(q35l));
    l->lin = false; l->li = e->natt;
    Q35T(attn_norm, "attn_norm.weight"); Q35T(post_norm, "post_attention_norm.weight");
    Q35T(m.sgate, "ffn_gate_inp_shexp.weight");
    Q35T(m.sg, "ffn_gate_shexp.weight"); Q35T(m.su, "ffn_up_shexp.weight"); Q35T(m.sd, "ffn_down_shexp.weight");
    Q35F(m.f_sg, "ffn_gate_shexp.weight"); Q35F(m.f_sd, "ffn_down_shexp.weight");
    experts(g, i, &l->m);
    Q35T(q, "attn_q.weight"); Q35T(k, "attn_k.weight"); Q35T(v, "attn_v.weight"); Q35T(o, "attn_output.weight");
    Q35T(q_norm, "attn_q_norm.weight"); Q35T(k_norm, "attn_k_norm.weight");
    Q35F(f_q, "attn_q.weight"); Q35F(f_v, "attn_v.weight"); Q35F(f_o, "attn_output.weight");
    snprintf(nm, sizeof nm, "blk.%d.nextn.eh_proj.weight", i); e->mtp_eh = g->data + find(g, nm)->off; e->mtp_feh = fmt_of(find(g, nm)->type);
    snprintf(nm, sizeof nm, "blk.%d.nextn.enorm.weight", i); e->mtp_enorm = g->data + find(g, nm)->off;
    snprintf(nm, sizeof nm, "blk.%d.nextn.hnorm.weight", i); e->mtp_hnorm = g->data + find(g, nm)->off;
    snprintf(nm, sizeof nm, "blk.%d.nextn.shared_head_norm.weight", i);
    e->mtp_hdnorm = find_opt(g, nm) ? g->data + find_opt(g, nm)->off : e->g.data + find(&e->g, "output_norm.weight")->off;
}

// The MoE's guess head (block L, e->mtp): from the model's file, or else from the side file beside it (SN_GUESS_HEAD:
// another path; SN_NOGUESS: none). to: the untied output table (only untied models take one)
static bool guess_wanted(const sn_engine * e, const tensor_t * to) { return e->moe && to && !getenv("SN_NOGUESS"); }
static void q35_guess_head(sn_engine * e, const char * path, const tensor_t * to) {
    gguf_t * g = &e->g;
    gguf_t * gm = g;  // the GGUF that holds the guess head's block: the model's own, or a side file with only that block
    if (g->n_nextn == 0 && guess_wanted(e, to)) gm = q35_side_head(e, path);
    if (gm->n_nextn == 1 && guess_wanted(e, to) && (gm != g || mtp_ok(g, e->L))) q35_head_load(e, gm);  // the guess head: block L
}

// an untied token table (to: the output layer, NULL when tied) stays off the GPU: the CPU copies the step's rows
static void emb_cpu_init(sn_engine * e, const tensor_t * te) {
    const gguf_t * g = &e->g;
    e->emb_cpu = true;
    e->etab = g->base + g->data + te->off;
    e->erow = tsize(te) / te->ne[1];
    for (int b = 0; b < 2; ++b) e->erowB[b] = [dev newBufferWithLength:(size_t) e->rmax * e->erow options:MTLResourceStorageModeShared];
}

// the untied token table (to set) staged by the CPU and left out of the GPU's buffers: 0, or -1 with err set
static int map_tables(sn_engine * e, const tensor_t * te, const tensor_t * to, const char * path, char * err, int err_len) {
    const gguf_t * g = &e->g;
    if (to) emb_cpu_init(e, te);  // untied: the token table stays off the GPU, the CPU copies the step's rows
    uint64_t holes[1][2] = {{g->data + te->off, g->data + te->off + tsize(te)}};
    if (map_file(e, holes, to ? 1 : 0) != 0) { set_err(err, err_len, "%s: cannot map the weights for the GPU", path); return -1; }
    return 0;
}

// mm formats of the token table and the output layer: 2 q6_K, 5 q8_0, 0 q4_0 (Llama's Q4_0 files), 1 q4_1 (output only),
// 6 q4_K; -1 none
static int head_fmt_of(int type) {
    switch (type) {
        case 14: return 2;
        case 8: return 5;
        case 2: return 0;
        case 3: return 1;
        case 12: return 6;
    }
    return -1;
}

// the token table and the output layer (to = te when tied): formats, handles, the output layer's buffer, the vocabulary
static void tables_set(sn_engine * e, const tensor_t * te, const tensor_t * to) {
    const gguf_t * g = &e->g;
    e->embd_fmt = te->type == 3 ? -1 : head_fmt_of(te->type);
    e->head_fmt = head_fmt_of(to->type);
    e->embd = g->data + te->off;
    e->head = g->data + to->off;
    e->W = gb(e, e->head);
    e->V = (int) te->ne[1];
}

// pool 0's K and V buffers of len bytes each (kv_alloc: placement sparse ones when it can; sparse only when both are)
static void kv_alloc_pool0(sn_engine * e, size_t len) {
    e->pl[0].kc = kv_alloc(e, len);
    const bool sk = e->sparse;
    e->pl[0].vc = kv_alloc(e, len);
    if (sk && !e->sparse) e->pl[0].kc = [dev newBufferWithLength:e->pl[0].kc.length options:MTLResourceStorageModePrivate];
    e->sparse = e->sparse && sk;
}

// Qwen3.5's step buffers: prompt tiles (G query heads a K/V head), the attention's q, the linear layers' conv output,
// alpha / beta, gate z and delta output
static void q35_step_buffers(sn_engine * e) {
    const gguf_t * g = &e->g;
    const int R = e->rmax, NH = e->NH, NKV = g->n_kv, D = g->hd, nv = g->ssm_rank, inner = g->ssm_inner, QD = NH * D, C = e->q35C;
    const int G = NH / NKV;
    e->qhb = priv_buf((size_t) (R / (32 / G) + e->slots + 1) * NKV * 32 * D * 2);
    e->qa = priv_buf((size_t) R * QD * 4);
    e->co = priv_buf((size_t) R * C * 4);
    e->ab = priv_buf((size_t) R * 2 * nv * 4);
    e->bg = priv_buf((size_t) R * 2 * nv * 4);
    e->zb = priv_buf((size_t) R * inner * 4);
    e->dout = priv_buf((size_t) R * inner * 4);
}

// the recurrent states: the pool (sparse when it can be), each sequence's entry, length, checkpoint and spare entries,
// and the step's run records in two sets
static void q35_state_pool(sn_engine * e) {
    const MTLResourceOptions PRIV = MTLResourceStorageModePrivate, SH = MTLResourceStorageModeShared;
    const int R = e->rmax, slots = e->slots, nv = e->g.ssm_rank, C = e->q35C;
    e->nst = MIN(slots, 2);  // the state pool: grows as sequences take entries (q35_entry)
    e->q35C = C;
    if (!q35_sparse_init(e)) {
        e->cst = [dev newBufferWithLength:(size_t) e->nst * e->nlin * 3 * C * 4 options:PRIV];
        e->sst = [dev newBufferWithLength:(size_t) e->nst * e->nlin * nv * 128 * 128 * 4 options:PRIV];
    }
    e->sidx = malloc((size_t) slots * sizeof(int));
    e->sfree = malloc((size_t) (slots + e->stgrow) * sizeof(int));  // a sparse pool's last chunk may pass the slots
    e->slen = calloc(slots, sizeof(int));
    e->ckent = malloc((size_t) slots * sizeof(int));
    e->ckpos = calloc(slots, sizeof(int));
    e->dent = malloc((size_t) slots * 3 * sizeof(int));
    e->dn = calloc(slots, sizeof(int));
    e->dpos = calloc(slots, sizeof(int));
    for (int b = 0; b < 2; ++b) {
        e->rsnapB[b] = [dev newBufferWithLength:(size_t) R * 4 options:SH];
        e->csnapB[b] = [dev newBufferWithLength:(size_t) R * 16 options:SH];
        e->runsB[b] = [dev newBufferWithLength:(size_t) R * 16 options:SH];
        e->rrunB[b] = [dev newBufferWithLength:(size_t) R * 4 options:SH];
        e->rsiB[b] = [dev newBufferWithLength:(size_t) R * 4 options:SH];
    }
}

// Qwen3.5 (qwen35, qwen35moe): weights, guess head, cache, the linear layers' state pool and the step buffers of its own;
// 0, or -1 with err set
static int q35_open(sn_engine * e, const char * path, char * err, int err_len) {
    gguf_t * g = &e->g;
    const int NH = e->NH;
    const int F = g->n_ff, NKV = g->n_kv, D = g->hd, nk = g->ssm_groups, inner = g->ssm_inner, QD = NH * D;
    const int C = 2 * nk * 128 + inner;
    e->q35C = C;
    e->QMAX = MAX(MAX(2 * QD, C), inner); e->KD = NKV * D; e->FMAX = MAX(F, 1);
    tensor_t * te = find(g, "token_embd.weight"), * to = find_opt(g, "output.weight");
    if (map_tables(e, te, to, path, err, err_len)) return -1;
    q35_load_layers(e);
    q35_guess_head(e, path, to);
    e->out_norm = g->data + find(g, "output_norm.weight")->off;
    if (!to) to = te;
    tables_set(e, te, to);
    e->kv_layer = (size_t) e->pl[0].n * 64 * e->KD;  // the attention layers' cache
    const int kvl = MAX(e->natt + (e->mtp ? 1 : 0), 1);  // the guess head's layer keeps cells like any attention layer
    kv_alloc_pool0(e, e->kv_layer * kvl * 2);
    q35_step_buffers(e);
    q35_state_pool(e);
    if (e->moe) moe_alloc(e, dev, e->rmax);
    return 0;
}

// Llama, Qwen2 / 3, Phi-3, SmolLM3's blocks: the weights' handles and formats
static void dense_load_layers(sn_engine * e) {
    gguf_t * g = &e->g;
    const int L = e->L, F = g->n_ff;
    char nm[128];
    e->ly = calloc(L, sizeof(layer_t));
    for (int i = 0; i < L; ++i) {
#define TQ(field, suffix) do { snprintf(nm, sizeof nm, "blk.%d." suffix, i); e->ly[i].field = g->data + find(g, nm)->off; } while (0)
        snprintf(nm, sizeof nm, "blk.%d.attn_qkv.weight", i);
        tensor_t * tqkv = find_opt(g, nm);
        if (tqkv) {  // Phi-3: q, k and v are rows of one matrix, gate and up the halves of ffn_up
            const uint64_t rb = tsize(tqkv) / tqkv->ne[1];
            e->ly[i].q = g->data + tqkv->off; e->ly[i].k = e->ly[i].q + (uint64_t) e->QMAX * rb; e->ly[i].v = e->ly[i].k + (uint64_t) e->KD * rb;
            snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", i);
            tensor_t * tu = find(g, nm);
            e->ly[i].gate = g->data + tu->off; e->ly[i].up = e->ly[i].gate + (uint64_t) F * (tsize(tu) / tu->ne[1]);
        } else {
            TQ(q, "attn_q.weight"); TQ(k, "attn_k.weight"); TQ(v, "attn_v.weight");
            TQ(gate, "ffn_gate.weight"); TQ(up, "ffn_up.weight");
        }
        TQ(o, "attn_output.weight"); TQ(down, "ffn_down.weight");
        TQ(attn_norm, "attn_norm.weight"); TQ(ffn_norm, "ffn_norm.weight");
        snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", i);
        if (find_opt(g, nm)) { TQ(q_norm, "attn_q_norm.weight"); TQ(k_norm, "attn_k_norm.weight"); }  // Qwen3's; Llama has none
        snprintf(nm, sizeof nm, "blk.%d.attn_q.bias", i);
        if (find_opt(g, nm)) { TQ(qb, "attn_q.bias"); TQ(kb, "attn_k.bias"); TQ(vb, "attn_v.bias"); }  // Qwen2's
        snprintf(nm, sizeof nm, tqkv ? "blk.%d.attn_qkv.weight" : "blk.%d.attn_q.weight", i);
        e->ly[i].fmt = fmt_of(find(g, nm)->type);  // q, k, o, gate, up (model_check: one type)
        snprintf(nm, sizeof nm, tqkv ? "blk.%d.attn_qkv.weight" : "blk.%d.attn_v.weight", i);
        e->ly[i].v_fmt = fmt_of(find(g, nm)->type);
        snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", i);
        e->ly[i].down_fmt = fmt_of(find(g, nm)->type);
    }
}

// the rope's frequency factors: Llama 3.1's, or Phi-3's LongRoPE ones (llama.cpp's choice by the sequence's window ctx)
static tensor_t * rope_factors(gguf_t * g, int ctx) {
    tensor_t * tf = find_opt(g, "rope_freqs.weight");
    if (!tf && find_opt(g, "rope_factors_short.weight"))  // Phi-3's LongRoPE: llama.cpp's choice by the sequence's window
        tf = find(g, g->ctx_orig > 0 && ctx > g->ctx_orig ? "rope_factors_long.weight" : "rope_factors_short.weight");
    return tf;
}

// qk_rope's mode (ff: with frequency factors). Qwen2: no norm, NEOX, biases; Phi-3: no norm, NEOX, factors
static int qk_mode_of(const gguf_t * g, bool ff) {
    const bool llama_like = !strcmp(g->arch, "llama") || !strcmp(g->arch, "smollm3");
    return llama_like ? 1 | 4 | (ff ? 2 : 0) : !strcmp(g->arch, "qwen2") ? 1 | 16 : !strcmp(g->arch, "phi3") ? 1 | 2 : 0;
}

// rope: Qwen3 norms q / k per head (NEOX pairs); Llama does not and pairs adjacent dims, Llama 3.1 with frequency factors
static void dense_rope(sn_engine * e) {
    gguf_t * g = &e->g;
    tensor_t * tf = rope_factors(g, e->ctx);
    e->rope_ff = tf ? g->data + tf->off : 0;
    e->mscale = g->rope_attn > 0 ? g->rope_attn : 1.0f;
    e->qk_mode = qk_mode_of(g, tf != NULL);
    e->nope_every = !strcmp(g->arch, "smollm3") ? 4 : 0;  // llama.cpp's n_no_rope_layer_step: layer i has no rope when (i + 1) % 4 == 0
}

// the dense attention models (no Qwen3.5, no Gemma / gpt-oss): weights, rope and cache. G: query heads per K/V head;
// 0, or -1 with err set
static int dense_open(sn_engine * e, const char * path, int G, char * err, int err_len) {
    gguf_t * g = &e->g;
    const int R = e->rmax, L = e->L, NH = e->NH, slots = e->slots;
    const int F = g->n_ff, NKV = g->n_kv;
    e->QMAX = NH * g->hd; e->KD = NKV * g->hd; e->FMAX = F;
    tensor_t * te = find(g, "token_embd.weight"), * to = find_opt(g, "output.weight");
    if (map_tables(e, te, to, path, err, err_len)) return -1;
    dense_load_layers(e);
    e->out_norm = g->data + find(g, "output_norm.weight")->off;
    if (!to) to = te;  // tied
    tables_set(e, te, to);
    if (e->embd_fmt < 0 || e->head_fmt < 0) { set_err(err, err_len, "%s: token_embd and output must be q6_K, q8_0 or q4_0", path); return -1; }
    dense_rope(e);
    e->kv_layer = (size_t) e->pl[0].n * 64 * e->KD;
    kv_alloc_pool0(e, e->kv_layer * L * 2);

    e->qhb = [dev newBufferWithLength:(size_t) (R / (32 / G) + slots + 1) * NKV * 32 * 128 * 2 options:MTLResourceStorageModePrivate];  // tiles of 32 / G tokens
    return 0;
}
