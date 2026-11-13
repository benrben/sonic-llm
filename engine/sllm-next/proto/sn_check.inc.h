// sllm-next engine library, a part of sn.m (one translation unit; sn.m includes its parts in order): model checks: what sllm-next runs, from a GGUF's header and tensor table (sn_check, sn_quiet_rows)

// mm's format of a GGUF weight type; -1 when there is no kernel for it
static int fmt_of(int type) {
    switch (type) {
        case 2: return 0;
        case 3: return 1;
        case 8: return 5;
        case 12: return 6;
        case 14: return 7;
        case 13: return 8;
        case 39: return 9;
    }
    return -1;
}

// a layer matrix sllm-next reads: a type it has kernels for, rows of whole blocks (256 weights for the K-quants)
static int mat_ok(const tensor_t * t) { const int f = fmt_of(t->type); return f >= 0 && t->ne[0] % (f >= 6 && f <= 8 ? 256 : 32) == 0; }

static tensor_t * find_opt(gguf_t * g, const char * name) {
    for (int i = 0; i < g->nt; ++i) if (!strcmp(g->t[i].name, name)) return &g->t[i];
    return NULL;
}

// tensor "blk.<i>.<name><sfx>" of g or NULL, its name in nm (128 bytes)
static tensor_t * blk_find(gguf_t * g, char * nm, int i, const char * name, const char * sfx) {
    snprintf(nm, 128, "blk.%d.%s%s", i, name, sfx);
    return find_opt(g, nm);
}

// whether layer i's n tensors "blk.<i>.<names[j]><sfx>" are all there and f32; else false with the first that is not in nm
static bool blk_f32_all(gguf_t * g, char * nm, int i, const char * const * names, int n, const char * sfx) {
    for (int j = 0; j < n; ++j) {
        tensor_t * t = blk_find(g, nm, i, names[j], sfx);
        if (!t || t->type != 0) return false;
    }
    return true;
}

// a layer's matrices (gpt-oss: the first 4)
static const char * const LAYER_MATS[7] = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"};

// a layer's matrices q, k, o, gate and up (j < 6 but v, j = 2) of one type: f0 the first one's (-2: none yet); false at
// one of another type
static bool one_fmt(int j, int f, int * f0) {
    if (j >= 6 || j == 2) return true;
    if (*f0 != -2 && f != *f0) return false;
    *f0 = f;
    return true;
}

// an output layer that is not q6_K, q8_0, q4_0 or q4_1 (an untied one; NULL: tied)
static bool head_type_bad(const tensor_t * to) { return to && to->type != 14 && to->type != 8 && to->type != 2 && to->type != 3; }

// a token table that is not q6_K, q8_0, q4_0 or q4_K, or not one sllm-next reads
static bool table_type_bad(const tensor_t * te) { return !te || (te->type != 14 && te->type != 8 && te->type != 2 && te->type != 12) || !mat_ok(te); }

// table_type_bad, or an output layer (to; NULL: tied) sllm-next does not read
static bool untied_table_bad(const tensor_t * te, const tensor_t * to) { return table_type_bad(te) || (to && !mat_ok(to)); }

// layer i's three K-quant experts (q4_K / q5_K / q6_K, whole blocks, n_expert of them) for the interactive rows' kernels;
// false with the first that is not in nm
static bool kq_experts_ok(gguf_t * g, char * nm, int i) {
    const char * ex[] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    for (int j = 0; j < 3; ++j) {
        tensor_t * t = blk_find(g, nm, i, ex[j], ".weight");
        if (!t || (t->type != 12 && t->type != 13 && t->type != 14) || t->ne[0] % 256 || t->ne[2] != g->n_expert) return false;
    }
    return true;
}

static id<MTLBuffer> gb(sn_engine * e, uint64_t off) {
    for (int i = 0; i < 4; ++i) if (e->wb[i] && off >= e->rb[i][0] && off < e->rb[i][1]) return e->wb[i];
    return nil;
}
static uint64_t go(sn_engine * e, uint64_t off) {
    for (int i = 0; i < 4; ++i) if (e->wb[i] && off >= e->rb[i][0] && off < e->rb[i][1]) return off - e->rb[i][0];
    return 0;
}

// Gemma 3's window layers (5 of each 6) and gpt-oss's (every other one): layer i's when the GGUF gives no pattern
static int window_layer(const gguf_t * g, int i) {
    return g->window > 0 && (!strcmp(g->arch, "gemma3") ? i % 6 < 5 : !strcmp(g->arch, "gpt-oss") ? i % 2 == 0 : 0);
}

// Gemma 3 / 4 and gpt-oss: the per-layer values a GGUF may give once for all layers, and the fixed window patterns (Gemma 3:
// 5 window layers, 1 full; gpt-oss: window, full)
static void gemma_defaults(gguf_t * g) {
    for (int i = 0; i < MIN(g->n_layer, 128); ++i) {
        if (!g->n_kva) g->kva[i] = g->n_kv;
        if (!g->n_ffa) g->ffa[i] = g->n_ff;
        if (!g->n_swa) g->swa[i] = window_layer(g, i);
    }
    g->n_kva = g->n_ffa = g->n_swa = MIN(g->n_layer, 128);
    if (!g->hd_swa) g->hd_swa = g->hd;
    if (!g->rope_base_swa) g->rope_base_swa = !strcmp(g->arch, "gpt-oss") ? g->rope_base : 10000.f;  // llama.cpp's defaults
}

// kernels.metal's g4_pre instances: head size x query heads per K/V head
static const int G4_PRE[][2] = {{128, 2}, {256, 2}, {256, 4}, {256, 8}, {512, 8}, {512, 16}};  // gpt-oss's 64 x 8: oss_check
static bool g4_pre_has(int D, int G) {
    bool ok = false;
    for (int j = 0; j < 6; ++j) ok |= G4_PRE[j][0] == D && G4_PRE[j][1] == G;
    return ok;
}

// layer i's heads: a g4_pre instance takes them, and the layers have at most two groupings of query heads (gs, ng: the
// ones so far); false with why set
static bool gemma_heads_ok(const gguf_t * g, int i, int * gs, int * ng, char * why, size_t len) {
    const int D = g->swa[i] ? g->hd_swa : g->hd, nkv = g->kva[i], G = nkv > 0 && g->n_head % nkv == 0 ? g->n_head / nkv : 0;
    if (!g4_pre_has(D, G)) { snprintf(why, len, "layer %d: head size %d with %d query heads per K/V head (sllm-next's Gemma kernels take 128 x 2, 256 x 2 / 4 / 8 and 512 x 8 / 16)", i, D, G); return false; }
    int k = 0;
    while (k < *ng && gs[k] != G) ++k;
    if (k == *ng && *ng == 2) { snprintf(why, len, "more than two query-head groupings"); return false; }
    if (k == *ng) gs[(*ng)++] = G;
    return true;
}

// layer i's matrix j (t, named nm): there, of a type sllm-next reads, q, k, o, gate and up of one type (f0: one_fmt's);
// false with why set
static bool layer_mat_ok(const tensor_t * t, const char * nm, int i, int j, int * f0, char * why, size_t len) {
    if (!t) { snprintf(why, len, "no tensor %s", nm); return false; }
    if (!mat_ok(t)) { snprintf(why, len, "%s is GGUF type %d (sllm-next reads q4_0, q4_1, q8_0, q4_K and q6_K matrices)", nm, t->type); return false; }
    if (!one_fmt(j, fmt_of(t->type), f0)) { snprintf(why, len, "layer %d's matrices are of different types", i); return false; }
    return true;
}

// a matrix layer i may lack: Gemma 4's v (v = k), the shared-KV layers' (from KVL) k and v
static bool gemma_mat_optional(const gguf_t * g, int i, int j, int KVL) {
    return (j == 2 && !strcmp(g->arch, "gemma4")) || ((j == 1 || j == 2) && i >= KVL);
}

// layer i's matrices: there (gemma_mat_optional), of types sllm-next reads, q, k, o, gate and up of one type
static bool gemma_mats_ok(gguf_t * g, int i, int KVL, char * why, size_t len) {
    char nm[128];
    int f0 = -2;
    for (int j = 0; j < 7; ++j) {
        tensor_t * t = blk_find(g, nm, i, LAYER_MATS[j], ".weight");
        if (!t && gemma_mat_optional(g, i, j, KVL)) continue;  // Gemma 4: v = k
        if (!layer_mat_ok(t, nm, i, j, &f0, why, len)) return false;
    }
    return true;
}

// layer i's norms (with per-layer embeddings: its gate, projection and norm), and Gemma 4's full layers' frequency factors
static bool gemma_vecs_ok(gguf_t * g, int i, char * why, size_t len) {
    char nm[128];
    const char * vecs[] = {"attn_norm", "ffn_norm", "attn_q_norm", "post_attention_norm", "post_ffw_norm", "inp_gate", "proj", "post_norm"};
    for (int j = 0; j < (g->ple ? 8 : 5); ++j)
        if (!blk_find(g, nm, i, vecs[j], ".weight")) { snprintf(why, len, "no tensor %s", nm); return false; }
    if (!g->swa[i] && !strcmp(g->arch, "gemma4") && !find_opt(g, "rope_freqs.weight")) { snprintf(why, len, "no tensor rope_freqs.weight"); return false; }
    return true;
}

// per-layer embeddings other than 256 a layer from a q5_K table through a bf16 projection
static bool gemma_ple_bad(gguf_t * g) {
    tensor_t * tp = find_opt(g, "per_layer_token_embd.weight"), * pp = find_opt(g, "per_layer_model_proj.weight");
    return g->ple != 256 || !tp || tp->type != 13 || !pp || pp->type != 30 || !find_opt(g, "per_layer_proj_norm.weight");
}

// a Gemma token table that is not q4_K, q6_K, q8_0 or q4_0, or a K-quant one of a width of part blocks
static bool gemma_table_bad(const gguf_t * g, const tensor_t * te) {
    return !te || (te->type != 12 && te->type != 14 && te->type != 8 && te->type != 2) || ((te->type == 12 || te->type == 14) && g->n_embd % 256);
}

// layer i: its heads (gs, ng: the groupings so far), matrices (KVL: the first layer without its own K/V) and vectors
static bool gemma_layer_ok(gguf_t * g, int i, int KVL, int * gs, int * ng, char * why, size_t len) {
    return gemma_heads_ok(g, i, gs, ng, why, len) && gemma_mats_ok(g, i, KVL, why, len) && gemma_vecs_ok(g, i, why, len);
}

static void gemma_check(gguf_t * g, char * why, size_t len) {
    gemma_defaults(g);
    if (g->n_layer > 128 || g->n_head < 1) { snprintf(why, len, "%d layers, %d heads", g->n_layer, g->n_head); return; }
    const int KVL = g->n_layer - g->shared_kv;
    int gs[2], ng = 0;
    for (int i = 0; i < g->n_layer; ++i)
        if (!gemma_layer_ok(g, i, KVL, gs, &ng, why, len)) return;
    if (g->ple && gemma_ple_bad(g)) {
        snprintf(why, len, "per-layer embeddings: 256 a layer, a q5_K table and a bf16 projection");
        return;
    }
    tensor_t * te = find_opt(g, "token_embd.weight"), * to = find_opt(g, "output.weight");
    if (gemma_table_bad(g, te))
        snprintf(why, len, "the token table must be q4_K, q6_K, q8_0 or q4_0");
    else if (head_type_bad(to))
        snprintf(why, len, "the output layer must be q6_K, q8_0, q4_0 or q4_1");
}

// gpt-oss's heads: 64 wide, 8 query heads a K/V head
static bool oss_heads_ok(const gguf_t * g, char * why, size_t len) {
    if (g->n_layer > 128 || g->hd != 64 || g->n_kv < 1 || g->n_head != 8 * g->n_kv) {
        snprintf(why, len, "head size %d with %d query heads per K/V head (sllm-next's gpt-oss kernels take 64 x 8)", g->hd, g->n_kv ? g->n_head / g->n_kv : 0);
        return false;
    }
    return true;
}

// gpt-oss's experts: 1 .. 512 of them, 1 .. 16 used, widths whole 32s
static bool oss_experts_ok(const gguf_t * g, char * why, size_t len) {
    if (g->n_expert < 1 || g->n_expert > 512 || g->n_expert_used < 1 || g->n_expert_used > 16 || g->n_ff_exp < 1 || g->n_ff_exp % 32 || g->n_embd % 32) {
        snprintf(why, len, "gpt-oss: %d experts, %d used, %d wide", g->n_expert, g->n_expert_used, g->n_ff_exp);
        return false;
    }
    return true;
}

// layer i's experts: mxfp4 or K-quant matrices of n_expert experts
static bool oss_layer_experts_ok(gguf_t * g, int i, char * why, size_t len) {
    char nm[128];
    const char * exps[] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    for (int j = 0; j < 3; ++j) {
        tensor_t * t = blk_find(g, nm, i, exps[j], ".weight");
        const int f = t ? fmt_of(t->type) : -1;
        if (!t || !mat_ok(t) || f < 6 || t->ne[2] != g->n_expert) { snprintf(why, len, "%s: experts sllm-next reads are mxfp4, q4_K, q5_K or q6_K", nm); return false; }
    }
    return true;
}

// layer i's matrix j (q / k / v / output; its name into nm): one sllm-next reads, q, k and o of one type (f0: one_fmt's)
static bool oss_mat_ok(gguf_t * g, char * nm, int i, int j, int * f0, char * why, size_t len) {
    tensor_t * t = blk_find(g, nm, i, LAYER_MATS[j], ".weight");
    if (!t || !mat_ok(t) || fmt_of(t->type) > 8) { snprintf(why, len, "%s: not a matrix sllm-next reads", nm); return false; }
    if (!one_fmt(j, fmt_of(t->type), f0)) { snprintf(why, len, "layer %d's matrices are of different types", i); return false; }
    return true;
}

// layer i: q / k / v / output matrices (q, k, o of one type), the f32 biases, sinks, norms and router, and the experts
static bool oss_layer_ok(gguf_t * g, int i, char * why, size_t len) {
    char nm[128];
    int f0 = -2;
    for (int j = 0; j < 4; ++j)
        if (!oss_mat_ok(g, nm, i, j, &f0, why, len)) return false;
    const char * vecs[] = {"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias", "attn_sinks.weight", "attn_norm.weight",
                           "post_attention_norm.weight", "ffn_gate_inp.weight", "ffn_gate_inp.bias", "ffn_gate_exps.bias", "ffn_up_exps.bias",
                           "ffn_down_exps.bias"};
    if (!blk_f32_all(g, nm, i, vecs, 12, "")) { snprintf(why, len, "%s must be f32", nm); return false; }
    return oss_layer_experts_ok(g, i, why, len);
}

// a gpt-oss token table that is not q6_K, q8_0 or q4_0
static bool oss_table_bad(const tensor_t * te) { return !te || (te->type != 14 && te->type != 8 && te->type != 2); }

// gpt-oss: Gemma's layer machinery (its window layers' pool) with q / k / v / output biases, a sink per head, YaRN and
// MoE layers (MXFP4 or K-quant experts with biases)
static void oss_check(gguf_t * g, char * why, size_t len) {
    gemma_defaults(g);
    if (!oss_heads_ok(g, why, len) || !oss_experts_ok(g, why, len)) return;
    if (g->rope_scale <= 0 || g->ctx_orig <= 0) { snprintf(why, len, "gpt-oss: no YaRN scaling"); return; }
    for (int i = 0; i < g->n_layer; ++i)
        if (!oss_layer_ok(g, i, why, len)) return;
    tensor_t * te = find_opt(g, "token_embd.weight"), * to = find_opt(g, "output.weight");
    if (oss_table_bad(te))
        snprintf(why, len, "the token table must be q6_K, q8_0 or q4_0");
    else if (head_type_bad(to))
        snprintf(why, len, "the output layer must be q6_K, q8_0, q4_0 or q4_1");
}

// bytes of n weights of a block-quantized type (0: not one sn_open maps)
static uint64_t tsize_blocks(int type, uint64_t n) {
    switch (type) {
        case 2: return n / 32 * 18;     case 3: return n / 32 * 20;     case 8: return n / 32 * 34;
        case 12: return n / 256 * 144;  case 13: return n / 256 * 176;  case 14: return n / 256 * 210;
        case 39: return n / 32 * 17;
    }
    return 0;
}

// bytes of a tensor (the types sn_open maps)
static uint64_t tsize(const tensor_t * t) {
    const uint64_t n = (uint64_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
    switch (t->type) {
        case 0: return n * 4;           case 1: case 30: return n * 2;
    }
    return tsize_blocks(t->type, n);
}

// an MoE model's step buffers (moe_ffn's) for R rows of k_used pairs each; tiles of 16 pairs, at most one partial tile an
// expert; the interactive pairs' (4 rows) apart
static void moe_alloc(sn_engine * e, id<MTLDevice> dev, int R) {
    const gguf_t * g = &e->g;
    const int E = e->E, NE = g->n_expert, KU = g->n_expert_used, FE = g->n_ff_exp, FS = g->n_ff_shexp, P = R * KU;
    const MTLResourceOptions PRIV = MTLResourceStorageModePrivate;
    e->ntiles = (P + 7) / 8 + NE;  // tiles of 8 pairs or more
    e->mlog = [dev newBufferWithLength:(size_t) R * NE * 4 options:PRIV];
    e->mids = [dev newBufferWithLength:(size_t) P * 4 options:PRIV];
    e->mw = [dev newBufferWithLength:(size_t) P * 4 options:PRIV];
    e->mpos = [dev newBufferWithLength:(size_t) P * 4 options:PRIV];
    e->mlist = [dev newBufferWithLength:(size_t) P * 4 options:PRIV];
    e->mtiles = [dev newBufferWithLength:(size_t) e->ntiles * 16 options:PRIV];
    e->mxg = [dev newBufferWithLength:(size_t) P * E * 2 options:PRIV];
    e->mact = [dev newBufferWithLength:(size_t) P * FE * 4 options:PRIV];
    e->mxg2 = [dev newBufferWithLength:(size_t) P * FE * 2 options:PRIV];
    e->myd = [dev newBufferWithLength:(size_t) P * E * 4 options:PRIV];
    e->mai = [dev newBufferWithLength:(size_t) 4 * KU * FE * 4 options:PRIV];
    e->myi = [dev newBufferWithLength:(size_t) 4 * KU * E * 4 options:PRIV];
    e->msgt = [dev newBufferWithLength:(size_t) R * 4 options:PRIV];
    e->mshg = [dev newBufferWithLength:(size_t) R * MAX(FS, 1) * 4 options:PRIV];
    e->mshx = [dev newBufferWithLength:(size_t) (FS ? R * E : 1) * 4 options:PRIV];
    if (e->oss) {
        e->mau = [dev newBufferWithLength:(size_t) 4 * KU * FE * 4 options:PRIV];
        e->mact2 = [dev newBufferWithLength:(size_t) P * FE * 4 options:PRIV];
    }
    e->FMAX = MAX(e->FMAX, MAX(FE, FS));
}

// layer i's experts (both MoE architectures): gate / up / down with their formats and strides, the router; gpt-oss's
// router and expert biases (0: none)
static void experts(gguf_t * g, int i, moe_t * m) {
    char nm[128];
    const char * sfx[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    uint64_t * w[3] = {&m->eg, &m->eu, &m->ed}, * b[3] = {&m->bg, &m->bu, &m->bd};
    int * f[3] = {&m->f_eg, &m->f_eu, &m->f_ed}, * st[3] = {&m->s_eg, &m->s_eu, &m->s_ed};
    for (int j = 0; j < 3; ++j) {
        snprintf(nm, sizeof nm, "blk.%d.%s.weight", i, sfx[j]);
        tensor_t * t = find(g, nm);
        *w[j] = g->data + t->off; *f[j] = fmt_of(t->type); *st[j] = (int) (tsize(t) / t->ne[2]);
        snprintf(nm, sizeof nm, "blk.%d.%s.bias", i, sfx[j]);
        tensor_t * tb = find_opt(g, nm);
        *b[j] = tb ? g->data + tb->off : 0;
    }
    snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.weight", i);
    m->router = g->data + find(g, nm)->off;
    snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.bias", i);
    tensor_t * tr = find_opt(g, nm);
    m->router_b = tr ? g->data + tr->off : 0;
}

enum : uint64_t { SIDE_BASE = 1ull << 48 };
// Whether sllm-next can run the guess-head block blk.L of a qwen35moe GGUF: a full-attention block of the model's shapes with
// K-quant experts and an f32 router and shared gate (as graft.py writes them), eh_proj over 2 E inputs, its norms f32
static bool mtp_ok(gguf_t * g, int L) {
    char nm[128];
    const char * mats[] = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "nextn.eh_proj"};
    const char * vecs[] = {"attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm", "nextn.enorm", "nextn.hnorm", "ffn_gate_inp", "ffn_gate_inp_shexp"};
    for (int j = 0; j < 8; ++j) {
        tensor_t * t = blk_find(g, nm, L, mats[j], ".weight");
        if (!t || !mat_ok(t)) return false;
        if (j == 7 && (t->ne[0] != 2 * (uint64_t) g->n_embd || t->ne[1] != (uint64_t) g->n_embd)) return false;
    }
    return blk_f32_all(g, nm, L, vecs, 8, ".weight") && kq_experts_ok(g, nm, L);
}

// The GPU's buffers (up to 3): the file without the nh tables in holes[] (byte ranges) that the GPU never reads in place
static void holes_sort(uint64_t holes[][2], int nh) {
    for (int i = 0; i < nh; ++i)
        for (int j = i + 1; j < nh; ++j)
            if (holes[j][0] < holes[i][0]) { uint64_t a0 = holes[i][0], a1 = holes[i][1]; holes[i][0] = holes[j][0]; holes[i][1] = holes[j][1]; holes[j][0] = a0; holes[j][1] = a1; }
}
static int map_file(sn_engine * e, uint64_t holes[][2], int nh) {
    const gguf_t * g = &e->g;
    const size_t page = getpagesize();
    holes_sort(holes, nh);
    uint64_t at = 0;
    for (int i = 0; i <= nh; ++i) {
        const uint64_t end = i < nh ? (holes[i][0] + page - 1) / page * page : g->size;
        e->rb[i][0] = at; e->rb[i][1] = end;
        e->wb[i] = end > at ? [dev newBufferWithBytesNoCopy:(void *) (g->base + at) length:end - at options:MTLResourceStorageModeShared deallocator:nil] : nil;
        if (end > at && !e->wb[i]) return -1;
        if (i < nh) at = holes[i][1] / page * page;
    }
    return 0;
}

// Qwen3.5-MoE's experts: 1 .. 512, 1 .. 16 used
static bool q35_moe_shape_bad(const gguf_t * g) {
    return g->n_expert < 1 || g->n_expert > 512 || g->n_expert_used < 1 || g->n_expert_used > 16 || g->n_ff_exp < 1;
}
// Qwen3.5's heads (256 wide, rope 64, G query heads a K/V head) and Gated DeltaNet shapes sllm-next runs
static bool q35_heads_bad(const gguf_t * g, int G) {
    return g->hd != 256 || g->n_rot != 64 || (G != 2 && G != 4 && G != 6 && G != 8);
}
static bool q35_ssm_bad(const gguf_t * g) {
    return g->ssm_state != 128 || g->ssm_conv != 4 || g->fa_interval < 1 || g->ssm_groups < 1 || g->ssm_inner != g->ssm_rank * 128 || g->n_layer > 128;
}

// a layer matrix sllm-next cannot read (f32_ok: alpha / beta, which may be f32: Qwen3.6)
static bool q35_mat_bad(const tensor_t * t, bool f32_ok) { return !t || (!mat_ok(t) && !(f32_ok && t->type == 0)); }

// the FFN's matrices: dense, or (MoE) the shared expert's
static const char * const Q35_FFN[2][3] = {{"ffn_gate", "ffn_up", "ffn_down"}, {"ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"}};

// layer i's matrices (lin: a linear-attention layer's, else an attention layer's)
static bool q35_mats_ok(gguf_t * g, int i, bool lin, bool moe, char * why, size_t len) {
    char nm[128];
    const char * const * ffn = Q35_FFN[moe];
    const char * mats[] = {"attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out", ffn[0], ffn[1], ffn[2]};
    const char * amats[] = {"attn_q", "attn_k", "attn_v", "attn_output", ffn[0], ffn[1], ffn[2]};
    for (int j = 0; j < (lin ? 8 : 7); ++j) {
        tensor_t * t = blk_find(g, nm, i, lin ? mats[j] : amats[j], ".weight");
        const bool f32_ok = lin && (j == 2 || j == 3);  // alpha / beta may be f32 (Qwen3.6)
        if (q35_mat_bad(t, f32_ok)) { snprintf(why, len, "%s: %s", nm, t ? "a type sllm-next does not read" : "missing"); return false; }
    }
    return true;
}

// layer i's f32 vectors, and (MoE) its experts: K-quants (the interactive rows' kernels), the router and the shared expert's
// gate f32
static bool q35_vecs_ok(gguf_t * g, int i, bool lin, bool moe, char * why, size_t len) {
    char nm[128];
    const char * vecs[] = {"attn_norm.weight", "post_attention_norm.weight", "ssm_a", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_norm.weight"};
    const char * avecs[] = {"attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight"};
    if (!blk_f32_all(g, nm, i, lin ? vecs : avecs, lin ? 6 : 4, "")) { snprintf(why, len, "%s: not an f32 tensor", nm); return false; }
    if (!moe) return true;
    if (!kq_experts_ok(g, nm, i)) { snprintf(why, len, "%s: not q4_K / q5_K / q6_K experts", nm); return false; }
    const char * fv[] = {"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"};
    if (!blk_f32_all(g, nm, i, fv, 2, "")) { snprintf(why, len, "%s: not an f32 tensor", nm); return false; }
    return true;
}

// layer i: a linear-attention layer (but every fa_interval-th, an attention layer)
static bool q35_layer_ok(gguf_t * g, int i, bool moe, char * why, size_t len) {
    const bool lin = (i + 1) % g->fa_interval != 0;
    return q35_mats_ok(g, i, lin, moe, why, len) && q35_vecs_ok(g, i, lin, moe, why, len);
}

// Qwen3.5 / 3.6: Gated DeltaNet layers (128-wide heads, a 4-tap conv) and full attention layers every fa_interval
static void q35_check(gguf_t * g, char * why, size_t len) {
    const bool moe = !strcmp(g->arch, "qwen35moe");
    if (moe && q35_moe_shape_bad(g)) {
        snprintf(why, len, "qwen35moe: %d experts, %d used", g->n_expert, g->n_expert_used);
        return;
    }
    const int G = g->n_kv ? g->n_head / g->n_kv : 0;
    if (q35_heads_bad(g, G) || q35_ssm_bad(g)) {
        snprintf(why, len, "qwen35 shapes sllm-next does not run (head %d, rope %d, %d query heads a K/V head, state %d, conv %d)", g->hd, g->n_rot, G, g->ssm_state, g->ssm_conv);
        return;
    }
    for (int i = 0; i < g->n_layer; ++i)
        if (!q35_layer_ok(g, i, moe, why, len)) return;
    tensor_t * te = find_opt(g, "token_embd.weight"), * to = find_opt(g, "output.weight");
    if (untied_table_bad(te, to))
        snprintf(why, len, "the token table must be q6_K, q4_K, q8_0 or q4_0");
}

// Phi-3's layer i: fused q / k / v and gate / up, each matrix of its rows, the first three of the first layer's type (t0)
static void phi3_layer_check(gguf_t * g, int i, const tensor_t * t0, char * why, size_t len) {
    char nm[128];
    const char * mats[] = {"attn_qkv", "attn_output", "ffn_up", "ffn_down"};
    const int64_t rows[] = {(int64_t) (g->n_head + 2 * g->n_kv) * g->hd, g->n_embd, 2 * (int64_t) g->n_ff, g->n_embd};
    for (int j = 0; j < 4 && !why[0]; ++j) {
        tensor_t * t = blk_find(g, nm, i, mats[j], ".weight");
        if (!t || !mat_ok(t) || t->ne[1] != rows[j]) snprintf(why, len, "%s: not a matrix of %lld rows sllm-next reads", nm, (long long) rows[j]);
        else if (j < 3 && (!t0 || t->type != t0->type)) snprintf(why, len, "layer %d's matrices are of different types", i);
    }
}

// Phi-3: fused q / k / v and gate / up, rope factors
static void phi3_check(gguf_t * g, char * why, size_t len) {
    tensor_t * t0 = find_opt(g, "blk.0.attn_qkv.weight");
    for (int i = 0; i < g->n_layer && !why[0]; ++i) phi3_layer_check(g, i, t0, why, len);
    tensor_t * te = find_opt(g, "token_embd.weight"), * to = find_opt(g, "output.weight");
    if (!why[0] && (!find_opt(g, "rope_factors_short.weight") || !find_opt(g, "rope_factors_long.weight"))) snprintf(why, len, "no rope factors");
    else if (!why[0] && untied_table_bad(te, to))
        snprintf(why, len, "the token table must be q6_K, q4_K, q8_0 or q4_0");
}

// a dense layer i's matrices: there, of types sllm-next reads, q, k, o, gate and up of one type
static void dense_mats_check(gguf_t * g, int i, char * why, size_t len) {
    char nm[128];
    int f0 = -2;  // (v may differ: Q4_K_M files, q4_K or q6_K)
    for (int j = 0; j < 7 && !why[0]; ++j) layer_mat_ok(blk_find(g, nm, i, LAYER_MATS[j], ".weight"), nm, i, j, &f0, why, len);
}

// Qwen3's q / k norms and Qwen2's q / k / v biases (f32) of layer i
static void dense_extras_check(gguf_t * g, int i, char * why, size_t len) {
    char nm[128];
    snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", i);
    if (!why[0] && !strcmp(g->arch, "qwen3") && !find_opt(g, nm)) snprintf(why, len, "no tensor %s", nm);
    const char * bias[] = {"attn_q", "attn_k", "attn_v"};
    if (!why[0] && !strcmp(g->arch, "qwen2") && !blk_f32_all(g, nm, i, bias, 3, ".bias")) snprintf(why, len, "%s must be an f32 bias", nm);
}

// Qwen2 / 3, Llama, SmolLM3: the layers' matrices, Qwen3's norms, Qwen2's biases, the token table and the output layer
static void dense_check(gguf_t * g, char * why, size_t len) {
    for (int i = 0; i < g->n_layer && !why[0]; ++i) {
        dense_mats_check(g, i, why, len);
        dense_extras_check(g, i, why, len);
    }
    tensor_t * te = find_opt(g, "token_embd.weight"), * to = find_opt(g, "output.weight");
    if (!to) to = te;
    if (!why[0] && table_type_bad(te))
        snprintf(why, len, "the token table must be q6_K, q4_K, q8_0 or q4_0");
    else if (!why[0] && !mat_ok(to))
        snprintf(why, len, "the output layer must be q6_K, q4_K, q8_0, q4_0 or q4_1");
}

// an architecture none of the families' checks takes
static bool dense_arch_unknown(const char * a) {
    return strcmp(a, "qwen3") && strcmp(a, "qwen2") && strcmp(a, "llama") && strcmp(a, "smollm3") && strcmp(a, "phi3");
}
// the dense attention kernels' heads: 128 wide with 2, 4, 7 or 8 query heads a K/V head, or 96 with 1
static bool dense_heads_bad(int hd, int G) {
    return !(hd == 128 && (G == 2 || G == 4 || G == 7 || G == 8)) && !(hd == 96 && G == 1);
}

// the family's check of g (why: the reason it cannot run, empty when it can)
static void arch_check(gguf_t * g, char * why, size_t len) {
    const int G = g->n_kv ? g->n_head / g->n_kv : 0;
    if (!strcmp(g->arch, "gemma4") || !strcmp(g->arch, "gemma3")) {
        gemma_check(g, why, len);
    } else if (!strcmp(g->arch, "qwen35") || !strcmp(g->arch, "qwen35moe")) {
        q35_check(g, why, len);
    } else if (!strcmp(g->arch, "gpt-oss")) {
        oss_check(g, why, len);
    } else if (dense_arch_unknown(g->arch)) {
        snprintf(why, len, "architecture %s (sllm-next runs qwen2, qwen3, qwen35, qwen35moe, llama, smollm3, phi3, gemma3, gemma4 and gpt-oss)", g->arch);
    } else if (dense_heads_bad(g->hd, G)) {
        snprintf(why, len, "head size %d with %d query heads per K/V head (sllm-next takes 128 with 2, 4, 7 or 8, and 96 with 1)", g->hd, G);
    } else if (!strcmp(g->arch, "phi3")) {  // fused q / k / v and gate / up, rope factors
        phi3_check(g, why, len);
    } else {
        dense_check(g, why, len);
    }
}

// What sn_open needs of a model, from the GGUF's header and tensor table alone (0, or -1 with the reason in err)
static int model_check(gguf_t * g, const char * path, char * err, int err_len) {
    char why[256];
    why[0] = 0;
    if (!g->hd && g->n_head) g->hd = g->n_embd / g->n_head;  // Llama's GGUFs may leave the head size out
    arch_check(g, why, sizeof why);
    if (!why[0]) return 0;
    if (err && err_len > 0) snprintf(err, err_len, "%s: %s", path, why);
    return -1;
}

// The prompt rows a step for a model that reads faster in bigger steps when no chat is about (0: none). A MoE step pays the
// bytes of every expert it touches once (Qwen3.6-35B-A3B: about 66 ms when all 256 are), so more rows a step share them:
// a 4,000-token read 2,283 -> 2,455 tok/s, 8 agents of 1,000 tokens 2,309-2,383 -> 2,558; the step buffers +365 MB.
int sn_quiet_rows(const char * path) {
    @autoreleasepool {
        if (getenv("SN_QUIET_ROWS")) return atoi(getenv("SN_QUIET_ROWS"));
        if (access(path, R_OK) != 0) return 0;
        gguf_t g;
        memset(&g, 0, sizeof g);
        gguf_open(path, &g);
        const int r = g.n_expert >= 128 ? 2048 : 0;  // (gpt-oss, 32 experts: not measured)
        munmap((void *) g.base, g.size);
        free(g.t);
        return r;
    }
}

// whether some layer is a window layer, and the widest FFN and K/V heads over the layers (ff, kvh: the GGUF's single values)
static void layer_widest(const gguf_t * g, bool * swa, int * ff, int * kvh) {
    for (int i = 0; i < g->n_swa; ++i) *swa |= g->swa[i] != 0 && g->window > 0;
    for (int i = 0; i < g->n_ffa; ++i) *ff = MAX(*ff, g->ffa[i]);
    for (int i = 0; i < g->n_kva; ++i) *kvh = MAX(*kvh, g->kva[i]);
}

int sn_check(const char * path, int * rows, char * err, int err_len) {
    @autoreleasepool {
        if (access(path, R_OK) != 0) { set_err(err, err_len, "cannot read %s", path); return -1; }
        gguf_t g;
        memset(&g, 0, sizeof g);
        gguf_open(path, &g);
        const int rc = model_check(&g, path, err, err_len);
        bool swa = false;
        int ff = g.n_ff, kvh = g.n_kv;
        layer_widest(&g, &swa, &ff, &kvh);
        // a step's buffers a row (sn_open's x, h, xo, ho, q, ao, k, v, fg in float, the half copy of the widest input)
        const size_t row = (size_t) (4 * g.n_embd + 2 * g.n_head * g.hd + 2 * kvh * g.hd + ff) * 4 + (size_t) MAX(ff, g.n_embd) * 2;
        // MoE models 1024 (twice the tokens an expert a step: Qwen3.6-35B doc read +14%, gpt-oss +10%; one agent's RAM still
        // under ggml's: 35B 20.8-21.0 vs 21.1-21.3 GB, gpt-oss 11.2-11.4 vs 11.4-11.7 in alternating rounds)
        if (rows) *rows = g.n_expert > 0 ? 1024 : swa || row > 144 * 1024 ? 512 : 1024;  // Qwen3-4B 140 KB, SmolLM3 119: 1024; Phi-3.5 147, Llama 8B / Mistral 192, Qwen2.5 204: 512
        if (rows && getenv("SN_ROWS")) *rows = atoi(getenv("SN_ROWS"));  // (experiments: the prompt rows a step)
        munmap((void *) g.base, g.size);
        free(g.t);
        return rc;
    }
}
