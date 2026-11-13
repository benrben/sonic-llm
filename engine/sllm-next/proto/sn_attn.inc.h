// sllm-next engine library, a part of sn.m (one translation unit; sn.m includes its parts in order): the step's kernels: attention, the Qwen3 / Llama layer program, f32 mat-vecs, the MoE experts' matmuls

static void dispatch_rows(id<MTLComputeCommandEncoder> c, NSString * name, int rows, int threads) {
    [c setComputePipelineState:K(name)];
    if ((g_skip & 8) && ![name hasPrefix:@"embed"] && ![name hasPrefix:@"g4_embed"] && ![name isEqual:@"gather"] && ![name isEqual:@"argmax"] && ![name isEqual:@"argmax_p"]) return;
    [c dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

// the embedding kernel of a token table of mm format fmt
static NSString * embed_name(int fmt) { return fmt == 5 ? @"embed_q8_0" : fmt == 0 ? @"embed_q4_0" : fmt == 6 ? @"embed_q4k" : @"embed_q6k"; }

// ---- attention ------------------------------------------------------------------------------------------------------
// a layer's attention as its kernels see it: q rows (ld = ldq = ldo wide), the layer's cache (K / V buffers from byte ko,
// the pool's page table), head size D with G query heads a K/V head, the window (0: none), the scale; gpt-oss's sinks
// (sinks_on: the merge and the prompt kernels read them in place of a scratch buffer)
typedef struct {
    __unsafe_unretained id<MTLBuffer> q, kc, vc, pt, sinks;
    size_t ko, sinks_o;
    int D, G, NH, NKV, ld, win, sinks_on;
    float scale;
} layer_attn;

// a write attention kernel's buffers: q, the layer's K and V, the rows' sequences and positions, the partials, the page table
static inline void attn_kv_bufs(id<MTLComputeCommandEncoder> c, const sn_engine * e, const layer_attn * a) {
    set3(c, 1, a->q, 0, a->kc, a->ko, a->vc, a->ko);
    set3(c, 4, e->rseq, 0, e->rpos, 0, e->apart, 0);
    [c setBuffer:a->pt offset:0 atIndex:7];
}

// a merge kernel's buffers 1 .. 3: the partials, the rows' positions, the output
static inline void attn_merge_bufs(id<MTLComputeCommandEncoder> c, const sn_engine * e) { set3(c, 1, e->apart, 0, e->rpos, 0, e->ao, 0); }

// a prompt q kernel's buffers: q, the tiles (list at byte bo), the staged tiles
static inline void prompt_q_bufs(id<MTLComputeCommandEncoder> c, const sn_engine * e, const layer_attn * a, size_t bo) { set3(c, 1, a->q, 0, e->rblk, bo, e->qhb, 0); }

// a prompt attention kernel's buffers 1 .. 7: the staged tiles, the layer's K and V, the tiles, the positions, the output,
// the page table
static inline void prompt_kv_bufs(id<MTLComputeCommandEncoder> c, const sn_engine * e, const layer_attn * a, size_t bo) {
    set3(c, 1, e->qhb, 0, a->kc, a->ko, a->vc, a->ko);
    set3(c, 4, e->rblk, bo, e->rpos, 0, e->ao, 0);
    [c setBuffer:a->pt offset:0 atIndex:7];
}

// Gemma's kernels: g4_attn_part's write rows (run: whether it dispatches; the encoder and arguments are set either way)
static inline void g4_attn_part(const sn_engine * e, const layer_attn * a, const g4a_args * aa, int n_write, int maxc_w, bool run) {
    const int hg = MIN(a->NH, 8);
    id<MTLComputeCommandEncoder> c = kernel_begin("attn", [NSString stringWithFormat:@"g4_attn_part_%d", a->D], aa, sizeof *aa);
    attn_kv_bufs(c, e, a);
    if (!(g_skip & 1) && run) [c dispatchThreadgroups:MTLSizeMake(n_write, a->NH / hg, maxc_w) threadsPerThreadgroup:MTLSizeMake(32 * hg, 1, 1)];
}

// Gemma's kernels: the write rows' partials merged in chunk order (with gpt-oss's sinks)
static inline void g4_attn_merge(const sn_engine * e, const layer_attn * a, const g4a_args * aa, int n_write) {
    id<MTLComputeCommandEncoder> c = kernel_begin("attn", [NSString stringWithFormat:@"g4_attn_merge_%d", a->D], aa, sizeof *aa);
    attn_merge_bufs(c, e);
    [c setBuffer:a->sinks_on ? a->sinks : e->apart offset:a->sinks_on ? a->sinks_o : 0 atIndex:4];
    if (!(g_skip & 17)) [c dispatchThreadgroups:MTLSizeMake(n_write, a->NH, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// Gemma's kernels, write rows: the interactive ones fa2 (fa2r: rows sharing threadgroups), the others g4_attn_part, merged
// in chunk order (Phi-3's head size 96, gpt-oss)
static void g4_attn_fa2(const sn_engine * e, const layer_attn * a, int n_write, int maxc_w) {
    const int NH = a->NH, NKV = a->NKV, D = a->D, G = a->G, hg = MIN(NH, 8);
    const int nA = g_fa1 ? MIN(g_nAtt, n_write) : 0, maxch = 1 << 20;
    g4a_args aa = {NH, NKV, e->ctx, a->scale, a->ld, a->ld, maxc_w, a->win, nA, maxch, hg, a->sinks_on, g_msp};  // partials maxc_w chunks apart
    if (nA > 0) {
        const int rpt = fa2_rpt(nA, G);
        fa1_args fa = {NH, NKV, e->ctx, a->scale, a->ld, maxc_w, a->win, NKV * D, maxch, rpt, nA, g_msp};
        id<MTLComputeCommandEncoder> c = kernel_begin("attn", [NSString stringWithFormat:rpt > 1 ? @"fa2r_%d_g%d" : @"fa2_%d_g%d", D, G], &fa, sizeof fa);
        attn_kv_bufs(c, e, a);
        if (!(g_skip & 1)) [c dispatchThreadgroups:MTLSizeMake((nA + rpt - 1) / rpt, NKV, fa2_z(maxc_w, a->win)) threadsPerThreadgroup:MTLSizeMake(32 * G * rpt, 1, 1)];
    }
    if (nA < n_write) g4_attn_part(e, a, &aa, n_write, maxc_w, true);
    g4_attn_merge(e, a, &aa, n_write);
}

// whether one of the nA interactive rows spans more than maxch 64-cell chunks of its window (win; 0: every cell before it)
static bool fa1_long_rows(const sn_engine * e, int nA, int win, int maxch) {
    for (int r = 0; r < nA; ++r) {
        const int pr = ((const int *) e->rpos.contents)[r], lo = win > 0 ? MAX(0, pr - win + 1) : 0;
        if (pr / 64 - lo / 64 + 1 > maxch) return true;
    }
    return false;
}

// Gemma's kernels, write rows: the interactive ones with short attention spans fa1, the others g4_attn_part (a background
// row, or an interactive one with a long span), merged in chunk order (Gemma, Qwen3.5)
static void g4_attn_fa1(const sn_engine * e, const layer_attn * a, int n_write, int maxc_w) {
    const int NH = a->NH, NKV = a->NKV, D = a->D, hg = MIN(NH, 8);
    const int nA = g_fa1 ? MIN(g_nAtt, n_write) : 0, maxch = FA1_MAXCH_G4;
    g4a_args aa = {NH, NKV, e->ctx, a->scale, a->ld, a->ld, maxc_w, a->win, nA, maxch, hg};  // partials maxc_w chunks apart
    if (nA > 0) {
        fa1_args fa = {NH, NKV, e->ctx, a->scale, a->ld, maxc_w, a->win, NKV * D, maxch};
        id<MTLComputeCommandEncoder> c = kernel_begin("attn", [NSString stringWithFormat:@"fa1_%d", D], &fa, sizeof fa);
        attn_kv_bufs(c, e, a);
        if (!(g_skip & 1)) [c dispatchThreadgroups:MTLSizeMake(nA, NH, maxc_w) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    const bool part = nA < n_write || fa1_long_rows(e, nA, a->win, maxch);  // g4_attn_part has rows
    g4_attn_part(e, a, &aa, n_write, maxc_w, part);
    g4_attn_merge(e, a, &aa, n_write);
}

// Gemma's kernels, prompt rows: g4_pre_q stages nb tiles of 32 / G tokens (tile list at byte bo), g4_pre attends over the cache
static void g4_prompt(const sn_engine * e, const layer_attn * a, int nb, size_t bo) {
    const int NKV = a->NKV, D = a->D, G = a->G;
    g4p_args pa = {a->NH, NKV, e->ctx, a->scale, a->ld, a->ld, a->win, a->sinks_on};
    id<MTLComputeCommandEncoder> c = kernel_begin("attn", [NSString stringWithFormat:@"g4_pre_q_%d_g%d", D, G], &pa, sizeof pa);
    prompt_q_bufs(c, e, a, bo);
    [c dispatchThreads:MTLSizeMake(D / 4, 32, nb * NKV) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    c = kernel_begin("attn", [NSString stringWithFormat:@"g4_pre_%d_g%d", D, G], &pa, sizeof pa);
    prompt_kv_bufs(c, e, a, bo);
    [c setBuffer:a->sinks_on ? a->sinks : e->ao offset:a->sinks_on ? a->sinks_o : 0 atIndex:8];
    [c dispatchThreadgroups:MTLSizeMake(nb, NKV, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

// Qwen3: one layer program over the step's rows
// Attention of layer i for a head size other than 128 (Phi-3's 96): Gemma's kernels for that size over the Qwen path's
// cache (cells of NKV x D), scale 1 / sqrt(D): interactive rows fa2, the other write rows g4_attn_part, merged in chunk
// order; prompt rows g4_pre in tiles of 32 / G tokens
// cache layer li of pool 0 (Qwen3 / Llama, Qwen3.5) as the attention kernels read it: q rows ld wide, head size g.hd
static inline layer_attn pool0_attn(const sn_engine * e, id<MTLBuffer> q, int li, int ld) {
    const int NH = e->NH, NKV = e->g.n_kv, D = e->g.hd;
    return (layer_attn) {q, e->pl[0].kc, e->pl[0].vc, e->pl[0].pt, nil, e->kv_layer * li * 2, 0, D, NH / NKV, NH, NKV, ld, 0, 0, 1.0f / sqrtf((float) D)};
}

static void attn_d(sn_engine * e, int i, int n_write, int nblk, int maxc_w) {
    const layer_attn a = pool0_attn(e, e->q, i, e->QMAX);
    if (n_write > 0) g4_attn_fa2(e, &a, n_write, maxc_w);
    if (nblk > 0) g4_prompt(e, &a, nblk, 0);
}

// rmsnorm of the step's R residual rows (x) with weight w into h
static inline void x_norm(sn_engine * e, const norm_args * nrm, uint64_t w, int R) {
    enc("rmsnorm"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:nrm length:sizeof *nrm atIndex:0];
    [c setBuffer:e->x offset:0 atIndex:1]; [c setBuffer:gb(e, w) offset:go(e, w) atIndex:2]; [c setBuffer:e->h offset:0 atIndex:3];
    dispatch_rows(c, @"rmsnorm", R, 256);
}

// Qwen3's q / k norms (Llama: none), rope (SmolLM3: none every nope_every-th layer) and the K / V cache write of layer i
static inline void qwen_rope(sn_engine * e, const layer_t * l, int i, int R, qkr_args * qa) {
    enc("qk_rope"); id<MTLComputeCommandEncoder> c = g_e;
    qa->mode = e->qk_mode | (e->nope_every && (i + 1) % e->nope_every == 0 ? 8 : 0);
    [c setBytes:qa length:sizeof *qa atIndex:0];
    [c setBuffer:e->q offset:0 atIndex:1]; [c setBuffer:e->k offset:0 atIndex:2]; [c setBuffer:e->v offset:0 atIndex:3];
    [c setBuffer:gb(e, l->q_norm) offset:go(e, l->q_norm) atIndex:4]; [c setBuffer:gb(e, l->k_norm) offset:go(e, l->k_norm) atIndex:5];
    [c setBuffer:e->pl[0].kc offset:e->kv_layer * i * 2 atIndex:6]; [c setBuffer:e->pl[0].vc offset:e->kv_layer * i * 2 atIndex:7];
    [c setBuffer:e->rseq offset:0 atIndex:8]; [c setBuffer:e->rpos offset:0 atIndex:9]; [c setBuffer:gb(e, e->rope_ff) offset:go(e, e->rope_ff) atIndex:10]; [c setBuffer:e->pl[0].pt offset:0 atIndex:11];
    [c setBuffer:gb(e, l->qb) offset:go(e, l->qb) atIndex:12]; [c setBuffer:gb(e, l->kb) offset:go(e, l->kb) atIndex:13]; [c setBuffer:gb(e, l->vb) offset:go(e, l->vb) atIndex:14];
    [c setComputePipelineState:K(e->g.hd == 128 ? @"qk_rope" : [NSString stringWithFormat:@"qk_rope_d%d", e->g.hd])];
    [c dispatchThreadgroups:MTLSizeMake(R, 8, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];  // kernels.metal's QKR_TGS
}

// Qwen3's q, k, v of R rows (v of its own type by itself)
static inline void qwen_qkv(sn_engine * e, const layer_t * l, int R) {
    const int E = e->E, QD = e->QMAX, KD = e->KD;
    const int M[3] = {QD, KD, KD}, ld[3] = {QD, KD, KD};
    id<MTLBuffer> w[3] = {gb(e, l->q), gb(e, l->k), gb(e, l->v)}, y[3] = {e->q, e->k, e->v};
    const uint64_t wo[3] = {go(e, l->q), go(e, l->k), go(e, l->v)};
    mm_qkv(l->fmt, e->h, E, R, l->v_fmt == l->fmt ? 3 : 2, w, wo, M, y, ld);
    if (l->v_fmt != l->fmt) mm(g_e, l->v_fmt, e->h, 0, w[2], wo[2], nil, e->v, 0, E, KD, R, E, KD, 0);  // v of its own type
}

// the interactive rows' one-row write attention: fa2 (fa2r: rows sharing threadgroups) per KV head, or fa1
static void qwen_attn_rows(const sn_engine * e, const layer_attn * a, int nA, int maxc_w, int maxch) {
    const int NH = a->NH, NKV = a->NKV, QD = a->ld;
    const int rpt = g_fa2 ? fa2_rpt(nA, NH / NKV) : 1;
    fa1_args fa = {NH, NKV, e->ctx, 1.0f / sqrtf(128.f), QD, maxc_w, 0, e->KD, maxch, rpt, nA, g_msp};
    id<MTLComputeCommandEncoder> c = kernel_begin("attn", g_fa2 ? [NSString stringWithFormat:rpt > 1 ? @"fa2r_128_g%d" : @"fa2_128_g%d", NH / NKV] : @"fa1_128", &fa, sizeof fa);
    attn_kv_bufs(c, e, a);
    if (!(g_skip & 1) && g_fa2) [c dispatchThreadgroups:MTLSizeMake((nA + rpt - 1) / rpt, NKV, fa2_z(maxc_w, 0)) threadsPerThreadgroup:MTLSizeMake(32 * (NH / NKV) * rpt, 1, 1)];
    else if (!(g_skip & 1)) [c dispatchThreadgroups:MTLSizeMake(nA, NH, MIN(maxc_w, FA1_MAXCH)) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

// write rows: chunked attention over their own cells, merged in order
static void qwen_attn_write(const sn_engine * e, int i, int n_write, int maxc_w) {
    const layer_attn a = pool0_attn(e, e->q, i, e->QMAX);
    const int NH = a.NH, NKV = a.NKV, QD = a.ld;
    // interactive rows with short contexts take the one-row attention, every other write row attn_part (each row
    // by its own position, so the choice never depends on the other rows)
    const int nA = g_fa1 ? MIN(g_nAtt, n_write) : 0;
    const int maxch = g_fa2 ? 1 << 20 : FA1_MAXCH;  // fa2 takes every interactive row
    attn2_args a2 = {NH, NKV, e->ctx, 1.0f / sqrtf(128.f), QD, QD, maxc_w, nA, maxch, g_fa2 ? g_msp : 0};  // partials maxc_w chunks apart
    if (nA > 0) qwen_attn_rows(e, &a, nA, maxc_w, maxch);
    id<MTLComputeCommandEncoder> c = kernel_begin("attn", @"attn_part", &a2, sizeof a2);
    attn_kv_bufs(c, e, &a);
    if (!(g_skip & 1) && (nA < n_write || maxc_w > maxch)) [c dispatchThreadgroups:MTLSizeMake(n_write, NKV, maxc_w) threadsPerThreadgroup:MTLSizeMake(32 * (NH / NKV), 1, 1)];
    c = kernel_begin("attn", @"attn_merge", &a2, sizeof a2);
    attn_merge_bufs(c, e);
    [c setBuffer:e->apart offset:0 atIndex:4];
    if (!(g_skip & 17)) [c dispatchThreadgroups:MTLSizeMake(n_write, NH, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// prompt rows: the matrix-unit prompt attention over 8-token tiles
static void qwen_attn_prompt(const sn_engine * e, int i, int nblk) {
    const layer_attn a = pool0_attn(e, e->q, i, e->QMAX);
    const int NH = a.NH, NKV = a.NKV, QD = a.ld;
    attnp_args ap = {NH, NKV, e->ctx, 1.0f / sqrtf(128.f), QD, QD, g_pre2swap};
    id<MTLComputeCommandEncoder> c = kernel_begin("attn", NH / NKV == 4 ? @"pre2_q" : [NSString stringWithFormat:@"pre2_q_g%d", NH / NKV], &ap, sizeof ap);
    prompt_q_bufs(c, e, &a, 0);
    [c dispatchThreads:MTLSizeMake(32, 32, nblk * NKV) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    c = kernel_begin("attn", NH / NKV == 4 ? @"pre2" : [NSString stringWithFormat:@"pre2_g%d", NH / NKV], &ap, sizeof ap);
    prompt_kv_bufs(c, e, &a, 0);
    [c setBuffer:e->ao offset:0 atIndex:8];
    if (!(g_skip & 32)) [c dispatchThreadgroups:(g_pre2swap ? MTLSizeMake(nblk, NKV, 1) : MTLSizeMake(NKV, nblk, 1)) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];  // SN_SKIP=32: timing without the prompt attention
}

// the FFN: its norm, the gated unit, the down projection added to x
static inline void qwen_ffn(sn_engine * e, const layer_t * l, const norm_args * nrm, int R) {
    const int E = e->E, F = e->FMAX;
    x_norm(e, nrm, l->ffn_norm, R);
    mm_glu(l->fmt, e->h, E, R, F, gb(e, l->gate), go(e, l->gate), gb(e, l->up), go(e, l->up), e->fg, F, 2);
    mm(g_e, l->down_fmt, e->fg, 0, gb(e, l->down), go(e, l->down), nil, e->x, 0, F, E, R, F, E, 1);
}

static void qwen_layers(sn_engine * e, int R, int n_write, int nblk, int maxc_w) {
    const int E = e->E, NH = e->NH, NKV = e->g.n_kv, QD = e->QMAX, KD = e->KD;
    norm_args nrm = {E, e->g.eps, E, E};
    qkr_args qa = {NH, NKV, e->ctx, e->slots, e->g.eps, e->g.rope_base, QD, KD, e->qk_mode, e->mscale};
    for (int i = 0; i < e->L; ++i) {
        const layer_t * l = &e->ly[i];
        x_norm(e, &nrm, l->attn_norm, R);
        qwen_qkv(e, l, R);
        qwen_rope(e, l, i, R, &qa);
        if (e->g.hd != 128) attn_d(e, i, n_write, nblk, maxc_w);
        if (e->g.hd == 128 && n_write > 0) qwen_attn_write(e, i, n_write, maxc_w);
        if (e->g.hd == 128 && nblk > 0) qwen_attn_prompt(e, i, nblk);  // prompt rows: the matrix-unit prompt attention over 8-token tiles
        mm(g_e, l->fmt, e->ao, 0, gb(e, l->o), go(e, l->o), nil, e->x, 0, QD, E, R, QD, E, 1);
        qwen_ffn(e, l, &nrm, R);
    }
}

// Qwen3.5 attention of layer l over the cache (D 256, G query heads a K/V head, gemma's kernels): interactive rows fa1
// (long ones g4_attn_part), the other write rows g4_attn_part, merged in chunk order; prompt rows g4_pre
static void q35_attn(sn_engine * e, const q35l * l, int n_write, int nblk, int maxc_w) {
    const layer_attn a = pool0_attn(e, e->qa, l->li, e->NH * e->g.hd);
    if (n_write > 0) g4_attn_fa1(e, &a, n_write, maxc_w);
    if (nblk > 0) g4_prompt(e, &a, nblk, 0);
}

// SN_SKIP (timing only): 128 every f32 mat-vec, 16384 the one-output ones, 32768 the 32-output ones
static inline bool mvf_skip(int M) { return (g_skip & 128) || ((g_skip & 16384) && M == 1) || ((g_skip & 32768) && M == 32); }

// f32 weights for every row (mvf: any output count; a row's result its own): y[r][j] = W[j] . x[r]
static void mvf_rows(sn_engine * e, uint64_t w, id<MTLBuffer> x, id<MTLBuffer> y, size_t yo, int K_, int M, int R, int ldx, int ldy) {
    // the interactive rows (all rows when the outputs are not 32s) mvf_ks; the batch rows mm's f32 path on the matrix units
    const int nA = M % 32 ? R : MIN(g_nA, R);
    if (nA > 0) {
        struct { int K, M, R, ldx, ldy; } fa = {K_, M, nA, ldx, ldy};
        enc("mm f32"); id<MTLComputeCommandEncoder> c = g_e;
        [c setComputePipelineState:K(nA == 1 && g_mvf1 ? @"mvf_ks1" : @"mvf_ks")];  // K % 16 == 0 (q35_check / oss_check: the width a multiple of 32)
        [c setBytes:&fa length:sizeof fa atIndex:0];
        [c setBuffer:gb(e, w) offset:go(e, w) atIndex:1]; [c setBuffer:x offset:0 atIndex:2]; [c setBuffer:y offset:yo atIndex:3];
        if (!mvf_skip(M)) [c dispatchThreadgroups:MTLSizeMake(M, (nA + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    if (nA < R) {
        const int sv = g_nA;
        g_nA = 0;
        mm(g_e, 3, x, (size_t) nA * ldx * 4, gb(e, w), go(e, w), nil, y, yo + (size_t) nA * ldy * 4, K_, M, R - nA, ldx, ldy, 0);
        g_nA = sv;
    }
}

// SN_JOIN 2: two one-row f32 mat-vecs over the same x in one dispatch (mvf_ks1x2), only where mvf_rows would take mvf_ks1 for both
// (one row, g_mvf1, interactive rows >= 1 when the outputs are 32s); y0 / y1 at their own offsets
static bool mvf_join_ok(int M0, int M1, int R) {
    if (!(g_join & 2) || R != 1 || !g_mvf1) return false;
    return (M0 % 32 == 0 ? g_nA >= 1 : true) && (M1 % 32 == 0 ? g_nA >= 1 : true);
}
static void mvf_rows2(sn_engine * e, uint64_t w0, uint64_t w1, id<MTLBuffer> x, id<MTLBuffer> y0, size_t yo0, id<MTLBuffer> y1, size_t yo1, int K_, int M0, int M1, int ldx, int ldy) {
    struct { int K, M, R, ldx, ldy; } fa = {K_, M0, 1, ldx, ldy};
    id<MTLComputeCommandEncoder> c = kernel_begin("mm f32", @"mvf_ks1x2", &fa, sizeof fa);
    [c setBuffer:gb(e, w0) offset:go(e, w0) atIndex:1]; [c setBuffer:x offset:0 atIndex:2]; [c setBuffer:y0 offset:yo0 atIndex:3];
    [c setBuffer:gb(e, w1) offset:go(e, w1) atIndex:4]; [c setBuffer:y1 offset:yo1 atIndex:5];
    const bool sk0 = mvf_skip(M0);
    const bool sk1 = mvf_skip(M1);
    if (!(sk0 && sk1)) [c dispatchThreadgroups:MTLSizeMake(M0 + M1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

// SN_JOIN 8: n (2 or 3) one-row Q8_0 mat-vecs over the same x in one dispatch (mv_q8_0kj: mv_q8_0k_r1's arithmetic), only where mm() would take
// mv_q8_0k_r1 for each (one row, interactive, g_q8k, all fmt 5); returns false (nothing encoded) otherwise
static bool join_q8_ok(int n, const int * fmt, int R) {
    if (!(g_join & 8) || R != 1 || g_f32 || g_nA < 1 || !g_q8k) return false;
    for (int i = 0; i < n; ++i) if (fmt[i] != 5) return false;
    return true;
}

// mv_q8_0kj's arguments: each matrix's mv_args and where its threadgroups end (tg1, tg2); the threadgroups in all
typedef struct { mv_args m[3]; int tg1, tg2; } join_q8_args;
static int join_q8_fill(join_q8_args * a, int n, int K_, const int * M) {
    int tgs = 0;
    for (int i = 0; i < 3; ++i) {
        const int j = i < n ? i : 0;  // (unused third: matrix 0's arguments, no threadgroups)
        a->m[i] = (mv_args){K_, M[j], 1, K_, M[j], 0, K_ / 32 * 34, g_eps};
        if (i < n) tgs += (M[i] + 1) / 2;
        if (i == 0) a->tg1 = tgs;
        if (i == 1) a->tg2 = tgs;
    }
    if (n == 2) a->tg2 = tgs;
    return tgs;
}

static bool mm_join_q8(sn_engine * e, int n, const int * fmt, id<MTLBuffer> x, int K_, const uint64_t * w, const int * M, id<MTLBuffer> const * y, int R) {
    if (!join_q8_ok(n, fmt, R)) return false;
    g_xh_same = 0; g_xhn_ok = 0; g_xhn_want = 0; g_amode = 0;  // (what mm() takes on entry)
    if (g_skip & 2) return true;
    join_q8_args a;
    memset(&a, 0, sizeof a);
    const int tgs = join_q8_fill(&a, n, K_, M);
    char cat[40];
    snprintf(cat, sizeof cat, "mm %dx%d join", M[0], K_);
    enc(cat);
    id<MTLComputeCommandEncoder> c = g_e;
    [c setComputePipelineState:K(@"mv_q8_0kj")];
    [c setBytes:&a length:sizeof a atIndex:0];
    [c setBuffer:gb(e, w[0]) offset:go(e, w[0]) atIndex:1]; [c setBuffer:x offset:0 atIndex:2]; [c setBuffer:y[0] offset:0 atIndex:3];
    [c setBuffer:gb(e, w[1]) offset:go(e, w[1]) atIndex:4]; [c setBuffer:y[1] offset:0 atIndex:5];
    const int k2 = n > 2 ? 2 : 0;
    [c setBuffer:gb(e, w[k2]) offset:go(e, w[k2]) atIndex:6]; [c setBuffer:y[k2] offset:0 atIndex:7];
    [c dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return true;
}

// the MoE experts' formats: mm()'s and 9 (mxfp4)
static const char * fmt_name(int fmt) { return fmt == 9 ? "mxfp4" : mm_fmt_name(fmt); }
static int fmt_nb01(int fmt, int K_) { return fmt == 9 ? K_ / 32 * 17 : mm_nb01(fmt, K_); }

// the experts' mat-vec over npairs pairs, per outputs a threadgroup
static void mvid_dispatch(id<MTLComputeCommandEncoder> c, int per, int M, int npairs) {
    if (!(g_skip & 64)) [c dispatchThreadgroups:MTLSizeMake((M + per - 1) / per, npairs, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

// an MoE layer's experts for the interactive pairs (rows 0 .. nA): one pair per tg.y, its expert's matrix
static void mvid(sn_engine * e, int fmt, uint64_t w, int estride, id<MTLBuffer> x, int xpair, id<MTLBuffer> y, int K_, int M, int npairs, int ldx, int ldy, int acc) {
    mvid_args a = {{K_, M, 1, ldx, ldy, acc, fmt_nb01(fmt, K_), g_eps}, e->g.n_expert_used, xpair, estride};
    id<MTLComputeCommandEncoder> c = kernel_begin(xpair ? "mm moe down" : "mm moe gate/up", [NSString stringWithFormat:@"mvid_%s", fmt_name(fmt)], &a, sizeof a);
    [c setBuffer:gb(e, w) offset:go(e, w) atIndex:1]; [c setBuffer:x offset:0 atIndex:2]; [c setBuffer:y offset:0 atIndex:3]; [c setBuffer:e->mids offset:0 atIndex:4];
    mvid_dispatch(c, fmt == 9 ? 4 : mv_per(fmt, K_), M, npairs);
}

// SN_JOIN 16: mvid for the gate (acc 0) and then the up (acc 2) of the same pairs in one dispatch (mvid2_*; K-quants, one interactive row)
static void mvid_glu(sn_engine * e, int fmt, uint64_t wg, int esg, uint64_t wu, int esu, id<MTLBuffer> x, id<MTLBuffer> y, int K_, int M, int npairs, int ldx, int ldy) {
    struct { mv_args mg, mu; int k_used, xpair, estride_g, estride_u; } a = {{K_, M, 1, ldx, ldy, 0, fmt_nb01(fmt, K_), g_eps}, {K_, M, 1, ldx, ldy, 2, fmt_nb01(fmt, K_), g_eps}, e->g.n_expert_used, 0, esg, esu};
    id<MTLComputeCommandEncoder> c = kernel_begin("mm moe gate/up", [NSString stringWithFormat:@"mvid2_%s", fmt_name(fmt)], &a, sizeof a);
    [c setBuffer:gb(e, wg) offset:go(e, wg) atIndex:1]; [c setBuffer:x offset:0 atIndex:2]; [c setBuffer:y offset:0 atIndex:3]; [c setBuffer:e->mids offset:0 atIndex:4];
    [c setBuffer:gb(e, wu) offset:go(e, wu) atIndex:5];
    mvid_dispatch(c, mv_per(fmt, K_), M, npairs);
}

// an MoE layer's experts for the other pairs: one grouped block matmul over the tiles moe_lists made (rows in list order);
// yh: the result in half into it (flag 32, rows M apart) instead of y
static void qmm_grouped(sn_engine * e, int fmt, uint64_t w, int estride, id<MTLBuffer> xh, id<MTLBuffer> y, int K_, int M, int ldx, int ldy, int flags, int ntiles, id<MTLBuffer> yh) {
    const int G = g_gg, RT = g_grt;
    qmm_args a = {K_, M, 0, fmt_nb01(fmt, K_), ldx, ldy, K_, 1 | 16 | flags | (yh ? 32 : 0), estride, yh ? M : 0};
    id<MTLComputeCommandEncoder> c = kernel_begin("mm moe", [NSString stringWithFormat:@"qmmh_ks_%s_g%d_nt%d_r%d", fmt_name(fmt), G, g_gnt, RT], &a, sizeof a);
    [c setBuffer:gb(e, w) offset:go(e, w) atIndex:1]; [c setBuffer:xh offset:0 atIndex:2]; [c setBuffer:y offset:0 atIndex:3]; [c setBuffer:e->mtiles offset:0 atIndex:4];
    [c setBuffer:yh ? yh : y offset:0 atIndex:5];
    [c dispatchThreadgroups:MTLSizeMake(ntiles, (M + 32 * RT - 1) / (32 * RT), 1) threadsPerThreadgroup:MTLSizeMake(32 * G * RT, 1, 1)];  // moe_lists' bound
}
