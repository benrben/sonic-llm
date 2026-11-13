// sllm-next: a part of core.h (included where core.h defines it): the matmul dispatch mm() and the interactive rows' fused
// mat-vecs (mm_qkv, mm_glu)

// mm()'s weight formats (0, 1, 5 .. 8): the kernel name part and a weight row's bytes for K weights
static const char * mm_fmt_name(int fmt) {
    switch (fmt) {
        case 8: return "q5_K";
        case 7: return "q6_K";
        case 6: return "q4_K";
        case 5: return "q8_0";
    }
    return fmt ? "q4_1" : "q4_0";
}
static int mm_nb01(int fmt, int K_) {
    switch (fmt) {
        case 7: return K_ / 256 * 210;
        case 6: return K_ / 256 * 144;
        case 8: return K_ / 256 * 176;
    }
    return K_ / 32 * (fmt == 5 ? 34 : fmt ? 20 : 18);
}

// one mm() call's rows as its phases see them: each phase that runs rows moves xo / yo past them and takes them off R;
// e is g_e as the last enc() left it
typedef struct {
    __unsafe_unretained id<MTLComputeCommandEncoder> e;
    __unsafe_unretained id<MTLBuffer> x, w, y;
    size_t xo, yo;
    uint64_t wo;
    int fmt, K, M, R, ldx, ldy, acc;
} mm_job;

static inline void mm_skip_rows(mm_job * j, int n) {
    j->xo += (size_t) n * j->ldx * 4; j->yo += (size_t) n * j->ldy * 4; j->R -= n;
}

// the weights, the input and the output at indices 1, 2, 3
static inline void set_wxy(id<MTLComputeCommandEncoder> e, const mm_job * j) { set3(e, 1, j->w, j->wo, j->x, j->xo, j->y, j->yo); }

// f32 / bf16 weights: the interactive rows, one simdgroup per output column; true when no rows are left
static inline bool mmf_rows_mv(mm_job * j) {
    const int nA = MIN(g_nA, j->R);
    if (nA <= 0) return false;
    struct { int K, M, R, ldx, ldy; } a = {j->K, j->M, nA, j->ldx, j->ldy};
    id<MTLComputeCommandEncoder> e = j->e;
    [e setComputePipelineState:K(j->fmt == 3 ? @"mvf_f32" : @"mvf_bf16")];
    [e setBytes:&a length:sizeof a atIndex:0];
    set_wxy(e, j);
    [e dispatchThreadgroups:MTLSizeMake((j->M + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (j->R == nA) return true;
    mm_skip_rows(j, nA);
    return false;
}

// f32 / bf16 weights: the batch rows on the matrix units, the weights straight from the file
static inline void mmf_tmm(const mm_job * j) {
    const int NT = j->R <= 16 ? 16 : 32, G = j->M <= 1024 ? 8 : 4;
    qmm_args a = {j->K, j->M, j->R, 0, j->ldx, j->ldy, j->K, 0};
    id<MTLComputeCommandEncoder> e = j->e;
    [e setComputePipelineState:K([NSString stringWithFormat:@"tmm_%s_g%d_nt%d", j->fmt == 3 ? "f32" : "bf16", G, NT])];
    [e setBytes:&a length:sizeof a atIndex:0];
    set_wxy(e, j);
    [e dispatchThreadgroups:MTLSizeMake((j->R + NT - 1) / NT, j->M / 32, 1) threadsPerThreadgroup:MTLSizeMake(32 * G, 1, 1)];
}

// f32 / bf16 weights: the batch rows as simdgroup-MMA tiles over S K-slices (S fixed by the shape), the partials added in
// slice order
static inline void mmf_sk(const mm_job * j) {
    const int K_ = j->K, M = j->M, R = j->R;
    id<MTLComputeCommandEncoder> e = j->e;
    const int S = M <= 512 ? 8 : M <= 2048 ? 2 : 1;
    mm_args a = {K_, M, R, j->ldx, j->ldy, 0};
    [e setComputePipelineState:K(j->fmt == 3 ? @"mm_sk_f32" : @"mm_sk_bf16")];
    [e setBytes:&a length:sizeof a atIndex:0];
    [e setBuffer:j->x offset:j->xo atIndex:1];
    [e setBuffer:j->w offset:j->wo atIndex:2];
    const bool direct = S == 1 && j->ldy == M;  // one slice: straight into y
    [e setBuffer:(direct ? j->y : g_skpart) offset:(direct ? j->yo : 0) atIndex:3];
    [e setBytes:&S length:4 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(M / 32, (R + 31) / 32, S) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (direct) return;
    enc("mm sum"); e = g_e;
    [e setComputePipelineState:K(@"mm_sk_sum")];
    [e setBytes:&a length:sizeof a atIndex:0];
    [e setBuffer:g_skpart offset:0 atIndex:1];
    [e setBuffer:j->y offset:j->yo atIndex:2];
    [e setBytes:&S length:4 atIndex:3];
    [e dispatchThreads:MTLSizeMake(M, R, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// f32 / bf16 weights (Gemma 4's small matrices)
static inline void mm_float(mm_job * j) {
    if (mmf_rows_mv(j)) return;
    if (g_tmm) { mmf_tmm(j); return; }
    mmf_sk(j);
}

// the interactive rows' mat-vec of an mm() call, as the call found them (its rows 0 .. nA; mode: g_amode's input)
typedef struct {
    __unsafe_unretained id<MTLBuffer> w, x, y;
    uint64_t wo;
    size_t xo, yo;
    int fmt, K, M, nA, mode, ldx, ldy, acc;
} mv_call;

static NSString * mv_name(int fmt, int nA, int mode) {
    if (fmt == 2) return [NSString stringWithFormat:@"mv_q6k_r%d", nA];
    if (fmt == 5) return [NSString stringWithFormat:g_q8k ? @"mv_q8_0k_r%d" : @"mv_q8_0_r%d", nA];
    if (fmt >= 6) return [NSString stringWithFormat:@"mv_%s_r%d", fmt == 6 ? "q4_K" : fmt == 8 ? "q5_K" : "q6_K", nA];
    return [NSString stringWithFormat:@"mv_%s_r%d_m%d", fmt ? "q4_1" : "q4_0", nA, mode];
}

// outputs a threadgroup (2 simdgroups): the K-quants' 4; mv_q4 8, 4 from 3 token rows (kernels.metal's J)
static inline int mv_rows_per(int fmt, int K_, int nA) {
    return fmt >= 6 ? mv_per(fmt, K_) : fmt == 5 && g_q8k ? 2 : (fmt == 0 || fmt == 1) && nA >= 3 ? 4 : 8;
}

static inline int mv_rows_tg(int fmt) { return fmt == 5 && g_q8k ? 128 : 64; }  // threads a threadgroup (mv_q8_0k: 4 simdgroups)

// the interactive rows on the mat-vec kernels (SN_LATE: after the background rows' block matmul)
static inline void mv_rows(const mv_call * v) {
    const int fmt = v->fmt, K_ = v->K, mode = v->mode;
    const int nb01 = fmt == 2 ? K_ / 256 * 210 : mm_nb01(fmt, K_);
    mv_args va = {K_, v->M, v->nA, mode == 1 ? M_E_ROWS : v->ldx, v->ldy, v->acc, nb01, g_eps};
    id<MTLComputeCommandEncoder> c = g_e;
    [c setComputePipelineState:K(mv_name(fmt, v->nA, mode))];
    [c setBytes:&va length:sizeof va atIndex:0];
    [c setBuffer:(fmt == 2 ? W : v->w) offset:(fmt == 2 ? g_head : v->wo) atIndex:1];
    [c setBuffer:(mode == 1 ? g_ax : v->x) offset:(mode == 1 ? 0 : v->xo) atIndex:2];
    [c setBuffer:v->y offset:v->yo atIndex:3];
    [c setBuffer:(mode ? g_aux : v->x) offset:(mode ? g_auxo : 0) atIndex:4];
    const int per = mv_rows_per(fmt, K_, v->nA);
    [c dispatchThreadgroups:MTLSizeMake((v->M + per - 1) / per, 1, 1) threadsPerThreadgroup:MTLSizeMake(mv_rows_tg(fmt), 1, 1)];
}

// the interactive rows' mat-vec first (unless SN_LATE puts it after the block matmul); true when no rows are left
static inline bool mm_mv_first(mm_job * j, const mv_call * mv, bool late) {
    if (mv->nA <= 0) return false;
    if (!late) mv_rows(mv);
    if (j->R == mv->nA) return true;
    mm_skip_rows(j, mv->nA);
    return false;
}

// a wide check's rows, or a small step's background write rows (SN_SMALL; its prompt rows go on to the block matmuls): the
// few-row MMA kernel, on a float input only (an input the last block matmul wrote in half has no float rows; a call that
// runs rows here writes no half output either: the next call may take other rows here, by its own weight type)
static inline int mvm_rows(int fmt, int R, bool xhn) {
    return g_f32 || !g_mvm || fmt != 0 || xhn ? 0 : g_wide ? (R <= 8 ? R : 0) : g_small_step ? MIN(g_nsw, R) : 0;
}

static inline void mm_mvm(const mm_job * j, int nm) {
    const int K_ = j->K, M = j->M;
    const int nb = K_ / 32;
    int nsg = 8, nt = 2;  // 8 K slices of 16 weight rows: Phi-3.5's 5-row check 12.9 ms (ggml's choice 13.2, 4 x 32 rows 14.0)
    while (nsg > 4 && nsg > nb) nsg /= 2;
    if (g_mvm_nsg) nsg = g_mvm_nsg;  // SN_MVM_NSG / SN_MVM_NT: tuning
    if (g_mvm_nt) nt = g_mvm_nt;
    mvm_args a = {K_, M, nm, j->ldx, j->ldy, j->acc, K_ / 32 * 18};
    id<MTLComputeCommandEncoder> e = j->e;
    [e setComputePipelineState:K([NSString stringWithFormat:@"mvm_q4_0_nt%d_s%d", nt, nsg])];
    [e setBytes:&a length:sizeof a atIndex:0];
    set_wxy(e, j);
    [e dispatchThreadgroups:MTLSizeMake((M + 8 * nt - 1) / (8 * nt), 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * nsg, 1, 1)];
}

// the mvm rows first (see mvm_rows); true when no rows are left (the late mat-vec run)
static inline bool mm_mvm_first(mm_job * j, const mv_call * mv, bool late, int nm) {
    if (nm <= 0) return false;
    mm_mvm(j, nm);
    g_xhb_stale = 1;
    if (nm == j->R) {
        if (late) mv_rows(mv);
        return true;
    }
    mm_skip_rows(j, nm);
    return false;
}

// 64-token tiles for a long K (a down projection's) from 128 rows: each weight read for twice the tokens (kq_bench, 512
// rows: Qwen2.5's 3584 x 18944 -17%, Gemma 4 31B's -27%, Mistral's q6_K -21%; 128 rows -7 to -20%; 64 rows mixed; at K
// 9728 and below 32 tokens are 10% faster)
static inline int qmm_nt(int fmt, int K_, int R) {
    return R <= 8 && g_nt_min == 8 ? 8 : R <= 16 ? 16 : (R >= g_nt64 || (K_ >= 12288 && R >= 128)) && fmt != 2 ? 64 : 32;
}

// the block matmuls' input in half: the gated unit's block rows wrote it (xhn, mm_glu), or converted here (f2h) unless the
// last block matmul took the same one (xh_same)
static inline void qmm_half_input(mm_job * j, qmm_args * a, bool xhn, int xh_same, const char * cat) {
    if (xhn) {
        j->x = g_xhn; j->xo = 0; a->ldx = j->K;
        return;
    }
    if (!xh_same || g_xhb_stale) {
        const size_t need = (size_t) j->R * j->K * 2;
        if (!g_xhb || g_xhb.length < need) g_xhb = [dev newBufferWithLength:need + need / 2 options:MTLResourceStorageModePrivate];
        struct { int K, ldx; } fa = {j->K, j->ldx};
        enc("f2h"); id<MTLComputeCommandEncoder> e = g_e;
        [e setComputePipelineState:K(@"f2h")];
        [e setBytes:&fa length:sizeof fa atIndex:0];
        [e setBuffer:j->x offset:j->xo atIndex:1];
        [e setBuffer:g_xhb offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake(j->K / 4, j->R, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, j->K / 4), 1, 1)];
        enc(cat); j->e = g_e;
    }
    j->x = g_xhb; j->xo = 0; a->ldx = j->K;
    g_xhb_stale = 0;
}

// K slices: a wide draft check (one writer, a few rows) 8 slices, 4% faster; row tiles: two share the activations
static inline int qmm_slices(int M) { return g_bg ? g_bg : M <= 1024 || g_wide ? 8 : 4; }
static inline int qmm_row_tiles(int NT, int K_, int R) { return NT == 64 || (R > 64 && K_ >= 2048) ? 2 : 1; }
static inline int qmm_acc_flags(int acc) { return acc == 1 ? 2 : acc == 2 ? 4 : acc == 3 ? 8 : 0; }  // acc: 1 add, 2 / 3 silu / gelu gate in D
// SN_QMMS (4 K slices only: its 8-slice sum order is not qmm_ks's)
static inline bool qmms_ok(int fmt, bool xh, int R, int G) {
    return xh && g_qmms && R >= g_qmms && G == 4 && (fmt == 5 || fmt == 7 || fmt == 8);
}

// a layer matrix's block matmul (not the output layer's): its kernel name, a's input and flags, grid and threadgroup
static inline NSString * qmm_layer_setup(mm_job * j, qmm_args * a, MTLSize * grid, MTLSize * tpg, int NT, bool xhn, int xh_same,
                                         const char * cat) {
    const int fmt = j->fmt, K_ = j->K, M = j->M, R = j->R;
    const int G = qmm_slices(M);
    const int RT = qmm_row_tiles(NT, K_, R);
    const bool xh = g_xh || fmt >= 6;  // the K-quants have half-input kernels only
    if (xh) qmm_half_input(j, a, xhn, xh_same, cat);
    NSString * name = [NSString stringWithFormat:xh ? @"qmmh_ks_%s_g%d_nt%d_r%d" : @"qmm_ks_%s_g%d_nt%d_r%d", mm_fmt_name(fmt), G, NT, RT];
    a->nb01 = mm_nb01(fmt, K_);
    a->flags = 1 | qmm_acc_flags(j->acc);
    *grid = MTLSizeMake((R + NT - 1) / NT, (M + 32 * RT - 1) / (32 * RT), 1);
    *tpg = MTLSizeMake(32 * G * RT, 1, 1);
    if (qmms_ok(fmt, xh, R, G)) {
        name = [NSString stringWithFormat:@"qmms_%s_g%d_t2", mm_fmt_name(fmt), G];
        *grid = MTLSizeMake((R + 63) / 64, (M + 31) / 32, 1);
        *tpg = MTLSizeMake(64 * G, 1, 1);
    }
    return name;
}

// the block matmul's result in half (R x M) into g_xhn, for the next block matmul
static inline void qmm_half_output(qmm_args * a, int R, int M) {
    const size_t need = (size_t) R * M * 2;
    if (!g_xhn || g_xhn.length < need) g_xhn = [dev newBufferWithLength:need + need / 2 options:MTLResourceStorageModePrivate];
    a->flags |= 32; a->ldh = M;
}

// the block matmuls (the rows the mat-vec and mvm left); want_h: the result in half for the next block matmul if it can be
static inline void mm_block(mm_job * j, bool xhn, int xh_same, bool want_h, int nm, const char * cat) {
    const int fmt = j->fmt, K_ = j->K, M = j->M, R = j->R;
    const int NT = qmm_nt(fmt, K_, R);
    qmm_args a = {K_, M, R, 0, j->ldx, j->ldy, K_, 0};
    NSString * name;
    MTLSize grid, tpg;
    if (fmt == 2) {
        name = [NSString stringWithFormat:@"qmm_q6p_nt%d", MAX(NT, 16)];  // its activation staging needs 16 rows or more
        a.nb01 = K_ / 256 * 210;
        j->w = W; j->wo = g_head;
        grid = MTLSizeMake((R + MAX(NT, 16) - 1) / MAX(NT, 16), (M + 127) / 128, 1);
        tpg = MTLSizeMake(128, 1, 1);
    } else {
        name = qmm_layer_setup(j, &a, &grid, &tpg, NT, xhn, xh_same, cat);
    }
    const bool hout = want_h && fmt != 2 && (g_xh || fmt >= 6) && nm == 0;  // the result in half for the next block matmul (no float rows written)
    if (hout) qmm_half_output(&a, R, M);
    id<MTLComputeCommandEncoder> e = j->e;
    [e setComputePipelineState:K(name)];
    [e setBytes:&a length:sizeof a atIndex:0];
    set_wxy(e, j);
    [e setBuffer:hout ? g_xhn : j->y offset:0 atIndex:5];
    [e dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    if (hout) { g_xhn_ok = 1; g_xhn_src = j->y; g_xhn_k = M; }
}

// SN f32 mode (g_f32): the simple f32 kernels
static inline void mm_simple(const mm_job * j, id<MTLBuffer> s) {
    static NSString * names[] = {@"mm_q4_0", @"mm_q4_1", @"mm_q8s16"};
    id<MTLComputeCommandEncoder> e = j->e;
    [e setComputePipelineState:K(names[j->fmt])];
    mm_args a = {j->K, j->M, j->R, j->ldx, j->ldy, j->acc};
    [e setBytes:&a length:sizeof a atIndex:0];
    [e setBuffer:j->x offset:j->xo atIndex:1];
    [e setBuffer:j->w offset:j->wo atIndex:2];
    [e setBuffer:(s ? s : j->x) offset:0 atIndex:3];
    [e setBuffer:j->y offset:j->yo atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(j->M / 32, (j->R + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

// the quantized weights: the interactive rows on the mat-vec kernels, a few rows on mvm, the rest on the block matmuls
static inline void mm_quant(mm_job * j, id<MTLBuffer> s, int amode, bool xhn, int xh_same, bool want_h, const char * cat) {
    const int fmt = j->fmt;
    const int nA = g_f32 ? 0 : MIN(g_nA, j->R), mode = fmt == 2 || fmt >= 5 ? 0 : amode;
    const mv_call mv = {j->w, j->x, j->y, j->wo, j->xo, j->yo, fmt, j->K, j->M, nA, mode, j->ldx, j->ldy, j->acc};
    const bool late = nA > 0 && nA < j->R && (g_late & 1);
    if (mm_mv_first(j, &mv, late)) return;
    const int nm = mvm_rows(fmt, j->R, xhn);
    if (mm_mvm_first(j, &mv, late, nm)) return;
    if (!g_f32) {
        mm_block(j, xhn, xh_same, want_h, nm, cat);
        if (late) mv_rows(&mv);
        return;
    }
    mm_simple(j, s);
}

// Y[R][M] (+)= X[R][K] W[M][K]^T; fmt 0 q4_0, 1 q4_1, 2 the output layer (q6_K at g_head), 3 f32, 4 bf16, 5 q8_0, 6 q4_K,
// 7 q6_K (a layer's: at w), 8 q5_K
static void mm(id<MTLComputeCommandEncoder> e, int fmt, id<MTLBuffer> x, size_t xo, id<MTLBuffer> w, uint64_t wo, id<MTLBuffer> s,
               id<MTLBuffer> y, size_t yo, int K_, int M, int R, int ldx, int ldy, int acc) {
    const int xh_same = g_xh_same;  // taken here, so no early return leaves it set for another call
    g_xh_same = 0;
    const bool xhn = g_xhn_ok && x == g_xhn_src && K_ == g_xhn_k, want_h = g_xhn_want;  // (taken here as g_xh_same)
    g_xhn_ok = 0; g_xhn_want = 0;
    if (g_skip & (fmt == 2 ? 4 : 2)) { g_amode = 0; return; }
    char cat[24];
    snprintf(cat, sizeof cat, fmt == 2 ? "mm head" : "mm %dx%d", M, K_);
    enc(cat);
    mm_job j = {g_e, x, w, y, xo, yo, wo, fmt, K_, M, R, ldx, ldy, acc};
    const int amode = g_fold ? g_amode : 0;
    g_amode = 0;
    if (fmt == 3 || fmt == 4) {  // f32 / bf16 weights (Gemma 4's small matrices)
        mm_float(&j);
        return;
    }
    mm_quant(&j, s, amode, xhn, xh_same, want_h, cat);
}

// The interactive rows' fused mat-vecs (kernels mv3 / mvglu: the per-output arithmetic of mm()'s mat-vec path, fewer
// dispatches). SN_NOFUSE=1 in sn.m turns them off.
static int g_fuse = 1;
static int g_fa2 = 1;  // Qwen3 / Llama: the interactive rows' attention per KV head (fa2) at every depth instead of fa1 + attn_part
                       // (1k cells: 94.6-94.9 against 90.5-91.5 tok/s, short level); SN_NOFA2=1 gives fa1 back
static int g_nan = 1;  // Gemma 4: a residual add's norm and the next input norm in one dispatch (g4_norm_add_norm, the same bits); SN_NONAN=1
static int g_fa1 = 1;  // the interactive rows' one-row write attention (kernel fa1); SN_NOFA1=1 in sn.m turns it off
typedef struct { int n_head, n_kv, ctx; float scale; int ldq, maxc, window, kvstride, maxch, rpt, nrows, msp; } fa1_args;
// fa2r: an interactive step's rows share threadgroups (each simdgroup the one-row kernel's work: the same bits), so a draft
// check's rows read a chunk's K / V once; only with 1 or 2 query heads a K / V head, whose rows read the most K / V
// (Phi-3.5's 5-row wide check 13.44 -> 12.98 ms, 8 rows 15.01 -> 14.13; with 4 or 8 heads 1-2% slower). SN_NOFA2R=1: off
static int g_fa2r = 1;
static int g_pre2swap = 0;
static int g_kvskip = 1;  // SN_KVSKIP=0: Gemma 4's shared-KV layers run every row (1: only rows with an output)  // SN_PRE2_SWAP=1: pre2's grid with the tiles fastest, so tiles of one KV head run side by side and share K/V in cache
// fa2: at most g_msp partials a row (SN_MSP; kernels.metal's fa_span): a long row's chunks go sp at a time, merged in the
// kernel (0: a partial per 64-cell chunk at every depth)
static int g_msp = 64;
static int fa2_z(int maxc, int window) { return g_msp > 0 && window == 0 ? MIN(maxc, g_msp) : maxc; }
static int fa2_rpt(int nA, int G) { return g_fa2r && nA > 1 && G <= 2 ? MIN(nA, 8) : 1; }
typedef struct { int n, nban, ban[8]; } argmax_args;  // nban: the end-of-generation tokens rows with skip[r] never take
// fa1 takes rows whose cells span at most this many 64-cell chunks: Qwen3 8 (its per-head reads of the shared KV heads are
// slower than attn_part at 1k cells), Gemma 4 64 (one KV head; still faster at 1k)
#define FA1_MAXCH 8
#define FA1_MAXCH_G4 64
static int mv_nb01(int fmt, int K) { return K / 32 * (fmt == 5 ? 34 : fmt ? 20 : 18); }
static const char * mv_bq(int fmt) { return fmt == 5 ? "q8_0" : fmt ? "q4_1" : "q4_0"; }

// n rows of up to three matrices with one input (q, k, v) in one dispatch
static void mv3(int fmt, id<MTLBuffer> x, size_t xo, int ldx, int Kd, int n, int ns, id<MTLBuffer> const * w, const uint64_t * wo,
                const int * M, id<MTLBuffer> const * y, const size_t * yo, const int * ldy) {
    enc("mm qkv");
    struct { mv_args s[3]; int tg1, tg2; } a;
    int tg = 0, ends[3];
    for (int i = 0; i < 3; ++i) {
        const int j = MIN(i, ns - 1);
        a.s[i] = (mv_args){Kd, M[j], n, ldx, ldy[j], 0, mv_nb01(fmt, Kd), g_eps};
        if (i < ns) tg += (M[i] + 7) / 8;
        ends[i] = tg;
    }
    a.tg1 = ends[0]; a.tg2 = ends[1];
    [g_e setComputePipelineState:K([NSString stringWithFormat:@"mv3_%s_r%d", mv_bq(fmt), n])];
    [g_e setBytes:&a length:sizeof a atIndex:0];
    for (int i = 0; i < 3; ++i) {
        const int j = MIN(i, ns - 1);
        [g_e setBuffer:w[j] offset:wo[j] atIndex:1 + i];
        [g_e setBuffer:y[j] offset:yo[j] atIndex:5 + i];
    }
    [g_e setBuffer:x offset:xo atIndex:4];
    [g_e dispatchThreadgroups:MTLSizeMake(tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

// n rows of act(x Wg^T) * (x Wu^T) in one dispatch (act: 2 silu, 3 gelu: mm()'s acc codes)
static void mvglu(int fmt, id<MTLBuffer> x, size_t xo, int ldx, int Kd, int M, int n, id<MTLBuffer> wg, uint64_t wgo, id<MTLBuffer> wu,
                  uint64_t wuo, id<MTLBuffer> y, size_t yo, int ldy, int act) {
    enc("mm glu");
    mv_args a = {Kd, M, n, ldx, ldy, act, mv_nb01(fmt, Kd), g_eps};
    [g_e setComputePipelineState:K([NSString stringWithFormat:@"mvglu_%s_r%d", mv_bq(fmt), n])];
    [g_e setBytes:&a length:sizeof a atIndex:0];
    [g_e setBuffer:wg offset:wgo atIndex:1];
    [g_e setBuffer:wu offset:wuo atIndex:2];
    [g_e setBuffer:x offset:xo atIndex:3];
    [g_e setBuffer:y offset:yo atIndex:4];
    [g_e dispatchThreadgroups:MTLSizeMake((M + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

// whether the interactive rows skip the fused mat-vecs (mv3 / mvglu) and go through mm() with the rest: no fused mat-vec
// for the K-quants; from 3 rows mv_q4's smaller groups are faster, with the same sums; Q8_0 stays fused up to 4 rows: mm()'s
// mv_q8_0k splits K, so a draft row's sums would not be a one-row step's. late (SN_LATE 1): the fused mat-vec after the
// block matmuls (2: none, but for Q8_0, whose mm() mat-vec mv_q8_0k splits K: not mv3's sums)
static inline bool unfused(int fmt, int nA, bool late) {
    return !g_fuse || nA == 0 || (nA >= 3 && fmt != 5) || fmt >= 6 || (late && (g_late & 2) && !(fmt == 5 && g_q8k));
}

// mm_qkv's matrices: weights, outputs (M wide, rows ldy apart), ns of them
typedef struct { id<MTLBuffer> const * w; const uint64_t * wo; const int * M; id<MTLBuffer> const * y; const int * ldy; int ns; } qkv_set;

// every row of each matrix through mm()
static inline void mm_qkv_each(int fmt, id<MTLBuffer> x, int K, int R, const qkv_set * q, int nA, bool late) {
    for (int i = 0; i < q->ns; ++i) {
        g_xh_same = i > 0 && (nA == 0 || late);  // with early interactive rows each mm() call converts its own rest rows
        g_nobar = g_xh_same && g_conc_step;  // (k / v beside q: no f2h of their own, their own outputs)
        mm(g_e, fmt, x, 0, q->w[i], q->wo[i], nil, q->y[i], 0, K, q->M[i], R, K, q->ldy[i], 0);
    }
}

// the rows after the nA interactive ones through mm()
static inline void mm_qkv_rest(int fmt, id<MTLBuffer> x, int K, int R, const qkv_set * q, int nA) {
    for (int i = 0; i < q->ns; ++i) {
        g_xh_same = i > 0;
        g_nobar = i > 0 && g_conc_step;  // (k / v beside q: their own outputs)
        mm(g_e, fmt, x, (size_t) nA * K * 4, q->w[i], q->wo[i], nil, q->y[i], (size_t) nA * q->ldy[i] * 4, K, q->M[i], R - nA, K, q->ldy[i], 0);
    }
}

// q, k, v (ns = 1: q only) of R rows with input x (K = ldx): the interactive rows fused, the rest as mm() runs them
static void mm_qkv(int fmt, id<MTLBuffer> x, int K, int R, int ns, id<MTLBuffer> const * w, const uint64_t * wo, const int * M,
                   id<MTLBuffer> const * y, const int * ldy) {
    const int fast = g_nA, nA = MIN(fast, R);
    const bool late = nA > 0 && nA < R && (g_late & 1);  // SN_LATE: the fused mat-vec after the block matmuls (see unfused)
    const qkv_set q = {w, wo, M, y, ldy, ns};
    if (unfused(fmt, nA, late)) {
        mm_qkv_each(fmt, x, K, R, &q, nA, late);
        return;
    }
    const size_t yo[3] = {0, 0, 0};
    if (!late) mv3(fmt, x, 0, K, K, nA, ns, w, wo, M, y, yo, ldy);
    if (R == nA) return;
    g_nA = 0;
    mm_qkv_rest(fmt, x, K, R, &q, nA);
    g_nA = fast;
    if (late) mv3(fmt, x, 0, K, K, nA, ns, w, wo, M, y, yo, ldy);
}

// y = act(gate in y) * u (kernels.metal's glu_comb): the up projection's epilogue after gate and up ran side by side
static void glu_comb(id<MTLBuffer> y, id<MTLBuffer> u, int M, int ldy, int act, int R) {
    enc("glu");
    const int ga[4] = {M, ldy, act, R};
    [g_e setComputePipelineState:K(@"glu_comb")];
    [g_e setBytes:ga length:sizeof ga atIndex:0];
    [g_e setBuffer:y offset:0 atIndex:1];
    [g_e setBuffer:u offset:0 atIndex:2];
    [g_e dispatchThreads:MTLSizeMake(M, R, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, M), 1, 1)];
}

// an mm_glu call: input x (K wide), gate and up weights, output y (M wide, rows ldy apart), act (2 silu, 3 gelu)
typedef struct {
    __unsafe_unretained id<MTLBuffer> x, wg, wu, y;
    uint64_t wgo, wuo;
    int fmt, K, R, M, ldy, act;
} glu_job;

// SN_CONCSMALL: gate and up side by side, then act(gate) * up
static inline bool glu_side_ok(int nA, int act) { return g_conc_step && nA == 0 && (act == 2 || act == 3); }
static inline void glu_side(const glu_job * j) {
    static id<MTLBuffer> ub;
    const size_t need = (size_t) j->R * j->M * 4;
    if (!ub || ub.length < need) ub = [dev newBufferWithLength:need options:MTLResourceStorageModePrivate];
    mm(g_e, j->fmt, j->x, 0, j->wg, j->wgo, nil, j->y, 0, j->K, j->M, j->R, j->K, j->ldy, 0);
    g_xh_same = 1;
    g_nobar = 1;
    mm(g_e, j->fmt, j->x, 0, j->wu, j->wuo, nil, ub, 0, j->K, j->M, j->R, j->K, j->M, 0);
    glu_comb(j->y, ub, j->M, j->ldy, j->act, j->R);
}

// every row of gate and up through mm(), act(gate) * up in the up projection's epilogue
static inline void glu_unfused(const glu_job * j, int nA, bool late) {
    mm(g_e, j->fmt, j->x, 0, j->wg, j->wgo, nil, j->y, 0, j->K, j->M, j->R, j->K, j->ldy, 0);
    g_xh_same = nA == 0 || late;
    g_xhn_want = g_hout;
    mm(g_e, j->fmt, j->x, 0, j->wu, j->wuo, nil, j->y, 0, j->K, j->M, j->R, j->K, j->ldy, j->act);
}

// the rows after the nA interactive ones through mm()
static inline void glu_rest(const glu_job * j, int nA) {
    const size_t xo = (size_t) nA * j->K * 4, yo = (size_t) nA * j->ldy * 4;
    mm(g_e, j->fmt, j->x, xo, j->wg, j->wgo, nil, j->y, yo, j->K, j->M, j->R - nA, j->K, j->ldy, 0);
    g_xh_same = 1;
    g_xhn_want = g_hout;
    mm(g_e, j->fmt, j->x, xo, j->wu, j->wuo, nil, j->y, yo, j->K, j->M, j->R - nA, j->K, j->ldy, j->act);
}

// gate and up into y = act(gate) * up
static void mm_glu(int fmt, id<MTLBuffer> x, int K, int R, int M, id<MTLBuffer> wg, uint64_t wgo, id<MTLBuffer> wu, uint64_t wuo,
                   id<MTLBuffer> y, int ldy, int act) {
    const int fast = g_nA, nA = MIN(fast, R);
    const bool late = nA > 0 && nA < R && (g_late & 1);
    const glu_job j = {x, wg, wu, y, wgo, wuo, fmt, K, R, M, ldy, act};
    if (glu_side_ok(nA, act)) {
        glu_side(&j);
        return;
    }
    if (unfused(fmt, nA, late)) {
        glu_unfused(&j, nA, late);
        return;
    }
    if (!late) mvglu(fmt, x, 0, K, K, M, nA, wg, wgo, wu, wuo, y, 0, ldy, act);
    if (R == nA) return;
    g_nA = 0;
    glu_rest(&j, nA);
    g_nA = fast;
    if (late) mvglu(fmt, x, 0, K, K, M, nA, wg, wgo, wu, wuo, y, 0, ldy, act);
}

static double now_s(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e9; }
static double g_wall_ms;  // the last step's wall time: encoding, GPU and the wait
static NSMutableArray * g_profs;  // profile mode: per-kernel GPU time of each profiled step
