// sllm-next engine library, a part of sn.m (one translation unit; sn.m includes its parts in order): the layer programs: the MoE FFN, Qwen3.5, Gemma 3 / 4, gpt-oss

// the MoE FFN's top-k log (SN_IDSLOG)
enum { IDS_CAP = 256 << 20, IDS_CALLS = 1 << 20 };
static const char * g_idslog;
static id<MTLBuffer> g_idsbuf, g_idsw;
static size_t g_idsoff;
static int g_idsn, g_idsr[IDS_CALLS];
// the router's logits and (Qwen3.5-MoE) the shared expert: its gate, its gated unit and down projection
// (SN_CONC: the router, the shared expert's gate and its first matmul read h only, so they may overlap; the top k with
// the shared expert's second)
static void moe_router(sn_engine * e, const moe_t * m, int R) {
    const gguf_t * g = &e->g;
    const int E = e->E, NE = g->n_expert, FS = g->n_ff_shexp;
    const bool j2 = !e->oss && mvf_join_ok(NE, 1, R);
    if (j2) mvf_rows2(e, m->router, m->sgate, e->h, e->mlog, 0, e->msgt, 0, E, NE, 1, E, NE);
    else mvf_rows(e, m->router, e->h, e->mlog, 0, E, NE, R, E, NE);
    if (e->oss) return;
    if (!j2) {
        g_nobar = 1;
        mvf_rows(e, m->sgate, e->h, e->msgt, 0, E, 1, R, E, 1);
    }
    g_nobar = 1;
    mm_glu(m->f_sg, e->h, E, R, FS, gb(e, m->sg), go(e, m->sg), gb(e, m->su), go(e, m->su), e->mshg, FS, 2);
    mm(g_e, m->f_sd, e->mshg, 0, gb(e, m->sd), go(e, m->sd), nil, e->mshx, 0, FS, E, R, FS, E, 0);
    g_nobar = 1;
}

// SN_IDSLOG=FILE (a measurement): the same top k once more into a log, written at sn_close (c: the top k's encoder)
static void moe_ids_log(sn_engine * e, id<MTLComputeCommandEncoder> c, int R, int KU, bool tk1) {
    static int ids_init;
    if (!ids_init) { g_idslog = getenv("SN_IDSLOG"); ids_init = 1; }
    if (!g_idslog || !tk1) return;
    if (!g_idsbuf) {
        g_idsbuf = [e->mids.device newBufferWithLength:IDS_CAP options:MTLResourceStorageModeShared];
        g_idsw = [e->mids.device newBufferWithLength:e->mw.length options:MTLResourceStorageModePrivate];
    }
    if (g_idsoff + (size_t) R * KU * 4 <= IDS_CAP && g_idsn < IDS_CALLS) {
        [c setBuffer:g_idsbuf offset:g_idsoff atIndex:2]; [c setBuffer:g_idsw offset:0 atIndex:3];
        [c dispatchThreadgroups:MTLSizeMake(R, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        g_idsr[g_idsn++] = R; g_idsoff += (size_t) R * KU * 4;
    }
}

// each row's top k experts and their weights (gpt-oss: the router's bias first)
static void moe_topk(sn_engine * e, const moe_t * m, const moe_args * ma, int R) {
    const int NE = e->g.n_expert, KU = e->g.n_expert_used;
    enc("moe topk"); id<MTLComputeCommandEncoder> c = g_e;
    const bool tk1 = g_topk1 && NE <= 256 && KU <= 16;  // one simdgroup a row, no barriers (the same bits)
    [c setComputePipelineState:K(tk1 ? @"moe_topk1" : @"moe_topk")];
    [c setBytes:ma length:sizeof *ma atIndex:0];
    [c setBuffer:e->mlog offset:0 atIndex:1]; [c setBuffer:e->mids offset:0 atIndex:2]; [c setBuffer:e->mw offset:0 atIndex:3];
    [c setBuffer:e->oss ? gb(e, m->router_b) : e->mlog offset:e->oss ? go(e, m->router_b) : 0 atIndex:4];
    if (!(g_skip & 512)) [c dispatchThreadgroups:MTLSizeMake(R, 1, 1) threadsPerThreadgroup:MTLSizeMake(tk1 ? 32 : 256, 1, 1)];
    moe_ids_log(e, c, R, KU, tk1);
}

// gpt-oss's activation: (gate, up) of P pairs -> g in place (interactive) or the half rows of the down projection
static void moe_oai(sn_engine * e, const moe_t * m, const moe_args * ma, id<MTLBuffer> gt, id<MTLBuffer> up, int P, int grouped) {
    const int FE = e->g.n_ff_exp;
    id<MTLComputeCommandEncoder> c_ = kernel_begin("moe", @"moe_oai", ma, sizeof *ma);
    [c_ setBuffer:gt offset:0 atIndex:1]; [c_ setBuffer:up offset:0 atIndex:2]; [c_ setBuffer:e->mids offset:0 atIndex:3]; [c_ setBuffer:e->mlist offset:0 atIndex:4];
    [c_ setBuffer:gb(e, m->bg) offset:go(e, m->bg) atIndex:5]; [c_ setBuffer:gb(e, m->bu) offset:go(e, m->bu) atIndex:6]; [c_ setBuffer:e->mxg2 offset:0 atIndex:7];
    [c_ setBytes:&FE length:4 atIndex:8]; [c_ setBytes:&grouped length:4 atIndex:9];
    [c_ dispatchThreads:MTLSizeMake(FE, P, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, FE), 1, 1)];
}

// the interactive rows' pairs (rows 0 .. nA) on the experts' mat-vecs
static void moe_experts_mv(sn_engine * e, const moe_t * m, const moe_args * ma, int nA) {
    const int E = e->E, KU = e->g.n_expert_used, FE = e->g.n_ff_exp;
    if (e->oss) {
        mvid(e, m->f_eg, m->eg, m->s_eg, e->h, 0, e->mai, E, FE, nA * KU, E, FE, 0);
        mvid(e, m->f_eu, m->eu, m->s_eu, e->h, 0, e->mau, E, FE, nA * KU, E, FE, 0);
        moe_oai(e, m, ma, e->mai, e->mau, nA * KU, 0);
    } else if ((g_join & 16) && nA == 1 && m->f_eg == m->f_eu && m->f_eg >= 6 && m->f_eg <= 8) {
        mvid_glu(e, m->f_eg, m->eg, m->s_eg, m->eu, m->s_eu, e->h, e->mai, E, FE, KU, E, FE);
    } else {
        mvid(e, m->f_eg, m->eg, m->s_eg, e->h, 0, e->mai, E, FE, nA * KU, E, FE, 0);
        mvid(e, m->f_eu, m->eu, m->s_eu, e->h, 0, e->mai, E, FE, nA * KU, E, FE, 2);
    }
    mvid(e, m->f_ed, m->ed, m->s_ed, e->mai, 1, e->myi, FE, E, nA * KU, FE, E, 0);
}

// the other rows' pairs (RB rows): listed by expert in tiles, gathered (half), the grouped block matmuls
static void moe_experts_grouped(sn_engine * e, const moe_t * m, const moe_args * ma, int RB) {
    const int NE = e->g.n_expert, KU = e->g.n_expert_used, FE = e->g.n_ff_exp, E = e->E;
    const int P = RB * KU, nt = (P + g_gnt - 1) / g_gnt + NE;  // moe_lists writes exactly these tiles
    id<MTLComputeCommandEncoder> c = kernel_begin("moe lists", @"moe_lists", ma, sizeof *ma);
    [c setBuffer:e->mids offset:0 atIndex:1]; [c setBuffer:e->mlist offset:0 atIndex:2]; [c setBuffer:e->mpos offset:0 atIndex:3]; [c setBuffer:e->mtiles offset:0 atIndex:4];
    [c dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    c = kernel_begin("moe gather", @"moe_gather", ma, sizeof *ma);
    [c setBuffer:e->h offset:0 atIndex:1]; [c setBuffer:e->mlist offset:0 atIndex:2]; [c setBuffer:e->mxg offset:0 atIndex:3]; [c setBytes:&E length:4 atIndex:4];
    [c dispatchThreads:MTLSizeMake(E / 4, P, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, E / 4), 1, 1)];
    qmm_grouped(e, m->f_eg, m->eg, m->s_eg, e->mxg, e->mact, E, FE, E, FE, 0, nt, nil);
    if (e->oss) {
        qmm_grouped(e, m->f_eu, m->eu, m->s_eu, e->mxg, e->mact2, E, FE, E, FE, 0, nt, nil);
        moe_oai(e, m, ma, e->mact, e->mact2, P, 1);
    } else {  // silu(gate) * up in the epilogue, in half for the down projection (f2h's rounding)
        qmm_grouped(e, m->f_eu, m->eu, m->s_eu, e->mxg, e->mact, E, FE, E, FE, 4, nt, e->mxg2);
    }
    qmm_grouped(e, m->f_ed, m->ed, m->s_ed, e->mxg2, e->myd, FE, E, FE, E, 0, nt, nil);
}

// an interactive row of Qwen3.5-MoE: moe_out + the next layer's rmsnorm (weights nnw at nnwo) in one dispatch
static void moe_out_norm(sn_engine * e, const moe_args * ma, int R, id<MTLBuffer> nnw, size_t nnwo, const norm_args * nrm) {
    id<MTLComputeCommandEncoder> c = kernel_begin("moe out", @"moe_out_norm", ma, sizeof *ma);
    [c setBuffer:e->x offset:0 atIndex:1]; [c setBuffer:e->myi offset:0 atIndex:2]; [c setBuffer:e->mw offset:0 atIndex:3];
    [c setBuffer:e->mshx offset:0 atIndex:4]; [c setBuffer:e->msgt offset:0 atIndex:5]; [c setBytes:nrm length:sizeof(norm_args) atIndex:6];
    [c setBuffer:nnw offset:nnwo atIndex:7]; [c setBuffer:e->h offset:0 atIndex:8];
    if (!((g_skip & 1024) && (g_skip & 8))) [c dispatchThreadgroups:MTLSizeMake(R, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// the experts' outputs by their weights, the shared expert's (Qwen3.5-MoE) or the down projection's bias (gpt-oss), into x
static void moe_out(sn_engine * e, const moe_t * m, const moe_args * ma, int R) {
    const int E = e->E;
    id<MTLComputeCommandEncoder> c = kernel_begin("moe out", @"moe_out", ma, sizeof *ma);
    [c setBuffer:e->x offset:0 atIndex:1]; [c setBuffer:e->myi offset:0 atIndex:2]; [c setBuffer:e->myd offset:0 atIndex:3]; [c setBuffer:e->mpos offset:0 atIndex:4];
    [c setBuffer:e->mw offset:0 atIndex:5]; [c setBuffer:e->mshx offset:0 atIndex:6]; [c setBuffer:e->msgt offset:0 atIndex:7]; [c setBytes:&E length:4 atIndex:8];
    [c setBuffer:e->mids offset:0 atIndex:9]; [c setBuffer:e->oss ? gb(e, m->bd) : e->mshx offset:e->oss ? go(e, m->bd) : 0 atIndex:10];
    if (!(g_skip & 1024)) [c dispatchThreads:MTLSizeMake(E, R, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// an MoE FFN over h into x: the router's top k experts per row, each a gated unit, summed in slot order. Qwen3.5-MoE
// (mode 0): softmax over the experts, the top k renormalized, SwiGLU experts, plus the shared expert times sigmoid(its
// gate). gpt-oss (mode 1): the router's bias, a softmax over the top k, biased experts through swiglu_oai. Interactive rows
// take the experts' mat-vecs, the rest the grouped block matmuls (a row's kernels by its kind, never by the other rows)
// nnw / nrm: the next layer's first rmsnorm weights (SN_JOIN 4: then moe_out and that rmsnorm are one dispatch when it applies and
// moe_ffn returns true: h holds the next layer's attn_norm output)
static bool moe_ffn(sn_engine * e, const moe_t * m, int R, id<MTLBuffer> nnw, size_t nnwo, const norm_args * nrm) {
    const gguf_t * g = &e->g;
    const int NE = g->n_expert, KU = g->n_expert_used, nA = MIN(g_nA, R), RB = R - nA;
    const int oss = e->oss;
    moe_router(e, m, R);
    moe_args ma = {NE, KU, nA, RB * KU, g_gnt, NE, oss};
    moe_topk(e, m, &ma, R);
    if (nA > 0) moe_experts_mv(e, m, &ma, nA);
    if (RB > 0) moe_experts_grouped(e, m, &ma, RB);
    if ((g_join & 4) && nnw && !oss && R == 1 && nA == 1) {  // an interactive row of Qwen3.5-MoE: moe_out + the next rmsnorm
        moe_out_norm(e, &ma, R, nnw, nnwo, nrm);
        return true;
    }
    moe_out(e, m, &ma, R);
    return false;
}

// layer i's MoE FFN, the next layer's (if any) first norm joined to it: true when it was (see moe_ffn)
static bool q35_moe(sn_engine * e, const q35l * ls, int nls, int i, int R, const norm_args * nrm) {
    const q35l * nl = i + 1 < nls ? &ls[i + 1] : NULL;
    return moe_ffn(e, &ls[i].m, R, nl ? gb(e, nl->attn_norm) : nil, nl ? go(e, nl->attn_norm) : 0, nrm);
}

// a linear-attention layer's inputs: qkv and the gate z (SN_JOIN 8: in one dispatch), alpha and beta
static void q35_lin_in(sn_engine * e, const q35l * l, int R) {
    const int E = e->E, C = e->q35C, nv = e->g.ssm_rank, inner = e->g.ssm_inner;
    {
        const int jf[2] = {l->f_qkv, l->f_z}, jm[2] = {C, inner};
        const uint64_t jw[2] = {l->qkv, l->z};
        id<MTLBuffer> const jy[2] = {e->q, e->zb};
        if (!mm_join_q8(e, 2, jf, e->h, E, jw, jm, jy, R)) {
            mm(g_e, l->f_qkv, e->h, 0, gb(e, l->qkv), go(e, l->qkv), nil, e->q, 0, E, C, R, E, C, 0);
            mm(g_e, l->f_z, e->h, 0, gb(e, l->z), go(e, l->z), nil, e->zb, 0, E, inner, R, E, inner, 0);
        }
    }
    if (l->f_ab == 3) {  // f32 alpha / beta: mvf_rows for every row (any count of outputs; mm's f32 batch path wants 32s)
        if (mvf_join_ok(nv, nv, R)) mvf_rows2(e, l->alpha, l->beta, e->h, e->ab, 0, e->ab, (size_t) nv * 4, E, nv, nv, E, 2 * nv);
        else {
            mvf_rows(e, l->alpha, e->h, e->ab, 0, E, nv, R, E, 2 * nv);
            mvf_rows(e, l->beta, e->h, e->ab, (size_t) nv * 4, E, nv, R, E, 2 * nv);
        }
    } else {
        mm(g_e, l->f_ab, e->h, 0, gb(e, l->alpha), go(e, l->alpha), nil, e->ab, 0, E, nv, R, E, 2 * nv, 0);
        mm(g_e, l->f_ab, e->h, 0, gb(e, l->beta), go(e, l->beta), nil, e->ab, (size_t) nv * 4, E, nv, R, E, 2 * nv, 0);
    }
}

// the runs' conv states: the draft rows' ones after them (read from the runs' old states, before q35_conv_save), then each
// run's saved
static void q35_conv_states(sn_engine * e, const q35_args * qa) {
    const int C = e->q35C;
    id<MTLComputeCommandEncoder> c;
    if (e->nsnap) {  // draft rows: the conv states after them, read from the runs' old states (before q35_conv_save)
        c = kernel_begin("q35", @"q35_conv_snap", qa, sizeof *qa);
        [c setBuffer:e->q offset:0 atIndex:1]; [c setBuffer:e->cst offset:0 atIndex:2]; [c setBuffer:e->csnap offset:0 atIndex:3];
        [c dispatchThreads:MTLSizeMake(C, e->nsnap, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    c = kernel_begin("q35", @"q35_conv_save", qa, sizeof *qa);
    [c setBuffer:e->q offset:0 atIndex:1]; [c setBuffer:e->cst offset:0 atIndex:2]; [c setBuffer:e->runs offset:0 atIndex:3];
    if (!(g_skip & 256)) [c dispatchThreads:MTLSizeMake(C, e->nruns, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// the 4-tap conv over each run's last cells and its rows (SN_JOIN 1, every run one row: the state shift inside q35_conv8s)
static void q35_conv(sn_engine * e, const q35l * l, int R, const q35_args * qa) {
    const int C = e->q35C;
    enc("q35"); id<MTLComputeCommandEncoder> c = g_e;
    const bool cs1 = (g_join & 1) && g_conv8 && e->nruns == R;  // every run one row: the state shift inside q35_conv8s
    [c setComputePipelineState:K(cs1 ? @"q35_conv8s" : g_conv8 ? @"q35_conv8" : @"q35_conv")];
    [c setBytes:qa length:sizeof *qa atIndex:0];
    [c setBuffer:e->q offset:0 atIndex:1]; [c setBuffer:gb(e, l->conv) offset:go(e, l->conv) atIndex:2]; [c setBuffer:e->cst offset:0 atIndex:3];
    [c setBuffer:e->rsi offset:0 atIndex:4]; [c setBuffer:e->rrun offset:0 atIndex:5]; [c setBuffer:e->co offset:0 atIndex:6];
    [c setBuffer:e->ab offset:0 atIndex:7]; [c setBuffer:gb(e, l->dt) offset:go(e, l->dt) atIndex:8]; [c setBuffer:gb(e, l->a) offset:go(e, l->a) atIndex:9];
    [c setBuffer:e->bg offset:0 atIndex:10];
    if (!(g_skip & 8192) || (cs1 && !(g_skip & 256))) [c dispatchThreadgroups:MTLSizeMake(R, g_conv8 ? (C / 128 + 7) / 8 : 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (!cs1) q35_conv_states(e, qa);
}

// the delta rule over each run (with the draft rows' state snapshots), the gated norm, the output projection added to x
static void q35_delta_out(sn_engine * e, const q35l * l, int R, const q35_args * qa) {
    const int E = e->E, nv = e->g.ssm_rank, inner = e->g.ssm_inner;
    id<MTLComputeCommandEncoder> c = kernel_begin("q35", e->nsnap ? @"q35_delta_s" : @"q35_delta", qa, sizeof *qa);
    [c setBuffer:e->co offset:0 atIndex:1]; [c setBuffer:e->bg offset:0 atIndex:2]; [c setBuffer:e->sst offset:0 atIndex:3];
    [c setBuffer:e->runs offset:0 atIndex:4]; [c setBuffer:e->dout offset:0 atIndex:5];
    if (e->nsnap) [c setBuffer:e->rsnap offset:0 atIndex:6];
    if (!(g_skip & 4096)) [c dispatchThreadgroups:MTLSizeMake(e->nruns, nv, 128 / 16) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    c = kernel_begin("q35", @"q35_out", qa, sizeof *qa);
    [c setBuffer:e->dout offset:0 atIndex:1]; [c setBuffer:e->zb offset:0 atIndex:2]; [c setBuffer:gb(e, l->norm) offset:go(e, l->norm) atIndex:3];
    [c setBuffer:e->ao offset:0 atIndex:4];
    if (!(g_skip & 2048)) [c dispatchThreadgroups:MTLSizeMake(R, nv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    mm(g_e, l->f_out, e->ao, 0, gb(e, l->out), go(e, l->out), nil, e->x, 0, inner, E, R, inner, E, 1);
}

// a Gated DeltaNet layer
static void q35_linear(sn_engine * e, const q35l * l, int R) {
    const int C = e->q35C, nk = e->g.ssm_groups, nv = e->g.ssm_rank;
    q35_lin_in(e, l, R);
    q35_args qa = {C, nk, nv, C, 2 * nv, e->nlin, e->g.eps, l->li};
    q35_conv(e, l, R, &qa);
    q35_delta_out(e, l, R, &qa);
}

// Qwen3.5's q / k norms, rope (its q from the doubled q rows: q and its gate) and the K / V cache write of attention layer l
static void q35_qk(sn_engine * e, const q35l * l, int R) {
    const gguf_t * g = &e->g;
    const int NH = e->NH, NKV = g->n_kv, D = g->hd, QD = NH * D, KVS = NKV * D;
    q35q_args qq = {NH, NKV, e->ctx, 2 * QD, QD, KVS, g->eps, g->rope_base, g->n_rot};
    const size_t ko = e->kv_layer * l->li * 2;
    id<MTLComputeCommandEncoder> c = kernel_begin("qk_rope", @"q35_qk", &qq, sizeof qq);
    [c setBuffer:e->q offset:0 atIndex:1]; [c setBuffer:e->qa offset:0 atIndex:2]; [c setBuffer:e->k offset:0 atIndex:3]; [c setBuffer:e->v offset:0 atIndex:4];
    [c setBuffer:gb(e, l->q_norm) offset:go(e, l->q_norm) atIndex:5]; [c setBuffer:gb(e, l->k_norm) offset:go(e, l->k_norm) atIndex:6];
    [c setBuffer:e->pl[0].kc offset:ko atIndex:7]; [c setBuffer:e->pl[0].vc offset:ko atIndex:8];
    [c setBuffer:e->rseq offset:0 atIndex:9]; [c setBuffer:e->rpos offset:0 atIndex:10]; [c setBuffer:e->pl[0].pt offset:0 atIndex:11];
    [c dispatchThreadgroups:MTLSizeMake(R, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// a full attention layer: q (with its gate) / k / v (SN_JOIN 8: in one dispatch), rope, attention, the gate, the output
// projection added to x
static void q35_full(sn_engine * e, const q35l * l, int R, int n_write, int nblk, int maxc_w) {
    const gguf_t * g = &e->g;
    const int E = e->E, NH = e->NH, NKV = g->n_kv, D = g->hd, QD = NH * D, KVS = NKV * D;
    {
        const int jf[3] = {l->f_q, l->f_q, l->f_v}, jm[3] = {2 * QD, KVS, KVS};
        const uint64_t jw[3] = {l->q, l->k, l->v};
        id<MTLBuffer> const jy[3] = {e->q, e->k, e->v};
        if (!mm_join_q8(e, 3, jf, e->h, E, jw, jm, jy, R)) {
            mm(g_e, l->f_q, e->h, 0, gb(e, l->q), go(e, l->q), nil, e->q, 0, E, 2 * QD, R, E, 2 * QD, 0);
            mm(g_e, l->f_q, e->h, 0, gb(e, l->k), go(e, l->k), nil, e->k, 0, E, KVS, R, E, KVS, 0);
            mm(g_e, l->f_v, e->h, 0, gb(e, l->v), go(e, l->v), nil, e->v, 0, E, KVS, R, E, KVS, 0);
        }
    }
    q35q_args qq = {NH, NKV, e->ctx, 2 * QD, QD, KVS, g->eps, g->rope_base, g->n_rot};
    q35_qk(e, l, R);
    q35_attn(e, l, n_write, nblk, maxc_w);
    id<MTLComputeCommandEncoder> c = kernel_begin("q35", @"q35_gate", &qq, sizeof qq);
    [c setBuffer:e->q offset:0 atIndex:1]; [c setBuffer:e->ao offset:0 atIndex:2];
    [c dispatchThreads:MTLSizeMake(QD, R, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    mm(g_e, l->f_o, e->ao, 0, gb(e, l->o), go(e, l->o), nil, e->x, 0, QD, E, R, QD, E, 1);
}

// Qwen3.5 / 3.6: one layer program over the step's rows (sn.m's runs give each sequence's rows in order)
static void q35_layers(sn_engine * e, const q35l * ls, int nls, int R, int n_write, int nblk, int maxc_w) {
    const gguf_t * g = &e->g;
    const int E = e->E, F = e->FMAX;
    norm_args nrm = {E, g->eps, E, E};
    bool pre = false;  // the layer's first rmsnorm already done by the last layer's moe_out_norm
    for (int i = 0; i < nls; ++i) {
        const q35l * l = &ls[i];
        if (!pre) x_norm(e, &nrm, l->attn_norm, R);
        pre = false;
        if (l->lin) q35_linear(e, l, R);
        else q35_full(e, l, R, n_write, nblk, maxc_w);
        x_norm(e, &nrm, l->post_norm, R);
        if (e->moe) {
            pre = q35_moe(e, ls, nls, i, R, &nrm);
            continue;
        }
        mm_glu(l->fmt, e->h, E, R, F, gb(e, l->gate), go(e, l->gate), gb(e, l->up), go(e, l->up), e->fg, F, 2);
        mm(g_e, l->down_fmt, e->fg, 0, gb(e, l->down), go(e, l->down), nil, e->x, 0, F, E, R, F, E, 1);
    }
}

// Gemma 4's per-layer embeddings: the step's rows of the table (praw, staged by the CPU) and the projection of x combined
// into each layer's input (ple)
static void gemma_ple(sn_engine * e, int R) {
    const gguf_t * g = &e->g;
    const int E = e->E, L = e->L, PN = e->PN;
    g4_args gr = {E, 16.0f, E, PN, L, g->eps};
    enc("ple"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&gr length:sizeof gr atIndex:0];
    [c setBuffer:e->praw offset:0 atIndex:1]; [c setBuffer:e->pq offset:0 atIndex:2];
    dispatch_rows(c, @"g4_ple_raw", R, 256);
    mm(g_e, 4, e->x, 0, gb(e, e->pl_proj), go(e, e->pl_proj), nil, e->pp, 0, E, PN, R, E, PN, 0);
    g4_args gc = {E, 1.0f / sqrtf((float) E), E, PN, L, g->eps};
    c = kernel_begin("ple", @"g4_ple_combine", &gc, sizeof gc);
    [c setBuffer:e->pp offset:0 atIndex:1]; [c setBuffer:e->pq offset:0 atIndex:2]; [c setBuffer:gb(e, e->pl_norm) offset:go(e, e->pl_norm) atIndex:3];
    [c setBuffer:e->ple offset:0 atIndex:4];
    [c dispatchThreadgroups:MTLSizeMake(R, L, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// a Gemma / gpt-oss layer's cache as the attention kernels read it: its pool's buffers at koff, the pool's page table
// (gpt-oss: its sinks)
static inline layer_attn g4_attn_of(sn_engine * e, const g4layer_t * l) {
    const pool_t * pool = &e->pl[l->pool];
    return (layer_attn) {e->q, LKB(e, l), LVB(e, l), pool->pt, e->oss ? gb(e, l->sinks) : nil, LKO(e, l), e->oss ? go(e, l->sinks) : 0, l->D,
                         e->NH / l->nkv, e->NH, l->nkv, e->QMAX, l->swa ? e->g.window : 0, e->oss, e->ascale};
}

// Gemma's rope kernel for head size D over R rows (its arguments and buffers set)
static void g4_rope_dispatch(id<MTLComputeCommandEncoder> c, int D, int R) {
    [c setComputePipelineState:K([NSString stringWithFormat:@"g4_qk_rope_%d", D])];
    [c dispatchThreadgroups:MTLSizeMake(R, 4, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];  // kernels.metal's G4QK_TGS
}

// a Gemma / gpt-oss layer's attention output projection of R rows into y (acc 1: added)
static void g4_out_proj(sn_engine * e, const g4layer_t * l, int R, id<MTLBuffer> y, int acc) {
    mm(g_e, l->fmt, e->ao, 0, gb(e, l->o), go(e, l->o), nil, y, 0, e->NH * l->D, e->E, R, e->QMAX, e->E, acc);
}

// a Gemma / gpt-oss layer's q, k, v of R rows: ns of them as mm_qkv takes them, then v by itself when v_own (its own type)
static void g4_qkv(sn_engine * e, const g4layer_t * l, int R, int ns, bool v_own) {
    const int E = e->E, QD = e->NH * l->D, KVS = l->nkv * l->D, QMAX = e->QMAX, KD = e->KD;
    id<MTLBuffer> w[3] = {gb(e, l->q), gb(e, l->k), gb(e, l->v)}, y[3] = {e->q, e->k, e->v};
    const uint64_t wo[3] = {go(e, l->q), go(e, l->k), go(e, l->v)};
    const int M[3] = {QD, KVS, KVS}, ld[3] = {QMAX, KD, KD};
    mm_qkv(l->fmt, e->h, E, R, ns, w, wo, M, y, ld);
    if (v_own) mm(g_e, l->v_fmt, e->h, 0, w[2], wo[2], nil, e->v, 0, E, KVS, R, E, KD, 0);
}

// Gemma's q, k, v (the shared-KV layers: q only; Gemma 4's layers without v: q and k)
static void gemma_qkv(sn_engine * e, const g4layer_t * l, int Ri) {
    const bool v_own = l->has_kv && l->has_v && l->v_fmt != l->fmt;  // v of its own type: by itself
    g4_qkv(e, l, Ri, l->has_kv ? (l->has_v && !v_own ? 3 : 2) : 1, v_own);
}

// Gemma's q / k / v norms and rope (the full layers: Gemma 4's frequency factors, Gemma 3's scaled positions) and the
// K / V cache write (a layer with its own K/V; no v: v = k)
static void gemma_rope(sn_engine * e, const g4layer_t * l, const layer_attn * a, int Ri) {
    const gguf_t * g = &e->g;
    const bool full_ff = !l->swa && e->g4;
    g4q_args qa = {e->NH, l->nkv, e->ctx, g->eps, l->swa ? g->rope_base_swa : g->rope_base, !l->swa && g->rope_scale > 0 ? 1.0f / g->rope_scale : 1.0f,
                   e->QMAX, e->KD, l->has_kv, full_ff, e->g4};
    enc("qk_rope"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&qa length:sizeof qa atIndex:0];
    [c setBuffer:e->q offset:0 atIndex:1]; [c setBuffer:e->k offset:0 atIndex:2]; [c setBuffer:l->has_v ? e->v : e->k offset:0 atIndex:3];  // no v: v = k
    [c setBuffer:gb(e, l->q_norm) offset:go(e, l->q_norm) atIndex:4]; [c setBuffer:l->k_norm ? gb(e, l->k_norm) : e->k offset:go(e, l->k_norm) atIndex:5];
    [c setBuffer:a->kc offset:a->ko atIndex:6]; [c setBuffer:a->vc offset:a->ko atIndex:7];
    [c setBuffer:e->rseq offset:0 atIndex:8]; [c setBuffer:e->rpos offset:0 atIndex:9];
    [c setBuffer:full_ff ? gb(e, e->rope_ff) : e->k offset:full_ff ? go(e, e->rope_ff) : 0 atIndex:10]; [c setBuffer:a->pt offset:0 atIndex:11]; [c setBuffer:e->k offset:0 atIndex:12];
    g4_rope_dispatch(c, l->D, Ri);
}

// the attention's output projection into y, its post-norm's residual add into x, the FFN's input norm into h
static void gemma_attn_out(sn_engine * e, const g4layer_t * l, int Ri, const norm_args * nrm) {
    const int E = e->E;
    g4_out_proj(e, l, Ri, e->y, 0);
    na_args na = {E, e->g.eps, E, E, 0};
    if (g_nan) {  // the post-attention norm's residual add and the FFN's input norm in one dispatch
        struct { na_args na; norm_args nr; } nn = {na, *nrm};
        enc("norm_add"); id<MTLComputeCommandEncoder> c = g_e;
        [c setBytes:&nn length:sizeof nn atIndex:0];
        [c setBuffer:e->x offset:0 atIndex:1]; [c setBuffer:e->y offset:0 atIndex:2]; [c setBuffer:gb(e, l->post_attn) offset:go(e, l->post_attn) atIndex:3]; [c setBuffer:e->x offset:0 atIndex:4];
        [c setBuffer:gb(e, l->ffn_norm) offset:go(e, l->ffn_norm) atIndex:5]; [c setBuffer:e->h offset:0 atIndex:6];
        dispatch_rows(c, @"g4_norm_add_norm", Ri, 256);
        return;
    }
    enc("norm_add"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&na length:sizeof na atIndex:0];
    [c setBuffer:e->x offset:0 atIndex:1]; [c setBuffer:e->y offset:0 atIndex:2]; [c setBuffer:gb(e, l->post_attn) offset:go(e, l->post_attn) atIndex:3]; [c setBuffer:e->x offset:0 atIndex:4];
    dispatch_rows(c, @"g4_norm_add", Ri, 256);
    x_norm(e, nrm, l->ffn_norm, Ri);
}

// the gate's GELU times layer i's per-layer embedding rows (in pg)
static void gemma_ple_gelu(sn_engine * e, int i, int Ri) {
    const int PLE = e->PLE;
    gm_args gp_ = {PLE, PLE, e->PN, Ri};
    id<MTLComputeCommandEncoder> c = kernel_begin("gelu", @"g4_gelu_mul", &gp_, sizeof gp_);
    [c setBuffer:e->pg offset:0 atIndex:1]; [c setBuffer:e->ple offset:(size_t) i * PLE * 4 atIndex:2];
    [c dispatchThreads:MTLSizeMake(PLE, Ri, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Gemma 4's per-layer embedding of layer i: the FFN's post-norm residual add, the gate of x times the layer's ple rows
// (GELU), projected into y
static void gemma_ple_layer(sn_engine * e, const g4layer_t * l, int i, int Ri) {
    const int E = e->E, PLE = e->PLE;
    na_args na = {E, e->g.eps, E, E, 0};
    enc("norm_add"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&na length:sizeof na atIndex:0];
    [c setBuffer:e->x offset:0 atIndex:1]; [c setBuffer:e->y offset:0 atIndex:2]; [c setBuffer:gb(e, l->post_ffw) offset:go(e, l->post_ffw) atIndex:3]; [c setBuffer:e->x offset:0 atIndex:4];
    dispatch_rows(c, @"g4_norm_add", Ri, 256);
    mm(g_e, 3, e->x, 0, gb(e, l->inp_gate), go(e, l->inp_gate), nil, e->pg, 0, E, PLE, Ri, E, PLE, 0);
    gemma_ple_gelu(e, i, Ri);
    mm(g_e, 3, e->pg, 0, gb(e, l->proj), go(e, l->proj), nil, e->y, 0, PLE, E, Ri, PLE, E, 0);
}

// the layer's last residual add (x = (x + norm(y) w) s, s its output scale when it has one) and the next layer's
// attention norm in one dispatch: after the FFN's post-norm, or after the per-layer embedding's
static void gemma_layer_end(sn_engine * e, int i, int Ri, const norm_args * nrm) {
    const g4layer_t * l = &e->gl[i];
    const int E = e->E, L = e->L;
    uint64_t wlast = l->post_ffw;
    if (e->PLE) {
        gemma_ple_layer(e, l, i, Ri);
        wlast = l->post_norm;
    }
    na_args ns = {E, e->g.eps, E, E, l->out_scale != 0};
    enc("norm_add"); id<MTLComputeCommandEncoder> c = g_e;
    if (g_nan && i + 1 < L) {
        struct { na_args na; norm_args nr; } nn = {ns, *nrm};
        [c setBytes:&nn length:sizeof nn atIndex:0];
        [c setBuffer:gb(e, e->gl[i + 1].attn_norm) offset:go(e, e->gl[i + 1].attn_norm) atIndex:5]; [c setBuffer:e->h offset:0 atIndex:6];
    } else {
        [c setBytes:&ns length:sizeof ns atIndex:0];
    }
    [c setBuffer:e->x offset:0 atIndex:1]; [c setBuffer:e->y offset:0 atIndex:2]; [c setBuffer:gb(e, wlast) offset:go(e, wlast) atIndex:3];
    [c setBuffer:l->out_scale ? gb(e, l->out_scale) : e->x offset:go(e, l->out_scale) atIndex:4];
    dispatch_rows(c, g_nan && i + 1 < L ? @"g4_norm_add_norm" : @"g4_norm_add", Ri, 256);
}

// Gemma 3 / 4: per-layer embeddings (Gemma 4 E2B), then one layer program over the step's rows. nblk: the prompt tiles of
// each tile list (rblk holds list k at k * rmax).
// r_late: the rows layers from e->kvl on run (R, or n_write when no prompt row has an output: the prompt rows' cells are
// all written by then, and the layers after only make their hidden states, which nothing reads)
static void gemma_layers(sn_engine * e, int R, int n_write, const int * nblk, int maxc_w, int r_late) {
    const int E = e->E, FMAX = e->FMAX;
    if (e->PLE) gemma_ple(e, R);
    norm_args nrm = {E, e->g.eps, E, E};
    for (int i = 0; i < e->L; ++i) {
        // past the last layer that writes the cache (Gemma 4's shared-KV layers) only the rows with an output matter
        const int Ri = i >= e->kvl ? r_late : R;
        if (Ri == 0) break;
        const g4layer_t * l = &e->gl[i];
        const layer_attn a = g4_attn_of(e, l);
        if (i == 0 || !g_nan) x_norm(e, &nrm, l->attn_norm, Ri);  // later layers' attention norm came with the layer before's last residual add
        gemma_qkv(e, l, Ri);
        gemma_rope(e, l, &a, Ri);
        // interactive rows with short attention spans: fa1 (as Qwen3's); fa2 measured 102 against 133 tok/s here (8 heads of
        // 512 per K/V head: too many registers)
        if (n_write > 0) g4_attn_fa1(e, &a, n_write, maxc_w);
        const int nb = Ri < R ? 0 : nblk[l->list];
        if (nb > 0) g4_prompt(e, &a, nb, (size_t) l->list * e->rmax * 16);
        gemma_attn_out(e, l, Ri, &nrm);
        mm_glu(l->fmt, e->h, E, Ri, l->ff, gb(e, l->gate), go(e, l->gate), gb(e, l->up), go(e, l->up), e->fg, FMAX, 3);
        mm(g_e, l->down_fmt, e->fg, 0, gb(e, l->down), go(e, l->down), nil, e->y, 0, l->ff, E, Ri, FMAX, E, 0);
        gemma_layer_end(e, i, Ri, &nrm);
    }
}

// gpt-oss's q / k / v biases and YaRN rope, and the K / V cache write
static void oss_rope(sn_engine * e, const g4layer_t * l, const layer_attn * a, int R) {
    const gguf_t * g = &e->g;
    g4q_args qa = {e->NH, l->nkv, e->ctx, g->eps, l->swa ? g->rope_base_swa : g->rope_base, 1.0f / g->rope_scale, e->QMAX, e->KD, 1, 0, 0, 1, e->ylo, e->yhi, e->ymscale};
    enc("qk_rope"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&qa length:sizeof qa atIndex:0];
    [c setBuffer:e->q offset:0 atIndex:1]; [c setBuffer:e->k offset:0 atIndex:2]; [c setBuffer:e->v offset:0 atIndex:3];
    [c setBuffer:gb(e, l->qb) offset:go(e, l->qb) atIndex:4]; [c setBuffer:gb(e, l->kb) offset:go(e, l->kb) atIndex:5];
    [c setBuffer:a->kc offset:a->ko atIndex:6]; [c setBuffer:a->vc offset:a->ko atIndex:7];
    [c setBuffer:e->rseq offset:0 atIndex:8]; [c setBuffer:e->rpos offset:0 atIndex:9]; [c setBuffer:e->k offset:0 atIndex:10]; [c setBuffer:a->pt offset:0 atIndex:11];
    [c setBuffer:gb(e, l->vb) offset:go(e, l->vb) atIndex:12];
    g4_rope_dispatch(c, l->D, R);
}

// the output projection added to x, its bias with the FFN's input norm into h
static void oss_attn_out(sn_engine * e, const g4layer_t * l, int R, const norm_args * nrm) {
    g4_out_proj(e, l, R, e->x, 1);
    enc("norm"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:nrm length:sizeof *nrm atIndex:0];
    [c setBuffer:e->x offset:0 atIndex:1]; [c setBuffer:gb(e, l->ob) offset:go(e, l->ob) atIndex:2];
    [c setBuffer:gb(e, l->ffn_norm) offset:go(e, l->ffn_norm) atIndex:3]; [c setBuffer:e->h offset:0 atIndex:4];
    dispatch_rows(c, @"bias_norm", R, 256);
}

// gpt-oss: one layer program over the step's rows. Attention with Gemma's kernels (q / k / v plus biases, YaRN, a sink per
// head; every other layer a 128-cell window in the windows' pool): interactive rows fa2, other write rows g4_attn_part,
// prompt rows g4_pre; the
// output projection added to x, its bias with the FFN's input norm, then the MoE FFN (moe_ffn's mode 1) added to x
static void oss_layers(sn_engine * e, int R, int n_write, const int * nblk, int maxc_w) {
    norm_args nrm = {e->E, e->g.eps, e->E, e->E};
    for (int i = 0; i < e->L; ++i) {
        const g4layer_t * l = &e->gl[i];
        const layer_attn a = g4_attn_of(e, l);
        x_norm(e, &nrm, l->attn_norm, R);
        g4_qkv(e, l, R, l->v_fmt == l->fmt ? 3 : 2, l->v_fmt != l->fmt);
        oss_rope(e, l, &a, R);
        if (n_write > 0) g4_attn_fa2(e, &a, n_write, maxc_w);  // interactive rows fa2 (as Phi-3's), the other write rows g4_attn_part, merged in chunk order
        const int nb = nblk[l->list];
        if (nb > 0) g4_prompt(e, &a, nb, (size_t) l->list * e->rmax * 16);
        oss_attn_out(e, l, R, &nrm);
        moe_ffn(e, &l->m, R, nil, 0, NULL);
    }
}
