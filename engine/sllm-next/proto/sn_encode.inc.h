// sllm-next engine library, a part of sn.m (one translation unit; sn.m includes its parts in order): a step's encoding: its rows, the layer program, the outputs

// ---- a step's encoding --------------------------------------------------------------------------------------------------
// a step as sn_step's caller asked for it: R rows (the first n_write write rows, of which n_fast interactive; n_fast < 0: a
// wide draft check of -n_fast rows), their tokens (NULL: the other set's outputs), sequences and positions, n_out output rows
typedef struct {
    int R, n_write, n_fast, n_out;
    const int32_t * tok, * seq, * pos, * out_rows;
} step_req;

// the step's row records: row set `set`'s buffers
static void rows_select(sn_engine * e, int set) {
    e->rtok = e->rtokB[set]; e->rseq = e->rseqB[set]; e->rpos = e->rposB[set]; e->ridx = e->ridxB[set];
    e->otok = e->otokB[set]; e->rblk = e->rblkB[set]; e->praw = e->prawB[set];
}

// row r's slot, position or token (when given) out of range
static bool row_bad(const sn_engine * e, const step_req * q, int r) {
    return q->seq[r] < 0 || q->seq[r] >= e->slots || q->pos[r] < 0 || q->pos[r] >= e->ctx || (q->tok && (q->tok[r] < 0 || q->tok[r] >= e->V));
}

// the rows' tokens, sequences and positions into the row records; maxw: the write rows' last position. -1 with err set
// when a row is out of range
static int rows_stage(sn_engine * e, const step_req * q, int * maxw, char * err, int err_len) {
    int * tk = e->rtok.contents, * sq = e->rseq.contents, * ps = e->rpos.contents;
    for (int r = 0; r < q->R; ++r) {
        if (row_bad(e, q, r)) {
            set_err(err, err_len, "%s", "a row is out of range (slot, position or token)");
            return -1;
        }
        if (q->tok) tk[r] = q->tok[r];
        sq[r] = q->seq[r]; ps[r] = q->pos[r];
        if (r < q->n_write) *maxw = MAX(*maxw, q->pos[r]);
    }
    return 0;
}

// a new run (its first row's state entry st): no other run of the step has the entry, and the sequence goes on from its
// last cell (pos); -1 with err set otherwise
static int q35_run_check(const sn_engine * e, const int4 * rn, int n, int st, int seq, int pos, char * err, int err_len) {
    for (int j = 0; j < n; ++j) if (rn[j].z == st) { set_err(err, err_len, "%s", "a sequence's rows must be one run of consecutive positions"); return -1; }
    if (pos != e->slen[seq]) { set_err(err, err_len, "%s", "a recurrent sequence goes on from its last cell only"); return -1; }
    return 0;
}

// the runs: each sequence's rows (consecutive positions) with its state entry (e->nruns of them); 0, or -1 / -2 with err set
static int q35_runs_build(sn_engine * e, int R, char * err, int err_len) {
    const int * sq = e->rseq.contents, * ps = e->rpos.contents;
    int4 * rn = e->runs.contents;
    int * rr = e->rrun.contents, * rs = e->rsi.contents, n = 0;
    for (int r = 0; r < R; ++r) {
        const int st = q35_entry(e, sq[r]);
        if (st < 0) { set_err(err, err_len, "%s", "out of recurrent states"); return -2; }
        if (r > 0 && sq[r] == sq[r - 1] && ps[r] == ps[r - 1] + 1) { rr[r] = rr[r - 1] + 1; ++rn[n - 1].y; }
        else {
            if (q35_run_check(e, rn, n, st, sq[r], ps[r], err, err_len)) return -1;
            rr[r] = 0; rn[n++] = (int4){r, 1, st, 0};
        }
        rs[r] = st;
        e->slen[sq[r]] = ps[r] + 1;
    }
    e->nruns = n;
    return 0;
}

// run `run` (y rows) of sequence s takes its spare entries d: the state after each row but the last into one
static void q35_snap_run(sn_engine * e, int4 run, int s, const int * d, int * sp, int4 * cp) {
    const int * ps = e->rpos.contents;
    const int y = run.y;
    for (int i = 0; i < y - 1; ++i) { sp[run.x + i] = d[i]; cp[e->nsnap++] = (int4){run.x + i, i + 1, run.z, d[i]}; }
    e->dn[s] = y - 1; e->dpos[s] = ps[run.x];
    if (e->drow) e->drow[s] = run.x;
}

// an interactive run of 2 .. 4 rows whose sequence holds the spare entries: the state after each row but the last
static void q35_snaps(sn_engine * e, int set, int R, int n_fast) {
    const int * sq = e->rseq.contents;
    const int4 * rn = e->runs.contents;
    e->rsnap = e->rsnapB[set]; e->csnap = e->csnapB[set]; e->nsnap = 0;
    int * sp = e->rsnap.contents;
    int4 * cp = e->csnap.contents;
    for (int r = 0; r < R; ++r) sp[r] = -1;
    for (int j = 0; j < e->nruns; ++j) {
        const int s = sq[rn[j].x], y = rn[j].y;
        e->dn[s] = 0;
        if (n_fast <= 0 || rn[j].x + y > n_fast || y < 2 || y > 4) continue;
        const int * d = e->dent + s * 3;
        if (d[y - 2] < 0) continue;
        q35_snap_run(e, rn[j], s, d, sp, cp);
    }
}

// Qwen3.5: the runs (each sequence's rows with its state entry) and the draft rows' snapshots; 0, or -1 / -2 with err set
static int q35_runs(sn_engine * e, int set, const step_req * q, char * err, int err_len) {
    e->runs = e->runsB[set]; e->rrun = e->rrunB[set]; e->rsi = e->rsiB[set];
    const int rc = q35_runs_build(e, q->R, err, err_len);
    if (rc) return rc;
    q35_snaps(e, set, q->R, q->n_fast);
    return 0;
}

// the write rows' attention partials (maxc_w chunks each, their last position maxw) and the outputs' scores (4 rows apart
// for a queued step): sized by the steps that run, not by the slots (a step already committed keeps the buffer it was
// given); returns maxc_w
static int step_buffers(sn_engine * e, const step_req * q, int maxw) {
    const int maxc_w = maxw / 64 + 1;
    fit(&e->apart, (size_t) MAX(q->n_write, 1) * e->NH * maxc_w * ((e->gemma || e->q35 ? e->g.hd : 128) + 2) * 4);
    fit(&e->logits, (size_t) (MAX(q->n_out, 1) + 4) * e->V * 4);
    return maxc_w;
}

// rows n_write .. R into tiles of consecutive positions of one sequence, tt tokens at most (bl); their count
static int tile_rows(int4 * bl, const int * sq, const int * ps, int n_write, int R, int tt) {
    int n = 0;
    for (int r = n_write; r < R; ++r) {
        if (n > 0 && bl[n - 1].z == sq[r] && bl[n - 1].y < tt && ps[r] == ps[r - 1] + 1) bl[n - 1].y++;
        else bl[n++] = (int4){r, 1, sq[r], 0};
    }
    return n;
}

// the output rows, and the prompt rows in tiles of consecutive positions of one sequence, 32 / G tokens (Qwen3 G = 4: 8;
// Gemma: a list per G; nblk: each list's tiles); -1 with err set when Gemma's tiles do not fit
static int step_tiles(sn_engine * e, const step_req * q, int nblk[2], char * err, int err_len) {
    int * ix = e->ridx.contents;
    const int * sq = e->rseq.contents, * ps = e->rpos.contents;
    for (int j = 0; j < q->n_out; ++j) ix[j] = q->out_rows[j];
    for (int k = 0; k < (e->gemma ? e->ntt : 1); ++k) {
        const int tt = e->gemma ? e->tt[k] : 32 / (e->NH / e->g.n_kv);
        const int n = tile_rows((int4 *) e->rblk.contents + (size_t) k * e->rmax, sq, ps, q->n_write, q->R, tt);
        if (e->gemma && (size_t) n * e->ttb[k] > e->qhb.length) { set_err(err, err_len, "%s", "too many prompt tiles (rows of many sequences)"); return -1; }
        nblk[k] = n;
    }
    return 0;
}

// the rows the CPU copies (an untied table off the GPU, Gemma 4's per-layer rows) of the step's tokens (when given)
static void rows_cpu_stage(sn_engine * e, int set, const step_req * q) {
    const int * tk = e->rtok.contents;
    e->erow_b = e->erowB[set];
    if (e->emb_cpu && q->tok) for (int r = 0; r < q->R; ++r) memcpy((uint8_t *) e->erow_b.contents + r * e->erow, e->etab + (size_t) tk[r] * e->erow, e->erow);
    if (e->PLE && q->tok) for (int r = 0; r < q->R; ++r) memcpy((uint8_t *) e->praw.contents + r * e->ple_row, e->ple_tab + (size_t) tk[r] * e->ple_row, e->ple_row);
}

// soft words: the rows that read a vector, the outputs that make one (as (row or output, vector slot) pairs for the kernels)
static void soft_stage(sn_engine * e, int set, const step_req * q) {
    e->soft_nin[set] = e->soft_nout[set] = 0;
    if (!e->soft_sv || (e->sp_nin <= 0 && e->sp_nout <= 0)) return;
    const int * sq = e->rseq.contents;
    soft_i2 * si = e->soft_inB[set].contents, * so = e->soft_outB[set].contents;
    for (int j = 0; j < e->sp_nin; ++j) {
        const int s = sq[e->sp_in[j]];
        si[j] = (soft_i2){e->sp_in[j], e->soft_par[s] * e->slots + s};
    }
    for (int j = 0; j < e->sp_nout; ++j) {
        const int s = sq[q->out_rows[e->sp_out[j]]];
        so[j] = (soft_i2){e->sp_out[j], (e->soft_par[s] ^ 1) * e->slots + s};
        e->soft_oseq[set * (e->slots + 4) + j] = s;
    }
    for (int j = 0; j < e->sp_nout; ++j) e->soft_par[e->soft_oseq[set * (e->slots + 4) + j]] ^= 1;
    e->soft_nin[set] = e->sp_nin; e->soft_nout[set] = e->sp_nout;
}

// a queued step's tokens: the other set's outputs, copied on the GPU (after its per-layer / staged rows are in: ev)
static void rows_from_prev(sn_engine * e, int set, int R) {
    if (e->PLE || e->emb_cpu) [g_cb encodeWaitForEvent:e->ev value:e->ev_val + 1];
    id<MTLBlitCommandEncoder> be = [g_cb blitCommandEncoder];
    [be copyFromBuffer:e->otokB[set ^ 1] sourceOffset:0 toBuffer:e->rtok destinationOffset:0 size:(size_t) R * 4];
    [be endEncoding];
}

// the step's kinds of rows for the kernels (core.h's g_*); returns n_fast at 0 or more
static int step_rows_kinds(const step_req * q) {
    int n_fast = q->n_fast;
    g_e = nil;
    g_np = 0;
    g_nAtt = n_fast < 0 ? -n_fast : n_fast;  // a wide draft check: its rows' attention on the one-row kernels, their matmuls blocks
    g_wide = n_fast < 0;
    n_fast = MAX(n_fast, 0);
    g_nA = n_fast;
    if (g_chatbatch && n_fast > 0 && q->n_write > n_fast) g_nA = 0;  // SN_CHATBATCH: the chat's matmul rows join the background rows'
    g_nsw = q->n_write - (g_nA ? n_fast : 0);  // (a wide check's n_fast < 0 counted here: g_small_step off for it anyway)
    return n_fast;
}

// SN_LATE 8: a mixed write step, the chat on its own kernels
static bool late_conc(int R, int n_write) { return (g_late & 8) && g_nA > 0 && g_nA < R && n_write == R; }

// the step's encoder kind (concurrent or not), its few-row steps and profiling
static void step_encoders(int R, int n_write, int n_fast) {
    g_small_step = !g_wide && g_nsw >= 1 && g_nsw <= g_small;
    g_conc_step = g_concsmall && n_fast == 0 && n_write == R && R <= 16;
    g_conc = g_conc_env || g_conc_step || late_conc(R, n_write);
    g_prof = g_prof_r && R == g_prof_r;  // SN_PROF: this step's kernels timed one by one
}

// the token rows' embeddings into x
static void embed_x(sn_engine * e, int R) { embed_into(e, e->x, R); }

// rows n of x replaced by vectors (src; rows idx) as kernel `name` takes them: latent rows, soft silent rows
static void x_rows_in(sn_engine * e, const char * cat, NSString * name, id<MTLBuffer> src, id<MTLBuffer> idx, int n) {
    n_args la = {e->E};
    enc(cat); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&la length:sizeof la atIndex:0];
    [c setBuffer:src offset:0 atIndex:1]; [c setBuffer:idx offset:0 atIndex:2]; [c setBuffer:e->x offset:0 atIndex:3];
    dispatch_rows(c, name, n, 256);
}

// the rows' inputs: their tokens' embeddings, latent rows' vectors, soft silent rows' vectors
static void embed_stage(sn_engine * e, int set, int R) {
    embed_x(e, R);
    if (e->lat_n > 0) x_rows_in(e, "latin", @"latin", e->lat_in, e->lat_idx, e->lat_n);  // latent rows: their vectors in place of the embeddings just written
    if (e->soft_nin[set] > 0) x_rows_in(e, "latin_sv", @"latin_sv", e->soft_sv, e->soft_inB[set], e->soft_nin[set]);  // soft silent rows: the vector their sequence's step before made
}

// the rows Gemma's layers from e->kvl on run (SN_KVSKIP=0: every row through every layer): the write rows when no prompt
// row has an output and no row reads a vector
static int gemma_r_late(const sn_engine * e, int set, const step_req * q) {
    int r_late = q->R;  // SN_KVSKIP=0: every row through every layer
    if (e->kvl < e->L && g_kvskip) {
        r_late = q->n_write;
        for (int j = 0; j < q->n_out; ++j) if (q->out_rows[j] >= q->n_write) r_late = q->R;
        if (e->lat_n > 0 || e->soft_nin[set] > 0) r_late = q->R;
    }
    return r_late;
}

// the family's layer program over the step's rows (Qwen3.5-MoE guessing: the guess head's cells too)
static void layers_run(sn_engine * e, int set, const step_req * q, const int * nblk, int maxc_w) {
    const int R = q->R, n_write = q->n_write;
    if (e->oss) oss_layers(e, R, n_write, nblk, maxc_w);
    else if (e->gemma) gemma_layers(e, R, n_write, nblk, maxc_w, gemma_r_late(e, set, q));
    else if (e->q35) {
        q35_layers(e, e->ql, e->L, R, n_write, nblk[0], maxc_w);
        if (e->mtp && e->guess_on) mtp_fill(e, R);
    }
    else qwen_layers(e, R, n_write, nblk[0], maxc_w);
}

// whether the interactive rows (the first n_fast) are the first outputs, in order (the output layer's one-row path)
static bool fast_first(const step_req * q, int n_fast) {
    bool ff = n_fast <= q->n_out;
    for (int j = 0; j < n_fast && ff; ++j) ff = q->out_rows[j] == j;
    return ff;
}

// the output rows gathered (xo) and normed (ho); latent: their hidden states for the caller (sn_latent)
static void out_norm(sn_engine * e, int n_out) {
    const int E = e->E;
    n_args na1 = {E};
    enc("gather"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&na1 length:sizeof na1 atIndex:0];
    [c setBuffer:e->x offset:0 atIndex:1]; [c setBuffer:e->ridx offset:0 atIndex:2]; [c setBuffer:e->xo offset:0 atIndex:3];
    dispatch_rows(c, @"gather", n_out, 256);
    norm_args nrm = {E, e->g.eps, E, E};
    enc("rmsnorm"); c = g_e;
    [c setBytes:&nrm length:sizeof nrm atIndex:0];
    [c setBuffer:e->xo offset:0 atIndex:1]; [c setBuffer:gb(e, e->out_norm) offset:go(e, e->out_norm) atIndex:2];
    [c setBuffer:e->ho offset:0 atIndex:3];
    dispatch_rows(c, @"rmsnorm", n_out, 256);
    if (e->lat_dst) {  // the output rows' hidden states after the final norm, for the caller (sn_latent)
        enc("gather"); c = g_e;
        [c setBytes:&na1 length:sizeof na1 atIndex:0];
        [c setBuffer:e->ho offset:0 atIndex:1]; [c setBuffer:e->lat_iota offset:0 atIndex:2]; [c setBuffer:e->lat_out offset:0 atIndex:3];
        dispatch_rows(c, @"gather", n_out, 256);
    }
}

// each output row's best token (sn_no_end rows never an end-of-generation one); sn_want_probs: argmax_p, which also
// writes each row's record
static void out_argmax(sn_engine * e, int set, const step_req * q, size_t lo) {
    const int n_out = q->n_out;
    argmax_args va = {e->V, 0, {0}};
    uint8_t skip[n_out + 16];  // output rows that skip the end-of-generation tokens (sn_no_end)
    memset(skip, 0, sizeof skip);
    for (int j = 0; j < n_out && e->neog; ++j) if ((skip[j] = e->no_end[q->seq[q->out_rows[j]]])) va.nban = e->neog;
    memcpy(va.ban, e->eog, sizeof va.ban);
    const bool p = e->want_probs;  // the same choice as argmax, and each row's two best scores and log-sum-exp
    if (p) e->pwant[set] = 1;
    enc(p ? "argmax_p" : "argmax"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&va length:sizeof va atIndex:0];
    [c setBuffer:e->logits offset:lo atIndex:1]; [c setBuffer:e->otok offset:0 atIndex:2];
    [c setBytes:skip length:(n_out + 15) / 16 * 16 atIndex:3];
    if (p) [c setBuffer:e->pB[set] offset:0 atIndex:4];
    dispatch_rows(c, p ? @"argmax_p" : @"argmax", n_out, 1024);
}

// soft words: each soft output row's mix of its best words' input rows, for the next step
static void out_soft_mix(sn_engine * e, int set, size_t lo) {
    const int E = e->E;
    struct { int V, E, K, fmt; float temp, scale; int neog; int eog[32]; } sa = {e->V, E, e->soft_k, e->embd_fmt, e->soft_t,
        e->gemma && !e->oss ? sqrtf((float) E) : 1.0f, e->soft_neog, {0}};
    memcpy(sa.eog, e->soft_eog, sizeof sa.eog);
    enc("soft_mix"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&sa length:sizeof sa atIndex:0];
    [c setBuffer:e->logits offset:lo atIndex:1]; [c setBuffer:e->soft_outB[set] offset:0 atIndex:2];
    [c setBuffer:gb(e, e->embd) offset:go(e, e->embd) atIndex:3]; [c setBuffer:e->soft_sv offset:0 atIndex:4];
    dispatch_rows(c, @"soft_mix", e->soft_nout[set], 1024);
}

// the rows asked for: final norm, output layer, greedy argmax. The interactive rows are the first n_fast rows of
// the step and the first of out_rows only when the caller lists them first; the output layer's one-row path needs
// that, so it takes it only then. A few outputs go to the logits rows of their set (4 apart).
static void outputs(sn_engine * e, int set, const step_req * q, int n_fast) {
    const int E = e->E, n_out = q->n_out;
    e->lofs[set] = n_out <= 4 && n_out <= e->slots ? 4 * set : 0;  // rows 4 .. 4 + n_out fit the slots + 4 rows
    e->pwant[set] = 0;
    if (n_out <= 0) return;
    const size_t lo = (size_t) e->lofs[set] * e->V * 4;
    g_nA = fast_first(q, n_fast) ? n_fast : 0;
    out_norm(e, n_out);
    if (e->gemma && !e->head) mm(g_e, 1, e->ho, 0, e->head41, 0, nil, e->logits, lo, E, e->V, n_out, E, e->V, 0);
    else mm(g_e, e->head_fmt, e->ho, 0, gb(e, e->head), go(e, e->head), nil, e->logits, lo, E, e->V, n_out, E, e->V, 0);
    g_nA = 0;
    out_argmax(e, set, q, lo);
    if (e->soft_nout[set] > 0) out_soft_mix(e, set, lo);
}

// Encode and commit one step in row set `set`. tok NULL: the tokens are the outputs of the other set's step, copied on the
// GPU (a queued write step; Gemma 4's also waits for ev, signalled once its per-layer rows are in).
static int encode(sn_engine * e, int set, int R, const int32_t * tok, const int32_t * seq, const int32_t * pos, int n_write, int n_fast,
                  int n_out, const int32_t * out_rows, char * err, int err_len) {
    const double t_in = g_time ? now_s() : 0;
    const step_req q = {R, n_write, n_fast, n_out, tok, seq, pos, out_rows};
    rows_select(e, set);
    int maxw = 0;
    if (rows_stage(e, &q, &maxw, err, err_len)) return -1;
    if (pages_for(e, R, seq, pos) != 0) { set_err(err, err_len, "%s", "out of cache pages"); return -2; }
    const int rc = e->q35 ? q35_runs(e, set, &q, err, err_len) : 0;  // the runs: each sequence's rows (consecutive positions) with its state entry
    if (rc) return rc;
    const int maxc_w = step_buffers(e, &q, maxw);
    int nblk[2] = {0, 0};
    if (step_tiles(e, &q, nblk, err, err_len)) return -1;
    rows_cpu_stage(e, set, &q);
    g_cb = [e->queue commandBuffer];
    e->cbs[set] = g_cb;
    soft_stage(e, set, &q);
    if (!tok) rows_from_prev(e, set, R);
    n_fast = step_rows_kinds(&q);
    step_encoders(R, n_write, n_fast);
    embed_stage(e, set, R);
    layers_run(e, set, &q, nblk, maxc_w);
    outputs(e, set, &q, n_fast);
    [g_e endEncoding];
    g_e = nil;
    if (g_time) g_t_enc += now_s() - t_in;
    [g_cb commit];
    return 0;
}
