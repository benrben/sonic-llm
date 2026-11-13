// sllm-next engine library, a part of sn.m (one translation unit; sn.m includes its parts in order): steps: finishing, the queued step, sn_step; soft words, latent rows, embeddings

// Wait for set `set`'s step; its first n_out tokens
static int finish(sn_engine * e, int set, int n_out, int32_t * out_tok, char * err, int err_len) {
    id<MTLCommandBuffer> cb = e->cbs[set];
    const double t_c = g_time ? now_s() : 0;
    [cb waitUntilCompleted];
    e->cbs[set] = nil;
    if (g_time) {
        g_t_wait += now_s() - t_c; ++g_t_n;
        g_t_gpu += cb.GPUEndTime - cb.GPUStartTime; g_t_lat += cb.GPUStartTime - cb.kernelStartTime;
    }
    if (cb.error) { set_err(err, err_len, "step failed: %s", cb.error.localizedDescription.UTF8String); return -1; }
    e->gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    if (g_prof && g_np > 0) {  // SN_PROF: each kernel's GPU time, summed by its name over the profiled steps
        NSData * d = [g_sb resolveCounterRange:NSMakeRange(0, 2 * g_np)];
        const MTLCounterResultTimestamp * ts = d.bytes;
        for (int i = 0; i < g_np; ++i) { NSString * k = @(g_cat[i]); g_prof_sum[k] = @([g_prof_sum[k] doubleValue] + (ts[2 * i + 1].timestamp - ts[2 * i].timestamp) / 1e6); }
        ++g_prof_steps;
        g_prof = 0;
    }
    if (n_out > 0) memcpy(out_tok, ((id<MTLBuffer>) e->otokB[set]).contents, n_out * 4);
    e->last_pn = 0;
    if (n_out > 0 && e->pwant[set]) { memcpy(e->last_probs, ((id<MTLBuffer>) e->pB[set]).contents, (size_t) n_out * 32); e->last_pn = n_out; }
    e->lastofs = e->lofs[set];
    return 0;
}

static void clear_cells(sn_engine * e, int n, const int * seq, const int * pos);

// The queued step is not wanted (the next step is another, or a cache call comes first): let it end, zero its cells
static void settle(sn_engine * e) {
    if (e->ch_set < 0) return;
    const int s = e->ch_set;
    e->ch_set = -1;
    ++g_ch_drop;
    if ((e->PLE || e->emb_cpu) && e->ev.signaledValue < e->ev_val + 1) e->ev.signaledValue = ++e->ev_val;  // never left waiting
    [e->cbs[s] waitUntilCompleted];
    e->cbs[s] = nil;
    // the soft vectors the dropped step made are not wanted: each sequence's current half is the one before (the dropped step
    // read it and wrote the other)
    for (int j = 0; j < e->soft_nout[s]; ++j) e->soft_par[e->soft_oseq[s * (e->slots + 4) + j]] ^= 1;
    e->soft_nout[s] = e->soft_nin[s] = 0;
    clear_cells(e, e->ch_R, e->ch_seq, e->ch_pos);
}

// the queued step's positions (each row's next), p1; false when one passes the window
static bool next_positions(const sn_engine * e, int R, const int32_t * pos, int32_t * p1) {
    for (int r = 0; r < R; ++r) { p1[r] = pos[r] + 1; if (p1[r] >= e->ctx) return false; }
    return true;
}

// soft silent rows go on silent: the same rows read the vectors this step makes (a step is chained as such only when each
// soft row is both: its output k is its row); false when set's step cannot be chained so
static bool soft_chain(sn_engine * e, int set, const int32_t * out_rows) {
    if (!e->soft_sv || (e->soft_nin[set] <= 0 && e->soft_nout[set] <= 0)) return true;
    id<MTLBuffer> sib = e->soft_inB[set], sob = e->soft_outB[set];
    const soft_i2 * si = sib.contents, * so = sob.contents;
    if (e->soft_nin[set] != e->soft_nout[set]) return false;
    for (int j = 0; j < e->soft_nout[set]; ++j) if (si[j].x != so[j].x || out_rows[so[j].x] != so[j].x) return false;
    for (int j = 0; j < e->soft_nout[set]; ++j) e->sp_in[j] = e->sp_out[j] = so[j].x;
    e->sp_nin = e->sp_nout = e->soft_nout[set];
    return true;
}

// Queue the write step after the one in `set` (rows seq at pos + 1, fed that step's outputs) when it can be
static void queue_next(sn_engine * e, int set, int R, const int32_t * seq, const int32_t * pos, int n_fast, const int32_t * out_rows) {
    int32_t p1[R];
    if (!next_positions(e, R, pos, p1)) return;
    e->sp_nin = e->sp_nout = 0;
    if (!soft_chain(e, set, out_rows)) return;
    const int rc = encode(e, set ^ 1, R, NULL, seq, p1, R, n_fast, R, out_rows, NULL, 0);
    e->sp_nin = e->sp_nout = 0;
    if (rc != 0) return;
    e->ch_set = set ^ 1; e->ch_R = R; e->ch_fast = n_fast;
    for (int r = 0; r < R; ++r) { e->ch_seq[r] = seq[r]; e->ch_pos[r] = p1[r]; }
}

// The step before the queued one has its tokens: they are what the queued step is fed (the rows the CPU copies, then go)
static void release_next(sn_engine * e, const int32_t * toks) {
    if (e->ch_set < 0) return;
    for (int r = 0; r < e->ch_R; ++r) e->ch_tok[r] = toks[r];
    if (e->PLE) {
        uint8_t * pr = ((id<MTLBuffer>) e->prawB[e->ch_set]).contents;
        for (int r = 0; r < e->ch_R; ++r) memcpy(pr + r * e->ple_row, e->ple_tab + (size_t) toks[r] * e->ple_row, e->ple_row);
    }
    if (e->emb_cpu) {
        uint8_t * er = ((id<MTLBuffer>) e->erowB[e->ch_set]).contents;
        for (int r = 0; r < e->ch_R; ++r) memcpy(er + r * e->erow, e->etab + (size_t) toks[r] * e->erow, e->erow);
    }
    if (e->PLE || e->emb_cpu) e->ev.signaledValue = ++e->ev_val;
}

static void lat_reset(sn_engine * e) { e->lat_n = 0; e->lat_dst = NULL; }  // latent rows are for one step

// a step's shape sn_step does not take
static bool step_shape_bad(const sn_engine * e, int R, int n_write, int n_fast, int n_out) {
    return R <= 0 || R > e->rmax || n_write > R || n_write > e->slots + 4 || n_fast > 4 || -n_fast > 8 || abs(n_fast) > n_write || n_out > e->slots + 4;
}

// soft rows (sn_soft_rows) outside the step (R rows, n_out outputs), or soft rows without sn_soft_config
static bool rows_within(const int * rows, int n, int lim) {
    for (int j = 0; j < n; ++j) if (!(rows[j] >= 0 && rows[j] < lim)) return false;
    return true;
}
static bool soft_rows_bad(const sn_engine * e, int R, int n_out) {
    if (e->sp_nin <= 0 && e->sp_nout <= 0) return false;
    if (!(e->soft_sv != nil && (n_out > 0 || e->sp_nout == 0))) return true;
    return !rows_within(e->sp_in, e->sp_nin, R) || !rows_within(e->sp_out, e->sp_nout, n_out);
}

// set s's step waited for and its tokens taken; the queued step behind it released (or, on failure, let go)
static int finish_step(sn_engine * e, int s, int n_out, int32_t * out_tok, char * err, int err_len) {
    const int rc = finish(e, s, n_out, out_tok, err, err_len);
    if (rc == 0) release_next(e, out_tok);
    else settle(e);
    return rc;
}

// latent rows (sn_latent): a step of its own, never the queued one, none queued behind
static int step_latent(sn_engine * e, const step_req * q, int32_t * out_tok, char * err, int err_len) {
    for (int j = 0; j < e->lat_n; ++j) {
        const int r = ((const int32_t *) e->lat_idx.contents)[j];
        if (r < 0 || r >= q->R) { set_err(err, err_len, "%s", "a latent row is outside the step"); lat_reset(e); return -1; }
    }
    settle(e);
    const int s = e->ch_last ^ 1;
    e->ch_last = s;
    float * dst = e->lat_dst;
    int rc = encode(e, s, q->R, q->tok, q->seq, q->pos, q->n_write, q->n_fast, q->n_out, q->out_rows, err, err_len);
    lat_reset(e);
    if (rc == 0) rc = finish(e, s, q->n_out, out_tok, err, err_len);
    if (rc == 0 && dst) memcpy(dst, e->lat_out.contents, (size_t) q->n_out * e->E * 4);
    return rc;
}

// a write step that samples each row in order (q35: a dropped queued step would have moved the states)
static bool queue_shape(const sn_engine * e, const step_req * q) {
    bool shape = !e->q35 && q->R == q->n_write && q->n_out == q->R && q->n_out <= e->slots;
    for (int j = 0; j < q->n_out && shape; ++j) shape = q->out_rows[j] == j;
    return shape;
}

// soft silent rows: the queued step reads and makes the same vectors (silent: its silent rows, whose tokens are not used)
static bool queued_soft_same(const sn_engine * e, uint8_t * silent) {
    const int cs = e->ch_set;
    id<MTLBuffer> sib = e->soft_inB[cs], sob = e->soft_outB[cs];
    bool same = e->sp_nin == e->soft_nin[cs] && e->sp_nout == e->soft_nout[cs];
    for (int j = 0; j < e->sp_nin && same; ++j) { same = e->sp_in[j] == ((const soft_i2 *) sib.contents)[j].x; if (same) silent[e->sp_in[j]] = 1; }
    for (int j = 0; j < e->sp_nout && same; ++j) same = e->sp_out[j] == ((const soft_i2 *) sob.contents)[j].x;
    return same;
}

// the queued step's rows are the step's: sequences, positions and tokens (but the silent rows': it was fed the last step's
// best words, the caller passes any)
static bool queued_rows_same(const sn_engine * e, const step_req * q, const uint8_t * silent) {
    for (int r = 0; r < q->R; ++r)
        if (!(q->seq[r] == e->ch_seq[r] && q->pos[r] == e->ch_pos[r] && (silent[r] || q->tok[r] == e->ch_tok[r]))) return false;
    return true;
}

// whether the step asked for is the queued one
static bool queued_same(const sn_engine * e, const step_req * q, bool shape) {
    bool same = shape && q->R == e->ch_R && q->n_fast == e->ch_fast && e->pwant[e->ch_set] == (e->want_probs != 0);
    // soft silent rows: the queued step must read and make the same vectors; their tokens are not used (it was fed the
    // last step's best words, the caller passes any)
    uint8_t silent[q->R];
    memset(silent, 0, q->R);
    if (same) same = queued_soft_same(e, silent);
    return same && queued_rows_same(e, q, silent);
}

// The queued step when it is the step asked for (whether or not the caller chains on from it; next: queue the one after):
// its rc; else 1, the queued step let go
static int queued_take(sn_engine * e, const step_req * q, bool shape, bool next, int32_t * out_tok, char * err, int err_len) {
    if (!queued_same(e, q, shape)) {
        settle(e);
        return 1;
    }
    const int s = e->ch_set;
    e->ch_set = -1;
    ++g_ch_used;
    if (next) queue_next(e, s, q->R, q->seq, q->pos, q->n_fast, q->out_rows);
    return finish_step(e, s, q->n_out, out_tok, err, err_len);
}

static int step_inner(sn_engine * e, int R, const int32_t * tok, const int32_t * seq, const int32_t * pos, int n_write, int n_fast,
                      int n_out, const int32_t * out_rows, int32_t * out_tok, int chain, char * err, int err_len) {
    @autoreleasepool {
        use(e);
        if (step_shape_bad(e, R, n_write, n_fast, n_out)) {
            set_err(err, err_len, "%s", "bad step shape");
            lat_reset(e);
            return -1;
        }
        if (soft_rows_bad(e, R, n_out)) { set_err(err, err_len, "%s", "a soft row is outside the step"); lat_reset(e); return -1; }  // soft rows (sn_soft_rows): inside the step
        const step_req q = {R, n_write, n_fast, n_out, tok, seq, pos, out_rows};
        if (e->lat_n > 0 || e->lat_dst) return step_latent(e, &q, out_tok, err, err_len);
        // a write step that samples each row in order can have the next one queued behind it
        const bool shape = queue_shape(e, &q);
        const bool next = chain && shape;
        if (e->ch_set >= 0) {  // the step asked for may be the queued one (whether or not the caller chains on from it)
            const int rc = queued_take(e, &q, shape, next, out_tok, err, err_len);
            if (rc != 1) return rc;
        }
        const int s = e->ch_last ^ 1;
        e->ch_last = s;
        const int rc0 = encode(e, s, R, tok, seq, pos, n_write, n_fast, n_out, out_rows, err, err_len);
        if (rc0 != 0) return rc0;
        if (next) queue_next(e, s, R, seq, pos, n_fast, out_rows);
        return finish_step(e, s, n_out, out_tok, err, err_len);
    }
}

int sn_step(sn_engine * e, int R, const int32_t * tok, const int32_t * seq, const int32_t * pos, int n_write, int n_fast,
            int n_out, const int32_t * out_rows, int32_t * out_tok, int chain, char * err, int err_len) {
    const int rc = step_inner(e, R, tok, seq, pos, n_write, n_fast, n_out, out_rows, out_tok, chain, err, err_len);
    e->sp_nin = e->sp_nout = 0;  // a soft request is for one step
    return rc;
}

// sn_soft_config's model and arguments: 0, or -1 with err set
// a model whose embedding stage the soft words cannot take (per-layer rows, a staged or q4_1 table, recurrent)
static bool soft_model_bad(const sn_engine * e) { return e->PLE || e->emb_cpu || (e->gemma && e->head41) || e->q35; }
// K outside 1 .. 64 (or more than the vocabulary less the end ids), T not above 0, more than 32 end ids
static bool soft_args_bad(const sn_engine * e, float temp, int k, int n_eog) {
    return k < 1 || k > 64 || n_eog < 0 || n_eog > 32 || !(temp > 0) || k > e->V - n_eog;
}

static int soft_config_check(const sn_engine * e, float temp, int k, const int32_t * eog, int n_eog, char * err, int err_len) {
    if (soft_model_bad(e)) {
        set_err(err, err_len, "%s", "soft words on the GPU: not for this model's embedding stage (per-layer rows, a staged or q4_1 table, recurrent)");
        return -1;
    }
    if (soft_args_bad(e, temp, k, n_eog)) {
        set_err(err, err_len, "%s", "soft words on the GPU: K is 1 .. 64, T above 0, at most 32 end-of-generation ids");
        return -1;
    }
    for (int i = 0; i < n_eog; ++i) if (eog[i] < 0 || eog[i] >= e->V) { set_err(err, err_len, "%s", "soft words on the GPU: end-of-generation id out of range"); return -1; }
    return 0;
}

// the soft vectors (two a slot), the steps' soft row records and the requests' tables; false when out of memory
static bool soft_alloc(sn_engine * e) {
    const int nmax = e->slots + 4;
    e->soft_sv = [dev newBufferWithLength:(size_t) 2 * e->slots * e->E * 4 options:MTLResourceStorageModePrivate];
    for (int s = 0; s < 2; ++s) {
        e->soft_inB[s] = [dev newBufferWithLength:(size_t) e->rmax * 8 options:MTLResourceStorageModeShared];
        e->soft_outB[s] = [dev newBufferWithLength:(size_t) nmax * 8 options:MTLResourceStorageModeShared];
    }
    e->soft_par = calloc(e->slots, 1);
    e->sp_in = calloc((size_t) e->rmax + nmax + 2 * nmax, sizeof(int));
    e->sp_out = e->sp_in + e->rmax;
    e->soft_oseq = e->sp_out + nmax;
    return e->soft_sv && e->soft_par && e->sp_in;
}

int sn_soft_config(sn_engine * e, float temp, int k, const int32_t * eog, int n_eog, char * err, int err_len) {
    @autoreleasepool {
        if (soft_config_check(e, temp, k, eog, n_eog, err, err_len)) return -1;
        use(e);
        settle(e);
        if (!e->soft_sv && !soft_alloc(e)) { set_err(err, err_len, "%s", "soft words on the GPU: out of memory"); return -1; }
        e->soft_t = temp; e->soft_k = k; e->soft_neog = n_eog;
        memset(e->soft_eog, 0, sizeof e->soft_eog);
        for (int i = 0; i < n_eog; ++i) e->soft_eog[i] = eog[i];
        e->sp_nin = e->sp_nout = 0;
        return 0;
    }
}

int sn_soft_rows(sn_engine * e, int n_in, const int32_t * in_rows, int n_out, const int32_t * out_ks, char * err, int err_len) {
    e->sp_nin = e->sp_nout = 0;
    if (!e->soft_sv) { set_err(err, err_len, "%s", "soft rows: sn_soft_config first"); return -1; }
    if (n_in < 0 || n_in > e->rmax || n_out < 0 || n_out > e->slots + 4) { set_err(err, err_len, "%s", "soft rows: too many"); return -1; }
    for (int j = 0; j < n_in; ++j) e->sp_in[j] = in_rows[j];
    for (int j = 0; j < n_out; ++j) e->sp_out[j] = out_ks[j];
    e->sp_nin = n_in; e->sp_nout = n_out;
    return 0;
}

int sn_n_embd(const sn_engine * e) { return e->E; }

int sn_latent(sn_engine * e, int n, const int32_t * rows, const float * vec, float * hid, char * err, int err_len) {
    @autoreleasepool {
        e->lat_n = 0; e->lat_dst = NULL;
        if (e->PLE) { set_err(err, err_len, "%s", "latent rows: not for a model with per-layer token embeddings"); return -1; }
        if (n < 0 || n > e->rmax) { set_err(err, err_len, "%s", "latent rows: too many"); return -1; }
        const int E = e->E, nmax = e->slots + 4;
        if (!e->lat_out) {
            e->lat_out = [dev newBufferWithLength:(size_t) nmax * E * 4 options:MTLResourceStorageModeShared];
            e->lat_iota = [dev newBufferWithLength:(size_t) nmax * 4 options:MTLResourceStorageModeShared];
            for (int i = 0; i < nmax; ++i) ((int32_t *) e->lat_iota.contents)[i] = i;
        }
        if (n > e->lat_cap) {
            e->lat_in = [dev newBufferWithLength:(size_t) n * E * 4 options:MTLResourceStorageModeShared];
            e->lat_idx = [dev newBufferWithLength:(size_t) n * 4 options:MTLResourceStorageModeShared];
            e->lat_cap = n;
        }
        if (n > 0) {
            memcpy(e->lat_in.contents, vec, (size_t) n * E * 4);
            memcpy(e->lat_idx.contents, rows, (size_t) n * 4);
        }
        e->lat_n = n;
        e->lat_dst = hid;
        return 0;
    }
}

// The input rows of n token ids (n <= C), as the step's embedding stage makes them, into ob (n x E floats).
static void embed_ids(sn_engine * e, int n, const int32_t * ids, id<MTLBuffer> tb, id<MTLBuffer> ob, id<MTLBuffer> sb) {
    for (int i = 0; i < n; ++i) ((int32_t *) tb.contents)[i] = ids[i];
    if (sb) for (int i = 0; i < n; ++i) memcpy((char *) sb.contents + (size_t) i * e->erow, e->etab + (size_t) ids[i] * e->erow, e->erow);
    id<MTLCommandBuffer> cb = [e->queue commandBuffer];
    id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
    embed_rows(e, c, tb, sb, ob, n);  // as encode's embedding stage
    [c endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

int sn_embed_rows(sn_engine * e, int n, const int32_t * ids, float * dst, char * err, int err_len) {
    @autoreleasepool {
        if (n < 0 || n > 1024) { set_err(err, err_len, "%s", "sn_embed_rows: 0..1024 ids"); return -1; }
        for (int i = 0; i < n; ++i) if (ids[i] < 0 || ids[i] >= e->V) { set_err(err, err_len, "%s", "sn_embed_rows: token out of range"); return -1; }
        if (n == 0) return 0;
        use(e);
        settle(e);
        id<MTLBuffer> tb = [dev newBufferWithLength:(size_t) n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [dev newBufferWithLength:(size_t) n * e->E * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> sb = e->emb_cpu ? [dev newBufferWithLength:(size_t) n * e->erow options:MTLResourceStorageModeShared] : nil;
        embed_ids(e, n, ids, tb, ob, sb);
        memcpy(dst, ob.contents, (size_t) n * e->E * 4);
        return 0;
    }
}

float sn_embd_norm(sn_engine * e) {
    @autoreleasepool {
        if (e->emb_norm > 0) return e->emb_norm;
        use(e);
        settle(e);
        const int E = e->E, C = 1024;
        id<MTLBuffer> tb = [dev newBufferWithLength:(size_t) C * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [dev newBufferWithLength:(size_t) C * E * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> sb = e->emb_cpu ? [dev newBufferWithLength:(size_t) C * e->erow options:MTLResourceStorageModeShared] : nil;
        double sum = 0;
        int32_t ids[1024];
        for (int t0 = 0; t0 < e->V; t0 += C) {
            const int n = MIN(C, e->V - t0);
            for (int i = 0; i < n; ++i) ids[i] = t0 + i;
            embed_ids(e, n, ids, tb, ob, sb);
            const float * o = ob.contents;
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int k = 0; k < E; ++k) s += (double) o[(size_t) i * E + k] * o[(size_t) i * E + k];
                sum += sqrt(s);
            }
        }
        e->emb_norm = (float) (sum / e->V);
        return e->emb_norm;
    }
}
