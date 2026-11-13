// sllm-next engine library, a part of sn.m (one translation unit; sn.m includes its parts in order): Qwen3.5's recurrent state pool, and the guess head (MTP): its cache cells and the output layer's sketch

// ---- Qwen3.5's recurrent states ----------------------------------------------------------------------------------------
static size_t q35_cbytes(const sn_engine * e) { return (size_t) e->nlin * 3 * e->q35C * 4; }                   // an entry's conv states
static size_t q35_sbytes(const sn_engine * e) { return (size_t) e->nlin * e->g.ssm_rank * 128 * 128 * 4; }     // its delta states

static void q35_reset(sn_engine * e) {
    for (int s = 0; s < e->slots; ++s) { e->sidx[s] = -1; e->slen[s] = 0; e->ckent[s] = -1; e->ckpos[s] = 0; }
    for (int s = 0; s < e->slots; ++s) { e->dn[s] = 0; for (int j = 0; j < 3; ++j) e->dent[s * 3 + j] = -1; }
    e->nsfree = 0;
    for (int i = e->nst - 1; i >= 0; --i) e->sfree[e->nsfree++] = i;
}

// zero entries [a, b) of the pool
static void q35_zero(sn_engine * e, int a, int b) {
    if (b <= a) return;
    id<MTLCommandBuffer> cb = [e->queue commandBuffer];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    [be fillBuffer:e->cst range:NSMakeRange((size_t) a * q35_cbytes(e), (size_t) (b - a) * q35_cbytes(e)) value:0];
    [be fillBuffer:e->sst range:NSMakeRange((size_t) a * q35_sbytes(e), (size_t) (b - a) * q35_sbytes(e)) value:0];
    [be endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// entries [n0, n1) on the free list (the lowest handed out first), the pool n1 entries
static void q35_pool_add(sn_engine * e, int n0, int n1) {
    for (int i = n1 - 1; i >= n0; --i) e->sfree[e->nsfree++] = i;
    e->nst = n1;
}

// the sparse pool: the next entries mapped in, nothing copied, nothing held twice
static int q35_grow_sparse(sn_engine * e) {
    const int n0 = e->nst, n1 = MIN((e->slots + e->stgrow - 1) / e->stgrow * e->stgrow, n0 + e->stgrow);
    if (n1 <= n0) return -1;
    if (q35_map(e, n0, n1, true)) return -1;
    q35_pool_add(e, n0, n1);
    if (getenv("SN_SPARSE_LOG")) fprintf(stderr, "sn: state memory %d entries (%.2f GB)\n", n1, (double) n1 * (q35_cbytes(e) + q35_sbytes(e)) / 1e9);
    return 0;
}

// new buffers of n1 entries, the n0 old ones copied to their start and the rest zero (waits for the GPU): 0, or -1
static int q35_grow_copy(sn_engine * e, int n0, int n1) {
    id<MTLBuffer> c1 = [dev newBufferWithLength:(size_t) n1 * q35_cbytes(e) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> s1 = [dev newBufferWithLength:(size_t) n1 * q35_sbytes(e) options:MTLResourceStorageModePrivate];
    if (!c1 || !s1) return -1;
    id<MTLCommandBuffer> cb = [e->queue commandBuffer];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    [be copyFromBuffer:e->cst sourceOffset:0 toBuffer:c1 destinationOffset:0 size:(size_t) n0 * q35_cbytes(e)];
    [be copyFromBuffer:e->sst sourceOffset:0 toBuffer:s1 destinationOffset:0 size:(size_t) n0 * q35_sbytes(e)];
    [be fillBuffer:c1 range:NSMakeRange((size_t) n0 * q35_cbytes(e), (size_t) (n1 - n0) * q35_cbytes(e)) value:0];
    [be fillBuffer:s1 range:NSMakeRange((size_t) n0 * q35_sbytes(e), (size_t) (n1 - n0) * q35_sbytes(e)) value:0];
    [be endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) return -1;
    e->cst = c1; e->sst = s1;
    return 0;
}

// the pool grows (to one entry a sequence at most): new zeroed buffers, the old entries copied to their start
static int q35_grow(sn_engine * e) {
    if (e->q35sp) return q35_grow_sparse(e);  // map the next entries in: nothing copied, nothing held twice
    const int n0 = e->nst, n1 = MIN(e->slots, MAX(2 * n0, n0 + 2));
    if (n1 <= n0) return -1;
    for (int s = 0; s < 2; ++s) if (e->cbs[s]) [e->cbs[s] waitUntilCompleted];
    if (q35_grow_copy(e, n0, n1)) return -1;
    q35_pool_add(e, n0, n1);
    return 0;
}

// the lowest free state entry taken (nsfree > 0): the entries in use stay packed at the pool's start for sn_trim
static int sfree_take(sn_engine * e) {
    int k = e->nsfree - 1;
    for (int i = 0; i < e->nsfree - 1; ++i) if (e->sfree[i] < e->sfree[k]) k = i;
    const int i = e->sfree[k];
    e->sfree[k] = e->sfree[--e->nsfree];
    return i;
}

// the entry *ent holds (if any) back to the pool, zeroed; *ent none
static void q35_entry_free(sn_engine * e, int * ent) {
    const int i = *ent;
    if (i < 0) return;
    q35_zero(e, i, i + 1);
    e->sfree[e->nsfree++] = i;
    *ent = -1;
}

// a sequence's state entry, a zero one taken (and the pool grown) when it has none; -1: none to take
static void q35_ck_drop(sn_engine * e, int seq);
static int q35_entry_keep(sn_engine * e, int seq, int keep) {
    if (e->sidx[seq] >= 0) return e->sidx[seq];
    // checkpoints give way before the pool grows (52 MB a state on Qwen3.5 4B); `keep`'s is the one being copied from
    for (int s = 0; s < e->slots && !e->nsfree; ++s) if (e->ckent[s] >= 0 && s != keep) q35_ck_drop(e, s);
    if (!e->nsfree && q35_grow(e) != 0) return -1;
    return e->sidx[seq] = sfree_take(e);
}

static int q35_entry(sn_engine * e, int seq) { return q35_entry_keep(e, seq, -1); }

// a sequence's checkpoint back to the pool, zeroed
static void q35_ck_drop(sn_engine * e, int seq) { q35_entry_free(e, &e->ckent[seq]); }

// entry src's states into entry dst
static void q35_copy(sn_engine * e, int src, int dst) {
    id<MTLCommandBuffer> cb = [e->queue commandBuffer];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    [be copyFromBuffer:e->cst sourceOffset:(size_t) src * q35_cbytes(e) toBuffer:e->cst destinationOffset:(size_t) dst * q35_cbytes(e) size:q35_cbytes(e)];
    [be copyFromBuffer:e->sst sourceOffset:(size_t) src * q35_sbytes(e) toBuffer:e->sst destinationOffset:(size_t) dst * q35_sbytes(e) size:q35_sbytes(e)];
    [be endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// a sequence's spare entries (draft rows) back to the pool, zeroed
static void q35_spare_drop(sn_engine * e, int seq) {
    e->dn[seq] = 0;
    for (int j = 0; j < 3; ++j) q35_entry_free(e, &e->dent[seq * 3 + j]);
}

static void q35_release(sn_engine * e, int seq) {
    q35_ck_drop(e, seq);
    q35_spare_drop(e, seq);
    if (e->mpend) {  // (a new sequence's first row has no hidden state before it)
        id<MTLCommandBuffer> cb = [e->queue commandBuffer];
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        [be fillBuffer:e->mpend range:NSMakeRange((size_t) seq * e->E * 4, (size_t) e->E * 4) value:0];
        [be endEncoding];
        [cb commit];
    }
    q35_entry_free(e, &e->sidx[seq]);
}

// A recurrent model's state entry for `seq`, taken now (at admission) so a step never runs out of them: 0, or -1 when none
// is free and the pool cannot grow (the caller waits for a sequence to end). Other models: always 0.
int sn_state_take(sn_engine * e, int seq) { return !e->q35 || q35_entry(e, seq) >= 0 ? 0 : -1; }

// ---- the guess head (MTP) ------------------------------------------------------------------------------------------------
static void mtp_alloc(sn_engine * e) {
    if (e->mhn) return;
    const size_t RE = (size_t) e->rmax * e->E * 4;
    const MTLResourceOptions PRIV = MTLResourceStorageModePrivate;
    e->mhn = [dev newBufferWithLength:RE options:PRIV]; e->mhs = [dev newBufferWithLength:RE options:PRIV];
    e->mem = [dev newBufferWithLength:RE options:PRIV]; e->mx = [dev newBufferWithLength:RE options:PRIV];
    e->mcat = [dev newBufferWithLength:2 * RE options:PRIV];
    e->mpend = [dev newBufferWithLength:(size_t) e->slots * e->E * 4 options:PRIV];
    e->mhg = [dev newBufferWithLength:(size_t) e->E * 4 options:PRIV];
    e->msrc = [dev newBufferWithLength:(size_t) e->rmax * 4 options:MTLResourceStorageModeShared];
    e->drow = calloc(e->slots, sizeof(int));
    for (int j = 0; j < 4; ++j) e->gpos[j] = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    {
        const gguf_t * g = &e->g;
        const size_t page = getpagesize(), a0 = e->embd / page * page, a1 = MIN((e->embd + (size_t) e->V * e->erow + page - 1) / page * page, g->size);
        e->mtab = [dev newBufferWithBytesNoCopy:(void *) (g->base + a0) length:a1 - a0 options:MTLResourceStorageModeShared deallocator:nil];
        e->mtab_off = e->embd - a0;
    }
    id<MTLCommandBuffer> cb = [e->queue commandBuffer];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    [be fillBuffer:e->mpend range:NSMakeRange(0, e->mpend.length) value:0];
    [be fillBuffer:e->mhg range:NSMakeRange(0, e->mhg.length) value:0];
    [be endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// The sketch of a q6_K output layer for guesses: a bit a weight (its sign) and the row's mean |weight|, made on the CPU once
enum { SK_NC = 1024, SK_TOP = 4 };  // chunks of the vocabulary, candidates a chunk; the sketch covers the first half of the dims
// (measured on 1,164 hidden states: the best row is among its chunk's top 4 on 99.8% with half the dims, top 2 with all: 99.7%)
typedef struct __attribute__((packed)) { uint8_t ql[128], qh[64]; int8_t scales[16]; _Float16 d; } cpu_q6k;
// group gi (16 weights) of a q6_K block b (scale d): the bits of its positive weights; |w| of each added to *mag in order
static inline uint32_t sketch_bits16(const cpu_q6k * b, float d, int gi, double * mag) {
    const int j0 = gi * 16, n = j0 / 128, jj = j0 % 128, qu = jj / 32, l0 = jj % 32;
    const uint8_t * ql = b->ql + n * 64 + l0 + (qu & 1) * 32, * qh = b->qh + n * 32 + l0;
    const float sc = d * (float) b->scales[gi];
    uint32_t m = 0;
    for (int i = 0; i < 16; ++i) {
        const int lo = qu < 2 ? (ql[i] & 0xF) : (ql[i] >> 4);
        const float w = sc * (float) ((lo | (((qh[i] >> (2 * qu)) & 3) << 4)) - 32);
        if (w > 0) m |= 1u << i;
        *mag += w < 0 ? -w : w;
    }
    return m;
}

// a row of the output layer (E wide): the signs of its first nw * 32 weights into out, its mean |weight| into *rs
static void sketch_row(const cpu_q6k * row, int E, int nw, uint32_t * out, float * rs) {
    double mag = 0;
    for (int sb = 0; sb < E / 256; ++sb) {
        const cpu_q6k * b = &row[sb];
        const float d = (float) b->d;
        for (int gi = 0; gi < 16; ++gi) {
            const uint32_t m = sketch_bits16(b, d, gi, &mag);
            const int at = sb * 256 + gi * 16;  // the row's weights at .. at + 15: 16 bits of word at / 32
            if (at / 32 >= nw) continue;
            if (at % 32 == 0) out[at / 32] = m; else out[at / 32] |= m << 16;
        }
    }
    *rs = (float) (mag / E);
}

// a sketch is wanted: none yet, a q6_K output layer of whole blocks, SN_GUESS_SKETCH not 0
static bool sketch_wanted(const sn_engine * e) {
    return !(e->sk_bits || e->head_fmt != 2 || e->E % 256 || (getenv("SN_GUESS_SKETCH") && !atoi(getenv("SN_GUESS_SKETCH"))));
}

static void sketch_build(sn_engine * e) {
    if (!sketch_wanted(e)) return;
    const int V = e->V, E = e->E, nw = E / 64;
    const MTLResourceOptions SH = MTLResourceStorageModeShared;
    id<MTLBuffer> bits = [dev newBufferWithLength:(size_t) V * nw * 4 options:SH], rs = [dev newBufferWithLength:(size_t) V * 4 options:SH];
    if (!bits || !rs) return;
    uint32_t * B = bits.contents;
    float * RS = rs.contents;
    const cpu_q6k * W = (const cpu_q6k *) (e->g.base + e->head);
    dispatch_apply((size_t) V, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
        sketch_row(W + r * (E / 256), E, nw, B + r * nw, &RS[r]);
    });
    e->sk_bits = bits; e->sk_rs = rs;
    e->sk_tab = priv_buf((size_t) (E / 8) * 256 * 4);
    e->sk_cand = priv_buf((size_t) SK_NC * SK_TOP * 4);
    e->sk_sc = priv_buf((size_t) SK_NC * SK_TOP * 4);
}

// a guess's best token from hidden state h (E floats in buffer hb) into out (an int at byte oo), through the sketch
static void sketch_argmax(sn_engine * e, id<MTLBuffer> hb, id<MTLBuffer> out, size_t oo) {
    const int V = e->V, E = e->E;
    struct { int V, E, per, nc; } a = {V, E, (V + SK_NC - 1) / SK_NC, SK_NC};  // (sk_pick / sk_best: SK_TOP candidates a chunk, E / 64 words a row)
    id<MTLComputeCommandEncoder> c;
    static int skx = -1;  // SN_SKX (timing only, wrong guesses): 1 no sk_tab, 2 no sk_pick, 4 no sk_exact, 8 no sk_best
    if (skx < 0) skx = getenv("SN_SKX") ? atoi(getenv("SN_SKX")) : 0;
    c = kernel_begin("sketch", @"sk_tab", &a, sizeof a);
    [c setBuffer:hb offset:0 atIndex:1]; [c setBuffer:e->sk_tab offset:0 atIndex:2];
    if (!(skx & 1)) [c dispatchThreads:MTLSizeMake(256, E / 16, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    c = kernel_begin("sketch", @"sk_pick", &a, sizeof a);
    [c setBuffer:e->sk_bits offset:0 atIndex:1]; [c setBuffer:e->sk_rs offset:0 atIndex:2]; [c setBuffer:e->sk_tab offset:0 atIndex:3];
    [c setBuffer:e->sk_cand offset:0 atIndex:4];
    if (!(skx & 2)) [c dispatchThreadgroups:MTLSizeMake(SK_NC, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    c = kernel_begin("sketch", @"sk_exact", &a, sizeof a);
    [c setBuffer:gb(e, e->head) offset:go(e, e->head) atIndex:1]; [c setBuffer:hb offset:0 atIndex:2]; [c setBuffer:e->sk_cand offset:0 atIndex:3];
    [c setBuffer:e->sk_sc offset:0 atIndex:4];
    if (!(skx & 4)) [c dispatchThreadgroups:MTLSizeMake(SK_TOP * SK_NC, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    c = kernel_begin("sketch", @"sk_best", &a, sizeof a);
    [c setBuffer:e->sk_sc offset:0 atIndex:1]; [c setBuffer:e->sk_cand offset:0 atIndex:2]; [c setBuffer:out offset:oo atIndex:3];
    if (!(skx & 8)) [c dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

// the embedding stage over n rows into ob on encoder c: token ids in tb; staged (nil: none): the table's rows the CPU copied
// (row r of the step at r), else the table on the GPU
static void embed_rows(sn_engine * e, id<MTLComputeCommandEncoder> c, id<MTLBuffer> tb, id<MTLBuffer> staged, id<MTLBuffer> ob, int n) {
    const int E = e->E;
    if (e->gemma && e->head41) {  // the q4_1 copy of a Q4_K table, times sqrt(n_embd)
        g4_args ga = {E, sqrtf((float) E), E, e->PN, e->L, e->g.eps};
        [c setBytes:&ga length:sizeof ga atIndex:0];
        [c setBuffer:tb offset:0 atIndex:1]; [c setBuffer:e->head41 offset:0 atIndex:2]; [c setBuffer:ob offset:0 atIndex:3];
        dispatch_rows(c, @"g4_embed", n, 256);
        return;
    }
    // Gemma's embeddings times sqrt(n_embd) (not gpt-oss's)
    struct { int n; float scale; int staged; } na = {E, e->gemma && !e->oss ? sqrtf((float) E) : 1.0f, e->emb_cpu};
    [c setBytes:&na length:sizeof na atIndex:0];
    [c setBuffer:tb offset:0 atIndex:1];
    [c setBuffer:staged ? staged : gb(e, e->embd) offset:staged ? 0 : go(e, e->embd) atIndex:2];
    [c setBuffer:ob offset:0 atIndex:3];
    dispatch_rows(c, embed_name(e->embd_fmt), n, 256);
}

// the step's token rows as the embedding stage makes them, into y (the rows the CPU staged, or the table on the GPU)
static void embed_into(sn_engine * e, id<MTLBuffer> y, int R) {
    enc("embed");
    embed_rows(e, g_e, e->rtok, e->emb_cpu ? e->erow_b : nil, y, R);
}

// rmsnorm of R rows of x (E wide, from byte xo) with weight w into y (rows ldy apart, from byte yo)
static void mtp_norm(sn_engine * e, id<MTLBuffer> x, size_t xo, uint64_t w, id<MTLBuffer> y, size_t yo, int ldy, int R) {
    norm_args nrm = {e->E, e->g.eps, e->E, ldy};
    enc("rmsnorm"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&nrm length:sizeof nrm atIndex:0];
    [c setBuffer:x offset:xo atIndex:1]; [c setBuffer:gb(e, w) offset:go(e, w) atIndex:2]; [c setBuffer:y offset:yo atIndex:3];
    dispatch_rows(c, @"rmsnorm", R, 256);
}

// After the model's layers (x: the rows' last hidden states): the guess head's cache cells of the step's rows. Row r of
// position q is the head's row (its token, the model's hidden state after the output norm at q - 1): the row before it in
// the step, or what its sequence's last step left (mpend). Only K and V are needed: eh_proj, the block's first norm, its k
// and v, the cache write. x is left as it is (the output stage reads it); h, q, k, v are free by now.
static void mtp_fill(sn_engine * e, int R) {
    const gguf_t * g = &e->g;
    const q35l * l = e->mtp;
    const int E = e->E, NKV = g->n_kv, D = g->hd, KVS = NKV * D;
    const int * sq = e->rseq.contents, * ps = e->rpos.contents;
    int * src = e->msrc.contents;
    for (int r = 0; r < R; ++r) src[r] = r > 0 && sq[r] == sq[r - 1] && ps[r] == ps[r - 1] + 1 ? r - 1 : ps[r] == 0 ? -2 : -1;
    id<MTLComputeCommandEncoder> c;
    n_args na = {E};
    mtp_norm(e, e->x, 0, e->out_norm, e->mhn, 0, E, R);
    enc("mtp"); c = g_e;
    [c setBytes:&na length:sizeof na atIndex:0];
    [c setBuffer:e->mhn offset:0 atIndex:1]; [c setBuffer:e->mpend offset:0 atIndex:2]; [c setBuffer:e->msrc offset:0 atIndex:3];
    [c setBuffer:e->rseq offset:0 atIndex:4]; [c setBuffer:e->mhs offset:0 atIndex:5];
    dispatch_rows(c, @"mtp_hsel", R, 256);
    embed_into(e, e->mem, R);
    mtp_norm(e, e->mem, 0, e->mtp_enorm, e->mcat, 0, 2 * E, R);
    mtp_norm(e, e->mhs, 0, e->mtp_hnorm, e->mcat, (size_t) E * 4, 2 * E, R);
    mm(g_e, e->mtp_feh, e->mcat, 0, gb(e, e->mtp_eh), go(e, e->mtp_eh), nil, e->mx, 0, 2 * E, E, R, 2 * E, E, 0);
    mtp_norm(e, e->mx, 0, l->attn_norm, e->h, 0, E, R);
    mm(g_e, l->f_q, e->h, 0, gb(e, l->k), go(e, l->k), nil, e->k, 0, E, KVS, R, E, KVS, 0);
    mm(g_e, l->f_v, e->h, 0, gb(e, l->v), go(e, l->v), nil, e->v, 0, E, KVS, R, E, KVS, 0);
    q35_qk(e, l, R);  // (its q half reads what the model's last attention layer left in q: nothing uses that output)
    c = kernel_begin("mtp", @"mtp_keep", &na, sizeof na);  // each sequence's last hidden state of the step, for its next step's first row
    [c setBuffer:e->mhn offset:0 atIndex:1]; [c setBuffer:e->runs offset:0 atIndex:2]; [c setBuffer:e->rseq offset:0 atIndex:3]; [c setBuffer:e->mpend offset:0 atIndex:4];
    [c dispatchThreads:MTLSizeMake(E, e->nruns, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
