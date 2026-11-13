// sllm-next engine library, a part of sn.m (one translation unit; sn.m includes its parts in order): scores and probabilities; the cache's layers and pages; guesses (sn_guess) and draft rows

// The scores of n output rows ks of the last step into dst (n x V floats), in one copy.
int sn_logits_rows(sn_engine * e, int n, const int * ks, float * dst, char * err, int err_len) {
    @autoreleasepool {
        if (n <= 0) return 0;
        for (int i = 0; i < n; ++i)
            if (ks[i] < 0 || (size_t) (e->lastofs + ks[i] + 1) * e->V * 4 > e->logits.length) { set_err(err, err_len, "%s", "output row out of range"); return -1; }
        if (e->ch_set >= 0 && e->lofs[e->ch_set] == e->lastofs) { set_err(err, err_len, "%s", "a queued step of more than 4 rows took these scores' rows"); return -1; }
        id<MTLBuffer> rows = [dev newBufferWithLength:(size_t) n * e->V * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [e->queue commandBuffer];
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        for (int i = 0; i < n; ++i)
            [be copyFromBuffer:e->logits sourceOffset:(size_t) (e->lastofs + ks[i]) * e->V * 4 toBuffer:rows destinationOffset:(size_t) i * e->V * 4 size:(size_t) e->V * 4];
        [be endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(dst, rows.contents, (size_t) n * e->V * 4);
        return 0;
    }
}

// Token probabilities (see sn.h): steps encoded while it is on take argmax_p in place of argmax
void sn_want_probs(sn_engine * e, int on) { e->want_probs = on != 0; }

// The records of the last returned step's first n_out output rows (8 words each) into dst
int sn_probs(sn_engine * e, int n_out, float * dst, char * err, int err_len) {
    if (n_out < 0 || n_out > e->last_pn) { set_err(err, err_len, "%s", "the last step has no token probabilities for these rows"); return -1; }
    memcpy(dst, e->last_probs, (size_t) n_out * 32);
    return 0;
}

int sn_logits(sn_engine * e, int k, float * dst, char * err, int err_len) {
    @autoreleasepool {
        if (k < 0 || (size_t) (e->lastofs + k + 1) * e->V * 4 > e->logits.length) { set_err(err, err_len, "%s", "output row out of range"); return -1; }
        if (e->ch_set >= 0 && e->lofs[e->ch_set] == e->lastofs) { set_err(err, err_len, "%s", "a queued step of more than 4 rows took these scores' rows"); return -1; }
        id<MTLBuffer> row = [dev newBufferWithLength:(size_t) e->V * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [e->queue commandBuffer];
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        [be copyFromBuffer:e->logits sourceOffset:(size_t) (e->lastofs + k) * e->V * 4 toBuffer:row destinationOffset:0 size:(size_t) e->V * 4];
        [be endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(dst, row.contents, (size_t) e->V * 4);
        return 0;
    }
}

// kv_layers' lists: each layer's byte offset and cell bytes, its K and V buffers (kb NULL: not wanted)
typedef struct { size_t * off, * cell; __unsafe_unretained id<MTLBuffer> * kb, * vb; } kv_list;

// whether layer i keeps cells in pool w
static bool kv_layer_in(const sn_engine * e, int i, int w) {
    if (e->gemma && (!e->gl[i].has_kv || e->gl[i].pool != w)) return false;
    if (!e->gemma && w) return false;
    return !(e->q35 && e->ql[i].lin);  // Qwen3.5's linear layers keep states, not cells
}

// entry n: cache layer li of pool 0's kc / vc (Qwen3 / Llama's layers, Qwen3.5's attention layers and guess head)
static void kv_dense_at(const sn_engine * e, int li, int n, const kv_list * k) {
    k->off[n] = e->kv_layer * li * 2;
    k->cell[n] = (size_t) e->KD * 2;
    if (k->kb) { k->kb[n] = e->pl[0].kc; k->vb[n] = e->pl[0].vc; }
}

// entry n: Gemma / gpt-oss layer l's region of its pool
static void kv_gemma_at(const sn_engine * e, const g4layer_t * l, int n, const kv_list * k) {
    k->off[n] = LKO(e, l);
    k->cell[n] = (size_t) (l->nkv * l->D) * 2;
    if (k->kb) { k->kb[n] = LKB(e, l); k->vb[n] = LVB(e, l); }
}

// Pool w's cache layers: byte offset of the layer in the pool's kc / vc and bytes of one cell. Returns the count.
static int kv_layers(const sn_engine * e, int w, size_t * off, size_t * cell, __unsafe_unretained id<MTLBuffer> * kb, __unsafe_unretained id<MTLBuffer> * vb) {
    const kv_list k = {off, cell, kb, vb};
    int n = 0;
    for (int i = 0; i < e->L; ++i) {
        if (!kv_layer_in(e, i, w)) continue;
        if (e->gemma) kv_gemma_at(e, &e->gl[i], n, &k);
        else kv_dense_at(e, e->q35 ? e->ql[i].li : i, n, &k);
        ++n;
    }
    if (e->q35 && e->mtp && !w) kv_dense_at(e, e->mtp->li, n++, &k);  // the guess head's cache layer
    return n;
}

static bool slot_ok(const sn_engine * e, int slot, int a, int b) { return slot >= 0 && slot < e->slots && a >= 0 && a <= b && b <= e->ctx; }

// One blit command buffer over pool w's layers; f(encoder, K buffer, V buffer, layer offset, cell bytes, layer index).
static int kv_blit(sn_engine * e, int w, void (^f)(id<MTLBlitCommandEncoder>, id<MTLBuffer>, id<MTLBuffer>, size_t, size_t, int), char * err, int err_len) {
    size_t off[256], cell[256];
    __unsafe_unretained id<MTLBuffer> kb[256], vb[256];
    const int n = kv_layers(e, w, off, cell, kb, vb);
    id<MTLCommandBuffer> cb = [e->queue commandBuffer];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    for (int i = 0; i < n; ++i) f(be, kb[i], vb[i], off[i], cell[i], i);
    [be endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { set_err(err, err_len, "cache copy failed: %s", cb.error.localizedDescription.UTF8String); return -1; }
    return 0;
}

// ---- the cache's pages ---------------------------------------------------------------------------------------------
// Pages of 64 cells (every layer of a pool at the same page index); a pool's pt[seq * maxc + chunk] is the page of the
// sequence's cells chunk * 64 .., -1 none. ref: the sequences that have the page in their table; a page nobody holds is
// zero and on the free list, so a sequence's cells past its end always read as zero. The window pool (1) holds a
// sequence's chunks from win_first(its length) on: what its next row's window reaches, and one chunk more.
static int * seq_pages(const sn_engine * e, int w, int seq) { return (int *) e->pl[w].pt.contents + (size_t) seq * e->maxc; }

// the first chunk the window pool keeps for a sequence of n cells (its next row sees cells n - win + 1 ..; one chunk spare,
// so a queued step's release never takes a chunk the running step reads, and short cuts back keep their window)
static int win_first(const sn_engine * e, int n) { return MAX(0, (n - e->win) / 64 - 1); }
// the first window chunk a sequence of n cells keeps: what it needs and what a cut back by up to wback cells needs
static int win_keep(const sn_engine * e, int n) { return win_first(e, MAX(0, n - e->wback)); }

// pages [n0, n1) of a pool free (zero, on the free list), the pool n1 pages
static void pool_add(pool_t * pool, int n0, int n1) {
    for (int p = n1 - 1; p >= n0; --p) { pool->ref[p] = 0; pool->freel[pool->nfree++] = p; }
    pool->n = n1;
}

// sparse: the next chunk mapped in, nothing copied
static int pool_grow_sparse(sn_engine * e) {
    pool_t * pool = &e->pl[1];
    const int n0 = pool->n, n1 = MIN(e->wcap, n0 + e->wgrow);
    if (n1 <= n0 || pool_map(e, 1, n0, n1, true)) return -1;
    pool_add(pool, n0, n1);
    if (getenv("SN_SPARSE_LOG")) fprintf(stderr, "sn: window memory %d pages (%.2f GB)\n", n1, (double) n1 * e->wunits * 256 / 1e9);
    return 0;
}

// window layer l's buffers from n0 to n1 pages: a new zeroed pair with its pages copied in (waits for the GPU); 0 (also
// when a try that failed further on grew it already), or -1 (the layers before keep their larger buffers: still valid for
// n0 pages)
static int pool_grow_layer(sn_engine * e, const g4layer_t * l, int n0, int n1) {
    pool_t * pool = &e->pl[1];
    @autoreleasepool {
        const size_t len0 = (size_t) l->nkv * l->D * n0 * 128, len1 = (size_t) l->nkv * l->D * n1 * 128;
        if (pool->lk[l->pi].length >= len1) return 0;  // grown by a try that failed further on
        id<MTLBuffer> k1 = [dev newBufferWithLength:len1 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> v1 = [dev newBufferWithLength:len1 options:MTLResourceStorageModePrivate];
        if (!k1 || !v1) return -1;  // the layers before keep their larger buffers: still valid for n0 pages
        id<MTLCommandBuffer> cb = [e->queue commandBuffer];
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        [be fillBuffer:k1 range:NSMakeRange(len0, len1 - len0) value:0];
        [be fillBuffer:v1 range:NSMakeRange(len0, len1 - len0) value:0];
        [be copyFromBuffer:pool->lk[l->pi] sourceOffset:0 toBuffer:k1 destinationOffset:0 size:len0];
        [be copyFromBuffer:pool->lv[l->pi] sourceOffset:0 toBuffer:v1 destinationOffset:0 size:len0];
        [be endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) return -1;
        pool->lk[l->pi] = k1; pool->lv[l->pi] = v1;
    }
    return 0;
}

// The window pool grows (to pool 0's size at most) by a quarter: a layer at a time, a new zeroed buffer with the layer's
// pages copied in, so the copy needs one layer's buffers twice, never the whole pool's (Gemma 3 27B with 32 long
// sequences: 9 GB held while 18 GB were made did not fit in 48 GB)
static int pool_grow(sn_engine * e) {
    pool_t * P = &e->pl[1];
    if (e->wsp) return pool_grow_sparse(e);  // sparse: the next chunk mapped in, nothing copied
    const int n0 = P->n, n1 = MIN(e->wcap, MAX(n0 + n0 / 4, n0 + 8));
    if (n1 <= n0) return -1;
    for (int s = 0; s < 2; ++s) if (e->cbs[s]) [e->cbs[s] waitUntilCompleted];
    for (int i = 0, j = 0; i < e->L; ++i) {
        const g4layer_t * l = &e->gl[i];
        if (!l->has_kv || !l->pool || l->pi != j) continue;
        ++j;
        if (pool_grow_layer(e, l, n0, n1)) return -1;
    }
    pool_add(P, n0, n1);
    return 0;
}

// a free page of pool w, the lowest (the pages in use stay packed at the pool's start, so the chunks above them stay
// unmapped, and sn_trim has little to move); -1 when there is none
static int page_new(sn_engine * e, int w) {
    pool_t * P = &e->pl[w];
    if (!P->nfree && (w == 0 || pool_grow(e) != 0)) return -1;
    int k = P->nfree - 1;
    for (int i = 0; i < P->nfree - 1; ++i) if (P->freel[i] < P->freel[k]) k = i;
    const int p = P->freel[k];
    if (w == 0 && e->sparse && p >= e->kc_pages && kv_commit(e, p + 1) != 0) return -1;  // memory for it first
    P->freel[k] = P->freel[--P->nfree];
    P->ref[p] = 1;
    return p;
}

// one reference less to each of n pages of pool w; the ones nobody holds any more are zeroed and free
static void pages_drop(sn_engine * e, int w, const int * ps, int n) {
    pool_t * P = &e->pl[w];
    int * z = malloc((size_t) MAX(n, 1) * 4), nz = 0;
    for (int i = 0; i < n; ++i) if (ps[i] >= 0 && --P->ref[ps[i]] == 0) z[nz++] = ps[i];
    if (nz) kv_blit(e, w, ^(id<MTLBlitCommandEncoder> be, id<MTLBuffer> kb, id<MTLBuffer> vb, size_t off, size_t cell, int li) {
        for (int j = 0; j < nz; ++j)
            for (id<MTLBuffer> b in @[kb, vb]) [be fillBuffer:b range:NSMakeRange(off + (size_t) z[j] * 64 * cell, 64 * cell) value:0];
    }, NULL, 0);
    for (int j = 0; j < nz; ++j) P->freel[P->nfree++] = z[j];
    free(z);
}

// the window pool's chunks [a, b) of a sequence given back
static void win_drop(sn_engine * e, int seq, int a, int b) {
    int * t = seq_pages(e, 1, seq);
    if (b > a) pages_drop(e, 1, t + a, b - a);
    for (int c = a; c < b; ++c) t[c] = -1;
}

// a page of pool w for every row's cell that has none; -1 when the pool is out of pages
static int rows_pages(sn_engine * e, int w, int R, const int32_t * seq, const int32_t * pos) {
    for (int r = 0; r < R; ++r) {
        int * t = seq_pages(e, w, seq[r]) + pos[r] / 64;
        if (*t < 0 && (*t = page_new(e, w)) < 0) return -1;
    }
    return 0;
}

// the window pool's chunks each sequence's first row of the step no longer sees, given back
static void win_release(sn_engine * e, int R, const int32_t * seq, const int32_t * pos) {
    for (int r = 0; r < R; ++r) e->wmin[seq[r]] = INT_MAX;
    for (int r = 0; r < R; ++r) e->wmin[seq[r]] = MIN(e->wmin[seq[r]], pos[r]);
    for (int r = 0; r < R; ++r) {
        const int s = seq[r];
        if (e->wmin[s] == INT_MAX) continue;  // done
        const int lo = win_keep(e, e->wmin[s] + 1);
        if (lo > e->wlo[s]) { win_drop(e, s, e->wlo[s], lo); e->wlo[s] = lo; }
        e->wmin[s] = INT_MAX;
    }
}

// pages for every row's cell (a write allocates the chunk it enters); the window pool first gives back what each
// sequence's first row of the step no longer sees. -1 when a pool is out of pages.
static int pages_for(sn_engine * e, int R, const int32_t * seq, const int32_t * pos) {
    if (rows_pages(e, 0, R, seq, pos)) return -1;
    if (e->npool < 2) return 0;
    win_release(e, R, seq, pos);
    return rows_pages(e, 1, R, seq, pos);
}

// cells [a, b) of page src into page dst, every layer of pool w
static int page_copy(sn_engine * e, int w, int src, int dst, int a, int b, char * err, int err_len) {
    return kv_blit(e, w, ^(id<MTLBlitCommandEncoder> be, id<MTLBuffer> kb, id<MTLBuffer> vb, size_t off, size_t cell, int li) {
        for (id<MTLBuffer> buf in @[kb, vb])
            [be copyFromBuffer:buf sourceOffset:off + ((size_t) src * 64 + a) * cell toBuffer:buf destinationOffset:off + ((size_t) dst * 64 + a) * cell size:(size_t) (b - a) * cell];
    }, err, err_len);
}

// the cells (seq[r], pos[r]) of n rows, zeroed in both pools (a queued step that is not wanted wrote them)
static void clear_cells(sn_engine * e, int n, const int * seq, const int * pos) {
    for (int w = 0; w < e->npool; ++w) {
        int pgs[MAX(n, 1)], * pg = pgs;  // a block takes the pointer, not the array (the blit is synchronous)
        for (int r = 0; r < n; ++r) pg[r] = seq_pages(e, w, seq[r])[pos[r] / 64];
        kv_blit(e, w, ^(id<MTLBlitCommandEncoder> be, id<MTLBuffer> kb, id<MTLBuffer> vb, size_t off, size_t cell, int i) {
            for (int r = 0; r < n; ++r)
                if (pg[r] >= 0)
                    for (id<MTLBuffer> b in @[kb, vb]) [be fillBuffer:b range:NSMakeRange(off + ((size_t) pg[r] * 64 + pos[r] % 64) * cell, cell) value:0];
        }, NULL, 0);
    }
}

// whether the window pool holds what a sequence of n cells needs: chunks win_first(n) .. (n - 1) / 64
static bool win_holds(const sn_engine * e, int slot, int n) {
    if (e->npool < 2 || n == 0) return true;
    const int * t = seq_pages(e, 1, slot);
    for (int c = win_first(e, n); c <= (n - 1) / 64; ++c) if (t[c] < 0) return false;
    return true;
}

// Qwen3.5: n cells end at a draft row of the sequence's last step (its spare entries hold the state after it)
static bool q35_draft_cut(const sn_engine * e, int slot, int n) { return e->dn[slot] > 0 && n > e->dpos[slot] && n <= e->dpos[slot] + e->dn[slot]; }
// Qwen3.5: n cells end where the sequence has a state: none, its end, or its checkpoint
static bool q35_state_cut(const sn_engine * e, int slot, int n) { return n == 0 || n == e->slen[slot] || (e->ckent[slot] >= 0 && n == e->ckpos[slot]); }

int sn_cut_ok(const sn_engine * e, int slot, int n) {
    if (!slot_ok(e, slot, 0, n)) return 0;
    if (e->q35 && q35_draft_cut(e, slot, n)) return 1;  // a draft row of its last step
    if (e->q35) return q35_state_cut(e, slot, n);  // a state: the end, or the checkpoint
    return win_holds(e, slot, n);
}

int sn_small_rows(const sn_engine * e) { return g_mvm && e->q40 ? g_small : 0; }  // (no Q4_0 layers: no few-row kernel, small steps are plain)
int sn_chat_exact(const sn_engine * e) { (void) e; return !g_chatbatch; }

int sn_recurrent(const sn_engine * e) { return e->q35; }
int sn_guess_head(const sn_engine * e) { return e->mtp != NULL; }

int sn_guess_on(sn_engine * e, int on) {
    if (!e->mtp) return 0;
    @autoreleasepool {
        settle(e);
        if (on) { mtp_alloc(e); sketch_build(e); }
        e->guess_on = on != 0;
    }
    return e->guess_on;
}

// sn_guess's slot, position, token or count out of range
static bool guess_args_bad(const sn_engine * e, int seq, int32_t tok, int pos, int k) {
    return seq < 0 || seq >= e->slots || pos < 0 || k < 1 || k > 4 || pos + k > e->ctx || tok < 0 || tok >= e->V;
}

// a guess's rows in set 0's records: the token, the sequence, the k positions (gpos), their cache pages; 0, or -2 with
// err set when out of pages
static int guess_stage(sn_engine * e, int seq, int32_t tok, int pos, int k, char * err, int err_len) {
    const int set = 0;
    rows_select(e, set);
    e->erow_b = e->erowB[set];
    ((int *) e->rtok.contents)[0] = tok; ((int *) e->rseq.contents)[0] = seq; ((int *) e->ridx.contents)[0] = 0;
    int32_t sqs[4], pss[4];
    for (int j = 0; j < k; ++j) { sqs[j] = seq; pss[j] = pos + j; ((int *) e->gpos[j].contents)[0] = pos + j; }
    if (pages_for(e, k, sqs, pss) != 0) { set_err(err, err_len, "%s", "out of cache pages"); return -2; }
    return 0;
}

// the guess's step: its buffers, command buffer and one interactive row's kernels (core.h's g_*)
static void guess_begin(sn_engine * e, int pos, int k) {
    const int set = 0;
    fit(&e->apart, (size_t) e->NH * ((pos + k - 1) / 64 + 1) * (e->g.hd + 2) * 4);
    fit(&e->logits, (size_t) 5 * e->V * 4);
    g_cb = [e->queue commandBuffer];
    e->cbs[set] = g_cb;
    g_e = nil; g_np = 0;
    g_nAtt = 1; g_wide = 0; g_nsw = 0; g_small_step = 0; g_conc_step = 0; g_conc = g_conc_env; g_prof = 0;
}

// row j's token embedded (the caller's token, then the guess before it), from the table mapped for the GPU (mtab)
static void guess_embed(sn_engine * e, int j) {
    struct { int n; float scale; int staged; } na = {e->E, 1.0f, 0};
    enc("embed"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&na length:sizeof na atIndex:0];
    [c setBuffer:j ? e->otok : e->rtok offset:j ? (size_t) (j - 1) * 4 : 0 atIndex:1];
    [c setBuffer:e->mtab offset:e->mtab_off atIndex:2];
    [c setBuffer:e->mem offset:0 atIndex:3];
    dispatch_rows(c, embed_name(e->embd_fmt), 1, 256);
}

// guess j's token from the head's hidden state (mhg): the output layer and argmax
static void guess_head_argmax(sn_engine * e, int j) {
    const int E = e->E;
    g_nA = 1;
    mm(g_e, e->head_fmt, e->mhg, 0, gb(e, e->head), go(e, e->head), nil, e->logits, 0, E, e->V, 1, E, e->V, 0);
    g_nA = 0;
    argmax_args va = {e->V, 0, {0}};
    uint8_t skip[16] = {0};
    enc("argmax"); id<MTLComputeCommandEncoder> c = g_e;
    [c setBytes:&va length:sizeof va atIndex:0];
    [c setBuffer:e->logits offset:0 atIndex:1]; [c setBuffer:e->otok offset:(size_t) j * 4 atIndex:2];
    [c setBytes:skip length:16 atIndex:3];
    dispatch_rows(c, @"argmax", 1, 1024);
}

// row j: its token (the caller's, then the guess before it) and the hidden state before it (the sequence's last one, then
// the guess before's), through the guess head's block, to the next guess
static void guess_row(sn_engine * e, int j, int seq, int pos) {
    const int E = e->E;
    e->rpos = e->gpos[j];
    g_nA = 1;
    guess_embed(e, j);
    mtp_norm(e, e->mem, 0, e->mtp_enorm, e->mcat, 0, 2 * E, 1);
    mtp_norm(e, j ? e->mhg : e->mpend, j ? 0 : (size_t) seq * E * 4, e->mtp_hnorm, e->mcat, (size_t) E * 4, 2 * E, 1);
    mm(g_e, e->mtp_feh, e->mcat, 0, gb(e, e->mtp_eh), go(e, e->mtp_eh), nil, e->x, 0, 2 * E, E, 1, 2 * E, E, 0);
    q35_layers(e, e->mtp, 1, 1, 1, 0, (pos + j) / 64 + 1);
    mtp_norm(e, e->x, 0, e->mtp_hdnorm, e->mhg, 0, E, 1);
    if (e->sk_bits) { sketch_argmax(e, e->mhg, e->otok, (size_t) j * 4); return; }
    guess_head_argmax(e, j);
}

int sn_guess(sn_engine * e, int seq, int32_t tok, int pos, int k, int32_t * out_tok, char * err, int err_len) {
    @autoreleasepool {
        if (!e->mtp || !e->guess_on || !e->mtab) { set_err(err, err_len, "%s", "sn_guess: no guess head, or guessing is off (sn_guess_on)"); return -1; }
        if (guess_args_bad(e, seq, tok, pos, k)) { set_err(err, err_len, "%s", "sn_guess: slot, position, token or count out of range"); return -1; }
        use(e);
        settle(e);
        const int set = 0;
        if (guess_stage(e, seq, tok, pos, k, err, err_len)) return -2;
        guess_begin(e, pos, k);
        for (int j = 0; j < k; ++j) guess_row(e, j, seq, pos);  // row j: its token (the caller's, then the guess before it) and the hidden state before it
        e->rpos = e->rposB[set];
        [g_e endEncoding];
        g_e = nil;
        [g_cb commit];
        e->lofs[set] = 0; e->pwant[set] = 0;
        return finish(e, set, k, out_tok, err, err_len);
    }
}

// spare entries d[have] .. d[want - 1] taken from the pool (grown when it can be); the count held then
static int q35_spares_add(sn_engine * e, int * d, int have, int want) {
    while (have < want) {
        if (!e->nsfree && q35_grow(e) != 0) break;
        d[have++] = sfree_take(e);
    }
    return have;
}

int sn_draft_rows(sn_engine * e, int seq, int want) {
    if (!e->q35) return want;
    if (seq < 0 || seq >= e->slots) return 0;
    int * d = e->dent + seq * 3, have = 0;
    while (have < 3 && d[have] >= 0) ++have;
    want = MIN(MAX(want, 0), 3);
    if (have >= want && want > 0) return want;
    @autoreleasepool {
        settle(e);
        if (want == 0) { q35_spare_drop(e, seq); return 0; }
        have = q35_spares_add(e, d, have, want);
    }
    return have;
}

int sn_set_eog(sn_engine * e, const int32_t * ids, int n) {
    if (n < 0 || n > 8) return -1;
    e->neog = n;
    memcpy(e->eog, ids, (size_t) n * 4);
    return 0;
}

void sn_no_end(sn_engine * e, int slot, int on) {
    if (slot >= 0 && slot < e->slots) e->no_end[slot] = on != 0;
}

int sn_mark(sn_engine * e, int slot) {
    @autoreleasepool {
        settle(e);
        if (!e->q35 || slot < 0 || slot >= e->slots || e->sidx[slot] < 0) return 0;
        if (e->ckent[slot] < 0) {  // a free entry only: the pool grows for one only when it is sparse (no copy, a chunk of memory)
            if (!e->nsfree && !(e->q35sp && q35_grow(e) == 0)) return 0;
            e->ckent[slot] = sfree_take(e);
        }
        q35_copy(e, e->sidx[slot], e->ckent[slot]);
        e->ckpos[slot] = e->slen[slot];
        return 1;
    }
}

// Qwen3.5: the longest cut of at most n cells the sequence has a state for (its end, its checkpoint, a draft row)
static int q35_cut_best(const sn_engine * e, int slot, int n) {
    int best = 0;
    if (e->slen[slot] <= n) best = e->slen[slot];
    if (e->ckent[slot] >= 0 && e->ckpos[slot] <= n) best = MAX(best, e->ckpos[slot]);
    if (e->dn[slot] > 0 && n > e->dpos[slot]) best = MAX(best, MIN(n, e->dpos[slot] + e->dn[slot]));
    return best;
}

int sn_cut_best(const sn_engine * e, int slot, int n) {
    if (slot < 0 || slot >= e->slots || n <= 0) return 0;
    if (!e->q35) return sn_cut_ok(e, slot, n) ? n : 0;
    return q35_cut_best(e, slot, n);
}
