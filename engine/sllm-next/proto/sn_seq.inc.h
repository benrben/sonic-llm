// sllm-next engine library, a part of sn.m (one translation unit; sn.m includes its parts in order): a sequence's cache: copy, cut, save and load; packing and giving back memory (sn_trim); sn_close

// sequence `to` takes `from`'s first n cells in pool w (the window pool: only the chunks from win_first(n) on): the full
// pages shared (neither sequence writes below its end again), the last, partly filled one copied; 0, or -1 / -2 with err set
static int copy_pages(sn_engine * e, int w, int from, int to, int n, char * err, int err_len) {
    int * tf = seq_pages(e, w, from), * tt = seq_pages(e, w, to);
    pages_drop(e, w, tt, e->maxc);  // what `to` held
    for (int c = 0; c < e->maxc; ++c) tt[c] = -1;
    const int c0 = w ? win_first(e, n) : 0;
    for (int c = c0; c < n / 64; ++c) {
        if (tf[c] < 0) { set_err(err, err_len, "%s", "sn_copy: the source has no cells there"); return -1; }
        tt[c] = tf[c];
        ++e->pl[w].ref[tf[c]];
    }
    if (n % 64) {
        const int c = n / 64, np = page_new(e, w);
        if (np < 0) { set_err(err, err_len, "%s", "out of cache pages"); return -2; }
        tt[c] = np;
        if (tf[c] >= 0 && page_copy(e, w, tf[c], np, 0, n % 64, err, err_len) != 0) return -1;
    }
    return 0;
}

// Qwen3.5: the source's state (its end's, or at_ck its checkpoint's) into the copy's own entry; 0, or -2 with err set
static int q35_copy_state(sn_engine * e, int from, int to, int n, bool at_ck, char * err, int err_len) {
    q35_release(e, to);
    const int st = n > 0 ? q35_entry_keep(e, to, at_ck ? from : -1) : -1, sf = at_ck ? e->ckent[from] : e->sidx[from];
    if (n > 0 && (sf < 0 || st < 0)) { set_err(err, err_len, "%s", "out of recurrent states"); return -2; }
    if (n > 0) q35_copy(e, sf, st);
    e->slen[to] = n;
    return 0;
}

// Qwen3.5: n cells of `from` end where it has a state (at_ck: its checkpoint); true for the other models
static bool q35_copy_from(const sn_engine * e, int from, int n, bool * at_ck) {
    *at_ck = e->q35 && n != e->slen[from] && e->ckent[from] >= 0 && n == e->ckpos[from];  // the source's checkpoint
    return !(e->q35 && n != e->slen[from] && !*at_ck);
}

int sn_copy(sn_engine * e, int from, int to, int n, char * err, int err_len) {
    @autoreleasepool {
        settle(e);
        if (!slot_ok(e, from, 0, n) || !slot_ok(e, to, 0, n) || from == to) { set_err(err, err_len, "%s", "sn_copy: bad slots or cells"); return -1; }
        if (!win_holds(e, from, n)) { set_err(err, err_len, "%s", "sn_copy: the source's window layers no longer hold those cells (sn_cut_ok)"); return -3; }
        bool at_ck;
        if (!q35_copy_from(e, from, n, &at_ck)) { set_err(err, err_len, "%s", "sn_copy: a recurrent state is only at the source's end or checkpoint (sn_cut_ok)"); return -3; }
        for (int w = 0; w < e->npool; ++w) {
            const int rc = copy_pages(e, w, from, to, n, err, err_len);
            if (rc) return rc;
        }
        e->wlo[to] = win_first(e, n);
        return e->q35 ? q35_copy_state(e, from, to, n, at_ck, err, err_len) : 0;  // the source's states into the copy's own entry
    }
}

// Qwen3.5: back to the checkpoint (from > 0 its cell count, below the end)
static bool q35_ck_cut(const sn_engine * e, int slot, int from) {
    return from != 0 && from < e->slen[slot] && e->ckent[slot] >= 0 && from == e->ckpos[slot];
}

// back to a draft row of the last step: its spare entry holds the state after it (the entries swap; nothing is copied);
// guessing: the head's next row pairs with the hidden state of the last row kept
static void q35_to_draft(sn_engine * e, int slot, int from) {
    int * d = e->dent + slot * 3 + (from - e->dpos[slot] - 1);
    const int t = e->sidx[slot];
    e->sidx[slot] = *d; *d = t;
    e->slen[slot] = from;
    if (e->guess_on && e->mhn) {  // the guess head's next row pairs with the hidden state of the last row kept
        id<MTLCommandBuffer> cb = [e->queue commandBuffer];
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        const size_t EB = (size_t) e->E * 4;
        [be copyFromBuffer:e->mhn sourceOffset:(size_t) (e->drow[slot] + from - e->dpos[slot] - 1) * EB toBuffer:e->mpend destinationOffset:(size_t) slot * EB size:EB];
        [be endEncoding];
        [cb commit];
    }
}

// Qwen3.5's state for a cut back to `from` cells: none (0), the checkpoint, or a draft row of the last step; 0, or -3 with
// err set when the state cannot go back there
static int q35_cut_state(sn_engine * e, int slot, int from, char * err, int err_len) {
    // back to a draft row of the last step: its spare entry holds the state after it (the entries swap; nothing is copied)
    const bool to_dr = from < e->slen[slot] && q35_draft_cut(e, slot, from);
    const bool to_ck = !to_dr && q35_ck_cut(e, slot, from);
    if (from != 0 && from < e->slen[slot] && !to_ck && !to_dr) { set_err(err, err_len, "%s", "sn_clear: a recurrent state cannot go back but to its checkpoint (sn_cut_ok)"); return -3; }
    if (from == 0) { q35_release(e, slot); e->slen[slot] = 0; }
    if (to_ck) { q35_copy(e, e->ckent[slot], e->sidx[slot]); e->slen[slot] = from; }  // back to where the prompt ended
    if (to_dr) q35_to_draft(e, slot, from);
    e->dn[slot] = 0;
    return 0;
}

// a shared page p: the sequence takes its own copy (*tc, the page of its chunk) of the cells it keeps (n; the rest of a new
// page is zero); 0, or -1 / -2 with err set
static int page_own(sn_engine * e, int w, int * tc, int p, int n, char * err, int err_len) {
    const int np = page_new(e, w);
    if (np < 0) { set_err(err, err_len, "%s", "out of cache pages"); return -2; }
    --e->pl[w].ref[p];
    *tc = np;
    if (page_copy(e, w, p, np, 0, n, err, err_len) != 0) return -1;
    return 0;
}

// cells [a, b) of page p zeroed, every layer of pool w; 0, or -1 with err set
static int page_zero(sn_engine * e, int w, int p, int a, int b, char * err, int err_len) {
    return kv_blit(e, w, ^(id<MTLBlitCommandEncoder> be, id<MTLBuffer> kb, id<MTLBuffer> vb, size_t off, size_t cell, int i) {
        for (id<MTLBuffer> buf in @[kb, vb]) [be fillBuffer:buf range:NSMakeRange(off + ((size_t) p * 64 + a) * cell, (size_t) (b - a) * cell) value:0];
    }, err, err_len);
}

// pool w's cells of `slot` from `from` on (up to `to` in the page `from` is in) given back or zeroed; 0, or -1 / -2 with
// err set
static int clear_pages(sn_engine * e, int w, int slot, int from, int to, char * err, int err_len) {
    int * t = seq_pages(e, w, slot);
    // every chunk from the first one wholly at or past `from` goes (also one a dropped queued step began)
    const int c1 = (from + 63) / 64;
    pages_drop(e, w, t + c1, e->maxc - c1);
    for (int c = c1; c < e->maxc; ++c) t[c] = -1;
    if (w) {  // the window chunks the shorter sequence no longer needs
        const int lo = win_keep(e, from);
        if (lo > e->wlo[slot]) { win_drop(e, slot, e->wlo[slot], lo); e->wlo[slot] = lo; }
    }
    const int c = from / 64, p = from % 64 ? t[c] : -1;
    if (p < 0 || from == to) return 0;
    if (e->pl[w].ref[p] > 1) return page_own(e, w, &t[c], p, from % 64, err, err_len);  // shared: the sequence takes its own copy of the cells it keeps (the rest of a new page is zero)
    return page_zero(e, w, p, from % 64, MIN(64, to - c * 64), err, err_len);
}

int sn_clear(sn_engine * e, int slot, int from, int to, char * err, int err_len) {
    @autoreleasepool {
        settle(e);
        if (!slot_ok(e, slot, from, to)) { set_err(err, err_len, "%s", "sn_clear: bad slot or cells"); return -1; }
        if (!win_holds(e, slot, from)) { set_err(err, err_len, "%s", "sn_clear: the window layers no longer hold the cells before the cut (sn_cut_ok)"); return -3; }
        const int rc = e->q35 ? q35_cut_state(e, slot, from, err, err_len) : 0;
        if (rc) return rc;
        for (int w = 0; w < e->npool; ++w) {
            const int rcw = clear_pages(e, w, slot, from, to, err, err_len);
            if (rcw) return rcw;
        }
        if (from == 0) e->wlo[slot] = 0;
        return 0;
    }
}

// The host copy of a sequence of n cells: pool 0's layers, then the window pool's, each layer K then V; pool 0 keeps
// cells 0 .. n, the window pool cells win_first(n) * 64 .. n.
static size_t kv_bytes_w(const sn_engine * e, int w, int n) {
    size_t off[256], cell[256], b = 0;
    const int nl = kv_layers(e, w, off, cell, NULL, NULL), m = w ? n - win_first(e, n) * 64 : n;
    for (int i = 0; i < nl; ++i) b += 2 * (size_t) m * cell[i];
    return b;
}

size_t sn_kv_bytes(const sn_engine * e, int n) {
    return kv_bytes_w(e, 0, n) + (e->npool > 1 ? kv_bytes_w(e, 1, n) : 0) + (e->q35 && n > 0 ? q35_cbytes(e) + q35_sbytes(e) : 0);  // + the states
}

int sn_win_need(const sn_engine * e, int n) { return e->npool < 2 || n <= 0 ? 0 : MIN((n + 63) / 64, (e->win + e->wback + e->rmax) / 64 + 3); }
int sn_win_room(const sn_engine * e) { return e->npool < 2 ? 1 << 30 : e->wcap - e->pl[1].n + e->pl[1].nfree; }
int sn_win_pages(const sn_engine * e, int seq) {
    if (e->npool < 2) return 0;
    const int * t = seq_pages(e, 1, seq);
    int n = 0;
    for (int c = 0; c < e->maxc; ++c) n += t[c] >= 0;
    return n;
}
int sn_pages_total(const sn_engine * e) { return e->pl[0].n; }
int sn_pages_free(const sn_engine * e) { return e->pl[0].nfree; }
int sn_trim_pending(sn_engine * e) { @autoreleasepool { return e->pend ? sp_release(e) : 0; } }

size_t sn_mem_used(const sn_engine * e) {
    size_t b = (size_t) (e->pl[0].n - e->pl[0].nfree) * kv_bytes_w(e, 0, 64);
    if (e->npool > 1) b += (size_t) (e->pl[1].n - e->pl[1].nfree) * e->wunits * 256;
    if (e->q35) b += (size_t) (e->nst - e->nsfree) * (q35_cbytes(e) + q35_sbytes(e));
    return b;
}
size_t sn_state_bytes(const sn_engine * e) { return e->q35 ? q35_cbytes(e) + q35_sbytes(e) : 0; }
int sn_state_entries(const sn_engine * e, int seq) {
    if (!e->q35 || seq < 0 || seq >= e->slots) return 0;
    int n = (e->sidx[seq] >= 0) + (e->ckent[seq] >= 0);
    for (int j = 0; j < 3; ++j) n += e->dent[seq * 3 + j] >= 0;
    return n;
}
size_t sn_page_bytes(const sn_engine * e) { return kv_bytes_w(e, 0, 64) + (e->npool > 1 ? kv_bytes_w(e, 1, 64) : 0); }

int sn_seq_pages(const sn_engine * e, int seq, int * own) {
    const int * t = seq_pages(e, 0, seq);
    int n = 0, o = 0;
    for (int c = 0; c < e->maxc; ++c) if (t[c] >= 0) { ++n; o += e->pl[0].ref[t[c]] == 1; }
    if (own) *own = o;
    return n;
}

// a sequence's cache moved to (save) or from its host copy: the copy's buffer, where the next layer's cells go in it
typedef struct { __unsafe_unretained id<MTLBuffer> st; size_t at; int slot, n; bool save; } kv_move_t;

// one buffer's cells (the layer at byte off, cell bytes a cell; t: the sequence's pages, from chunk c0) to or from the
// host copy at m->at
static void kv_move_cells(id<MTLBlitCommandEncoder> be, id<MTLBuffer> b, size_t off, size_t cell, const int * t, int c0, const kv_move_t * m) {
    const int n = m->n;
    for (int c = c0; c * 64 < n; ++c) {
        const size_t k = (size_t) MIN(64, n - c * 64) * cell, dev_at = off + (size_t) t[c] * 64 * cell, host_at = m->at + (size_t) (c - c0) * 64 * cell;
        if (m->save) [be copyFromBuffer:b sourceOffset:dev_at toBuffer:m->st destinationOffset:host_at size:k];
        else [be copyFromBuffer:m->st sourceOffset:host_at toBuffer:b destinationOffset:dev_at size:k];
    }
}

// pool w's cells of the sequence (the window pool's from win_first(n) on) to or from the host copy; a restore first gives
// the sequence (which holds nothing) its pages. 0, or -1 / -2 with err set
static int kv_move_pool(sn_engine * e, int w, kv_move_t * m, char * err, int err_len) {
    int * t = seq_pages(e, w, m->slot);
    const int n = m->n, c0 = w ? win_first(e, n) : 0;
    if (!m->save) {  // a restore fills a sequence that holds nothing
        for (int c = c0; c < (n + 63) / 64; ++c)
            if (t[c] < 0 && (t[c] = page_new(e, w)) < 0) { set_err(err, err_len, "%s", "out of cache pages"); return -2; }
        if (w) e->wlo[m->slot] = c0;
    }
    for (int c = c0; c < (n + 63) / 64; ++c) if (t[c] < 0) { set_err(err, err_len, "%s", "the sequence has no cells there"); return -1; }
    const int cells = n - c0 * 64;
    return kv_blit(e, w, ^(id<MTLBlitCommandEncoder> be, id<MTLBuffer> kb, id<MTLBuffer> vb, size_t off, size_t cell, int i) {
        for (id<MTLBuffer> b in @[kb, vb]) {
            kv_move_cells(be, b, off, cell, t, c0, m);
            m->at += (size_t) cells * cell;
        }
    }, err, err_len);
}

// Qwen3.5: the states after the cells (a save of the whole sequence only) to or from the host copy; 0, or -1 / -2 with
// err set
static int kv_move_state(sn_engine * e, const kv_move_t * m, char * err, int err_len) {
    const int slot = m->slot, n = m->n;
    const bool save = m->save;
    if (save && n != e->slen[slot]) { set_err(err, err_len, "%s", "a recurrent state is only at the sequence's end"); return -1; }
    const int ent = save ? e->sidx[slot] : q35_entry(e, slot);
    if (ent < 0) { set_err(err, err_len, "%s", save ? "the sequence has no state" : "out of recurrent states"); return -2; }
    id<MTLCommandBuffer> cb = [e->queue commandBuffer];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    const size_t cb_ = q35_cbytes(e), sb = q35_sbytes(e), at = m->at;
    if (save) {
        [be copyFromBuffer:e->cst sourceOffset:(size_t) ent * cb_ toBuffer:m->st destinationOffset:at size:cb_];
        [be copyFromBuffer:e->sst sourceOffset:(size_t) ent * sb toBuffer:m->st destinationOffset:at + cb_ size:sb];
    } else {
        [be copyFromBuffer:m->st sourceOffset:at toBuffer:e->cst destinationOffset:(size_t) ent * cb_ size:cb_];
        [be copyFromBuffer:m->st sourceOffset:at + cb_ toBuffer:e->sst destinationOffset:(size_t) ent * sb size:sb];
    }
    [be endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (!save) { e->slen[slot] = n; e->dn[slot] = 0; }
    return 0;
}

static int kv_move(sn_engine * e, int slot, int n, void * host, bool save, char * err, int err_len) {
    @autoreleasepool {
        settle(e);
        if (!slot_ok(e, slot, 0, n)) { set_err(err, err_len, "%s", "bad slot or cells"); return -1; }
        if (n == 0) return 0;
        const size_t total = sn_kv_bytes(e, n);
        NS_VALID_UNTIL_END_OF_SCOPE id<MTLBuffer> st = [dev newBufferWithLength:total options:MTLResourceStorageModeShared];  // (m holds it unretained)
        if (!st) { set_err(err, err_len, "%s", "no memory for the cache copy"); return -1; }
        if (!save) memcpy(st.contents, host, total);
        kv_move_t m = {st, 0, slot, n, save};
        for (int w = 0; w < e->npool; ++w) {
            const int rc = kv_move_pool(e, w, &m, err, err_len);
            if (rc != 0) return rc;
        }
        const int rc = e->q35 ? kv_move_state(e, &m, err, err_len) : 0;  // the states after the cells (a save of the whole sequence only)
        if (rc != 0) return rc;
        if (save) memcpy(host, st.contents, total);
        return 0;
    }
}

int sn_kv_save(sn_engine * e, int slot, int n, void * dst, char * err, int err_len) { return kv_move(e, slot, n, dst, true, err, err_len); }
int sn_kv_load(sn_engine * e, int slot, int n, const void * src, char * err, int err_len) { return kv_move(e, slot, n, (void *) src, false, err, err_len); }

// Pool w's pages in use at or above T (below `top`) moved to the lowest free pages below T (every layer, the page tables
// and holders with them); the free list rebuilt from the pages nobody holds below `end`. Nothing may run on the GPU.
// pool_pack's moves: each page in use at or above T (below top) to the lowest free page below T (src[j] -> dst[j]);
// their count, or -1 when the pages below T are too few
static int pack_plan(const pool_t * pool, int T, int top, int * src, int * dst) {
    int nm = 0, q = 0;
    for (int p = T; p < top; ++p) {
        if (pool->ref[p] <= 0) continue;
        while (q < T && pool->ref[q] > 0) ++q;
        if (q >= T) return -1;
        src[nm] = p; dst[nm++] = q++;
    }
    return nm;
}

// the nm pages' cells copied over every layer of pool w, then the pages moved from zeroed (they stay so if they cannot be
// unmapped); -1 when the copy failed
static int pack_move(sn_engine * e, int w, const int * sp, const int * dp, int nm) {
    if (kv_blit(e, w, ^(id<MTLBlitCommandEncoder> be, id<MTLBuffer> kb, id<MTLBuffer> vb, size_t off, size_t cell, int li) {
            for (int j = 0; j < nm; ++j)
                for (id<MTLBuffer> b in @[kb, vb])
                    [be copyFromBuffer:b sourceOffset:off + (size_t) sp[j] * 64 * cell toBuffer:b destinationOffset:off + (size_t) dp[j] * 64 * cell size:64 * cell];
        }, NULL, 0)) return -1;
    // the pages moved from are free now, so zero (they stay so if they cannot be unmapped)
    kv_blit(e, w, ^(id<MTLBlitCommandEncoder> be, id<MTLBuffer> kb, id<MTLBuffer> vb, size_t off, size_t cell, int li) {
        for (int j = 0; j < nm; ++j)
            for (id<MTLBuffer> b in @[kb, vb]) [be fillBuffer:b range:NSMakeRange(off + (size_t) sp[j] * 64 * cell, 64 * cell) value:0];
    }, NULL, 0);
    return 0;
}

// the holders and page tables of pool w follow the nm moves (pages [T, top) moved)
static void pack_remap(sn_engine * e, pool_t * pool, const int * src, const int * dst, int nm, int T, int top) {
    int * to = malloc((size_t) top * sizeof(int));
    for (int p = 0; p < top; ++p) to[p] = p;
    for (int j = 0; j < nm; ++j) { to[src[j]] = dst[j]; pool->ref[dst[j]] = pool->ref[src[j]]; pool->ref[src[j]] = 0; }
    int * t = (int *) pool->pt.contents;
    for (size_t i = 0; i < (size_t) e->slots * e->maxc; ++i) if (t[i] >= T && t[i] < top) t[i] = to[t[i]];
    free(to);
}

static int pool_pack(sn_engine * e, int w, int T, int top, int end) {
    pool_t * P = &e->pl[w];
    int * src = malloc((size_t) MAX(top - T, 1) * sizeof(int)), * dst = malloc((size_t) MAX(top - T, 1) * sizeof(int));
    const int nm = pack_plan(P, T, top, src, dst);
    if (nm < 0 || (nm && pack_move(e, w, src, dst, nm))) { free(src); free(dst); return -1; }
    if (nm) pack_remap(e, P, src, dst, nm, T, top);
    free(src); free(dst);
    P->nfree = 0;
    for (int p = end - 1; p >= 0; --p) if (P->ref[p] <= 0) { P->ref[p] = 0; P->freel[P->nfree++] = p; }
    return nm;
}

// what a pool of n (pages or entries, used of them in use, mapped g at a time) keeps: the used ones' chunks and `spare`
// chunks more, at least lo
static int trim_target(int n, int used, int g, int lo, int spare) { return MIN(n, MAX(lo, (used + g - 1) / g * g + spare * g)); }

// sn_trim's moves need the GPU idle: the queued step let go and both sets' steps done (once)
static void trim_quiet(sn_engine * e, bool * settled) {
    if (*settled) return;
    settle(e);
    for (int s_ = 0; s_ < 2; ++s_) if (e->cbs[s_]) [e->cbs[s_] waitUntilCompleted];
    *settled = true;
}

// pool 0: its pages in use packed below T (the used pages' chunks, `spare` chunks more) and the mapped chunks above given
// back; the bytes given back
static size_t trim_pool0(sn_engine * e, int spare, bool * settled) {
    pool_t * P = &e->pl[0];
    const int used = P->n - P->nfree, T = trim_target(P->n, used, e->kgrow, e->kgrow, spare);
    if (getenv("SN_SPARSE_LOG")) fprintf(stderr, "sn: trim pool 0: used %d, target %d, mapped %d\n", used, T, e->kc_pages);
    if (T >= e->kc_pages) return 0;
    trim_quiet(e, settled);
    const int pk = pool_pack(e, 0, T, e->kc_pages, P->n);
    if (getenv("SN_SPARSE_LOG")) fprintf(stderr, "sn: trim pool 0: moved %d\n", pk);
    if (pk < 0 || pool_map(e, 0, T, e->kc_pages, false) != 0) return 0;
    const size_t given = (size_t) (e->kc_pages - T) * kv_bytes_w(e, 0, 64);
    e->kc_pages = T;
    return given;
}

// the window pool: as pool 0 (never below its first chunk); the bytes given back
static size_t trim_window(sn_engine * e, int spare, bool * settled) {
    pool_t * P = &e->pl[1];
    const int T = trim_target(P->n, P->n - P->nfree, e->wgrow, e->wn0, spare);
    if (T >= P->n) return 0;
    trim_quiet(e, settled);
    const int n0 = P->n;
    if (pool_pack(e, 1, T, n0, n0) < 0) return 0;
    const int n1 = pool_map(e, 1, T, n0, false) == 0 ? T : n0;
    P->nfree = 0;  // the pages left
    for (int p = n1 - 1; p >= 0; --p) if (!P->ref[p]) P->freel[P->nfree++] = p;
    P->n = n1;
    return (size_t) (n0 - n1) * e->wunits * 256;
}

// the state entries held: each sequence's state, checkpoint and spare entries (calloc'd, nst bytes)
static char * states_busy(const sn_engine * e) {
    char * busy = calloc((size_t) e->nst, 1);
    for (int s = 0; s < e->slots; ++s) {
        if (e->sidx[s] >= 0) busy[e->sidx[s]] = 1;
        if (e->ckent[s] >= 0) busy[e->ckent[s]] = 1;
        for (int j = 0; j < 3; ++j) if (e->dent[s * 3 + j] >= 0) busy[e->dent[s * 3 + j]] = 1;
    }
    return busy;
}

// sequence s's k-th entry: 0 its state, 1 its checkpoint, 2 .. 4 its spare entries
static int * state_ent(sn_engine * e, int s, int k) { return k == 0 ? &e->sidx[s] : k == 1 ? &e->ckent[s] : &e->dent[s * 3 + k - 2]; }

// entry *ent moved to the lowest free entry below T (*q: where the search goes on); false when there is none
static bool state_move_down(sn_engine * e, char * busy, int * ent, int * q, int T) {
    while (*q < T && busy[*q]) ++*q;
    if (*q >= T) return false;
    q35_copy(e, *ent, *q);
    q35_zero(e, *ent, *ent + 1);  // free entries are zero
    busy[*ent] = 0; busy[*q] = 1; *ent = *q;
    return true;
}

// every entry held at or above T moved below it; false when the entries below T are too few
static bool states_pack(sn_engine * e, char * busy, int T) {
    int q = 0;
    for (int s = 0; s < e->slots; ++s)
        for (int k = 0; k < 5; ++k) {
            int * ent = state_ent(e, s, k);
            if (*ent >= T && !state_move_down(e, busy, ent, &q, T)) return false;
        }
    return true;
}

// Qwen3.5's state entries: a sequence's state or checkpoint above T moves down, the chunks above given back; the bytes
static size_t trim_states(sn_engine * e, int spare, bool * settled) {
    const int T = trim_target(e->nst, e->nst - e->nsfree, e->stgrow, e->stgrow, spare);
    if (T >= e->nst) return 0;
    trim_quiet(e, settled);
    char * busy = states_busy(e);
    const bool ok = states_pack(e, busy, T);
    const int n0 = e->nst, n1 = ok && q35_map(e, T, n0, false) == 0 ? T : n0;
    e->nsfree = 0;  // the entries left (the free ones are zero)
    for (int i = n1 - 1; i >= 0; --i) if (!busy[i]) e->sfree[e->nsfree++] = i;
    size_t given = 0;
    if (n1 < n0) { e->nst = n1; given = (size_t) (n0 - n1) * (q35_cbytes(e) + q35_sbytes(e)); }
    free(busy);
    return given;
}

// see sn.h
size_t sn_trim(sn_engine * e, int spare) {
    @autoreleasepool {
        size_t given = 0;
        if (e->pend) sp_release(e);
        spare = MAX(spare, 0);
        bool settled = false;
        if (e->sparse) given += trim_pool0(e, spare, &settled);  // pool 0
        if (e->wsp) given += trim_window(e, spare, &settled);  // the window pool
        if (e->q35sp) given += trim_states(e, spare, &settled);  // Qwen3.5's state entries: a sequence's state or checkpoint above T moves down
        if (given && getenv("SN_SPARSE_LOG")) fprintf(stderr, "sn: trim gave back %.2f GB (cache %d pages, window %d, states %d)\n", given / 1e9, e->kc_pages, e->pl[1].n, e->nst);
        return given;
    }
}

void sn_close(sn_engine * e) {
    if (!e) return;
    settle(e);
    if (g_idslog && g_idsbuf) {  // calls (int), each call's rows (ints), then every call's rows x k ids
        FILE * f = fopen(g_idslog, "wb");
        if (f) { fwrite(&g_idsn, 4, 1, f); fwrite(g_idsr, 4, g_idsn, f); fwrite(g_idsbuf.contents, 1, g_idsoff, f); fclose(f); }
    }
    if (g_time && g_t_n) {
        fprintf(stderr, "sn time per step (%ld): encode %.3f ms (to the layers %.3f), commit->done %.3f, gpu %.3f, queue->gpu start %.3f\n", g_t_n,
                g_t_enc / g_t_n * 1e3, g_t_layers / g_t_n * 1e3, g_t_wait / g_t_n * 1e3, g_t_gpu / g_t_n * 1e3, g_t_lat / g_t_n * 1e3);
        fprintf(stderr, "sn queued steps: %ld taken, %ld dropped\n", g_ch_used, g_ch_drop);
        g_t_n = 0; g_t_enc = g_t_wait = g_t_gpu = g_t_lat = g_t_layers = 0; g_ch_used = g_ch_drop = 0;
    }
    if (g_prof_steps) {
        double tot = 0;
        for (NSString * k in g_prof_sum) tot += g_prof_sum[k].doubleValue;
        fprintf(stderr, "sn profile: %d steps of %d rows, %.3f ms a step\n", g_prof_steps, g_prof_r, tot / g_prof_steps);
        for (NSString * k in [g_prof_sum keysSortedByValueUsingSelector:@selector(compare:)].reverseObjectEnumerator)
            fprintf(stderr, "  %-16s %8.3f ms  %5.1f%%\n", k.UTF8String, g_prof_sum[k].doubleValue / g_prof_steps, 100 * g_prof_sum[k].doubleValue / tot);
        g_prof_steps = 0;
        [g_prof_sum removeAllObjects];
    }
    free(e->ly); free(e->gl); free(e->g.t); free(e->pl[0].ref); free(e->pl[0].freel); free(e->pl[1].ref); free(e->pl[1].freel);
    free(e->sp_in); free(e->soft_par); free(e->wlo); free(e->wmin); free(e->no_end); free(e->last_probs); free(e->ch_seq); free(e->ql); free(e->sidx); free(e->sfree); free(e->slen);
    munmap((void *) e->g.base, e->g.size);
    free(e);
}
