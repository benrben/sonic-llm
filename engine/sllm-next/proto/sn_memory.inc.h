// sllm-next engine library, a part of sn.m (one translation unit; sn.m includes its parts in order): the cache's and the states' memory: placement sparse buffers mapped and given back a chunk at a time

static int kv_layers(const sn_engine * e, int w, size_t * off, size_t * cell, __unsafe_unretained id<MTLBuffer> * kb, __unsafe_unretained id<MTLBuffer> * vb);
static size_t kv_bytes_w(const sn_engine * e, int w, int n);

// pool 0's K or V buffer: a placement sparse one when it can be (see sn_engine.sparse)
static id<MTLBuffer> kv_alloc(sn_engine * e, size_t len) {
    if (@available(macOS 26.0, *)) {
        const char * s = getenv("SN_SPARSE");
        if ((!s || strcmp(s, "0")) && len <= dev.maxBufferLength) {
            id<MTLBuffer> b = [dev newBufferWithLength:len options:MTLResourceStorageModePrivate placementSparsePageSize:MTLSparsePageSize64];
            if (b) { e->sparse = true; return b; }
        }
    }
    e->sparse = false;
    return [dev newBufferWithLength:len options:MTLResourceStorageModePrivate];
}

// whether placement sparse buffers are wanted (macOS 26; SN_SPARSE=0: plain buffers)
static bool sp_wanted(void) {
    if (@available(macOS 26.0, *)) {
        const char * s = getenv("SN_SPARSE");
        return !s || strcmp(s, "0");
    }
    return false;
}

static int sp_release(sn_engine * e);

// the Metal 4 queue and its event, the kind's heap table, the tile size: true when chunk `chunk` of `kind` (n ranges) can
// be mapped
static bool sp_ready(sn_engine * e, int kind, int chunk, int n) API_AVAILABLE(macos(26.0)) {
    if (!e->mq4) { e->mq4 = [dev newMTL4CommandQueue]; e->mev = [dev newSharedEvent]; }
    if (!e->hp[kind]) e->hp[kind] = [NSMutableDictionary new];
    if (!e->mq4 || !e->mev || n > 512 || e->hp[kind][@(chunk)]) return false;
    if (e->pend) sp_release(e);
    if (!e->tile) e->tile = [dev sparseTileSizeInBytesForSparsePageSize:MTLSparsePageSize64];
    return true;
}

// a placement heap of `bytes` for sparse pages
static id<MTLHeap> sp_heap(size_t bytes) API_AVAILABLE(macos(26.0)) {
    MTLHeapDescriptor * hd = [MTLHeapDescriptor new];
    hd.type = MTLHeapTypePlacement;
    hd.storageMode = MTLStorageModePrivate;
    hd.size = bytes;
    hd.maxCompatiblePlacementSparsePageSize = MTLSparsePageSize64;
    return [dev newHeapWithDescriptor:hd];
}

// the n ranges mapped onto heap h one after another (h nil: unmapped), one call a run of ranges in the same buffer
static void sp_update(sn_engine * e, id<MTLHeap> h, __unsafe_unretained id<MTLBuffer> * bufs, const size_t * a, const size_t * len, int n)
    API_AVAILABLE(macos(26.0)) {
    MTL4UpdateSparseBufferMappingOperation ops[512];
    id<MTL4CommandQueue> q4 = e->mq4;
    size_t at = 0;
    for (int i = 0, i0 = 0; i < n; ++i) {  // one call a run of ranges in the same buffer
        ops[i] = (MTL4UpdateSparseBufferMappingOperation) { h ? MTLSparseTextureMappingModeMap : MTLSparseTextureMappingModeUnmap,
                                                            NSMakeRange(a[i] / e->tile, len[i] / e->tile), h ? at / e->tile : 0 };
        at += len[i];
        if (i + 1 == n || bufs[i + 1] != bufs[i]) { [q4 updateBufferMappings:bufs[i] heap:h operations:ops + i0 count:i + 1 - i0]; i0 = i + 1; }
    }
}

// the mapping queue's work so far done: true, or false after 10 s (the mapping never came)
static bool sp_wait(sn_engine * e) API_AVAILABLE(macos(26.0)) {
    [e->mq4 signalEvent:e->mev value:++e->mevv];
    for (int k = 0; e->mev.signaledValue < e->mevv; ++k) {
        if (k > 200000) return false;  // 10 s: the mapping never came
        usleep(50);
    }
    return true;
}

// the n ranges zeroed (waits for the GPU): 0, or -1 when the blit failed
static int sp_zero(sn_engine * e, __unsafe_unretained id<MTLBuffer> * bufs, const size_t * a, const size_t * len, int n) {
    id<MTLCommandBuffer> cb = [e->queue commandBuffer];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    for (int i = 0; i < n; ++i) [be fillBuffer:bufs[i] range:NSMakeRange(a[i], len[i]) value:0];
    [be endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return cb.error ? -1 : 0;
}

// Back n ranges (buffer bufs[i], byte offset a[i], length len[i], tile multiples) of placement sparse buffers with one new
// heap, and zero them; the heap is kept as chunk `chunk` of kind `kind` (sn_engine.hp): 0, or -1 when the memory cannot be had
static int sp_map_now(sn_engine * e, int kind, int chunk, __unsafe_unretained id<MTLBuffer> * bufs, const size_t * a, const size_t * len, int n)
    API_AVAILABLE(macos(26.0)) {
    if (!sp_ready(e, kind, chunk, n)) return -1;
    size_t bytes = 0;
    for (int i = 0; i < n; ++i) bytes += len[i];
    if (!bytes) return 0;
    id<MTLHeap> h = sp_heap(bytes);
    if (!h) return -1;
    sp_update(e, h, bufs, a, len, n);
    if (!sp_wait(e) || sp_zero(e, bufs, a, len, n)) return -1;
    e->hp[kind][@(chunk)] = h;
    return 0;
}

static int sp_map(sn_engine * e, int kind, int chunk, __unsafe_unretained id<MTLBuffer> * bufs, const size_t * a, const size_t * len, int n) {
    if (@available(macOS 26.0, *)) {
        @autoreleasepool {
            return sp_map_now(e, kind, chunk, bufs, a, len, n);
        }
    }
    return -1;
}

// the heaps of given-back chunks whose unmap has run, let go (their memory back to the system); the number still waiting
static int sp_release(sn_engine * e) {
    while (e->pend.count && e->mev.signaledValue >= [e->pend[0][0] unsignedLongLongValue]) {
        if (getenv("SN_SPARSE_LOG")) fprintf(stderr, "sn: unmap done after %.0f ms\n", (CFAbsoluteTimeGetCurrent() - [e->pend[0][2] doubleValue]) * 1e3);
        [e->pend removeObjectAtIndex:0];
    }
    return (int) e->pend.count;
}

// the heaps of kind `kind`'s chunks [c0, c1), taken off its table
static NSMutableArray * sp_take_heaps(sn_engine * e, int kind, int c0, int c1) {
    NSMutableArray * hs = [NSMutableArray new];
    for (int c = c0; c < c1; ++c) {
        id h = e->hp[kind][@(c)];
        if (h) { [hs addObject:h]; [e->hp[kind] removeObjectForKey:@(c)]; }
    }
    return hs;
}

// Give back chunks [c0, c1) of kind `kind` (sp_map's heaps), whose ranges are the n ranges (bufs[i], a[i], len[i]): one
// unmap on the Metal 4 queue, not waited for (it takes ~70 ms a 256 MB chunk; the first after a busy run took 12 s
// waited one chunk at a time). Their heaps are kept until it has run (sp_release). Nothing on the GPU may use the ranges.
static int sp_unmap(sn_engine * e, int kind, int c0, int c1, __unsafe_unretained id<MTLBuffer> * bufs, const size_t * a, const size_t * len, int n) {
    if (@available(macOS 26.0, *)) {
        @autoreleasepool {
            if (!e->mq4 || n > 512) return -1;
            NSMutableArray * hs = sp_take_heaps(e, kind, c0, c1);
            sp_update(e, nil, bufs, a, len, n);
            [e->mq4 signalEvent:e->mev value:++e->mevv];
            if (!e->pend) e->pend = [NSMutableArray new];
            [e->pend addObject:@[@(e->mevv), hs, @(CFAbsoluteTimeGetCurrent())]];
            return 0;
        }
    }
    return -1;
}

// pool w's pages [p0, p1) as ranges of its layers' buffers (every K, then every V): their count
static int pool_ranges(sn_engine * e, int w, int p0, int p1, __unsafe_unretained id<MTLBuffer> * bufs, size_t * a, size_t * len) {
    size_t off[256], cell[256];
    __unsafe_unretained id<MTLBuffer> kb[256], vb[256];
    const int n = kv_layers(e, w, off, cell, kb, vb);
    for (int i = 0; i < n; ++i) {
        a[i] = a[n + i] = off[i] + (size_t) p0 * 64 * cell[i];
        len[i] = len[n + i] = (size_t) (p1 - p0) * 64 * cell[i];
        bufs[i] = kb[i]; bufs[n + i] = vb[i];
    }
    return 2 * n;
}

// map pool w's pages [p0, p1) (its chunk p0 / grow), or give back pages [p0, p1) (whole chunks, p1 the pool's end or a
// chunk boundary)
static int pool_map(sn_engine * e, int w, int p0, int p1, bool map) {
    __unsafe_unretained id<MTLBuffer> bufs[512];
    size_t a[512], len[512];
    const int n = pool_ranges(e, w, p0, p1, bufs, a, len), g = w ? e->wgrow : e->kgrow;
    return map ? sp_map(e, w, p0 / g, bufs, a, len, n) : sp_unmap(e, w, p0 / g, (p1 + g - 1) / g, bufs, a, len, n);
}

// Back pool 0's pages [0, upto) with memory (whole kgrow chunks, a heap each, at most the pool), zeroed: 0, or -1 when the
// memory cannot be had
static int kv_commit(sn_engine * e, int upto) {
    if (!e->sparse || upto <= e->kc_pages) return 0;
    const int c1 = MIN(e->pl[0].n, (upto + e->kgrow - 1) / e->kgrow * e->kgrow);
    while (e->kc_pages < c1) {  // kc_pages: a chunk boundary (or the pool's end)
        const int q = MIN(e->pl[0].n, e->kc_pages + e->kgrow);
        if (pool_map(e, 0, e->kc_pages, q, true)) return -1;
        e->kc_pages = q;
    }
    if (getenv("SN_SPARSE_LOG")) fprintf(stderr, "sn: cache memory %d of %d pages (%.2f GB)\n", c1, e->pl[0].n, (double) c1 * kv_bytes_w(e, 0, 64) / 1e9);
    return 0;
}

static size_t q35_cbytes(const sn_engine * e);
static size_t q35_sbytes(const sn_engine * e);

// map / unmap Qwen3.5's state entries [n0, n1), its chunk n0 / stgrow
static int q35_map(sn_engine * e, int n0, int n1, bool map) {
    __unsafe_unretained id<MTLBuffer> bufs[2] = {e->cst, e->sst};
    const size_t cb = q35_cbytes(e), sb = q35_sbytes(e);
    const size_t a[2] = {(size_t) n0 * cb, (size_t) n0 * sb}, len[2] = {(size_t) (n1 - n0) * cb, (size_t) (n1 - n0) * sb};
    const int c = n0 / e->stgrow;
    return map ? sp_map(e, 2, c, bufs, a, len, 2) : sp_unmap(e, 2, c, (n1 + e->stgrow - 1) / e->stgrow, bufs, a, len, 2);
}

// Qwen3.5's state pool as sparse buffers (SN_SPARSE=0: plain): room for every slot's entry, about 512 MB mapped at a time
// in whole tiles; true when made (e->nst entries mapped)
// the state pool's growth step (e->stgrow): whole tiles of entries (cb + sb bytes each), about 512 MB; false when no count
// of up to 64 entries is whole tiles
static bool q35_sparse_step(sn_engine * e, size_t cb, size_t sb) {
    int k = 1;
    while (k < 64 && ((k * cb) % e->tile || (k * sb) % e->tile)) ++k;
    if ((k * cb) % e->tile || (k * sb) % e->tile) return false;
    e->stgrow = MAX(k, (int) ((512u << 20) / (cb + sb)) / k * k);
    return true;
}

// the state pool's sparse buffers of cap entries, the first step mapped: true when made
static bool q35_sparse_make(sn_engine * e, int cap, size_t cb, size_t sb) API_AVAILABLE(macos(26.0)) {
    e->cst = [dev newBufferWithLength:(size_t) cap * cb options:MTLResourceStorageModePrivate placementSparsePageSize:MTLSparsePageSize64];
    e->sst = [dev newBufferWithLength:(size_t) cap * sb options:MTLResourceStorageModePrivate placementSparsePageSize:MTLSparsePageSize64];
    if (!e->cst || !e->sst) { e->cst = e->sst = nil; return false; }
    if (q35_map(e, 0, e->stgrow, true) == 0) {
        e->q35sp = true;
        e->nst = e->stgrow;  // q35_reset hands them out
        return true;
    }
    e->cst = e->sst = nil;
    e->nst = MIN(e->slots, 2);
    return false;
}

static bool q35_sparse_init(sn_engine * e) {
    if (@available(macOS 26.0, *)) {
        const char * s = getenv("SN_SPARSE");
        if (s && !strcmp(s, "0")) return false;
        e->tile = [dev sparseTileSizeInBytesForSparsePageSize:MTLSparsePageSize64];
        const size_t cb = q35_cbytes(e), sb = q35_sbytes(e);
        if (!q35_sparse_step(e, cb, sb)) return false;
        const int cap = (e->slots + e->stgrow - 1) / e->stgrow * e->stgrow;
        if ((size_t) cap * sb > dev.maxBufferLength) return false;
        return q35_sparse_make(e, cap, cb, sb);
    }
    return false;
}

// Pool 0 as sparse buffers: every layer's pages tile-aligned in kc / vc, the first chunk backed; else plain buffers
// pool 0's layers all in kc / vc, each tile-aligned with pages of whole tiles: true, page the bytes of a page over them
static bool kv_sparse_aligned(sn_engine * e, size_t * page) API_AVAILABLE(macos(26.0)) {
    size_t off[256], cell[256];
    __unsafe_unretained id<MTLBuffer> kb[256], vb[256];
    const int n = kv_layers(e, 0, off, cell, kb, vb);
    e->tile = [dev sparseTileSizeInBytesForSparsePageSize:MTLSparsePageSize64];
    bool ok = n > 0 && e->tile > 0;
    *page = 0;
    for (int i = 0; i < n && ok; ++i) {
        ok = kb[i] == e->pl[0].kc && vb[i] == e->pl[0].vc && off[i] % e->tile == 0 && (64 * cell[i]) % e->tile == 0;
        *page += 64 * cell[i];
    }
    return ok;
}

static void kv_sparse_init(sn_engine * e) {
    if (!e->sparse) return;
    if (@available(macOS 26.0, *)) {
        size_t page;
        if (kv_sparse_aligned(e, &page)) {
            const char * mb = getenv("SN_SPARSE_MB");
            const size_t chunk = (size_t) (mb && atoi(mb) > 0 ? atoi(mb) : 256) << 20;
            e->kgrow = MAX(1, (int) (chunk / (2 * page)));
            if (kv_commit(e, 1) == 0) return;
        }
    }
    const size_t kl = e->pl[0].kc.length, vl = e->pl[0].vc.length;
    e->sparse = false;
    e->kc_pages = 0;
    e->pl[0].kc = [dev newBufferWithLength:kl options:MTLResourceStorageModePrivate];
    e->pl[0].vc = [dev newBufferWithLength:vl options:MTLResourceStorageModePrivate];
}
