// sllm-next engine library, a part of sn.m (one translation unit; sn.m includes its parts in order): Gemma / gpt-oss's weights, cache pools and buffers; every family's step buffers; sn_open

// Gemma 3 / 4 and gpt-oss's layer i: its tensors (gpt-oss: biases, sinks and experts; Gemma: the FFN, the norms and the
// per-layer embedding's) and formats
static void gemma_layer_tensors(sn_engine * e, int i, g4layer_t * l) {
    gguf_t * g = &e->g;
    char nm[128];
#define TG2(field, suffix) do { snprintf(nm, sizeof nm, "blk.%d." suffix, i); l->field = g->data + find(g, nm)->off; } while (0)
#define TG2O(field, suffix) do { snprintf(nm, sizeof nm, "blk.%d." suffix, i); tensor_t * t_ = find_opt(g, nm); l->field = t_ ? g->data + t_->off : 0; } while (0)
    TG2(q, "attn_q.weight"); TG2O(k, "attn_k.weight"); TG2O(v, "attn_v.weight"); TG2(o, "attn_output.weight");
    if (e->oss) {  // the FFN's input norm is post_attention_norm
        TG2(attn_norm, "attn_norm.weight"); TG2(ffn_norm, "post_attention_norm.weight");
        TG2(qb, "attn_q.bias"); TG2(kb, "attn_k.bias"); TG2(vb, "attn_v.bias"); TG2(ob, "attn_output.bias"); TG2(sinks, "attn_sinks.weight");
        experts(g, i, &l->m);
    } else {
        TG2(gate, "ffn_gate.weight"); TG2(up, "ffn_up.weight"); TG2(down, "ffn_down.weight");
        TG2(attn_norm, "attn_norm.weight"); TG2(ffn_norm, "ffn_norm.weight"); TG2(q_norm, "attn_q_norm.weight"); TG2O(k_norm, "attn_k_norm.weight");
        TG2(post_attn, "post_attention_norm.weight"); TG2(post_ffw, "post_ffw_norm.weight"); TG2O(out_scale, "layer_output_scale.weight");
        if (g->ple) { TG2(inp_gate, "inp_gate.weight"); TG2(proj, "proj.weight"); TG2(post_norm, "post_norm.weight"); }
        snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", i);
        l->down_fmt = fmt_of(find(g, nm)->type);
    }
    snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", i);
    l->fmt = fmt_of(find(g, nm)->type);
    snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", i);
    l->v_fmt = find_opt(g, nm) ? fmt_of(find(g, nm)->type) : l->fmt;
}

// the prompt-tile list of tiles of tt tokens (a new one when no list has them)
static int tile_list(sn_engine * e, int tt) {
    int k = 0;
    while (k < e->ntt && e->tt[k] != tt) ++k;
    if (k == e->ntt) e->tt[e->ntt++] = tt;
    return k;
}

// layer i's shape (head size, K/V heads, FFN width, the layer whose K/V it reads: its own before KVL) and place: its
// prompt-tile list, the widest tile and K/V row, its pool and first element in a cell of the pool (units: K elements of
// a cell over each pool's layers so far; npi: the window pool's buffers so far; dmax: the widest head so far)
static void gemma_layer_place(sn_engine * e, int i, int KVL, size_t units[2], int * npi, int * dmax) {
    const gguf_t * g = &e->g;
    g4layer_t * l = &e->gl[i];
    l->swa = g->swa[i]; l->D = g->swa[i] ? g->hd_swa : g->hd; l->nkv = g->kva[i]; l->ff = g->ffa[i];
    l->has_kv = i < KVL; l->has_v = l->v != 0; l->kv_src = i < KVL ? i : KVL - (g->swa[i] ? 2 : 1);
    const int k = tile_list(e, 32 / (e->NH / l->nkv));
    l->list = k;
    e->ttb[k] = MAX(e->ttb[k], (size_t) l->nkv * 32 * l->D * 2);
    e->KD = MAX(e->KD, l->nkv * l->D);
    *dmax = MAX(*dmax, l->D);
    l->pool = l->swa && e->win > 0;  // a layer's place: its pool, and its first element in a cell of the pool (bytes: koff x pages x 128)
    if (l->has_kv) { l->koff = units[l->pool]; units[l->pool] += (size_t) l->nkv * l->D; }
    if (l->has_kv && l->pool) l->pi = (*npi)++;  // its buffer in the window pool
}

// Gemma 3 / 4 and gpt-oss's blocks: each layer's tensors, its K/V widths, its prompt-tile list and its place in the
// cache pools; units: K elements of a cell over each pool's layers
static void gemma_load_layers(sn_engine * e, size_t units[2]) {
    gguf_t * g = &e->g;
    const int L = e->L, NH = e->NH;
    const int KVL = L - g->shared_kv;
    e->kvl = KVL;
    int npi = 0;
    e->win = g->window;
    // A window model drops a sequence's old window chunks as it goes on, so a cut back to an earlier cell (the same prompt
    // sent again after an answer, a follow-up question) found nothing to keep and read everything again: 15.6 s against
    // llama.cpp's 0.08 s for the second of two equal 10,900-token requests on Gemma 4 12B. The chunks of the last 512
    // cells stay now (8 chunks a sequence; the cells themselves were never dropped from the full-attention pool).
    e->wback = getenv("SN_WINBACK") ? MAX(0, atoi(getenv("SN_WINBACK"))) / 64 * 64 : 512;
    int dmax = 0;
    e->gl = calloc(L, sizeof(g4layer_t));
    for (int i = 0; i < L; ++i) {
        gemma_layer_tensors(e, i, &e->gl[i]);
        gemma_layer_place(e, i, KVL, units, &npi, &dmax);
    }
    for (int i = KVL; i < L; ++i) { const g4layer_t * src = &e->gl[e->gl[i].kv_src]; e->gl[i].koff = src->koff; e->gl[i].pool = src->pool; e->gl[i].pi = src->pi; }
    e->npool = units[1] ? 2 : 1;
    e->wunits = units[1];
    e->QMAX = NH * dmax;
}

// gpt-oss: its rope's YaRN numbers and its MoE's buffers
static void oss_setup(sn_engine * e) {
    const gguf_t * g = &e->g;
    const int R = e->rmax;
    // YaRN as llama.cpp sets it up: the correction pairs from beta fast / slow over the original window, and
    // attn_factor get_mscale(f) / (1 + 0.1 ln f) (1), which the rope kernel times 1 + 0.1 ln f
    const float f = g->rope_scale, nd = (float) g->hd, bf = g->yarn_bf > 0 ? g->yarn_bf : 32.0f, bs = g->yarn_bs > 0 ? g->yarn_bs : 1.0f;
    const float c_bf = nd * logf((float) g->ctx_orig / (bf * 2 * (float) M_PI)) / (2 * logf(g->rope_base));
    const float c_bs = nd * logf((float) g->ctx_orig / (bs * 2 * (float) M_PI)) / (2 * logf(g->rope_base));
    e->ylo = MAX(0.0f, floorf(c_bf)); e->yhi = MIN(nd - 1, ceilf(c_bs));
    const float af = (0.1f * logf(f) + 1.0f) * (1.0f / (1.0f + 0.1f * logf(f))) * (g->rope_attn > 0 ? g->rope_attn : 1.0f);
    e->ymscale = af * (1.0f + 0.1f * logf(1.0f / (1.0f / f)));
    moe_alloc(e, dev, R);
}

// the window pool's layer buffers as sparse buffers with room for pool 0's pages, the first chunk mapped (e->wsp); else
// none made (the caller makes plain ones)
// a sparse K and a V buffer for each window layer with room for pool 0's pages (whole tiles of `tile` bytes a cell row):
// false when one cannot be made
static bool window_sparse_buffers(sn_engine * e, pool_t * pool, size_t tile) API_AVAILABLE(macos(26.0)) {
    for (int i = 0; i < e->L; ++i) {
        const g4layer_t * l = &e->gl[i];
        if (!l->has_kv || !l->pool) continue;
        const size_t len = (size_t) l->nkv * l->D * e->pl[0].n * 128;
        id<MTLBuffer> k = (((size_t) l->nkv * l->D * 128) % tile || len > dev.maxBufferLength) ? nil
            : [dev newBufferWithLength:len options:MTLResourceStorageModePrivate placementSparsePageSize:MTLSparsePageSize64];
        id<MTLBuffer> v = k ? [dev newBufferWithLength:len options:MTLResourceStorageModePrivate placementSparsePageSize:MTLSparsePageSize64] : nil;
        if (!k || !v) return false;
        [pool->lk addObject:k]; [pool->lv addObject:v];
    }
    return true;
}

// the window pool's first chunk mapped (wgrow pages a chunk, about 256 MB): true when it is
static bool window_sparse_map(sn_engine * e, pool_t * pool, size_t tile) {
    e->tile = tile;
    e->wgrow = MAX(1, (int) ((256u << 20) / ((size_t) e->wunits * 256)));
    const int n0 = MIN(e->pl[0].n, (pool->n + e->wgrow - 1) / e->wgrow * e->wgrow);
    const bool ok = pool_map(e, 1, 0, n0, true) == 0;
    if (ok) pool->n = e->wn0 = n0;
    return ok;
}

static void window_sparse_init(sn_engine * e, pool_t * P) {
    if (@available(macOS 26.0, *)) {
        const size_t tile = [dev sparseTileSizeInBytesForSparsePageSize:MTLSparsePageSize64];
        e->wsp = tile > 0 && window_sparse_buffers(e, P, tile);
        if (e->wsp) e->wsp = window_sparse_map(e, P, tile);
        if (!e->wsp) { [P->lk removeAllObjects]; [P->lv removeAllObjects]; e->hp[1] = nil; }
    }
}

// the window pool (pool 1, the sliding-window layers'): one sequence's window, the chunk it straddles, a full step of
// rows and one spare
static void window_pool_init(sn_engine * e) {
    const MTLResourceOptions PRIV = MTLResourceStorageModePrivate, SH = MTLResourceStorageModeShared;
    const int R = e->rmax, L = e->L, slots = e->slots;
    pool_t * P = &e->pl[1];
    P->n = MIN(e->pl[0].n, (e->win + R) / 64 + 3);
    e->wcap = e->pl[0].n;  // until the end of sn_open sets it from the working set
    P->lk = [NSMutableArray new]; P->lv = [NSMutableArray new];
    if (sp_wanted()) window_sparse_init(e, P);  // sparse layer buffers with room for pool 0's pages, the first chunk mapped (see wsp)
    for (int i = 0; i < L && !e->wsp; ++i) {
        const g4layer_t * l = &e->gl[i];
        if (!l->has_kv || !l->pool) continue;
        [P->lk addObject:[dev newBufferWithLength:(size_t) l->nkv * l->D * P->n * 128 options:PRIV]];
        [P->lv addObject:[dev newBufferWithLength:(size_t) l->nkv * l->D * P->n * 128 options:PRIV]];
    }
    P->pt = [dev newBufferWithLength:(size_t) slots * e->maxc * 4 options:SH];
    P->ref = calloc(e->pl[0].n, sizeof(int));
    P->freel = malloc((size_t) e->pl[0].n * sizeof(int));
}

// Gemma 4's per-layer embeddings: the CPU copies the step's rows of the table (tple), the GPU's buffers for the rest
static void ple_init(sn_engine * e, const tensor_t * tple) {
    const gguf_t * g = &e->g;
    const MTLResourceOptions PRIV = MTLResourceStorageModePrivate, SH = MTLResourceStorageModeShared;
    const int R = e->rmax;
    e->ple_tab = g->base + g->data + tple->off;
    e->ple_row = (size_t) e->PN / 256 * 176;
    for (int b = 0; b < 2; ++b) e->prawB[b] = [dev newBufferWithLength:(size_t) R * e->ple_row options:SH];
    e->pq = [dev newBufferWithLength:(size_t) R * e->PN * 4 options:PRIV];
    e->pp = [dev newBufferWithLength:(size_t) R * e->PN * 4 options:PRIV];
    e->ple = [dev newBufferWithLength:(size_t) R * e->PN * 4 options:PRIV];
    e->pg = [dev newBufferWithLength:(size_t) R * e->PLE * 4 options:PRIV];
    g_skpart = e->skpart = [dev newBufferWithLength:(size_t) MAX(8 * R * 512, 2 * R * 2048) * 4 options:PRIV];
}

// Gemma's tables: the per-layer token table (Gemma 4 E2B; NULL: none), the token table, the output layer (NULL: tied);
// q41: a Q4_K token table, read through a q4_1 copy that gives the embeddings (and the output layer when tied)
typedef struct { tensor_t * tple, * te, * to; bool q41; } gemma_tabs;

// every tensor the GPU reads in place is in one of its buffers: 0, or -1 with err set
static int gemma_mapped_all(sn_engine * e, const gemma_tabs * tb, char * err, int err_len) {
    gguf_t * g = &e->g;
    for (int i = 0; i < g->nt; ++i) {
        tensor_t * t = &g->t[i];
        if (t != tb->tple && !((tb->q41 || e->emb_cpu) && t == tb->te) && !gb(e, g->data + t->off)) { set_err(err, err_len, "tensor %s is in no GPU buffer", t->name); return -1; }
    }
    return 0;
}

// the GPU's buffers: the file without the tables it never reads in place (the per-layer table, whose rows the CPU copies,
// a Q4_K table, read through its copy, and gpt-oss's untied token table); 0, or -1 with err set
static int gemma_map(sn_engine * e, const gemma_tabs * tb, const char * path, char * err, int err_len) {
    const gguf_t * g = &e->g;
    uint64_t holes[2][2];
    int nh = 0;
    if (tb->tple) { holes[nh][0] = g->data + tb->tple->off; holes[nh][1] = holes[nh][0] + tsize(tb->tple); ++nh; }
    if (tb->q41 || (e->oss && tb->to)) { holes[nh][0] = g->data + tb->te->off; holes[nh][1] = holes[nh][0] + tsize(tb->te); ++nh; }
    if (e->oss && tb->to) emb_cpu_init(e, tb->te);  // gpt-oss's untied token table stays off the GPU too: the CPU copies the step's rows (as Qwen3's)
    if (map_file(e, holes, nh) != 0) { set_err(err, err_len, "%s: cannot map the weights for the GPU", path); return -1; }
    return gemma_mapped_all(e, tb, err, err_len);
}

// the final norm, Gemma 4's rope frequency factors, the per-layer embedding's projection and norm, the attention scale
static void gemma_globals(sn_engine * e) {
    gguf_t * g = &e->g;
    e->out_norm = g->data + find(g, "output_norm.weight")->off;
    tensor_t * tf = find_opt(g, "rope_freqs.weight");
    e->rope_ff = tf ? g->data + tf->off : 0;
    if (g->ple) {
        e->pl_proj = g->data + find(g, "per_layer_model_proj.weight")->off;
        e->pl_norm = g->data + find(g, "per_layer_proj_norm.weight")->off;
    }
    e->ascale = e->g4 ? 1.0f : 1.0f / sqrtf((float) (e->L == 62 && !e->oss ? e->E / e->NH : g->hd));  // llama.cpp's Gemma 3 27B rule (62 layers)
}

// mm formats of a Gemma token table (q6_K, q8_0, q4_0) and output layer (q6_K, q8_0, q4_0, q4_1)
static int gemma_embd_fmt(int type) { return type == 14 ? 2 : type == 8 ? 5 : 0; }
static int gemma_head_fmt(int type) { return type == 14 ? 2 : type == 8 ? 5 : type == 2 ? 0 : 1; }

// the token table (a Q4_K one's q4_1 copy, head41) and the output layer (head 0: head41)
static void gemma_tables(sn_engine * e, const gemma_tabs * tb) {
    const gguf_t * g = &e->g;
    const int E = e->E;
    e->V = (int) tb->te->ne[1];
    if (tb->q41) {
        e->head41 = [dev newBufferWithLength:(size_t) e->V * (E / 32) * 20 options:MTLResourceStorageModeShared];
        q4k_to_q41(g->base + g->data + tb->te->off, e->V, E, e->head41.contents);
    } else {
        e->embd = g->data + tb->te->off;
        e->embd_fmt = gemma_embd_fmt(tb->te->type);
    }
    if (tb->to || !tb->q41) {
        tensor_t * th = tb->to ? tb->to : tb->te;
        e->head = g->data + th->off;
        e->head_fmt = gemma_head_fmt(th->type);
        e->W = gb(e, e->head);
    }
}

// the cache pools (units: K elements of a cell over each pool's layers), the per-layer embedding's buffers, the prompt
// tiles and the attention output
static void gemma_buffers(sn_engine * e, const size_t units[2], const tensor_t * tple) {
    const MTLResourceOptions PRIV = MTLResourceStorageModePrivate;
    const int R = e->rmax;
    kv_alloc_pool0(e, MAX(units[0], 1) * e->pl[0].n * 128);
    if (e->npool == 2) window_pool_init(e);
    if (e->g.ple) ple_init(e, tple);
    size_t qb = 0;  // the prompt tiles: a step's rows in tiles of tt tokens of one sequence (encode checks the count)
    for (int k = 0; k < e->ntt; ++k) qb = MAX(qb, (size_t) (R / e->tt[k] + e->slots + 1) * e->ttb[k]);
    e->qhb = [dev newBufferWithLength:qb options:PRIV];
    e->y = [dev newBufferWithLength:(size_t) R * e->E * 4 options:PRIV];
}

// Gemma 3 / 4 and gpt-oss: weights, layers, rope, output table, the two cache pools and the step buffers of their own;
// 0, or -1 with err set
static int gemma_open(sn_engine * e, const char * path, char * err, int err_len) {
    gguf_t * g = &e->g;
    const int L = e->L;
    e->g4 = !strcmp(g->arch, "gemma4");
    e->PLE = g->ple; e->PN = g->ple * L;
    for (int i = 0; i < L; ++i) e->FMAX = MAX(e->FMAX, g->ffa[i]);
    gemma_tabs tb = {g->ple ? find(g, "per_layer_token_embd.weight") : NULL, find(g, "token_embd.weight"), find_opt(g, "output.weight"), false};
    tb.q41 = tb.te->type == 12;  // a Q4_K table: a q4_1 copy gives the embeddings (and the output layer when tied)
    if (gemma_map(e, &tb, path, err, err_len)) return -1;
    size_t units[2] = {0, 0};  // K elements of a cell over each pool's layers
    gemma_load_layers(e, units);
    gemma_globals(e);
    if (e->oss) oss_setup(e);
    gemma_tables(e, &tb);
    gemma_buffers(e, units, tb.tple);
    return 0;
}

// every family's step buffers, the step's double-buffered row records, pool 0's page table and the sequences' tables
static void common_buffers(sn_engine * e) {
    const MTLResourceOptions PRIV = MTLResourceStorageModePrivate, SH = MTLResourceStorageModeShared;
    const int R = e->rmax, E = e->E, slots = e->slots;
    e->x = [dev newBufferWithLength:(size_t) R * E * 4 options:PRIV];
    e->h = [dev newBufferWithLength:(size_t) R * E * 4 options:PRIV];
    e->q = [dev newBufferWithLength:(size_t) R * e->QMAX * 4 options:PRIV];
    e->ao = [dev newBufferWithLength:(size_t) R * e->QMAX * 4 options:PRIV];
    e->k = [dev newBufferWithLength:(size_t) R * e->KD * 4 options:PRIV];
    e->v = [dev newBufferWithLength:(size_t) R * e->KD * 4 options:PRIV];
    e->fg = [dev newBufferWithLength:(size_t) R * e->FMAX * 4 options:PRIV];
    e->xo = [dev newBufferWithLength:(size_t) (slots + 4) * E * 4 options:PRIV];  // the output rows only (n_out <= slots + 4)
    e->ho = [dev newBufferWithLength:(size_t) (slots + 4) * E * 4 options:PRIV];

    for (int b = 0; b < 2; ++b) {
        e->rtokB[b] = [dev newBufferWithLength:R * 4 options:SH]; e->rseqB[b] = [dev newBufferWithLength:R * 4 options:SH];
        e->rposB[b] = [dev newBufferWithLength:R * 4 options:SH]; e->ridxB[b] = [dev newBufferWithLength:R * 4 options:SH];
        e->otokB[b] = [dev newBufferWithLength:R * 4 options:SH];
        e->pB[b] = [dev newBufferWithLength:(size_t) R * 32 options:SH];  // a record per output row (up to every row of a step)
        e->rblkB[b] = [dev newBufferWithLength:(size_t) R * 16 * 2 options:SH];  // two tile lists (Gemma) of up to R tiles
    }
    e->ev = [dev newSharedEvent];
    e->pl[0].pt = [dev newBufferWithLength:(size_t) slots * e->maxc * 4 options:SH];
    e->pl[0].ref = calloc(e->pl[0].n, sizeof(int));
    e->pl[0].freel = malloc((size_t) e->pl[0].n * sizeof(int));
    if (!e->npool) e->npool = 1;
    e->wlo = calloc(slots, sizeof(int));
    e->wmin = malloc((size_t) slots * sizeof(int));
    e->no_end = calloc(slots, 1);
    e->last_probs = calloc((size_t) e->rmax * 8, sizeof(float));
    e->ch_set = -1;
    e->ch_seq = malloc((size_t) 3 * e->rmax * sizeof(int)); e->ch_pos = e->ch_seq + e->rmax; e->ch_tok = e->ch_pos + e->rmax;
}

// zero the caches after the warm-up: pool 0's (only the backed pages when sparse), the window pool's (the mapped pages
// when sparse) and Qwen3.5's states; waits for the GPU
static void zero_caches(sn_engine * e) {
    id<MTLCommandBuffer> cb = [e->queue commandBuffer];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    if (e->sparse) {  // only the backed pages
        size_t off[256], cell[256];
        const int n = kv_layers(e, 0, off, cell, NULL, NULL);
        for (int i = 0; i < n; ++i) {
            const NSRange r = NSMakeRange(off[i], (size_t) e->kc_pages * 64 * cell[i]);
            [be fillBuffer:e->pl[0].kc range:r value:0];
            [be fillBuffer:e->pl[0].vc range:r value:0];
        }
    } else {
        [be fillBuffer:e->pl[0].kc range:NSMakeRange(0, e->pl[0].kc.length) value:0];
        [be fillBuffer:e->pl[0].vc range:NSMakeRange(0, e->pl[0].vc.length) value:0];
    }
    if (e->wsp) {  // the mapped pages
        __unsafe_unretained id<MTLBuffer> bufs[512];
        size_t a[512], len[512];
        const int n = pool_ranges(e, 1, 0, e->pl[1].n, bufs, a, len);
        for (int i = 0; i < n; ++i) [be fillBuffer:bufs[i] range:NSMakeRange(a[i], len[i]) value:0];
    } else {
        for (id<MTLBuffer> b in e->pl[1].lk) [be fillBuffer:b range:NSMakeRange(0, b.length) value:0];
        for (id<MTLBuffer> b in e->pl[1].lv) [be fillBuffer:b range:NSMakeRange(0, b.length) value:0];
    }
    if (e->q35) {  // the warm-up's states too
        [be fillBuffer:e->cst range:NSMakeRange(0, (size_t) e->nst * q35_cbytes(e)) value:0];  // the mapped entries (sparse)
        [be fillBuffer:e->sst range:NSMakeRange(0, (size_t) e->nst * q35_sbytes(e)) value:0];
    }
    [be endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// the window pool's cap (e->wcap): what the GPU's working set has left after the engine's buffers, less 2 GiB spare
static void window_pool_cap(sn_engine * e) {
    const uint64_t ws = dev.recommendedMaxWorkingSetSize, used = dev.currentAllocatedSize, spare = 2ull << 30;
    const uint64_t wpage = (uint64_t) e->wunits * 128 * 2;
    e->wcap = (int) MIN((uint64_t) e->pl[0].n, e->pl[1].n + (used + spare < ws ? (ws - used - spare) / wpage : 0));
    if (getenv("SN_VERBOSE")) fprintf(stderr, "sn: window pool %d pages, up to %d (working set %.1f GiB, in use %.1f GiB, a page %.1f MiB)\n",
                                      e->pl[1].n, e->wcap, ws / 1073741824.0, used / 1073741824.0, wpage / 1048576.0);
}

// the device, the queue, the SN_* switches, the profiler and (the process's first engine) the kernels' library: false with
// err set when the kernels do not compile
static bool metal_init(sn_engine * e, const char * kernels, char * err, int err_len) {
    dev = MTLCreateSystemDefaultDevice();
    e->queue = [dev newCommandQueue];
    MTLCompileOptions * opt = [MTLCompileOptions new];
    opt.mathMode = MTLMathModeSafe;
    opt.languageVersion = MTLLanguageVersion4_0;
    NSError * nerr = nil;
    env_switches();
    prof_init(&nerr);
    if (!pso) {  // a second engine in the process (a draft model) shares the library and the pipelines built
        // pipelines are built on first use (core.h K): only the kernels this model's steps take, not every instance in
        // the library (each one's GPU code stays in memory)
        g_lib = [dev newLibraryWithSource:@(kernels) options:opt error:&nerr];
        if (!g_lib) { set_err(err, err_len, "kernels: %s", nerr.localizedDescription.UTF8String); return false; }
        pso = [NSMutableDictionary new];
    }
    return true;
}

// the family's weights, cache and buffers of its own (on failure each sets err and sn_open frees e)
static int family_open(sn_engine * e, const char * path, char * err, int err_len) {
    const gguf_t * g = &e->g;
    const int G = g->n_kv ? g->n_head / g->n_kv : 0;  // query heads per K/V head (model_check took 2, 4 or 8)
    return e->q35 ? q35_open(e, path, err, err_len) : !e->gemma ? dense_open(e, path, G, err, err_len) : gemma_open(e, path, err, err_len);
}

// pool 0's K and V buffers were made: Metal returns nil past its largest buffer: say so instead of a SIGSEGV in fillBuffer
static bool kv_made(const sn_engine * e, char * err, int err_len) {
    if (e->pl[0].kc && e->pl[0].vc) return true;
    char m[200];
    snprintf(m, sizeof m, "cannot allocate the KV cache (%d pages): above the GPU's largest buffer (%llu bytes); use fewer --kv-pages",
             e->pl[0].n, (unsigned long long) dev.maxBufferLength);
    set_err(err, err_len, "%s", m);
    return false;
}

// warm-up: the first steps map the weights for the GPU (about 20 ms off the first request); the caller zeroes the cache
static bool warm_up(sn_engine * e, char * err, int err_len) {
    const int32_t wt[8] = {2, 2, 2, 2, 2, 2, 2, 2}, ws[8] = {0}, wp[8] = {0, 1, 2, 3, 4, 5, 6, 7}, wo = 7, w1 = 8, z = 0;
    int32_t out;
    return !(sn_step(e, 8, wt, ws, wp, 0, 0, 1, &wo, &out, 0, err, err_len) || sn_step(e, 1, wt, ws, &w1, 1, 1, 1, &z, &out, 0, err, err_len));
}

sn_engine * sn_open(const char * path, const char * kernels, int slots, int ctx, int max_rows, int pages, char * err, int err_len) {
    @autoreleasepool {
        sn_engine * e = calloc(1, sizeof(sn_engine));
        if (access(path, R_OK) != 0) { set_err(err, err_len, "cannot read %s", path); free(e); return NULL; }
        gguf_open(path, &e->g);
        gguf_t * g = &e->g;
        if (model_check(g, path, err, err_len)) { munmap((void *) g->base, g->size); free(g->t); free(e); return NULL; }
        arch_flags(e);
        engine_sizes(e, slots, ctx, max_rows, pages);
        if (!metal_init(e, kernels, err, err_len) || family_open(e, path, err, err_len)) { free(e); return NULL; }
        common_buffers(e);
        g_eps = g->eps; M_E_ROWS = e->E;
        if (e->pl[0].kc && e->pl[0].vc) kv_sparse_init(e);
        if (!kv_made(e, err, err_len)) { free(e); return NULL; }
        pages_reset(e);
        if (!warm_up(e, err, err_len)) { free(e); return NULL; }
        zero_caches(e);
        pages_reset(e);  // the warm-up's pages back, every page zero and free
        if (e->npool == 2) window_pool_cap(e);  // the window pool may grow into what the GPU's working set has left (2 GiB spare for the steps' buffers)
        return e;
    }
}

int sn_n_vocab(const sn_engine * e) { return e->V; }
int sn_n_ctx_train(const sn_engine * e) { return e->ctx; }
const char * sn_arch(const sn_engine * e) { return e->g.arch; }
double sn_last_gpu_ms(const sn_engine * e) { return e->gpu_ms; }
