// sllm-next engine library (sn.h): the step of next.m's bench path as a library for sllm-server, both architectures, with
// write rows and prompt rows in the same step. Steps are synchronous (the caller needs each token before the next step).
#include "core.h"
#include "sn.h"

// a cache pool: pages of 64 cells for a set of layers (layer-major: a layer's pages, then the next layer's; the window pool
// a buffer a layer in lk / lv, so it grows a layer at a time); pt[seq * maxc + chunk] is the page of the sequence's cells
// chunk * 64 .. (-1 none); ref: each page's holders; freel: the free pages (a stack)
typedef struct {
    id<MTLBuffer> kc, vc, pt;
    NSMutableArray<id<MTLBuffer>> * lk, * lv;
    int n, nfree, * ref, * freel;
} pool_t;

// a Gemma / gpt-oss layer's K and V buffers and its region's byte offset in them (pool 0: koff into the pool's buffers)
#define LKB(e, l) ((l)->pool ? (e)->pl[1].lk[(l)->pi] : (e)->pl[0].kc)
#define LVB(e, l) ((l)->pool ? (e)->pl[1].lv[(l)->pi] : (e)->pl[0].vc)
#define LKO(e, l) ((l)->pool ? (size_t) 0 : (size_t) (l)->koff * (e)->pl[0].n * 128)

// a Qwen3.5 layer: a Gated DeltaNet layer (lin; li its index among them, the states') or a full attention layer (li: its
// index among those, the cache's); matrices with their own types
typedef struct {
    bool lin;
    int li;
    uint64_t attn_norm, post_norm, gate, up, down;
    int fmt, down_fmt;
    uint64_t qkv, z, alpha, beta, a, conv, dt, norm, out;
    int f_qkv, f_z, f_ab, f_out;
    uint64_t q, k, v, o, q_norm, k_norm;
    int f_q, f_v, f_o;
    moe_t m;  // an MoE layer's experts
} q35l;

typedef struct { int x, y; } soft_i2;  // (row or output, vector slot) of a soft row

struct sn_engine {
    gguf_t g;
    bool gemma;
    id<MTLCommandQueue> queue;
    int E, L, NH, V, ctx, slots, rmax, maxc, QMAX, KD, FMAX, PLE, PN;
    double gpu_ms;
    // Qwen3
    layer_t * ly;
    uint64_t out_norm, embd, head;  // the token table and the output layer (the same tensor when tied)
    int embd_fmt, head_fmt;          // mm formats: 2 q6_K, 5 q8_0, 0 q4_0, 1 q4_1
    int qk_mode;                     // qk_rope's mode: 0 Qwen3, Llama 1 | 4 (| 2 with frequency factors)
    int nope_every;                  // every this many layers one has no rope (SmolLM3: 4), 0 none
    float mscale;                    // rope's cos / sin factor (Phi-3's rope.scaling.attn_factor), 1 else
    // Qwen3.5 / 3.6: the layers, conv channels, attention / linear layer counts, step buffers; recurrent states in a pool of
    // entries ([entry][linear layer]: conv 3 x C, delta 128 x 128 per v head), a sequence's entry (sidx, -1 none) handed out
    // zeroed and grown on demand; slen: each sequence's cells (a cut keeps a recurrent state only at 0 or its end); runs: a
    // step's rows of one sequence (first row, rows, state entry), rrun a row's place in its run, rsi its state entry
    bool q35;
    q35l * ql;
    int q35C, natt, nlin, nst, nsfree, * sidx, * sfree, * slen, nruns;
    // a checkpoint a sequence (sn_mark): its state at ckpos cells (ckent: an entry of the pool, -1 none), taken only into a
    // free entry and given back first when a sequence needs one, so a cut back near the end of the last prompt (a chat's next
    // turn) keeps the prompt instead of reading it again
    int * ckent, * ckpos;
    // the makers' guess head (MTP; a GGUF with nextn_predict_layers 1 whose block sllm-next can read, else NULL): one more
    // full-attention MoE block (its cache layer: natt), eh_proj (2 E -> E) over the normed token row and hidden state
    q35l * mtp;
    uint64_t mtp_eh, mtp_enorm, mtp_hnorm, mtp_hdnorm;
    int mtp_feh;
    // guessing (sn_guess_on): every step also writes the head's cache cells of its rows: row (token x_q, the model's hidden
    // state after the output norm at q - 1). mhn: the step's hidden states; mpend: each sequence's last one (the next step's
    // first row pairs with it); mhs / mem / mcat / mx: the rows' picked hidden states, token rows, eh_proj's input and
    // output; msrc: a row's hidden-state source (a row of mhn, -1 its sequence's mpend, -2 none); mhg: the last guess's
    // output hidden state (a chained guess feeds on it); drow: the first row of a sequence's draft run in the last step
    // mtab: the token table mapped for the GPU by itself (a guessed token's row is looked up there: the model's own steps
    // stage their rows from the CPU); gpos: the positions of a call's chained guess rows
    int guess_on, * drow;
    id<MTLBuffer> mhn, mpend, mhs, mem, mcat, mx, msrc, mhg, mtab, gpos[4];
    size_t mtab_off;
    // a guess's output layer in small (a q6_K head; SN_GUESS_SKETCH=0: the head itself): the sign of every weight (sk_bits,
    // a bit each) and each row's mean size (sk_rs) rank the vocabulary by chunks (sk_tab: the hidden state's partial sums
    // by byte), the best SK_TOP rows of each of SK_NC chunks are scored with the real weights (sk_cand, sk_sc), the best wins
    id<MTLBuffer> sk_bits, sk_rs, sk_tab, sk_cand, sk_sc;
    // draft rows (sn_draft_rows): a sequence's spare entries (dent: 3 a sequence, -1 none; never zeroed while it holds them, a
    // step writes them whole). A step whose interactive run of 2 .. 4 rows has them writes the state after each row but the
    // last into one (rsnap: a row's entry or -1; csnap: the conv states' items), so the sequence can be cut back to any of
    // those rows: dn states after the rows from position dpos on. The next step or cut of the sequence ends them.
    int * dent, * dn, * dpos, nsnap;
    id<MTLBuffer> rsnapB[2], csnapB[2], rsnap, csnap;
    id<MTLBuffer> qa, co, ab, bg, zb, dout, cst, sst, runsB[2], rrunB[2], rsiB[2], runs, rrun, rsi;
    // Qwen3.5-MoE: router logits, the pairs' experts / weights / list places, the list, tiles, gathered rows (half), the
    // experts' gated units and outputs (batch: in list order; interactive: in pair order), the shared expert's
    // gpt-oss (oss: Gemma's layer machinery, the windows' pool, with its own layer program): the experts' up projections
    // apart (mau interactive, mact2 batch); YaRN's ramp over the rotation pairs ylo .. yhi and its cos / sin factor
    bool moe, oss;
    int ntiles;
    id<MTLBuffer> mlog, mids, mw, mpos, mlist, mtiles, mxg, mact, mxg2, myd, mai, myi, msgt, mshg, mshx, mau, mact2;
    float ylo, yhi, ymscale;
    // core.h's globals, set from these at every call (a draft model and its target live in one process)
    id<MTLBuffer> W, skpart;
    size_t kv_layer;
    // Gemma 3 / 4 (embd, head: the token table and the output layer, head 0: the table's q4_1 copy head41)
    g4layer_t * gl;
    id<MTLBuffer> wb[4], head41;  // (3: the guess head's side file, its offsets from SIDE_BASE)
    uint64_t rb[4][2], rope_ff, pl_proj, pl_norm;
    gguf_t g2;  // the guess head's side file (SN_GUESS_HEAD, or MODEL minus .gguf plus .guess-head.gguf), when the model has no head block
    const uint8_t * ple_tab;
    size_t ple_row;
    bool g4;          // Gemma 4: v RMS-normed (Gemma 3 keeps v as is)
    // a token table only the embeddings read (an untied output layer) stays off the GPU: the CPU copies the step's rows
    // (etab: the table, erow: bytes of a row) into erowB[set] (erow_b: the set being encoded), as Gemma 4 E2B's per-layer rows
    bool emb_cpu;
    const uint8_t * etab;
    size_t erow;
    id<MTLBuffer> erowB[2], erow_b;
    // latent rows (sn_latent), for the next step only: its rows lat_idx[j] take lat_in row j in place of their tokens'
    // embeddings (lat_n of them); lat_dst (when set) receives its output rows' hidden states after the final norm
    int lat_n, lat_cap;
    id<MTLBuffer> lat_in, lat_idx, lat_out, lat_iota;
    float * lat_dst;
    // soft words on the GPU (sn_soft_config): soft_sv holds two vectors a slot (soft_par[slot]: the half with its current one; a step
    // reads that half and writes the other, so a queued step that is dropped leaves it as it was). sp_in / sp_out: the request for
    // the next step (sn_soft_rows: input rows, output rows); soft_inB / soft_outB: what each set's step took, as (row or output,
    // vector slot) pairs for the kernels; soft_nin / soft_nout their counts, soft_oseq the slots its outputs turned over
    int soft_k, soft_neog, soft_eog[32], sp_nin, sp_nout, soft_nin[2], soft_nout[2], * sp_in, * sp_out, * soft_oseq;
    float soft_t;
    uint8_t * soft_par;
    id<MTLBuffer> soft_sv, soft_inB[2], soft_outB[2];
    float emb_norm;   // the mean L2 norm of a token's row as the first layer reads it (0: not measured yet)
    float ascale;    // attention scale: Gemma 4 1 (q is normed), Gemma 3 1 / sqrt(head) (27B: 1 / sqrt(n_embd / n_head))
    int ntt, tt[2];   // the prompt tile lists: tokens a tile (32 / G for the layers with G query heads per K/V head)
    size_t ttb[2];    // the largest qhb tile (bytes) of a list's layers
    // activations and row data
    id<MTLBuffer> x, h, y, q, k, v, ao, fg, xo, ho, logits, rtok, rseq, rpos, ridx, otok, rblk, qhb, apart;
    id<MTLBuffer> praw, pq, pp, ple, pg;
    // the cache in pages of 64 cells, in two pools: 0 every layer of Qwen3 / Llama and Gemma's full layers, 1 Gemma's
    // sliding-window layers, which keep only a sequence's last window of cells (the pool starts at one sequence's need
    // and grows to pool 0's size when more sequences hold windows). win: the window (0: one pool); wlo[seq]: chunks below
    // it hold no window page; wmin: pages_for's scratch
    pool_t pl[2];
    // pool 0's K / V buffers as placement sparse buffers (macOS 26; SN_SPARSE=0: plain ones): memory is mapped in a chunk of
    // kgrow pages at a time (SN_SPARSE_MB, 256 MB of K and V) as pages are first used (kv_commit), so the cache costs what its
    // busiest moment held, not every page it might hold (Qwen3-4B, 238 agents of 1k on one document: 25 GB at start).
    // kc_pages: pages [0, kc_pages) are backed; mq4 / mev: the Metal 4 queue that maps them, and its event; hp[k]: the heap
    // of each mapped chunk by chunk index (0: pool 0's chunks of kgrow pages, 1: the window pool's of wgrow, 2: Qwen3.5's
    // state entries' of stgrow), so sn_trim can give a chunk back
    bool sparse;
    int kc_pages, kgrow;
    size_t tile;
    id mq4;
    id<MTLSharedEvent> mev;
    uint64_t mevv;
    NSMutableDictionary * hp[3];
    NSMutableArray * pend;  // chunks given back whose unmap the Metal 4 queue still runs: @[event value, their heaps]
    // Qwen3.5's state pool (cst / sst) as sparse buffers too: entries mapped stgrow at a time, never copied when it grows
    bool q35sp;
    int stgrow;
    int kvl;  // Gemma: the layers [0, kvl) write the cache, the rest read an earlier layer's (Gemma 4's shared KV)
    // the window pool as sparse buffers too (wsp): a layer's buffer has room for pool 0's pages, mapped wgrow pages at a
    // time from the first wn0 (the pool's start); it grows without a copy
    bool wsp;
    int wgrow, wn0;
    int npool, win, wcap, * wlo, * wmin;  // wcap: the window pool's most pages (the GPU's working set)
    int wback;  // cells behind a sequence's end whose window chunks stay, so a cut back that far holds (SN_WINBACK)
    size_t wunits;  // pool 1: K (or V) elements of one cell over its layers
    // the row data of a step in two sets (rtok .. praw above point at the set being encoded): a write step can be queued
    // while the one before runs. ch: that queued step (set ch_set, -1: none), its rows and the tokens it will be fed
    id<MTLBuffer> rtokB[2], rseqB[2], rposB[2], ridxB[2], otokB[2], rblkB[2], prawB[2];
    // sn_want_probs: the steps encoded while it is on dispatch argmax_p, which also writes each output row's 8-word record
    // (kernels.metal) to pB[set]; pwant[set]: whether set's step has them; finish() copies them to last_probs (last_pn rows: the
    // returned step's; 0: none)
    id<MTLBuffer> pB[2];
    int want_probs, pwant[2], last_pn;
    float * last_probs;
    id<MTLCommandBuffer> cbs[2];
    int lofs[2], lastofs;  // logits row of each set's first output (a queued step of up to 4 outputs never covers the returned
                           // step's scores; a longer one does, and sn_logits refuses those)
    id<MTLSharedEvent> ev;  // Gemma 4: the queued step waits for its per-layer rows (copied once the token is known)
    uint64_t ev_val;
    int ch_set, ch_last, ch_R, ch_fast, * ch_seq, * ch_pos, * ch_tok;  // the queued step's rows (rmax each)
    int neog, eog[8];  // sn_set_eog: the tokens a slot with no_end[slot] never takes
    uint8_t * no_end;
    int q40;  // some layer matrix is Q4_0: small steps take the few-row kernel (sn_small_rows)
};

// SN_PROF=R: the GPU time of each kernel of the steps of R rows (each its own encoder, with timestamps), summed by name
// and printed by sn_close (steps must not be queued behind each other: SLLM_NEXT_CHAIN=0 or a synchronous caller)
static int g_prof_r, g_prof_steps;
static NSMutableDictionary<NSString *, NSNumber *> * g_prof_sum;

// the grouped (MoE) matmuls' row tiles, K slices and pairs a tile (SN_GRT, SN_GG, SN_GNT): 4 slices of tiles of 32 pairs
// measured best on Qwen3.6-35B-A3B's prompt steps (expert matmuls 164 -> 106 ms a 512-row step; 2 row tiles, 8 slices,
// tiles of 16 or 64 slower)
static int g_grt = 1, g_gg = 4, g_gnt = 32;

static void set_err(char * err, int n, const char * fmt, const char * a) {
    if (err && n > 0) snprintf(err, n, fmt, a);
}

// SN_TIME=1: where the host time of a step goes (printed at sn_close)
static int g_time;
static double g_t_enc, g_t_wait, g_t_gpu, g_t_lat, g_t_layers;
static long g_t_n, g_ch_used, g_ch_drop;  // and the queued steps taken / dropped

static int pages_for(sn_engine * e, int R, const int32_t * seq, const int32_t * pos);

// a private (GPU-only) buffer of len bytes
static id<MTLBuffer> priv_buf(size_t len) { return [dev newBufferWithLength:len options:MTLResourceStorageModePrivate]; }

// a buffer of at least `need` bytes (grown by half again when it is short)
static void fit(id<MTLBuffer> __strong * b, size_t need) {
    if (!*b || (*b).length < need) *b = [dev newBufferWithLength:need + need / 2 options:MTLResourceStorageModePrivate];
}

// every sequence without pages, every page free (the cache is zero)
static void q35_reset(sn_engine * e);

static void pages_reset(sn_engine * e) {
    for (int w = 0; w < e->npool; ++w) {
        pool_t * P = &e->pl[w];
        for (int i = 0; i < e->slots * e->maxc; ++i) ((int *) P->pt.contents)[i] = -1;
        for (int i = 0; i < P->n; ++i) { P->ref[i] = 0; P->freel[i] = P->n - 1 - i; }
        P->nfree = P->n;
    }
    for (int s = 0; s < e->slots; ++s) e->wlo[s] = 0;
    if (e->q35) q35_reset(e);
}

static id<MTLBuffer> gb(sn_engine * e, uint64_t off);
static uint64_t go(sn_engine * e, uint64_t off);

static void use(sn_engine * e) {
    // mm's q6_K output layer reads W at g_head: W is the buffer that holds it
    W = e->W; g_head = go(e, e->head); g_eps = e->g.eps; M_E_ROWS = e->E; g_skpart = e->skpart;
}

// the library's parts, in this order (each sees what the ones before define)
#include "sn_check.inc.h"
#include "sn_memory.inc.h"
#include "sn_load.inc.h"
#include "sn_open.inc.h"
#include "sn_attn.inc.h"
#include "sn_layers.inc.h"
#include "sn_state.inc.h"
#include "sn_encode.inc.h"
#include "sn_step.inc.h"
#include "sn_cache.inc.h"
#include "sn_seq.inc.h"
