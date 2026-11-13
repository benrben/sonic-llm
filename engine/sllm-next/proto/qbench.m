// qbench: the quick bench. One model, every scenario in one process, a few seconds: GPU ms a step (median of QB_REPS),
// tok/s, and a hash of every output row's scores (bits against a saved reference), plus the exactness checks in process.
// Scenarios: read (1,024 prompt tokens, best of 3), w1 (one chat row after them), w5 (a wide draft check of 5 rows), b8 / b32
// (background rows of 8 / 32 sequences of 512 cells), mix (the chat + 31 background), w1_end (w1 again: did the Mac heat?).
// The background sequences share one 512-token prompt's pages (sn_copy; QB_DISTINCT=1: each reads its own, slower).
// Exactness (models that can cut back): chat alone == chat in the mix (with SN_CHATBATCH=0), the mix's background rows == b32's, one background
// writer alone (b1) == b8's first, b8 == b8 beside 64 prompt rows of another sequence (b8p), and b8 == b32's first 8 unless
// SN_SMALL (sn_small_rows: up to 8 writers take the few-row kernel, more the block matmuls). Recurrent models step on one
// position at a time (no cut back) and skip w5, b8p and the exactness checks. read_chunks: the document read in steps of
// 160 rows gives the chat's next row the same scores (a prompt token's cells do not depend on the chunking).
// usage: qbench MODEL KERNELS REF.json   (one JSON line on stdout; env QB_REPS, QB_ONLY=w1,mix)
#import <Foundation/Foundation.h>
#include "sn.h"
#include <time.h>

static char err[512];
static sn_engine * e;
static int V, rec, NREPS;
static int32_t tok[2048], seq[2048], pos[2048], orow[64], out[64];
static int npos[64];  // each slot's next position (a recurrent model's moves on; the others' are cut back to it)
static float * lg;
static NSMutableArray * P;  // the prompt tokens (tiled to 2,048)

static double now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }
static int cmp_d(const void * a, const void * b) { const double x = *(const double *) a, y = *(const double *) b; return x < y ? -1 : x > y; }
static uint64_t fnv(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * b = p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// prompt steps: tokens P[off .. off + n) into slot s at positions 0 .. n
static int read_into(int s, int off, int n, int rows) {
    for (int b = 0; b < n; b += rows) {
        const int k = MIN(rows, n - b);
        for (int i = 0; i < k; ++i) { tok[i] = [P[off + b + i] intValue]; seq[i] = s; pos[i] = b + i; }
        int32_t last = k - 1;
        if (sn_step(e, k, tok, seq, pos, 0, 0, 1, &last, out, 0, err, sizeof err) != 0) return -1;
    }
    return 0;
}

// a write scenario: R rows (row r: slot sl[r]; rows of one slot at its next positions in turn), n_fast as sn_step takes it.
// First one step whose rows' scores are hashed into hs, then the timed steps; cut back after each (not recurrent).
static int run(const int * sl, int R, int n_fast, int reps, uint64_t * hs, double * med, double * mn) {
    double t[256];
    for (int it = 0; it < reps + 4; ++it) {
        for (int r = 0; r < R; ++r) {
            pos[r] = r > 0 && sl[r - 1] == sl[r] ? pos[r - 1] + 1 : npos[sl[r]];
            tok[r] = [P[(sl[r] * 7 + pos[r] - npos[sl[r]]) % 1024] intValue]; seq[r] = sl[r]; orow[r] = r;  // a slot's own tokens
        }
        if (sn_step(e, R, tok, seq, pos, R, n_fast, R, orow, out, 0, err, sizeof err) != 0) return -1;
        if (it == 0 && hs) for (int r = 0; r < R; ++r) { if (sn_logits(e, r, lg, err, sizeof err) != 0) return -1; hs[r] = fnv(lg, (size_t) V * 4); }
        if (it >= 4) t[it - 4] = sn_last_gpu_ms(e);
        for (int r = 0; r < R; ++r) if (r == R - 1 || sl[r + 1] != sl[r]) {  // a slot's last row
            if (rec) npos[sl[r]] = pos[r] + 1;
            else if (sn_clear(e, sl[r], npos[sl[r]], pos[r] + 1, err, sizeof err) != 0) return -1;
        }
    }
    qsort(t, reps, sizeof t[0], cmp_d);
    *med = t[reps / 2]; *mn = t[0];
    return 0;
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 4) { fprintf(stderr, "usage: qbench MODEL KERNELS REF.json\n"); return 2; }
        NSString * src = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
        NSArray * doc = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[3])] options:0 error:nil][@"prompts"][0];
        P = [NSMutableArray array];
        while (P.count < 2048) [P addObjectsFromArray:doc];
        NREPS = getenv("QB_REPS") ? MIN(atoi(getenv("QB_REPS")), 256) : 30;
        const char * only = getenv("QB_ONLY");
        const bool distinct = getenv("QB_DISTINCT") != NULL;
        int rows = 1024;
        if (sn_check(argv[1], &rows, err, sizeof err) != 0) { printf("{\"error\": \"%s\"}\n", err); return 1; }
        const int NB = 32, SRC = NB + 1, S2 = NB + 2, SLOTS = NB + 3, CTX = 1152;
        const int pages = 2 * (CTX / 64 + 2) + (distinct ? NB * 10 : 10 + NB * 2) + 16;
        const double t_open = now_ms();
        e = sn_open(argv[1], src.UTF8String, SLOTS, CTX, rows, pages, err, sizeof err);
        if (!e) { printf("{\"error\": \"open: %s\"}\n", err); return 1; }
        V = sn_n_vocab(e); rec = sn_recurrent(e);
        lg = malloc((size_t) V * 4);
        NSMutableDictionary * res = [NSMutableDictionary dictionary], * hashes = [NSMutableDictionary dictionary], * exact = [NSMutableDictionary dictionary];
        res[@"open_ms"] = @(round(now_ms() - t_open));
        #define WANT(s) (!only || strstr(only, s))
        #define FAIL(s) do { printf("{\"error\": \"%s: %s\"}\n", s, err); return 1; } while (0)
        // read: 1,024 tokens into slot 0, best of 3 (the last read stays: the chat's document)
        double best = 1e9;
        for (int k = 0; k < 3; ++k) {
            if (k > 0 && sn_clear(e, 0, 0, 1024, err, sizeof err) != 0) FAIL("clear");
            double g = 0;
            for (int b = 0; b < 1024; b += rows) {
                const int n = MIN(rows, 1024 - b);
                for (int i = 0; i < n; ++i) { tok[i] = [P[b + i] intValue]; seq[i] = 0; pos[i] = b + i; }
                int32_t last = n - 1;
                if (sn_step(e, n, tok, seq, pos, 0, 0, 1, &last, out, 0, err, sizeof err) != 0) FAIL("read");
                g += sn_last_gpu_ms(e);
            }
            best = MIN(best, g);
        }
        res[@"read"] = @{@"ms": @(best), @"tok_s": @(round(1024 / best * 1e3))};
        // the background sequences: 512 cells each
        if (distinct) {
            for (int s = 1; s <= NB; ++s) if (read_into(s, s * 16, 512, rows) != 0) FAIL("bg read");
        } else {
            if (read_into(SRC, 0, 512, rows) != 0) FAIL("src read");
            for (int s = 1; s <= NB; ++s) if (sn_copy(e, SRC, s, 512, err, sizeof err) != 0) FAIL("copy");
        }
        int sl[64];
        npos[0] = 1024;
        for (int s = 1; s <= NB; ++s) npos[s] = 512;
        uint64_t h_w1[1], h_b1[1], h_b8[8], h_b32[32], h_mix[32], h_w5[5], h_b8p[8];
        const int small = sn_small_rows(e);
        double med, mn;
        #define SCEN(name, R, nf, hs, mul) do { \
            if (run(sl, R, nf, NREPS, hs, &med, &mn) != 0) FAIL(name); \
            res[@name] = @{@"ms": @(med), @"min": @(mn), @"tok_s": @(round((mul) * 1e3 / med))}; \
            NSMutableArray * hx = [NSMutableArray array]; for (int r = 0; r < R; ++r) [hx addObject:[NSString stringWithFormat:@"%016llx", hs[r]]]; \
            hashes[@name] = hx; } while (0)
        sl[0] = 0;
        if (WANT("w1")) SCEN("w1", 1, 1, h_w1, 1);
        if (WANT("w1") && !rec) {  // the same document read in steps of 160 rows: the chat's next row must not change
            if (read_into(S2, 0, 1024, 160) != 0) FAIL("chunked read");
            tok[0] = [P[0] intValue]; seq[0] = S2; pos[0] = 1024; orow[0] = 0;
            if (sn_step(e, 1, tok, seq, pos, 1, 1, 1, orow, out, 0, err, sizeof err) != 0 || sn_logits(e, 0, lg, err, sizeof err) != 0) FAIL("chunked w1");
            exact[@"read_chunks"] = @(fnv(lg, (size_t) V * 4) == h_w1[0]);
            if (sn_clear(e, S2, 0, 1025, err, sizeof err) != 0) FAIL("chunked clear");
        }
        if (WANT("w5") && !rec) { for (int r = 0; r < 5; ++r) sl[r] = 0; SCEN("w5", 5, -5, h_w5, 5); }
        for (int r = 0; r < 32; ++r) sl[r] = r + 1;
        if (WANT("b8")) SCEN("b1", 1, 0, h_b1, 1);
        if (WANT("b8")) SCEN("b8", 8, 0, h_b8, 8);
        if (WANT("b8") && !rec) {  // b8 beside 64 prompt rows of the source sequence (its cells 512 .. 576, cut back after)
            for (int r = 0; r < 8; ++r) { tok[r] = [P[((r + 1) * 7) % 1024] intValue]; seq[r] = r + 1; pos[r] = 512; orow[r] = r; }
            for (int i = 0; i < 64; ++i) { tok[8 + i] = [P[512 + i] intValue]; seq[8 + i] = SRC; pos[8 + i] = 512 + i; }
            if (sn_step(e, 72, tok, seq, pos, 8, 0, 8, orow, out, 0, err, sizeof err) != 0) FAIL("b8p");
            NSMutableArray * hx = [NSMutableArray array];
            for (int r = 0; r < 8; ++r) {
                if (sn_logits(e, r, lg, err, sizeof err) != 0) FAIL("b8p");
                h_b8p[r] = fnv(lg, (size_t) V * 4);
                [hx addObject:[NSString stringWithFormat:@"%016llx", h_b8p[r]]];
            }
            hashes[@"b8p"] = hx;
            for (int r = 0; r < 8; ++r) if (sn_clear(e, r + 1, 512, 513, err, sizeof err) != 0) FAIL("b8p clear");
            if (sn_clear(e, SRC, 512, 576, err, sizeof err) != 0) FAIL("b8p clear");
        }
        if (WANT("b32")) SCEN("b32", 32, 0, h_b32, 32);
        sl[0] = 0;
        for (int r = 1; r < 32; ++r) sl[r] = r;
        if (WANT("mix")) {
            SCEN("mix", 32, 1, h_mix, 1);
            res[@"mix"] = @{@"ms": @(med), @"min": @(mn), @"tok_s": @(round(1e3 / med)), @"bg_tok_s": @(round(31e3 / med))};
        }
        sl[0] = 0;
        if (WANT("w1")) { uint64_t h2[1]; SCEN("w1_end", 1, 1, h2, 1); }
        if (!rec) {
            if (WANT("w1") && WANT("mix") && sn_chat_exact(e)) exact[@"chat_alone_eq_mix"] = @(h_w1[0] == h_mix[0]);
            if (WANT("b8") && WANT("b32") && !small) { bool ok = true; for (int r = 0; r < 8; ++r) ok &= h_b8[r] == h_b32[r]; exact[@"b8_eq_b32"] = @(ok); }
            if (WANT("b8")) {
                exact[@"b1_eq_b8"] = @(h_b1[0] == h_b8[0]);
                bool ok = true;
                for (int r = 0; r < 8; ++r) ok &= h_b8[r] == h_b8p[r];
                exact[@"b8_eq_b8p"] = @(ok);
            }
            if (WANT("b32") && WANT("mix")) { bool ok = true; for (int r = 1; r < 32; ++r) ok &= h_mix[r] == h_b32[r - 1]; exact[@"mix_bg_eq_b32"] = @(ok); }
        }
        NSDictionary * o = @{@"arch": @(sn_arch(e)), @"rows": @(rows), @"recurrent": @(rec), @"small": @(small), @"res": res, @"hash": hashes, @"exact": exact};
        printf("%s\n", [[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:o options:NSJSONWritingSortedKeys error:nil] encoding:NSUTF8StringEncoding].UTF8String);
        sn_close(e);
    }
    return 0;
}
