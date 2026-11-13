// One interactive writer through the library with the server's prompt lookup (next_engine.rs + spec.rs: k 7 starting at
// 2, n 3, min_keep 0.2, wide checks from 4 rows, the plain step queued unless the next lookup can match), writing a GIVEN
// text: each row's token is the text's, not the argmax (a step's time does not depend on its tokens), so two engines'
// speeds can be had on the same text. Prints tok/s from the first token to the last (as srv_one.py), steps, drafts, kept,
// and how many rows' argmax differed from the text.
// usage: forced_bench MODEL KERNELS TOKS.json PROMPT_KEY TEXT_KEY [REPS]   (spec_toks.py's JSON: doc / doc_out ...)
#import <Foundation/Foundation.h>
#include "sn.h"
#include <time.h>

static double now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }
static char err[512];
#define CK(x) do { if ((x) != 0) { fprintf(stderr, "%s: %s\n", #x, err); exit(1); } } while (0)

enum { K = 7, N = 3, START = 2 };
static const float MIN_KEEP = 0.2f;

typedef struct { int k_cur, miss, cooldown, steps, gated; float keep; } spec_t;

// spec.rs propose: the up to k tokens after the most recent earlier occurrence of ctx's last N tokens
static int propose(spec_t * s, const int32_t * ctx, int len, int32_t * d) {
    if (len < N + 1) return 0;
    int found = -1;
    for (int e = len - 2; e >= N - 1 && found < 0; --e)
        if (!memcmp(ctx + e + 1 - N, ctx + len - N, N * 4)) found = e;
    if (s->cooldown > 0) { --s->cooldown; return 0; }
    if (found < 0) return 0;
    const int k = s->k_cur ? MIN(s->k_cur, K) : K;
    int n = 0;
    for (int i = found + 1; i < len && n < k; ++i) d[n++] = ctx[i];
    return n;
}

static void record(spec_t * s, int prop, int acc) {
    if (!prop) return;
    const int cur = s->k_cur ? MIN(s->k_cur, K) : K;
    const float obs = (float) acc / prop;
    s->keep = s->steps == 0 ? obs : 0.75f * s->keep + 0.25f * obs;
    if (++s->steps >= 3 && s->keep < MIN_KEEP) {
        s->cooldown = MAX(s->cooldown, 16 << MIN(s->gated, 3));
        ++s->gated;
        s->keep = MIN_KEEP;
    }
    if (acc == 0) {
        ++s->miss;
        s->k_cur = MAX(cur / 2, 1);
        if (s->miss >= 3) s->cooldown = 1 << MIN(s->miss - 2, 5);
    } else {
        s->miss = 0;
        s->k_cur = acc >= prop ? MIN(cur + 1, K) : acc * 2 < prop ? MAX(cur / 2, 1) : cur;
    }
}

// spec.rs may_match_next: the bigram (cells' last, last) occurs earlier with a token after it
static bool may_match_next(const int32_t * cells, int len, int32_t last) {
    for (int e = 1; e < len; ++e)
        if (cells[e] == last && cells[e - 1] == cells[len - 1]) return true;
    return false;
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 6) { fprintf(stderr, "usage: forced_bench MODEL KERNELS TOKS.json PROMPT_KEY TEXT_KEY [REPS]\n"); return 2; }
        NSString * src = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
        NSDictionary * js = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[3])] options:0 error:nil];
        NSArray * pr = js[@(argv[4])], * tx = js[@(argv[5])];
        const int P = (int) pr.count, T = (int) tx.count, reps = argc > 6 ? atoi(argv[6]) : 3;
        int32_t * ctx = malloc((P + T + 8) * 4), tok[1024], seq[1024], pos[1024], rows[8] = {0, 1, 2, 3, 4, 5, 6, 7}, out[8];
        for (int i = 0; i < P; ++i) ctx[i] = [pr[i] intValue];
        for (int i = 0; i < T; ++i) ctx[P + i] = [tx[i] intValue];
        sn_engine * e = sn_open(argv[1], src.UTF8String, 4, 4096, 1024, 0, err, sizeof err);
        if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
        if (getenv("NOEND")) {  // NOEND=1: the slot skips Gemma's end-of-generation tokens (the server's ignore_eos)
            const int32_t eog[2] = {1, 106};
            sn_set_eog(e, eog, 2);
            sn_no_end(e, 0, 1);
        }
        for (int rep = 0; rep < reps; ++rep) {
            for (int b = 0; b < P; b += 512) {  // the prompt
                const int n = MIN(512, P - b);
                for (int i = 0; i < n; ++i) { tok[i] = ctx[b + i]; seq[i] = 0; pos[i] = b + i; }
                int32_t last = n - 1;
                CK(sn_step(e, n, tok, seq, pos, 0, 0, b + n == P, &last, out, 0, err, sizeof err));
            }
            spec_t s = {0};
            int i = 1, steps = 0, drafts = 0, drafted = 0, kept = 0, differ = 0;  // i: tokens of the text out (the first from the prompt)
            const double t0 = now_ms();
            while (i < T) {
                const int p0 = P + i - 1;  // the pending token ctx[p0]'s cell
                if (!s.k_cur) s.k_cur = START;
                int32_t d[K];
                int k = getenv("NODRAFT") ? 0 : propose(&s, ctx, p0 + 1, d);  // NODRAFT=1: plain steps, each queued
                k = MIN(k, T - i - 1);
                const int R = k + 1;
                for (int r = 0; r < R; ++r) { tok[r] = r ? d[r - 1] : ctx[p0]; seq[r] = 0; pos[r] = p0 + r; }  // [pending, drafts...]
                const bool chain = k == 0 && i + 1 < T && (getenv("NODRAFT") || !(s.cooldown == 0 && may_match_next(ctx, p0, ctx[p0])));
                CK(sn_step(e, R, tok, seq, pos, R, k ? (R >= 4 ? -R : R) : 1, R, rows, out, chain, err, sizeof err));
                int acc = 0;
                while (acc < k && d[acc] == ctx[P + i + acc]) ++acc;
                for (int r = 0; r <= acc; ++r) {
                    differ += out[r] != ctx[P + i + r];
                    if (getenv("SHOW") && rep == 0 && out[r] != ctx[P + i + r]) printf("  token %d: step of %d rows, row %d: argmax %d, text %d\n", i + r, R, r, out[r], ctx[P + i + r]);
                }
                if (acc < k) CK(sn_clear(e, 0, p0 + acc + 1, p0 + R, err, sizeof err));
                if (k) { record(&s, k, acc); ++drafts; drafted += k; kept += acc; }
                ++steps;
                i += acc + 1;
            }
            const double ms = now_ms() - t0;
            printf("rep %d: %.1f tok/s (%d tokens in %.0f ms), %d steps, %d drafts, %d drafted, %d kept, %d rows' argmax not the text's\n",
                   rep, (T - 1) / ms * 1e3, T - 1, ms, steps, drafts, drafted, kept, differ);
            CK(sn_clear(e, 0, 0, P + T, err, sizeof err));
        }
        sn_close(e);
    }
    return 0;
}
