// Draft rows against one-row steps, bit for bit: a sequence reads C tokens of a text and writes its next N one row a step
// (the scores of every step kept); then again in steps of 2 .. 4 interactive rows (sn_draft_rows first), each cut back to
// 1 .. R of its rows (sn_cut_ok, sn_cut_best, sn_clear) before the next. Every row's scores must equal the one-row step's.
// Prints the rows checked and how many differ, and the mean GPU ms of a step by its rows. Exit 1 on a difference.
// usage: draft_test MODEL KERNELS REF.json [C] [N]
#import <Foundation/Foundation.h>
#include "sn.h"
#include <string.h>
#include <time.h>
static double now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }
static double g_cut_ms; static int g_cuts;

static char err[512];
#define CK(x) do { if ((x) != 0) { fprintf(stderr, "%s: %s\n", #x, err); exit(1); } } while (0)

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 4) { fprintf(stderr, "usage: draft_test MODEL KERNELS REF.json [C] [N]\n"); return 2; }
        NSString * src = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
        NSArray * doc = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[3])] options:0 error:nil][@"prompts"][0];
        const int C = argc > 4 ? atoi(argv[4]) : 300, N = argc > 5 ? atoi(argv[5]) : 48;
        if ((int) doc.count < C + N + 4) { fprintf(stderr, "the text is too short\n"); return 2; }
        sn_engine * e = sn_open(argv[1], src.UTF8String, 4, 4096, 1024, 0, err, sizeof err);
        if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
        const int V = sn_n_vocab(e);
        static int32_t tok[1024], seq[1024], pos[1024];
        int32_t orow[4] = {0, 1, 2, 3}, out[4];
        float * ref = malloc((size_t) N * V * 4), * got = malloc((size_t) V * 4);
        int32_t * rtok = malloc((size_t) N * 4);
        void (^prompt)(void) = ^{
            for (int i = 0; i < C; ++i) { tok[i] = [doc[i] intValue]; seq[i] = 0; pos[i] = i; }
            int32_t last = C - 1, o;
            CK(sn_step(e, C, tok, seq, pos, 0, 0, 1, &last, &o, 0, err, sizeof err));
        };
        prompt();
        double ms[5] = {0}; int cnt[5] = {0};
        for (int i = 0; i < N; ++i) {
            tok[0] = [doc[C + i] intValue]; seq[0] = 0; pos[0] = C + i;
            CK(sn_step(e, 1, tok, seq, pos, 1, 1, 1, orow, out, 0, err, sizeof err));
            CK(sn_logits(e, 0, ref + (size_t) i * V, err, sizeof err));
            rtok[i] = out[0];
            ms[1] += sn_last_gpu_ms(e); ++cnt[1];
        }
        CK(sn_clear(e, 0, 0, C + N, err, sizeof err));
        prompt();
        const int have = sn_draft_rows(e, 0, 3);
        printf("%s: draft rows %d\n", sn_recurrent(e) ? "recurrent" : "not recurrent", have);
        if (have < 3) { fprintf(stderr, "no spare state entries\n"); return 1; }
        static const int plan[9][2] = {{2, 1}, {2, 2}, {3, 1}, {3, 2}, {3, 3}, {4, 1}, {4, 2}, {4, 3}, {4, 4}};
        int i = 0, rows = 0, bad = 0, badtok = 0, cuts = 0;
        for (int it = 0; i + 4 <= N; ++it) {
            const int R = plan[it % 9][0], a = plan[it % 9][1];
            for (int j = 0; j < R; ++j) { tok[j] = [doc[C + i + j] intValue]; seq[j] = 0; pos[j] = C + i + j; }
            CK(sn_step(e, R, tok, seq, pos, R, R, R, orow, out, 0, err, sizeof err));
            ms[R] += sn_last_gpu_ms(e); ++cnt[R];
            for (int j = 0; j < R; ++j) {
                CK(sn_logits(e, j, got, err, sizeof err));
                ++rows;
                if (memcmp(got, ref + (size_t) (i + j) * V, (size_t) V * 4)) { ++bad; if (bad <= 5) printf("  differs: step %d (rows %d), row %d, text token %d\n", it, R, j, i + j); }
                badtok += out[j] != rtok[i + j];
            }
            if (a < R) {
                if (!sn_cut_ok(e, 0, C + i + a) || sn_cut_best(e, 0, C + i + a) != C + i + a) { fprintf(stderr, "cut to row %d of %d refused\n", a, R); return 1; }
                { const double w0 = now_ms(); CK(sn_clear(e, 0, C + i + a, C + i + R, err, sizeof err)); g_cut_ms += now_ms() - w0; ++g_cuts; }
                ++cuts;
            }
            i += a;
        }
        printf("rows checked %d (cuts %d): %d differ, %d other tokens\n", rows, cuts, bad, badtok);
        if (g_cuts) printf("  a cut back: wall %.2f ms\n", g_cut_ms / g_cuts);
        for (int r = 1; r <= 4; ++r) if (cnt[r]) printf("  %d rows: gpu %.2f ms a step (%d steps)\n", r, ms[r] / cnt[r], cnt[r]);
        if (sn_recurrent(e)) {  // without the spare entries a cut into a step stays refused
            sn_draft_rows(e, 0, 0);
            for (int j = 0; j < 2; ++j) { tok[j] = [doc[C + i + j] intValue]; seq[j] = 0; pos[j] = C + i + j; }
            CK(sn_step(e, 2, tok, seq, pos, 2, 2, 2, orow, out, 0, err, sizeof err));
            if (sn_cut_ok(e, 0, C + i + 1)) { fprintf(stderr, "a cut without spare entries was allowed\n"); return 1; }
        }
        sn_close(e);
        return bad != 0;
    }
}
