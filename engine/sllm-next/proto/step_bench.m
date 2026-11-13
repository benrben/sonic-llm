// One agent's write step by its row count: a sequence holds a context of C tokens, then steps of R = 1 .. 4 interactive
// rows at its next positions (a draft step's verify rows: one sequence, consecutive positions, every row sampled) are
// timed, GPU and wall, the cells cut back after each (a recurrent model, which cannot cut back: the steps go on from
// there instead). EXACT=1 (non-recurrent models): the last row's scores of an R-row step against R one-row steps, bit for bit.
// DUMP=PREFIX: each R-row step's scores to PREFIX.R (compare two kernel builds). SEQS=N (with BG): N sequences each hold
// the context, a step's rows go round them (8 rows of 8 sequences: 8 agents writing); SLOTS: the engine's (default 4).
// SOAK=S (with BG, SEQS): steps of SEQS rows for S seconds, the GPU time of each 5 s (does the step slow as the Mac heats?).
// usage: step_bench MODEL KERNELS REF.json [C]
#import <Foundation/Foundation.h>
#include "sn.h"
#include <time.h>
#include <math.h>

static double now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }
static char err[512];
#define CK(x) do { if ((x) != 0) { fprintf(stderr, "%s: %s\n", #x, err); exit(1); } } while (0)

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 4) { fprintf(stderr, "usage: step_bench MODEL KERNELS REF.json [C]\n"); return 2; }
        NSString * src = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
        NSArray * doc = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[3])] options:0 error:nil][@"prompts"][0];
        const int C = MIN(argc > 4 ? atoi(argv[4]) : 1024, (int) doc.count);
        sn_engine * e = sn_open(argv[1], src.UTF8String, getenv("SLOTS") ? atoi(getenv("SLOTS")) : 4, 4096, 1024, 0, err, sizeof err);
        if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
        int32_t tok[1024], seq[1024], pos[1024], orow[8] = {0, 1, 2, 3, 4, 5, 6, 7}, out[8];
        const int V = sn_n_vocab(e);
        float * l1 = malloc((size_t) V * 4), * l2 = malloc((size_t) V * 4);
        double pre_gpu = 0, pre_wall = 0;
        const int NS = getenv("SEQS") ? atoi(getenv("SEQS")) : 1;
        for (int b = 0; b < NS * C; b += 512) {  // the context, in prompt steps
            const int n = MIN(512, C - b % C);
            for (int i = 0; i < n; ++i) { tok[i] = [doc[b % C + i] intValue]; seq[i] = b / C; pos[i] = b % C + i; }
            int32_t last = n - 1;
            const double t0 = now_ms();
            CK(sn_step(e, n, tok, seq, pos, 0, 0, 1, &last, out, 0, err, sizeof err));
            pre_wall += now_ms() - t0; pre_gpu += sn_last_gpu_ms(e);
        }
        if (getenv("PREFILL")) {  // PREFILL=N: read the context N more times (cut back each time), report the best
            double best = pre_gpu;
            for (int it = 0; it < atoi(getenv("PREFILL")); ++it) {
                CK(sn_clear(e, 0, 0, C, err, sizeof err));
                double g = 0, w = 0;
                for (int b = 0; b < C; b += 512) {
                    const int n = MIN(512, C - b);
                    for (int i = 0; i < n; ++i) { tok[i] = [doc[b + i] intValue]; seq[i] = 0; pos[i] = b + i; }
                    int32_t last = n - 1;
                    const double t0 = now_ms();
                    CK(sn_step(e, n, tok, seq, pos, 0, 0, 1, &last, out, 0, err, sizeof err));
                    w += now_ms() - t0; g += sn_last_gpu_ms(e);
                }
                if (getenv("EACH")) printf(" %.0f", C / w * 1e3);
                if (g < best) { best = g; pre_wall = w; }
            }
            printf("read %d tokens: gpu %.1f ms (%.0f tok/s), wall %.1f ms (%.0f tok/s)\n", C, best, C / best * 1e3, pre_wall, C / pre_wall * 1e3);
            sn_close(e);
            return 0;
        }
        printf("context %d cells\n", C);
        if (getenv("SOAK")) {
            const double t_end = now_ms() + atof(getenv("SOAK")) * 1e3;
            double t_win = now_ms() + 5e3, gpu = 0;
            int n = 0;
            while (now_ms() < t_end) {
                for (int r = 0; r < NS; ++r) { tok[r] = [doc[r] intValue]; seq[r] = r; pos[r] = C; }
                CK(sn_step(e, NS, tok, seq, pos, NS, 0, NS, orow, out, 0, err, sizeof err));
                gpu += sn_last_gpu_ms(e); ++n;
                for (int q = 0; q < NS; ++q) CK(sn_clear(e, q, C, C + 1, err, sizeof err));
                if (now_ms() >= t_win) { printf(" %.2f", gpu / n); fflush(stdout); gpu = 0; n = 0; t_win += 5e3; }
            }
            printf("\n");
            sn_close(e);
            return 0;
        }
        if (getenv("ROWS")) {  // ROWS=R NFAST=F (SEQS=R): one step of R rows, the first F interactive (mat-vec), the rest background
            // (block matmuls), each row its own sequence: the GPU time; DUMP=PREFIX: every row's scores to PREFIX.R.F
            const int R = atoi(getenv("ROWS")), F = getenv("NFAST") ? atoi(getenv("NFAST")) : 1;
            int32_t orw[1024];
            for (int r = 0; r < R; ++r) orw[r] = r;
            if (getenv("ONCE")) {  // ONCE=PREFIX: one step only (recurrent models cannot cut back), every row's scores to PREFIX.R.F
                for (int r = 0; r < R; ++r) { tok[r] = [doc[r] intValue]; seq[r] = r % NS; pos[r] = C; }
                CK(sn_step(e, R, tok, seq, pos, R, F, R, orw, out, 0, err, sizeof err));
                FILE * f = fopen([NSString stringWithFormat:@"%s.%d.%d", getenv("ONCE"), R, F].UTF8String, "wb");
                for (int r = 0; r < R; ++r) { CK(sn_logits(e, r, l1, err, sizeof err)); fwrite(l1, 4, V, f); }
                fclose(f);
                sn_close(e);
                return 0;
            }
            double gpu = 0, wall = 0, gmin = 1e9;
            const int reps = getenv("REPS") ? atoi(getenv("REPS")) : 60;
            const bool rec = sn_recurrent(e);  // a recurrent model cannot cut back: its steps go on one position at a time
            for (int it = 0; it < reps + 5; ++it) {
                for (int r = 0; r < R; ++r) { tok[r] = [doc[r] intValue]; seq[r] = r % NS; pos[r] = C + (rec ? it : 0); }
                const double t0 = now_ms();
                CK(sn_step(e, R, tok, seq, pos, R, F, R, orw, out, 0, err, sizeof err));
                const double t1 = now_ms();
                if (it >= 5) { gpu += sn_last_gpu_ms(e); wall += t1 - t0; gmin = fmin(gmin, sn_last_gpu_ms(e)); }
                for (int q = 0; q < MIN(NS, R) && !rec; ++q) CK(sn_clear(e, q, C, C + 1, err, sizeof err));
            }
            printf("  %d rows (%d interactive): gpu %.2f ms (min %.2f), wall %.2f ms\n", R, F, gpu / reps, gmin, wall / reps);
            if (getenv("DUMP")) {
                for (int r = 0; r < R; ++r) { tok[r] = [doc[r] intValue]; seq[r] = r % NS; pos[r] = C; }
                CK(sn_step(e, R, tok, seq, pos, R, F, R, orw, out, 0, err, sizeof err));
                FILE * f = fopen([NSString stringWithFormat:@"%s.%d.%d", getenv("DUMP"), R, F].UTF8String, "wb");
                for (int r = 0; r < R; ++r) { CK(sn_logits(e, r, l1, err, sizeof err)); fwrite(l1, 4, V, f); }
                fclose(f);
                for (int q = 0; q < MIN(NS, R); ++q) CK(sn_clear(e, q, C, C + 1, err, sizeof err));
            }
            sn_close(e);
            return 0;
        }
        const bool rec = sn_recurrent(e);
        int at = C;  // a recurrent model's next position
        for (int R = 1; R <= (getenv("BG") || getenv("WIDEB") ? 8 : 4); ++R) {
            double gpu = 0, wall = 0;
            const int reps = 40;
            for (int it = 0; it < reps + 5; ++it) {
                for (int r = 0; r < R; ++r) { tok[r] = [doc[r] intValue]; seq[r] = r % NS; pos[r] = (rec ? at : C) + r / NS; }
                const double t0 = now_ms();
                CK(sn_step(e, R, tok, seq, pos, R, getenv("BG") ? 0 : getenv("WIDEB") ? -R : R, R, orow, out, 0, err, sizeof err));  // BG: background rows (block matmuls)
                const double t1 = now_ms();
                if (it >= 5) { gpu += sn_last_gpu_ms(e); wall += t1 - t0; }
                if (rec) at += R;
                else for (int q = 0; q < MIN(NS, R); ++q) CK(sn_clear(e, q, C, C + (R + NS - 1) / NS, err, sizeof err));
            }
            printf("  %d row%s: gpu %.2f ms, wall %.2f ms", R, R > 1 ? "s" : " ", gpu / reps, wall / reps);
            if (getenv("EXACT") && !rec) {
                for (int r = 0; r < R; ++r) { tok[r] = [doc[r] intValue]; seq[r] = 0; pos[r] = C + r; }
                CK(sn_step(e, R, tok, seq, pos, R, getenv("WIDEB") ? -R : R, R, orow, out, 0, err, sizeof err));
                CK(sn_logits(e, R - 1, l1, err, sizeof err));
                CK(sn_clear(e, 0, C, C + R, err, sizeof err));
                for (int r = 0; r < R; ++r) {
                    int32_t t1 = [doc[r] intValue], s1 = 0, p1 = C + r, o0 = 0;
                    CK(sn_step(e, 1, &t1, &s1, &p1, 1, 1, 1, &o0, out, 0, err, sizeof err));
                }
                CK(sn_logits(e, 0, l2, err, sizeof err));
                CK(sn_clear(e, 0, C, C + R, err, sizeof err));
                printf(", last row's scores %s one-row steps'", memcmp(l1, l2, (size_t) V * 4) ? "DIFFER from" : "= the");
                if (getenv("WIDEB")) {  // a wide check is not bitwise: how far, and the same top token?
                    float dmax = 0, smax = 0;
                    int a1 = 0, a2 = 0;
                    for (int i = 0; i < V; ++i) {
                        dmax = fmaxf(dmax, fabsf(l1[i] - l2[i])); smax = fmaxf(smax, fabsf(l2[i]));
                        if (l1[i] > l1[a1]) a1 = i;
                        if (l2[i] > l2[a2]) a2 = i;
                    }
                    printf(" (largest difference %.4f of scores up to %.1f, top token %s)", dmax, smax, a1 == a2 ? "same" : "DIFFERENT");
                }
            }
            if (getenv("DUMP") && !rec) {  // DUMP=PREFIX: every row's scores of an R-row step to PREFIX.R (two kernel builds: cmp)
                for (int r = 0; r < R; ++r) { tok[r] = [doc[r] intValue]; seq[r] = 0; pos[r] = C + r; }
                CK(sn_step(e, R, tok, seq, pos, R, getenv("BG") ? 0 : getenv("WIDEB") ? -R : R, R, orow, out, 0, err, sizeof err));
                FILE * f = fopen([NSString stringWithFormat:@"%s.%d", getenv("DUMP"), R].UTF8String, "wb");
                for (int r = 0; r < R; ++r) { CK(sn_logits(e, r, l1, err, sizeof err)); fwrite(l1, 4, V, f); }
                fclose(f);
                CK(sn_clear(e, 0, C, C + R, err, sizeof err));
            }
            printf("\n");
        }
        sn_close(e);
    }
    return 0;
}
