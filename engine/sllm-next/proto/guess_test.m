// The guess head against real answers: each line of SEQS.txt is "NP t0 t1 ..." (a prompt of NP tokens, then the model's own
// greedy answer). The prompt is read, then the answer is written one row a step with its own tokens; before each step the
// head guesses the next K tokens (one sn_guess call) and the guesses are scored against the answer. Prints how often guess
// 1 is right, guess 2 when 1 was, guess 3 when 1 and 2 were, the tokens a draft step of 1 .. K guesses would keep, the GPU
// ms of a guess, and how many of the model's own tokens differ from the answer (0 expected: guessing never changes them).
// usage: guess_test MODEL KERNELS SEQS.txt [K=3] [MAXSEQ]
#import <Foundation/Foundation.h>
#include "sn.h"
#include <time.h>
static double now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }
static char err[512];
#define CK(x) do { if ((x) != 0) { fprintf(stderr, "%s: %s\n", #x, err); exit(1); } } while (0)
int main(int argc, char ** argv) {
    @autoreleasepool {
        NSString * src = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
        const int K = argc > 4 ? atoi(argv[4]) : 3, MAXSEQ = argc > 5 ? atoi(argv[5]) : 1 << 30;
        sn_engine * e = sn_open(argv[1], src.UTF8String, 4, 8192, 1024, 0, err, sizeof err);
        if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
        if (!sn_guess_head(e) || !sn_guess_on(e, 1)) { fprintf(stderr, "no guess head in this GGUF\n"); return 1; }
        static int32_t tok[1024], seq[1024], pos[1024];
        long n[4] = {0}, ok[4] = {0}, kept[4] = {0}, steps = 0, bad = 0;
        double gms = 0, wms = 0; long gn = 0, nseq = 0;
        NSArray * lines = [[NSString stringWithContentsOfFile:@(argv[3]) encoding:NSUTF8StringEncoding error:nil] componentsSeparatedByString:@"\n"];
        for (NSString * ln in lines) {
            NSArray * ids = [[ln stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet] componentsSeparatedByString:@" "];
            if (ids.count < 4 || nseq >= MAXSEQ) continue;
            ++nseq;
            const int NP = [ids[0] intValue], C = (int) ids.count - 1;
            int32_t * t = malloc((size_t) C * 4);
            for (int i = 0; i < C; ++i) t[i] = [ids[i + 1] intValue];
            for (int b = 0; b < NP; b += 512) {
                const int m = MIN(512, NP - b);
                for (int i = 0; i < m; ++i) { tok[i] = t[b + i]; seq[i] = 0; pos[i] = b + i; }
                int32_t last = m - 1, o;
                CK(sn_step(e, m, tok, seq, pos, 0, 0, 1, &last, &o, 0, err, sizeof err));
                if (b + m == NP) bad += o != t[NP];
            }
            for (int p = NP; p < C; ++p) {  // token t[p] goes in at position p; the answer's next token is t[p + 1]
                int32_t gs[4], out;
                bool all = true;
                const double w0 = now_ms();
                CK(sn_guess(e, 0, t[p], p, K, gs, err, sizeof err));
                wms += now_ms() - w0;
                gms += sn_last_gpu_ms(e); gn += K;
                for (int j = 1; j <= K && p + j < C; ++j) {
                    if (all) { ++n[j]; if (gs[j - 1] == t[p + j]) ++ok[j]; else all = false; }
                    for (int k = j; k <= K; ++k) if (all) ++kept[k];
                }
                ++steps;
                tok[0] = t[p]; seq[0] = 0; pos[0] = p;
                int32_t r0 = 0;
                CK(sn_step(e, 1, tok, seq, pos, 1, 1, 1, &r0, &out, 0, err, sizeof err));
                if (p + 1 < C) bad += out != t[p + 1];
            }
            CK(sn_clear(e, 0, 0, C, err, sizeof err));
            free(t);
        }
        printf("%ld answers, %ld write steps; the model's own tokens that differ from the answer: %ld\n", nseq, steps, bad);
        for (int j = 1; j <= K; ++j) printf("guess %d right (when the ones before were): %ld of %ld = %.1f%%\n", j, ok[j], n[j], n[j] ? 100.0 * ok[j] / n[j] : 0);
        for (int k = 1; k <= K; ++k) printf("a step with %d guesses keeps %.2f tokens\n", k, 1.0 + (double) kept[k] / steps);
        printf("a guess: gpu %.2f ms; a call of %d guesses: wall %.2f ms\n", gn ? gms / gn : 0, K, steps ? wms / steps : 0);
        sn_close(e);
    }
    return 0;
}
