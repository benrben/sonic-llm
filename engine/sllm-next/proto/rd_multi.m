// A measurement: NS prompts (lines of SEQS.txt, each cut to C tokens) read in ONE step of NS x C rows; best GPU ms of 5.
// usage: rd_multi MODEL KERNELS SEQS.txt NS C
#import <Foundation/Foundation.h>
#include "sn.h"
static char err[512];
#define CK(x) do { if ((x) != 0) { fprintf(stderr, "%s: %s\n", #x, err); exit(1); } } while (0)
int main(int argc, char ** argv) {
    @autoreleasepool {
        NSString * src = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
        const int NS = atoi(argv[4]), C = atoi(argv[5]);
        sn_engine * e = sn_open(argv[1], src.UTF8String, 16, 4096, 2048, 0, err, sizeof err);
        if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
        static int32_t tok[2048], seq[2048], pos[2048];
        NSArray * lines = [[NSString stringWithContentsOfFile:@(argv[3]) encoding:NSUTF8StringEncoding error:nil] componentsSeparatedByString:@"\n"];
        int R = 0, s = 0;
        for (NSString * ln in lines) {
            NSArray * ids = [[ln stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet] componentsSeparatedByString:@" "];
            if ((int) ids.count < C) continue;
            for (int i = 0; i < C; ++i) { tok[R] = [ids[i] intValue]; seq[R] = s; pos[R] = i; ++R; }
            if (++s == NS) break;
        }
        if (s < NS) { fprintf(stderr, "only %d prompts of %d tokens\n", s, C); return 1; }
        double best = 1e9;
        for (int it = 0; it < 6; ++it) {
            int32_t last = R - 1, out;
            CK(sn_step(e, R, tok, seq, pos, 0, 0, 1, &last, &out, 0, err, sizeof err));
            if (it) best = MIN(best, sn_last_gpu_ms(e));
            for (int q = 0; q < NS; ++q) CK(sn_clear(e, q, 0, C, err, sizeof err));
        }
        printf("%d prompts x %d tokens in one step: gpu %.1f ms, %.0f tok/s\n", NS, C, best, R / best * 1e3);
        sn_close(e);
    }
    return 0;
}
