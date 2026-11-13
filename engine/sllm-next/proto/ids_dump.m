// A measurement: read token sequences (one a line of SEQS.txt, ids separated by spaces) as prompts, one after another in slot
// 0, with SN_IDSLOG set so sn.m logs every step's expert ids. usage: SN_IDSLOG=OUT ids_dump MODEL KERNELS SEQS.txt [STEP]
#import <Foundation/Foundation.h>
#include "sn.h"
static char err[512];
#define CK(x) do { if ((x) != 0) { fprintf(stderr, "%s: %s\n", #x, err); exit(1); } } while (0)
int main(int argc, char ** argv) {
    @autoreleasepool {
        NSString * src = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
        const int STEP = argc > 4 ? atoi(argv[4]) : 512;
        sn_engine * e = sn_open(argv[1], src.UTF8String, 4, 8192, 1024, 0, err, sizeof err);
        if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
        static int32_t tok[1024], seq[1024], pos[1024];
        NSArray * lines = [[NSString stringWithContentsOfFile:@(argv[3]) encoding:NSUTF8StringEncoding error:nil] componentsSeparatedByString:@"\n"];
        for (NSString * ln in lines) {
            NSArray * ids = [[ln stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet] componentsSeparatedByString:@" "];
            const int C = (int) ids.count;
            if (C < 2) continue;
            for (int b = 0; b < C; b += STEP) {
                const int n = MIN(STEP, C - b);
                for (int i = 0; i < n; ++i) { tok[i] = [ids[b + i] intValue]; seq[i] = 0; pos[i] = b + i; }
                int32_t last = n - 1, out;
                CK(sn_step(e, n, tok, seq, pos, 0, 0, 1, &last, &out, 0, err, sizeof err));
            }
            CK(sn_clear(e, 0, 0, C, err, sizeof err));
        }
        sn_close(e);
    }
    return 0;
}
