// A measurement: read token sequences (one a line, ids separated by spaces) 8 rows a step and write every row's hidden state
// after the final norm (n_embd floats) and its greedy next token. usage: hid_dump MODEL KERNELS SEQS.txt OUT.f32 OUT.tok
#import <Foundation/Foundation.h>
#include "sn.h"
static char err[512];
#define CK(x) do { if ((x) != 0) { fprintf(stderr, "%s: %s\n", #x, err); exit(1); } } while (0)
int main(int argc, char ** argv) {
    @autoreleasepool {
        NSString * src = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
        sn_engine * e = sn_open(argv[1], src.UTF8String, 4, 8192, 1024, 0, err, sizeof err);
        if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
        const int E = sn_n_embd(e);
        int32_t tok[8], seq[8], pos[8], orow[8] = {0, 1, 2, 3, 4, 5, 6, 7}, out[8];
        float * hid = malloc((size_t) 8 * E * 4);
        FILE * fh = fopen(argv[4], "wb"), * ft = fopen(argv[5], "wb");
        NSArray * lines = [[NSString stringWithContentsOfFile:@(argv[3]) encoding:NSUTF8StringEncoding error:nil] componentsSeparatedByString:@"\n"];
        for (NSString * ln in lines) {
            NSArray * ids = [[ln stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet] componentsSeparatedByString:@" "];
            const int C = (int) ids.count;
            if (C < 2) continue;
            for (int b = 0; b < C; b += 8) {
                const int n = MIN(8, C - b);
                for (int i = 0; i < n; ++i) { tok[i] = [ids[b + i] intValue]; seq[i] = 0; pos[i] = b + i; }
                CK(sn_latent(e, 0, NULL, NULL, hid, err, sizeof err));
                CK(sn_step(e, n, tok, seq, pos, 0, 0, n, orow, out, 0, err, sizeof err));
                fwrite(hid, 4, (size_t) n * E, fh); fwrite(out, 4, n, ft);
            }
            CK(sn_clear(e, 0, 0, C, err, sizeof err));
        }
        fclose(fh); fclose(ft);
        sn_close(e);
    }
    return 0;
}
