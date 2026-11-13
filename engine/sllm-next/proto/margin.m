// Greedy tokens of a prompt (REF.json prompts[i], its first C tokens) with each step's top-2 logit gap: where another
// engine's tokens part from these, a small gap is a near tie, a large one a bug (ONE=1: the prompt read a token at a time).
// usage: margin MODEL KERNELS REF.json I C N
#import <Foundation/Foundation.h>
#include "sn.h"

static char err[512];
#define CK(x) do { if ((x) != 0) { fprintf(stderr, "%s: %s\n", #x, err); exit(1); } } while (0)

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 7) { fprintf(stderr, "usage: margin MODEL KERNELS REF.json I C N\n"); return 2; }
        NSString * src = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
        NSArray * doc = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[3])] options:0 error:nil][@"prompts"][atoi(argv[4])];
        const int C = MIN(atoi(argv[5]), (int) doc.count), N = atoi(argv[6]);
        sn_engine * e = sn_open(argv[1], src.UTF8String, 4, 4096, 512, 0, err, sizeof err);
        if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
        const int V = sn_n_vocab(e);
        float * lg = malloc((size_t) V * 4);
        int32_t tok[512], seq[512], pos[512], out = 0;
        const int step = getenv("ONE") ? 1 : getenv("STEP") ? atoi(getenv("STEP")) : 512;  // ONE: a token at a time, as interactive rows (the mat-vec path)
        for (int b = 0; b < C; b += step) {
            const int n = MIN(step, C - b);
            for (int i = 0; i < n; ++i) { tok[i] = [doc[b + i] intValue]; seq[i] = 0; pos[i] = b + i; }
            int32_t last = n - 1;
            const int nw = step == 1 || (getenv("WRITE") && n <= 4) ? n : 0;  // WRITE (steps of up to 4): the rows as write rows
            CK(sn_step(e, n, tok, seq, pos, nw, step == 1, 1, &last, &out, 0, err, sizeof err));
        }
        printf("{\"tokens\": [");
        for (int k = 0; k < N; ++k) {
            CK(sn_logits(e, 0, lg, err, sizeof err));
            if (k == 0 && getenv("DUMP")) { FILE * f = fopen(getenv("DUMP"), "wb"); fwrite(lg, 4, V, f); fclose(f); }  // DUMP: the first scores
            int a = 0, b = -1;
            for (int i = 1; i < V; ++i) if (lg[i] > lg[a]) a = i;
            for (int i = 0; i < V; ++i) if (i != a && (b < 0 || lg[i] > lg[b])) b = i;
            printf("%s[%d, %.4f]", k ? ", " : "", out, lg[a] - lg[b]);
            if (a != out) fprintf(stderr, "step %d: argmax %d, sampled %d\n", k, a, out);
            tok[0] = out; seq[0] = 0; pos[0] = C + k;
            int32_t z = 0;
            CK(sn_step(e, 1, tok, seq, pos, 1, 1, 1, &z, &out, 0, err, sizeof err));
        }
        printf("]}\n");
        sn_close(e);
    }
    return 0;
}
