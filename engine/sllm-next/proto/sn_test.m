// The library on a job of next.m's form: every prompt read then written greedily for n_out tokens, in one of three
// schedules: "solo" (one sequence at a time), "together" (all read, then all write), "staggered" (sequence i starts at
// step 3i, so prompt rows and write rows share steps). usage: sn_test MODEL KERNELS JOB OUT MODE [FAST]
#import <Foundation/Foundation.h>
#include "sn.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 6) { fprintf(stderr, "usage: sn_test MODEL KERNELS JOB OUT MODE [FAST]\n"); return 2; }
        NSString * src = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
        NSDictionary * job = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[3])] options:0 error:nil];
        NSArray * prompts = job[@"prompts"];
        const int n = (int) prompts.count, n_out = [job[@"n_out"] intValue], fast = argc > 6 ? atoi(argv[6]) : 0;
        const char * mode = argv[5];
        int maxp = 0;
        for (NSArray * p in prompts) maxp = MAX(maxp, (int) p.count);
        char err[256];
        sn_engine * e = sn_open(argv[1], src.UTF8String, n, maxp + n_out + 8, 2048, getenv("SN_PAGES") ? atoi(getenv("SN_PAGES")) : 0, err, sizeof err);
        if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
        int * fed = calloc(n, sizeof(int)), * gen = calloc(n, sizeof(int)), * last = calloc(n, sizeof(int)), * start = calloc(n, sizeof(int));
        int ** out = calloc(n, sizeof(int *));
        for (int s = 0; s < n; ++s) { out[s] = calloc(n_out, sizeof(int)); start[s] = !strcmp(mode, "staggered") ? 3 * s : 0; }
        int32_t * tok = malloc(4096 * 4), * seq = malloc(4096 * 4), * pos = malloc(4096 * 4), * orow = malloc(4096 * 4), * otok = malloc(4096 * 4), * oseq = malloc(4096 * 4);
        double gpu = 0; int steps = 0;
        const bool solo = !strcmp(mode, "solo");
        for (int step = 0, cur = 0;; ++step) {
            int R = 0, nw = 0, no = 0, nf = 0;
            bool any = false;
            // write rows first (interactive ones first), then prompt rows
            for (int pass = 0; pass < 2; ++pass)
                for (int s = 0; s < n; ++s) {
                    if (solo && s != cur) continue;
                    const bool inter = s < fast;
                    if ((pass == 0) != inter) continue;
                    if (step >= start[s] && fed[s] == (int) [prompts[s] count] && gen[s] > 0 && gen[s] < n_out) {
                        tok[R] = last[s]; seq[R] = s; pos[R] = fed[s] + gen[s] - 1; orow[no] = R; oseq[no++] = s; ++R; ++nw; nf += inter;
                    }
                }
            for (int s = 0; s < n; ++s) {
                if (solo && s != cur) continue;
                NSArray * p = prompts[s];
                if (step < start[s] || fed[s] == (int) p.count) continue;
                const int take = MIN((int) p.count - fed[s], 2048 - R);
                if (take <= 0) break;
                for (int j = 0; j < take; ++j, ++R) { tok[R] = [p[fed[s] + j] intValue]; seq[R] = s; pos[R] = fed[s] + j; }
                fed[s] += take;
                if (fed[s] == (int) p.count) { orow[no] = R - 1; oseq[no++] = s; }
            }
            for (int s = 0; s < n; ++s) any |= gen[s] < n_out;
            if (!any) break;
            if (R == 0) { if (solo) ++cur; continue; }
            if (sn_step(e, R, tok, seq, pos, nw, nf, no, orow, otok, getenv("SN_CHAIN") != NULL, err, sizeof err)) { fprintf(stderr, "step: %s\n", err); return 1; }
            gpu += sn_last_gpu_ms(e); ++steps;
            for (int j = 0; j < no; ++j) { const int s = oseq[j]; out[s][gen[s]++] = otok[j]; last[s] = otok[j]; }
            if (solo && gen[cur] == n_out) ++cur;
        }
        NSMutableArray * toks = [NSMutableArray new];
        for (int s = 0; s < n; ++s) {
            NSMutableArray * a = [NSMutableArray new];
            for (int i = 0; i < n_out; ++i) [a addObject:@(out[s][i])];
            [toks addObject:a];
        }
        [[NSJSONSerialization dataWithJSONObject:@{@"tokens": toks, @"steps": @(steps), @"gpu_ms": @(gpu)} options:0 error:nil] writeToFile:@(argv[4]) atomically:YES];
        sn_close(e);
    }
    return 0;
}
