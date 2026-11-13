// A draft model for one agent: the draft writes k tokens one at a time, the target checks them in one step of k + 1 write
// rows (the one-row kernels take up to 4 rows and give each row the bits of a one-row step), the agreeing prefix and the
// target's next token are kept. The tokens must equal the target's own greedy tokens.
// usage: draft_bench TARGET.gguf DRAFT.gguf KERNELS REF.json N_PROMPTS N_OUT K...
#import <Foundation/Foundation.h>
#include "sn.h"
#include <time.h>

static double now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }
static char err[512];
static int g_chain;  // DB_CHAIN=1: write steps queued one ahead (sn_step chain)
#define CK(x) do { if ((x) != 0) { fprintf(stderr, "%s: %s\n", #x, err); exit(1); } } while (0)

// read a prompt into slot 0; the target's first token
static int32_t prefill(sn_engine * e, NSArray * p, bool sample) {
    const int n = (int) p.count;
    int32_t * tok = malloc(n * 4), * seq = calloc(n, 4), * pos = malloc(n * 4), last = n - 1, out = -1;
    for (int i = 0; i < n; ++i) { tok[i] = [p[i] intValue]; pos[i] = i; }
    for (int c = 0; c < n; c += 512) {  // steps of at most 512 rows; the last one samples
        const int m = MIN(512, n - c), lr = m - 1;
        CK(sn_step(e, m, tok + c, seq + c, pos + c, 0, 0, sample && c + m == n ? 1 : 0, &lr, &out, 0, err, sizeof err));
    }
    (void) last;
    free(tok); free(seq); free(pos);
    return out;
}

// one write row (or a few of one sequence) at positions p0..; next tokens of every row
static void write_rows(sn_engine * e, const int32_t * tok, int n, int p0, int32_t * out) {
    int32_t seq[4] = {0}, pos[4], rows[4] = {0, 1, 2, 3};
    for (int i = 0; i < n; ++i) pos[i] = p0 + i;
    CK(sn_step(e, n, tok, seq, pos, n, n, n, rows, out, g_chain, err, sizeof err));
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 7) { fprintf(stderr, "usage: draft_bench TARGET DRAFT KERNELS REF.json N_PROMPTS N_OUT [K...]\n"); return 2; }
        NSString * src = [NSString stringWithContentsOfFile:@(argv[3]) encoding:NSUTF8StringEncoding error:nil];
        NSDictionary * ref = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[4])] options:0 error:nil];
        const int np = atoi(argv[5]), n_out = atoi(argv[6]), ctx = 2048;
        g_chain = getenv("DB_CHAIN") != NULL;
        sn_engine * T = sn_open(argv[1], src.UTF8String, 1, ctx, 1024, 0, err, sizeof err);
        sn_engine * D = sn_open(argv[2], src.UTF8String, 1, ctx, 1024, 0, err, sizeof err);
        if (!T || !D) { fprintf(stderr, "open: %s\n", err); return 1; }
        for (int pi = 0; pi < np; ++pi) {
            NSArray * p = ref[@"prompts"][pi + 1];
            const int n0 = (int) p.count;
            // the target alone, one token per step
            CK(sn_clear(T, 0, 0, ctx, err, sizeof err));
            const double tp0 = now_ms();
            int32_t base[4096], t = prefill(T, p, true);
            printf("{\"prefill_rows\": %d, \"prefill_ms\": %.2f, \"prefill_gpu_ms\": %.2f}\n", (int) p.count, now_ms() - tp0, sn_last_gpu_ms(T));
            double t0 = now_ms(), tg = 0, dg = 0;
            for (int i = 0; i < n_out; ++i) {
                base[i] = t;
                write_rows(T, &t, 1, n0 + i, &t);
                tg += sn_last_gpu_ms(T);
            }
            const double base_ms = now_ms() - t0;
            int32_t v2[2] = {t, t}, o2[2];
            write_rows(T, v2, 2, n0 + n_out, o2);
            const double v2g = sn_last_gpu_ms(T);
            CK(sn_clear(T, 0, n0 + n_out, n0 + n_out + 2, err, sizeof err));
            // the draft alone
            CK(sn_clear(D, 0, 0, ctx, err, sizeof err));
            int32_t d = prefill(D, p, true);
            t0 = now_ms();
            for (int i = 0; i < n_out; ++i) { write_rows(D, &d, 1, n0 + i, &d); dg += sn_last_gpu_ms(D); }
            const double draft_ms = (now_ms() - t0) / n_out;
            printf("{\"prompt\": %d, \"target_tok_s\": %.1f, \"target_ms\": [%.2f, %.2f], \"check2_gpu_ms\": %.2f, \"draft_ms\": [%.2f, %.2f]", pi,
                   n_out * 1000.0 / base_ms, base_ms / n_out, tg / n_out, v2g, draft_ms, dg / n_out);
            for (int ki = 7; ki < argc; ++ki) {
                const int k = atoi(argv[ki]);
                CK(sn_clear(T, 0, 0, ctx, err, sizeof err));
                CK(sn_clear(D, 0, 0, ctx, err, sizeof err));
                int32_t got[4200], last = prefill(T, p, true);
                prefill(D, p, false);
                int n = n0, dn = n0, ng = 0, rounds = 0, kept = 0;  // cells: target n, draft dn; got: tokens so far
                int32_t dk_pending = -1;  // the last draft token when every draft was kept (not in the draft's cells yet)
                t0 = now_ms();
                while (ng < n_out) {
                    got[ng++] = last;
                    if (ng >= n_out) break;
                    // draft k tokens after `last` (first catching up on a kept draft token not fed yet)
                    int32_t dr[4], out[4];
                    if (dk_pending >= 0) {
                        int32_t two[2] = {dk_pending, last};
                        write_rows(D, two, 2, dn, out);
                        dr[0] = out[1];
                        dn += 2;
                    } else {
                        write_rows(D, &last, 1, dn, out);
                        dr[0] = out[0];
                        dn += 1;
                    }
                    for (int i = 1; i < k; ++i) { write_rows(D, &dr[i - 1], 1, dn, out); dr[i] = out[0]; dn += 1; }
                    // check: rows last, dr[0..k-1] at n..n+k
                    int32_t rows[4] = {last, dr[0], dr[1], dr[2]}, tt[4];
                    write_rows(T, rows, k + 1, n, tt);
                    int j = 0;
                    while (j < k && dr[j] == tt[j]) ++j;
                    for (int i = 0; i < j && ng < n_out; ++i) got[ng++] = dr[i];
                    kept += j; rounds += 1;
                    // the target keeps last + j drafts; the rest of its rows go
                    CK(sn_clear(T, 0, n + j + 1, n + k + 1, err, sizeof err));
                    // the draft holds last, dr[0..k-2]: keep last + j drafts of them
                    const int keep = n + 1 + MIN(j, k - 1);
                    CK(sn_clear(D, 0, keep, dn, err, sizeof err));
                    dn = keep;
                    dk_pending = j == k ? dr[k - 1] : -1;
                    n += j + 1;
                    last = tt[j];
                }
                const double ms = now_ms() - t0;
                int same = 1;
                for (int i = 0; i < n_out; ++i) same &= got[i] == base[i];
                printf(", \"k%d\": {\"tok_s\": %.1f, \"kept\": \"%d/%d\", \"same\": %s}", k, n_out * 1000.0 / ms, kept, rounds * k, same ? "true" : "false");
            }
            printf("}\n");
            fflush(stdout);
        }
        sn_close(D);
        sn_close(T);
    }
    return 0;
}
