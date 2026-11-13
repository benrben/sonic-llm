// g4_pre (prompt attention) against g4_attn_part + g4_attn_merge (write attention) on the same random rows and cache: one
// sequence of P cells (one page a chunk), R rows at positions P - R .. P - 1, head size D, G query heads per K/V head,
// a window, sinks; prints the largest difference. usage: pre_test KERNELS D G NKV P R WINDOW SINKS [QSCALE]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdlib.h>

typedef struct { int n_head, n_kv, ctx; float scale; int ldq, ldo, maxc, window, n_fast, maxch, hg, sinks; } g4a_args;
typedef struct { int n_head, n_kv, ctx; float scale; int ldq, ldo, window, sinks; } g4p_args;

static float frand(void) { return (float) rand() / RAND_MAX * 2 - 1; }

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 9) { fprintf(stderr, "usage: pre_test KERNELS D G NKV P R WINDOW SINKS\n"); return 2; }
        const int D = atoi(argv[2]), G = atoi(argv[3]), NKV = atoi(argv[4]), P = atoi(argv[5]), R = atoi(argv[6]), W = atoi(argv[7]), SK = atoi(argv[8]);
        const int NH = G * NKV, QD = NH * D, KVS = NKV * D, CTX = (P + 63) / 64 * 64, maxc = CTX / 64, TT = 32 / G;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions * o = [MTLCompileOptions new];
        o.languageVersion = MTLLanguageVersion4_0;
        o.mathMode = MTLMathModeSafe;
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:nil] options:o error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> (^K)(NSString *) = ^(NSString * n) { return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:nil]; };
        const MTLResourceOptions SH = MTLResourceStorageModeShared;
        srand(7);
        id<MTLBuffer> q = [dev newBufferWithLength:(size_t) R * QD * 4 options:SH], kc = [dev newBufferWithLength:(size_t) CTX * KVS * 2 options:SH],
                      vc = [dev newBufferWithLength:(size_t) CTX * KVS * 2 options:SH], pt = [dev newBufferWithLength:maxc * 4 options:SH],
                      seq = [dev newBufferWithLength:R * 4 options:SH], pos = [dev newBufferWithLength:R * 4 options:SH],
                      sk = [dev newBufferWithLength:NH * 4 options:SH], apart = [dev newBufferWithLength:(size_t) R * NH * maxc * (D + 2) * 4 options:SH],
                      ao1 = [dev newBufferWithLength:(size_t) R * QD * 4 options:SH], ao2 = [dev newBufferWithLength:(size_t) R * QD * 4 options:SH],
                      blk = [dev newBufferWithLength:(size_t) (R / TT + 2) * 16 options:SH], qh = [dev newBufferWithLength:(size_t) (R / TT + 2) * NKV * 32 * D * 2 options:SH];
        const float qs = argc > 9 ? atof(argv[9]) : 3.0f;  // the queries' size (small: the sinks matter)
        for (int i = 0; i < R * QD; ++i) ((float *) q.contents)[i] = qs * frand();
        for (int i = 0; i < CTX * KVS; ++i) { ((__fp16 *) kc.contents)[i] = (__fp16) (3 * frand()); ((__fp16 *) vc.contents)[i] = (__fp16) frand(); }
        for (int c = 0; c < maxc; ++c) ((int *) pt.contents)[maxc - 1 - c] = c;  // pages in reverse, as a pool hands them out
        for (int r = 0; r < R; ++r) { ((int *) seq.contents)[r] = 0; ((int *) pos.contents)[r] = P - R + r; }
        for (int h = 0; h < NH; ++h) ((float *) sk.contents)[h] = 2 * frand();
        int nb = 0;
        for (int r = 0; r < R; r += TT, ++nb) {  // tiles: (first row, rows, sequence)
            int * b = (int *) blk.contents + 4 * nb;
            b[0] = r; b[1] = MIN(TT, R - r); b[2] = 0; b[3] = 0;
        }
        const float scale = 1.0f / sqrtf((float) D);
        id<MTLCommandQueue> cq = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [cq commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        // write attention
        g4a_args aa = {NH, NKV, CTX, scale, QD, QD, maxc, W, 0, 0, MIN(NH, 8), SK};
        [e setComputePipelineState:K([NSString stringWithFormat:@"g4_attn_part_%d", D])];
        [e setBytes:&aa length:sizeof aa atIndex:0];
        [e setBuffer:q offset:0 atIndex:1]; [e setBuffer:kc offset:0 atIndex:2]; [e setBuffer:vc offset:0 atIndex:3];
        [e setBuffer:seq offset:0 atIndex:4]; [e setBuffer:pos offset:0 atIndex:5]; [e setBuffer:apart offset:0 atIndex:6]; [e setBuffer:pt offset:0 atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(R, NH / MIN(NH, 8), maxc) threadsPerThreadgroup:MTLSizeMake(32 * MIN(NH, 8), 1, 1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:K([NSString stringWithFormat:@"g4_attn_merge_%d", D])];
        [e setBytes:&aa length:sizeof aa atIndex:0];
        [e setBuffer:apart offset:0 atIndex:1]; [e setBuffer:pos offset:0 atIndex:2]; [e setBuffer:ao1 offset:0 atIndex:3]; [e setBuffer:sk offset:0 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(R, NH, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        // prompt attention
        g4p_args pa = {NH, NKV, CTX, scale, QD, QD, W, SK};
        [e setComputePipelineState:K([NSString stringWithFormat:@"g4_pre_q_%d_g%d", D, G])];
        [e setBytes:&pa length:sizeof pa atIndex:0];
        [e setBuffer:q offset:0 atIndex:1]; [e setBuffer:blk offset:0 atIndex:2]; [e setBuffer:qh offset:0 atIndex:3];
        [e dispatchThreads:MTLSizeMake(D / 4, 32, nb * NKV) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:K([NSString stringWithFormat:@"g4_pre_%d_g%d", D, G])];
        [e setBytes:&pa length:sizeof pa atIndex:0];
        [e setBuffer:qh offset:0 atIndex:1]; [e setBuffer:kc offset:0 atIndex:2]; [e setBuffer:vc offset:0 atIndex:3];
        [e setBuffer:blk offset:0 atIndex:4]; [e setBuffer:pos offset:0 atIndex:5]; [e setBuffer:ao2 offset:0 atIndex:6]; [e setBuffer:pt offset:0 atIndex:7]; [e setBuffer:sk offset:0 atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake(nb, NKV, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "%s\n", cb.error.localizedDescription.UTF8String); return 1; }
        const float * a1 = ao1.contents, * a2 = ao2.contents;
        double worst = 0, mag = 0;
        int wr = 0, wh = 0;
        for (int r = 0; r < R; ++r)
            for (int h = 0; h < NH; ++h)
                for (int d = 0; d < D; ++d) {
                    const double x = fabs(a1[(size_t) r * QD + h * D + d] - a2[(size_t) r * QD + h * D + d]);
                    mag = fmax(mag, fabs(a1[(size_t) r * QD + h * D + d]));
                    if (x > worst) { worst = x; wr = r; wh = h; }
                }
        printf("D %d G %d P %d R %d window %d sinks %d: largest difference %.5f (row %d, head %d; outputs up to %.3f)\n", D, G, P, R, W, SK, worst, wr, wh, mag);
    }
    return 0;
}
