// The block matmul (qmmh_ks) alone, per weight type, at a model's prompt-step shapes: GPU time and TFLOP/s of N rows x
// an M x K matrix in q4_0, q4_1, q8_0, q4_K and q6_K (random bits, the scales set small), the kernels mm() picks for that
// shape (NT / RT env: another tile), the types interleaved, 5 rounds. TG=NRB: also qmmt (the shared weight tile) with NRB
// token rows a tile, its largest difference from qmmh_ks and whether its 32- and 128-token tiles give the same bits.
// usage: kq_bench KERNELS [N M K]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { int K, M, N, nb01, ldx, ldd, ksplit, flags, estride, ldh; } qmm_args;

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: kq_bench KERNELS [N M K]\n"); return 2; }
        const int N = argc > 4 ? atoi(argv[2]) : 512, M = argc > 4 ? atoi(argv[3]) : 14336, K = argc > 4 ? atoi(argv[4]) : 4096;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions * o = [MTLCompileOptions new];
        o.languageVersion = MTLLanguageVersion4_0;
        o.mathMode = MTLMathModeSafe;
        NSError * e = nil;
        NSString * src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:nil];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&e];
        if (!lib) { fprintf(stderr, "%s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        // the tile mm() takes: 16 / 32 tokens, two row tiles for reads of 64+ rows with K >= 2048, 8 K-slices for M <= 1024
        const int NT = getenv("NT") ? atoi(getenv("NT")) : N <= 16 ? 16 : 32, RT = getenv("RT") ? atoi(getenv("RT")) : N > 64 && K >= 2048 ? 2 : 1,
                  G = M <= 1024 ? 8 : 4;  // NT / RT env: another tile
        const char * names[] = {"q4_0", "q4_1", "q8_0", "q4_K", "q6_K", "q5_K"};
        const int bsz[] = {18, 20, 34, 144, 210, 176}, bw[] = {32, 32, 32, 256, 256, 256}, hoff[] = {0, 0, 0, 0, 208, 0};  // a block's bytes, weights, its d
        const int nf = 6;
        id<MTLComputePipelineState> ps[6], pt[6], pt32[6], pt128[6], pq[6];
        const int QG = getenv("QG") ? atoi(getenv("QG")) : 0, QS = QG ? 1 : getenv("QS") ? atoi(getenv("QS")) : 0;  // QG=NRB: qmmg (32 x NRB tiles) in qmms' place  // QS=TS: also qmms with TS token tiles a threadgroup (must be qmmh_ks bit for bit)
        const int TG = getenv("TG") ? atoi(getenv("TG")) : 0, TGM = getenv("TGM") ? atoi(getenv("TGM")) : 64;  // TGM: qmmt's weight rows a tile
        id<MTLBuffer> Wb[6];
        for (int f = 0; f < nf; ++f) {
            NSString * n = [NSString stringWithFormat:@"qmmh_ks_%s_g%d_nt%d_r%d", names[f], G, NT, RT];
            ps[f] = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&e];
            if (!ps[f]) { fprintf(stderr, "no %s\n", n.UTF8String); return 1; }
            if (QS) {
                pq[f] = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:(QG ? [NSString stringWithFormat:@"qmmg_%s_g%d_n%d", names[f], G, QG] : [NSString stringWithFormat:@"qmms_%s_g%d_t%d", names[f], G, QS])] error:&e];
                if (!pq[f]) { fprintf(stderr, "no qmms %s\n", names[f]); return 1; }
            }
            if (TG) {
                pt[f] = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithFormat:@"qmmt_%s_m%d_n%d", names[f], TGM, TG]] error:&e];
                pt32[f] = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithFormat:@"qmmt_%s_m%d_n32", names[f], TGM]] error:&e];
                pt128[f] = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithFormat:@"qmmt_%s_m%d_n128", names[f], TGM]] error:&e];
                if (!pt[f] || !pt32[f] || !pt128[f]) { fprintf(stderr, "no qmmt %s: %s\n", names[f], e.localizedDescription.UTF8String); return 1; }
            }
            const size_t nb = (size_t) M * (K / bw[f]), bytes = nb * bsz[f];
            Wb[f] = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            uint8_t * w = Wb[f].contents;
            for (size_t i = 0; i < bytes; ++i) w[i] = (uint8_t) (rand() >> 7);
            for (size_t b = 0; b < nb; ++b) {
                uint16_t * h = (uint16_t *) (w + b * bsz[f] + hoff[f]);
                h[0] = 0x2000;                      // d = 2^-7
                if (f == 1 || f == 3 || f == 5) h[1] = 0x1800;  // m / dmin small
            }
        }
        id<MTLBuffer> X = [dev newBufferWithLength:(size_t) N * K * 2 options:MTLResourceStorageModeShared];
        for (size_t i = 0; i < (size_t) N * K; ++i) ((uint16_t *) X.contents)[i] = 0x2000 + (uint16_t) (i % 512);
        id<MTLBuffer> D = [dev newBufferWithLength:(size_t) N * M * 4 options:MTLResourceStorageModePrivate];
        double best[6] = {1e9, 1e9, 1e9, 1e9, 1e9, 1e9}, bestt[6] = {1e9, 1e9, 1e9, 1e9, 1e9, 1e9};
        for (int round = 0; round < 6; ++round) {
            for (int f = 0; f < nf && QS; ++f) {
                qmm_args a = {K, M, N, K / bw[f] * bsz[f], K, M, K, 1, 0, 0};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
                [c setComputePipelineState:pq[f]];
                [c setBytes:&a length:sizeof a atIndex:0];
                [c setBuffer:Wb[f] offset:0 atIndex:1]; [c setBuffer:X offset:0 atIndex:2]; [c setBuffer:D offset:0 atIndex:3];
                [c setBuffer:D offset:0 atIndex:4]; [c setBuffer:D offset:0 atIndex:5];
                for (int rep = 0; rep < 10; ++rep)
                    [c dispatchThreadgroups:MTLSizeMake(QG ? (N + QG - 1) / QG : (N + 32 * QS - 1) / (32 * QS), (M + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(QG ? 128 : 32 * G * QS, 1, 1)];
                [c endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / 10;
                if (round > 0 && ms < bestt[f]) bestt[f] = ms;
            }
            for (int f = 0; f < nf && TG; ++f) {
                qmm_args a = {K, M, N, K / bw[f] * bsz[f], K, M, K, 1, 0, 0};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
                [c setComputePipelineState:pt[f]];
                [c setBytes:&a length:sizeof a atIndex:0];
                [c setBuffer:Wb[f] offset:0 atIndex:1]; [c setBuffer:X offset:0 atIndex:2]; [c setBuffer:D offset:0 atIndex:3];
                [c setBuffer:D offset:0 atIndex:4]; [c setBuffer:D offset:0 atIndex:5];
                for (int rep = 0; rep < 10; ++rep)
                    [c dispatchThreadgroups:MTLSizeMake((N + TG - 1) / TG, (M + TGM - 1) / TGM, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [c endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / 10;
                if (round > 0 && ms < bestt[f]) bestt[f] = ms;
            }
            for (int f = 0; f < nf; ++f) {
                qmm_args a = {K, M, N, K / bw[f] * bsz[f], K, M, K, 1, 0, 0};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
                [c setComputePipelineState:ps[f]];
                [c setBytes:&a length:sizeof a atIndex:0];
                [c setBuffer:Wb[f] offset:0 atIndex:1]; [c setBuffer:X offset:0 atIndex:2]; [c setBuffer:D offset:0 atIndex:3];
                for (int rep = 0; rep < 10; ++rep)
                    [c dispatchThreadgroups:MTLSizeMake((N + NT - 1) / NT, (M + 32 * RT - 1) / (32 * RT), 1) threadsPerThreadgroup:MTLSizeMake(32 * G * RT, 1, 1)];
                [c endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / 10;
                if (round > 0 && ms < best[f]) best[f] = ms;  // round 0 warms up
            }
        }
        printf("N %d M %d K %d (nt %d, rt %d, g %d)\n", N, M, K, NT, RT, G);
        for (int f = 0; f < nf; ++f)
            printf("  %-5s %7.3f ms  %5.1f TFLOP/s  (%.2fx q4_0's time)%s\n", names[f], best[f], 2.0 * N * M * K / best[f] / 1e9, best[f] / best[0],
                   TG || QS ? [NSString stringWithFormat:@"   %s %7.3f ms (%.2fx)", QS ? "qmms" : "qmmt", bestt[f], bestt[f] / best[f]].UTF8String : "");
        if (QS) {  // qmms against qmmh_ks, bit for bit
            id<MTLBuffer> D1 = [dev newBufferWithLength:(size_t) N * M * 4 options:MTLResourceStorageModeShared], D2 = [dev newBufferWithLength:(size_t) N * M * 4 options:MTLResourceStorageModeShared];
            for (int f = 0; f < nf; ++f) {
                qmm_args a = {K, M, N, K / bw[f] * bsz[f], K, M, K, 1, 0, 0};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
                [c setBytes:&a length:sizeof a atIndex:0];
                [c setBuffer:Wb[f] offset:0 atIndex:1]; [c setBuffer:X offset:0 atIndex:2];
                [c setComputePipelineState:ps[f]]; [c setBuffer:D1 offset:0 atIndex:3]; [c setBuffer:D1 offset:0 atIndex:5];
                [c dispatchThreadgroups:MTLSizeMake((N + NT - 1) / NT, (M + 32 * RT - 1) / (32 * RT), 1) threadsPerThreadgroup:MTLSizeMake(32 * G * RT, 1, 1)];
                [c setComputePipelineState:pq[f]]; [c setBuffer:D2 offset:0 atIndex:3]; [c setBuffer:D2 offset:0 atIndex:5];
                [c dispatchThreadgroups:MTLSizeMake(QG ? (N + QG - 1) / QG : (N + 32 * QS - 1) / (32 * QS), (M + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(QG ? 128 : 32 * G * QS, 1, 1)];
                [c endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                printf("  %-5s qmms vs qmmh_ks: %s\n", names[f], memcmp(D1.contents, D2.contents, (size_t) N * M * 4) ? "DIFFER" : "same bits");
            }
        }
        if (TG) {  // the outputs: qmmh_ks against qmmt; qmmt's 32- and 128-token tiles bit for bit
            id<MTLBuffer> D1 = [dev newBufferWithLength:(size_t) N * M * 4 options:MTLResourceStorageModeShared], D2 = [dev newBufferWithLength:(size_t) N * M * 4 options:MTLResourceStorageModeShared],
                          D3 = [dev newBufferWithLength:(size_t) N * M * 4 options:MTLResourceStorageModeShared];
            for (int f = 0; f < nf; ++f) {
                qmm_args a = {K, M, N, K / bw[f] * bsz[f], K, M, K, 1, 0, 0};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
                [c setBytes:&a length:sizeof a atIndex:0];
                [c setBuffer:Wb[f] offset:0 atIndex:1]; [c setBuffer:X offset:0 atIndex:2];
                [c setComputePipelineState:ps[f]]; [c setBuffer:D1 offset:0 atIndex:3]; [c setBuffer:D1 offset:0 atIndex:5];
                [c dispatchThreadgroups:MTLSizeMake((N + NT - 1) / NT, (M + 32 * RT - 1) / (32 * RT), 1) threadsPerThreadgroup:MTLSizeMake(32 * G * RT, 1, 1)];
                [c setComputePipelineState:pt32[f]]; [c setBuffer:D2 offset:0 atIndex:3]; [c setBuffer:D2 offset:0 atIndex:5];
                [c dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + TGM - 1) / TGM, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [c setComputePipelineState:pt128[f]]; [c setBuffer:D3 offset:0 atIndex:3]; [c setBuffer:D3 offset:0 atIndex:5];
                [c dispatchThreadgroups:MTLSizeMake((N + 127) / 128, (M + TGM - 1) / TGM, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [c endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const float * d1 = D1.contents, * d2 = D2.contents;
                double dmax = 0, mag = 0;
                for (size_t i = 0; i < (size_t) N * M; ++i) { dmax = fmax(dmax, fabs(d1[i] - d2[i])); mag = fmax(mag, fabs(d1[i])); }
                printf("  %-5s qmmt vs qmmh_ks: largest difference %.2g of values up to %.3g; qmmt n32 vs n128: %s\n", names[f], dmax, mag,
                       memcmp(D2.contents, D3.contents, (size_t) N * M * 4) ? "DIFFER" : "same bits");
            }
        }
    }
    return 0;
}
