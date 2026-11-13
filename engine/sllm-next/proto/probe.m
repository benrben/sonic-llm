// What arithmetic does the matrix unit (matmul2d in qmm_ks) do? Random Q4_0 weights and activations through the real
// kernel, then candidate CPU formulas compared bit for bit. usage: probe [N]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int K, M, N, nb01, ldx, ldd, ksplit, flags; } qmm_args;
typedef struct { uint16_t d; uint8_t qs[16]; } __attribute__((packed)) bq4;

static float h2f(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float) x; }
static uint16_t f2h(float f) { __fp16 x = (__fp16) f; uint16_t h; memcpy(&h, &x, 2); return h; }
static float half_round(float f) { return (float) (__fp16) f; }
static float trunc10(float f) {  // mantissa cut to half's 10 bits toward zero, exponent kept
    uint32_t u; memcpy(&u, &f, 4); u &= 0xFFFFE000u; memcpy(&f, &u, 4); return f;
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        const int N = argc > 1 ? atoi(argv[1]) : 1, M = 256, K = 2560, G = 4;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSString * src = [NSString stringWithContentsOfFile:[[@(argv[0]) stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"kernels.metal"]
                                                   encoding:NSUTF8StringEncoding error:nil];
        MTLCompileOptions * opt = [MTLCompileOptions new];
        opt.mathMode = MTLMathModeSafe;
        opt.languageVersion = MTLLanguageVersion4_0;
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"qmm_ks_q4_0_g4_nt16_r1"] error:&err];
        id<MTLBuffer> wb = [dev newBufferWithLength:(size_t) M * (K / 32) * 18 options:MTLResourceStorageModeShared];
        id<MTLBuffer> xb = [dev newBufferWithLength:(size_t) N * K * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> db = [dev newBufferWithLength:(size_t) N * M * 4 options:MTLResourceStorageModeShared];
        srand(7);
        bq4 * w = wb.contents;
        for (int i = 0; i < M * (K / 32); ++i) {
            w[i].d = f2h(((rand() % 2000) - 1000) / 40000.0f);
            for (int j = 0; j < 16; ++j) w[i].qs[j] = rand() & 0xFF;
        }
        float * x = xb.contents;
        for (int i = 0; i < N * K; ++i) x[i] = ((rand() % 200001) - 100000) / 25000.0f * powf(2.0f, (rand() % 7) - 3);
        qmm_args a = {K, M, N, K / 32 * 18, K, M, K, 1};
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps];
        [e setBytes:&a length:sizeof a atIndex:0];
        [e setBuffer:wb offset:0 atIndex:1]; [e setBuffer:xb offset:0 atIndex:2]; [e setBuffer:db offset:0 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((N + 15) / 16, M / 32, 1) threadsPerThreadgroup:MTLSizeMake(32 * G, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float * d = db.contents;
        // candidates: per slice (interleaved chunks s, s+G, ...), per chunk of 32 k, then ((p0 + p1) + p2) + p3
        const char * names[] = {"fma chain, x cut", "fma chain, x rounded", "chunk exact then add, x cut", "chunk rounded then add, x cut",
                                "exact acc+chunk, x cut", "8-groups exact then add, x cut", "16-groups exact then add, x cut"};
        int hits[7] = {0};
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
                float res[7];
                for (int c = 0; c < 7; ++c) {
                    float part[8] = {0};
                    for (int s = 0; s < G; ++s) {
                        float acc = 0.f;
                        for (int kb = s; kb < K / 32; kb += G) {
                            const bq4 * b = &w[(size_t) m * (K / 32) + kb];
                            float wv[32], xv[32];
                            for (int j = 0; j < 16; ++j) {
                                wv[j] = half_round((float) ((b->qs[j] & 15) - 8) * h2f(b->d));
                                wv[j + 16] = half_round((float) ((b->qs[j] >> 4) - 8) * h2f(b->d));
                            }
                            for (int k = 0; k < 32; ++k) {
                                const float xr = x[(size_t) n * K + kb * 32 + k];
                                xv[k] = c == 1 ? half_round(xr) : trunc10(xr);
                            }
                            if (c <= 1) {
                                for (int k = 0; k < 32; ++k) acc = fmaf(wv[k], xv[k], acc);
                            } else if (c == 2 || c == 4) {
                                double t = 0;
                                for (int k = 0; k < 32; ++k) t += (double) wv[k] * xv[k];
                                acc = c == 2 ? acc + (float) t : (float) ((double) acc + t);
                            } else if (c == 3) {
                                float t = 0;
                                for (int k = 0; k < 32; ++k) t += wv[k] * xv[k];
                                acc = acc + t;
                            } else {
                                const int gsz = c == 5 ? 8 : 16;
                                for (int g0 = 0; g0 < 32; g0 += gsz) {
                                    double t = 0;
                                    for (int k = g0; k < g0 + gsz; ++k) t += (double) wv[k] * xv[k];
                                    acc = (float) ((double) acc + t);
                                }
                            }
                        }
                        part[s] = acc;
                    }
                    res[c] = ((part[0] + part[1]) + part[2]) + part[3];
                }
                const float got = d[(size_t) n * M + m];
                for (int c = 0; c < 7; ++c) hits[c] += memcmp(&got, &res[c], 4) == 0;
            }
        }
        for (int c = 0; c < 7; ++c) printf("%-34s %d / %d\n", names[c], hits[c], N * M);
    }
    return 0;
}
