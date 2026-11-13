// The one-row Q4_0 mat-vec (mv_q4_0_r1_m0) alone on random weights: GB/s for an M x K matrix as one dispatch and as C
// dispatches of M / C rows (does a very tall matrix, a Gemma output layer, stream slower?). usage: mv_bench KERNELS M K C
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct { int K, M, R, ldx, ldy, acc, nb01; float eps; } mv_args;

int main(int argc, char ** argv) {
    @autoreleasepool {
        const int M = atoi(argv[2]), K = atoi(argv[3]), C = atoi(argv[4]), nb01 = K / 32 * 18;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions * o = [MTLCompileOptions new];
        o.languageVersion = MTLLanguageVersion4_0;
        o.mathMode = MTLMathModeSafe;
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:nil] options:o error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mv_q4_0_r1_m0"] error:&err];
        const size_t wbytes = (size_t) M * nb01;
        id<MTLBuffer> w;
        if (getenv("FILEMAP")) {  // weights in a file mapped no-copy, as sn.m maps a GGUF
            char path[] = "/private/tmp/claude-501/-Users-benreich-Projects-crazyidea/d765eb19-2996-47f2-b57b-b9f82d97ec9b/scratchpad/mvw.bin";
            const size_t len = (wbytes + 16383) / 16384 * 16384;
            FILE * f = fopen(path, "wb");
            uint8_t * tmp = malloc(len);
            for (size_t i = 0; i < len; ++i) tmp[i] = (uint8_t) rand();
            for (size_t r = 0; r < (size_t) M; ++r) for (int b = 0; b < K / 32; ++b) { __fp16 d = 0.01f; memcpy(tmp + r * nb01 + b * 18, &d, 2); }
            fwrite(tmp, 1, len, f); fclose(f); free(tmp);
            const int fd = open(path, O_RDONLY);
            void * m = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
            w = [dev newBufferWithBytesNoCopy:m length:len options:MTLResourceStorageModeShared deallocator:nil];
        } else w = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
        uint8_t * wp = getenv("FILEMAP") ? NULL : w.contents;
        if (wp) {
            for (size_t i = 0; i < wbytes; ++i) wp[i] = (uint8_t) rand();
            for (size_t r = 0; r < (size_t) M; ++r) for (int b = 0; b < K / 32; ++b) { __fp16 d = 0.01f; memcpy(wp + r * nb01 + b * 18, &d, 2); }
        }
        id<MTLBuffer> x = [dev newBufferWithLength:(size_t) K * 4 options:MTLResourceStorageModeShared];
        for (int i = 0; i < K; ++i) ((float *) x.contents)[i] = (float) rand() / RAND_MAX;
        id<MTLBuffer> y = [dev newBufferWithLength:(size_t) M * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        for (int pass = 0; pass < 2; ++pass) {
            const int chunks = pass ? C : 1, mc = M / chunks;
            double best = 1e9;
            for (int it = 0; it < 12; ++it) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p];
                for (int c = 0; c < chunks; ++c) {
                    mv_args a = {K, mc, 1, K, M, 0, nb01, 1e-6f};
                    [e setBytes:&a length:sizeof a atIndex:0];
                    [e setBuffer:w offset:(size_t) c * mc * nb01 atIndex:1]; [e setBuffer:x offset:0 atIndex:2];
                    [e setBuffer:y offset:(size_t) c * mc * 4 atIndex:3]; [e setBuffer:x offset:0 atIndex:4];
                    [e dispatchThreadgroups:MTLSizeMake((mc + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                }
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (it >= 2) best = MIN(best, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
            }
            printf("%d x %d as %d dispatch%s: %.3f ms, %.0f GB/s\n", M, K, chunks, chunks > 1 ? "es" : "", best, wbytes / best / 1e6);
        }
    }
    return 0;
}
