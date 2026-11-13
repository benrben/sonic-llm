// The matrix units' ceiling for qmm_ks's operation: matmul2d (32 x NT x 32, a register-held half left input, a float right
// input from memory, float accumulate), looped with nothing loaded but one small L1-resident tile. usage: peak [NT]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>

static const char * SRC = R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
template <int NT, typename TX>
kernel void peak(device const TX * X [[buffer(0)]], device float * D [[buffer(1)]], constant int & iters [[buffer(2)]],
                 uint tg [[threadgroup_position_in_grid]], ushort ln [[thread_index_in_simdgroup]]) {
    constexpr auto desc = matmul2d_descriptor(32, NT, 32, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> mm;
    auto tX = tensor((device TX *) X, dextents<int32_t, 2>(32, NT), array<int, 2>({1, 32}));
    auto wL = mm.template get_left_input_cooperative_tensor<half, TX, float>();
    auto cT = mm.template get_destination_cooperative_tensor<decltype(wL), decltype(tX), float>();
    for (uint16_t i = 0; i < wL.get_capacity(); ++i) wL[i] = half(0.001h * (ln + i));
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.f;
    for (int it = 0; it < iters; ++it) mm.run(wL, tX, cT);
    float s = 0.f;
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) s += cT[i];
    if (ln == 0) D[tg] = s;
}
// the same with the right input streamed: a 512-token x K matrix, each threadgroup its NT tokens, K walked in 32s (qmm's reads)
template <int NT, typename TX>
kernel void peakx(device const TX * X [[buffer(0)]], device float * D [[buffer(1)]], constant int & iters [[buffer(2)]],
                  uint tg [[threadgroup_position_in_grid]], ushort ln [[thread_index_in_simdgroup]]) {
    constexpr int K = 2560;
    constexpr auto desc = matmul2d_descriptor(32, NT, 32, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> mm;
    device const TX * xb = X + (size_t) (tg % (512 / NT)) * NT * K;
    auto tX0 = tensor((device TX *) xb, dextents<int32_t, 2>(32, NT), array<int, 2>({1, K}));
    auto wL = mm.template get_left_input_cooperative_tensor<half, TX, float>();
    auto cT = mm.template get_destination_cooperative_tensor<decltype(wL), decltype(tX0), float>();
    for (uint16_t i = 0; i < wL.get_capacity(); ++i) wL[i] = half(0.001h * (ln + i));
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.f;
    for (int it = 0; it < iters; ++it) {
        auto tX = tensor((device TX *) xb + (it % (K / 32)) * 32, dextents<int32_t, 2>(32, NT), array<int, 2>({1, K}));
        mm.run(wL, tX, cT);
    }
    float s = 0.f;
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) s += cT[i];
    if (ln == 0) D[tg] = s;
}
// streamed half input and a q4_0 weight tile dequantized every step (qmm_ks's cv_ks), the raw blocks from an L1-resident buffer
struct bq4 { half d; uchar qs[16]; };
kernel void peakd(device const half * X [[buffer(0)]], device float * D [[buffer(1)]], constant int & iters [[buffer(2)]],
                  device const bq4 * Wq [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], ushort ln [[thread_index_in_simdgroup]]) {
    constexpr int K = 2560, NT = 32;
    constexpr auto desc = matmul2d_descriptor(32, NT, 32, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> mm;
    device const half * xb = X + (size_t) (tg % (512 / NT)) * NT * K;
    auto tX0 = tensor((device half *) xb, dextents<int32_t, 2>(32, NT), array<int, 2>({1, K}));
    auto wL = mm.template get_left_input_cooperative_tensor<half, half, float>();
    auto cT = mm.template get_destination_cooperative_tensor<decltype(wL), decltype(tX0), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.f;
    const auto ix0 = wL.get_multidimensional_index((uint16_t) 0);
    const int k0l = ix0[0], rl = ix0[1];
    for (int it = 0; it < iters; ++it) {
        for (int q = 0; q < 4; ++q) {
            device const bq4 & b = Wq[((rl + 8 * (q % 2) + 16 * (q / 2)) * 4 + it) % 256];
            device const ushort * qs = (device const ushort *) (b.qs + k0l);
            const uint qq = (uint) qs[0] | ((uint) qs[1] << 16);
            const uint4 bb = uint4(qq, qq >> 8, qq >> 16, qq >> 24) & 0xFF;
            const half4 lo = (half4(bb & 0xF) - 8.h) * b.d, hi = (half4(bb >> 4) - 8.h) * b.d;
            const int a_ = q / 2, c_ = q % 2;
            for (int e = 0; e < 4; ++e) { wL[16 * a_ + 0 + 4 * c_ + e] = lo[e]; wL[16 * a_ + 8 + 4 * c_ + e] = hi[e]; }
        }
        auto tX = tensor((device half *) xb + (it % (K / 32)) * 32, dextents<int32_t, 2>(32, NT), array<int, 2>({1, K}));
        mm.run(wL, tX, cT);
    }
    float s = 0.f;
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) s += cT[i];
    if (ln == 0) D[tg] = s;
}
template [[host_name("peakx32")]] kernel void peakx<32, float>(device const float *, device float *, constant int &, uint, ushort);
template [[host_name("peakxh32")]] kernel void peakx<32, half>(device const half *, device float *, constant int &, uint, ushort);
template [[host_name("peak16")]] kernel void peak<16, float>(device const float *, device float *, constant int &, uint, ushort);
template [[host_name("peak32")]] kernel void peak<32, float>(device const float *, device float *, constant int &, uint, ushort);
template [[host_name("peak64")]] kernel void peak<64, float>(device const float *, device float *, constant int &, uint, ushort);
template [[host_name("peakh16")]] kernel void peak<16, half>(device const half *, device float *, constant int &, uint, ushort);
template [[host_name("peakh32")]] kernel void peak<32, half>(device const half *, device float *, constant int &, uint, ushort);
template [[host_name("peakh64")]] kernel void peak<64, half>(device const half *, device float *, constant int &, uint, ushort);
)";

int main(int argc, char ** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions * o = [MTLCompileOptions new];
        o.languageVersion = MTLLanguageVersion4_0;
        NSError * e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(SRC) options:o error:&e];
        if (!lib) { fprintf(stderr, "%s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLBuffer> X = [dev newBufferWithLength:512 * 2560 * 4 options:MTLResourceStorageModeShared];
        for (int i = 0; i < 512 * 2560; ++i) ((float *) X.contents)[i] = 0.01f * (i % 7);
        const int groups = 8192, iters = 2048;
        id<MTLBuffer> D = [dev newBufferWithLength:groups * 4 options:MTLResourceStorageModeShared];
        NSArray * names = @[@"peak32", @"peakh32", @"peakx32", @"peakxh32", @"peakd"];
        id<MTLBuffer> Wq = [dev newBufferWithLength:256 * 18 options:MTLResourceStorageModeShared];
        for (int i = 0; i < 256 * 18; ++i) ((uint8_t *) Wq.contents)[i] = (uint8_t) (i * 37);
        for (int i = 0; i < 256; ++i) ((uint16_t *) ((uint8_t *) Wq.contents + i * 18))[0] = 0x2000;  // d = 2^-7
        for (int k = 0; k < 5; ++k) {
            const int NT = 32;
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:names[k]] error:&e];
            for (int rep = 0; rep < 3; ++rep) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
                [c setComputePipelineState:p];
                [c setBuffer:X offset:0 atIndex:0]; [c setBuffer:D offset:0 atIndex:1]; [c setBytes:&iters length:4 atIndex:2]; [c setBuffer:Wq offset:0 atIndex:3];
                [c dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [c endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double s = cb.GPUEndTime - cb.GPUStartTime, flop = 2.0 * 32 * NT * 32 * (double) iters * groups;
                if (rep == 2) printf("%s: %.1f TFLOP/s (%.1f ms)\n", [names[k] UTF8String], flop / s / 1e12, s * 1e3);
            }
        }
    }
    return 0;
}
