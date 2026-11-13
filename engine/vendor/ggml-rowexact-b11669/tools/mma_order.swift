// Which scalar order reproduces simdgroup_multiply_accumulate(C, A(half 8x8), B(float 8x8), C) bit for bit?
// Many random tiles go through the hardware op; every output is compared with candidate CPU formulas.
// Run: swiftc -O mma_order.swift -o /tmp/mma_order && /tmp/mma_order   (kernel_sllm_mme_q4_0_f32 relies on "seq fma k=0..7": 0)
import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void mma(device const half * A [[buffer(0)]], device const float * B [[buffer(1)]], device float * C [[buffer(2)]],
                uint tg [[threadgroup_position_in_grid]]) {
    simdgroup_half8x8 a; simdgroup_float8x8 b; simdgroup_float8x8 c;
    simdgroup_load(a, A + tg*64, 8);
    simdgroup_load(b, B + tg*64, 8);
    simdgroup_load(c, C + tg*64, 8);
    simdgroup_multiply_accumulate(c, a, b, c);
    simdgroup_store(c, C + tg*64, 8);
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "mma")!)
let q = dev.makeCommandQueue()!

let tiles = 20000
var rng = SystemRandomNumberGenerator()
func rf(_ lo: Float, _ hi: Float) -> Float { Float.random(in: lo..<hi, using: &rng) }
for mode in ["q4-like ints", "general halves"] {
    var A = [Float16](repeating: 0, count: tiles*64)
    var B = [Float](repeating: 0, count: tiles*64)
    var C = [Float](repeating: 0, count: tiles*64)
    for i in 0..<A.count {
        A[i] = mode == "q4-like ints" ? Float16(Float(Int.random(in: -8...7)) * (Bool.random() ? 1 : 16)) : Float16(rf(-4, 4))
        // activations of mixed magnitudes, no subnormals
        B[i] = rf(-1, 1) * powf(2, Float(Int.random(in: -6...6)))
        C[i] = rf(-1, 1) * powf(2, Float(Int.random(in: -4...8)))
    }
    let bA = dev.makeBuffer(bytes: A, length: A.count*2)!
    let bB = dev.makeBuffer(bytes: B, length: B.count*4)!
    let bC = dev.makeBuffer(bytes: C, length: C.count*4)!
    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(bA, offset: 0, index: 0); enc.setBuffer(bB, offset: 0, index: 1); enc.setBuffer(bC, offset: 0, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: tiles, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let G = Array(UnsafeBufferPointer(start: bC.contents().bindMemory(to: Float.self, capacity: C.count), count: C.count))

    // candidates for c[i][j] = C[i][j] + sum_k A[i][k] B[k][j]
    typealias F = ([Float], [Float], Float) -> Float  // (a row, b column, c)
    let cands: [(String, F)] = [
        ("seq fma k=0..7", { a, b, c in var s = c; for k in 0..<8 { s = fmaf(a[k], b[k], s) }; return s }),
        ("seq fma k=7..0", { a, b, c in var s = c; for k in (0..<8).reversed() { s = fmaf(a[k], b[k], s) }; return s }),
        ("seq mul+add", { a, b, c in var s = c; for k in 0..<8 { s = s + a[k]*b[k] }; return s }),
        ("fma chain from 0, then + c", { a, b, c in var s: Float = 0; for k in 0..<8 { s = fmaf(a[k], b[k], s) }; return c + s }),
        ("pairwise products, then + c", { a, b, c in let p = (0..<8).map { a[$0]*b[$0] }; return c + (((p[0]+p[1])+(p[2]+p[3]))+((p[4]+p[5])+(p[6]+p[7]))) }),
        ("c + pairwise", { a, b, c in let p = (0..<8).map { a[$0]*b[$0] }; return (((c+p[0])+p[1]) + (p[2]+p[3])) + ((p[4]+p[5])+(p[6]+p[7])) }),
        ("exact sum, one rounding", { a, b, c in var s = Double(c); for k in 0..<8 { s += Double(a[k])*Double(b[k]) }; return Float(s) }),
        ("two fma chains (even/odd k), then add", { a, b, c in var e = c, o: Float = 0; for k in stride(from: 0, to: 8, by: 2) { e = fmaf(a[k], b[k], e); o = fmaf(a[k+1], b[k+1], o) }; return e + o }),
        ("fma 4-chains k0..3 onto c, k4..7 from 0", { a, b, c in var x = c, y: Float = 0; for k in 0..<4 { x = fmaf(a[k], b[k], x); y = fmaf(a[k+4], b[k+4], y) }; return x + y }),
    ]
    print("== \(mode): \(tiles*64) outputs")
    for (name, f) in cands {
        var bad = 0
        for t in 0..<tiles {
            for i in 0..<8 {
                let a = (0..<8).map { Float(A[t*64 + i*8 + $0]) }
                for j in 0..<8 {
                    let b = (0..<8).map { B[t*64 + $0*8 + j] }
                    if f(a, b, C[t*64 + i*8 + j]).bitPattern != G[t*64 + i*8 + j].bitPattern { bad += 1 }
                }
            }
        }
        print(String(format: "  %-42@ %8d mismatches (%.3f%%)", name as NSString, bad, 100*Double(bad)/Double(tiles*64)))
    }
}
