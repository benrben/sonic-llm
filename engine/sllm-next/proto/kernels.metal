// sllm-next prototype kernels. Every token in flight (a prompt token being read or an agent's next token being written) is
// one row; every kernel takes R rows and computes each row with the same operations in the same order whatever R is, so a
// row's result never depends on how many agents share the step.
#include <metal_stdlib>
using namespace metal;

struct block_q4_0 { half d; uchar qs[16]; };
struct block_q4_1 { half d; half m; uchar qs[16]; };
struct block_q8_0 { half d; char qs[32]; };

// ---- weight matmul: Y[r][c] (+)= sum_k X[r][k] * W[c][k] -------------------------------------------------------------
// One threadgroup = 4 simdgroups = 32 weight rows (8 each) x up to 32 token rows (4 tiles of 8). K advances one 32-wide
// block at a time: the block is dequantized to float in threadgroup memory, the X tile is staged beside it, and every
// 8x8 product is a simdgroup multiply-accumulate (a sequential fma chain over its 8 k). The sum order of an output is the
// k order, the same for every R and every tile, which is what makes rows exact.
struct mm_args { int K; int M; int R; int ldx; int ldy; int acc; };

template <int FMT>
inline void mm_deq(device const uchar * w, device const float * s, int K, int c, int kb, int p, threadgroup float * ws) {
    // thread part p (0..3) of weight row c, k block kb: 8 values into ws (that row's 32 floats)
    if (FMT == 0) {
        device const block_q4_0 & b = ((device const block_q4_0 *) w)[(size_t) c * (K / 32) + kb];
        const float d = b.d;
        for (int i = 0; i < 4; ++i) {
            const uchar q = b.qs[p * 4 + i];
            ws[p * 4 + i] = (float((int) (q & 15) - 8)) * d;
            ws[p * 4 + i + 16] = (float((int) (q >> 4) - 8)) * d;
        }
    } else if (FMT == 1) {
        device const block_q4_1 & b = ((device const block_q4_1 *) w)[(size_t) c * (K / 32) + kb];
        const float d = b.d, m = b.m;
        for (int i = 0; i < 4; ++i) {
            const uchar q = b.qs[p * 4 + i];
            ws[p * 4 + i] = float(q & 15) * d + m;
            ws[p * 4 + i + 16] = float(q >> 4) * d + m;
        }
    } else if (FMT == 3) {  // f32 weights
        device const float * q = (device const float *) w + (size_t) c * K + kb * 32 + p * 8;
        for (int i = 0; i < 8; ++i) ws[p * 8 + i] = q[i];
    } else if (FMT == 4) {  // bf16 weights
        device const ushort * q = (device const ushort *) w + (size_t) c * K + kb * 32 + p * 8;
        for (int i = 0; i < 8; ++i) ws[p * 8 + i] = as_type<float>((uint) q[i] << 16);
    } else {  // q8s16: int8 values, one f32 scale per 16
        device const char * q = (device const char *) w + (size_t) c * K + kb * 32 + p * 8;
        const float sc = s[(size_t) c * (K / 16) + kb * 2 + p / 2];
        for (int i = 0; i < 8; ++i) {
            ws[p * 8 + i] = sc * float(q[i]);
        }
    }
}

template <int FMT>
kernel void mm(constant mm_args & a [[buffer(0)]], device const float * x [[buffer(1)]], device const uchar * w [[buffer(2)]],
               device const float * s [[buffer(3)]], device float * y [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
               ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ws[32 * 32];
    threadgroup float xs[32 * 32];
    const int c0 = tg.x * 32, r0 = tg.y * 32;
    const int nt = min(4, (a.R - r0 + 7) / 8);
    simdgroup_float8x8 acc[4];
    for (int i = 0; i < 4; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    }
    const int wc = t / 4, wp = t % 4;  // dequant: weight row, part
    for (int kb = 0; kb < a.K / 32; ++kb) {
        mm_deq<FMT>(w, s, a.K, c0 + wc, kb, wp, ws + wc * 32);
        for (int i = t; i < 32 * 32; i += 128) {
            const int r = i / 32, k = i % 32;
            xs[i] = r0 + r < a.R ? x[(size_t) (r0 + r) * a.ldx + kb * 32 + k] : 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_float8x8 wb;
            simdgroup_load(wb, ws + sg * 8 * 32 + kk * 8, 32, ulong2(0, 0), true);
            for (int i = 0; i < nt; ++i) {
                simdgroup_float8x8 xa;
                simdgroup_load(xa, xs + i * 8 * 32 + kk * 8, 32);
                simdgroup_multiply_accumulate(acc[i], xa, wb, acc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // through threadgroup memory to the rows that exist
    for (int i = 0; i < nt; ++i) {
        simdgroup_store(acc[i], xs + i * 8 * 32 + sg * 8, 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < 32 * 32; i += 128) {
        const int r = i / 32, c = i % 32;
        if (r0 + r < a.R && r < nt * 8) {
            device float * o = y + (size_t) (r0 + r) * a.ldy + c0 + c;
            *o = a.acc ? *o + xs[i] : xs[i];
        }
    }
}

template [[host_name("mm_q4_0")]] kernel void mm<0>(constant mm_args &, device const float *, device const uchar *, device const float *,
                                                    device float *, uint2, ushort, ushort);
template [[host_name("mm_q4_1")]] kernel void mm<1>(constant mm_args &, device const float *, device const uchar *, device const float *,
                                                    device float *, uint2, ushort, ushort);
template [[host_name("mm_f32w")]] kernel void mm<3>(constant mm_args &, device const float *, device const uchar *, device const float *,
                                                    device float *, uint2, ushort, ushort);
template [[host_name("mm_bf16")]] kernel void mm<4>(constant mm_args &, device const float *, device const uchar *, device const float *,
                                                    device float *, uint2, ushort, ushort);
template [[host_name("mm_q8s16")]] kernel void mm<2>(constant mm_args &, device const float *, device const uchar *, device const float *,
                                                     device float *, uint2, ushort, ushort);

// ---- token embedding from the q8s16 copy of token_embd ---------------------------------------------------------------
struct embed_args { int n; };
kernel void embed(constant embed_args & a [[buffer(0)]], device const int * tok [[buffer(1)]], device const char * q [[buffer(2)]],
                  device const float * s [[buffer(3)]], device float * x [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                  ushort t [[thread_index_in_threadgroup]]) {
    const size_t row = (size_t) tok[r];
    for (int i = t; i < a.n; i += 256) {
        x[(size_t) r * a.n + i] = s[row * (a.n / 16) + i / 16] * float(q[row * a.n + i]);
    }
}

// ---- RMS norm of each row, times a weight -------------------------------------------------------------------------
struct norm_args { int n; float eps; int ldx; int ldy; };
kernel void rmsnorm(constant norm_args & a [[buffer(0)]], device const float * x [[buffer(1)]], device const float * w [[buffer(2)]],
                    device float * y [[buffer(3)]], uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    device const float4 * xr = (device const float4 *) (x + (size_t) r * a.ldx);
    float s = 0.f;
    for (int i = t; i < a.n / 4; i += 256) s += dot(xr[i], xr[i]);
    s = simd_sum(s);
    if (ln == 0) {
        part[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int i = 0; i < 8; ++i) {
        tot += part[i];
    }
    const float sc = 1.0f / sqrt(tot / a.n + a.eps);
    device float4 * yr = (device float4 *) (y + (size_t) r * a.ldy);
    for (int i = t; i < a.n / 4; i += 256) yr[i] = (xr[i] * sc) * ((device const float4 *) w)[i];
}

// ---- per-head RMS norm of q and k, NEOX rope, k and v into the cache ---------------------------------------------
// One threadgroup per row, one head per simdgroup turn. A lane holds dims l, l+32 (rotated with l+64, l+96).
// mode (0 = Qwen3): 1 no q / k norm (Llama), 2 frequency factors ff (Llama 3.1's rope_freqs), 4 adjacent pairs (llama.cpp's
// NORM rope: dims 2j, 2j + 1; GGUF permutes Llama's q / k for it) instead of NEOX's j, j + 64, 8 no rotation (NoPE layers),
// 16 biases qb / kb / vb added to q, k and v first (Qwen2)
struct qkr_args { int n_head; int n_kv; int ctx; int n_seq; float eps; float base; int ldq; int ldk; int mode; float mscale; };  // mscale: qk_rope_d's
constant int QKR_TGS = 8;  // a row's threadgroups (sn.m's dispatch), 4 heads each in turn: one threadgroup a row ran all heads
kernel void qk_rope(constant qkr_args & a [[buffer(0)]], device float * q [[buffer(1)]], device const float * k [[buffer(2)]],
                    device const float * v [[buffer(3)]], device const float * qw [[buffer(4)]], device const float * kw [[buffer(5)]],
                    device half * kc [[buffer(6)]], device half * vc [[buffer(7)]], device const int * seq [[buffer(8)]],
                    device const int * pos [[buffer(9)]], device const float * ff [[buffer(10)]], device const int * pt [[buffer(11)]],
                    device const float * qb [[buffer(12)]], device const float * kb [[buffer(13)]], device const float * vb [[buffer(14)]],
                    uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort ln [[thread_index_in_simdgroup]]) {
    const int r = tg.x, p = pos[r];
    const size_t cell = ((size_t) pt[(size_t) seq[r] * (a.ctx / 64) + p / 64] * 64 + p % 64) * a.n_kv * 128;  // pt: the sequence's pages
    float th0 = float(p) * pow(a.base, -float(2 * ln) / 128.f);
    float th1 = float(p) * pow(a.base, -float(2 * (ln + 32)) / 128.f);
    if (a.mode & 2) { th0 /= ff[ln]; th1 /= ff[ln + 32]; }
    if (a.mode & 8) { th0 = 0.f; th1 = 0.f; }  // a layer without rope (SmolLM3's every fourth): cos 1, sin 0 leave q and k as they are
    // the lane's two pairs: (ln, ln + 64) and (ln + 32, ln + 96), or adjacent (2 ln, 2 ln + 1) and (64 + 2 ln, 65 + 2 ln)
    const bool adj = (a.mode & 4) != 0;
    const int i0 = adj ? 2 * ln : ln, i1 = adj ? 64 + 2 * ln : ln + 32, i2 = adj ? 2 * ln + 1 : ln + 64, i3 = adj ? 65 + 2 * ln : ln + 96;
    const float c0 = cos(th0), s0 = sin(th0), c1 = cos(th1), s1 = sin(th1);
    for (int h = tg.y * 4 + sg; h < a.n_head + a.n_kv; h += 4 * QKR_TGS) {  // threadgroup tg.y of QKR_TGS: heads 4 tg.y + sg, ..
        const bool isq = h < a.n_head;
        device const float * src = isq ? q + (size_t) r * a.ldq + h * 128 : k + (size_t) r * a.ldk + (h - a.n_head) * 128;
        device const float * w = isq ? qw : kw;
        float x0 = src[i0], x1 = src[i1], x2 = src[i2], x3 = src[i3];
        if (a.mode & 16) {
            device const float * b = isq ? qb + h * 128 : kb + (h - a.n_head) * 128;
            x0 += b[i0]; x1 += b[i1]; x2 += b[i2]; x3 += b[i3];
        }
        if (!(a.mode & 1)) {
            const float ss = simd_sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3);
            const float sc = 1.0f / sqrt(ss / 128.f + a.eps);
            x0 = (x0 * sc) * w[i0];
            x1 = (x1 * sc) * w[i1];
            x2 = (x2 * sc) * w[i2];
            x3 = (x3 * sc) * w[i3];
        }
        const float y0 = x0 * c0 - x2 * s0, y2 = x0 * s0 + x2 * c0;
        const float y1 = x1 * c1 - x3 * s1, y3 = x1 * s1 + x3 * c1;
        if (isq) {
            device float * o = q + (size_t) r * a.ldq + h * 128;
            o[i0] = y0; o[i1] = y1; o[i2] = y2; o[i3] = y3;
        } else {
            device half * o = kc + cell + (h - a.n_head) * 128;
            o[i0] = half(y0); o[i1] = half(y1); o[i2] = half(y2); o[i3] = half(y3);
        }
    }
    for (int i = t; i < a.n_kv * 128; i += 128) {
        const float x = v[(size_t) r * a.ldk + i];
        vc[cell + i] = half((a.mode & 16) ? x + vb[i] : x);
    }
}

// qk_rope for another head size D (Phi-3's 96): pair j = ln + 32 t (j < D / 2) of every head, NEOX (j, j + D / 2) or adjacent
// (2 j, 2 j + 1), the modes of qk_rope, cos / sin times mscale (llama.cpp's attn_factor: Phi-3's rope.scaling.attn_factor)
template <int D>
kernel void qk_rope_d(constant qkr_args & a [[buffer(0)]], device float * q [[buffer(1)]], device const float * k [[buffer(2)]],
                      device const float * v [[buffer(3)]], device const float * qw [[buffer(4)]], device const float * kw [[buffer(5)]],
                      device half * kc [[buffer(6)]], device half * vc [[buffer(7)]], device const int * seq [[buffer(8)]],
                      device const int * pos [[buffer(9)]], device const float * ff [[buffer(10)]], device const int * pt [[buffer(11)]],
                      device const float * qb [[buffer(12)]], device const float * kb [[buffer(13)]], device const float * vb [[buffer(14)]],
                      uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort ln [[thread_index_in_simdgroup]]) {
    constexpr int NP = (D / 2 + 31) / 32;
    const int r = tg.x, p = pos[r];
    const size_t cell = ((size_t) pt[(size_t) seq[r] * (a.ctx / 64) + p / 64] * 64 + p % 64) * a.n_kv * D;
    const bool adj = (a.mode & 4) != 0;
    float cs[NP], sn[NP];
    int ia[NP], ib[NP];
    for (int u = 0; u < NP; ++u) {
        const int j = min(ln + 32 * u, D / 2 - 1);
        float th = float(p) * pow(a.base, -float(2 * j) / float(D));
        if (a.mode & 2) th /= ff[j];
        if (a.mode & 8) th = 0.f;
        cs[u] = cos(th) * a.mscale; sn[u] = sin(th) * a.mscale;
        ia[u] = adj ? 2 * j : j; ib[u] = adj ? 2 * j + 1 : j + D / 2;
    }
    for (int h = tg.y * 4 + sg; h < a.n_head + a.n_kv; h += 4 * QKR_TGS) {
        const bool isq = h < a.n_head;
        device const float * src = isq ? q + (size_t) r * a.ldq + h * D : k + (size_t) r * a.ldk + (h - a.n_head) * D;
        device const float * b = isq ? qb + h * D : kb + (h - a.n_head) * D;
        float xa[NP], xb[NP], ss = 0.f;
        for (int u = 0; u < NP; ++u) {
            const bool ok = ln + 32 * u < D / 2;
            xa[u] = ok ? src[ia[u]] : 0.f; xb[u] = ok ? src[ib[u]] : 0.f;
            if (a.mode & 16) { xa[u] += ok ? b[ia[u]] : 0.f; xb[u] += ok ? b[ib[u]] : 0.f; }
        }
        if (!(a.mode & 1)) {
            for (int u = 0; u < NP; ++u) ss += xa[u] * xa[u];
            for (int u = 0; u < NP; ++u) ss += xb[u] * xb[u];
            const float sc = 1.0f / sqrt(simd_sum(ss) / float(D) + a.eps);
            device const float * w = isq ? qw : kw;
            for (int u = 0; u < NP; ++u) { xa[u] = (xa[u] * sc) * w[ia[u]]; xb[u] = (xb[u] * sc) * w[ib[u]]; }
        }
        for (int u = 0; u < NP; ++u) {
            if (ln + 32 * u >= D / 2) continue;
            const float ya = xa[u] * cs[u] - xb[u] * sn[u], yb = xa[u] * sn[u] + xb[u] * cs[u];
            if (isq) { device float * o = q + (size_t) r * a.ldq + h * D; o[ia[u]] = ya; o[ib[u]] = yb; }
            else { device half * o = kc + cell + (h - a.n_head) * D; o[ia[u]] = half(ya); o[ib[u]] = half(yb); }
        }
    }
    for (int i = t; i < a.n_kv * D; i += 128) {
        const float x = v[(size_t) r * a.ldk + i];
        vc[cell + i] = half((a.mode & 16) ? x + vb[i] : x);
    }
}
template [[host_name("qk_rope_d96")]] kernel void qk_rope_d<96>(constant qkr_args &, device float *, device const float *, device const float *,
    device const float *, device const float *, device half *, device half *, device const int *, device const int *, device const float *,
    device const int *, device const float *, device const float *, device const float *, uint2, ushort, ushort, ushort);

// ---- attention: each row against its own sequence's cells 0..pos ---------------------------------------------------
// One threadgroup per (row, kv head), one query head per simdgroup (the GQA group shares the threadgroup's K/V reads
// through the cache). Online softmax over the cells in order.
struct attn_args { int n_head; int n_kv; int ctx; float scale; int ldq; int ldo; };
kernel void attn(constant attn_args & a [[buffer(0)]], device const float * q [[buffer(1)]], device const half * kc [[buffer(2)]],
                 device const half * vc [[buffer(3)]], device const int * seq [[buffer(4)]], device const int * pos [[buffer(5)]],
                 device float * o [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort ln [[thread_index_in_simdgroup]]) {
    const int r = tg.x, kvh = tg.y, g = a.n_head / a.n_kv;
    const int h = kvh * g + sg;
    const int p = pos[r];
    const float4 qv = *((device const float4 *) (q + (size_t) r * a.ldq + h * 128) + ln);
    device const half4 * kb = (device const half4 *) (kc + (size_t) seq[r] * a.ctx * a.n_kv * 128 + kvh * 128) + ln;
    device const half4 * vb = (device const half4 *) (vc + (size_t) seq[r] * a.ctx * a.n_kv * 128 + kvh * 128) + ln;
    const int stride = a.n_kv * 128 / 4;
    float m = -INFINITY, l = 0.f;
    float4 acc = 0.f;
    for (int c = 0; c <= p; ++c) {
        const float sc = simd_sum(dot(qv, float4(kb[c * stride]))) * a.scale;
        const float mn = max(m, sc);
        const float e0 = exp(m - mn), e1 = exp(sc - mn);
        acc = acc * e0 + float4(vb[c * stride]) * e1;
        l = l * e0 + e1;
        m = mn;
    }
    *((device float4 *) (o + (size_t) r * a.ldo + h * 128) + ln) = acc / l;
}

// ---- small row ops ---------------------------------------------------------------------------------------------------
struct ew_args { int n; };
kernel void swiglu(constant ew_args & a [[buffer(0)]], device float * g [[buffer(1)]], device const float * u [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if ((int) i < a.n) {
        const float x = g[i];
        g[i] = (x / (1.0f + exp(-x))) * u[i];
    }
}

struct gather_args { int n; };
kernel void gather(constant gather_args & a [[buffer(0)]], device const float * x [[buffer(1)]], device const int * idx [[buffer(2)]],
                   device float * y [[buffer(3)]], uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
    for (int i = t; i < a.n; i += 256) {
        y[(size_t) r * a.n + i] = x[(size_t) idx[r] * a.n + i];
    }
}

// the guess head (MTP): each row's hidden-state input: row src[r] of hn, its sequence's kept one (src -1), or none (-2)
kernel void mtp_hsel(constant gather_args & a [[buffer(0)]], device const float * hn [[buffer(1)]], device const float * pend [[buffer(2)]],
                     device const int * src [[buffer(3)]], device const int * seq [[buffer(4)]], device float * y [[buffer(5)]],
                     uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
    const int s = src[r];
    for (int i = t; i < a.n; i += 256) {
        y[(size_t) r * a.n + i] = s >= 0 ? hn[(size_t) s * a.n + i] : s == -1 ? pend[(size_t) seq[r] * a.n + i] : 0.f;
    }
}
// each run's last row of hn into its sequence's kept hidden state
kernel void mtp_keep(constant gather_args & a [[buffer(0)]], device const float * hn [[buffer(1)]], device const int4 * runs [[buffer(2)]],
                     device const int * seq [[buffer(3)]], device float * pend [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    if ((int) g.x >= a.n) return;
    const int4 rn = runs[g.y];
    pend[(size_t) seq[rn.x] * a.n + g.x] = hn[(size_t) (rn.x + rn.y - 1) * a.n + g.x];
}

// latent rows (sn_latent): row idx[j] of x takes row j of v, a vector in place of its token's embedding
kernel void latin(constant gather_args & a [[buffer(0)]], device const float * v [[buffer(1)]], device const int * idx [[buffer(2)]],
                  device float * x [[buffer(3)]], uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
    for (int i = t; i < a.n; i += 256) {
        x[(size_t) idx[r] * a.n + i] = v[(size_t) r * a.n + i];
    }
}

// greedy: the first index of the largest logit
struct argmax_args { int n; int nban; int ban[8]; };  // nban: tokens a row with skip[r] never takes (its request may not end)
kernel void argmax(constant argmax_args & a [[buffer(0)]], device const float * x [[buffer(1)]], device int * out [[buffer(2)]],
                   constant uchar * skip [[buffer(3)]], uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup int bi[32];
    device const float * xr = x + (size_t) r * a.n;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    const int nb = skip[r] ? a.nban : 0;
    for (int i = t; i < a.n; i += 1024) {
        bool ban = false;
        for (int b = 0; b < nb; ++b) ban = ban || i == a.ban[b];
        if (xr[i] > best && !ban) { best = xr[i]; idx = i; }
    }
    for (int off = 16; off > 0; off /= 2) {
        const float ov = simd_shuffle_down(best, off);
        const int oi = simd_shuffle_down(idx, off);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if (ln == 0) { bv[sg] = best; bi[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        for (int i = 1; i < 32; ++i) {
            if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
        }
        out[r] = idx;
    }
}

// argmax plus the numbers for token probabilities (sn_want_probs steps only). The chosen token comes from the same loop, ban
// rule and tie rule as argmax. Over the whole row (banned words too): the two best scores (ties: the lower id) and the
// log-sum-exp. The record of row r is 8 words at pr[8 * r]: chosen id, chosen score, best id, best score, second id, second
// score, log-sum-exp, 0 (ids as ints, scores as float bits).
struct top2_t { float m1; int i1; float m2; int i2; };
// the better of two (score, id): the higher score, the lower id among equal ones
static inline bool top2_better(float v, int i, float w, int j) { return v > w || (v == w && i < j); }
// the two best of two sets of disjoint ids
static inline top2_t top2_merge(top2_t a, top2_t b) {
    top2_t w = a, l = b;
    if (top2_better(b.m1, b.i1, a.m1, a.i1)) { w = b; l = a; }
    top2_t o = w;
    if (top2_better(l.m1, l.i1, w.m2, w.i2)) { o.m2 = l.m1; o.i2 = l.i1; }
    return o;
}
kernel void argmax_p(constant argmax_args & a [[buffer(0)]], device const float * x [[buffer(1)]], device int * out [[buffer(2)]],
                     constant uchar * skip [[buffer(3)]], device uint * pr [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                     ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup int bi[32];
    threadgroup float tv1[32], tv2[32], ts[32];
    threadgroup int ti1[32], ti2[32];
    threadgroup float row_max;
    device const float * xr = x + (size_t) r * a.n;
    // the chosen token: argmax's loop
    float best = -INFINITY;
    int idx = 0x7fffffff;
    // the two best scores of the whole row
    top2_t tp;
    tp.m1 = -INFINITY; tp.i1 = 0x7fffffff; tp.m2 = -INFINITY; tp.i2 = 0x7fffffff;
    const int nb = skip[r] ? a.nban : 0;
    for (int i = t; i < a.n; i += 1024) {
        bool ban = false;
        for (int b = 0; b < nb; ++b) ban = ban || i == a.ban[b];
        if (xr[i] > best && !ban) { best = xr[i]; idx = i; }
        // (this thread's ids rise, so an equal score never replaces an earlier one)
        if (xr[i] > tp.m1) { tp.m2 = tp.m1; tp.i2 = tp.i1; tp.m1 = xr[i]; tp.i1 = i; }
        else if (xr[i] > tp.m2) { tp.m2 = xr[i]; tp.i2 = i; }
    }
    for (int off = 16; off > 0; off /= 2) {
        const float ov = simd_shuffle_down(best, off);
        const int oi = simd_shuffle_down(idx, off);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
        top2_t o;
        o.m1 = simd_shuffle_down(tp.m1, off); o.i1 = simd_shuffle_down(tp.i1, off);
        o.m2 = simd_shuffle_down(tp.m2, off); o.i2 = simd_shuffle_down(tp.i2, off);
        tp = top2_merge(tp, o);
    }
    if (ln == 0) { bv[sg] = best; bi[sg] = idx; tv1[sg] = tp.m1; ti1[sg] = tp.i1; tv2[sg] = tp.m2; ti2[sg] = tp.i2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        for (int i = 1; i < 32; ++i) {
            if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
            top2_t o;
            o.m1 = tv1[i]; o.i1 = ti1[i]; o.m2 = tv2[i]; o.i2 = ti2[i];
            tp = top2_merge(tp, o);
        }
        out[r] = idx;
        row_max = tp.m1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the sum of exp(score - the row's largest score) over the whole row
    const float mx = row_max;
    float s = 0.0f;
    if (mx > -INFINITY) {
        for (int i = t; i < a.n; i += 1024) s += precise::exp(xr[i] - mx);
    }
    s = simd_sum(s);
    if (ln == 0) ts[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float z = 0.0f;
        for (int i = 0; i < 32; ++i) z += ts[i];
        device uint * p = pr + (size_t) r * 8;
        p[0] = as_type<uint>(idx);
        p[1] = as_type<uint>(best);
        p[2] = as_type<uint>(tp.i1);
        p[3] = as_type<uint>(tp.m1);
        p[4] = as_type<uint>(tp.i2);
        p[5] = as_type<uint>(tp.m2);
        p[6] = as_type<uint>(mx > -INFINITY ? mx + precise::log(z) : -INFINITY);
        p[7] = 0u;
    }
}

// ---- token embedding straight from Q6_K blocks (ggml's numbers: (d * scale) * q), times a.scale (Gemma: sqrt(n_embd)) --
struct block_q6_K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };
struct embs_args { int n; float scale; int staged; };  // staged: w holds the step's rows (row r at r), not the table
kernel void embed_q6k(constant embs_args & a [[buffer(0)]], device const int * tok [[buffer(1)]], device const block_q6_K * w [[buffer(2)]],
                      device float * x [[buffer(3)]], uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
    device const block_q6_K * row = w + (size_t) (a.staged ? (int) r : tok[r]) * (a.n / 256);
    for (int i = t; i < a.n; i += 256) {
        device const block_q6_K & b = row[i / 256];
        const int j = i % 256, n = j / 128, jj = j % 128, qu = jj / 32, l = jj % 32;
        const uchar lb = b.ql[n * 64 + l + (qu & 1) * 32];
        const int lo = qu < 2 ? (lb & 0xF) : (lb >> 4);
        const int hi = (b.qh[n * 32 + l] >> (2 * qu)) & 3;
        const int q = (lo | (hi << 4)) - 32;
        x[(size_t) r * a.n + i] = ((float(b.d) * float(b.scales[n * 8 + l / 16 + 2 * qu])) * float(q)) * a.scale;
    }
}

kernel void embed_q4_0(constant embs_args & a [[buffer(0)]], device const int * tok [[buffer(1)]], device const block_q4_0 * w [[buffer(2)]],
                       device float * x [[buffer(3)]], uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
    device const block_q4_0 * row = w + (size_t) (a.staged ? (int) r : tok[r]) * (a.n / 32);
    for (int i = t; i < a.n; i += 256) {
        device const block_q4_0 & b = row[i / 32];
        const int j = i % 32, q = j < 16 ? (b.qs[j] & 15) : (b.qs[j - 16] >> 4);
        x[(size_t) r * a.n + i] = (float(b.d) * float(q - 8)) * a.scale;
    }
}
kernel void embed_q8_0(constant embs_args & a [[buffer(0)]], device const int * tok [[buffer(1)]], device const block_q8_0 * w [[buffer(2)]],
                       device float * x [[buffer(3)]], uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
    device const block_q8_0 * row = w + (size_t) (a.staged ? (int) r : tok[r]) * (a.n / 32);
    for (int i = t; i < a.n; i += 256) x[(size_t) r * a.n + i] = (float(row[i / 32].d) * float(row[i / 32].qs[i % 32])) * a.scale;
}

// ---- a guess's output layer through a sketch (sn.m sketch_build): one bit a weight and a size a row --------------------
struct sk_args { int V, E, per, nc; };
// the hidden state's partial sums: tab[byte b][value v] = the sum of h[8 b + i] over the set bits i of v
kernel void sk_tab(constant sk_args & a [[buffer(0)]], device const float * h [[buffer(1)]], device float * tab [[buffer(2)]],
                   uint2 g [[thread_position_in_grid]]) {
    float s = 0.f;
    for (int i = 0; i < 8; ++i) if ((g.x >> i) & 1) s += h[g.y * 8 + i];
    tab[g.y * 256 + g.x] = s;
}
// a chunk of `per` vocabulary rows a threadgroup (one simdgroup; lane l the word l of a row's E / 64: the sketch covers the
// first half of the dims): a row's score is its size times (2 x the sum of h over its set bits - the sum of h); the
// chunk's best 4 rows into cand
kernel void sk_pick(constant sk_args & a [[buffer(0)]], device const uint * bits [[buffer(1)]], device const float * rs [[buffer(2)]],
                    device const float * tab [[buffer(3)]], device int * cand [[buffer(4)]], uint c [[threadgroup_position_in_grid]],
                    ushort ln [[thread_index_in_simdgroup]]) {
    const int nw = a.E / 64;
    float tot = 0.f;
    for (int w = ln; w < nw; w += 32) for (int b = 0; b < 4; ++b) tot += tab[(w * 4 + b) * 256 + 255];
    const float H = simd_sum(tot);
    float bv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    int bi[4] = {-1, -1, -1, -1};
    const int r0 = c * a.per, r1 = min(r0 + a.per, a.V);
    for (int r = r0; r < r1; ++r) {
        float p = 0.f;
        for (int w = ln; w < nw; w += 32) {
            const uint m = bits[(size_t) r * nw + w];
            device const float * t = tab + (size_t) w * 4 * 256;
            p += t[m & 255] + t[256 + ((m >> 8) & 255)] + t[512 + ((m >> 16) & 255)] + t[768 + (m >> 24)];
        }
        const float s = rs[r] * (2.f * simd_sum(p) - H);
        if (s > bv[3]) {
            int k = 3;
            for (; k > 0 && s > bv[k - 1]; --k) { bv[k] = bv[k - 1]; bi[k] = bi[k - 1]; }
            bv[k] = s; bi[k] = r;
        }
    }
    if (ln == 0) for (int k = 0; k < 4; ++k) cand[4 * c + k] = bi[k] < 0 ? max(bi[0], 0) : bi[k];
}
// the candidates' real scores: row cand[c] of the q6_K head times h (a simdgroup a candidate; lane l the 16-weight groups
// l, l + 32, ..)
kernel void sk_exact(constant sk_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * h [[buffer(2)]],
                     device const int * cand [[buffer(3)]], device float * sc [[buffer(4)]], uint c [[threadgroup_position_in_grid]],
                     ushort ln [[thread_index_in_simdgroup]]) {
    device const block_q6_K * row = (device const block_q6_K *) (W + (size_t) cand[c] * (a.E / 256) * 210);
    float sum = 0.f;
    for (int g = ln; g < a.E / 16; g += 32) {
        const int sb = g / 16, j0 = (g % 16) * 16, n = j0 / 128, jj = j0 % 128, qu = jj / 32, l0 = jj % 32;
        device const block_q6_K & b = row[sb];
        device const uchar * ql = b.ql + n * 64 + l0 + (qu & 1) * 32;
        device const uchar * qh = b.qh + n * 32 + l0;
        float s_ = 0.f;
        for (int i = 0; i < 16; ++i) {
            const int lo = qu < 2 ? (ql[i] & 0xF) : (ql[i] >> 4);
            s_ += float((lo | (((qh[i] >> (2 * qu)) & 3) << 4)) - 32) * h[g * 16 + i];
        }
        sum += float(b.d) * float(b.scales[g % 16]) * s_;
    }
    const float v = simd_sum(sum);
    if (ln == 0) sc[c] = v;
}
// the best candidate's row id (the lowest id among equal scores) into out
kernel void sk_best(constant sk_args & a [[buffer(0)]], device const float * sc [[buffer(1)]], device const int * cand [[buffer(2)]],
                    device int * out [[buffer(3)]], ushort t [[thread_index_in_threadgroup]]) {
    threadgroup float bv[1024];
    threadgroup int bi[1024];
    float v = -INFINITY;
    int id = 1 << 30;
    for (int c = t; c < 4 * a.nc; c += 1024) if (sc[c] > v || (sc[c] == v && cand[c] < id)) { v = sc[c]; id = cand[c]; }
    bv[t] = v; bi[t] = id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        for (int i = 1; i < 1024; ++i) if (bv[i] > v || (bv[i] == v && bi[i] < id)) { v = bv[i]; id = bi[i]; }
        out[0] = id;
    }
}

// ---- matrix-unit matmuls (the qmm kernels SLLM wrote for its ggml build, phase2-qmm32/qmm32b, Metal 4 tensor API) ----
// One kernel per weight type for every row count: a token tile of 16 or 32 rows gives each row bitwise the same result,
// so a row never depends on how many rows share the step. Activations are rounded to half inside matmul2d.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct qmm_args { int K; int M; int N; int nb01; int ldx; int ldd; int ksplit; int flags; int estride; int ldh; };  // flags: 1 interleaved K, 2 accumulate,
// 4 / 8: D holds the gate, write silu(gate) * result / gelu(gate) * result (the gated unit in the up projection's epilogue),
// 16 grouped (an MoE layer's experts): tile tg.x is tiles[tg.x] = (expert, first row, rows) of rows gathered in expert order,
// the expert's matrix estride bytes apart; 32: the result in half into DH (rows ldh apart) instead of D (a gated unit's output
// as the next block matmul's input: no float copy written and converted back)
static inline float act_silu(float x) { return x / (1.0f + exp(-x)); }
static inline float act_gelu(float x) { return 0.5f * x * (1.0f + precise::tanh(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x))); }
static inline float qmm_epi(int flags, float old, float s) {
    return (flags & 2) ? old + s : (flags & 4) ? act_silu(old) * s : (flags & 8) ? act_gelu(old) * s : s;
}

struct raw_ks_q4_0 { half d; ushort q0, q1; };
struct raw_ks_q4_1 { half d, m; ushort q0, q1; };
template <typename BQ> struct raw_ks_of { typedef raw_ks_q4_0 type; };
template <> struct raw_ks_of<block_q4_1> { typedef raw_ks_q4_1 type; };
static inline void ld_ks(device const block_q4_0 * row, int kk, int k0, thread raw_ks_q4_0 & r) {
    device const block_q4_0 & b = row[kk / 32];
    device const ushort * q = (device const ushort *) (b.qs + k0);
    r.d = b.d; r.q0 = q[0]; r.q1 = q[1];
}
static inline void ld_ks(device const block_q4_1 * row, int kk, int k0, thread raw_ks_q4_1 & r) {
    device const block_q4_1 & b = row[kk / 32];
    device const ushort * q = (device const ushort *) (b.qs + k0);
    r.d = b.d; r.m = b.m; r.q0 = q[0]; r.q1 = q[1];
}
static inline void cv_ks(thread const raw_ks_q4_0 & r, thread half4 & lo, thread half4 & hi) {
    const uint q = (uint) r.q0 | ((uint) r.q1 << 16);
    const uint4 b = uint4(q, q >> 8, q >> 16, q >> 24) & 0xFF;
    lo = (half4(b & 0xF) - 8.h) * r.d;
    hi = (half4(b >> 4) - 8.h) * r.d;
}
static inline void cv_ks(thread const raw_ks_q4_1 & r, thread half4 & lo, thread half4 & hi) {
    const uint q = (uint) r.q0 | ((uint) r.q1 << 16);
    const uint4 b = uint4(q, q >> 8, q >> 16, q >> 24) & 0xFF;
    lo = half4(b & 0xF) * r.d + r.m;
    hi = half4(b >> 4) * r.d + r.m;
}

// Q8_0: the lane's 4 weights k0.. (lo) and k0 + 16.. (hi) are bytes of the block as they are
struct raw_ks_q8_0 { half d; ushort l0, l1, h0, h1; };
template <> struct raw_ks_of<block_q8_0> { typedef raw_ks_q8_0 type; };
static inline void ld_ks(device const block_q8_0 * row, int kk, int k0, thread raw_ks_q8_0 & r) {
    device const block_q8_0 & b = row[kk / 32];
    device const ushort * lo = (device const ushort *) (b.qs + k0);
    device const ushort * hi = (device const ushort *) (b.qs + k0 + 16);
    r.d = b.d; r.l0 = lo[0]; r.l1 = lo[1]; r.h0 = hi[0]; r.h1 = hi[1];
}
static inline void cv_ks(thread const raw_ks_q8_0 & r, thread half4 & lo, thread half4 & hi) {
    lo = half4(as_type<char4>((uint) r.l0 | ((uint) r.l1 << 16))) * r.d;
    hi = half4(as_type<char4>((uint) r.h0 | ((uint) r.h1 << 16))) * r.d;
}

// MXFP4 (gpt-oss's experts): 32 weights a 17-byte block, an E8M0 scale (2^(e - 127)) and q4_0's nibble order of FP4 E2M1
// values (0, .5, 1, 1.5, 2, 3, 4, 6, signed). Blocks sit a byte apart, so byte loads. An E2M1 nibble shifted into a half's
// sign, exponent and top mantissa bits is its value times 2^-14 (exponent 0: the half's subnormals), 2 nibbles a half2.
struct block_mxfp4 { uchar e; uchar qs[16]; };
static inline float mx_scale(uchar e) { return as_type<float>(e ? (uint) e << 23 : 0x00400000u) * 16384.0f; }  // times 2^14
static inline half2 mx_h2(uint n) { return as_type<half2>(((n & 0x00070007u) << 9) | ((n & 0x00080008u) << 12)); }  // nibbles at bits 0, 16
static inline uint mx_u32(device const uchar * p) { return (uint) p[0] | ((uint) p[1] << 8) | ((uint) p[2] << 16) | ((uint) p[3] << 24); }
// a word's 4 bytes -> the low nibbles' values (lo) and the high nibbles' (hi), times 2^-14
static inline void mx_4(uint q, thread float4 & lo, thread float4 & hi) {
    const half2 a = mx_h2(q), b = mx_h2(q >> 8), c = mx_h2(q >> 4), d = mx_h2(q >> 12);
    lo = float4(a.x, b.x, a.y, b.y); hi = float4(c.x, d.x, c.y, d.y);
}
struct raw_ks_mxfp4 { float d; uint q; };
template <> struct raw_ks_of<block_mxfp4> { typedef raw_ks_mxfp4 type; };
static inline void ld_ks(device const block_mxfp4 * row, int kk, int k0, thread raw_ks_mxfp4 & r) {
    device const block_mxfp4 & b = row[kk / 32];
    r.d = mx_scale(b.e); r.q = mx_u32(b.qs + k0);
}
static inline void cv_ks(thread const raw_ks_mxfp4 & r, thread half4 & lo, thread half4 & hi) {
    float4 l, h;
    mx_4(r.q, l, h);
    lo = half4(l * r.d); hi = half4(h * r.d);
}

// A block matmul's input rows in half (what matmul2d takes from a float input anyway; streamed as half it runs at the matrix
// units' rate, measured 33 against 26 TFLOP/s): rounded to nearest, clamped to half's range. One float4 per thread.
struct f2h_args { int K; int ldx; };
// a gated unit's two halves computed side by side (SN_CONCSMALL): y = act(gate in y) * up, the expression of qmm_epi / mvm's
// epilogue (act 2 silu, 3 gelu), so the same bits as the up projection's epilogue
kernel void glu_comb(constant int4 & a [[buffer(0)]], device float * y [[buffer(1)]], device const float * u [[buffer(2)]],
                     uint2 id [[thread_position_in_grid]]) {
    const int M = a.x, ldy = a.y, act = a.z, R = a.w;
    if ((int) id.x >= M || (int) id.y >= R) return;
    device float * o = y + (size_t) id.y * ldy + id.x;
    const float s = u[(size_t) id.y * M + id.x];
    *o = act == 2 ? act_silu(*o) * s : act_gelu(*o) * s;
}

kernel void f2h(constant f2h_args & a [[buffer(0)]], device const float * x [[buffer(1)]], device half * y [[buffer(2)]],
                uint2 gid [[thread_position_in_grid]]) {
    if ((int) gid.x * 4 >= a.K) return;
    const float4 v = *((device const float4 *) (x + (size_t) gid.y * a.ldx) + gid.x);
    *((device half4 *) (y + (size_t) gid.y * a.K) + gid.x) = half4(clamp(v, -65504.0f, 65504.0f));
}

// ---- llama.cpp's K-quants in the block matmul: Q4_K (a 6-bit scale and min per 32 weights) and Q6_K (an 8-bit scale per
// 16), 256-weight blocks. A lane's 4 + 4 weights of a 32-weight piece (k0 .., k0 + 16 ..) as ggml dequantizes them: the
// value in float (d * scale * q - dmin * min), then half. 2-byte loads: a Q6_K block is 210 bytes, so blocks sit 2 apart.
struct block_q4_K { half d; half dmin; uchar scales[12]; uchar qs[128]; };
static inline void q4k_sm(int j, device const uchar * q, thread int & sc, thread int & m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
struct raw_ks_q4_K { half dl, ml; uint a, b; ushort sh; };
template <> struct raw_ks_of<block_q4_K> { typedef raw_ks_q4_K type; };
// one 16-byte load for d, dmin and the 12 scale bytes (a block's first 16 bytes; blocks are 144 bytes, 16-aligned), the
// piece's scale and min unpacked with shifts (llama.cpp's get_scale_min_k4), two 4-byte loads for its weights
static inline void ld_ks(device const block_q4_K * row, int kk, int k0, thread raw_ks_q4_K & r) {
    device const block_q4_K & b = row[kk / 256];
    const int sub = (kk % 256) / 32;
    const uint4 h = *((device const uint4 *) &b);
    uint sc, m;
    if (sub < 4) {
        sc = (h.y >> (8 * sub)) & 63; m = (h.z >> (8 * sub)) & 63;
    } else {
        const int i = 8 * (sub - 4);
        sc = ((h.w >> i) & 0xF) | (((h.y >> (i + 6)) & 3) << 4);
        m = ((h.w >> (i + 4)) & 0xF) | (((h.z >> (i + 6)) & 3) << 4);
    }
    const half2 dd = as_type<half2>(h.x);
    r.dl = half(float(dd.x) * float(sc)); r.ml = half(float(dd.y) * float(m));
    device const uint * q = (device const uint *) (b.qs + 32 * (sub / 2) + k0);
    r.a = q[0]; r.b = q[4];
    r.sh = 4 * (sub % 2);
}
static inline void cv_ks(thread const raw_ks_q4_K & r, thread half4 & lo, thread half4 & hi) {
    const uint a = r.a >> r.sh, b = r.b >> r.sh;
    lo = half4(uint4(a, a >> 8, a >> 16, a >> 24) & 0xF) * r.dl - r.ml;
    hi = half4(uint4(b, b >> 8, b >> 16, b >> 24) & 0xF) * r.dl - r.ml;
}
// Q5_K: Q4_K's scales and nibbles plus a fifth bit per weight in qh (bit 2 g + hi of byte l for weight 64 g + 32 hi + l)
struct block_q5_K { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; };
struct raw_ks_q5_K { half dl, ml; uint a, b, ha, hb; ushort sh, hs; };
template <> struct raw_ks_of<block_q5_K> { typedef raw_ks_q5_K type; };
static inline void ld_ks(device const block_q5_K * row, int kk, int k0, thread raw_ks_q5_K & r) {
    device const block_q5_K & b = row[kk / 256];
    const int sub = (kk % 256) / 32;
    const uint4 h = *((device const uint4 *) &b);
    uint sc, m;
    if (sub < 4) {
        sc = (h.y >> (8 * sub)) & 63; m = (h.z >> (8 * sub)) & 63;
    } else {
        const int i = 8 * (sub - 4);
        sc = ((h.w >> i) & 0xF) | (((h.y >> (i + 6)) & 3) << 4);
        m = ((h.w >> (i + 4)) & 0xF) | (((h.z >> (i + 6)) & 3) << 4);
    }
    const half2 dd = as_type<half2>(h.x);
    r.dl = half(float(dd.x) * float(sc)); r.ml = half(float(dd.y) * float(m));
    device const uint * q = (device const uint *) (b.qs + 32 * (sub / 2) + k0);
    device const uint * qh = (device const uint *) (b.qh + k0);
    r.a = q[0]; r.b = q[4]; r.ha = qh[0]; r.hb = qh[4];
    r.sh = 4 * (sub % 2); r.hs = sub;
}
static inline void cv_ks(thread const raw_ks_q5_K & r, thread half4 & lo, thread half4 & hi) {
    const uint a = r.a >> r.sh, b = r.b >> r.sh, ha = r.ha >> r.hs, hb = r.hb >> r.hs;
    lo = half4((uint4(a, a >> 8, a >> 16, a >> 24) & 0xF) | ((uint4(ha, ha >> 8, ha >> 16, ha >> 24) & 1) << 4)) * r.dl - r.ml;
    hi = half4((uint4(b, b >> 8, b >> 16, b >> 24) & 0xF) | ((uint4(hb, hb >> 8, hb >> 16, hb >> 24) & 1) << 4)) * r.dl - r.ml;
}
// Q6_K: 4-byte loads at 2-byte alignment (packed_ushort2: a 210-byte block every other one sits 2 bytes off), the two
// scales of the piece one 2-byte load
struct raw_ks_q6_K { half d0, d1; uint l0, l1, h0, h1; ushort shl, shh; };
template <> struct raw_ks_of<block_q6_K> { typedef raw_ks_q6_K type; };
static inline uint ld_u32(device const uchar * p) { const packed_ushort2 v = *((device const packed_ushort2 *) p); return (uint) v[0] | ((uint) v[1] << 16); }
static inline void ld_ks(device const block_q6_K * row, int kk, int k0, thread raw_ks_q6_K & r) {
    device const block_q6_K & b = row[kk / 256];
    const int sub = (kk % 256) / 32, n = sub / 4, qu = sub % 4;
    device const uchar * ql = b.ql + n * 64 + (qu & 1) * 32 + k0, * qh = b.qh + n * 32 + k0;
    r.l0 = ld_u32(ql); r.l1 = ld_u32(ql + 16); r.h0 = ld_u32(qh); r.h1 = ld_u32(qh + 16);
    const ushort s2 = *((device const ushort *) (b.scales + n * 8 + 2 * qu));
    const float d = b.d;
    r.d0 = half(d * float((char) (s2 & 0xFF))); r.d1 = half(d * float((char) (s2 >> 8)));
    r.shl = qu < 2 ? 0 : 4; r.shh = 2 * qu;
}
static inline int4 q6_4(uint l, uint h, uint shl, uint shh) {
    l >>= shl; h >>= shh;
    return int4((uint4(l, l >> 8, l >> 16, l >> 24) & 0xF) | ((uint4(h, h >> 8, h >> 16, h >> 24) & 3) << 4)) - 32;
}
static inline void cv_ks(thread const raw_ks_q6_K & r, thread half4 & lo, thread half4 & hi) {
    lo = half4(q6_4(r.l0, r.h0, r.shl, r.shh)) * r.d0;
    hi = half4(q6_4(r.l1, r.h1, r.shl, r.shh)) * r.d1;
}

// RT x 32 weight rows x NT token rows per threadgroup; simdgroup sg = (row tile rt, K slice ks): the G simdgroups of a row
// tile take K slices (interleaved 32-chunks), each into its own partial, summed in a fixed order at the end. With RT = 2
// (reads) the row tiles share each activation chunk through L1; each simdgroup's arithmetic is the RT = 1 one.
template <typename BQ, int G, int NT, int RT, typename TX = float>
kernel void qmm_ks(constant qmm_args & a [[buffer(0)]], device const char * W0 [[buffer(1)]], device const TX * X [[buffer(2)]],
                   device float * D [[buffer(3)]], device const int4 * tiles [[buffer(4)]], device half * DH [[buffer(5)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int SN = 32, NTH = 32 * G * RT;
    const int rt = (int) sg / G, ks = (int) sg % G;
    int n0 = tg.x * NT, nrem = clamp(a.N - n0, 0, NT);
    device const char * W = W0;
    if (a.flags & 16) {
        const int4 t = tiles[tg.x];
        if (t.z <= 0) return;
        W = W0 + (size_t) t.x * a.estride; n0 = t.y; nrem = t.z;
    }
    const int r0 = (tg.y * RT + rt) * SN;
    const bool il = (a.flags & 1) != 0;
    const int nk = a.K / 32;
    const int kb = il ? 32 * ks : 32 * ((ks * nk) / G);
    const int ke = il ? a.K : 32 * (((ks + 1) * nk) / G);
    const int kst = il ? 32 * G : 32;
    constexpr auto desc = matmul2d_descriptor(SN, NT, 32, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> mm;
    device const TX * xb = X + (uint64_t) n0 * a.ldx;
    auto tX0 = tensor((device TX *) xb, dextents<int32_t, 2>(32, NT), array<int, 2>({1, a.ldx}));
    auto wL = mm.template get_left_input_cooperative_tensor<half, TX, float>();
    auto cT = mm.template get_destination_cooperative_tensor<decltype(wL), decltype(tX0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }
    // left-input layout probed on the M5: element 16a + 8b + 4c + e = (k0 + 16b + e, row r0 + 8c + 16a)
    const auto ix0 = wL.get_multidimensional_index((uint16_t) 0);
    const int k0l = ix0[0], rl = ix0[1];
    typedef typename raw_ks_of<BQ>::type RAW;
    RAW raw[4];
    device const BQ * rps[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int gr = min(r0 + rl + 8 * (q % 2) + 16 * (q / 2), a.M - 1);
        rps[q] = (device const BQ *) (W + (uint64_t) gr * a.nb01);
    }
    if (kb < ke) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) ld_ks(rps[q], kb, k0l, raw[q]);
    }
    for (int k0 = kb; k0 < ke; k0 += kst) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            half4 lo, hi;
            cv_ks(raw[q], lo, hi);
            const int a_ = q / 2, c_ = q % 2;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                wL[16 * a_ + 0 + 4 * c_ + e] = lo[e];
                wL[16 * a_ + 8 + 4 * c_ + e] = hi[e];
            }
        }
        if (k0 + kst < ke) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) ld_ks(rps[q], k0 + kst, k0l, raw[q]);
        }
        if (nrem > 0) {
            auto tX = tensor((device TX *) xb + k0, dextents<int32_t, 2>(32, nrem), array<int, 2>({1, a.ldx}));
            mm.run(wL, tX, cT);
        }
    }
    threadgroup float red[4 * 1024];
    constexpr int NH = NT > 32 ? NT / 32 : 1;  // token halves of 32 through the 32 x 32 partial buffers
    for (int t = 0; t < RT * NH; ++t) {  // the row tiles' partials in turn, each in token halves
        const int tt = t / NH, h0 = (t % NH) * 32;
        if (t > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (G == 8) {
            if (rt == tt && ks >= 4) {
                threadgroup float * p = red + (ks - 4) * 1024;
                #pragma unroll
                for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                    if (cT.is_valid_element(i)) {
                        const auto ix = cT.get_multidimensional_index(i);
                        if (ix[0] >= h0 && ix[0] < h0 + 32) p[(ix[0] - h0) * 32 + ix[1]] = cT[i];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (rt == tt && ks < 4) {
                threadgroup const float * p = red + ks * 1024;
                #pragma unroll
                for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                    if (cT.is_valid_element(i)) {
                        const auto ix = cT.get_multidimensional_index(i);
                        if (ix[0] >= h0 && ix[0] < h0 + 32) cT[i] += p[(ix[0] - h0) * 32 + ix[1]];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (rt == tt && ks < 4) {
            threadgroup float * p = red + ks * 1024;
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) {
                    const auto ix = cT.get_multidimensional_index(i);
                    if (ix[0] >= h0 && ix[0] < h0 + 32) p[(ix[0] - h0) * 32 + ix[1]] = cT[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int rb = (tg.y * RT + tt) * SN;
        for (int e = tid; e < 1024; e += NTH) {
            const int n = h0 + e / 32, m = e % 32;
            const float s = ((red[e] + red[1024 + e]) + red[2048 + e]) + red[3072 + e];
            if (n < nrem && rb + m < a.M) {
                device float * o = D + (uint64_t) (n0 + n) * a.ldd + rb + m;
                const float v = (a.flags & 14) ? qmm_epi(a.flags, *o, s) : s;
                if (a.flags & 32) DH[(uint64_t) (n0 + n) * a.ldh + rb + m] = half(clamp(v, -65504.0f, 65504.0f));
                else *o = v;
            }
        }
    }
}

// ---- qmm_ks with each unpacked weight tile shared by TS token tiles ---------------------------------------------------------
// The simdgroups of qmm_ks (RT = 1): G K slices, now for TS token tiles of 32 (G x TS simdgroups). For each of its chunks a
// slice's 32 x 32 weights are unpacked once into threadgroup memory (by its TS simdgroups together; qmm_ks unpacks them
// in each token tile's threadgroup) and each simdgroup takes them into its left input, so its matmul2d, its slice's sum
// and the slices' order are qmm_ks's: the same bits for any TS (and any rows). Two tile buffers, one barrier a chunk.
template <typename BQ, int G, int TS, typename TX>
kernel void qmms(constant qmm_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const TX * X [[buffer(2)]],
                 device float * D [[buffer(3)]], device const int4 * tiles [[buffer(4)]], device half * DH [[buffer(5)]],
                 uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort ln [[thread_index_in_simdgroup]]) {
    constexpr int SN = 32, NT = 32, NTH = 32 * G * TS, NPL = 4 / TS;  // a lane's 8-weight pieces of its slice's tile
    threadgroup float shm[4096];  // the tile buffers [2][G][32 x 32] halves, then the slices' partials
    threadgroup half * wt = (threadgroup half *) shm;
    const int ts = (int) sg / G, ks = (int) sg % G;
    const int n0 = (tg.x * TS + ts) * NT, nrem = clamp(a.N - n0, 0, NT);
    const int r0 = tg.y * SN;
    const bool il = (a.flags & 1) != 0;
    const int nk = a.K / 32;
    const int kb = il ? 32 * ks : 32 * ((ks * nk) / G);
    const int ke = il ? a.K : 32 * (((ks + 1) * nk) / G);
    const int kst = il ? 32 * G : 32;
    const int nit = il ? (nk + G - 1) / G : (nk + G - 1) / G + 1;  // the most chunks a slice has (every simdgroup meets every barrier)
    constexpr auto desc = matmul2d_descriptor(SN, NT, 32, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> mm;
    device const TX * xb = X + (uint64_t) n0 * a.ldx;
    auto tX0 = tensor((device TX *) xb, dextents<int32_t, 2>(32, NT), array<int, 2>({1, a.ldx}));
    auto wL = mm.template get_left_input_cooperative_tensor<half, TX, float>();
    auto cT = mm.template get_destination_cooperative_tensor<decltype(wL), decltype(tX0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }
    const auto ix0 = wL.get_multidimensional_index((uint16_t) 0);
    const int k0l = ix0[0], rl = ix0[1];
    typedef typename raw_ks_of<BQ>::type RAW;
    RAW raw[NPL];
    device const BQ * rp[NPL];
    int so[NPL], kq[NPL];
    for (int j = 0; j < NPL; ++j) {
        const int pc = ts * 32 + ln + TS * 32 * j, r = pc / 4;  // piece pc of the tile: row r, 4 weights at k0 and k0 + 16
        kq[j] = 4 * (pc % 4); so[j] = r * 32 + kq[j];
        rp[j] = (device const BQ *) (W + (uint64_t) min(r0 + r, a.M - 1) * a.nb01);
        if (kb < ke) ld_ks(rp[j], kb, kq[j], raw[j]);
    }
    for (int it = 0, k0 = kb; it < nit; ++it, k0 += kst) {
        threadgroup half * t = wt + ((it & 1) * G + ks) * 1024;
        const bool live = k0 < ke;
        if (live) {
            for (int j = 0; j < NPL; ++j) {
                half4 lo, hi;
                cv_ks(raw[j], lo, hi);
                *((threadgroup half4 *) (t + so[j])) = lo;
                *((threadgroup half4 *) (t + so[j] + 16)) = hi;
            }
            if (k0 + kst < ke) {
                for (int j = 0; j < NPL; ++j) ld_ks(rp[j], k0 + kst, kq[j], raw[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) {  // qmm_ks's left input: element 16a + 8b + 4c + e = (k0l + 16b + e, row rl + 8c + 16a)
                const int a_ = q / 2, c_ = q % 2;
                threadgroup const half * tr = t + (rl + 8 * c_ + 16 * a_) * 32 + k0l;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    wL[16 * a_ + 0 + 4 * c_ + e] = tr[e];
                    wL[16 * a_ + 8 + 4 * c_ + e] = tr[16 + e];
                }
            }
            if (nrem > 0) {
                auto tX = tensor((device TX *) xb + k0, dextents<int32_t, 2>(32, nrem), array<int, 2>({1, a.ldx}));
                mm.run(wL, tX, cT);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * red = shm;
    for (int tt = 0; tt < TS; ++tt) {  // the token tiles' partials in turn, as qmm_ks's row tiles
        if (tt > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (G == 8) {
            if (ts == tt && ks >= 4) {
                threadgroup float * p = red + (ks - 4) * 1024;
                #pragma unroll
                for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                    if (cT.is_valid_element(i)) {
                        const auto ix = cT.get_multidimensional_index(i);
                        p[ix[0] * 32 + ix[1]] = cT[i];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ts == tt && ks < 4) {
                threadgroup const float * p = red + ks * 1024;
                #pragma unroll
                for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                    if (cT.is_valid_element(i)) {
                        const auto ix = cT.get_multidimensional_index(i);
                        cT[i] += p[ix[0] * 32 + ix[1]];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (ts == tt && ks < 4) {
            threadgroup float * p = red + ks * 1024;
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) {
                    const auto ix = cT.get_multidimensional_index(i);
                    p[ix[0] * 32 + ix[1]] = cT[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int nt0 = (tg.x * TS + tt) * NT, nr = clamp(a.N - nt0, 0, NT);
        for (int e = tid; e < 1024; e += NTH) {
            const int n = e / 32, m = e % 32;
            const float s = ((red[e] + red[1024 + e]) + red[2048 + e]) + red[3072 + e];
            if (n < nr && r0 + m < a.M) {
                device float * o = D + (uint64_t) (nt0 + n) * a.ldd + r0 + m;
                const float v = (a.flags & 14) ? qmm_epi(a.flags, *o, s) : s;
                if (a.flags & 32) DH[(uint64_t) (nt0 + n) * a.ldh + r0 + m] = half(clamp(v, -65504.0f, 65504.0f));
                else *o = v;
            }
        }
    }
}
#define QMMS(BQ, G, TS) template [[host_name("qmms_" #BQ "_g" #G "_t" #TS)]] kernel void qmms<block_##BQ, G, TS, half>(constant qmm_args &, \
    device const char *, device const half *, device float *, device const int4 *, device half *, uint3, ushort, ushort, ushort);
#define QMMS_ALL(BQ) QMMS(BQ, 4, 2) QMMS(BQ, 4, 4) QMMS(BQ, 8, 2) QMMS(BQ, 8, 4)
QMMS_ALL(q4_0) QMMS_ALL(q4_1) QMMS_ALL(q8_0) QMMS_ALL(q4_K) QMMS_ALL(q5_K) QMMS_ALL(q6_K)  // (kq_bench QS=2: q8_0 0.72x, q6_K 0.78x, q5_K 0.86-0.93x qmmh_ks at 512 rows; q4_0 / q4_K no gain)

#define QMM_KS(BQ, G, NT, RT) template [[host_name("qmm_ks_" #BQ "_g" #G "_nt" #NT "_r" #RT)]] kernel void qmm_ks<block_##BQ, G, NT, RT>( \
    constant qmm_args &, device const char *, device const float *, device float *, device const int4 *, device half *, uint3, ushort, ushort); \
    template [[host_name("qmmh_ks_" #BQ "_g" #G "_nt" #NT "_r" #RT)]] kernel void qmm_ks<block_##BQ, G, NT, RT, half>( \
    constant qmm_args &, device const char *, device const half *, device float *, device const int4 *, device half *, uint3, ushort, ushort);
#define QMM_KS_ALL(BQ, G) QMM_KS(BQ, G, 8, 1) QMM_KS(BQ, G, 16, 1) QMM_KS(BQ, G, 32, 1) QMM_KS(BQ, G, 32, 2) QMM_KS(BQ, G, 64, 2)
QMM_KS_ALL(q4_0, 4) QMM_KS_ALL(q4_0, 8) QMM_KS_ALL(q4_1, 4) QMM_KS_ALL(q4_1, 8) QMM_KS_ALL(q8_0, 4) QMM_KS_ALL(q8_0, 8)
// the K-quants: half inputs only (sn.m's default; SN_XF32 takes the q4 / q8 formats' float inputs)
#define QMMH_KS(BQ, G, NT, RT) template [[host_name("qmmh_ks_" #BQ "_g" #G "_nt" #NT "_r" #RT)]] kernel void qmm_ks<block_##BQ, G, NT, RT, half>( \
    constant qmm_args &, device const char *, device const half *, device float *, device const int4 *, device half *, uint3, ushort, ushort);
#define QMMH_KS_ALL(BQ, G) QMMH_KS(BQ, G, 8, 1) QMMH_KS(BQ, G, 16, 1) QMMH_KS(BQ, G, 32, 1) QMMH_KS(BQ, G, 32, 2) QMMH_KS(BQ, G, 64, 2)
QMMH_KS_ALL(q4_K, 4) QMMH_KS_ALL(q4_K, 8) QMMH_KS_ALL(q6_K, 4) QMMH_KS_ALL(q6_K, 8) QMMH_KS_ALL(q5_K, 4) QMMH_KS_ALL(q5_K, 8)
QMMH_KS_ALL(mxfp4, 4) QMMH_KS_ALL(mxfp4, 8)
// the grouped (MoE) matmuls' tiles of 16 pairs with 2 row tiles
QMMH_KS(q4_K, 4, 16, 2) QMMH_KS(q4_K, 8, 16, 2) QMMH_KS(q5_K, 4, 16, 2) QMMH_KS(q5_K, 8, 16, 2) QMMH_KS(q6_K, 4, 16, 2) QMMH_KS(q6_K, 8, 16, 2)
QMMH_KS(mxfp4, 4, 16, 2) QMMH_KS(mxfp4, 8, 16, 2)
QMMH_KS(q4_K, 4, 64, 1) QMMH_KS(q5_K, 4, 64, 1) QMMH_KS(q6_K, 4, 64, 1) QMMH_KS(mxfp4, 4, 64, 1)

// Q6_K (the output layer): inside every 32-K chunk the K order is permuted so a lane loads whole words of ql/qh; the
// activations follow the permutation through threadgroup memory as half (double-buffered).
struct raw_q6p { uint l, h; ushort sc; half d; };
// par: the row's start in 2-byte units, mod 2 (a 210-byte block is 2 mod 4, so with an odd number of blocks a row every
// other row starts 2 bytes past a 4-byte boundary): block b starts 4-byte aligned when b + par is even
static inline void ld_q6p(device const block_q6_K * row, int par, int kk, int t, thread raw_q6p & r) {
    const int b = kk / 256, l0 = kk % 256, hh = l0 / 128, g = (l0 % 128) / 32;
    device const block_q6_K & bl = row[b];
    device const uchar * ql = bl.ql + 64 * hh + 32 * (g & 1) + 4 * t;
    device const uchar * qh = bl.qh + 32 * hh + 4 * t;
    if (((b + par) & 1) == 0) {
        r.l = *((device const uint *) ql);
        r.h = *((device const uint *) qh);
    } else {
        r.l = (uint) *((device const ushort *) ql) | ((uint) *((device const ushort *) (ql + 2)) << 16);
        r.h = (uint) *((device const ushort *) qh) | ((uint) *((device const ushort *) (qh + 2)) << 16);
    }
    r.sc = *((device const ushort *) (bl.scales + 8 * hh + 2 * g));
    r.d = bl.d;
}
static inline void cv_q6p(thread const raw_q6p & r, int kk, int t, thread half4 & w) {
    const int g = (kk % 128) / 32;
    const uint shl = (g >= 2) ? 4 : 0, shh = 2 * g;
    const uint4 lb = uint4(r.l, r.l >> 8, r.l >> 16, r.l >> 24) & 0xFF;
    const uint4 hb = uint4(r.h, r.h >> 8, r.h >> 16, r.h >> 24) & 0xFF;
    const uint4 q = ((lb >> shl) & 0xF) | (((hb >> shh) & 3) << 4);
    const float d = r.d;
    const half sc = (half) (d * (float) (char) ((t >= 4) ? (r.sc >> 8) : (r.sc & 0xFF)));
    w = half4(short4(q) - 32) * sc;
}

template <int NT>
kernel void qmm_q6p(constant qmm_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * X [[buffer(2)]],
                    device float * D [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int G = 4, SN = 32, NTH = 32 * G;
    constexpr int XPT = NT * 8 / NTH;
    const int n0 = tg.x * NT;
    const int r0 = tg.y * (SN * G) + sg * SN;
    const int nrem = clamp(a.N - n0, 0, NT);
    constexpr auto desc = matmul2d_descriptor(NT, SN, 32, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> mm;
    threadgroup half xs[2 * NT * 32];
    auto tX0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(xs, dextents<int32_t, 2>(32, NT));
    auto wR = mm.template get_right_input_cooperative_tensor<half, half, float>();
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX0), decltype(wR), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }
    const auto ix0 = wR.get_multidimensional_index((uint16_t) 0);
    const int nb = ix0[0], kbl = ix0[1];
    device const block_q6_K * rps[8];
    int rpar[8];
    #pragma unroll
    for (int nsub = 0; nsub < 2; ++nsub) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int gr = min(r0 + nb + 16 * nsub + j, a.M - 1);
            rps[4 * nsub + j] = (device const block_q6_K *) (W + (uint64_t) gr * a.nb01);
            rpar[4 * nsub + j] = (int) (((uint64_t) gr * a.nb01 / 2) & 1);
        }
    }
    float4 xr[XPT];
    #define Q6P_LOAD_X(kk) _Pragma("unroll") for (int i = 0; i < XPT; ++i) { const int f = tid + i * NTH; \
        const int n = min(n0 + f / 8, a.N - 1); xr[i] = *((device const float4 *) (X + (uint64_t) n * a.ldx + (kk)) + f % 8); }
    #define Q6P_STORE_X(buf) _Pragma("unroll") for (int i = 0; i < XPT; ++i) { const int f = tid + i * NTH; \
        threadgroup half * p = xs + (buf) * NT * 32 + (f / 8) * 32 + (f % 8); \
        const float4 xc = clamp(xr[i], -65504.0f, 65504.0f); \
        p[0] = (half) xc.x; p[8] = (half) xc.y; p[16] = (half) xc.z; p[24] = (half) xc.w; }
    Q6P_LOAD_X(0)
    Q6P_STORE_X(0)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int buf = 0;
    for (int k0 = 0; k0 < a.K; k0 += 32) {
        const bool more = k0 + 32 < a.K;
        if (more) { Q6P_LOAD_X(k0 + 32) }
        #pragma unroll
        for (int nsub = 0; nsub < 2; ++nsub) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                raw_q6p r;
                ld_q6p(rps[4 * nsub + j], rpar[4 * nsub + j], k0, kbl, r);
                half4 w4;
                cv_q6p(r, k0, kbl, w4);
                wR[8 * (nsub + 0) + 0 + j] = w4[0];
                wR[8 * (nsub + 0) + 4 + j] = w4[1];
                wR[8 * (nsub + 2) + 0 + j] = w4[2];
                wR[8 * (nsub + 2) + 4 + j] = w4[3];
            }
        }
        auto tX = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(xs + buf * NT * 32, dextents<int32_t, 2>(32, NT));
        mm.run(tX, wR, cT);
        if (more) { Q6P_STORE_X(buf ^ 1) }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    #undef Q6P_LOAD_X
    #undef Q6P_STORE_X
    if (r0 >= a.M || nrem <= 0) return;
    const int mrem = min(SN, a.M - r0);
    auto tD = tensor(D + (uint64_t) n0 * a.ldd + r0, dextents<int32_t, 2>(mrem, nrem), array<int, 2>({1, a.ldd}));
    cT.store(tD);
}
template [[host_name("qmm_q6p_nt16")]] kernel void qmm_q6p<16>(constant qmm_args &, device const char *, device const float *, device float *, uint3, ushort, ushort);
template [[host_name("qmm_q6p_nt32")]] kernel void qmm_q6p<32>(constant qmm_args &, device const char *, device const float *, device float *, uint3, ushort, ushort);

// ---- attention in fixed chunks of a sequence's cells, merged in chunk order ------------------------------------------
// attn_part: one threadgroup per (row, kv head, chunk of AC cells), one query head of the GQA group per simdgroup; 8 cells
// at a time (their dot products and sums are independent, so they overlap), one softmax update per 8. A row's chunks
// depend only on its own position, never on the other rows, so its result is the same in any step.
// attn_merge: per (row, head), the chunks in order: (m, l, acc) combined, then acc / l.
constant constexpr int AC = 64;
// fa2's chunks a partial for a row of nc chunks: 1 up to msp chunks, then as many as keep its partials at msp or fewer
// (0: always 1; windowed rows always 1). Its partials are merged here in order, so a long row writes and merges far fewer.
static inline int fa_span(int nc, int msp, int window) { return msp > 0 && window == 0 && nc > msp ? (nc + msp - 1) / msp : 1; }
// n_fast, maxch: rows [0, n_fast) whose cells span at most maxch chunks are fa1's (0: none)
struct attn2_args { int n_head; int n_kv; int ctx; float scale; int ldq; int ldo; int maxc; int n_fast; int maxch; int msp; };  // msp: fa2's (fa_span)
// Each lane scores one cell (its own 128-wide dot product, q from threadgroup memory), then the 32 scores take one max
// and one sum; for the values each lane owns 4 dims and walks the 32 cells in order.
kernel void attn_part(constant attn2_args & a [[buffer(0)]], device const float * q [[buffer(1)]], device const half * kc [[buffer(2)]],
                      device const half * vc [[buffer(3)]], device const int * seq [[buffer(4)]], device const int * pos [[buffer(5)]],
                      device float * part [[buffer(6)]], device const int * pt [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs[8][32];
    const int r = tg.x, kvh = tg.y, c = tg.z, g = a.n_head / a.n_kv;
    const int h = kvh * g + sg, p = pos[r];
    const int c0 = c * AC, c1 = min(c0 + AC, p + 1);
    if (c0 > p || (r < a.n_fast && p / AC + 1 <= a.maxch)) return;
    qs[sg][ln] = *((device const float4 *) (q + (size_t) r * a.ldq + h * 128) + ln) * a.scale;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const int stride = a.n_kv * 128;
    const size_t pg = (size_t) pt[(size_t) seq[r] * (a.ctx / AC) + c] * AC * stride;  // chunk c is one page
    device const half * kb = kc + pg + kvh * 128;
    device const half * vb = vc + pg + kvh * 128;
    float m = -INFINITY, l = 0.f;
    float4 acc = 0.f;
    for (int cb = c0; cb < c1; cb += 32) {
        const int cell = cb + ln;
        float sc = -INFINITY;
        if (cell < c1) {
            device const half4 * kr = (device const half4 *) (kb + (size_t) (cell - c0) * stride);
            float4 t = 0.f;
            for (int i = 0; i < 32; ++i) t = fma(float4(kr[i]), qs[sg][i], t);
            sc = (t.x + t.y) + (t.z + t.w);
        }
        const float mn = max(m, simd_max(sc));
        const float e0 = exp(m - mn);
        const float pc = cell < c1 ? exp(sc - mn) : 0.f;
        acc *= e0;
        l = l * e0 + simd_sum(pc);
        const int n = min(32, c1 - cb);
        for (int j = 0; j < n; ++j) {
            acc += float4(*((device const half4 *) (vb + (size_t) (cb + j - c0) * stride) + ln)) * simd_shuffle(pc, j);
        }
        m = mn;
    }
    device float * o = part + (((size_t) r * a.n_head + h) * a.maxc + c) * 130;
    *((device float4 *) o + ln) = acc;
    if (ln == 0) { o[128] = m; o[129] = l; }
}

kernel void attn_merge(constant attn2_args & a [[buffer(0)]], device const float * part [[buffer(1)]], device const int * pos [[buffer(2)]],
                       device float * out [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], ushort ln [[thread_index_in_simdgroup]]) {
    const int r = tg.x, h = tg.y, nc = pos[r] / AC + 1, sp = r < a.n_fast ? fa_span(nc, a.msp, 0) : 1;
    device const float * p0 = part + ((size_t) r * a.n_head + h) * a.maxc * 130;
    float m = -INFINITY, l = 0.f;
    float4 acc = 0.f;
    for (int c = 0; c < (nc + sp - 1) / sp; ++c) {
        device const float * pc = p0 + c * 130;
        const float mc = pc[128], lc = pc[129], mn = max(m, mc);
        const float e0 = exp(m - mn), e1 = exp(mc - mn);
        acc = acc * e0 + *((device const float4 *) pc + ln) * e1;
        l = l * e0 + lc * e1;
        m = mn;
    }
    *((device float4 *) (out + (size_t) r * a.ldo + h * 128) + ln) = acc / l;
}

// ---- prompt attention on simdgroup matrices --------------------------------------------------------------------------
// A query block = up to 16 consecutive prompt rows of one sequence; one threadgroup per (block, kv head) with 2 x G
// simdgroups: simdgroup sg = (rows 8 (sg / G) .. + 7, query head sg % G of the GQA group), so a K/V block staged in
// threadgroup memory serves all of them. Cells go in blocks of 32: S = Q K^T (half inputs, float sums over d in order), a
// per-row softmax update, O = diag(alpha) O + P V (the scaling skipped when every alpha is 1: x * 1 is x). Every row's
// arithmetic depends only on its own position (masked cells add exactly nothing), not on the block it sits in.
struct attnp_args { int n_head; int n_kv; int ctx; float scale; int ldq; int ldo; int swap; };
kernel void attn_pre(constant attnp_args & a [[buffer(0)]], device const float * q [[buffer(1)]], device const half * kc [[buffer(2)]],
                     device const half * vc [[buffer(3)]], device const int4 * blk [[buffer(4)]], device const int * pos [[buffer(5)]],
                     device float * o [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup half kvs[2 * 32 * 128];  // the Q tiles first, then K and V of a cell block, at the end the O tiles as float
    threadgroup half * ks = kvs, * vs = kvs + 32 * 128;
    threadgroup float sc[8][8 * 32];
    threadgroup half ps[8][8 * 32];
    threadgroup float dg[8][64];
    const int4 b = blk[tg.x];
    const int row0 = b.x, nr = b.y, s = b.z, kvh = tg.y, g = a.n_head / a.n_kv;
    const int hs = sg % g, rh = sg / g, h = kvh * g + hs;
    const int nth = 64 * g, r0 = row0 + 8 * rh, nr8 = clamp(nr - 8 * rh, 0, 8);
    const int pmax = pos[row0 + nr - 1];
    // Q: each simdgroup stages its 8 x 128 tile (rows beyond the block are zero) in kvs, then keeps it in registers
    threadgroup half * qs = kvs + sg * 8 * 128;
    for (int i = ln; i < 8 * 128; i += 32) {
        const int r = i / 128, d = i % 128;
        qs[i] = r < nr8 ? (half) q[(size_t) (r0 + r) * a.ldq + h * 128 + d] : 0.h;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qm[16];
    for (int t = 0; t < 16; ++t) simdgroup_load(qm[t], qs + t * 8, 128);
    // the lane's row for the softmax: r = ln / 4, cells 8 (ln % 4) .. + 7
    const int sr = ln / 4, sp = ln % 4;
    const int my_pos = sr < nr8 ? pos[r0 + sr] : -1;
    float m = -INFINITY, l = 0.f;
    simdgroup_float8x8 acc[16];
    for (int t = 0; t < 16; ++t) acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (int i = ln; i < 64; i += 32) dg[sg][i] = 0.f;
    device const half * kbase = kc + (size_t) s * a.ctx * a.n_kv * 128 + kvh * 128;
    device const half * vbase = vc + (size_t) s * a.ctx * a.n_kv * 128 + kvh * 128;
    for (int c0 = 0; c0 <= pmax; c0 += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < 32 * 128 / 4; i += nth) {
            const int c = i / 32, d4 = i % 32;
            const bool ok = c0 + c <= pmax;
            *((threadgroup half4 *) ks + i) = ok ? *((device const half4 *) (kbase + (size_t) (c0 + c) * a.n_kv * 128) + d4) : half4(0.h);
            *((threadgroup half4 *) vs + i) = ok ? *((device const half4 *) (vbase + (size_t) (c0 + c) * a.n_kv * 128) + d4) : half4(0.h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (nr8 == 0) continue;
        // S (8 rows x 32 cells) = Q K^T
        for (int j = 0; j < 4; ++j) {
            simdgroup_float8x8 st = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
            for (int t = 0; t < 16; ++t) {
                simdgroup_half8x8 kt;
                simdgroup_load(kt, ks + j * 8 * 128 + t * 8, 128, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(st, qm[t], kt, st);
            }
            simdgroup_store(st, sc[sg] + j * 8, 32);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // softmax update of the lane's row over its 8 cells; the row's 4 lanes agree through shuffles
        float v[8], mb = -INFINITY;
        for (int i = 0; i < 8; ++i) {
            const int c = c0 + sp * 8 + i;
            v[i] = c <= my_pos ? sc[sg][sr * 32 + sp * 8 + i] * a.scale : -INFINITY;
            mb = max(mb, v[i]);
        }
        mb = max(mb, simd_shuffle_xor(mb, 1));
        mb = max(mb, simd_shuffle_xor(mb, 2));
        const float mn = max(m, mb);
        const float al = (mn == -INFINITY) ? 1.f : exp(m - mn);
        float ls = 0.f;
        for (int i = 0; i < 8; ++i) {
            const float p = (v[i] == -INFINITY) ? 0.f : exp(v[i] - mn);
            ps[sg][sr * 32 + sp * 8 + i] = (half) p;
            ls += p;
        }
        ls += simd_shuffle_xor(ls, 1);
        ls += simd_shuffle_xor(ls, 2);
        l = l * al + ls;
        m = mn;
        const bool rescale = !simd_all(al == 1.f);
        if (rescale && sp == 0) dg[sg][sr * 9] = al;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dm;
        if (rescale) simdgroup_load(dm, dg[sg], 8);
        simdgroup_half8x8 pm[4];
        for (int j = 0; j < 4; ++j) simdgroup_load(pm[j], ps[sg] + j * 8, 32);
        // O = diag(alpha) O + P V
        for (int t = 0; t < 16; ++t) {
            if (rescale) simdgroup_multiply(acc[t], dm, acc[t]);
            for (int j = 0; j < 4; ++j) {
                simdgroup_half8x8 vt;
                simdgroup_load(vt, vs + j * 8 * 128 + t * 8, 128);
                simdgroup_multiply_accumulate(acc[t], pm[j], vt, acc[t]);
            }
        }
    }
    // O / l, the rows that exist: two passes of four simdgroups through kvs (16 KB = 4 tiles of 8 x 128 floats)
    for (int pass = 0; pass < 2; ++pass) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rh == pass && nr8 > 0) {
            threadgroup float * mine = (threadgroup float *) kvs + hs * 8 * 128;
            for (int t = 0; t < 16; ++t) simdgroup_store(acc[t], mine + t * 8, 128);
            if (sp == 0) sc[sg][sr] = l;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = ln; i < 8 * 128; i += 32) {
                const int r = i / 128, d = i % 128;
                if (r < nr8) o[(size_t) (r0 + r) * a.ldo + h * 128 + d] = mine[i] / sc[sg][r];
            }
        }
    }
}

// ---- one-row path (interactive agents): mat-vec kernels for up to 4 rows ----------------------------------------------
// Which path an agent takes is fixed by the agent (its lane), never by the step, so either way its rows are the same at
// any agent count. Each simdgroup takes 4 weight rows; a lane takes half a block (8 bytes = 16 weights) at a time, the
// half-blocks lane / 2 + 16 i, the sum of a weight row is per lane in block order, then simd_sum. R rows share each
// weight load; a row's arithmetic does not depend on R.
struct mv_args { int K; int M; int R; int ldx; int ldy; int acc; int nb01; float eps; };
// MODE 0: x as is; 1: x is the residual, RMS-normed with the weight in aux on the way in (the simdgroup takes the row's
// scale itself, lanes then simd_sum, the same for every simdgroup); 2: x is the gate, aux the up projection: silu(x) * up.
// So an interactive agent's step skips the norm and swiglu kernels (two thirds of its dispatches).
template <int MODE>
inline float4 mv_x(device const float * x, device const float * aux, int k, float sc) {
    const float4 v = *((device const float4 *) (x + k));
    if (MODE == 1) return (v * sc) * *((device const float4 *) (aux + k));
    if (MODE == 2) { const float4 u = *((device const float4 *) (aux + k)); return (v / (1.0f + exp(-v))) * u; }
    return v;
}
template <typename BQ, int R, int MODE>
kernel void mv_q4(constant mv_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * x [[buffer(2)]],
                  device float * y [[buffer(3)]], device const float * aux [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    constexpr int J = R >= 3 ? 2 : 4;  // outputs a simdgroup (fewer live sums with more token rows)
    const int r0 = (tg * 2 + sg) * J;
    if (r0 >= a.M) return;
    const int nb = a.K / 32, il = ln % 2;
    float sc[R];
    for (int r = 0; r < R; ++r) {
        sc[r] = 1.f;
        if (MODE == 1) {
            float ss = 0.f;
            for (int k = ln * 4; k < a.K; k += 128) { const float4 v = *((device const float4 *) (x + (size_t) r * a.ldx + k)); ss += dot(v, v); }
            sc[r] = 1.0f / sqrt(simd_sum(ss) / a.K + a.eps);
        }
    }
    float sum[J][R];
    for (int j = 0; j < J; ++j) for (int r = 0; r < R; ++r) sum[j][r] = 0.f;
    for (int ib = ln / 2; ib < nb; ib += 16) {
        if (R == 4) for (int j = 0; j < J; ++j) {  // 4 token rows: each row's inputs loaded where used, not held for all rows (the same arithmetic)
            device const BQ & b = *((device const BQ *) (W + (size_t) (r0 + j) * a.nb01) + ib);
            device const ushort * qs = (device const ushort *) (b.qs + il * 8);
            const ushort4 q = ushort4(qs[0], qs[1], qs[2], qs[3]);
            const float4 l0 = float4(q.x & 15, (q.x >> 8) & 15, q.y & 15, (q.y >> 8) & 15);
            const float4 l1 = float4(q.z & 15, (q.z >> 8) & 15, q.w & 15, (q.w >> 8) & 15);
            const float4 h0 = float4((q.x >> 4) & 15, q.x >> 12, (q.y >> 4) & 15, q.y >> 12);
            const float4 h1 = float4((q.z >> 4) & 15, q.z >> 12, (q.w >> 4) & 15, q.w >> 12);
            const float d = b.d;
            for (int r = 0; r < R; ++r) {
                device const float * xr = x + (size_t) r * a.ldx;
                device const float * ar = MODE == 2 ? aux + (size_t) r * a.ldx : aux;
                const int k = ib * 32 + il * 8;
                const float4 x0 = mv_x<MODE>(xr, ar, k, sc[r]), x1 = mv_x<MODE>(xr, ar, k + 4, sc[r]);
                const float4 x2 = mv_x<MODE>(xr, ar, k + 16, sc[r]), x3 = mv_x<MODE>(xr, ar, k + 20, sc[r]);
                if (is_same<BQ, block_q4_1>::value) {
                    const float mm_ = ((device const block_q4_1 &) b).m;
                    sum[j][r] += d * (dot(l0, x0) + dot(l1, x1) + dot(h0, x2) + dot(h1, x3)) + mm_ * dot(x0 + x1 + x2 + x3, float4(1.f));
                } else {
                    sum[j][r] += d * (dot(l0 - 8.f, x0) + dot(l1 - 8.f, x1) + dot(h0 - 8.f, x2) + dot(h1 - 8.f, x3));
                }
            }
        }
        if (R == 4) continue;
        float4 xl[R][2], xh[R][2];
        for (int r = 0; r < R; ++r) {
            device const float * xr = x + (size_t) r * (MODE == 2 ? a.ldx : a.ldx);
            device const float * ar = MODE == 2 ? aux + (size_t) r * a.ldx : aux;
            const int k = ib * 32 + il * 8;
            xl[r][0] = mv_x<MODE>(xr, ar, k, sc[r]); xl[r][1] = mv_x<MODE>(xr, ar, k + 4, sc[r]);
            xh[r][0] = mv_x<MODE>(xr, ar, k + 16, sc[r]); xh[r][1] = mv_x<MODE>(xr, ar, k + 20, sc[r]);
        }
        for (int j = 0; j < J; ++j) {
            device const BQ & b = *((device const BQ *) (W + (size_t) (r0 + j) * a.nb01) + ib);
            device const ushort * qs = (device const ushort *) (b.qs + il * 8);
            const ushort4 q = ushort4(qs[0], qs[1], qs[2], qs[3]);
            const float4 l0 = float4(q.x & 15, (q.x >> 8) & 15, q.y & 15, (q.y >> 8) & 15);
            const float4 l1 = float4(q.z & 15, (q.z >> 8) & 15, q.w & 15, (q.w >> 8) & 15);
            const float4 h0 = float4((q.x >> 4) & 15, q.x >> 12, (q.y >> 4) & 15, q.y >> 12);
            const float4 h1 = float4((q.z >> 4) & 15, q.z >> 12, (q.w >> 4) & 15, q.w >> 12);
            const float d = b.d;
            for (int r = 0; r < R; ++r) {
                if (is_same<BQ, block_q4_1>::value) {
                    const float mm_ = ((device const block_q4_1 &) b).m;
                    const float sx = dot(xl[r][0] + xl[r][1] + xh[r][0] + xh[r][1], float4(1.f));
                    const float s = dot(l0, xl[r][0]) + dot(l1, xl[r][1]) + dot(h0, xh[r][0]) + dot(h1, xh[r][1]);
                    sum[j][r] += d * s + mm_ * sx;
                } else {
                    const float s = dot(l0 - 8.f, xl[r][0]) + dot(l1 - 8.f, xl[r][1]) + dot(h0 - 8.f, xh[r][0]) + dot(h1 - 8.f, xh[r][1]);
                    sum[j][r] += d * s;
                }
            }
        }
    }
    for (int j = 0; j < J; ++j) {
        for (int r = 0; r < R; ++r) {
            const float v = simd_sum(sum[j][r]);
            if (ln == 0 && r0 + j < a.M) {
                device float * o = y + (size_t) r * a.ldy + r0 + j;
                *o = a.acc == 1 ? *o + v : a.acc == 2 ? act_silu(*o) * v : a.acc == 3 ? act_gelu(*o) * v : v;
            }
        }
    }
}
#define MV_Q4(BQ, R, MODE) template [[host_name("mv_" #BQ "_r" #R "_m" #MODE)]] kernel void mv_q4<block_##BQ, R, MODE>(constant mv_args &, \
    device const char *, device const float *, device float *, device const float *, uint, ushort, ushort);
#define MV_Q4_R(BQ, MODE) MV_Q4(BQ, 1, MODE) MV_Q4(BQ, 2, MODE) MV_Q4(BQ, 3, MODE) MV_Q4(BQ, 4, MODE)
MV_Q4_R(q4_0, 0) MV_Q4_R(q4_0, 1) MV_Q4_R(q4_0, 2) MV_Q4_R(q4_1, 0) MV_Q4_R(q4_1, 2)

// ---- few-row block matmul for a wide draft check (2-8 rows of one writer): ggml b11669's kernel_mul_mv_mma_blk for Q4_0 ----
// A threadgroup takes 8 NT weight rows x 8 token rows, its NSG simdgroups the K blocks in turn (block ib in simdgroup ib %
// NSG). A block is 4 steps of 8x8 MMAs: the weights as halves 1024 + q minus 1024 + 8 (exact; a high nibble left in place
// as 16 q, its inputs / 16), the inputs as floats, the next block's loads in flight; then d times the block onto the sums.
// The simdgroups' sums are added in simdgroup order. A row's bits are not the one-row kernels' (a wide check's are not).
struct mvm_args { int K; int M; int R; int ldx; int ldy; int acc; int nb01; };
static inline short mvm_fm(ushort l) { return ((l / 4) & 4) + ((l / 2) % 4); }  // the 8x8 matrix row a lane holds
static inline short mvm_fn(ushort l) { return ((l / 4) & 2) * 2 + (l % 2) * 2; }  // and its first column (of two)
static inline half2 mvm_frag(ushort2 q, short s) {
    const ushort2 qq = s < 2 ? q : q >> 8;
    return s % 2 == 0 ? as_type<half2>((qq & ushort2(0x000F)) | ushort2(0x6400)) - half2(1024.0h + 8.0h)
                      : as_type<half2>((qq & ushort2(0x00F0)) | ushort2(0x6400)) - half2(1024.0h + 128.0h);
}
template <int NT, int NSG>
kernel void mvm_q4_0(constant mvm_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * x [[buffer(2)]],
                     device float * y [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float red[NSG * NT * 64];
    const short fm = mvm_fm(ln), fn = mvm_fn(ln);
    const int m0 = tg * 8 * NT, nb = a.K / 32;
    device const ushort * wp[NT];  // the lane's two quant ushorts (fn, fn + 1) of its weight row, block 0
    for (int t = 0; t < NT; ++t) wp[t] = (device const ushort *) (W + (size_t) min(m0 + 8 * t + fm, a.M - 1) * a.nb01) + 1 + fn;
    device const float2 * xp[2];  // the lane's inputs 2 fm, 2 fm + 1 (and + 16) of token rows fn, fn + 1, block 0
    for (int e = 0; e < 2; ++e) xp[e] = (device const float2 *) (x + (size_t) min(fn + e, a.R - 1) * a.ldx + 2 * fm);
    float acc[NT][2];
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = 0.f;
    ushort2 q[NT];
    float d[NT];
    float2 b0[2], b1[2];
    {
        const int ib = min((int) sg, nb - 1);
        for (int t = 0; t < NT; ++t) { device const ushort * p = wp[t] + ib * 9; q[t] = ushort2(p[0], p[1]); d[t] = as_type<half>(*(p - 1 - fn)); }
        for (int e = 0; e < 2; ++e) { b0[e] = xp[e][ib * 16]; b1[e] = xp[e][ib * 16 + 8]; }
    }
    for (int ib = sg; ib < nb; ib += NSG) {
        ushort2 qc[NT];
        float dc[NT];
        float2 b0c[2], b1c[2];
        for (int t = 0; t < NT; ++t) { qc[t] = q[t]; dc[t] = d[t]; }
        for (int e = 0; e < 2; ++e) { b0c[e] = b0[e]; b1c[e] = b1[e] * (1.0f / 16); }
        const int in = min(ib + NSG, nb - 1);
        for (int t = 0; t < NT; ++t) { device const ushort * p = wp[t] + in * 9; q[t] = ushort2(p[0], p[1]); d[t] = as_type<half>(*(p - 1 - fn)); }
        for (int e = 0; e < 2; ++e) { b0[e] = xp[e][in * 16]; b1[e] = xp[e][in * 16 + 8]; }
        simdgroup_float8x8 mp[NT];
        for (int t = 0; t < NT; ++t) mp[t] = make_filled_simdgroup_matrix<float, 8>(0.f);
        for (short s = 0; s < 4; ++s) {
            simdgroup_float8x8 mb;
            const float2 v0 = s % 2 ? b1c[0] : b0c[0], v1 = s % 2 ? b1c[1] : b0c[1];
            mb.thread_elements()[0] = s >= 2 ? v0.y : v0.x;
            mb.thread_elements()[1] = s >= 2 ? v1.y : v1.x;
            for (int t = 0; t < NT; ++t) {
                const half2 h = mvm_frag(qc[t], s);
                simdgroup_half8x8 ma;
                ma.thread_elements()[0] = h.x;
                ma.thread_elements()[1] = h.y;
                simdgroup_multiply_accumulate(mp[t], ma, mb, mp[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            acc[t][0] = fma(dc[t], mp[t].thread_elements()[0], acc[t][0]);
            acc[t][1] = fma(dc[t], mp[t].thread_elements()[1], acc[t][1]);
        }
    }
    for (int t = 0; t < NT; ++t) { red[(sg * NT + t) * 64 + 2 * ln] = acc[t][0]; red[(sg * NT + t) * 64 + 2 * ln + 1] = acc[t][1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = sg * 32 + ln; i < NT * 64; i += NSG * 32) {
        float v = 0.f;
        for (int k = 0; k < NSG; ++k) v += red[k * NT * 64 + i];
        const int t = i / 64, l = (i % 64) / 2, r0 = m0 + 8 * t + mvm_fm(l), r1 = mvm_fn(l) + i % 2;
        if (r0 < a.M && r1 < a.R) {
            device float * o = y + (size_t) r1 * a.ldy + r0;
            *o = a.acc == 1 ? *o + v : a.acc == 2 ? act_silu(*o) * v : a.acc == 3 ? act_gelu(*o) * v : v;
        }
    }
}
#define MVM(NT, NSG) template [[host_name("mvm_q4_0_nt" #NT "_s" #NSG)]] kernel void mvm_q4_0<NT, NSG>(constant mvm_args &, \
    device const char *, device const float *, device float *, uint, ushort, ushort);
MVM(1, 4) MVM(2, 4) MVM(4, 4) MVM(1, 8) MVM(2, 8) MVM(4, 8) MVM(1, 16) MVM(2, 16) MVM(4, 16)

// Q4_K / Q6_K mat-vec for up to R = 4 token rows: ggml's kernel_mul_mv_q4_K / q6_K_f32 (2 weight rows a simdgroup, 2
// simdgroups a threadgroup: dispatch (M + 3) / 4 threadgroups of 64), each token row's inputs loaded once per 256-weight
// block and used by both weight rows; the epilogue of mv_q4 (acc: 1 add, 2 / 3 silu / gelu gate)
static inline float mv_epi(int acc, float old, float v) { return acc == 1 ? old + v : acc == 2 ? act_silu(old) * v : acc == 3 ? act_gelu(old) * v : v; }
template <int R>
inline void mvb_q4_K(constant mv_args & a, device const char * W, device const float * x,
                    device float * y, uint tg, ushort sg,
                    ushort ln) {
    constexpr int NR0 = 2;
    constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    const int nb = a.K / 256, rg = nb >= 4 ? 1 : 4 / nb;  // row groups (mv_rg): rows of under 4 blocks give idle lanes more rows
    const short ix0 = ln / 8, it = ln % 8, iq = it / 4, ir = it % 4, grp = ix0 / (4 / rg), ix = ix0 % (4 / rg);
    const int base = (tg * 2 + sg) * rg * NR0, first_row = base + grp * NR0;
    if (base >= a.M) return;
    device const block_q4_K * xw = (device const block_q4_K *) (W + (size_t) (first_row < a.M ? first_row : base) * a.nb01);
    float sumf[NR0][R];
    for (int i = 0; i < NR0; ++i) for (int r = 0; r < R; ++r) sumf[i][r] = 0.f;
    ushort sc16[4];
    thread const uchar * sc8 = (thread const uchar *) sc16;
    for (int ib = ix; ib < nb; ib += 4 / rg) {
        // one token row's inputs at a time (all rows' in registers at once spilled: 3 rows cost 4x one), the weights again from L1
        for (int r = 0; r < R; ++r) {
            float yl[16], yh[16];
            float4 sumy = 0.f;
            device const float * y4 = x + (size_t) r * a.ldx + ib * 256 + 64 * iq + 8 * ir;
            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }
            device const ushort * sc = (device const ushort *) xw[ib].scales + iq;
            device const ushort * q1 = (device const ushort *) xw[ib].qs + 16 * iq + 4 * ir;
            device const half * dh = &xw[ib].d;
            for (short row = 0; row < NR0; ++row) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
                device const ushort * q2 = q1 + 32;
                float4 acc1 = 0.f, acc2 = 0.f;
                for (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                }
                sumf[row][r] += float(dh[0]) * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] + (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                                (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] + (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                q1 += a.nb01 / 2; sc += a.nb01 / 2; dh += a.nb01 / 2;
            }
        }
    }
    for (int g = 0; g < rg; ++g)
        for (int row = 0; row < NR0; ++row)
            for (int r = 0; r < R; ++r) {
                const float v = simd_sum(grp == g ? sumf[row][r] : 0.f);
                const int m = base + g * NR0 + row;
                if (ln == 0 && m < a.M) { device float * o = y + (size_t) r * a.ldy + m; *o = mv_epi(a.acc, *o, v); }
            }
}
template <int R>
inline void mvb_q6_K(constant mv_args & a, device const char * W, device const float * x,
                    device float * y, uint tg, ushort sg,
                    ushort ln) {
    constexpr int NR0 = 2;
    constexpr uchar kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    const int nb = a.K / 256, first_row = (tg * 2 + sg) * NR0;
    if (first_row >= a.M) return;
    device const block_q6_K * xw = (device const block_q6_K *) (W + (size_t) first_row * a.nb01);
    const short tid = ln / 2, ix = ln % 2, ip = tid / 8, il = tid % 8, l0 = 4 * il, is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0, q_offset_l = 64 * ip + l0, q_offset_h = 32 * ip + l0;
    float sumf[NR0][R];
    for (int i = 0; i < NR0; ++i) for (int r = 0; r < R; ++r) sumf[i][r] = 0.f;
    for (int i = ix; i < nb; i += 2) {
        for (int r = 0; r < R; ++r) {  // one token row's inputs at a time (as mvb_q4_K)
            float yl[16];
            device const float * yy = x + (size_t) r * a.ldx + i * 256 + y_offset;
            for (short l = 0; l < 4; ++l) { yl[4 * l + 0] = yy[l]; yl[4 * l + 1] = yy[l + 32]; yl[4 * l + 2] = yy[l + 64]; yl[4 * l + 3] = yy[l + 96]; }
            device const uchar * q1 = xw[i].ql + q_offset_l;
            device const uchar * q2 = q1 + 32;
            device const uchar * qh = xw[i].qh + q_offset_h;
            device const char * sc = xw[i].scales + is;
            device const half * dh = &xw[i].d;
            for (short row = 0; row < NR0; ++row) {
                float4 sums = 0.f;
                for (short l = 0; l < 4; ++l) {
                    sums[0] += yl[4 * l + 0] * ((char) ((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                    sums[1] += yl[4 * l + 1] * ((char) ((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                    sums[2] += yl[4 * l + 2] * ((char) ((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                    sums[3] += yl[4 * l + 3] * ((char) ((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
                }
                sumf[row][r] += float(dh[0]) * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
                q1 += a.nb01; q2 += a.nb01; qh += a.nb01; sc += a.nb01; dh += a.nb01 / 2;
            }
        }
    }
    for (int row = 0; row < NR0; ++row)
        for (int r = 0; r < R; ++r) {
            const float v = simd_sum(sumf[row][r]);
            if (ln == 0 && first_row + row < a.M) { device float * o = y + (size_t) r * a.ldy + first_row + row; *o = mv_epi(a.acc, *o, v); }
        }
}
template <int R>
inline void mvb_q5_K(constant mv_args & a, device const char * W, device const float * x,
                    device float * y, uint tg, ushort sg,
                    ushort ln) {
    constexpr int NR0 = 2;
    constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    const int nb = a.K / 256, rg = nb >= 4 ? 1 : 4 / nb;  // row groups (mv_rg): rows of under 4 blocks give idle lanes more rows
    const short tid = ln / 4, ix0 = ln % 4, iq = tid / 4, ir = tid % 4, l0 = 8 * ir, q_offset = 32 * iq + l0, y_offset = 64 * iq + l0;
    const short grp = ix0 / (4 / rg), ix = ix0 % (4 / rg);
    const int base = (tg * 2 + sg) * rg * NR0, first_row = base + grp * NR0;
    if (base >= a.M) return;
    device const block_q5_K * xw = (device const block_q5_K *) (W + (size_t) (first_row < a.M ? first_row : base) * a.nb01);
    const uchar hm1 = 1u << (2 * iq), hm2 = hm1 << 1, hm3 = hm1 << 4, hm4 = hm2 << 4;
    float sumf[NR0][R];
    for (int i = 0; i < NR0; ++i) for (int r = 0; r < R; ++r) sumf[i][r] = 0.f;
    ushort sc16[4];
    thread const uchar * sc8 = (thread const uchar *) sc16;
    for (int i = ix; i < nb; i += 4 / rg) {
        for (int r = 0; r < R; ++r) {  // one token row's inputs at a time (as mvb_q4_K)
            float yl[16], yh[16];
            float4 sumy = 0.f;
            device const float * y1 = x + (size_t) r * a.ldx + i * 256 + y_offset, * y2 = y1 + 128;
            for (short l = 0; l < 8; ++l) {
                yl[l + 0] = y1[l + 0];  sumy[0] += yl[l + 0];
                yl[l + 8] = y1[l + 32]; sumy[1] += yl[l + 8];
                yh[l + 0] = y2[l + 0];  sumy[2] += yh[l + 0];
                yh[l + 8] = y2[l + 32]; sumy[3] += yh[l + 8];
            }
            device const uchar * q1 = xw[i].qs + q_offset;
            device const uchar * qh = xw[i].qh + l0;
            device const half * dh = &xw[i].d;
            device const ushort * sa = (device const ushort *) xw[i].scales + iq;
            for (short row = 0; row < NR0; ++row) {
                device const uchar * q2 = q1 + 64;
                sc16[0] = sa[0] & kmask1;
                sc16[1] = sa[2] & kmask1;
                sc16[2] = ((sa[4] >> 0) & kmask2) | ((sa[0] & kmask3) >> 2);
                sc16[3] = ((sa[4] >> 4) & kmask2) | ((sa[2] & kmask3) >> 2);
                float4 acc1 = 0.f, acc2 = 0.f;
                for (short l = 0; l < 8; ++l) {
                    const uchar h = qh[l];
                    acc1[0] += yl[l + 0] * (q1[l] & 0x0F);
                    acc1[1] += yl[l + 8] * (q1[l] & 0xF0);
                    acc1[2] += yh[l + 0] * (q2[l] & 0x0F);
                    acc1[3] += yh[l + 8] * (q2[l] & 0xF0);
                    acc2[0] += h & hm1 ? yl[l + 0] : 0.f;
                    acc2[1] += h & hm2 ? yl[l + 8] : 0.f;
                    acc2[2] += h & hm3 ? yh[l + 0] : 0.f;
                    acc2[3] += h & hm4 ? yh[l + 8] : 0.f;
                }
                sumf[row][r] += float(dh[0]) * (sc8[0] * (acc1[0] + 16.f * acc2[0]) + sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1]) +
                                                sc8[4] * (acc1[2] + 16.f * acc2[2]) + sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3])) -
                                float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                q1 += a.nb01; qh += a.nb01; dh += a.nb01 / 2; sa += a.nb01 / 2;
            }
        }
    }
    for (int g = 0; g < rg; ++g)
        for (int row = 0; row < NR0; ++row)
            for (int r = 0; r < R; ++r) {
                const float v = simd_sum(grp == g ? sumf[row][r] : 0.f);
                const int m = base + g * NR0 + row;
                if (ln == 0 && m < a.M) { device float * o = y + (size_t) r * a.ldy + m; *o = mv_epi(a.acc, *o, v); }
            }
}
template <int R>
kernel void mv_q4_K(constant mv_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * x [[buffer(2)]], device float * y [[buffer(3)]],
                    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) { mvb_q4_K<R>(a, W, x, y, tg, sg, ln); }
template <int R>
kernel void mv_q5_K(constant mv_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * x [[buffer(2)]], device float * y [[buffer(3)]],
                    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) { mvb_q5_K<R>(a, W, x, y, tg, sg, ln); }
template <int R>
kernel void mv_q6_K(constant mv_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * x [[buffer(2)]], device float * y [[buffer(3)]],
                    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) { mvb_q6_K<R>(a, W, x, y, tg, sg, ln); }
// MXFP4 for up to R = 4 rows: ggml's kernel_mul_mv_mxfp4_f32 lanes (2 a block: bytes 8 il .. of qs, weights 8 il .. and
// 16 + 8 il ..), 2 weight rows a simdgroup, 2 simdgroups a threadgroup (dispatch (M + 3) / 4 of 64)
template <int R>
inline void mvb_mxfp4(constant mv_args & a, device const char * W, device const float * x, device float * y, uint tg, ushort sg, ushort ln) {
    const int first_row = (tg * 2 + sg) * 2;
    if (first_row >= a.M) return;
    const int nb = a.K / 32, il = ln % 2;
    float sumf[2][R];
    for (int j = 0; j < 2; ++j) for (int r = 0; r < R; ++r) sumf[j][r] = 0.f;
    for (int ib = ln / 2; ib < nb; ib += 16) {
        float4 xl[R][2], xh[R][2];
        for (int r = 0; r < R; ++r) {
            device const float * xr = x + (size_t) r * a.ldx + ib * 32 + il * 8;
            xl[r][0] = *((device const float4 *) xr); xl[r][1] = *((device const float4 *) (xr + 4));
            xh[r][0] = *((device const float4 *) (xr + 16)); xh[r][1] = *((device const float4 *) (xr + 20));
        }
        for (int j = 0; j < 2; ++j) {
            device const block_mxfp4 & b = *((device const block_mxfp4 *) (W + (size_t) min(first_row + j, a.M - 1) * a.nb01) + ib);
            float4 l0, h0, l1, h1;
            mx_4(mx_u32(b.qs + il * 8), l0, h0);
            mx_4(mx_u32(b.qs + il * 8 + 4), l1, h1);
            const float d = mx_scale(b.e);
            for (int r = 0; r < R; ++r) sumf[j][r] += d * ((dot(l0, xl[r][0]) + dot(l1, xl[r][1])) + (dot(h0, xh[r][0]) + dot(h1, xh[r][1])));
        }
    }
    for (int j = 0; j < 2; ++j)
        for (int r = 0; r < R; ++r) {
            const float v = simd_sum(sumf[j][r]);
            if (ln == 0 && first_row + j < a.M) { device float * o = y + (size_t) r * a.ldy + first_row + j; *o = mv_epi(a.acc, *o, v); }
        }
}
// an MoE layer's interactive rows: one (row, slot) pair per tg.y, its expert's matrix (ids[pair], estride bytes apart), the
// input row of the pair (xpair) or of its row (pair / k_used), the output row of the pair
struct mvid_args { mv_args m; int k_used, xpair, estride; };
#define MVID(K_) kernel void mvid_##K_(constant mvid_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * x [[buffer(2)]], \
    device float * y [[buffer(3)]], device const int * ids [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort ln [[thread_index_in_simdgroup]]) { \
    const int p = tg.y; \
    mvb_##K_<1>(a.m, W + (size_t) ids[p] * a.estride, x + (size_t) (a.xpair ? p : p / a.k_used) * a.m.ldx, y + (size_t) p * a.m.ldy, tg.x, sg, ln); }
MVID(q4_K) MVID(q5_K) MVID(q6_K) MVID(mxfp4)
// SN_JOIN 16: an expert pair's gate then up in one dispatch (K-quants, Qwen mode): each threadgroup runs the unchanged mvb_ body for the
// gate (acc 0, stores the sum) and then for the up (acc 2: y = act_silu(stored gate) * sum) on the same rows and output; the lane that
// stores a row's gate sum is the lane that reads it back, so no barrier is needed and the gate value is the stored float
struct mvid2_args { mv_args mg, mu; int k_used, xpair, estride_g, estride_u; };
#define MVID2(K_) kernel void mvid2_##K_(constant mvid2_args & a [[buffer(0)]], device const char * Wg [[buffer(1)]], device const float * x [[buffer(2)]], \
    device float * y [[buffer(3)]], device const int * ids [[buffer(4)]], device const char * Wu [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], \
    ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) { \
    const int p = tg.y; \
    mvb_##K_<1>(a.mg, Wg + (size_t) ids[p] * a.estride_g, x + (size_t) (a.xpair ? p : p / a.k_used) * a.mg.ldx, y + (size_t) p * a.mg.ldy, tg.x, sg, ln); \
    mvb_##K_<1>(a.mu, Wu + (size_t) ids[p] * a.estride_u, x + (size_t) (a.xpair ? p : p / a.k_used) * a.mu.ldx, y + (size_t) p * a.mu.ldy, tg.x, sg, ln); }
MVID2(q4_K) MVID2(q5_K) MVID2(q6_K)
#define MV_QK(K_, R) template [[host_name("mv_" #K_ "_r" #R)]] kernel void mv_##K_<R>(constant mv_args &, device const char *, \
    device const float *, device float *, uint, ushort, ushort);
MV_QK(q4_K, 1) MV_QK(q4_K, 2) MV_QK(q4_K, 3) MV_QK(q4_K, 4) MV_QK(q6_K, 1) MV_QK(q6_K, 2) MV_QK(q6_K, 3) MV_QK(q6_K, 4)
MV_QK(q5_K, 1) MV_QK(q5_K, 2) MV_QK(q5_K, 3) MV_QK(q5_K, 4)

// token embedding from Q4_K blocks (a table, or the staged rows)
kernel void embed_q4k(constant embs_args & a [[buffer(0)]], device const int * tok [[buffer(1)]], device const block_q4_K * w [[buffer(2)]],
                      device float * x [[buffer(3)]], uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
    device const block_q4_K * row = w + (size_t) (a.staged ? (int) r : tok[r]) * (a.n / 256);
    for (int i = t; i < a.n; i += 256) {
        device const block_q4_K & b = row[i / 256];
        const int j = i % 256, sub = j / 32;
        int sc, m;
        q4k_sm(sub, b.scales, sc, m);
        const int q = (b.qs[32 * (sub / 2) + j % 32] >> (4 * (sub % 2))) & 0xF;
        x[(size_t) r * a.n + i] = (float(b.d) * float(sc) * float(q) - float(b.dmin) * float(m)) * a.scale;
    }
}

// ---- soft words on the GPU (sn_soft_config): the mix of a row's K best words' input rows, weighted by softmax(score / T) ----
// One threadgroup (1024 threads) a soft output row: so[r] = (the output row k of the step, the vector slot to write). The K best
// scores (end-of-generation words left out) by a radix select on the scores' ordered bits (11 + 11 + 10 bits, ties at the
// K-th score: the lowest ids), their weights, then the weighted sum of their table rows (embed_*'s arithmetic, times a.scale)
// into sv slot so[r].y.
struct soft_args { int V; int E; int K; int fmt; float temp; float scale; int neog; int eog[32]; };
inline uint soft_key(float v) {
    const uint b = as_type<uint>(v);
    return v != v ? 0u : ((b & 0x80000000u) ? ~b : (b | 0x80000000u));
}
inline float soft_emb(constant soft_args & a, device const uchar * w, int id, int i) {
    if (a.fmt == 5) {
        device const block_q8_0 * row = (device const block_q8_0 *) w + (size_t) id * (a.E / 32);
        return (float(row[i / 32].d) * float(row[i / 32].qs[i % 32])) * a.scale;
    }
    if (a.fmt == 0) {
        device const block_q4_0 * row = (device const block_q4_0 *) w + (size_t) id * (a.E / 32);
        device const block_q4_0 & b = row[i / 32];
        const int j = i % 32, q = j < 16 ? (b.qs[j] & 15) : (b.qs[j - 16] >> 4);
        return (float(b.d) * float(q - 8)) * a.scale;
    }
    if (a.fmt == 6) {
        device const block_q4_K * row = (device const block_q4_K *) w + (size_t) id * (a.E / 256);
        device const block_q4_K & b = row[i / 256];
        const int j = i % 256, sub = j / 32;
        int sc, m;
        q4k_sm(sub, b.scales, sc, m);
        const int q = (b.qs[32 * (sub / 2) + j % 32] >> (4 * (sub % 2))) & 0xF;
        return (float(b.d) * float(sc) * float(q) - float(b.dmin) * float(m)) * a.scale;
    }
    device const block_q6_K * row = (device const block_q6_K *) w + (size_t) id * (a.E / 256);
    device const block_q6_K & b = row[i / 256];
    const int j = i % 256, n = j / 128, jj = j % 128, qu = jj / 32, l = jj % 32;
    const uchar lb = b.ql[n * 64 + l + (qu & 1) * 32];
    const int lo = qu < 2 ? (lb & 0xF) : (lb >> 4);
    const int hi = (b.qh[n * 32 + l] >> (2 * qu)) & 3;
    const int q = (lo | (hi << 4)) - 32;
    return ((float(b.d) * float(b.scales[n * 8 + l / 16 + 2 * qu])) * float(q)) * a.scale;
}
kernel void soft_mix(constant soft_args & a [[buffer(0)]], device const float * lg [[buffer(1)]], device const int2 * so [[buffer(2)]],
                     device const uchar * w [[buffer(3)]], device float * sv [[buffer(4)]],
                     uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[2048];
    threadgroup atomic_uint cnt;
    threadgroup uint pre, krem, ties;
    threadgroup int sid[64];
    threadgroup float ssc[64], sw[64];
    device const float * x = lg + (size_t) so[r].x * a.V;
    if (t == 0) { pre = 0; krem = (uint) a.K; }
    for (int p = 0; p < 3; ++p) {
        const int bits = p < 2 ? 11 : 10, shift = p == 0 ? 21 : p == 1 ? 10 : 0;
        for (int i = t; i < 2048; i += 1024) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint pf = pre;
        for (int i = t; i < a.V; i += 1024) {
            bool ban = false;
            for (int b = 0; b < a.neog; ++b) ban = ban || i == a.eog[b];
            if (ban) continue;
            const uint k = soft_key(x[i]);
            if (p == 0 || (k >> (shift + bits)) == pf) atomic_fetch_add_explicit(&hist[(k >> shift) & ((1u << bits) - 1u)], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            uint cum = 0;
            int d = (1 << bits) - 1;
            for (; d > 0; --d) {
                const uint c = atomic_load_explicit(&hist[d], memory_order_relaxed);
                if (cum + c >= krem) break;
                cum += c;
            }
            ties = atomic_load_explicit(&hist[d], memory_order_relaxed);
            krem -= cum;
            pre = (pf << bits) | (uint) d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // pre: the K-th best key; krem of the `ties` words with that key are among the K (all of them when they are as many)
    const uint T = pre;
    const bool all_ties = krem == ties;
    if (t == 0) atomic_store_explicit(&cnt, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < a.V; i += 1024) {
        bool ban = false;
        for (int b = 0; b < a.neog; ++b) ban = ban || i == a.eog[b];
        if (ban) continue;
        const uint k = soft_key(x[i]);
        if (k > T || (k == T && all_ties)) {
            const uint at = atomic_fetch_add_explicit(&cnt, 1u, memory_order_relaxed);
            if (at < 64) { sid[at] = i; ssc[at] = x[i]; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        uint n = atomic_load_explicit(&cnt, memory_order_relaxed);
        if (!all_ties) {  // the lowest ids among the equal scores
            uint need = krem;
            for (int i = 0; i < a.V && need > 0; ++i) {
                bool ban = false;
                for (int b = 0; b < a.neog; ++b) ban = ban || i == a.eog[b];
                if (!ban && soft_key(x[i]) == T) { sid[n] = i; ssc[n] = x[i]; ++n; --need; }
            }
        }
        for (int k = 1; k < a.K; ++k) {  // by id: the same sums in the same order every run (the atomic order is not)
            const int id = sid[k];
            const float sc = ssc[k];
            int j = k - 1;
            for (; j >= 0 && sid[j] > id; --j) { sid[j + 1] = sid[j]; ssc[j + 1] = ssc[j]; }
            sid[j + 1] = id; ssc[j + 1] = sc;
        }
        float mx = -INFINITY;
        for (int k = 0; k < a.K; ++k) mx = max(mx, ssc[k]);
        float z = 0.0f;
        for (int k = 0; k < a.K; ++k) { sw[k] = exp((ssc[k] - mx) / a.temp); z += sw[k]; }
        for (int k = 0; k < a.K; ++k) sw[k] /= z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float * dst = sv + (size_t) so[r].y * a.E;
    for (int i = t; i < a.E; i += 1024) {
        float acc = 0.0f;
        for (int k = 0; k < a.K; ++k) acc += sw[k] * soft_emb(a, w, sid[k], i);
        dst[i] = acc;
    }
}

// a silent row's input: row si[r].x of x takes vector slot si[r].y of sv (soft_mix's output, the step before)
kernel void latin_sv(constant gather_args & a [[buffer(0)]], device const float * sv [[buffer(1)]], device const int2 * si [[buffer(2)]],
                     device float * x [[buffer(3)]], uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
    for (int i = t; i < a.n; i += 256) x[(size_t) si[r].x * a.n + i] = sv[(size_t) si[r].y * a.n + i];
}

// ---- the interactive rows' fused mat-vecs: the arithmetic of mv_q4 / mv_q8_0 (MODE 0) per output, fewer dispatches ----
// One 4-output group of rows r0.. of a q4_0 / q4_1 matrix for R rows: the partial sums of each lane (mv_q4's loop as is).
template <typename BQ, int R>
inline void mvq4_sums(constant mv_args & a, device const char * W, device const float * x, int r0, ushort ln, thread float (&sum)[4][R]) {
    const int nb = a.K / 32, il = ln % 2;
    for (int j = 0; j < 4; ++j) for (int r = 0; r < R; ++r) sum[j][r] = 0.f;
    for (int ib = ln / 2; ib < nb; ib += 16) {
        float4 xl[R][2], xh[R][2];
        for (int r = 0; r < R; ++r) {
            device const float * xr = x + (size_t) r * a.ldx;
            const int k = ib * 32 + il * 8;
            xl[r][0] = *((device const float4 *) (xr + k)); xl[r][1] = *((device const float4 *) (xr + k + 4));
            xh[r][0] = *((device const float4 *) (xr + k + 16)); xh[r][1] = *((device const float4 *) (xr + k + 20));
        }
        for (int j = 0; j < 4; ++j) {
            device const BQ & b = *((device const BQ *) (W + (size_t) (r0 + j) * a.nb01) + ib);
            device const ushort * qs = (device const ushort *) (b.qs + il * 8);
            const ushort4 q = ushort4(qs[0], qs[1], qs[2], qs[3]);
            const float4 l0 = float4(q.x & 15, (q.x >> 8) & 15, q.y & 15, (q.y >> 8) & 15);
            const float4 l1 = float4(q.z & 15, (q.z >> 8) & 15, q.w & 15, (q.w >> 8) & 15);
            const float4 h0 = float4((q.x >> 4) & 15, q.x >> 12, (q.y >> 4) & 15, q.y >> 12);
            const float4 h1 = float4((q.z >> 4) & 15, q.z >> 12, (q.w >> 4) & 15, q.w >> 12);
            const float d = b.d;
            for (int r = 0; r < R; ++r) {
                if (is_same<BQ, block_q4_1>::value) {
                    const float mm_ = ((device const block_q4_1 &) b).m;
                    const float sx = dot(xl[r][0] + xl[r][1] + xh[r][0] + xh[r][1], float4(1.f));
                    const float s = dot(l0, xl[r][0]) + dot(l1, xl[r][1]) + dot(h0, xh[r][0]) + dot(h1, xh[r][1]);
                    sum[j][r] += d * s + mm_ * sx;
                } else {
                    const float s = dot(l0 - 8.f, xl[r][0]) + dot(l1 - 8.f, xl[r][1]) + dot(h0 - 8.f, xh[r][0]) + dot(h1 - 8.f, xh[r][1]);
                    sum[j][r] += d * s;
                }
            }
        }
    }
}
// the same for q8_0 (mv_q8_0's loop as is)
template <int R>
inline void mvq8_sums(constant mv_args & a, device const char * W, device const float * x, int r0, ushort ln, thread float (&sum)[4][R]) {
    const int nb = a.K / 32, il = ln % 2;
    for (int j = 0; j < 4; ++j) for (int r = 0; r < R; ++r) sum[j][r] = 0.f;
    for (int ib = ln / 2; ib < nb; ib += 16) {
        float4 xl[R][2], xh[R][2];
        for (int r = 0; r < R; ++r) {
            device const float * xr = x + (size_t) r * a.ldx + ib * 32 + il * 8;
            xl[r][0] = *((device const float4 *) xr); xl[r][1] = *((device const float4 *) (xr + 4));
            xh[r][0] = *((device const float4 *) (xr + 16)); xh[r][1] = *((device const float4 *) (xr + 20));
        }
        for (int j = 0; j < 4; ++j) {
            device const block_q8_0 & b = *((device const block_q8_0 *) (W + (size_t) (r0 + j) * a.nb01) + ib);
            device const ushort * ql = (device const ushort *) (b.qs + il * 8);
            device const ushort * qh = (device const ushort *) (b.qs + 16 + il * 8);
            const float4 l0 = float4(as_type<char4>((uint) ql[0] | ((uint) ql[1] << 16))), l1 = float4(as_type<char4>((uint) ql[2] | ((uint) ql[3] << 16)));
            const float4 h0 = float4(as_type<char4>((uint) qh[0] | ((uint) qh[1] << 16))), h1 = float4(as_type<char4>((uint) qh[2] | ((uint) qh[3] << 16)));
            const float d = b.d;
            for (int r = 0; r < R; ++r) sum[j][r] += d * (dot(l0, xl[r][0]) + dot(l1, xl[r][1]) + dot(h0, xh[r][0]) + dot(h1, xh[r][1]));
        }
    }
}
template <typename BQ, int R>
inline void mv_sums(constant mv_args & a, device const char * W, device const float * x, int r0, ushort ln, thread float (&sum)[4][R]) {
    mvq4_sums<BQ, R>(a, W, x, r0, ln, sum);
}
template <>
inline void mv_sums<block_q8_0, 1>(constant mv_args & a, device const char * W, device const float * x, int r0, ushort ln, thread float (&sum)[4][1]) { mvq8_sums<1>(a, W, x, r0, ln, sum); }
template <>
inline void mv_sums<block_q8_0, 2>(constant mv_args & a, device const char * W, device const float * x, int r0, ushort ln, thread float (&sum)[4][2]) { mvq8_sums<2>(a, W, x, r0, ln, sum); }
template <>
inline void mv_sums<block_q8_0, 3>(constant mv_args & a, device const char * W, device const float * x, int r0, ushort ln, thread float (&sum)[4][3]) { mvq8_sums<3>(a, W, x, r0, ln, sum); }
template <>
inline void mv_sums<block_q8_0, 4>(constant mv_args & a, device const char * W, device const float * x, int r0, ushort ln, thread float (&sum)[4][4]) { mvq8_sums<4>(a, W, x, r0, ln, sum); }

// Up to three matrices with the same input (q, k, v) in one dispatch: threadgroups [0, tg1) the first, [tg1, tg2) the
// second, the rest the third; each output as mv_q4 / mv_q8_0 with acc 0.
struct mv3_args { mv_args s[3]; int tg1, tg2; };
template <typename BQ, int R>
kernel void mv3(constant mv3_args & a [[buffer(0)]], device const char * W0 [[buffer(1)]], device const char * W1 [[buffer(2)]],
                device const char * W2 [[buffer(3)]], device const float * x [[buffer(4)]], device float * y0 [[buffer(5)]],
                device float * y1 [[buffer(6)]], device float * y2 [[buffer(7)]], uint tg [[threadgroup_position_in_grid]],
                ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const int si = (int) tg < a.tg1 ? 0 : (int) tg < a.tg2 ? 1 : 2;
    const int lt = (int) tg - (si == 0 ? 0 : si == 1 ? a.tg1 : a.tg2);
    constant mv_args & s = a.s[si];
    device const char * W = si == 0 ? W0 : si == 1 ? W1 : W2;
    device float * y = si == 0 ? y0 : si == 1 ? y1 : y2;
    const int r0 = (lt * 2 + sg) * 4;
    if (r0 >= s.M) return;
    float sum[4][R];
    mv_sums<BQ, R>(s, W, x, r0, ln, sum);
    for (int j = 0; j < 4; ++j) {
        for (int r = 0; r < R; ++r) {
            const float v = simd_sum(sum[j][r]);
            if (ln == 0 && r0 + j < s.M) y[(size_t) r * s.ldy + r0 + j] = v;
        }
    }
}
// gate and up in one dispatch: act(gate) * up as the up projection's epilogue computes it (acc 2 silu, 3 gelu)
template <typename BQ, int R>
kernel void mvglu(constant mv_args & a [[buffer(0)]], device const char * Wg [[buffer(1)]], device const char * Wu [[buffer(2)]],
                  device const float * x [[buffer(3)]], device float * y [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const int r0 = (tg * 2 + sg) * 4;
    if (r0 >= a.M) return;
    float sg_[4][R], su[4][R];
    mv_sums<BQ, R>(a, Wg, x, r0, ln, sg_);
    mv_sums<BQ, R>(a, Wu, x, r0, ln, su);
    for (int j = 0; j < 4; ++j) {
        for (int r = 0; r < R; ++r) {
            const float g = simd_sum(sg_[j][r]), u = simd_sum(su[j][r]);
            if (ln == 0 && r0 + j < a.M) y[(size_t) r * a.ldy + r0 + j] = (a.acc == 2 ? act_silu(g) : act_gelu(g)) * u;
        }
    }
}
#define MVF(BQ, R) \
    template [[host_name("mv3_" #BQ "_r" #R)]] kernel void mv3<block_##BQ, R>(constant mv3_args &, device const char *, device const char *, \
        device const char *, device const float *, device float *, device float *, device float *, uint, ushort, ushort); \
    template [[host_name("mvglu_" #BQ "_r" #R)]] kernel void mvglu<block_##BQ, R>(constant mv_args &, device const char *, device const char *, \
        device const float *, device float *, uint, ushort, ushort);
#define MVF_R(BQ) MVF(BQ, 1) MVF(BQ, 2) MVF(BQ, 3) MVF(BQ, 4)
MVF_R(q4_0) MVF_R(q4_1) MVF_R(q8_0)

// Q8_0 for up to 4 rows (a draft model's weights): the lanes and activations of mv_q4, a byte per weight
template <int R>
kernel void mv_q8_0(constant mv_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * x [[buffer(2)]],
                    device float * y [[buffer(3)]], device const float * aux [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const int r0 = (tg * 2 + sg) * 4;
    if (r0 >= a.M) return;
    const int nb = a.K / 32, il = ln % 2;
    float sum[4][R];
    for (int j = 0; j < 4; ++j) for (int r = 0; r < R; ++r) sum[j][r] = 0.f;
    for (int ib = ln / 2; ib < nb; ib += 16) {
        float4 xl[R][2], xh[R][2];
        for (int r = 0; r < R; ++r) {
            device const float * xr = x + (size_t) r * a.ldx + ib * 32 + il * 8;
            xl[r][0] = *((device const float4 *) xr); xl[r][1] = *((device const float4 *) (xr + 4));
            xh[r][0] = *((device const float4 *) (xr + 16)); xh[r][1] = *((device const float4 *) (xr + 20));
        }
        for (int j = 0; j < 4; ++j) {
            device const block_q8_0 & b = *((device const block_q8_0 *) (W + (size_t) (r0 + j) * a.nb01) + ib);
            device const ushort * ql = (device const ushort *) (b.qs + il * 8);
            device const ushort * qh = (device const ushort *) (b.qs + 16 + il * 8);
            const float4 l0 = float4(as_type<char4>((uint) ql[0] | ((uint) ql[1] << 16))), l1 = float4(as_type<char4>((uint) ql[2] | ((uint) ql[3] << 16)));
            const float4 h0 = float4(as_type<char4>((uint) qh[0] | ((uint) qh[1] << 16))), h1 = float4(as_type<char4>((uint) qh[2] | ((uint) qh[3] << 16)));
            const float d = b.d;
            for (int r = 0; r < R; ++r) sum[j][r] += d * (dot(l0, xl[r][0]) + dot(l1, xl[r][1]) + dot(h0, xh[r][0]) + dot(h1, xh[r][1]));
        }
    }
    for (int j = 0; j < 4; ++j) {
        for (int r = 0; r < R; ++r) {
            const float v = simd_sum(sum[j][r]);
            if (ln == 0 && r0 + j < a.M) {
                device float * o = y + (size_t) r * a.ldy + r0 + j;
                *o = a.acc == 1 ? *o + v : a.acc == 2 ? act_silu(*o) * v : a.acc == 3 ? act_gelu(*o) * v : v;
            }
        }
    }
}
#define MV_Q8(R) template [[host_name("mv_q8_0_r" #R)]] kernel void mv_q8_0<R>(constant mv_args &, device const char *, device const float *, \
    device float *, device const float *, uint, ushort, ushort);
MV_Q8(1) MV_Q8(2) MV_Q8(3) MV_Q8(4)

// Q8_0 with K split over a threadgroup's 4 simdgroups (ggml's kernel_mul_mv_q8_0_f32 layout): 2 weight rows a threadgroup,
// simdgroup s takes blocks 8 s + ix, + 32, ... (lane: block ix = ln / 4, its quants 8 (ln % 4) ..), the 4 partials added in
// order; more threadgroups in flight than mv_q8_0's 8 rows each (dispatch (M + 1) / 2 of 128)
template <int R>
kernel void mv_q8_0k(constant mv_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * x [[buffer(2)]],
                     device float * y [[buffer(3)]], device const float * aux [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float part[4][2][R];
    const int r0 = tg * 2, nb = a.K / 32, ix = ln / 4, il = ln % 4;
    float sum[2][R];
    for (int j = 0; j < 2; ++j) for (int r = 0; r < R; ++r) sum[j][r] = 0.f;
    device const block_q8_0 * b0 = (device const block_q8_0 *) (W + (size_t) min(r0, a.M - 1) * a.nb01);
    device const block_q8_0 * b1 = (device const block_q8_0 *) (W + (size_t) min(r0 + 1, a.M - 1) * a.nb01);
    for (int ib = sg * 8 + ix; ib < nb; ib += 32) {
        float4 xl[R][2];
        for (int r = 0; r < R; ++r) {
            device const float4 * xr = (device const float4 *) (x + (size_t) r * a.ldx + ib * 32 + il * 8);
            xl[r][0] = xr[0]; xl[r][1] = xr[1];
        }
        for (int j = 0; j < 2; ++j) {
            device const block_q8_0 & b = (j ? b1 : b0)[ib];
            device const ushort * q = (device const ushort *) (b.qs + il * 8);
            const float4 q0 = float4(as_type<char4>((uint) q[0] | ((uint) q[1] << 16))), q1 = float4(as_type<char4>((uint) q[2] | ((uint) q[3] << 16)));
            const float d = b.d;
            for (int r = 0; r < R; ++r) sum[j][r] += d * (dot(q0, xl[r][0]) + dot(q1, xl[r][1]));
        }
    }
    for (int j = 0; j < 2; ++j)
        for (int r = 0; r < R; ++r) {
            const float v = simd_sum(sum[j][r]);
            if (ln == 0) part[sg][j][r] = v;
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && ln < 2 * R) {
        const int j = ln / R, r = ln % R;
        if (r0 + j < a.M) {
            const float v = (part[0][j][r] + part[1][j][r]) + (part[2][j][r] + part[3][j][r]);
            device float * o = y + (size_t) r * a.ldy + r0 + j;
            *o = a.acc == 1 ? *o + v : a.acc == 2 ? act_silu(*o) * v : a.acc == 3 ? act_gelu(*o) * v : v;
        }
    }
}
#define MV_Q8K(R) template [[host_name("mv_q8_0k_r" #R)]] kernel void mv_q8_0k<R>(constant mv_args &, device const char *, device const float *, \
    device float *, device const float *, uint, ushort, ushort);
MV_Q8K(1) MV_Q8K(2) MV_Q8K(3) MV_Q8K(4)

// SN_JOIN 8: up to 3 one-row Q8_0 mat-vecs over the same x in one dispatch. The body of mv_q8_0k<1> unchanged (same K split over
// the 4 simdgroups, 2 weight rows a threadgroup, same partial order), threadgroups [0, tg1) the first matrix, [tg1, tg2) the
// second, the rest the third; each matrix has its own mv_args (M, nb01, ldy) and output
struct mv3k_args { mv_args m[3]; int tg1, tg2; };
kernel void mv_q8_0kj(constant mv3k_args & aj [[buffer(0)]], device const char * W0 [[buffer(1)]], device const float * x [[buffer(2)]],
                      device float * y0 [[buffer(3)]], device const char * W1 [[buffer(4)]], device float * y1 [[buffer(5)]],
                      device const char * W2 [[buffer(6)]], device float * y2 [[buffer(7)]], uint tgj [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    constexpr int R = 1;
    const int sel = tgj < (uint) aj.tg1 ? 0 : tgj < (uint) aj.tg2 ? 1 : 2;
    constant mv_args & a = aj.m[sel];
    device const char * W = sel == 0 ? W0 : sel == 1 ? W1 : W2;
    device float * y = sel == 0 ? y0 : sel == 1 ? y1 : y2;
    const uint tg = tgj - (uint) (sel == 0 ? 0 : sel == 1 ? aj.tg1 : aj.tg2);
    threadgroup float part[4][2][R];
    const int r0 = tg * 2, nb = a.K / 32, ix = ln / 4, il = ln % 4;
    float sum[2][R];
    for (int j = 0; j < 2; ++j) for (int r = 0; r < R; ++r) sum[j][r] = 0.f;
    device const block_q8_0 * b0 = (device const block_q8_0 *) (W + (size_t) min(r0, a.M - 1) * a.nb01);
    device const block_q8_0 * b1 = (device const block_q8_0 *) (W + (size_t) min(r0 + 1, a.M - 1) * a.nb01);
    for (int ib = sg * 8 + ix; ib < nb; ib += 32) {
        float4 xl[R][2];
        for (int r = 0; r < R; ++r) {
            device const float4 * xr = (device const float4 *) (x + (size_t) r * a.ldx + ib * 32 + il * 8);
            xl[r][0] = xr[0]; xl[r][1] = xr[1];
        }
        for (int j = 0; j < 2; ++j) {
            device const block_q8_0 & b = (j ? b1 : b0)[ib];
            device const ushort * q = (device const ushort *) (b.qs + il * 8);
            const float4 q0 = float4(as_type<char4>((uint) q[0] | ((uint) q[1] << 16))), q1 = float4(as_type<char4>((uint) q[2] | ((uint) q[3] << 16)));
            const float d = b.d;
            for (int r = 0; r < R; ++r) sum[j][r] += d * (dot(q0, xl[r][0]) + dot(q1, xl[r][1]));
        }
    }
    for (int j = 0; j < 2; ++j)
        for (int r = 0; r < R; ++r) {
            const float v = simd_sum(sum[j][r]);
            if (ln == 0) part[sg][j][r] = v;
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && ln < 2 * R) {
        const int j = ln / R, r = ln % R;
        if (r0 + j < a.M) {
            const float v = (part[0][j][r] + part[1][j][r]) + (part[2][j][r] + part[3][j][r]);
            device float * o = y + (size_t) r * a.ldy + r0 + j;
            *o = a.acc == 1 ? *o + v : a.acc == 2 ? act_silu(*o) * v : a.acc == 3 ? act_gelu(*o) * v : v;
        }
    }
}

// Q6_K (the output layer) for up to 4 rows: a lane takes 16-weight groups ln, ln + 32, ... (one scale each)
template <int R>
kernel void mv_q6k(constant mv_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * x [[buffer(2)]],
                   device float * y [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort ln [[thread_index_in_simdgroup]]) {
    const int r0 = (tg * 2 + sg) * 4;
    if (r0 >= a.M) return;
    float sum[4][R];
    for (int j = 0; j < 4; ++j) for (int r = 0; r < R; ++r) sum[j][r] = 0.f;
    for (int g = ln; g < a.K / 16; g += 32) {
        const int sb = g / 16, j0 = (g % 16) * 16, n = j0 / 128, jj = j0 % 128, qu = jj / 32, l0 = jj % 32;
        // token rows in groups of RG (3 rows' inputs held at once; 4 rows in two halves of 2: all 4 held spilled, 3.6x one
        // row), the weights read again from L1 for the second half; the same arithmetic either way
        constexpr int RG = R == 4 ? 2 : R;
        for (int r1 = 0; r1 < R; r1 += RG) {
            float xv[RG][16];
            for (int r = 0; r < RG; ++r) for (int i = 0; i < 16; ++i) xv[r][i] = x[(size_t) (r1 + r) * a.ldx + g * 16 + i];
            for (int j = 0; j < 4; ++j) {
                device const block_q6_K & b = *((device const block_q6_K *) (W + (size_t) (r0 + j) * a.nb01) + sb);
                device const uchar * ql = b.ql + n * 64 + l0 + (qu & 1) * 32;
                device const uchar * qh = b.qh + n * 32 + l0;
                const float sc = float(b.d) * float(b.scales[g % 16]);
                float s_[RG];
                for (int r = 0; r < RG; ++r) s_[r] = 0.f;
                for (int i = 0; i < 16; ++i) {
                    const int lo = qu < 2 ? (ql[i] & 0xF) : (ql[i] >> 4);
                    const float qv = float((lo | (((qh[i] >> (2 * qu)) & 3) << 4)) - 32);
                    for (int r = 0; r < RG; ++r) s_[r] += qv * xv[r][i];
                }
                for (int r = 0; r < RG; ++r) sum[j][r1 + r] += sc * s_[r];
            }
        }
    }
    for (int j = 0; j < 4; ++j) {
        for (int r = 0; r < R; ++r) {
            const float v = simd_sum(sum[j][r]);
            if (ln == 0 && r0 + j < a.M) y[(size_t) r * a.ldy + r0 + j] = v;
        }
    }
}
template [[host_name("mv_q6k_r1")]] kernel void mv_q6k<1>(constant mv_args &, device const char *, device const float *, device float *, uint, ushort, ushort);
template [[host_name("mv_q6k_r2")]] kernel void mv_q6k<2>(constant mv_args &, device const char *, device const float *, device float *, uint, ushort, ushort);
template [[host_name("mv_q6k_r3")]] kernel void mv_q6k<3>(constant mv_args &, device const char *, device const float *, device float *, uint, ushort, ushort);
template [[host_name("mv_q6k_r4")]] kernel void mv_q6k<4>(constant mv_args &, device const char *, device const float *, device float *, uint, ushort, ushort);

// ---- prompt attention on the matrix units (the design of SLLM's pfa kernel, phase2-prefillattn, without ggml's mask) ---
// A tile = RT = 32 rows = 8 consecutive prompt tokens of one sequence x the G = 4 query heads of one KV head, so every K/V
// block feeds four heads. Cells in blocks of 64: S = Q K^T and O += P V on matmul2d, one simdgroup per row for the
// softmax (2 cells per lane, the lazy rescale of ggml's kernel: the running max moves only when it grows by more than 8),
// causal by position. A row's result depends only on its own position: masked cells give P = 0 and add exactly nothing
// (the cache is zero-filled, so no NaN from cells never written).
constant constexpr int PRT = 32, PC = 64, PNSG = 4;
struct pre2_args { int n_head; int n_kv; int ctx; float scale; int ldq; int ldo; int swap; };

// G query heads per KV head: 4 (Qwen3-4B, Llama 3.1 8B; the names without a suffix: 8-token tiles), 2 (Qwen3-0.6B: 16-token
// tiles), 8 (Qwen3-32B: 4-token tiles) or 7 (Qwen2.5-7B: 4-token tiles, rows 28 .. 31 unused)
template <int G>
kernel void pre2_q(constant pre2_args & a [[buffer(0)]], device const float * q [[buffer(1)]], device const int4 * blk [[buffer(2)]],
                   device half * qh [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    constexpr int PG = G, PTT = PRT / G;
    const int d4 = gid.x, rl = gid.y, hk = gid.z / 1;
    const int b = gid.z / a.n_kv, kvh = gid.z % a.n_kv;
    const int4 bl = blk[b];
    const int g = rl / PTT, tl = rl % PTT;
    float4 x = 0.f;
    if (tl < bl.y && g < PG) x = *((device const float4 *) (q + (size_t) (bl.x + tl) * a.ldq + (kvh * PG + g) * 128) + d4);
    *((device half4 *) (qh + (((size_t) b * a.n_kv + kvh) * PRT + rl) * 128) + d4) = (half4) (x * a.scale);
    (void) hk;
}

template <int G>
kernel void pre2(constant pre2_args & a [[buffer(0)]], device const half * qh [[buffer(1)]], device const half * kc [[buffer(2)]],
                 device const half * vc [[buffer(3)]], device const int4 * blk [[buffer(4)]], device const int * pos [[buffer(5)]],
                 device float * o [[buffer(6)]], device const int * pt [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                 ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int PG = G, PTT = PRT / G;
    constexpr int D = 128, NW = 32, NQ = PRT / PNSG, NC = PC / NW;
    threadgroup float ss[PRT * PC];
    threadgroup half sp[PRT * PC];
    threadgroup float sa[PRT];
    const int hk = a.swap ? tg.y : tg.x, b = a.swap ? tg.x : tg.y;
    const int4 bl = blk[b];
    const int row0 = bl.x, nr = bl.y, s = bl.z;
    const int pmax = pos[row0 + nr - 1], pmin = pos[row0];
    constexpr auto dqk = matmul2d_descriptor(PRT, PC, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto dpv = matmul2d_descriptor(PRT, D, PC, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dqk, execution_simdgroups<PNSG>> mqk;
    matmul2d<dpv, execution_simdgroups<PNSG>> mpv;
    const int ld = a.n_kv * 128;
    device const int * ps = pt + (size_t) s * (a.ctx / PC);  // block cb (a multiple of PC = 64) is page ps[cb / 64]
    device half * kh = (device half *) kc + (size_t) ps[0] * PC * ld + hk * 128;
    device half * vh = (device half *) vc + (size_t) ps[0] * PC * ld + hk * 128;
    auto tQ = tensor((device half *) qh + ((size_t) b * a.n_kv + hk) * PRT * D, dextents<int32_t, 2>(D, PRT), array<int, 2>({1, D}));
    auto tS = tensor(ss, dextents<int32_t, 2>(PC, PRT), array<int, 2>({1, PC}));
    auto tP = tensor(sp, dextents<int32_t, 2>(PC, PRT), array<int, 2>({1, PC}));
    auto tK0 = tensor(kh, dextents<int32_t, 2>(D, PC), array<int, 2>({1, ld}));
    auto tV0 = tensor(vh, dextents<int32_t, 2>(D, PC), array<int, 2>({1, ld}));
    auto cS = mqk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK0), float>();
    auto cO = mpv.template get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (cO.is_valid_element(i)) cO[i] = 0.0f;
    }
    float M[NQ], S[NQ];
    int rp[NQ];
    #pragma unroll
    for (int jj = 0; jj < NQ; ++jj) {
        const int j = jj * PNSG + sgitg, tl = j % PTT;
        rp[jj] = tl < nr ? pos[row0 + tl] : -1;
        M[jj] = -FLT_MAX / 2;
        S[jj] = 0.f;
    }
    for (int cb = 0; cb <= pmax; cb += PC) {
        const bool full = cb + PC - 1 <= pmin;  // every cell visible to every row of the tile
        auto tK = tensor((device half *) kc + (size_t) ps[cb / PC] * PC * ld + hk * 128, dextents<int32_t, 2>(D, PC), array<int, 2>({1, ld}));
        mqk.run(tQ, tK, cS);
        cS.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int jj = 0; jj < NQ; ++jj) {
            const int j = jj * PNSG + sgitg;
            float sv[NC];
            #pragma unroll
            for (int ii = 0; ii < NC; ++ii) {
                const int c = cb + ii * NW + tiisg;
                sv[ii] = (full || c <= rp[jj]) ? ss[j * PC + ii * NW + tiisg] : -INFINITY;
            }
            float m = M[jj];
            #pragma unroll
            for (int ii = 0; ii < NC; ++ii) m = max(m, sv[ii]);
            m = simd_max(m);
            float ms = 1.0f;
            if (m > M[jj] + 8.0f) {
                ms = exp(M[jj] - m);
                M[jj] = m;
            }
            float sum = 0.f;
            #pragma unroll
            for (int ii = 0; ii < NC; ++ii) {
                const half p = (half) exp(sv[ii] - M[jj]);
                sp[j * PC + ii * NW + tiisg] = p;
                sum += (float) p;
            }
            S[jj] = S[jj] * ms + simd_sum(sum);
            if (tiisg == 0) sa[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (cO.is_valid_element(i)) {
                const auto ix = cO.get_multidimensional_index(i);
                cO[i] *= sa[ix[1]];
            }
        }
        auto tV = tensor((device half *) vc + (size_t) ps[cb / PC] * PC * ld + hk * 128, dextents<int32_t, 2>(D, PC), array<int, 2>({1, ld}));
        mpv.run(tP, tV, cO);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (int jj = 0; jj < NQ; ++jj) {
        const int j = jj * PNSG + sgitg;
        if (tiisg == 0) sa[j] = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (cO.is_valid_element(i)) {
            const auto ix = cO.get_multidimensional_index(i);
            const int rr = ix[1], d = ix[0], tl = rr % PTT, g = rr / PTT;
            if (tl < nr && g < PG) o[(size_t) (row0 + tl) * a.ldo + (hk * PG + g) * D + d] = cO[i] * sa[rr];
        }
    }
}
#define PRE2(G, SUF) \
    template [[host_name("pre2_q" SUF)]] kernel void pre2_q<G>(constant pre2_args &, device const float *, device const int4 *, device half *, uint3); \
    template [[host_name("pre2" SUF)]] kernel void pre2<G>(constant pre2_args &, device const half *, device const half *, device const half *, \
                                                         device const int4 *, device const int *, device float *, device const int *, uint3, ushort, ushort);
PRE2(4, "") PRE2(2, "_g2") PRE2(8, "_g8") PRE2(7, "_g7")


// ==== Gemma 4 ===========================================================================================================
struct g4_args { int n; float scale; int ld; int ple_n; int n_layer; float eps; };

// token embedding from the q4_1 copy of token_embd (made at load from its Q4_K), times sqrt(n_embd)
kernel void g4_embed(constant g4_args & a [[buffer(0)]], device const int * tok [[buffer(1)]], device const block_q4_1 * w [[buffer(2)]],
                     device float * x [[buffer(3)]], uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
    device const block_q4_1 * row = w + (size_t) tok[r] * (a.n / 32);
    for (int i = t; i < a.n; i += 256) {
        const block_q4_1 b = row[i / 32];
        const int j = i % 32;
        const int q = j < 16 ? (b.qs[j] & 15) : (b.qs[j - 16] >> 4);
        x[(size_t) r * a.n + i] = (float(q) * float(b.d) + float(b.m)) * a.scale;
    }
}

// the per-layer token embeddings: raw Q5_K rows the CPU gathered (one per row of the step), times sqrt(256) (block_q5_K: above)
static inline void k_scale_min(int j, device const uchar * q, thread int & d, thread int & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
kernel void g4_ple_raw(constant g4_args & a [[buffer(0)]], device const block_q5_K * raw [[buffer(1)]], device float * out [[buffer(2)]],
                       uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
    for (int i = t; i < a.ple_n; i += 256) {
        device const block_q5_K & b = raw[(size_t) r * (a.ple_n / 256) + i / 256];
        const int j = i % 256, grp = j / 64, l = j % 32, hi = (j % 64) / 32;
        int sc, mn;
        k_scale_min(2 * grp + hi, b.scales, sc, mn);
        const int lo4 = hi ? (b.qs[grp * 32 + l] >> 4) : (b.qs[grp * 32 + l] & 0xF);
        const int h = (b.qh[l] >> (2 * grp + hi)) & 1;
        const float v = (float(b.d) * float(sc)) * float(lo4 + 16 * h) - float(b.dmin) * float(mn);
        out[(size_t) r * a.ple_n + i] = v * a.scale;
    }
}

// PLE inputs: proj (= per_layer_model_proj x, times 1/sqrt(n_embd)) RMS-normed per 256-slice with w, plus the raw table
// rows, times 1/sqrt(2); one simdgroup per (row, layer slice)
kernel void g4_ple_combine(constant g4_args & a [[buffer(0)]], device const float * proj [[buffer(1)]], device const float * raw [[buffer(2)]],
                           device const float * w [[buffer(3)]], device float * out [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                           ushort ln [[thread_index_in_simdgroup]]) {
    const int r = tg.x, il = tg.y;
    device const float * p = proj + (size_t) r * a.ple_n + il * 256;
    float v[8], ss = 0.f;
    for (int i = 0; i < 8; ++i) { v[i] = p[ln + 32 * i] * a.scale; ss += v[i] * v[i]; }
    const float sc = 1.0f / sqrt(simd_sum(ss) / 256.f + a.eps);
    device const float * q = raw + (size_t) r * a.ple_n + il * 256;
    for (int i = 0; i < 8; ++i) {
        const int j = ln + 32 * i;
        out[(size_t) r * a.ple_n + il * 256 + j] = ((v[i] * sc) * w[j] + q[j]) * 0.70710678118654752f;
    }
}

// x += b, then h = rmsnorm(x) w (gpt-oss: the attention output's bias with the FFN's input norm)
kernel void bias_norm(constant norm_args & a [[buffer(0)]], device float * x [[buffer(1)]], device const float * b [[buffer(2)]],
                      device const float * w [[buffer(3)]], device float * h [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                      ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    device float * xr = x + (size_t) r * a.ldx;
    float s = 0.f;
    for (int i = t; i < a.n; i += 256) { const float v = xr[i] + b[i]; xr[i] = v; s += v * v; }
    s = simd_sum(s);
    if (ln == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float sc = 1.0f / sqrt(tot / a.n + a.eps);
    for (int i = t; i < a.n; i += 256) h[(size_t) r * a.ldy + i] = (xr[i] * sc) * w[i];
}

// x = (x + rmsnorm(y) * w) * s: the post-norms with their residual adds (and the layer's output scale)
struct na_args { int n; float eps; int ldx; int ldy; int has_s; };
kernel void g4_norm_add(constant na_args & a [[buffer(0)]], device float * x [[buffer(1)]], device const float * y [[buffer(2)]],
                        device const float * w [[buffer(3)]], device const float * s [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                        ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    device const float * yr = y + (size_t) r * a.ldy;
    float ss = 0.f;
    for (int i = t; i < a.n; i += 256) ss += yr[i] * yr[i];
    ss = simd_sum(ss);
    if (ln == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float sc = 1.0f / sqrt(tot / a.n + a.eps);
    const float os = a.has_s ? s[0] : 1.0f;
    for (int i = t; i < a.n; i += 256) {
        device float * o = x + (size_t) r * a.ldx + i;
        *o = (*o + (yr[i] * sc) * w[i]) * os;
    }
}

// g4_norm_add then rmsnorm of the new x into h, in one dispatch: each half is its kernel's code as is (the norm reads x
// back after a device barrier), so the bits are those of the two dispatches
struct nan_args { na_args na; norm_args nr; };
kernel void g4_norm_add_norm(constant nan_args & a [[buffer(0)]], device float * x [[buffer(1)]], device const float * y [[buffer(2)]],
                             device const float * w [[buffer(3)]], device const float * s [[buffer(4)]], device const float * w2 [[buffer(5)]],
                             device float * h [[buffer(6)]], uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    {
        device const float * yr = y + (size_t) r * a.na.ldy;
        float ss = 0.f;
        for (int i = t; i < a.na.n; i += 256) ss += yr[i] * yr[i];
        ss = simd_sum(ss);
        if (ln == 0) part[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.f;
        for (int i = 0; i < 8; ++i) tot += part[i];
        const float sc = 1.0f / sqrt(tot / a.na.n + a.na.eps);
        const float os = a.na.has_s ? s[0] : 1.0f;
        for (int i = t; i < a.na.n; i += 256) {
            device float * o = x + (size_t) r * a.na.ldx + i;
            *o = (*o + (yr[i] * sc) * w[i]) * os;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    {
        device const float4 * xr = (device const float4 *) (x + (size_t) r * a.nr.ldx);
        float ss = 0.f;
        for (int i = t; i < a.nr.n / 4; i += 256) ss += dot(xr[i], xr[i]);
        ss = simd_sum(ss);
        if (ln == 0) part[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.f;
        for (int i = 0; i < 8; ++i) tot += part[i];
        const float sc = 1.0f / sqrt(tot / a.nr.n + a.nr.eps);
        device float4 * hr = (device float4 *) (h + (size_t) r * a.nr.ldy);
        for (int i = t; i < a.nr.n / 4; i += 256) hr[i] = (xr[i] * sc) * ((device const float4 *) w2)[i];
    }
}

// ggml's GELU (tanh form) of g times u; u has its own row stride (the PLE slice of a layer)
struct gm_args { int n; int ldg; int ldu; int rows; };
kernel void g4_gelu_mul(constant gm_args & a [[buffer(0)]], device float * g [[buffer(1)]], device const float * u [[buffer(2)]],
                        uint2 i [[thread_position_in_grid]]) {
    if ((int) i.x >= a.n || (int) i.y >= a.rows) return;
    device float * gp = g + (size_t) i.y * a.ldg + i.x;
    const float x = *gp;
    const float ge = 0.5f * x * (1.0f + precise::tanh(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
    *gp = ge * u[(size_t) i.y * a.ldu + i.x];
}

// per-head RMS norm of q (weight) and k (weight), plain RMS norm of v (Gemma 4; Gemma 3 keeps v as is), NEOX rope over the
// whole head (positions times fscale, with the full layers' frequency factors), each K/V head's k and v into the cache when
// the layer keeps notes (a cell holds n_kv heads). D = head size, a lane holds D/64 rotation pairs.
// oss (gpt-oss): no norms; q, k, v plus their biases (qw, kw, vb); YaRN (llama.cpp's rope_yarn, ext_factor 1): the angle mixes
// the interpolated (fscale) and the extrapolated one by the ramp over pairs ylo .. yhi, cos and sin times ymscale
struct g4q_args { int n_head; int n_kv; int ctx; float eps; float base; float fscale; int ldq; int ldk; int has_kv; int has_ff; int vnorm;
                  int oss; float ylo, yhi, ymscale; };
constant int G4QK_TGS = 4;  // a row's threadgroups (sn.m's dispatch): 8 heads each in turn
template <int D>
kernel void g4_qk_rope(constant g4q_args & a [[buffer(0)]], device float * q [[buffer(1)]], device const float * k [[buffer(2)]],
                       device const float * v [[buffer(3)]], device const float * qw [[buffer(4)]], device const float * kw [[buffer(5)]],
                       device half * kc [[buffer(6)]], device half * vc [[buffer(7)]], device const int * seq [[buffer(8)]],
                       device const int * pos [[buffer(9)]], device const float * ff [[buffer(10)]], device const int * pt [[buffer(11)]],
                       device const float * vb [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort ln [[thread_index_in_simdgroup]]) {
    constexpr int NP = D / 64;
    const int r = tg.x, p = pos[r];
    float cs[NP], sn[NP];
    for (int t = 0; t < NP; ++t) {
        const int i = ln + 32 * t;
        if (a.oss) {
            const float te = float(p) * pow(a.base, (-1.0f / float(D)) * float(2 * i)), ti = a.fscale * te;
            const float mix = 1.0f - min(1.0f, max(0.0f, (float(i) - a.ylo) / max(0.001f, a.yhi - a.ylo)));
            const float th = ti * (1 - mix) + te * mix;
            cs[t] = cos(th) * a.ymscale; sn[t] = sin(th) * a.ymscale;
            continue;
        }
        float th = float(p) * a.fscale * pow(a.base, -float(2 * i) / float(D));
        if (a.has_ff) th /= ff[i];
        cs[t] = cos(th); sn[t] = sin(th);
    }
    const size_t cell0 = ((size_t) pt[(size_t) seq[r] * (a.ctx / 64) + p / 64] * 64 + p % 64) * a.n_kv * D;
    const int nh = a.n_head + (a.has_kv ? 2 * a.n_kv : 0);
    for (int h = tg.y * 8 + sg; h < nh; h += 8 * G4QK_TGS) {  // threadgroup tg.y of G4QK_TGS: heads 8 tg.y + sg, ...
        const int kind = h < a.n_head ? 0 : h < a.n_head + a.n_kv ? 1 : 2;  // q, k, v
        const int kvh = kind == 1 ? h - a.n_head : h - a.n_head - a.n_kv;
        const size_t cell = cell0 + (size_t) kvh * D;
        device const float * src = kind == 0 ? q + (size_t) r * a.ldq + h * D : (kind == 1 ? k : v) + (size_t) r * a.ldk + kvh * D;
        float x0[NP], x1[NP], ss = 0.f;
        for (int t = 0; t < NP; ++t) { x0[t] = src[ln + 32 * t]; x1[t] = src[ln + 32 * t + D / 2]; ss += x0[t] * x0[t] + x1[t] * x1[t]; }
        if (a.oss) {
            device const float * b = kind == 0 ? qw + h * D : (kind == 1 ? kw : vb) + kvh * D;
            for (int t = 0; t < NP; ++t) { x0[t] += b[ln + 32 * t]; x1[t] += b[ln + 32 * t + D / 2]; }
        }
        const float sc = (kind == 2 && !a.vnorm) || a.oss ? 1.0f : 1.0f / sqrt(simd_sum(ss) / float(D) + a.eps);
        if (kind == 2) {
            for (int t = 0; t < NP; ++t) { vc[cell + ln + 32 * t] = half(x0[t] * sc); vc[cell + ln + 32 * t + D / 2] = half(x1[t] * sc); }
            continue;
        }
        device const float * w = kind == 0 ? qw : kw;
        for (int t = 0; t < NP; ++t) {
            const int i = ln + 32 * t;
            const float a0 = a.oss ? x0[t] : (x0[t] * sc) * w[i], a1 = a.oss ? x1[t] : (x1[t] * sc) * w[i + D / 2];
            const float y0 = a0 * cs[t] - a1 * sn[t], y1 = a0 * sn[t] + a1 * cs[t];
            if (kind == 0) { device float * o = q + (size_t) r * a.ldq + h * D; o[i] = y0; o[i + D / 2] = y1; }
            else { kc[cell + i] = half(y0); kc[cell + i + D / 2] = half(y1); }
        }
    }
}
#define G4QK(D) template [[host_name("g4_qk_rope_" #D)]] kernel void g4_qk_rope<D>(constant g4q_args &, device float *, device const float *, \
    device const float *, device const float *, device const float *, device half *, device half *, device const int *, device const int *, \
    device const float *, device const int *, device const float *, uint2, ushort, ushort);
G4QK(64) G4QK(128) G4QK(256) G4QK(512)

// write attention for head size D, G query heads per K/V head, an optional window: chunks of 64 cells (fixed by cell
// index), one cell per lane, merged in chunk order. Partials: (row, head, chunk) -> D accumulators, m, l.
// A threadgroup per (row, hg query heads, chunk), a simdgroup per head; a cell holds the n_kv heads' K (V) one after another.
struct g4a_args { int n_head; int n_kv; int ctx; float scale; int ldq; int ldo; int maxc; int window; int n_fast; int maxch; int hg; int sinks; int msp; };  // n_fast, maxch: attn2_args'; sinks: gpt-oss's (merge)
template <int D>
kernel void g4_attn_part(constant g4a_args & a [[buffer(0)]], device const float * q [[buffer(1)]], device const half * kc [[buffer(2)]],
                         device const half * vc [[buffer(3)]], device const int * seq [[buffer(4)]], device const int * pos [[buffer(5)]],
                         device float * part [[buffer(6)]], device const int * pt [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    constexpr int NV = D / 32;
    threadgroup float4 qs[8][D / 4];
    const int r = tg.x, c = tg.z, h = tg.y * a.hg + sg, p = pos[r], kvs = a.n_kv * D;
    const int kvh = h / (a.n_head / a.n_kv);
    const int lo = a.window > 0 ? max(0, p - a.window + 1) : 0;
    if (r < a.n_fast && p / 64 - lo / 64 + 1 <= a.maxch) return;
    const int c0 = max(c * 64, lo), c1 = min(c * 64 + 64, p + 1);
    device float * o = part + (((size_t) r * a.n_head + h) * a.maxc + c) * (D + 2);
    if (c0 >= c1) { if (ln == 0) { o[D] = -INFINITY; o[D + 1] = 0.f; } for (int j = 0; j < NV; ++j) o[ln + 32 * j] = 0.f; return; }
    for (int i = ln; i < D / 4; i += 32) qs[sg][i] = *((device const float4 *) (q + (size_t) r * a.ldq + h * D) + i) * a.scale;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const size_t pg = (size_t) pt[(size_t) seq[r] * (a.ctx / 64) + c] * 64 * kvs;  // chunk c is one page
    device const half * kb = kc + pg + kvh * D;
    device const half * vb = vc + pg + kvh * D;
    float m = -INFINITY, l = 0.f, acc[NV];
    for (int j = 0; j < NV; ++j) acc[j] = 0.f;
    for (int cb = c0; cb < c1; cb += 32) {
        const int cell = cb + ln;
        float sc = -INFINITY;
        if (cell < c1) {
            device const half4 * kr = (device const half4 *) (kb + (size_t) (cell - c * 64) * kvs);
            float4 t = 0.f;
            for (int i = 0; i < D / 4; ++i) t = fma(float4(kr[i]), qs[sg][i], t);
            sc = (t.x + t.y) + (t.z + t.w);
        }
        const float mn = max(m, simd_max(sc));
        const float e0 = exp(m - mn);
        const float pc = cell < c1 ? exp(sc - mn) : 0.f;
        l = l * e0 + simd_sum(pc);
        for (int j = 0; j < NV; ++j) acc[j] *= e0;
        const int n = min(32, c1 - cb);
        for (int jj = 0; jj < n; ++jj) {
            const float pj = simd_shuffle(pc, jj);
            device const half * vr = vb + (size_t) (cb + jj - c * 64) * kvs;
            for (int j = 0; j < NV; ++j) acc[j] += float(vr[ln + 32 * j]) * pj;
        }
        m = mn;
    }
    for (int j = 0; j < NV; ++j) o[ln + 32 * j] = acc[j];
    if (ln == 0) { o[D] = m; o[D + 1] = l; }
}
template <int D>
kernel void g4_attn_merge(constant g4a_args & a [[buffer(0)]], device const float * part [[buffer(1)]], device const int * pos [[buffer(2)]],
                          device float * out [[buffer(3)]], device const float * sk [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          ushort ln [[thread_index_in_simdgroup]]) {
    constexpr int NV = D / 32;
    const int r = tg.x, h = tg.y, p = pos[r];
    const int lo = a.window > 0 ? max(0, p - a.window + 1) : 0, sp = r < a.n_fast ? fa_span(p / 64 + 1, a.msp, a.window) : 1;
    device const float * p0 = part + ((size_t) r * a.n_head + h) * a.maxc * (D + 2);
    float m = -INFINITY, l = 0.f, acc[NV];
    for (int j = 0; j < NV; ++j) acc[j] = 0.f;
    for (int c = lo / 64; c <= p / 64 / sp; ++c) {
        device const float * pc = p0 + c * (D + 2);
        const float mc = pc[D], lc = pc[D + 1], mn = max(m, mc);
        const float e0 = exp(m - mn), e1 = exp(mc - mn);
        for (int j = 0; j < NV; ++j) acc[j] = acc[j] * e0 + pc[ln + 32 * j] * e1;
        l = l * e0 + lc * e1;
        m = mn;
    }
    if (a.sinks) l += exp(sk[h] - m);  // the sink: a logit with no value
    for (int j = 0; j < NV; ++j) out[(size_t) r * a.ldo + h * D + ln + 32 * j] = acc[j] / l;
}
#define G4ATT(D) template [[host_name("g4_attn_part_" #D)]] kernel void g4_attn_part<D>(constant g4a_args &, device const float *, \
    device const half *, device const half *, device const int *, device const int *, device float *, device const int *, uint3, ushort, ushort); \
    template [[host_name("g4_attn_merge_" #D)]] kernel void g4_attn_merge<D>(constant g4a_args &, device const float *, device const int *, \
    device float *, device const float *, uint2, ushort);
G4ATT(64) G4ATT(96) G4ATT(128) G4ATT(256) G4ATT(512)

// ---- one-row write attention for the interactive rows (Qwen3 and Gemma 4) ------------------------------------------------
// A threadgroup per (row, query head, 64-cell chunk): the scores with the lanes across the head (coalesced reads), the
// chunk's softmax and P V in threadgroup memory. The partial is attn_part's / g4_attn_part's (acc[D], max, sum), merged by
// attn_merge / g4_attn_merge in chunk order, so a row's result depends only on its own cells.
struct fa1_args { int n_head; int n_kv; int ctx; float scale; int ldq; int maxc; int window; int kvstride; int maxch; int rpt; int nrows; int msp; };  // msp: fa2's (fa_span)  // rpt, nrows: fa2's rows a threadgroup (0: 1), rows  // maxch: longer rows are attn_part's
template <int D>
kernel void fa1(constant fa1_args & a [[buffer(0)]], device const float * q [[buffer(1)]], device const half * kc [[buffer(2)]],
                device const half * vc [[buffer(3)]], device const int * seq [[buffer(4)]], device const int * pos [[buffer(5)]],
                device float * part [[buffer(6)]], device const int * pt [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    // the cache is far (a step streams the weights through it), so every pass issues all its loads before using one:
    // scores 8 lanes per cell (lane j: elements 4 (j + 8 i) .. + 3, coalesced), 16 cells a pass; P V 8 cells a pass
    constexpr int NT = 128, NI = D / 32, NO = D / NT;
    threadgroup float qs[D];
    threadgroup float ps[64];
    threadgroup float ml[2];
    const int r = tg.x, h = tg.y, c = tg.z, p = pos[r];
    const int kvh = h / (a.n_head / a.n_kv);
    const int lo = a.window > 0 ? max(0, p - a.window + 1) : 0;
    if (p / 64 - lo / 64 + 1 > a.maxch) return;  // a long row: attn_part / g4_attn_part's
    const int c0 = max(c * 64, lo), c1 = min(c * 64 + 64, p + 1);
    device float * o = part + (((size_t) r * a.n_head + h) * a.maxc + c) * (D + 2);
    if (c0 >= c1) {
        for (int d = tid; d < D; d += NT) o[d] = 0.f;
        if (tid == 0) { o[D] = -INFINITY; o[D + 1] = 0.f; }
        return;
    }
    for (int d = tid; d < D; d += NT) qs[d] = q[(size_t) r * a.ldq + h * D + d] * a.scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const size_t pg = (size_t) pt[(size_t) seq[r] * (a.ctx / 64) + c] * 64 * a.kvstride;  // chunk c is one page
    device const half * kb = kc + pg + kvh * D;
    device const half * vb = vc + pg + kvh * D;
    const int j8 = ln % 8, cg = (int) sg * 4 + ln / 8;  // lane in its cell's group; the group (16 a pass)
    float4 qv[NI];
    for (int i = 0; i < NI; ++i) qv[i] = *((threadgroup const float4 *) qs + j8 + 8 * i);
    const int b = c * 64;
    for (int pass = 0; pass < 4; ++pass) {
        const int cell = b + pass * 16 + cg;
        const bool ok = cell >= c0 && cell < c1;
        device const half4 * kr = (device const half4 *) (kb + (size_t) ((ok ? cell : c0) - b) * a.kvstride) + j8;
        half4 kv[NI];
        for (int i = 0; i < NI; ++i) kv[i] = kr[8 * i];
        float4 t = 0.f;
        for (int i = 0; i < NI; ++i) t = fma(float4(kv[i]), qv[i], t);
        float s_ = (t.x + t.y) + (t.z + t.w);
        s_ += simd_shuffle_xor(s_, 1);
        s_ += simd_shuffle_xor(s_, 2);
        s_ += simd_shuffle_xor(s_, 4);
        if (j8 == 0) ps[pass * 16 + cg] = s_;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {  // the chunk's max and sum, two cells per lane
        const int i0 = 2 * ln, i1 = 2 * ln + 1;
        const bool v0 = b + i0 >= c0 && b + i0 < c1, v1 = b + i1 >= c0 && b + i1 < c1;
        const float s0 = v0 ? ps[i0] : -INFINITY, s1 = v1 ? ps[i1] : -INFINITY;
        const float m = simd_max(max(s0, s1));
        const float e0 = v0 ? exp(s0 - m) : 0.f, e1 = v1 ? exp(s1 - m) : 0.f;
        ps[i0] = e0; ps[i1] = e1;
        const float l = simd_sum(e0 + e1);
        if (ln == 0) { ml[0] = m; ml[1] = l; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[NO];
    for (int j = 0; j < NO; ++j) acc[j] = 0.f;
    for (int c8 = c0; c8 < c1; c8 += 8) {
        half vv[8][NO];
        for (int u = 0; u < 8; ++u) {
            const int cell = min(c8 + u, c1 - 1);
            device const half * vr = vb + (size_t) (cell - b) * a.kvstride;
            for (int j = 0; j < NO; ++j) vv[u][j] = vr[tid + NT * j];
        }
        for (int u = 0; u < 8; ++u) {
            const float pc = c8 + u < c1 ? ps[c8 + u - b] : 0.f;
            for (int j = 0; j < NO; ++j) acc[j] = fma(float(vv[u][j]), pc, acc[j]);
        }
    }
    for (int j = 0; j < NO; ++j) o[tid + NT * j] = acc[j];
    if (tid == 0) { o[D] = ml[0]; o[D + 1] = ml[1]; }
}
// The same per KV head: a threadgroup per (row, KV head, 64-cell chunk), a simdgroup per query head of that KV head, so
// the chunk's K / V come from memory once and the heads share them in L1. A simdgroup's scores: 8 lanes per cell (lane j:
// elements 4 (j + 8 i) .., coalesced), 4 cells a pass, the loads of 4 passes in flight; its softmax and P V on its own
// (no threadgroup barrier). Partials as fa1's. A row of more than msp chunks: threadgroup z takes chunks z sp .. + sp - 1
// (fa_span) one after another, merged as the merge kernels do, into partial z.
template <int D, int GH, int RM>
kernel void fa2(constant fa1_args & a [[buffer(0)]], device const float * q [[buffer(1)]], device const half * kc [[buffer(2)]],
                device const half * vc [[buffer(3)]], device const int * seq [[buffer(4)]], device const int * pos [[buffer(5)]],
                device float * part [[buffer(6)]], device const int * pt [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    constexpr int NI = D / 32, NO = D / 32;  // half4s per lane per cell (scores); output elements per lane
    threadgroup float ps[GH * RM][64];
    // RM > 1 (fa2r): rpt <= RM rows a threadgroup, a simdgroup per (row, head), so a draft check's rows read the chunk's
    // K / V from memory once too (each simdgroup's work is the one-row threadgroup's: the same bits)
    const int rpt = RM > 1 ? a.rpt : 1, r = tg.x * rpt + sg / GH, kvh = tg.y, h = kvh * GH + sg % GH;
    if (RM > 1 && r >= a.nrows) return;
    const int p = pos[r];
    const int lo = a.window > 0 ? max(0, p - a.window + 1) : 0;
    const int nc = p / 64 + 1, sp = fa_span(nc, a.msp, a.window), cb = tg.z * sp, ce = min(cb + sp, nc);
    device float * o = part + (((size_t) r * a.n_head + h) * a.maxc + tg.z) * (D + 2);
    if (p / 64 - lo / 64 + 1 > a.maxch) return;  // a long row: attn_part / g4_attn_part's
    if (sp > 1 && cb >= nc) return;
    const int j8 = ln % 8, cg = ln / 8;
    float4 qv[NI];
    for (int i = 0; i < NI; ++i) qv[i] = *((device const float4 *) (q + (size_t) r * a.ldq + h * D) + j8 + 8 * i) * a.scale;
    float M = -INFINITY, L = 0.f, A[NO];
    for (int j = 0; j < NO; ++j) A[j] = 0.f;
    for (int c = cb; c < ce; ++c) {
        const int c0 = max(c * 64, lo), c1 = min(c * 64 + 64, p + 1), b = c * 64;
        if (c0 >= c1) {
            if (sp > 1) continue;
            for (int j = 0; j < NO; ++j) o[ln + 32 * j] = 0.f;
            if (ln == 0) { o[D] = -INFINITY; o[D + 1] = 0.f; }
            return;
        }
        const size_t pg = (size_t) pt[(size_t) seq[r] * (a.ctx / 64) + c] * 64 * a.kvstride;  // chunk c is one page
        device const half * kb = kc + pg + kvh * D;
        device const half * vb = vc + pg + kvh * D;
        for (int p4 = 0; p4 < 16; p4 += 4) {  // 16 passes of 4 cells, 4 passes' loads at a time
            half4 kv[4][NI];
            for (int u = 0; u < 4; ++u) {
                const int cell = b + (p4 + u) * 4 + cg;
                device const half4 * kr = (device const half4 *) (kb + (size_t) (cell >= c0 && cell < c1 ? cell - b : c0 - b) * a.kvstride) + j8;
                for (int i = 0; i < NI; ++i) kv[u][i] = kr[8 * i];
            }
            for (int u = 0; u < 4; ++u) {
                float4 t = 0.f;
                for (int i = 0; i < NI; ++i) t = fma(float4(kv[u][i]), qv[i], t);
                float s_ = (t.x + t.y) + (t.z + t.w);
                s_ += simd_shuffle_xor(s_, 1);
                s_ += simd_shuffle_xor(s_, 2);
                s_ += simd_shuffle_xor(s_, 4);
                if (j8 == 0) ps[sg][(p4 + u) * 4 + cg] = s_;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int i0 = 2 * ln, i1 = 2 * ln + 1;
        const bool v0 = b + i0 >= c0 && b + i0 < c1, v1 = b + i1 >= c0 && b + i1 < c1;
        const float s0 = v0 ? ps[sg][i0] : -INFINITY, s1 = v1 ? ps[sg][i1] : -INFINITY;
        const float m = simd_max(max(s0, s1));
        const float e0 = v0 ? exp(s0 - m) : 0.f, e1 = v1 ? exp(s1 - m) : 0.f;
        const float l = simd_sum(e0 + e1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        ps[sg][i0] = e0; ps[sg][i1] = e1;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float acc[NO];
        for (int j = 0; j < NO; ++j) acc[j] = 0.f;
        for (int c8 = c0; c8 < c1; c8 += 8) {
            half vv[8][NO];
            for (int u = 0; u < 8; ++u) {
                device const half * vr = vb + (size_t) (min(c8 + u, c1 - 1) - b) * a.kvstride;
                for (int j = 0; j < NO; ++j) vv[u][j] = vr[ln + 32 * j];
            }
            for (int u = 0; u < 8; ++u) {
                const float pc = c8 + u < c1 ? ps[sg][c8 + u - b] : 0.f;
                for (int j = 0; j < NO; ++j) acc[j] = fma(float(vv[u][j]), pc, acc[j]);
            }
        }
        if (sp == 1) {
            for (int j = 0; j < NO; ++j) o[ln + 32 * j] = acc[j];
            if (ln == 0) { o[D] = m; o[D + 1] = l; }
            return;
        }
        const float mn = max(M, m), f0 = exp(M - mn), f1 = exp(m - mn);
        for (int j = 0; j < NO; ++j) A[j] = A[j] * f0 + acc[j] * f1;
        L = L * f0 + l * f1;
        M = mn;
        simdgroup_barrier(mem_flags::mem_threadgroup);  // the next chunk rewrites ps
    }
    for (int j = 0; j < NO; ++j) o[ln + 32 * j] = A[j];
    if (ln == 0) { o[D] = M; o[D + 1] = L; }
}
#define FA2(D, GH) template [[host_name("fa2_" #D "_g" #GH)]] kernel void fa2<D, GH, 1>(constant fa1_args &, device const float *, device const half *, \
    device const half *, device const int *, device const int *, device float *, device const int *, uint3, ushort, ushort); \
    template [[host_name("fa2r_" #D "_g" #GH)]] kernel void fa2<D, GH, (32 / GH < 8 ? 32 / GH : 8)>(constant fa1_args &, device const float *, \
    device const half *, device const half *, device const int *, device const int *, device float *, device const int *, uint3, ushort, ushort);
FA2(128, 2) FA2(128, 4) FA2(128, 7) FA2(128, 8) FA2(96, 1)  // Gemma 4 keeps fa1: fa2 with 8 heads of 256 / 512 was 23% slower
FA2(64, 8)  // gpt-oss

#define FA1(D) template [[host_name("fa1_" #D)]] kernel void fa1<D>(constant fa1_args &, device const float *, device const half *, \
    device const half *, device const int *, device const int *, device float *, device const int *, uint3, ushort, ushort, ushort);
FA1(128) FA1(256) FA1(512)

// prompt attention for Gemma: pre2's design with G heads per K/V head (a tile = 32 / G tokens x G heads, one K/V head),
// head size D, the sliding window (cells older than pos - window + 1 hidden)
struct g4p_args { int n_head; int n_kv; int ctx; float scale; int ldq; int ldo; int window; int sinks; };  // sinks: gpt-oss's (g4_pre)
template <int D, int G>
kernel void g4_pre_q(constant g4p_args & a [[buffer(0)]], device const float * q [[buffer(1)]], device const int4 * blk [[buffer(2)]],
                     device half * qh [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    constexpr int TT = 32 / G;
    const int d4 = gid.x, rl = gid.y, b = gid.z / a.n_kv, kvh = gid.z % a.n_kv;
    if (d4 >= D / 4) return;
    const int4 bl = blk[b];
    const int g = rl / TT, tl = rl % TT;
    float4 x = 0.f;
    if (tl < bl.y && g < G) x = *((device const float4 *) (q + (size_t) (bl.x + tl) * a.ldq + (kvh * G + g) * D) + d4);  // G 6: rows 30, 31 unused
    *((device half4 *) (qh + ((size_t) gid.z * 32 + rl) * D) + d4) = (half4) (x * a.scale);
}
template <int D, int G>
kernel void g4_pre(constant g4p_args & a [[buffer(0)]], device const half * qh [[buffer(1)]], device const half * kc [[buffer(2)]],
                   device const half * vc [[buffer(3)]], device const int4 * blk [[buffer(4)]], device const int * pos [[buffer(5)]],
                   device float * o [[buffer(6)]], device const int * pt [[buffer(7)]], device const float * sk [[buffer(8)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int RT = 32, C = 64, NSG = 4, NW = 32, NQ = RT / NSG, NC = C / NW, TT = 32 / G;
    threadgroup float ss[RT * C];
    threadgroup half sp[RT * C];
    threadgroup float sa[RT];
    const int b = tg.x, kvh = tg.y, kvs = a.n_kv * D;
    const int4 bl = blk[b];
    const int row0 = bl.x, nr = bl.y, s = bl.z;
    const int pmax = pos[row0 + nr - 1], pmin = pos[row0];
    const int lo_all = a.window > 0 ? max(0, pmin - a.window + 1) : 0;
    constexpr auto dqk = matmul2d_descriptor(RT, C, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto dpv = matmul2d_descriptor(RT, D, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dqk, execution_simdgroups<NSG>> mqk;
    matmul2d<dpv, execution_simdgroups<NSG>> mpv;
    device const int * ps = pt + (size_t) s * (a.ctx / C);  // block cb (a multiple of C = 64) is page ps[cb / 64]
    device half * kh = (device half *) kc + (size_t) ps[0] * C * kvs + kvh * D;
    device half * vh = (device half *) vc + (size_t) ps[0] * C * kvs + kvh * D;
    auto tQ = tensor((device half *) qh + ((size_t) b * a.n_kv + kvh) * RT * D, dextents<int32_t, 2>(D, RT), array<int, 2>({1, D}));
    auto tS = tensor(ss, dextents<int32_t, 2>(C, RT), array<int, 2>({1, C}));
    auto tP = tensor(sp, dextents<int32_t, 2>(C, RT), array<int, 2>({1, C}));
    auto tK0 = tensor(kh, dextents<int32_t, 2>(D, C), array<int, 2>({1, kvs}));
    auto tV0 = tensor(vh, dextents<int32_t, 2>(D, C), array<int, 2>({1, kvs}));
    auto cS = mqk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK0), float>();
    auto cO = mpv.template get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (cO.is_valid_element(i)) cO[i] = 0.0f;
    }
    float M[NQ], S[NQ];
    int rp[NQ], rlo[NQ];
    #pragma unroll
    for (int jj = 0; jj < NQ; ++jj) {
        const int j = jj * NSG + sgitg, tl = j % TT;
        rp[jj] = tl < nr ? pos[row0 + tl] : -1;
        rlo[jj] = a.window > 0 ? max(0, rp[jj] - a.window + 1) : 0;
        M[jj] = -FLT_MAX / 2;
        S[jj] = 0.f;
    }
    for (int cb = lo_all / C * C; cb <= pmax; cb += C) {
        const bool full = cb + C - 1 <= pmin && (a.window <= 0 || cb >= pmax - a.window + 1);
        auto tK = tensor((device half *) kc + (size_t) ps[cb / C] * C * kvs + kvh * D, dextents<int32_t, 2>(D, C), array<int, 2>({1, kvs}));
        mqk.run(tQ, tK, cS);
        cS.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int jj = 0; jj < NQ; ++jj) {
            const int j = jj * NSG + sgitg;
            float sv[NC];
            #pragma unroll
            for (int ii = 0; ii < NC; ++ii) {
                const int c = cb + ii * NW + tiisg;
                sv[ii] = (full || (c <= rp[jj] && c >= rlo[jj])) ? ss[j * C + ii * NW + tiisg] : -INFINITY;
            }
            float m = M[jj];
            #pragma unroll
            for (int ii = 0; ii < NC; ++ii) m = max(m, sv[ii]);
            m = simd_max(m);
            float ms = 1.0f;
            if (m > M[jj] + 8.0f) { ms = exp(M[jj] - m); M[jj] = m; }
            float sum = 0.f;
            #pragma unroll
            for (int ii = 0; ii < NC; ++ii) {
                const half pp = (half) exp(sv[ii] - M[jj]);
                sp[j * C + ii * NW + tiisg] = pp;
                sum += (float) pp;
            }
            S[jj] = S[jj] * ms + simd_sum(sum);
            if (tiisg == 0) sa[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (cO.is_valid_element(i)) { const auto ix = cO.get_multidimensional_index(i); cO[i] *= sa[ix[1]]; }
        }
        auto tV = tensor((device half *) vc + (size_t) ps[cb / C] * C * kvs + kvh * D, dextents<int32_t, 2>(D, C), array<int, 2>({1, kvs}));
        mpv.run(tP, tV, cO);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (int jj = 0; jj < NQ; ++jj) {
        const int j = jj * NSG + sgitg;
        if (a.sinks && j / TT < G) S[jj] += exp(sk[kvh * G + j / TT] - M[jj]);  // the sink: a logit with no value
        if (tiisg == 0) sa[j] = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (cO.is_valid_element(i)) {
            const auto ix = cO.get_multidimensional_index(i);
            const int rr = ix[1], d = ix[0], tl = rr % TT, g = rr / TT;
            if (tl < nr && g < G) o[(size_t) (row0 + tl) * a.ldo + (kvh * G + g) * D + d] = cO[i] * sa[rr];
        }
    }
}
#define G4PRE(D, G) template [[host_name("g4_pre_q_" #D "_g" #G)]] kernel void g4_pre_q<D, G>(constant g4p_args &, device const float *, \
    device const int4 *, device half *, uint3); \
    template [[host_name("g4_pre_" #D "_g" #G)]] kernel void g4_pre<D, G>(constant g4p_args &, device const half *, device const half *, \
    device const half *, device const int4 *, device const int *, device float *, device const int *, device const float *, uint3, ushort, ushort);
// Gemma 4 E2B 256 / 512 g8, 12B 256 g2 / 512 g16, 31B 256 g2 / 512 g8; Gemma 3 1B 256 g4, 4B 256 g2, 27B 128 g2
G4PRE(128, 2) G4PRE(256, 2) G4PRE(256, 4) G4PRE(256, 8) G4PRE(512, 8) G4PRE(512, 16) G4PRE(96, 1) G4PRE(256, 6)  // 96 x 1: Phi-3; 256 x 6: Qwen3.5 27B
G4PRE(64, 8)  // gpt-oss

// small f32 / bf16 matrices (Gemma 4's per-layer embedding gate and projections): one simdgroup per output column c and
// block of up to 32 rows; lane l takes K positions 4l + 128i (float4s), sums per row in that order, then simd_sum. A row's
// arithmetic does not depend on how many rows there are.
struct mvf_args { int K; int M; int R; int ldx; int ldy; };
template <int BF>
kernel void mvf(constant mvf_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * x [[buffer(2)]],
                device float * y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                ushort ln [[thread_index_in_simdgroup]]) {
    const int c = tg.x * 4 + sg, r0 = tg.y * 32;
    if (c >= a.M) return;
    const int nr = min(32, a.R - r0);
    float acc[32];
    for (int r = 0; r < 32; ++r) acc[r] = 0.f;
    for (int k = ln * 4; k < a.K; k += 128) {
        float4 w;
        if (BF) { const ushort4 b = *((device const ushort4 *) ((device const ushort *) W + (size_t) c * a.K + k));
                  w = float4(as_type<float>((uint) b.x << 16), as_type<float>((uint) b.y << 16), as_type<float>((uint) b.z << 16), as_type<float>((uint) b.w << 16)); }
        else w = *((device const float4 *) ((device const float *) W + (size_t) c * a.K + k));
        for (int r = 0; r < nr; ++r) acc[r] += dot(w, *((device const float4 *) (x + (size_t) (r0 + r) * a.ldx + k)));
    }
    for (int r = 0; r < nr; ++r) {
        const float v = simd_sum(acc[r]);
        if (ln == 0) y[(size_t) (r0 + r) * a.ldy + c] = v;
    }
}
template [[host_name("mvf_f32")]] kernel void mvf<0>(constant mvf_args &, device const char *, device const float *, device float *, uint2, ushort, ushort);
// f32 matrices with few outputs (an MoE router, Qwen3.5's alpha / beta): one output column a threadgroup, its 4 simdgroups
// K slices of K / 4 (lanes float4s 4l + 128i in the slice), the slices' sums added in order. Fixed by the shape: a row's
// arithmetic does not depend on how many rows there are.
kernel void mvf_ks(constant mvf_args & a [[buffer(0)]], device const float * W [[buffer(1)]], device const float * x [[buffer(2)]],
                   device float * y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float part[4][32];
    const int c = tg.x, r0 = tg.y * 32, nr = min(32, a.R - r0), ks = a.K / 4, k0 = sg * ks;
    float acc[32];
    for (int r = 0; r < 32; ++r) acc[r] = 0.f;
    device const float * wr = W + (size_t) c * a.K;
    for (int k = k0 + ln * 4; k < k0 + ks; k += 128) {
        const float4 w = *((device const float4 *) (wr + k));
        for (int r = 0; r < nr; ++r) acc[r] += dot(w, *((device const float4 *) (x + (size_t) (r0 + r) * a.ldx + k)));
    }
    for (int r = 0; r < nr; ++r) {
        const float v = simd_sum(acc[r]);
        if (ln == 0) part[sg][r] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && ln < nr) y[(size_t) (r0 + ln) * a.ldy + c] = (part[0][ln] + part[1][ln]) + (part[2][ln] + part[3][ln]);
}
// mvf_ks for one row: the same lanes, slices and sums as mvf_ks (bit for bit) without its 32-row accumulator array
kernel void mvf_ks1(constant mvf_args & a [[buffer(0)]], device const float * W [[buffer(1)]], device const float * x [[buffer(2)]],
                    device float * y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float part[4];
    const int c = tg.x, r0 = tg.y, ks = a.K / 4, k0 = sg * ks;
    float acc = 0.f;
    device const float * wr = W + (size_t) c * a.K;
    device const float * xr = x + (size_t) r0 * a.ldx;
    for (int k = k0 + ln * 4; k < k0 + ks; k += 128) acc += dot(*((device const float4 *) (wr + k)), *((device const float4 *) (xr + k)));
    const float v = simd_sum(acc);
    if (ln == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && ln == 0) y[(size_t) r0 * a.ldy + c] = (part[0] + part[1]) + (part[2] + part[3]);
}
// two mvf_ks1 in one dispatch (one row, two matrices over the same x): threadgroups < a.M are matrix 0's outputs (c = tg.x,
// into y0), the rest matrix 1's (c = tg.x - a.M, into y1); each output the exact mvf_ks1 body (4 simdgroups' K slices, (p0+p1)+(p2+p3))
kernel void mvf_ks1x2(constant mvf_args & a [[buffer(0)]], device const float * W0 [[buffer(1)]], device const float * x [[buffer(2)]],
                      device float * y0 [[buffer(3)]], device const float * W1 [[buffer(4)]], device float * y1 [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float part[4];
    const bool second = (int) tg.x >= a.M;
    const int c = second ? (int) tg.x - a.M : (int) tg.x, r0 = tg.y, ks = a.K / 4, k0 = sg * ks;
    float acc = 0.f;
    device const float * wr = (second ? W1 : W0) + (size_t) c * a.K;
    device const float * xr = x + (size_t) r0 * a.ldx;
    for (int k = k0 + ln * 4; k < k0 + ks; k += 128) acc += dot(*((device const float4 *) (wr + k)), *((device const float4 *) (xr + k)));
    const float v = simd_sum(acc);
    if (ln == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && ln == 0) (second ? y1 : y0)[(size_t) r0 * a.ldy + c] = (part[0] + part[1]) + (part[2] + part[3]);
}
template [[host_name("mvf_bf16")]] kernel void mvf<1>(constant mvf_args &, device const char *, device const float *, device float *, uint2, ushort, ushort);

// batch rows of the small f32 / bf16 matrices: the simdgroup-MMA tile of mm<FMT> over K slice z of S (fixed by the shape),
// its partial to part[z]; mm_sk_sum adds the S partials in slice order
template <int FMT>
kernel void mm_sk(constant mm_args & a [[buffer(0)]], device const float * x [[buffer(1)]], device const uchar * w [[buffer(2)]],
                  device float * part [[buffer(3)]], constant int & S [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                  ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ws[32 * 32];
    threadgroup float xs[32 * 32];
    const int c0 = tg.x * 32, r0 = tg.y * 32, z = tg.z;
    const int nt = min(4, (a.R - r0 + 7) / 8);
    const int nkb = a.K / 32, kb0 = z * nkb / S, kb1 = (z + 1) * nkb / S;
    simdgroup_float8x8 acc[4];
    for (int i = 0; i < 4; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    const int wc = t / 4, wp = t % 4;
    for (int kb = kb0; kb < kb1; ++kb) {
        mm_deq<FMT>(w, (device const float *) w, a.K, c0 + wc, kb, wp, ws + wc * 32);
        for (int i = t; i < 32 * 32; i += 128) {
            const int r = i / 32, k = i % 32;
            xs[i] = r0 + r < a.R ? x[(size_t) (r0 + r) * a.ldx + kb * 32 + k] : 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_float8x8 wb;
            simdgroup_load(wb, ws + sg * 8 * 32 + kk * 8, 32, ulong2(0, 0), true);
            for (int i = 0; i < nt; ++i) {
                simdgroup_float8x8 xa;
                simdgroup_load(xa, xs + i * 8 * 32 + kk * 8, 32);
                simdgroup_multiply_accumulate(acc[i], xa, wb, acc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < nt; ++i) simdgroup_store(acc[i], xs + i * 8 * 32 + sg * 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < 32 * 32; i += 128) {
        const int r = i / 32, c = i % 32;
        if (r0 + r < a.R && r < nt * 8) part[((size_t) z * a.R + r0 + r) * a.M + c0 + c] = xs[i];
    }
}
template [[host_name("mm_sk_f32")]] kernel void mm_sk<3>(constant mm_args &, device const float *, device const uchar *, device float *, constant int &, uint3, ushort, ushort);
template [[host_name("mm_sk_bf16")]] kernel void mm_sk<4>(constant mm_args &, device const float *, device const uchar *, device float *, constant int &, uint3, ushort, ushort);
kernel void mm_sk_sum(constant mm_args & a [[buffer(0)]], device const float * part [[buffer(1)]], device float * y [[buffer(2)]],
                      constant int & S [[buffer(3)]], uint2 i [[thread_position_in_grid]]) {
    if ((int) i.x >= a.M || (int) i.y >= a.R) return;
    float s = 0.f;
    for (int z = 0; z < S; ++z) s += part[((size_t) z * a.R + i.y) * a.M + i.x];
    y[(size_t) i.y * a.ldy + i.x] = s;
}

// batch rows of the small f32 / bf16 matrices on the matrix units: qmm_ks's tiling (32 weight rows x NT tokens, G K slices,
// partials summed in order) with the weights read straight from the file as a device tensor (relaxed precision: the
// mantissa is cut, the f32 exponent kept, so tiny weights do not underflow as they would in half)
template <typename WT, int G, int NT>
kernel void tmm(constant qmm_args & a [[buffer(0)]], device const char * W [[buffer(1)]], device const float * X [[buffer(2)]],
                device float * D [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int SN = 32, NTH = 32 * G;
    const int ks = (int) sg;
    const int n0 = tg.x * NT, r0 = tg.y * SN;
    const int nrem = clamp(a.N - n0, 0, NT);
    constexpr auto desc = matmul2d_descriptor(SN, NT, 32, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> mm;
    device const float * xb = X + (uint64_t) n0 * a.ldx;
    device WT * wb = (device WT *) W + (uint64_t) r0 * a.K;
    auto tX0 = tensor((device float *) xb, dextents<int32_t, 2>(32, NT), array<int, 2>({1, a.ldx}));
    auto tW0 = tensor(wb, dextents<int32_t, 2>(32, SN), array<int, 2>({1, a.K}));
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tW0), decltype(tX0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }
    if (nrem > 0) {
        for (int k0 = 32 * ks; k0 < a.K; k0 += 32 * G) {
            auto tW = tensor(wb + k0, dextents<int32_t, 2>(32, SN), array<int, 2>({1, a.K}));
            auto tX = tensor((device float *) xb + k0, dextents<int32_t, 2>(32, nrem), array<int, 2>({1, a.ldx}));
            mm.run(tW, tX, cT);
        }
    }
    threadgroup float red[4 * 1024];
    if (G == 8) {
        if (ks >= 4) {
            threadgroup float * p = red + (ks - 4) * 1024;
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) { const auto ix = cT.get_multidimensional_index(i); p[ix[0] * 32 + ix[1]] = cT[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks < 4) {
            threadgroup const float * p = red + ks * 1024;
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) { const auto ix = cT.get_multidimensional_index(i); cT[i] += p[ix[0] * 32 + ix[1]]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (ks < 4) {
        threadgroup float * p = red + ks * 1024;
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) { const auto ix = cT.get_multidimensional_index(i); p[ix[0] * 32 + ix[1]] = cT[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < 1024; e += NTH) {
        const int n = e / 32, m = e % 32;
        const float s = ((red[e] + red[1024 + e]) + red[2048 + e]) + red[3072 + e];
        if (n < nrem && r0 + m < a.M) D[(uint64_t) (n0 + n) * a.ldd + r0 + m] = s;
    }
}
#define TMM(WT, NAME, G, NT) template [[host_name("tmm_" #NAME "_g" #G "_nt" #NT)]] kernel void tmm<WT, G, NT>(constant qmm_args &, \
    device const char *, device const float *, device float *, uint3, ushort, ushort);
TMM(float, f32, 4, 16) TMM(float, f32, 4, 32) TMM(float, f32, 8, 16) TMM(float, f32, 8, 32)
TMM(bfloat, bf16, 4, 16) TMM(bfloat, bf16, 4, 32) TMM(bfloat, bf16, 8, 16) TMM(bfloat, bf16, 8, 32)

// ==== Qwen3.5 / 3.6 (qwen35): Gated DeltaNet layers between full attention layers ==================================
// A step's rows of one sequence come as a run of consecutive positions (runs: first row, rows, sequence); rrun[r] is a
// row's place in its run. A linear layer: qkv (q, k: nk heads of 128, v: nv heads of 128, C channels in all) through a
// 4-tap causal convolution (the sequence's last 3 inputs in its conv state) and SiLU, q and k L2-normed per head; the
// delta rule per v head over a 128 x 128 state (columns independent); the output RMS-normed per head, times silu(z).
struct q35_args { int C, nk, nv, ld_qkv, ld_ab, nlin; float eps; int lin; };  // lin: the linear layer's index; states [entry][nlin]
static inline float q35_silu(float x) { return x / (1.0f + exp(-x)); }
static inline float q35_softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }  // ggml's
kernel void q35_conv(constant q35_args & a [[buffer(0)]], device const float * qkv [[buffer(1)]], device const float * w [[buffer(2)]],
                     device const float * cs [[buffer(3)]], device const int * seq [[buffer(4)]], device const int * rrun [[buffer(5)]],
                     device float * co [[buffer(6)]], device const float * ab [[buffer(7)]], device const float * dt [[buffer(8)]],
                     device const float * sa [[buffer(9)]], device float * bg [[buffer(10)]], uint r [[threadgroup_position_in_grid]],
                     ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const int run = rrun[r];
    device const float * st = cs + ((size_t) seq[r] * a.nlin + a.lin) * 3 * a.C;  // [3][C]: the inputs before the run (seq: the row's state entry)
    for (int c = t; c < a.C; c += 256) {
        float acc = 0.f;
        for (int i = 0; i < 4; ++i) {
            const int back = 3 - i;
            const float x = back <= run ? qkv[(size_t) (r - back) * a.ld_qkv + c] : st[(3 - back + run) * a.C + c];
            acc += w[c * 4 + i] * x;
        }
        co[(size_t) r * a.C + c] = q35_silu(acc);
    }
    if (t < a.nv) {  // beta = sigmoid(b), g = softplus(alpha + dt) * a  (ab: alpha then beta, nv each)
        const float al = ab[(size_t) r * a.ld_ab + t], be = ab[(size_t) r * a.ld_ab + a.nv + t];
        bg[((size_t) r * a.nv + t) * 2] = 1.0f / (1.0f + exp(-be));
        bg[((size_t) r * a.nv + t) * 2 + 1] = q35_softplus(al + dt[t]) * sa[t];
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (int h = sg; h < 2 * a.nk; h += 8) {  // q and k heads: x / sqrt(sum x^2 + eps) (llama.cpp's gdn l2 norm)
        device float * x = co + (size_t) r * a.C + h * 128;
        float v[4], ss = 0.f;
        for (int u = 0; u < 4; ++u) { v[u] = x[ln + 32 * u]; ss += v[u] * v[u]; }
        const float sc = 1.0f / sqrt(simd_sum(ss) + a.eps);
        for (int u = 0; u < 4; ++u) x[ln + 32 * u] = v[u] * sc;
    }
}
// q35_conv over (row, 8 heads of 128 channels) threadgroups (one threadgroup a row left one GPU core doing the whole layer):
// a simdgroup per head, lane l channels l + 32 u; the same sums as q35_conv bit for bit (each channel's conv, the norm's
// squares in u order and its simd_sum over the same lanes)
kernel void q35_conv8(constant q35_args & a [[buffer(0)]], device const float * qkv [[buffer(1)]], device const float * w [[buffer(2)]],
                      device const float * cs [[buffer(3)]], device const int * seq [[buffer(4)]], device const int * rrun [[buffer(5)]],
                      device float * co [[buffer(6)]], device const float * ab [[buffer(7)]], device const float * dt [[buffer(8)]],
                      device const float * sa [[buffer(9)]], device float * bg [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                      ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const int r = tg.x, run = rrun[r], h = tg.y * 8 + sg;
    if (tg.y == 0 && t < a.nv) {
        const float al = ab[(size_t) r * a.ld_ab + t], be = ab[(size_t) r * a.ld_ab + a.nv + t];
        bg[((size_t) r * a.nv + t) * 2] = 1.0f / (1.0f + exp(-be));
        bg[((size_t) r * a.nv + t) * 2 + 1] = q35_softplus(al + dt[t]) * sa[t];
    }
    if (h * 128 >= a.C) return;
    device const float * st = cs + ((size_t) seq[r] * a.nlin + a.lin) * 3 * a.C;
    float v[4];
    for (int u = 0; u < 4; ++u) {
        const int c = h * 128 + ln + 32 * u;
        float acc = 0.f;
        for (int i = 0; i < 4; ++i) {
            const int back = 3 - i;
            const float x = back <= run ? qkv[(size_t) (r - back) * a.ld_qkv + c] : st[(3 - back + run) * a.C + c];
            acc += w[c * 4 + i] * x;
        }
        v[u] = q35_silu(acc);
    }
    device float * x = co + (size_t) r * a.C + h * 128;
    if (h < 2 * a.nk) {
        float ss = 0.f;
        for (int u = 0; u < 4; ++u) ss += v[u] * v[u];
        const float sc = 1.0f / sqrt(simd_sum(ss) + a.eps);
        for (int u = 0; u < 4; ++u) v[u] *= sc;
    }
    for (int u = 0; u < 4; ++u) x[ln + 32 * u] = v[u];
}
// q35_conv8 with q35_conv_save folded in, for steps where every run is one row (nruns == R: run = 0, n = 1, last = r, the
// state entry rsi[r] = runs[r].z). A channel c belongs to exactly one thread (h * 128 + ln + 32 u of row r), which reads all
// its old taps in the conv loop and only then writes the shifted state exactly as q35_conv_save does for a one-row run
// (ascending j: st[0] = old st[1], st[1] = old st[2], st[2] = qkv[r][c]): pure copies, no other thread touches the channel
kernel void q35_conv8s(constant q35_args & a [[buffer(0)]], device const float * qkv [[buffer(1)]], device const float * w [[buffer(2)]],
                       device float * cs [[buffer(3)]], device const int * seq [[buffer(4)]], device const int * rrun [[buffer(5)]],
                       device float * co [[buffer(6)]], device const float * ab [[buffer(7)]], device const float * dt [[buffer(8)]],
                       device const float * sa [[buffer(9)]], device float * bg [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                       ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const int r = tg.x, run = rrun[r], h = tg.y * 8 + sg;
    if (tg.y == 0 && t < a.nv) {
        const float al = ab[(size_t) r * a.ld_ab + t], be = ab[(size_t) r * a.ld_ab + a.nv + t];
        bg[((size_t) r * a.nv + t) * 2] = 1.0f / (1.0f + exp(-be));
        bg[((size_t) r * a.nv + t) * 2 + 1] = q35_softplus(al + dt[t]) * sa[t];
    }
    if (h * 128 >= a.C) return;
    device float * st = cs + ((size_t) seq[r] * a.nlin + a.lin) * 3 * a.C;
    float v[4];
    for (int u = 0; u < 4; ++u) {
        const int c = h * 128 + ln + 32 * u;
        float acc = 0.f;
        for (int i = 0; i < 4; ++i) {
            const int back = 3 - i;
            const float x = back <= run ? qkv[(size_t) (r - back) * a.ld_qkv + c] : st[(3 - back + run) * a.C + c];
            acc += w[c * 4 + i] * x;
        }
        v[u] = q35_silu(acc);
        // the old taps are all read (acc is done); this thread alone owns channel c of this state entry
        st[c] = st[a.C + c];
        st[a.C + c] = st[2 * a.C + c];
        st[2 * a.C + c] = qkv[(size_t) r * a.ld_qkv + c];
    }
    device float * x = co + (size_t) r * a.C + h * 128;
    if (h < 2 * a.nk) {
        float ss = 0.f;
        for (int u = 0; u < 4; ++u) ss += v[u] * v[u];
        const float sc = 1.0f / sqrt(simd_sum(ss) + a.eps);
        for (int u = 0; u < 4; ++u) v[u] *= sc;
    }
    for (int u = 0; u < 4; ++u) x[ln + 32 * u] = v[u];
}
// the conv state after the step: each run's last 3 inputs (from its rows, or the old state when the run is shorter)
kernel void q35_conv_save(constant q35_args & a [[buffer(0)]], device const float * qkv [[buffer(1)]], device float * cs [[buffer(2)]],
                          device const int4 * runs [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    const int c = g.x;
    if (c >= a.C) return;
    const int4 rn = runs[g.y];
    const int last = rn.x + rn.y - 1, n = rn.y;
    device float * st = cs + ((size_t) rn.z * a.nlin + a.lin) * 3 * a.C;
    for (int j = 0; j < 3; ++j) {  // ascending: new j reads old j + n (> j), not yet written
        const int back = 2 - j;
        st[j * a.C + c] = back < n ? qkv[(size_t) (last - back) * a.ld_qkv + c] : st[(j + n) * a.C + c];
    }
}
// the delta rule: a threadgroup per (run, v head, 16 columns), a simdgroup per 4 columns, lane l the rows l + 32 u; the
// state [nv][column][row] per (layer, sequence) read once, the run's tokens in order, written back
kernel void q35_delta(constant q35_args & a [[buffer(0)]], device const float * co [[buffer(1)]], device const float * bg [[buffer(2)]],
                      device float * ss [[buffer(3)]], device const int4 * runs [[buffer(4)]], device float * out [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const int4 rn = runs[tg.x];
    const int h = tg.y, kh = h % a.nk, j0 = tg.z * 16 + sg * 4;
    device float * S = ss + (((size_t) rn.z * a.nlin + a.lin) * a.nv + h) * 128 * 128;
    float st[4][4];  // [column][u]
    for (int c = 0; c < 4; ++c) for (int u = 0; u < 4; ++u) st[c][u] = S[(j0 + c) * 128 + ln + 32 * u];
    const float scale = 1.0f / sqrt(128.0f);
    for (int r = rn.x; r < rn.x + rn.y; ++r) {
        device const float * row = co + (size_t) r * a.C;
        float q[4], k[4];
        for (int u = 0; u < 4; ++u) { q[u] = row[kh * 128 + ln + 32 * u]; k[u] = row[(a.nk + kh) * 128 + ln + 32 * u]; }
        const float beta = bg[((size_t) r * a.nv + h) * 2], eg = exp(bg[((size_t) r * a.nv + h) * 2 + 1]);
        for (int c = 0; c < 4; ++c) {
            const float vj = row[2 * a.nk * 128 + h * 128 + j0 + c];
            float kv = 0.f;
            for (int u = 0; u < 4; ++u) { st[c][u] *= eg; kv += st[c][u] * k[u]; }
            const float delta = (vj - simd_sum(kv)) * beta;
            float o = 0.f;
            for (int u = 0; u < 4; ++u) { st[c][u] += k[u] * delta; o += st[c][u] * q[u]; }
            o = simd_sum(o);
            if (ln == 0) out[(size_t) r * a.nv * 128 + h * 128 + j0 + c] = o * scale;
        }
    }
    for (int c = 0; c < 4; ++c) for (int u = 0; u < 4; ++u) S[(j0 + c) * 128 + ln + 32 * u] = st[c][u];
}
// q35_delta for a step with draft rows: snap[r] >= 0 is the state entry that takes the state after row r (the same sums as
// q35_delta; the run's own entry takes the state after its last row as always)
kernel void q35_delta_s(constant q35_args & a [[buffer(0)]], device const float * co [[buffer(1)]], device const float * bg [[buffer(2)]],
                        device float * ss [[buffer(3)]], device const int4 * runs [[buffer(4)]], device float * out [[buffer(5)]],
                        device const int * snap [[buffer(6)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const int4 rn = runs[tg.x];
    const int h = tg.y, kh = h % a.nk, j0 = tg.z * 16 + sg * 4;
    device float * S = ss + (((size_t) rn.z * a.nlin + a.lin) * a.nv + h) * 128 * 128;
    float st[4][4];  // [column][u]
    for (int c = 0; c < 4; ++c) for (int u = 0; u < 4; ++u) st[c][u] = S[(j0 + c) * 128 + ln + 32 * u];
    const float scale = 1.0f / sqrt(128.0f);
    for (int r = rn.x; r < rn.x + rn.y; ++r) {
        device const float * row = co + (size_t) r * a.C;
        float q[4], k[4];
        for (int u = 0; u < 4; ++u) { q[u] = row[kh * 128 + ln + 32 * u]; k[u] = row[(a.nk + kh) * 128 + ln + 32 * u]; }
        const float beta = bg[((size_t) r * a.nv + h) * 2], eg = exp(bg[((size_t) r * a.nv + h) * 2 + 1]);
        for (int c = 0; c < 4; ++c) {
            const float vj = row[2 * a.nk * 128 + h * 128 + j0 + c];
            float kv = 0.f;
            for (int u = 0; u < 4; ++u) { st[c][u] *= eg; kv += st[c][u] * k[u]; }
            const float delta = (vj - simd_sum(kv)) * beta;
            float o = 0.f;
            for (int u = 0; u < 4; ++u) { st[c][u] += k[u] * delta; o += st[c][u] * q[u]; }
            o = simd_sum(o);
            if (ln == 0) out[(size_t) r * a.nv * 128 + h * 128 + j0 + c] = o * scale;
        }
        const int sn = snap[r];
        if (sn >= 0) {
            device float * D = ss + (((size_t) sn * a.nlin + a.lin) * a.nv + h) * 128 * 128;
            for (int c = 0; c < 4; ++c) for (int u = 0; u < 4; ++u) D[(j0 + c) * 128 + ln + 32 * u] = st[c][u];
        }
    }
    for (int c = 0; c < 4; ++c) for (int u = 0; u < 4; ++u) S[(j0 + c) * 128 + ln + 32 * u] = st[c][u];
}
// q35_conv_save for the rows a draft step may be cut back to: an item is (the row, the rows of its run up to it, the run's
// state entry, the entry that takes the conv state after that row). It reads the run's state as it was before the step, so
// it is dispatched before q35_conv_save.
kernel void q35_conv_snap(constant q35_args & a [[buffer(0)]], device const float * qkv [[buffer(1)]], device float * cs [[buffer(2)]],
                          device const int4 * items [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    const int c = g.x;
    if (c >= a.C) return;
    const int4 it = items[g.y];
    const int last = it.x, n = it.y;
    device const float * src = cs + ((size_t) it.z * a.nlin + a.lin) * 3 * a.C;
    device float * dst = cs + ((size_t) it.w * a.nlin + a.lin) * 3 * a.C;
    for (int j = 0; j < 3; ++j) {
        const int back = 2 - j;
        dst[j * a.C + c] = back < n ? qkv[(size_t) (last - back) * a.ld_qkv + c] : src[(j + n) * a.C + c];
    }
}
// the gated norm: per v head, rmsnorm(out) * w * silu(z)
kernel void q35_out(constant q35_args & a [[buffer(0)]], device const float * o [[buffer(1)]], device const float * z [[buffer(2)]],
                    device const float * w [[buffer(3)]], device float * y [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                    ushort ln [[thread_index_in_simdgroup]]) {
    const int r = tg.x, h = tg.y;
    device const float * x = o + (size_t) r * a.nv * 128 + h * 128;
    device const float * zr = z + (size_t) r * a.nv * 128 + h * 128;
    float v[4], ss = 0.f;
    for (int u = 0; u < 4; ++u) { v[u] = x[ln + 32 * u]; ss += v[u] * v[u]; }
    const float sc = 1.0f / sqrt(simd_sum(ss) / 128.0f + a.eps);
    for (int u = 0; u < 4; ++u) {
        const int d = ln + 32 * u;
        y[(size_t) r * a.nv * 128 + h * 128 + d] = ((v[u] * sc) * w[d]) * q35_silu(zr[d]);
    }
}
// a full attention layer's q (from q / gate interleaved per head: ldqf apart, head h at 2 D h) and k: per-head RMS norm
// with their weights, NEOX rope over the first rot dims (pairs i, i + rot / 2), k and v into the cache; D = 256
struct q35q_args { int n_head, n_kv, ctx, ldqf, ldq, ldk; float eps, base; int rot; };
kernel void q35_qk(constant q35q_args & a [[buffer(0)]], device const float * qf [[buffer(1)]], device float * q [[buffer(2)]],
                   device const float * k [[buffer(3)]], device const float * v [[buffer(4)]], device const float * qw [[buffer(5)]],
                   device const float * kw [[buffer(6)]], device half * kc [[buffer(7)]], device half * vc [[buffer(8)]],
                   device const int * seq [[buffer(9)]], device const int * pos [[buffer(10)]], device const int * pt [[buffer(11)]],
                   uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort ln [[thread_index_in_simdgroup]]) {
    constexpr int D = 256, NU = D / 32;
    const int p = pos[r];
    const size_t cell = ((size_t) pt[(size_t) seq[r] * (a.ctx / 64) + p / 64] * 64 + p % 64) * a.n_kv * D;
    const int half_rot = a.rot / 2;  // lane ln < half_rot: pair (ln, ln + half_rot) = elements u 0 and u half_rot / 32
    const float th = float(p) * pow(a.base, -float(2 * ln) / float(a.rot));
    const float cs = cos(th), sn = sin(th);
    const int u1 = half_rot / 32;
    for (int h = sg; h < a.n_head + a.n_kv; h += 8) {
        const bool isq = h < a.n_head;
        device const float * src = isq ? qf + (size_t) r * a.ldqf + h * 2 * D : k + (size_t) r * a.ldk + (h - a.n_head) * D;
        float x[NU], ss = 0.f;
        for (int u = 0; u < NU; ++u) { x[u] = src[ln + 32 * u]; ss += x[u] * x[u]; }
        const float sc = 1.0f / sqrt(simd_sum(ss) / float(D) + a.eps);
        device const float * w = isq ? qw : kw;
        for (int u = 0; u < NU; ++u) x[u] = (x[u] * sc) * w[ln + 32 * u];
        if (ln < half_rot) {
            const float x0 = x[0], x1 = x[u1];
            x[0] = x0 * cs - x1 * sn; x[u1] = x0 * sn + x1 * cs;
        }
        if (isq) for (int u = 0; u < NU; ++u) q[(size_t) r * a.ldq + h * D + ln + 32 * u] = x[u];
        else for (int u = 0; u < NU; ++u) kc[cell + (h - a.n_head) * D + ln + 32 * u] = half(x[u]);
    }
    for (int i = t; i < a.n_kv * D; i += 256) vc[cell + i] = half(v[(size_t) r * a.ldk + i]);
}
// the attention output times sigmoid(gate) (the gate: the second half of each head's q / gate pair)
kernel void q35_gate(constant q35q_args & a [[buffer(0)]], device const float * qf [[buffer(1)]], device float * o [[buffer(2)]],
                     uint2 g [[thread_position_in_grid]]) {
    const int i = g.x, r = g.y;
    if (i >= a.n_head * 256) return;
    const float gt = qf[(size_t) r * a.ldqf + (i / 256) * 512 + 256 + i % 256];
    o[(size_t) r * a.ldq + i] *= 1.0f / (1.0f + exp(-gt));
}

// ==== mixture of experts (Qwen3.5-MoE) ================================================================================
// moe_topk: a row's router logits -> softmax over the experts, the k_used most probable (ties: the lower index), their
// probabilities renormalized to sum 1 (llama.cpp's norm_w, the sum floored at 6.1e-5): ids, w per (row, slot). Mode 1
// (gpt-oss, llama.cpp's SOFTMAX_WEIGHT): logits plus the router's bias, the k_used largest, a softmax over those k
struct moe_args { int ne, k_used, r0, npairs, nt, ldl, mode; };  // mode 1: gpt-oss (router bias, softmax over the top k, expert biases)
kernel void moe_topk(constant moe_args & a [[buffer(0)]], device const float * lg [[buffer(1)]], device int * ids [[buffer(2)]],
                     device float * w [[buffer(3)]], device const float * bias [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                     ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float p[512];
    threadgroup float red[8];
    threadgroup int redi[8];
    threadgroup int pick[16];
    device const float * l = lg + (size_t) r * a.ldl;
    float tot = 1.f;
    if (a.mode == 1) {
        for (int i = t; i < a.ne; i += 256) p[i] = l[i] + bias[i];
    } else {
        float m = -INFINITY;
        for (int i = t; i < a.ne; i += 256) m = max(m, l[i]);
        m = simd_max(m);
        if (ln == 0) red[sg] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        m = red[0];
        for (int i = 1; i < 8; ++i) m = max(m, red[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0.f;
        for (int i = t; i < a.ne; i += 256) { p[i] = exp(l[i] - m); s += p[i]; }
        s = simd_sum(s);
        if (ln == 0) red[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tot = 0.f;
        for (int i = 0; i < 8; ++i) tot += red[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the k_used largest in turn, all threads: each its best (ties: the lower index), the simdgroups', then the threadgroup's
    for (int k = 0; k < a.k_used; ++k) {
        float bv = -INFINITY;
        int bi = 1 << 30;
        for (int i = t; i < a.ne; i += 256) if (p[i] > bv) { bv = p[i]; bi = i; }
        const float mv = simd_max(bv);
        const int mi = simd_min(bv == mv ? bi : 1 << 30);
        if (ln == 0) { red[sg] = mv; redi[sg] = mi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            float v = red[0];
            int i0 = redi[0];
            for (int j = 1; j < 8; ++j) if (red[j] > v || (red[j] == v && redi[j] < i0)) { v = red[j]; i0 = redi[j]; }
            pick[k] = i0;
            p[i0] = -INFINITY;
            w[r * a.k_used + k] = v;  // its value (thread 0 weighs them below)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) {
        float pk[16], sel = 0.f, mx = -INFINITY;
        for (int k = 0; k < a.k_used; ++k) { pk[k] = w[r * a.k_used + k]; mx = max(mx, pk[k]); }
        if (a.mode == 1) {
            for (int k = 0; k < a.k_used; ++k) { pk[k] = exp(pk[k] - mx); sel += pk[k]; }
            for (int k = 0; k < a.k_used; ++k) { ids[r * a.k_used + k] = pick[k]; w[r * a.k_used + k] = pk[k] / sel; }
        } else {
            for (int k = 0; k < a.k_used; ++k) { pk[k] /= tot; sel += pk[k]; }
            sel = max(sel, 6.103515625e-5f);
            for (int k = 0; k < a.k_used; ++k) { ids[r * a.k_used + k] = pick[k]; w[r * a.k_used + k] = pk[k] / sel; }
        }
    }
}
// moe_topk1: moe_topk for ne <= 256 in ONE simdgroup per row, with no threadgroup memory and no barriers (moe_topk's 21
// barriers held the whole GPU on one threadgroup: ~28 us a layer). Lane ln holds the experts ln + 32 g (g = 0 .. 7) in
// registers, as moe_topk's simdgroup g held them on its lanes: the same simd_sum per g, the same sum over g in order, the
// same choices (the largest, ties: the lower index) and the same weights, so the same bits
kernel void moe_topk1(constant moe_args & a [[buffer(0)]], device const float * lg [[buffer(1)]], device int * ids [[buffer(2)]],
                      device float * w [[buffer(3)]], device const float * bias [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                      ushort ln [[thread_index_in_threadgroup]]) {
    device const float * l = lg + (size_t) r * a.ldl;
    float p[8];
    float tot = 1.f;
    if (a.mode == 1) {
        for (int g = 0; g < 8; ++g) { const int i = ln + 32 * g; p[g] = i < a.ne ? l[i] + bias[i] : -INFINITY; }
    } else {
        float v[8];
        float m = -INFINITY;
        for (int g = 0; g < 8; ++g) { const int i = ln + 32 * g; v[g] = i < a.ne ? l[i] : -INFINITY; m = max(m, v[g]); }
        m = simd_max(m);
        tot = 0.f;
        for (int g = 0; g < 8; ++g) {
            const bool in = ln + 32 * g < a.ne;
            const float e = in ? exp(v[g] - m) : 0.f;
            p[g] = in ? e : -INFINITY;
            tot += simd_sum(e);
        }
    }
    float pkv[16];
    int pick[16];
    for (int k = 0; k < a.k_used; ++k) {
        float bv = -INFINITY;
        int bi = 1 << 30;
        for (int g = 0; g < 8; ++g) if (p[g] > bv) { bv = p[g]; bi = ln + 32 * g; }
        const float mv = simd_max(bv);
        const int mi = simd_min(bv == mv ? bi : 1 << 30);
        for (int g = 0; g < 8; ++g) if (ln + 32 * g == mi) p[g] = -INFINITY;
        pkv[k] = mv; pick[k] = mi;
    }
    if (ln == 0) {
        float pk[16], sel = 0.f, mx = -INFINITY;
        for (int k = 0; k < a.k_used; ++k) { pk[k] = pkv[k]; mx = max(mx, pk[k]); }
        if (a.mode == 1) {
            for (int k = 0; k < a.k_used; ++k) { pk[k] = exp(pk[k] - mx); sel += pk[k]; }
            for (int k = 0; k < a.k_used; ++k) { ids[r * a.k_used + k] = pick[k]; w[r * a.k_used + k] = pk[k] / sel; }
        } else {
            for (int k = 0; k < a.k_used; ++k) { pk[k] /= tot; sel += pk[k]; }
            sel = max(sel, 6.103515625e-5f);
            for (int k = 0; k < a.k_used; ++k) { ids[r * a.k_used + k] = pick[k]; w[r * a.k_used + k] = pk[k] / sel; }
        }
    }
}
// moe_lists (one threadgroup): the pairs of rows r0 .. in expert order (list), each pair's place in it (pos), and the
// grouped matmuls' tiles (expert, first place, rows <= nt) in tiles[0 ..), the rest (.., .., 0); a pair's place among its
// expert's comes from an atomic, so it may differ run to run, its result does not
kernel void moe_lists(constant moe_args & a [[buffer(0)]], device const int * ids [[buffer(1)]], device int * list [[buffer(2)]],
                      device int * pos [[buffer(3)]], device int4 * tiles [[buffer(4)]], ushort t [[thread_index_in_threadgroup]]) {
    threadgroup atomic_int cnt[512];
    threadgroup int off[512];
    for (int e = t; e < a.ne; e += 1024) atomic_store_explicit(&cnt[e], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int p0 = a.r0 * a.k_used;
    for (int q = t; q < a.npairs; q += 1024) pos[p0 + q] = atomic_fetch_add_explicit(&cnt[ids[p0 + q]], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (t == 0) {
        int o = 0, nt = 0;
        for (int e = 0; e < a.ne; ++e) {
            const int c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
            off[e] = o;
            for (int n = 0; n < c; n += a.nt) tiles[nt++] = int4(e, o + n, min(a.nt, c - n), 0);
            o += c;
        }
        const int bound = (a.npairs + a.nt - 1) / a.nt + a.ne;
        for (; nt < bound; ++nt) tiles[nt] = int4(0, 0, 0, 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int q = t; q < a.npairs; q += 1024) {
        const int at = off[ids[p0 + q]] + pos[p0 + q];
        pos[p0 + q] = at;
        list[at] = p0 + q;
    }
}
// gpt-oss's experts: (gate + bias_g[e]) and (up + bias_u[e]) through llama.cpp's swiglu_oai (gate clamped above at 7, up to
// +-7, gate sigmoid(1.702 gate) (up + 1)). Interactive pairs (list null): pair i in place in g; the rest: place i of the
// grouped matmuls (pair list[i]) into half for the down projection
kernel void moe_oai(constant moe_args & a [[buffer(0)]], device float * g [[buffer(1)]], device const float * u [[buffer(2)]],
                    device const int * ids [[buffer(3)]], device const int * list [[buffer(4)]], device const float * bg [[buffer(5)]],
                    device const float * bu [[buffer(6)]], device half * oh [[buffer(7)]], constant int & n [[buffer(8)]],
                    constant int & grouped [[buffer(9)]], uint2 i [[thread_position_in_grid]]) {
    if ((int) i.x >= n) return;
    const int e = ids[grouped ? list[i.y] : i.y];
    const size_t at = (size_t) i.y * n + i.x;
    const float x0 = min(g[at] + bg[(size_t) e * n + i.x], 7.0f), x1 = clamp(u[at] + bu[(size_t) e * n + i.x], -7.0f, 7.0f);
    const float v = x0 / (1.0f + exp(-1.702f * x0)) * (1.0f + x1);
    if (grouped) oh[at] = half(clamp(v, -65504.0f, 65504.0f));
    else g[at] = v;
}
// the gathered rows: row i of x (in half) is the row of pair list[i]
kernel void moe_gather(constant moe_args & a [[buffer(0)]], device const float * x [[buffer(1)]], device const int * list [[buffer(2)]],
                       device half * xg [[buffer(3)]], constant int & n [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    if ((int) g.x >= n / 4) return;
    const int src = list[g.y] / a.k_used;
    *((device half4 *) (xg + (size_t) g.y * n) + g.x) = half4(clamp(*((device const float4 *) (x + (size_t) src * n) + g.x), -65504.0f, 65504.0f));
}
// x += the experts' outputs weighted in slot order (interactive rows' pairs from yi, in pair order; the rest from yd at
// their list place), then sigmoid(gate) times the shared expert's output. Mode 1 (gpt-oss): each output plus its expert's
// down bias, no shared expert
kernel void moe_out(constant moe_args & a [[buffer(0)]], device float * x [[buffer(1)]], device const float * yi [[buffer(2)]],
                    device const float * yd [[buffer(3)]], device const int * pos [[buffer(4)]], device const float * w [[buffer(5)]],
                    device const float * sh [[buffer(6)]], device const float * sgt [[buffer(7)]], constant int & n [[buffer(8)]],
                    device const int * ids [[buffer(9)]], device const float * bd [[buffer(10)]], uint2 g [[thread_position_in_grid]]) {
    const int c = g.x, r = g.y;
    if (c >= n) return;
    float s = 0.f;
    for (int k = 0; k < a.k_used; ++k) {
        const int pr = r * a.k_used + k;
        float v = r < a.r0 ? yi[(size_t) pr * n + c] : yd[(size_t) pos[pr] * n + c];
        if (a.mode == 1) v += bd[(size_t) ids[pr] * n + c];
        s += w[pr] * v;
    }
    if (a.mode == 0) s += (1.0f / (1.0f + exp(-sgt[r]))) * sh[(size_t) r * n + c];
    x[(size_t) r * n + c] += s;
}
// moe_out (mode 0, an interactive row: r < r0) and the next layer's first rmsnorm in one dispatch, a threadgroup of 256 a row:
// thread t holds the float4s t and t + 256 of the row (rmsnorm's lanes, n = 2048 or any n % 4 == 0 with n / 4 <= 512 per loop
// turn), computes each new x element exactly as moe_out does (the experts in slot order, then the shared expert term), stores
// it, and sums the squares of the new float4s in rmsnorm's order; the rest is rmsnorm's tail verbatim
kernel void moe_out_norm(constant moe_args & a [[buffer(0)]], device float * x [[buffer(1)]], device const float * yi [[buffer(2)]],
                         device const float * w [[buffer(3)]], device const float * sh [[buffer(4)]], device const float * sgt [[buffer(5)]],
                         constant norm_args & nm [[buffer(6)]], device const float * nw [[buffer(7)]], device float * y [[buffer(8)]],
                         uint r [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const int n = nm.n;
    device float4 * xr = (device float4 *) (x + (size_t) r * nm.ldx);
    float s = 0.f;
    for (int i = t; i < n / 4; i += 256) {
        float nx[4];
        for (int j = 0; j < 4; ++j) {
            const int c = i * 4 + j;
            float e = 0.f;
            for (int k = 0; k < a.k_used; ++k) {
                const int pr = r * a.k_used + k;
                float v = yi[(size_t) pr * n + c];
                e += w[pr] * v;
            }
            e += (1.0f / (1.0f + exp(-sgt[r]))) * sh[(size_t) r * n + c];
            nx[j] = x[(size_t) r * n + c] + e;
        }
        const float4 nv = float4(nx[0], nx[1], nx[2], nx[3]);
        xr[i] = nv;
        s += dot(nv, nv);
    }
    s = simd_sum(s);
    if (ln == 0) {
        part[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int i = 0; i < 8; ++i) {
        tot += part[i];
    }
    const float sc = 1.0f / sqrt(tot / nm.n + nm.eps);
    device float4 * yr = (device float4 *) (y + (size_t) r * nm.ldy);
    for (int i = t; i < n / 4; i += 256) yr[i] = (xr[i] * sc) * ((device const float4 *) nw)[i];
}
