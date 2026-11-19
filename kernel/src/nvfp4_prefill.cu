// Adapted from gpillon/ninfer, commit a00648cb828457986cf5b4b4f712b4cbcd7af0d1,
// src/ops/kernel/gqa_attention_prefill_bf16.cuh, Apache-2.0 (kernel/vendor/LICENSE).
// Owned adaptation: normal-frame linear bands, FP32 carry and per-query positions;
// no HQ transform, no parity claim for the new band-boundary reduction order.
// Adapted from kernel/vendor/src/ops/kernel/gqa_attention_prefill_bf16.cuh
// (vendored ninfer, Apache-2.0; see kernel/vendor/LICENSE). The 64x64 BF16
// tensor-core tile, swizzle, QK/PV fragment mapping and online-softmax math
// retain that implementation. Ignis changes: normal-frame, head-major decoded
// linear bands independent of HQ rotation, no paged access/split path, actual
// per-query positions, and FP32 unnormalized carry across bands.
#include "nvfp4_prefill.h"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <string>

namespace {
using namespace ninfer::ops;
using Geometry = Gqa27Geometry;

__device__ __forceinline__ void stage_kv(
    __nv_bfloat16 *dst, const __nv_bfloat16 *cache, int kv_head, int k0,
    int max_query_abs, int tid, int span, int begin, int end) {
  constexpr int D = kGqaPrefillHeadDim, Bc = kGqaPrefillBc;
  const int bound = min(max_query_abs, end - 1);
  const auto *base = cache + (static_cast<std::int64_t>(kv_head) * span + k0 - begin) * D;
#pragma unroll
  for (int chunk = tid; chunk < Bc * (D / 8); chunk += kGqaPrefillThreads) {
    const int row = chunk >> 5, d = (chunk & 31) << 3;
    auto *p = &dst[row * D + gqa_prefill_swz(row, d)];
    if (k0 + row <= bound) cp_async<16, Cache::cg>(p, &base[row * D + d]);
    else store_vec(p, make_int4(0, 0, 0, 0));
  }
}

__device__ void neutral_rows(const IgnisNv4PrefillBand &p, int head, int begin, int end,
                            int tid) {
  for (int i = tid; i < (end - begin) * 256; i += kGqaPrefillThreads) {
    const int row = begin + i / 256, d = i % 256;
    const auto index = gqa_prefill_q_index<Geometry>(head, d, row);
    if (p.save_carry) p.carry_acc[index] = 0.0f;
    else static_cast<__nv_bfloat16 *>(p.out)[index] = __float2bfloat16(0.0f);
  }
  if (p.save_carry) {
    for (int row = begin + tid; row < end; row += kGqaPrefillThreads) {
      const auto stat = static_cast<std::int64_t>(head) * p.tokens + row;
      p.carry_m[stat] = -CUDART_INF_F;
      p.carry_l[stat] = 0.0f;
    }
  }
}

__global__ __launch_bounds__(kGqaPrefillThreads, 1)
void prefill_band_kernel(IgnisNv4PrefillBand p) {
    constexpr int D = kGqaPrefillHeadDim, Br = kGqaPrefillBr, Bc = kGqaPrefillBc;
    constexpr int Threads = kGqaPrefillThreads;
    constexpr int QKNt = Bc / 8, QKKs = D / 16, PVNt = D / 8, PVKs = Bc / 16;
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    const auto *q = static_cast<const __nv_bfloat16 *>(p.q);
    const auto *cache_k = static_cast<const __nv_bfloat16 *>(p.k);
    const auto *cache_v = static_cast<const __nv_bfloat16 *>(p.v);
    const auto *positions = p.positions;
    const int width = p.tokens, key_begin = p.band_begin, key_end = p.band_end;
    const int scratch_span = p.band_span;
    const int tokens = p.valid ? max(0, min(width, *p.valid)) : width;
    const int q_head = blockIdx.y, q0 = blockIdx.x * Br;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kv_head = q_head / Geometry::GroupSize;
    if (q0 >= width) return;
    if (q0 >= tokens) {
      neutral_rows(p, q_head, q0, min(q0 + Br, width), tid);
      return;
    }
    extern __shared__ __align__(16) __nv_bfloat16 storage[];
    auto *q_s = storage;
    auto *k_s = q_s + Br * D;
    auto *v_s = k_s + Bc * D;
    // Actual positions, not a presumed positions[0]+row sequence. This also
    // keeps Q/K staging independent of inactive suffix positions.
    __shared__ int row_max[4], row_min[4];
    int hi = tid < Br && q0 + tid < tokens ? positions[q0 + tid] : -1;
    int lo = tid < Br && q0 + tid < tokens ? positions[q0 + tid] : INT_MAX;
#pragma unroll
    for (int step = 16; step; step >>= 1) {
      hi = max(hi, __shfl_xor_sync(FullMask, hi, step));
      lo = min(lo, __shfl_xor_sync(FullMask, lo, step));
    }
    if (lane == 0) { row_max[warp] = hi; row_min[warp] = lo; }
    __syncthreads();
    const int max_query_abs = max(max(row_max[0], row_max[1]), max(row_max[2], row_max[3]));
    const int min_query_abs = min(min(row_min[0], row_min[1]), min(row_min[2], row_min[3]));
    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16; // this warp owns rows [warp_row0, warp_row0+16)

    // Per-lane precomputed swizzled ldmatrix base addresses (see gqa_prefill_swz_addr).
    const unsigned q_sbase = smem_addr(q_s);
    const unsigned k_sbase = smem_addr(k_s);
    const unsigned v_sbase = smem_addr(v_s);
    // Q A-fragment: row = warp_row0 + a_rowoff, col = k*16 + a_coloff.
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    // K B-fragment via ldmatrix.x4 (2 n-tiles/instr): lanes 16-31 fetch the +8-key
    // half (extra 4096 bytes), lanes with bit3 set fetch the +8 d-contract half.
    const unsigned k_lane_base =
        k_sbase + static_cast<unsigned>(b_rin * 512) + (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
    // V B-fragment via ldmatrix.x4.trans (2 n-tiles/instr): row = k*16 + (bit3)*8 + b_rin,
    // col = n*8 + (lane>>4)*8.
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                                 static_cast<unsigned>(b_rin * 512);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    // Stage Q into smem once via cp.async (overlaps with the K(0) prologue load
    // below); it stays resident for the whole key loop. Global Q rows are 256 bf16
    // contiguous, with a token stride of 256*QHeads.
    {
        constexpr int VecPerRow      = D / 8;
        constexpr int QRowStride     = D * Geometry::QHeads; // global stride between tokens
        const __nv_bfloat16* q_block = q + gqa_prefill_q_index<Geometry>(q_head, 0, q0);
        if (q0 + Br <= tokens) {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                if (q0 + row < tokens) {
                    cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
                } else {
                    store_vec(p, make_int4(0, 0, 0, 0));
                }
            }
        }
    }

    float acc[PVNt][4];
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    const int carry_row0 = q0 + warp_row0 + gid, carry_row1 = carry_row0 + 8;
    if (key_begin > 0) {
      if (carry_row0 < tokens) {
        const auto stat = static_cast<std::int64_t>(q_head) * width + carry_row0;
        m0 = p.carry_m[stat]; l0 = .25f * p.carry_l[stat];
      }
      if (carry_row1 < tokens) {
        const auto stat = static_cast<std::int64_t>(q_head) * width + carry_row1;
        m1 = p.carry_m[stat]; l1 = .25f * p.carry_l[stat];
      }
    }
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
      const int d0 = n * 8 + 2 * lid;
      float2 a0 = make_float2(0.0f, 0.0f), a1 = a0;
      if (key_begin > 0 && carry_row0 < tokens && m0 != -CUDART_INF_F && l0 > 0.0f)
        a0 = *reinterpret_cast<const float2 *>(&p.carry_acc[gqa_prefill_q_index<Geometry>(q_head, d0, carry_row0)]);
      if (key_begin > 0 && carry_row1 < tokens && m1 != -CUDART_INF_F && l1 > 0.0f)
        a1 = *reinterpret_cast<const float2 *>(&p.carry_acc[gqa_prefill_q_index<Geometry>(q_head, d0, carry_row1)]);
      acc[n][0] = a0.x; acc[n][1] = a0.y; acc[n][2] = a1.x; acc[n][3] = a1.y;
    }
    const int kb_split_begin = key_begin / Bc;
    const int key_limit = min(key_end, max_query_abs + 1);
    const int kb_split_end = (key_limit + Bc - 1) / Bc;
    const float scale_l2 = p.scale * Log2E;
    cp_commit(); // Q; empty bands must still drain this copy before returning.
    if (kb_split_begin < kb_split_end) {
      stage_kv(k_s, cache_k, kv_head, kb_split_begin * Bc, max_query_abs,
               tid, scratch_span, key_begin, key_end);
      cp_commit();
    }
    for (int kb = kb_split_begin; kb < kb_split_end; ++kb) {
      const int k0 = kb * Bc;
      cp_wait<0>();
      __syncthreads();
      stage_kv(v_s, cache_v, kv_head, k0, max_query_abs, tid, scratch_span, key_begin, key_end);
      cp_commit();
        // S = Q Kᵀ for this warp's 16 rows over all Bc keys, in registers.
        // Software-pipelined like cute's gemm: issue the ldmatrix for contraction
        // step k+1 while the m16n8k16 MMAs for step k run, so the LSU (ldmatrix)
        // and tensor pipes overlap instead of stalling on each other.
        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        // Swizzled ldmatrix addresses via precomputed per-lane bases + immediates.
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        {
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        gqa_prefill_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                            gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096),
                                                 0u, k_as, k_r));
            }
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            gqa_prefill_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), ck,
                                             k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        const int row0             = warp_row0 + gid;
        const int row1             = warp_row0 + gid + 8;
        const int qrow0            = q0 + row0;
        const int qrow1            = q0 + row1;
        const int qabs0            = (qrow0 < tokens) ? positions[qrow0] : -1;
        const int qabs1            = (qrow1 < tokens) ? positions[qrow1] : -1;
        const bool full_score_tile =
            (q0 + Br <= tokens) && ((k0 + Bc - 1) <= min_query_abs) &&
            ((k0 + Bc) <= key_end);

        // block row-max on raw (unscaled) scores; scale is folded into exp2 below
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                const bool key0_ok =
                    key0 <= qabs0 && (key0 < key_end);
                const bool key1_ok =
                    key1 <= qabs0 && (key1 < key_end);
                score[nt][0]   = (qrow0 < tokens && key0_ok) ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = (qrow0 < tokens && key1_ok) ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = (qrow1 < tokens && key0 <= qabs1 && (key0 < key_end))
                                     ? score[nt][2]
                                     : -CUDART_INF_F;
                score[nt][3]   = (qrow1 < tokens && key1 <= qabs1 && (key1 < key_end))
                                     ? score[nt][3]
                                     : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(m0, bm0);
        const float nm1        = fmaxf(m1, bm1);
        const float nm0_scaled = nm0 * scale_l2;
        const float nm1_scaled = nm1 * scale_l2;
        // A split segment can lie entirely past a row's causal limit, so both m and the
        // block max stay -inf here: the naive exp2 would compute -inf - (-inf) = NaN. The
        // INT8 kernel guards the same case; the single-pass bf16 launch never met it.
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f
                                               : exp2_approx(__fmaf_rn(m0, scale_l2, -nm0_scaled));
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f
                                               : exp2_approx(__fmaf_rn(m1, scale_l2, -nm1_scaled));

        // P = exp2(S - m), repacked into the PV A-fragment layout, plus local block row-sum.
        // The row-sum allreduce is deferred to the epilogue; only row max must be reduced per tile.
        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
                const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
                const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
                const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = (score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = (score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = (score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = (score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        }

        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        ninfer::ops::cp_wait<0>(); // V(kb) landed; QK done reading k_s
        __syncthreads();

        // Prefetch K(kb+1) into the (now-free) K buffer, overlapping the PV MMA.
        if (kb + 1 < kb_split_end) {
            stage_kv(k_s, cache_k, kv_head, (kb + 1) * Bc, max_query_abs,
                     tid, scratch_span, key_begin, key_end);
            ninfer::ops::cp_commit();
        }

        // O += P V, contracting over the Bc keys. The (k, n) iteration space is
        // flattened and software-pipelined: the transposed ldmatrix for the next
        // V fragment is issued while the current MMA runs.
        // Each x4.trans load covers 2 output n-tiles (16 dims); pipeline the next
        // load against the current pair of MMAs.
        constexpr int PVHalf  = PVNt / 2;      // 16 n-tile pairs
        constexpr int PVLoads = PVKs * PVHalf; // 64 x4.trans loads
        // Swizzled V x4.trans addresses via precomputed per-lane base + immediates.
        unsigned vf[2][4];
        {
            ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                          gqa_prefill_swz_addr(v_lane_base, 0u, v_as, v_r));
        }
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int k   = li / PVHalf;
            const int n2  = (li % PVHalf) * 2;
            const int cur = li & 1;
            const int nxt = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int k2       = (li + 1) / PVHalf;
                const int n2b      = ((li + 1) % PVHalf) * 2;
                const unsigned ckv = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                              gqa_prefill_swz_addr(v_lane_base + static_cast<unsigned>(k2 * 8192),
                                                   ckv, v_as, v_r));
            }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                     p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                     p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
    }

    // An empty band has no tile-loop barrier: explicitly drain staged Q.
    cp_wait<0>();
    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);
    if (p.save_carry && lid == 0) {
      if (carry_row0 < tokens) {
        const auto stat = static_cast<std::int64_t>(q_head) * width + carry_row0;
        p.carry_m[stat] = m0; p.carry_l[stat] = l0;
      }
      if (carry_row1 < tokens) {
        const auto stat = static_cast<std::int64_t>(q_head) * width + carry_row1;
        p.carry_m[stat] = m1; p.carry_l[stat] = l1;
      }
    }
    const float inv0 = l0 > 0.0f ? __frcp_rn(l0) : 0.0f;
    const float inv1 = l1 > 0.0f ? __frcp_rn(l1) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
      const int d0 = n * 8 + 2 * lid;
      if (carry_row0 < tokens) {
        const auto index = gqa_prefill_q_index<Geometry>(q_head, d0, carry_row0);
        if (p.save_carry) *reinterpret_cast<float2 *>(&p.carry_acc[index]) = make_float2(acc[n][0], acc[n][1]);
        else *reinterpret_cast<unsigned *>(&static_cast<__nv_bfloat16 *>(p.out)[index]) =
            pack_bf16x2(l0 > 0.0f ? acc[n][0] * inv0 : 0.0f, l0 > 0.0f ? acc[n][1] * inv0 : 0.0f);
      }
      if (carry_row1 < tokens) {
        const auto index = gqa_prefill_q_index<Geometry>(q_head, d0, carry_row1);
        if (p.save_carry) *reinterpret_cast<float2 *>(&p.carry_acc[index]) = make_float2(acc[n][2], acc[n][3]);
        else *reinterpret_cast<unsigned *>(&static_cast<__nv_bfloat16 *>(p.out)[index]) =
            pack_bf16x2(l1 > 0.0f ? acc[n][2] * inv1 : 0.0f, l1 > 0.0f ? acc[n][3] * inv1 : 0.0f);
      }
    }
    neutral_rows(p, q_head, max(q0, tokens), min(q0 + Br, width), tid);
}

void check(cudaError_t result, const char *what) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string("ignis_nv4_prefill: ") + what + ": " + cudaGetErrorString(result));
}
} // namespace

void ignis_nv4_prefill_prepare() {
  check(cudaFuncSetAttribute(prefill_band_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kGqaPrefillSmemBytes), "shared-memory opt-in");
}

void ignis_nv4_prefill_band(const IgnisNv4PrefillBand &p, cudaStream_t stream) {
  const bool carry = p.save_carry || p.band_begin > 0;
  if (p.tokens < 0 || p.band_begin < 0 || p.band_begin % 64 || p.band_end < p.band_begin ||
      p.band_span < 0 || p.band_span > 16384 || p.band_end - p.band_begin > p.band_span ||
      !std::isfinite(p.scale) || p.scale <= 0.0f ||
      (p.tokens && (!p.q || !p.positions || (!p.save_carry && !p.out))) ||
      (p.tokens && carry && (!p.carry_acc || !p.carry_m || !p.carry_l)) ||
      (p.tokens && p.band_end > p.band_begin && (!p.k || !p.v)))
    throw std::invalid_argument("ignis_nv4_prefill: invalid band parameters");
  if (!p.tokens) return;
  const dim3 grid((p.tokens + kGqaPrefillBr - 1) / kGqaPrefillBr, Geometry::QHeads);
  prefill_band_kernel<<<grid, kGqaPrefillThreads, kGqaPrefillSmemBytes, stream>>>(p);
  check(cudaGetLastError(), "band launch");
}
