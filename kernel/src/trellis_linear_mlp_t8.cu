// Keep T=8 MLP scheduling edits independent of generic decode instantiations.
#include "trellis_decode_helpers.cuh"

namespace ignis_tl {
namespace {
// T=8 MLP: four warps retain the same 128-output work unit.
// Each warp owns two adjacent 16-output tiles; each output keeps the baseline K order.
template <int K2>
__device__ void mlp_unit4(const DecodeParams &p, const Smem &s, int wsel, int cb, int *s_last) {
  constexpr int words = ignis_trellis::tile_words(K2);
  const Weight &w = p.w[wsel];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const size_t row_words = static_cast<size_t>(w.stride) * words;
  const size_t slice_words = 8 * row_words;
  const uint32_t *base = w.trellis + (cb * 8 + warp * 2) * words;
  const auto plan = plan_for<K2>(lane);
  uint32_t ra[2][8][kRegs<K2>], rb[2][8][kRegs<K2>];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    load_block<K2>(base + j * words, row_words, ra[j]);
    load_block<K2>(base + slice_words + j * words, row_words, rb[j]);
  }
  grid_dependency_wait();
  // T=8 needs 128 activation copy threads, so the baseline staging routine also
  // covers every byte with this smaller CTA (its second loop iteration is unused).
  stage_block(p, s, 0, wsel, 0);
  cp_async_commit();
  stage_block(p, s, 1, wsel, 1);
  cp_async_commit();
  float acc[2][1][4] = {};
  for (int i = 0; i < p.nb; i += 2) {
    cp_async_wait<1>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 2; ++j)
      compute_block<K2, 1>(ra[j], plan, s.a[0], s.inv[0], 8, acc[j]);
    if (i + 2 < p.nb) {
#pragma unroll
      for (int j = 0; j < 2; ++j)
        load_block<K2>(base + (i + 2) * slice_words + j * words, row_words, ra[j]);
    }
    __syncthreads();
    if (i + 2 < p.nb) stage_block(p, s, 0, wsel, i + 2);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 2; ++j)
      compute_block<K2, 1>(rb[j], plan, s.a[1], s.inv[1], 8, acc[j]);
    if (i + 3 < p.nb) {
#pragma unroll
      for (int j = 0; j < 2; ++j)
        load_block<K2>(base + (i + 3) * slice_words + j * words, row_words, rb[j]);
    }
    __syncthreads();
    if (i + 3 < p.nb) stage_block(p, s, 1, wsel, i + 3);
    cp_async_commit();
  }
  cp_async_wait<0>();
  __syncthreads();
  const int g = lane >> 2, c = lane & 3;
  const size_t outw = static_cast<size_t>(2) * p.out;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int col = (warp * 2 + j) * 16 + g;
    float *part = p.partials + static_cast<size_t>(wsel) * p.out + cb * 128 + col;
    part[(2 * c) * outw] = acc[j][0][0];
    part[(2 * c) * outw + 8] = acc[j][0][2];
    part[(2 * c + 1) * outw] = acc[j][0][1];
    part[(2 * c + 1) * outw + 8] = acc[j][0][3];
  }
  // Publish every warp's partials before gate/up arrival; the last CTA performs
  // the complete 128-output rotation. Keep this fence even without ticket scheduling.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) *s_last = atomicAdd(&p.arrivals[cb], 1u) == 1u;
  __syncthreads();
  if (*s_last) {
    __threadfence();
    for (int t = warp; t < 8; t += 4) {
      float gate[4], up[4];
      sum_splits(p, t, 0, cb, gate);
      rotate_out(gate, p.w[0].svh, cb);
      sum_splits(p, t, 1, cb, up);
      rotate_out(up, p.w[1].svh, cb);
#pragma unroll
      for (int q = 0; q < 4; ++q) gate[q] = ignis_moe::silu(gate[q]) * up[q];
      store_bf16x4(p.y + static_cast<size_t>(t) * p.ldy + cb * 128 + 4 * lane, gate, false);
    }
    if (threadIdx.x == 0) p.arrivals[cb] = 0;
  }
}

// The fixed MLP shape has 272 independent work units. Launch all of them;
// scheduling capacity affects performance, never the number of outputs computed.
__global__ void __launch_bounds__(128) mlp_static4_kernel(const __grid_constant__ DecodeParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ int last;
  const Smem s = carve(smem_raw, 8);
  const int ticket = static_cast<int>(blockIdx.x);
  const int wsel = ticket % 2, cb = ticket / 2;
  dispatch_k2(p.w[wsel].k2, [&](auto k) {
    constexpr int K2 = decltype(k)::value;
    if constexpr (K2 < 10) mlp_unit4<K2>(p, s, wsel, cb, &last);
    else __trap();
  });
}


} // namespace
int32_t prepare_mlp_t8(DeviceState *st) {
  const cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &st->mlp4_ctas, mlp_static4_kernel, 128, decode_smem(8));
  if (err != cudaSuccess || st->mlp4_ctas < 1) return fail("MLP4 occupancy query failed");
  return 0;
}
cudaError_t launch_mlp_t8(cudaLaunchConfig_t &cfg, const DecodeParams &p) {
  return cudaLaunchKernelEx(&cfg, mlp_static4_kernel, p);
}
} // namespace ignis_tl
