// Keep T=8 MLP scheduling edits independent of generic decode instantiations.
#include "trellis_decode_helpers.cuh"

namespace ignis_tl {
namespace {
template <int K2>
__device__ __forceinline__ void accumulate_mlp_tiles(const DecodeParams &p, const Smem &s, int wsel,
    const uint32_t *base, size_t row_words, size_t slice_words,
    const typename PlanOf<K2>::type &plan, float (&acc)[2][1][4]) {
  constexpr int words = ignis_trellis::tile_words(K2);
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
}

// K3 weights stay in registers. Gather both output tiles of the current K
// block before reusing their packed storage for the next block.
template <>
__device__ __forceinline__ void accumulate_mlp_tiles<6>(const DecodeParams &p, const Smem &s, int wsel,
    const uint32_t *base, size_t row_words, size_t slice_words,
    const PlanOf<6>::type &plan, float (&acc)[2][1][4]) {
  constexpr int words = ignis_trellis::tile_words(6);
  uint32_t packed[2][8][1];
#pragma unroll
  for (int j = 0; j < 2; ++j)
    load_block<6>(base + j * words, row_words, packed[j]);
  grid_dependency_wait();
  stage_block(p, s, 0, wsel, 0);
  cp_async_commit();
  stage_block(p, s, 1, wsel, 1);
  cp_async_commit();
  for (int i = 0; i < p.nb; i += 2) {
#pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
      cp_async_wait<1>();
      __syncthreads();
      uint32_t gathered[2][8][2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
          const uint32_t word = packed[j][tile][0];
          gathered[j][tile][0] = __shfl_sync(0xffffffffu, word, plan.w0);
          gathered[j][tile][1] = __shfl_sync(0xffffffffu, word, plan.w1);
        }
      }
      const int next = i + stage + 1;
      if (next < p.nb) {
#pragma unroll
        for (int j = 0; j < 2; ++j)
          load_block<6>(base + next * slice_words + j * words, row_words, packed[j]);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j)
        compute_block<6, 1, true>(gathered[j], plan, s.a[stage], s.inv[stage], 8, acc[j]);
      __syncthreads();
      if (next + 1 < p.nb) stage_block(p, s, stage, wsel, next + 1);
      cp_async_commit();
    }
  }
}

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
  float acc[2][1][4] = {};
  accumulate_mlp_tiles<K2>(p, s, wsel, base, row_words, slice_words, plan, acc);
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

// T=8 down keeps the generic route's four K splits and accumulation order.
// Eight warps each own one 16-column tile of a 128-output CTA.
template <int K2>
__device__ void down_t8_partial_unit(const DecodeParams &p, const Smem &s, int cb, int split) {
  constexpr int words = ignis_trellis::tile_words(K2);
  const Weight &w = p.w[0];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int b0 = split * p.nb / p.splits;
  const int n = (split + 1) * p.nb / p.splits - b0;
  const size_t row_words = static_cast<size_t>(w.stride) * words;
  const size_t slice_words = 8 * row_words;
  const uint32_t *base = w.trellis + static_cast<size_t>(b0) * slice_words + (cb * 8 + warp) * words;
  const auto plan = plan_for<K2>(lane);
  uint32_t ra[8][kRegs<K2>], rb[8][kRegs<K2>];
  load_block<K2>(base, row_words, ra);
  if (n > 1) load_block<K2>(base + slice_words, row_words, rb);
  grid_dependency_wait();
  stage_block(p, s, 0, 0, b0);
  cp_async_commit();
  if (n > 1) stage_block(p, s, 1, 0, b0 + 1);
  cp_async_commit();
  float acc[1][4] = {};
  for (int i = 0; i < n; i += 2) {
    cp_async_wait<1>();
    __syncthreads();
    compute_block<K2, 1>(ra, plan, s.a[0], s.inv[0], 8, acc);
    if (i + 2 < n) load_block<K2>(base + (i + 2) * slice_words, row_words, ra);
    __syncthreads();
    if (i + 2 < n) stage_block(p, s, 0, 0, b0 + i + 2);
    cp_async_commit();
    if (i + 1 >= n) break;
    cp_async_wait<1>();
    __syncthreads();
    compute_block<K2, 1>(rb, plan, s.a[1], s.inv[1], 8, acc);
    if (i + 3 < n) load_block<K2>(base + (i + 3) * slice_words, row_words, rb);
    __syncthreads();
    if (i + 3 < n) stage_block(p, s, 1, 0, b0 + i + 3);
    cp_async_commit();
  }
  cp_async_wait<0>();
  __syncthreads();
  const int g = lane >> 2, c = lane & 3;
  const int col = warp * 16 + g;
  float *part = p.partials + static_cast<size_t>(split) * 8 * p.out + cb * 128 + col;
  part[(2 * c) * p.out] = acc[0][0];
  part[(2 * c) * p.out + 8] = acc[0][2];
  part[(2 * c + 1) * p.out] = acc[0][1];
  part[(2 * c + 1) * p.out + 8] = acc[0][3];
  // Ordinary stream ordering publishes partials to the reduction kernel.
}

// One shared stage covers 128 input features, 128 output features and 8 tokens.
struct alignas(16) DownK3Stage {
  uint32_t packed[8 * 8 * ignis_trellis::tile_words(6)];
  __half activation[8][kXRow];
  float inverse_scale[8];
};

__device__ __forceinline__ void fetch_down_stage(
    DownK3Stage &stage, const DecodeParams &p, int output_group, int block) {
  constexpr int words = ignis_trellis::tile_words(6);
  const Weight &weight = p.w[0];
  const size_t row_stride = size_t(weight.stride) * words;
  const uint32_t *base = weight.trellis + block * 8 * row_stride + output_group * 8 * words;

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1 && lane >= 16) break;
    const int word = 4 * (pass == 0 ? tid : 256 + 16 * warp + lane);
    const int row = word / (8 * words);
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(stage.packed + word));
    unsigned long long policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0;" : "=l"(policy));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(base + row * row_stride + word % (8 * words)), "l"(policy));
  }
  if (lane >= 16) {
    const int col = (lane - 16) * 8;
    cp_async16(&stage.activation[warp][col], p.xr + block * 1024 + warp * 128 + col);
  }
  if (lane == 0)
    cp_async4(&stage.inverse_scale[warp], p.inv + block * 8 + warp);
}

struct DownFragment {
  uint32_t packed;
  uint2 activation;
};

__device__ __forceinline__ DownFragment load_down_fragment(const DownK3Stage &stage, int k) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  constexpr int words = ignis_trellis::tile_words(6);
  return {lane < words ? stage.packed[(k * 8 + warp) * words + lane] : 0u,
          *reinterpret_cast<const uint2 *>(&stage.activation[lane / 4][(lane % 4) * 32 + 4 * k])};
}

__global__ __launch_bounds__(256) void down_k3_multistage_kernel(const __grid_constant__ DecodeParams p) {
  __shared__ DownK3Stage stages[2];
  const int split = blockIdx.x % p.splits;
  const int output_group = blockIdx.x / p.splits;
  const int begin = split * p.nb / p.splits;
  const int end = (split + 1) * p.nb / p.splits;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const auto plan = plan_for<6>(lane);

  grid_dependency_wait();
  fetch_down_stage(stages[0], p, output_group, begin);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  float acc[4] = {};
  for (int block = begin, slot = 0; block < end; ++block, slot ^= 1) {
    const int next = block + 1;
    if (next < end)
    fetch_down_stage(stages[slot ^ 1], p, output_group, next);
    cp_async_commit();
    DownFragment fragments[2];
    const DownK3Stage &current = stages[slot];
    fragments[0] = load_down_fragment(current, 0);
    const float scale0 = current.inverse_scale[2 * (lane % 4)];
    const float scale1 = current.inverse_scale[2 * (lane % 4) + 1];
    uint32_t half_acc[2] = {};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (k < 7) fragments[(k + 1) % 2] = load_down_fragment(current, k + 1);
      const DownFragment &frag = fragments[k % 2];
      const uint32_t packed[1] = {frag.packed};
      uint32_t decoded[4];
      tile_fragment<6>(packed, plan, decoded);
      const uint32_t operand[4] = {decoded[0], decoded[2], decoded[1], decoded[3]};
      mma_f16acc(half_acc, operand, frag.activation.x, frag.activation.y);
      if (k % 2 == 1) {
        const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&half_acc[0]));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&half_acc[1]));
        acc[0] = fmaf(lo.x, scale0, acc[0]);
        acc[1] = fmaf(lo.y, scale1, acc[1]);
        acc[2] = fmaf(hi.x, scale0, acc[2]);
        acc[3] = fmaf(hi.y, scale1, acc[3]);
        half_acc[0] = half_acc[1] = 0;
      }
    }
    cp_async_wait<0>();
    __syncthreads(); // Next slot is ready; all readers have released the current slot.
  }
  const int col = output_group * 128 + warp * 16 + lane / 4;
  float *part = p.partials + size_t(split) * 8 * p.out + col;
  const int token = 2 * (lane % 4);
  part[token * p.out] = acc[0];
  part[token * p.out + 8] = acc[2];
  part[(token + 1) * p.out] = acc[1];
  part[(token + 1) * p.out + 8] = acc[3];
}

// Each warp owns one token and a complete 128-output rotation group.
__global__ void __launch_bounds__(128) down_t8_reduce_kernel(const __grid_constant__ DecodeParams p) {
  const int lane = threadIdx.x & 31;
  const int item = static_cast<int>(blockIdx.x) * 4 + (threadIdx.x >> 5);
  const int cb = item / 8, t = item % 8;
  if (cb >= p.out / 128) return;
  float v[4];
  sum_splits(p, t, 0, cb, v);
  warp_hadamard128(v);
  float scale[4];
  load_half4(p.w[0].svh + cb * 128 + 4 * lane, scale);
#pragma unroll
  for (int q = 0; q < 4; ++q) v[q] = __fmul_rn(v[q], scale[q]);
  store_bf16x4(p.y + static_cast<size_t>(t) * p.ldy + cb * 128 + 4 * lane, v, true);
}

__global__ void __launch_bounds__(128) down_k3_reduce4_kernel(const __grid_constant__ DecodeParams p) {
  const int lane = threadIdx.x & 31;
  const int item = static_cast<int>(blockIdx.x) * 4 + (threadIdx.x >> 5);
  const int cb = item / 8, t = item % 8;
  // The caller selects exactly 40 column groups and 8 tokens.
  const float4 *src = reinterpret_cast<const float4 *>(p.partials) + t * 1280 + cb * 32 + lane;
  float v[4] = {};
#pragma unroll
  for (int split = 0; split < 4; ++split) {
    const float4 q = __ldcg(src + split * 10240);
    v[0] += q.x;
    v[1] += q.y;
    v[2] += q.z;
    v[3] += q.w;
  }
  warp_hadamard128(v);
  float scale[4];
  load_half4(p.w[0].svh + cb * 128 + 4 * lane, scale);
#pragma unroll
  for (int q = 0; q < 4; ++q) v[q] = __fmul_rn(v[q], scale[q]);
  store_bf16x4(p.y + static_cast<size_t>(t) * p.ldy + cb * 128 + 4 * lane, v, true);
}

__global__ void __launch_bounds__(256, 2) down_t8_partial_kernel(const __grid_constant__ DecodeParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const Smem s = carve(smem_raw, 8);
  const int ticket = static_cast<int>(blockIdx.x);
  dispatch_k2(p.w[0].k2, [&](auto k) {
    constexpr int K2 = decltype(k)::value;
    if constexpr (K2 == 4 || K2 == 6)
      down_t8_partial_unit<K2>(p, s, ticket / p.splits, ticket % p.splits);
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
cudaError_t launch_down_t8(cudaLaunchConfig_t &cfg, const DecodeParams &p) {
  cudaError_t err;
  if (p.w[0].k2 == 6) {
    // On the 84-SM RTX 5080, reserve two more stages' worth of shared memory
    // so this 160-CTA grid spreads across the card even with PDL enabled.
    cfg.dynamicSmemBytes = 2 * sizeof(DownK3Stage);
    err = cudaLaunchKernelEx(&cfg, down_k3_multistage_kernel, p);
  } else {
    err = cudaLaunchKernelEx(&cfg, down_t8_partial_kernel, p);
  }
  if (err != cudaSuccess) return err;
  if (p.w[0].k2 == 6)
    down_k3_reduce4_kernel<<<80, 128, 0, cfg.stream>>>(p);
  else
    down_t8_reduce_kernel<<<(p.out / 128) * 2, 128, 0, cfg.stream>>>(p);
  return cudaGetLastError();
}
} // namespace ignis_tl
