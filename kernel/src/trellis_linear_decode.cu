#include "trellis_decode_helpers.cuh"

namespace ignis_tl {
namespace {
template <int NG, bool kWide>
__global__ void __launch_bounds__(kThreads, !kWide && NG <= 2 ? 2 : 1) decode_kernel(const __grid_constant__ DecodeParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ int s_ticket, s_next, s_last;
  const Smem s = carve(smem_raw, p.tokens);
  if (sm_admitted(p.capped, p.admit)) {
    if (threadIdx.x == 0) s_ticket = static_cast<int>(atomicAdd(&p.counters[0], 1u));
    __syncthreads();
    while (s_ticket < p.units) {
      const int ticket = s_ticket;
      if (threadIdx.x == 0) s_next = static_cast<int>(atomicAdd(&p.counters[0], 1u));
      // Splits of one output block are consecutive tickets, and SwiGLU's gate and up of a block
      // are neighbours: the last arrival finds its partial sums fresh in the L2.
      const int split = ticket % p.splits;
      const int rest = ticket / p.splits;
      const int wsel = rest % p.weights;
      const int cb = rest / p.weights;
      const int k2 = p.w[wsel].k2;
      if constexpr (kWide) {
        dispatch_k2(k2, [&](auto k) { unit<decltype(k)::value, NG>(p, s, wsel, cb, split, &s_last); });
      } else {
        dispatch_k2(k2, [&](auto k) {
          constexpr int K2 = decltype(k)::value;
          if constexpr (K2 < 10) {
            unit<K2, NG>(p, s, wsel, cb, split, &s_last);
          } else {
            __trap();
          }
        });
      }
      __syncthreads();
      if (threadIdx.x == 0) s_ticket = s_next;
      __syncthreads();
    }
  }
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&p.counters[1], 1u) == gridDim.x - 1) {
      p.counters[0] = 0;
      p.counters[1] = 0;
    }
  }
}

template <int NG, bool kWide> int32_t prepare_one(DeviceState *st, int cls) {
  const size_t smem = decode_smem(8 * NG);
  cudaError_t err = cudaFuncSetAttribute(decode_kernel<NG, kWide>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem));
  if (err != cudaSuccess) return fail(std::string("ignis_trellis_prepare: decode attributes: ") + cudaGetErrorString(err));
  int ctas = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, decode_kernel<NG, kWide>, kThreads, smem);
  if (err != cudaSuccess || ctas < 1) return fail("ignis_trellis_prepare: the decode kernel does not fit an SM");
  st->decode_ctas[kWide ? 1 : 0][cls] = ctas;
  return 0;
}

template <bool kWide> cudaError_t launch_one(cudaLaunchConfig_t &cfg, int cls, const DecodeParams &p) {
  switch (cls) {
  case 0: return cudaLaunchKernelEx(&cfg, decode_kernel<1, kWide>, p);
  case 1: return cudaLaunchKernelEx(&cfg, decode_kernel<2, kWide>, p);
  case 2: return cudaLaunchKernelEx(&cfg, decode_kernel<4, kWide>, p);
  default: return cudaLaunchKernelEx(&cfg, decode_kernel<8, kWide>, p);
  }
}

}  // namespace

int32_t prepare_decode(DeviceState *st) {
  if (prepare_one<1, false>(st, 0) != 0 || prepare_one<2, false>(st, 1) != 0 || prepare_one<4, false>(st, 2) != 0 ||
      prepare_one<8, false>(st, 3) != 0 || prepare_one<1, true>(st, 0) != 0 || prepare_one<2, true>(st, 1) != 0 ||
      prepare_one<4, true>(st, 2) != 0 || prepare_one<8, true>(st, 3) != 0) {
    return -1;
  }
  return prepare_mlp_t8(st);
}

int32_t launch_decode(const DeviceState &st, const DecodeParams &p, cudaStream_t stream) {
  const int cls = decode_class(static_cast<uint32_t>(p.tokens));
  const bool wide = p.w[0].k2 >= 10 || (p.weights == 2 && p.w[1].k2 >= 10);
  const int resident = st.decode_ctas[wide ? 1 : 0][cls] * st.sms;
  // Capped, every SM gets its full share of CTAs and the unadmitted ones leave.
  const int grid = p.capped ? resident : (p.units < resident ? p.units : resident);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = decode_smem(p.tokens);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const bool mlp4 = st.sms == 84 && !p.capped && p.tokens == 8 && p.in == 5120 && p.out == 17408 &&
                    p.epilogue == kSwiGlu && p.splits == 1 && !wide && p.w[0].k2 == p.w[1].k2 &&
                    (p.w[0].k2 == 4 || p.w[0].k2 == 6 || p.w[0].k2 == 8);
  cudaError_t err;
  if (mlp4) {
    cfg.gridDim = dim3(p.units);
    cfg.blockDim = dim3(128);
    // Static work assignment does not pre-reserve another output group.
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    err = launch_mlp_t8(cfg, p);
  } else {
    err = wide ? launch_one<true>(cfg, cls, p) : launch_one<false>(cfg, cls, p);
  }
  if (err != cudaSuccess) return fail(std::string("ignis_trellis: decode launch: ") + cudaGetErrorString(err));
  return 0;
}

}  // namespace ignis_tl
