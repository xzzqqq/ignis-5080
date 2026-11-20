#pragma once
// Ignis-owned codec. NVFP4 storage, not FP4 MMA. Fixed global1 means no mutable
// calibration state across chunks, appends, COW, rollback or snapshot/restore.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace ignis::nv4 {
__device__ __forceinline__ float scale_float(std::uint8_t bits) {
  const __half h = __nv_cvt_fp8_to_halfraw(bits, __NV_E4M3);
  return __half2float(h);
}
__device__ __forceinline__ void load8(const std::uint8_t *codes,
                                     const std::uint8_t *scales, int d,
                                     __nv_bfloat16 *out) {
  const std::uint32_t packed = *reinterpret_cast<const std::uint32_t *>(codes + d / 2);
  const float sf = scale_float(scales[d / 16]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2 h = __nv_cvt_fp4x2_to_halfraw2(std::uint8_t(packed >> (8 * i)), __NV_E2M1);
    const float2 x = __half22float2(h);
    out[2 * i] = __float2bfloat16_rn(__fmul_rn(x.x, sf));
    out[2 * i + 1] = __float2bfloat16_rn(__fmul_rn(x.y, sf));
  }
}
}
