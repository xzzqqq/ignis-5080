#include "ignis_nvfp4_kv.h"
#include "nvfp4_codec.cuh"
#include "nvfp4_prefill.h"
#include "nvfp4_xqa.h"
#include "core/device.h"
#include "ninfer/ops/sigmoid_mul.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ignis {
namespace {
constexpr int D = 256, Heads = 4, QHeads = 24, Page = 64;
struct Cache {
  std::uint8_t *k, *v, *ks, *vs;
  const std::int32_t *tables;
  int stride, pages, rows;
};
Cache device_cache(const ninfer::PagedKVBatchLayerView &c) {
  if (c.k_pages.dtype!=ninfer::DType::U8 || c.v_pages.dtype!=ninfer::DType::U8 ||
      c.k_scale_pages.dtype!=ninfer::DType::U8 || c.v_scale_pages.dtype!=ninfer::DType::U8 ||
      c.block_tables.dtype!=ninfer::DType::I32 || !c.k_pages.is_contiguous() || !c.v_pages.is_contiguous() ||
      !c.k_scale_pages.is_contiguous() || !c.v_scale_pages.is_contiguous() ||
      !c.block_tables.is_contiguous() || !nvfp4_cache(c) || c.k_pages.ne[0] != 128 || c.v_pages.ne[0] != 128 ||
      c.k_scale_pages.ne[0] != 16 || c.v_scale_pages.ne[0] != 16 ||
      c.k_pages.ne[1] != Page || c.k_pages.ne[2] != Heads ||
      c.v_pages.ne[1] != Page || c.v_pages.ne[2] != Heads ||
      c.k_scale_pages.ne[1] != Page || c.k_scale_pages.ne[2] != Heads ||
      c.v_scale_pages.ne[1] != Page || c.v_scale_pages.ne[2] != Heads ||
      c.k_pages.ne[3] != c.v_pages.ne[3] || c.k_pages.ne[3] != c.k_scale_pages.ne[3] ||
      c.k_pages.ne[3] != c.v_scale_pages.ne[3])
    throw std::invalid_argument("nvfp4 KV: invalid four-plane geometry");
  return {static_cast<std::uint8_t *>(c.k_pages.data), static_cast<std::uint8_t *>(c.v_pages.data),
          static_cast<std::uint8_t *>(c.k_scale_pages.data), static_cast<std::uint8_t *>(c.v_scale_pages.data),
          static_cast<const std::int32_t *>(c.block_tables.data), int(c.block_tables.ne[0]),
          int(c.k_pages.ne[3]), int(c.block_tables.ne[1])};
}
__device__ int count(const std::int32_t *valid, int lane, int tokens) {
  return valid ? max(0, min(tokens, valid[lane])) : tokens;
}
__device__ void bump(std::uint64_t *stats, int i, unsigned n) {
  if (stats && n) atomicAdd(reinterpret_cast<unsigned long long *>(stats + i),
                           static_cast<unsigned long long>(n));
}
__global__ void append(Cache c, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
                        const std::int32_t *positions, const std::int32_t *valid,
                        const std::int32_t *rows, int tokens, std::uint64_t *stats) {
  const int token = blockIdx.x % tokens, lane = blockIdx.x / tokens;
  if (token >= count(valid, lane, tokens)) return;
  const int head = blockIdx.y, pair = threadIdx.x;
  const bool value = blockIdx.z != 0;
  const int pos = positions[lane * tokens + token], row = rows[lane];
  if (pos != positions[lane * tokens] + token) asm volatile("trap;");
  if (pos < 0 || pos / Page >= c.stride || row < 0 || row >= c.rows) asm volatile("trap;");
  const int physical = c.tables[row * c.stride + pos / Page];
  if (physical < 0 || physical >= c.pages) asm volatile("trap;");
  const auto i = (std::int64_t(lane * tokens + token) * Heads + head) * D + pair * 2;
  const float x = __bfloat162float((value ? v : k)[i]);
  const float y = __bfloat162float((value ? v : k)[i + 1]);
  const unsigned invalid = unsigned(!isfinite(x)) + unsigned(!isfinite(y));
  bump(stats, 0, invalid);
  if (invalid) asm volatile("trap;");
  float amax = fmaxf(fabsf(x), fabsf(y));
#pragma unroll
  for (int step = 1; step < 8; step <<= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, step, 8));
  // Unlike scratch prefix calibration, fixed global1 has a stable, explicit
  // representable domain. Refuse nonfinite/out-of-range activation groups rather
  // than silently changing already-written history or clipping an unseen outlier.
  if (amax > 2688.0f) {
    if ((pair & 7) == 0) bump(stats, 1, 1);
    asm volatile("trap;");
  }
  const std::uint8_t sf = __nv_cvt_float_to_fp8(__fdiv_rn(amax, 6.0f), __NV_SATFINITE, __NV_E4M3);
  const float factor = nv4::scale_float(sf);
  const float nx = factor > 0 ? __fdiv_rn(x, factor) : 0.0f;
  const float ny = factor > 0 ? __fdiv_rn(y, factor) : 0.0f;
  const auto packed = __nv_cvt_float2_to_fp4x2(make_float2(nx, ny), __NV_E2M1, cudaRoundNearest);
  const auto dst = (std::int64_t(physical) * Heads + head) * Page + pos % Page;
  (value ? c.v : c.k)[dst * 128 + pair] = packed;
  if ((pair & 7) == 0) {
    (value ? c.vs : c.ks)[dst * 16 + pair / 8] = sf;
    bump(stats, 3, sf == 0);
  }
  bump(stats, 2, unsigned(fabsf(nx) > 6.0f) + unsigned(fabsf(ny) > 6.0f));
}
__global__ void decode_band(Cache c, const std::int32_t *rows, const std::int32_t *positions,
                             const std::int32_t *valid, int lane, int tokens,
                             int begin, int span, __nv_bfloat16 *k, __nv_bfloat16 *v) {
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat >= Heads * span * (D / 8)) return;
  const int d = (flat % 32) * 8, token = (flat / 32) % span, head = flat / (32 * span);
  const int n = count(valid, lane, tokens);
  const int window = n > 0 ? positions[lane * tokens + n - 1] + 1 : 0;
  if (!n || begin >= window) return; // Empty bands are not read by the carry kernel.
  const int absolute = begin + token;
  __nv_bfloat16 *ko = k + std::int64_t(flat) * 8, *vo = v + std::int64_t(flat) * 8;
  if (absolute >= window) {
    *reinterpret_cast<int4 *>(ko) = make_int4(0, 0, 0, 0);
    *reinterpret_cast<int4 *>(vo) = make_int4(0, 0, 0, 0);
    return;
  }
  const int row = rows[lane];
  if (row < 0 || row >= c.rows || absolute / Page >= c.stride) asm volatile("trap;");
  const int physical = c.tables[row * c.stride + absolute / Page];
  if (physical < 0 || physical >= c.pages) asm volatile("trap;");
  const auto src = (std::int64_t(physical) * Heads + head) * Page + absolute % Page;
  nv4::load8(c.k + src * 128, c.ks + src * 16, d, ko);
  nv4::load8(c.v + src * 128, c.vs + src * 16, d, vo);
}
__global__ void gate_output(__nv_bfloat16 *out, const __nv_bfloat16 *gate,
                            const std::int32_t *valid, int tokens, int batch) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tokens * batch * QHeads * D) return;
  const int token = i / (QHeads * D), lane = token / tokens;
  if (token % tokens >= count(valid, lane, tokens)) { out[i] = __float2bfloat16(0.0f); return; }
  if (gate) {
    const float g = __bfloat162float(gate[i]);
    const float sigmoid = 1.0f / (1.0f + expf(-g));
    out[i] = __float2bfloat16_rn(__bfloat162float(out[i]) * sigmoid);
  }
}
std::size_t aligned(std::size_t n) { return (n + 255) & ~std::size_t(255); }
}
bool nvfp4_cache(const ninfer::PagedKVBatchLayerView &c) {
  return c.dtype == ninfer::DType::U8 && c.quant_group == 16 && c.head_dim == D && c.num_kv_heads == Heads;
}
void nvfp4_append(const ninfer::Tensor &k, const ninfer::Tensor &v,
                  const ninfer::Tensor &positions, const ninfer::Tensor &valid,
                  const ninfer::Tensor &rows, ninfer::PagedKVBatchLayerView cache,
                  cudaStream_t stream, std::uint64_t *stats) {
  const int tokens = int(k.ne[2]), batch = int(k.ne[3]);
  if (!k.data || !v.data || !positions.data || !rows.data || tokens < 1 || batch < 1 ||
      k.dtype != ninfer::DType::BF16 || v.dtype != ninfer::DType::BF16 ||
      !k.is_contiguous() || !v.is_contiguous() || !positions.is_contiguous() ||
      positions.dtype!=ninfer::DType::I32 || positions.ne[0]!=tokens || positions.ne[1]!=batch ||
      rows.dtype!=ninfer::DType::I32 || !rows.is_contiguous() || rows.ne[0]!=batch ||
      (valid.data && (valid.dtype!=ninfer::DType::I32 || !valid.is_contiguous() || valid.ne[0]!=batch)) ||
      k.ne[0] != D || k.ne[1] != Heads || v.ne[0] != k.ne[0] || v.ne[1] != k.ne[1] || v.ne[2] != k.ne[2] || v.ne[3] != k.ne[3])
    throw std::invalid_argument("nvfp4 append: invalid tensor geometry");
  append<<<dim3(tokens * batch, Heads, 2), 128, 0, stream>>>(device_cache(cache),
      static_cast<const __nv_bfloat16 *>(k.data), static_cast<const __nv_bfloat16 *>(v.data),
      static_cast<const std::int32_t *>(positions.data), static_cast<const std::int32_t *>(valid.data),
      static_cast<const std::int32_t *>(rows.data), tokens, stats);
  CUDA_CHECK(cudaGetLastError());
}
void nvfp4_decode_band(ninfer::PagedKVBatchLayerView cache,
                       const std::int32_t *rows, const std::int32_t *positions,
                       const std::int32_t *valid, int lane, int tokens, int begin, int span,
                       void *k, void *v, cudaStream_t stream) {
  if (!rows || !positions || !k || !v || tokens < 1 || lane < 0 || begin < 0 ||
      begin % Page || span < 1 || span % Page)
    throw std::invalid_argument("nvfp4 band: invalid span");
  decode_band<<<(Heads * span * 32 + 255) / 256, 256, 0, stream>>>(device_cache(cache),
      rows, positions, valid, lane, tokens, begin, span,
      static_cast<__nv_bfloat16 *>(k), static_cast<__nv_bfloat16 *>(v));
  CUDA_CHECK(cudaGetLastError());
}
std::size_t nvfp4_attention_workspace_bytes(std::uint32_t max_keys, int tokens, int batch) {
  if (!max_keys || max_keys > 524288 || tokens < 1 || batch < 1) return 0;
  const auto span = std::min<std::uint32_t>(kNv4BandKeys, (max_keys + 63) / 64 * 64);
  const std::size_t fallback = 2 * aligned(std::size_t(span) * Heads * D * 2) +
      aligned(std::size_t(tokens) * QHeads * D * 4) +
      2 * aligned(std::size_t(tokens) * QHeads * 4);
    // This is also the max-width capacity bound used by model planning: a
  // width9 fallback query must still cover the width8 XQA below it.
  return std::max(fallback, aligned(ignis_nv4_xqa_workspace(max_keys)));
}
void nvfp4_prepare() { ignis_nv4_xqa_prepare(); ignis_nv4_prefill_prepare(); }
void nvfp4_attention(const ninfer::Tensor &q, const ninfer::Tensor &k,
                     const ninfer::Tensor &v, const ninfer::Tensor &positions,
                     const ninfer::Tensor &valid, const ninfer::Tensor &rows,
                     const ninfer::Tensor &gate, float scale,
                     ninfer::PagedKVBatchLayerView cache, ninfer::ops::GqaExecutionEnvelope envelope,
                     ninfer::DeviceArena &workspace, ninfer::Tensor &out, cudaStream_t stream) {
  const int tokens = int(q.ne[2]), batch = int(q.ne[3]);
  const auto needed = nvfp4_attention_workspace_bytes(envelope.max_visible_keys, tokens, batch);
  if (!needed || !q.data || !out.data || !positions.data || !rows.data ||
      q.dtype!=ninfer::DType::BF16 || out.dtype!=ninfer::DType::BF16 ||
      !q.is_contiguous() || !out.is_contiguous() ||
      !positions.is_contiguous() || positions.dtype!=ninfer::DType::I32 ||
      positions.ne[0]!=tokens || positions.ne[1]!=batch ||
      rows.dtype!=ninfer::DType::I32 || !rows.is_contiguous() || rows.ne[0]!=batch ||
      (valid.data && (valid.dtype!=ninfer::DType::I32 || !valid.is_contiguous() || valid.ne[0]!=batch)) ||
      (gate.data && (gate.dtype!=ninfer::DType::BF16 || !gate.is_contiguous() ||
       gate.ne[0]!=D || gate.ne[1]!=QHeads || gate.ne[2]!=tokens || gate.ne[3]!=batch)) ||
      out.ne[0]!=D || out.ne[1]!=QHeads || out.ne[2]!=tokens || out.ne[3]!=batch ||
      !std::isfinite(scale) || scale<=0 ||
      k.ne[2]!=tokens || k.ne[3]!=batch ||
      q.ne[0] != D || q.ne[1] != QHeads || !nvfp4_cache(cache))
    throw std::invalid_argument("nvfp4 attention: unsupported geometry or envelope");
  nvfp4_append(k, v, positions, valid, rows, cache, stream);
  if (tokens <= 8) {
    const auto ws = workspace.alloc_bytes(ignis_nv4_xqa_workspace(envelope.max_visible_keys));
    // Rectangular lanes are replayed serially with one shared workspace.
    // Nominal M48/T8 shape stays fixed; qCuSeqLens carries actual T/valid.
    for(int lane=0;lane<batch;++lane){
      const auto q_offset=std::int64_t(lane)*tokens*QHeads*D;
      const IgnisNv4XqaCall call{
        static_cast<const __nv_bfloat16 *>(q.data)+q_offset,
        static_cast<__nv_bfloat16 *>(out.data)+q_offset,
        cache.k_pages.data,cache.v_pages.data,cache.k_scale_pages.data,cache.v_scale_pages.data,
        static_cast<const std::int32_t *>(cache.block_tables.data),
        static_cast<const std::int32_t *>(rows.data)+lane,
        static_cast<const std::int32_t *>(positions.data)+lane*tokens,
        valid.data ? static_cast<const std::int32_t *>(valid.data)+lane : nullptr,
        std::uint32_t(cache.block_tables.ne[0]),std::uint32_t(cache.block_tables.ne[1]),
        std::uint32_t(cache.k_pages.ne[3]),envelope.max_visible_keys,std::uint32_t(tokens),scale};
      ignis_nv4_xqa_run(call,ws.data,ws.bytes,stream);
    }
  } else {
    const int span = int(std::min<std::uint32_t>(kNv4BandKeys,
        (envelope.max_visible_keys + 63) / 64 * 64));
    const auto kb = workspace.alloc(ninfer::DType::BF16, {D, span, Heads});
    const auto vb = workspace.alloc(ninfer::DType::BF16, {D, span, Heads});
    const auto ca = workspace.alloc(ninfer::DType::FP32, {D, QHeads, tokens});
    const auto cm = workspace.alloc(ninfer::DType::FP32, {tokens, QHeads});
    const auto cl = workspace.alloc(ninfer::DType::FP32, {tokens, QHeads});
    for (int lane = 0; lane < batch; ++lane) {
      for (std::uint32_t begin = 0; begin < envelope.max_visible_keys; begin += span) {
        nvfp4_decode_band(cache, static_cast<const std::int32_t *>(rows.data),
            static_cast<const std::int32_t *>(positions.data), static_cast<const std::int32_t *>(valid.data),
            lane, tokens, int(begin), span, kb.data, vb.data, stream);
        const IgnisNv4PrefillBand call{static_cast<const __nv_bfloat16 *>(q.data) +
            std::int64_t(lane) * tokens * QHeads * D, kb.data, vb.data,
            static_cast<const std::int32_t *>(positions.data) + lane * tokens,
            valid.data ? static_cast<const std::int32_t *>(valid.data) + lane : nullptr,
            static_cast<__nv_bfloat16 *>(out.data) + std::int64_t(lane) * tokens * QHeads * D,
            static_cast<float *>(ca.data), static_cast<float *>(cm.data), static_cast<float *>(cl.data),
            tokens, int(begin), int(std::min<std::uint32_t>(begin + span, envelope.max_visible_keys)), span,
            begin + span < envelope.max_visible_keys, scale};
        ignis_nv4_prefill_band(call, stream);
      }
    }
  }
  gate_output<<<(tokens * batch * QHeads * D + 255) / 256, 256, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(out.data), static_cast<const __nv_bfloat16 *>(gate.data),
      static_cast<const std::int32_t *>(valid.data), tokens, batch);
  CUDA_CHECK(cudaGetLastError());
}
}
