#pragma once
// NVFP4 KV storage: adjacent E2M1 nibbles, E4M3 G16 scales, fixed global1.
// The attention math uses BF16 Tensor Core inputs and FP32 accumulation.
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include <cstddef>
#include <cstdint>
namespace ignis {
inline constexpr std::uint32_t kNv4BandKeys = 16384;
inline constexpr float kNv4Global = 1.0f;
bool nvfp4_cache(const ninfer::PagedKVBatchLayerView &cache);
std::size_t nvfp4_attention_workspace_bytes(std::uint32_t max_keys, std::int32_t tokens,
                                            std::int32_t batch = 1);
void nvfp4_prepare();
// Native C1/T8 verify with an exact BF16 window. The enclosing round owns
// round_metadata across all layers; workspace is private to this layer.
void nvfp4_verify_attention(const ninfer::Tensor &q, const ninfer::Tensor &k, const ninfer::Tensor &v,
                            const ninfer::Tensor &positions, const ninfer::Tensor &valid,
                            const ninfer::Tensor &rows, const ninfer::PagedKVBatchLayerView &cache,
                            std::uint32_t max_keys, const ninfer::Tensor &out, ninfer::DeviceSpan workspace,
                            void *round_metadata, bool first, bool last, cudaStream_t stream);
void nvfp4_store_window(const ninfer::Tensor &k, const ninfer::Tensor &v,
                        const ninfer::Tensor &positions, const ninfer::Tensor &valid,
                        const ninfer::Tensor &rows, ninfer::PagedKVBatchLayerView cache,
                        bool publish, cudaStream_t stream);
// T<=8 uses variable-valid M48 per lane; wider calls use FP32-carry bands.
void nvfp4_attention(const ninfer::Tensor &q, const ninfer::Tensor &k,
                     const ninfer::Tensor &v, const ninfer::Tensor &positions,
                     const ninfer::Tensor &valid, const ninfer::Tensor &rows,
                     const ninfer::Tensor &gate, float scale,
                     ninfer::PagedKVBatchLayerView cache,
                     ninfer::ops::GqaExecutionEnvelope envelope,
                     ninfer::DeviceArena &workspace, ninfer::Tensor &out, cudaStream_t stream);
// Tests use the actual production scatter/decode entry points and optional counters:
// nonfinite values, out-of-range groups, endpoint codes, zero-SF groups.
// Active positions are a contiguous prefix per lane; invalid suffix inputs are not read.
void nvfp4_append(const ninfer::Tensor &k, const ninfer::Tensor &v,
                  const ninfer::Tensor &positions, const ninfer::Tensor &valid,
                  const ninfer::Tensor &rows, ninfer::PagedKVBatchLayerView cache,
                  cudaStream_t stream, std::uint64_t *stats = nullptr);
void nvfp4_decode_band(ninfer::PagedKVBatchLayerView cache,
                       const std::int32_t *rows, const std::int32_t *positions,
                       const std::int32_t *valid, std::int32_t lane, std::int32_t tokens,
                       std::int32_t begin, std::int32_t span, void *k, void *v,
                       cudaStream_t stream);
}
