#pragma once
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
struct ignis_model;
namespace ignis {
bool gqa_candidate_applies(const ignis_model &, const ninfer::Tensor &, const ninfer::PagedKVBatchLayerView &);
uint64_t gqa_candidate_workspace_bytes(const ignis_model &, uint32_t max_keys);
void gqa_attention_dispatch(const ignis_model &, const ninfer::Tensor &q, const ninfer::Tensor &k,
    const ninfer::Tensor &v, const ninfer::Tensor &positions, const ninfer::Tensor &valid,
    const ninfer::Tensor &rows, const ninfer::Tensor &gate, float scale,
    ninfer::PagedKVBatchLayerView, ninfer::ops::GqaExecutionEnvelope, ninfer::DeviceArena &,
    ninfer::Tensor &out, cudaStream_t, void *replay_workspace=nullptr, uint64_t replay_workspace_bytes=0);
}
