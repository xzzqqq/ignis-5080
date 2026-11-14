#pragma once
#include "ignis_trellis_linear.h"
#include <cuda_runtime.h>
#include <cstdint>
struct ignis_model;
struct ignis_seq_pool;
namespace ignis {
struct LabTrace;
void lab_prefill_position(ignis_model *, uint64_t);
void lab_graph_begin(ignis_model *, bool verify, uint32_t width);
void *lab_before(const ignis_model &, const ignis_trellis_weight &, const ignis_trellis_weight *,
                 const void *, uint32_t, void *, uint32_t, uint32_t, cudaStream_t);
void lab_after(const ignis_model &, void *, const void *, cudaStream_t);
}
extern "C" {
int32_t ignis_lab_capture_start(ignis_model *, ignis_seq_pool *, const char *, uint64_t, uint64_t, const char *);
int32_t ignis_lab_capture_stop(ignis_model *, ignis_seq_pool *);
int32_t ignis_lab_trace_context(ignis_model *, uint32_t request, uint32_t phase, uint64_t position);
int32_t ignis_lab_trace_flush(ignis_model *);
const char *ignis_lab_trace_error();
int32_t ignis_lab_capture_minimum(ignis_model *, const char *, uint64_t tokens, uint64_t requests, uint64_t *bytes);
struct ignis_lab_weight_view {
  const char *name;
  uint64_t trellis_offset, suh_offset, svh_offset;
  uint32_t k2, in, out, tile_stride;
};
enum ignis_lab_replay_check : uint32_t { IGNIS_LAB_CHECK_FIRST = 0, IGNIS_LAB_CHECK_EVERY = 1 };
int32_t ignis_lab_replay(ignis_model *, const ignis_lab_weight_view *, const ignis_lab_weight_view *,
    uint32_t tokens, uint32_t ldy, uint32_t epilogue, const void *x, const void *initial,
    const void *reference, uint32_t warmup, uint32_t samples, uint32_t check, float *times_ms, uint64_t *mismatches, uint32_t *checked);
}
