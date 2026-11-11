// ignis kernel leaf - the 27B's trellis body in its layer programs -- OURS
// (spec 27b-16gb/05a, GitHub #339). See kernel/src/trellis_body.h.

#include "trellis_body.h"

#include "model_internal.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ignis::trellis_body {
namespace {

void require_activation(const ninfer::Tensor &x, uint32_t in, const char *op) {
  if (x.dtype != ninfer::DType::BF16 || x.ne[0] != static_cast<std::int32_t>(in) || x.ne[1] <= 0 ||
      x.ne[2] != 1 || x.ne[3] != 1 || !x.is_contiguous()) {
    throw std::invalid_argument(std::string(op) + ": the activation is not a BF16 [" + std::to_string(in) +
                                ", T] matrix");
  }
}

void check(const ignis_model &model, int32_t rc, const char *op) {
  if (rc != 0) {
    throw std::runtime_error(std::string(op) + ": " + (model.trellis_candidate ? model.trellis_candidate->api->last_error() : ignis_trellis_last_error()));
  }
}

}  // namespace

void linear(const ignis_model &model, const ignis_trellis_weight &w, const ninfer::Tensor &x, void *y,
            uint32_t ldy, uint32_t epilogue, cudaStream_t stream) {
  require_activation(x, w.in, "trellis linear");
  const auto call = model.trellis_candidate ? model.trellis_candidate->api->linear : ignis_trellis_linear;
  check(model, call(&w, x.data, static_cast<uint32_t>(x.ne[1]), y, ldy, epilogue,
                             &model.trellis_workspace, stream),
        "trellis linear");
}

void swiglu(const ignis_model &model, const ignis_trellis_weight &gate, const ignis_trellis_weight &up,
            const ninfer::Tensor &x, ninfer::Tensor &h, cudaStream_t stream) {
  require_activation(x, gate.in, "trellis swiglu");
  if (h.dtype != ninfer::DType::BF16 || h.ne[0] != static_cast<std::int32_t>(gate.out) || h.ne[1] != x.ne[1] ||
      !h.is_contiguous()) {
    throw std::invalid_argument("trellis swiglu: the output is not a BF16 [out, T] matrix");
  }
  const auto call = model.trellis_candidate ? model.trellis_candidate->api->swiglu : ignis_trellis_linear_swiglu;
  check(model, call(&gate, &up, x.data, static_cast<uint32_t>(x.ne[1]), h.data, gate.out,
                                    &model.trellis_workspace, stream),
        "trellis swiglu");
}

cudaError_t split_query_gate(const void *interleaved, int32_t heads, int32_t head_dim, int32_t tokens, void *query,
                             void *gate, cudaStream_t stream) {
  // One row per (token, head): `head_dim` BF16 values, read every 2 head_dim
  // from the interleaved rows -- the q half at offset 0, the gate half after
  // it -- and written densely.
  const std::size_t half = static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
  const std::size_t rows = static_cast<std::size_t>(heads) * static_cast<std::size_t>(tokens);
  const auto *src = static_cast<const std::uint8_t *>(interleaved);
  cudaError_t err = cudaMemcpy2DAsync(query, half, src, 2 * half, half, rows, cudaMemcpyDeviceToDevice, stream);
  if (err == cudaSuccess) {
    err = cudaMemcpy2DAsync(gate, half, src + half, 2 * half, half, rows, cudaMemcpyDeviceToDevice, stream);
  }
  return err;
}

void query_gate(const ignis_model &model, const ignis_trellis_weight &q_proj, const ninfer::Tensor &x,
                int32_t heads, int32_t head_dim, ninfer::DeviceArena &arena, ninfer::Tensor &query,
                ninfer::Tensor &gate, cudaStream_t stream) {
  const std::int32_t width = heads * head_dim;
  const std::int32_t tokens = x.ne[1];
  if (q_proj.out != static_cast<uint32_t>(2 * width) || query.ne[0] != width || gate.ne[0] != width ||
      query.ne[1] != tokens || gate.ne[1] != tokens) {
    throw std::invalid_argument("trellis query/gate: q_proj is not " + std::to_string(heads) + " heads of 2 x " +
                                std::to_string(head_dim) + " rows for this call's query and gate");
  }
  ninfer::Tensor interleaved = arena.alloc(ninfer::DType::BF16, {2 * width, tokens, 1, 1});
  linear(model, q_proj, x, interleaved.data, static_cast<uint32_t>(2 * width), IGNIS_TRELLIS_STORE, stream);
  const cudaError_t err = split_query_gate(interleaved.data, heads, head_dim, tokens, query.data, gate.data, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("trellis query/gate: cudaMemcpy2DAsync failed: ") + cudaGetErrorString(err));
  }
}

std::int32_t mlp_width(const ignis_model &model, const TrellisTail &tail, const ninfer::Weight &gate_up) {
  return model.is_trellis_body ? static_cast<std::int32_t>(tail.gate_proj.out) : gate_up.n / 2;
}

void output_and_mlp(const ignis_model &model, const TrellisTail &tail, const ninfer::Tensor &mixed,
                    const ninfer::Tensor &post_norm, ninfer::Tensor &residual, ninfer::Tensor &post,
                    ninfer::Tensor &fused, cudaStream_t stream) {
  const auto hidden = static_cast<uint32_t>(model.hidden);
  linear(model, tail.output, mixed, residual.data, hidden, IGNIS_TRELLIS_ADD, stream);
  ninfer::ops::rmsnorm(residual, post_norm, model.rms_norm_eps, /*unit_offset=*/true, post, stream);
  swiglu(model, tail.gate_proj, tail.up_proj, post, fused, stream);
  linear(model, tail.down, fused, residual.data, hidden, IGNIS_TRELLIS_ADD, stream);
}

void output_head(const ignis_model &model, const ninfer::Tensor &normalized, ninfer::Tensor &logits,
                 cudaStream_t stream) {
  if (!model.is_trellis_body) {
    ninfer::ops::linear(normalized, model.output_head, logits, stream);
    return;
  }
  if (logits.dtype != ninfer::DType::BF16 || logits.ne[0] != static_cast<std::int32_t>(model.trellis_head.out) ||
      logits.ne[1] != normalized.ne[1] || !logits.is_contiguous()) {
    throw std::invalid_argument("trellis output head: the logits are not a BF16 [vocab, N] matrix");
  }
  linear(model, model.trellis_head, normalized, logits.data, model.trellis_head.out, IGNIS_TRELLIS_STORE, stream);
}

}  // namespace ignis::trellis_body
