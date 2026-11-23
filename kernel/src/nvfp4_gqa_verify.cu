// Native Trellis C1/T8 token mixer for the 27B NVFP4 cache.
#include "layer_internal.h"
#include "ignis_nvfp4_kv.h"
#include "lab_profile.h"
#include "lab_segment.h"
#include "nvfp4_xqa.h"
#include "ops/kernel/rope.cuh"
#include "ninfer/ops/rmsnorm.h"
#include "trellis_decode_helpers.cuh"
#include "core/device.h"
#include "core/pdl.cuh"
#include <stdexcept>

extern "C" int32_t ignis_gqa_native_q_fixed_splits(const ignis_trellis_weight *, const void *, void *,
                                                    const ignis_trellis_workspace *, void *);
extern "C" int32_t ignis_gqa_native_kv_pair(const ignis_trellis_weight *, const ignis_trellis_weight *,
                                            const void *, void *, void *, const ignis_trellis_workspace *, void *);
extern "C" int32_t ignis_gdn_rotation_params(const ignis_trellis_weight *, const ignis_trellis_workspace *,
                                            ignis_tl::RotateParams *);
extern "C" int32_t ignis_gdn_output_from(const ignis_trellis_weight *, const void *, const void *, void *,
                                         const ignis_trellis_workspace *, void *);

namespace {
constexpr int Tokens = 8, Heads = 24, KvHeads = 4, Dim = 256, Hidden = 5120;
constexpr int Query = Heads * Dim, Blocks = Query / 128;

void check_linear(int rc) {
  if (rc) throw std::runtime_error(ignis_trellis_last_error());
}

__global__ void gated_rotation(const __nv_bfloat16 *attention, const __nv_bfloat16 *interleaved,
                               const std::int32_t *valid, const __grid_constant__ ignis_tl::RotateParams rotation) {
  ninfer::pdl::sync();
  ninfer::pdl::trigger_dependents();
  const int row = blockIdx.x * 4 + threadIdx.x / 32;
  const int token = row / Blocks, block = row % Blocks, col = block * 128 + (threadIdx.x % 32) * 4;
  float values[4]{};
  if (token < valid[0]) {
    const int gate_index = (token * Heads + col / Dim) * (2 * Dim) + Dim + col % Dim;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float g = __bfloat162float(interleaved[gate_index + j]);
      values[j] = __bfloat162float(attention[token * Query + col + j]) * (1.0f / (1.0f + expf(-g)));
    }
  }
  // rotate_bf16_block rounds the sigmoid product to BF16 before rotation.
  ignis_tl::rotate_bf16_block(rotation, values, block, token);
}
}

void ignis_nvfp4_gqa_verify(ignis_model &model, ignis_seq_pool &pool, uint32_t layer,
                            const ninfer::Tensor &input, ninfer::Tensor &output, void *round_metadata) {
  using ninfer::DType;
  using ninfer::Tensor;
  ignis::LabProfileRange profile(layer, "gqa");
  auto &arena = *model.decode_graph_scratch;
  auto scope = arena.scope();
  auto &w = model.layers[layer].gqa;
  auto &ws = model.trellis_workspace;
  const auto stream = model.stream;
  const auto ordinal = ignis_gqa_relative_layer(layer);
  auto cache = ignis_kv_batch_layer_view(&pool, ordinal);
  auto &verify = *model.verify;
  const Tensor positions(verify.positions->p, DType::I32, {Tokens, 1});
  const Tensor valid(verify.valid_columns->p, DType::I32, {1});
  const Tensor slots(model.decode_graph_slots->p, DType::I32, {1});
  const auto *rope_pos = static_cast<const int32_t *>(verify.rope_positions ? verify.rope_positions->p : positions.data);
  const auto norm = [](const ninfer::Weight &weight) {
    return Tensor(const_cast<void *>(weight.qdata), DType::BF16, {Hidden});
  };
  {
    auto mixer_scope = arena.scope();
    auto normalized = arena.alloc(DType::BF16, {Hidden, Tokens});
    auto q_gate = arena.alloc(DType::BF16, {2 * Query, Tokens});
    auto kv = arena.alloc(DType::BF16, {Dim, KvHeads, Tokens, 2});
    Tensor key(kv.data, DType::BF16, {Dim, KvHeads, Tokens});
    Tensor value(static_cast<__nv_bfloat16 *>(kv.data) + Dim * KvHeads * Tokens,
                 DType::BF16, {Dim, KvHeads, Tokens});
    auto rq = arena.alloc(DType::BF16, {Dim, Heads, Tokens});
    auto rk = arena.alloc(DType::BF16, {Dim, KvHeads, Tokens});
    auto attention = arena.alloc(DType::BF16, {Dim, Heads, Tokens});
    const auto scratch = arena.alloc_bytes(ignis_nv4_xqa_workspace(model.max_context_tokens));
    IGNIS_PROFILE_OP("rmsnorm", ninfer::ops::rmsnorm(input, norm(w.input_norm), model.rms_norm_eps, true, normalized, stream));
    IGNIS_PROFILE_OP("linear.q_proj", check_linear(ignis_gqa_native_q_fixed_splits(
        &w.trellis.q_proj, normalized.data, q_gate.data, &ws, stream)));
    IGNIS_PROFILE_OP("linear.kv_pair", check_linear(ignis_gqa_native_kv_pair(
        &w.trellis.k_proj, &w.trellis.v_proj, normalized.data, key.data, value.data, &ws, stream)));
    CUDA_CHECK(ninfer::pdl::launch_dependent({dim3((Heads + KvHeads) * Tokens / 4), dim3(128), 0, stream},
        ninfer::ops::rope_norm_fused_kernel<ninfer::ops::RopeKernelMode::Text1D, Heads, KvHeads, 128, false, 2>,
        static_cast<const __nv_bfloat16 *>(q_gate.data), static_cast<const __nv_bfloat16 *>(key.data),
        static_cast<const __nv_bfloat162 *>(w.query_norm.qdata), static_cast<const __nv_bfloat162 *>(w.key_norm.qdata),
        model.rms_norm_eps, rope_pos, model.text_rope, static_cast<__nv_bfloat16 *>(rq.data),
        static_cast<__nv_bfloat16 *>(rk.data), Tokens));
    IGNIS_PROFILE_OP("gqa_attention", ignis::nvfp4_verify_attention(rq, rk, value, positions, valid, slots, cache,
        model.max_context_tokens, attention, scratch, round_metadata, ordinal == 0, ordinal + 1 == pool.kv_num_layers, stream));
    ignis_tl::RotateParams rotation{};
    check_linear(ignis_gdn_rotation_params(&w.trellis.tail.output, &ws, &rotation));
    CUDA_CHECK(ninfer::pdl::launch_dependent({dim3(Blocks * Tokens / 4), dim3(128), 0, stream}, gated_rotation,
        static_cast<const __nv_bfloat16 *>(attention.data), static_cast<const __nv_bfloat16 *>(q_gate.data),
        static_cast<const int32_t *>(valid.data), rotation));
    IGNIS_PROFILE_OP("linear.output", check_linear(ignis_gdn_output_from(
        &w.trellis.tail.output, nullptr, input.data, output.data, &ws, stream)));
  }
  auto post = arena.alloc(DType::BF16, {Hidden, Tokens});
  auto fused = arena.alloc(DType::BF16, {static_cast<int32_t>(w.trellis.tail.gate_proj.out), Tokens});
  ignis::segment_mlp(model, layer, norm(w.post_attention_norm), output, post, fused, stream);
}
