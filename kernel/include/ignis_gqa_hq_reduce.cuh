#pragma once
#include "ops/kernel/gqa_attention_decode.cuh"
// SPDX-License-Identifier: Apache-2.0
// Adapted from gpillon/ninfer, commit a00648cb828457986cf5b4b4f712b4cbcd7af0d1,
// src/ops/kernel/gqa_attention_decode.cuh. See kernel/NOTICE and
// kernel/vendor/LICENSE for the upstream lineage and Apache-2.0 terms.
// This owned specialization retains the original FP32 reduction tree, shares
// identical per-split weights, and reduces all 256 output dimensions per CTA.
// The unchanged generic reducer is its independent exact oracle (ADR 0031).
namespace ninfer::ops {
inline constexpr int kIgnisGqaHqReduceThreads = 256;
inline constexpr int kIgnisGqaHqReduceDChunk = 256;
template <typename Geometry, int DChunk, bool Int8, bool MultiBatch, bool Masked, bool Offset>
__launch_bounds__(kIgnisGqaHqReduceThreads) __global__ void ignis_gqa_hq_reduce_kernel(
    const __nv_bfloat16* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns,
    const __nv_bfloat16* gate, std::int32_t tokens, std::int32_t full_width,
    std::int32_t column_begin, std::int32_t batch_size, std::int32_t split_count,
    __nv_bfloat16* out) {
    static_assert(DChunk == kIgnisGqaHqReduceDChunk && Geometry::QHeads == 24 && Geometry::KVHeads == 4);
    // Private HQ27 specialization: launcher and tests both use exactly 256 threads.
    static_assert(!Int8);
    pdl::sync();

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (Offset) { positions += column_begin; }
    if constexpr (MultiBatch) { positions += batch * full_width; }
    const int last_pos = positions[tokens - 1];
    int output_column  = token;
    if constexpr (Offset) { output_column += column_begin; }
    if constexpr (MultiBatch) { output_column += batch * full_width; }

    if constexpr (MultiBatch) {
        const std::int64_t partial_acc_row = static_cast<std::int64_t>(batch) * kGqaHeadDim *
                                             Geometry::QHeads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m += partial_stat_row;
        partial_l += partial_stat_row;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, Int8>(window, split_count, tokens);

    __shared__ float reduce[kIgnisGqaHqReduceThreads];
    __shared__ float split_weights[Geometry::DecodeSplits];

    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_split_count; split += kIgnisGqaHqReduceThreads) {
        local_m = fmaxf(local_m,
                        partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)]);
    }
    reduce[tid] = local_m;
    __syncthreads();

    for (int stride = kIgnisGqaHqReduceThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]); }
        __syncthreads();
    }
    const float head_m = reduce[0];
    __syncthreads();

    if (head_m == -CUDART_INF_F) {
        const int d = d_start + tid;
        if (tid < DChunk && d < kGqaHeadDim) {
            out[gqa_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(0.0f);
        }
        return;
    }

    float local_l = 0.0f;
    for (int split = tid; split < active_split_count; split += kIgnisGqaHqReduceThreads) {
        const float tile_l =
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
        // Compute even for nonpositive/NaN l: the original numerator uses
        // a different <=0 predicate. FP32 sharing adds no rounding.
        const float weight = expf(
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] - head_m);
        if (split < Geometry::DecodeSplits) { split_weights[split] = weight; }
        if (tile_l > 0.0f) { local_l += tile_l * weight; }
    }
    reduce[tid] = local_l;
    __syncthreads();

    for (int stride = kIgnisGqaHqReduceThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] += reduce[tid + stride]; }
        __syncthreads();
    }
    const float head_l = reduce[0];

    const int d = d_start + tid;
    if (tid >= DChunk || d >= kGqaHeadDim) { return; }

    float numerator = 0.0f;
    if (head_l > 0.0f) {
        if (active_split_count <= Geometry::DecodeSplits) {
            for (int split = 0; split < active_split_count; ++split) {
                const float tile_l =
                    partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
                if (tile_l <= 0.0f) { continue; }
                const float weight = split_weights[split];
                numerator +=
                    __bfloat162float(
                        partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)]) *
                    weight;
            }
        } else {
            for (int split = 0; split < active_split_count; ++split) {
                const float tile_l =
                    partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
                if (tile_l <= 0.0f) { continue; }
                const float weight = expf(
                    partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] - head_m);
                numerator +=
                    __bfloat162float(
                        partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)]) *
                    weight;
            }
        }
    }
    bool valid = true;
    if constexpr (Masked) {
        int absolute_column = token;
        if constexpr (Offset) { absolute_column += column_begin; }
        valid = absolute_column < valid_columns[batch];
    }
    const float value = (valid && head_l > 0.0f) ? numerator / head_l : 0.0f;
    const std::int64_t output_index = gqa_q_index<Geometry>(q_head, d, output_column);
    const __nv_bfloat16 attention   = __float2bfloat16(value);
    out[output_index]               = __float2bfloat16_rn(
        __bfloat162float(attention) * sigmoid(__bfloat162float(gate[output_index])));
    pdl::publish();
}

} // namespace ninfer::ops
