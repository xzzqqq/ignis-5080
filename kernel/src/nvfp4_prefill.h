#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

// One sequence, Qwen 27B geometry: Q/out [tokens,24,256], decoded K/V
// [4,band_span,256], all BF16. positions/valid are device pointers; valid
// is nullable (all tokens), otherwise its prefix count is clamped to [0,tokens].
// Bands are consecutive absolute key ranges [begin,end), begin divisible by64,
// 0 <= end-begin <= band_span <=16384. No query/HQ rotation is applied.
//
// Start at begin0, which initializes all carried rows without reading old scratch.
// For each intermediate band save_carry=true stores unnormalized FP32 acc/m/l;
// the final call normalizes to BF16 out. carry_acc is [tokens,24,256], carry_m/l
// are [24,tokens]. m stores the raw QK maximum (before scale). Empty bands preserve
// valid carried rows; invalid query rows always have neutral carry or +0 output.
// Q, positions, valid count and scale must remain unchanged across the bands.
// Positions are absolute nonnegative keys within the caller's validated context.
// All bands and any reuse/decode of K/V scratch must be ordered on the same stream.
// The caller owns storage/lifetime and must not alias Q/out/carry with K/V scratch.
struct IgnisNv4PrefillBand {
  const void *q, *k, *v;
  const std::int32_t *positions, *valid;
  void *out;
  float *carry_acc, *carry_m, *carry_l;
  std::int32_t tokens, band_begin, band_end, band_span;
  bool save_carry;
  float scale;
};

// Call before graph capture; sets the kernel's shared-memory opt-in attribute.
void ignis_nv4_prefill_prepare();
void ignis_nv4_prefill_band(const IgnisNv4PrefillBand &band, cudaStream_t stream);
