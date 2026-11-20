#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace ignis::nv4 {
inline constexpr int Sink = 32, Recent = 512, Rows = Sink + Recent;

// All offsets are relative to this layer. Fresh pointers address the first
// token of one lane. Ring pointers address the first slot of the cache view.
struct Window {
  __nv_bfloat16 *k{}, *v{};
  std::uint32_t *valid{};
  const __nv_bfloat16 *fresh_k{}, *fresh_v{};

  __device__ static int side_row(int pos) { return pos < Sink ? pos : Sink + pos % Recent; }

  __device__ bool overlaps(int tile, int width, int start) const {
    return k && (tile < Sink || tile + width > start - Recent);
  }

  __device__ const __nv_bfloat16 *exact(bool value, int slot, int head,
                                         int pos, int start, int end) const {
    if (!k || pos >= end) return nullptr;
    if (pos >= start)
      return (value ? fresh_v : fresh_k) + ((pos - start) * 4 + head) * 256;
    const int bit = pos % Recent;
    if (pos >= Sink && (pos < start - Recent ||
        !(valid[slot * (Recent / 32) + bit / 32] & (1u << (bit % 32))))) return nullptr;
    return (value ? v : k) + ((slot * Rows + side_row(pos)) * 4 + head) * 256;
  }
};
}
