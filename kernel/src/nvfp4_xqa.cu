// FlashInfer XQA v0.7.2 adapted for Ignis NVFP4 paged storage and graph replay.
// Source, Apache-2.0 license, hashes and recorded patch: third_party/flashinfer_xqa.
// Values are FP4 in memory; QK/PV execute BF16 Tensor Core math, not FP4 MMA.
#include "nvfp4_xqa.h"
#include "csrc/xqa/mha.cu"
#include "core/device.h"
#include <algorithm>
#include <stdexcept>

namespace {
static_assert(rowsPerBlock==48 && nbValidRows==48 && headGrpSize==6 && headElems==256,
              "private XQA workspace/semantics require M48, T8, group6, D256");
constexpr std::uint32_t SplitCapacity = 21;
using ScratchSlice = Array2D<LdGrain, nbValidRows, SharedMem::XSmemBuffer::cols>;
using ScratchBlock = Vec<ScratchSlice, gemm1WarpsPerGrp * nbHeadSplits>;
constexpr std::size_t ScratchAlignment = sizeof(ScratchBlock);
static_assert(ScratchAlignment % sizeof(SMemWarpRowMax) == 0);
std::uint32_t multiprocessors = 0;
std::size_t align256(std::size_t n) { return (n + 255) & ~std::size_t(255); }
struct Metadata {
  std::uint32_t seq_len, cu[2], mask[8];
};
__global__ void metadata(IgnisNv4XqaCall call, Metadata *m, std::int32_t *table) {
  const int valid = call.valid ? max(0, min(int(call.tokens), call.valid[0])) : int(call.tokens);
  const int row = valid ? call.table_rows[0] : 0;
  if (valid && (row < 0 || row >= call.table_count)) asm volatile("trap;");
  const int window = valid ? call.positions[valid - 1] + 1 : 0;
  if (window < valid || window > call.max_keys) asm volatile("trap;");
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 8) {
    if (i < valid && call.positions[i] != window - valid + i) asm volatile("trap;");
    m->mask[i] = (1u << (i + 1)) - 1;
  }
  if (i == 0) { m->seq_len = window; m->cu[0] = 0; m->cu[1] = valid; }
  const int pages = (call.max_keys + 63) / 64;
  if (i < pages && valid) {
    const int physical = call.tables[row * call.table_stride + i];
    if (i * 64 < window && (physical < 0 || physical >= call.physical_pages)) asm volatile("trap;");
    table[i] = physical;
  }
}
std::size_t partial_bytes() {
  const std::size_t count = SplitCapacity * 4;
  // MemSegmenter::newSeg aligns to sizeof(T), including non-power-of-two
  // ScratchTile sizes. Match that layout, not just CUDA allocation alignment.
  Segmenter<std::uint64_t> segments;
  segments.newSeg<SMemWarpRowMax>(count, sizeof(SMemWarpRowMax));
  segments.newSeg<SMemWarpRowMax>(count, sizeof(SMemWarpRowMax));
  segments.newSeg<ScratchBlock>(count, sizeof(ScratchBlock));
  return std::size_t(segments.getEndOffset());
}
}
std::size_t ignis_nv4_xqa_workspace(std::uint32_t max_keys) {
  if (!max_keys || max_keys > 524288) return 0;
  return 256 + align256(std::size_t((max_keys + 63) / 64) * 4) + 256 + (ScratchAlignment - 1) + partial_bytes();
}
void ignis_nv4_xqa_prepare() {
  std::uint32_t size = 0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&size, smemSize, sizeof(size)));
  CUDA_CHECK(cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize, size));
  hostSmemSize = size;
  int sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  multiprocessors = std::uint32_t(sms);
}
void ignis_nv4_xqa_run(const IgnisNv4XqaCall &c, void *workspace, std::size_t bytes,
                       cudaStream_t stream) {
  const auto need = ignis_nv4_xqa_workspace(c.max_keys);
  if (!need || !workspace || bytes < need ||
      reinterpret_cast<std::uintptr_t>(workspace) % alignof(Metadata) ||
      !hostSmemSize || !c.q || !c.out ||
      !c.tables || !c.table_rows || !c.positions || !c.tokens || c.tokens>8 || c.table_stride < (c.max_keys + 63) / 64 ||
      !c.k_codes || !c.v_codes || !c.k_scales || !c.v_scales || !c.table_count || !c.physical_pages)
    throw std::invalid_argument("nvfp4 XQA: invalid graph invocation/workspace");
  auto *base = static_cast<char *>(workspace);
  auto *m = reinterpret_cast<Metadata *>(base);
  auto *table = reinterpret_cast<std::int32_t *>(base + 256);
  auto *sem = reinterpret_cast<std::uint32_t *>(base + 256 +
      align256(std::size_t((c.max_keys + 63) / 64) * 4));
  const auto raw_scratch = reinterpret_cast<std::uintptr_t>(sem) + 256;
  // MemSegmenter also declares sizeof(T) base alignment. This is not a
  // power-of-two for M48, so align by division, not a bit mask.
  void *scratch = reinterpret_cast<void *>(
      ((raw_scratch + ScratchAlignment - 1) / ScratchAlignment) * ScratchAlignment);
  metadata<<<(((c.max_keys + 63) / 64) + 255) / 256, 256, 0, stream>>>(c, m, table);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemsetAsync(sem, 0, 4 * sizeof(std::uint32_t), stream));
  // C1/T8 physical grid is21x4, graph-stable. Device seq_len chooses
  // activeS=min(21,ceil(actual_window/256)); inactive CTAs leave before barriers.
  launchMHAFlashInfer(multiprocessors, 4, 0, c.scale * 16.0f, nullptr,
      reinterpret_cast<OutputHead *>(c.out), reinterpret_cast<const InputHead *>(c.q), nullptr,
      reinterpret_cast<GMemCacheHead *>(c.k_codes), reinterpret_cast<GMemCacheHead *>(c.v_codes),
      reinterpret_cast<GMemCacheHeadSf *>(c.k_scales), reinterpret_cast<GMemCacheHeadSf *>(c.v_scales),
      reinterpret_cast<const KVCachePageIndex *>(table), ((c.max_keys + 63) / 64) * 64,
      &m->seq_len, 1, 1.0f, nullptr, 8, m->cu, m->mask, sem, scratch, false,
      64ull * 4 * 128, 128, 64ull * 128,
      64ull * 4 * 16, 16, 64ull * 16, c.max_keys, stream);
  CUDA_CHECK(cudaGetLastError());
}
