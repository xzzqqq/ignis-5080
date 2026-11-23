#pragma once
#include <cuda_runtime.h>
#include "nvfp4_window.cuh"
#include <cstddef>
#include <cstdint>
struct IgnisNv4XqaCall {
  const void *q;
  void *out;
  void *k_codes, *v_codes, *k_scales, *v_scales;
  const std::int32_t *tables, *table_rows, *positions, *valid;
  std::uint32_t table_stride, table_count, physical_pages, max_keys, tokens;
  float scale;
  ignis::nv4::Window window{};
};
std::size_t ignis_nv4_xqa_workspace(std::uint32_t max_keys);
std::size_t ignis_nv4_xqa_metadata_bytes(std::uint32_t max_keys);
void ignis_nv4_xqa_prepare();
void ignis_nv4_xqa_run(const IgnisNv4XqaCall &, void *workspace, std::size_t bytes,
                       cudaStream_t stream, void *round_metadata = nullptr, bool prepare_metadata = true);
