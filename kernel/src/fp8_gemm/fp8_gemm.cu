// ignis kernel leaf: the image model's FP8 W8A8 GEMM (spec image-model/02 acceptance 2, GitHub
// #319) -- OURS (kernel/include/ignis_image_gemm.h; ADR 0051 tier 1). The kernel is
// fp8_gemm_kernel.cuh's; this unit instantiates its default tile configuration only, the one the
// C ABI runs. The other configurations exist for the bench's sweep and are built into the test
// and the bench, never into the leaf (kernel/tests/image_gemm_sweep.cu).

#include "fp8_gemm_kernel.cuh"
#include "ignis_image_gemm.h"

#include <atomic>

static_assert(IGNIS_FP8_MAX_SEGMENTS == ignis_image::kMaxSegments, "the grouped GEMM takes ignis_fp8_segment's count");

namespace ignis_image {
namespace {

constexpr int kMaxDevices = 64;
std::atomic<bool> g_prepared[kMaxDevices];

int32_t require_prepared(const char *op) {
  int device = 0;
  const cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return fail(std::string(op) + ": " + cudaGetErrorString(err));
  if (device < 0 || device >= kMaxDevices || !g_prepared[device].load()) {
    return fail(std::string(op) + ": device " + std::to_string(device) +
                " is not prepared; call ignis_image_gemm_prepare at load");
  }
  return 0;
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" int32_t ignis_image_gemm_prepare(void) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) err = gemm_w8a8_set_attributes<GemmDefaultConfig>();
  if (err != cudaSuccess) return fail(std::string("ignis_image_gemm_prepare: ") + cudaGetErrorString(err));
  if (device < 0 || device >= kMaxDevices) return fail("ignis_image_gemm_prepare: device id out of range");
  g_prepared[device].store(true);
  return 0;
}

extern "C" const char *ignis_image_gemm_last_error(void) { return last_error_cstr(); }

extern "C" int32_t ignis_image_gemm_w8a8(const void *weight, uint32_t rows, uint32_t cols, const void *codes,
                                         const float *scales, uint32_t tokens, void *y, uint32_t y_f32, void *stream) {
  if (require_prepared("ignis_image_gemm_w8a8") != 0) return -1;
  return gemm_w8a8_launch<GemmDefaultConfig>(weight, rows, cols, codes, scales, tokens, y, y_f32, stream);
}

extern "C" int32_t ignis_image_gemm_w8a8_grouped(const ignis_fp8_segment *segments, uint32_t count, uint32_t cols,
                                                 const void *codes, const float *scales, uint32_t tokens,
                                                 uint32_t y_f32, void *stream) {
  const char *op = "ignis_image_gemm_w8a8_grouped";
  if (segments == nullptr) return fail(std::string(op) + ": NULL pointer");
  if (count == 0 || count > static_cast<uint32_t>(kMaxSegments)) {
    return fail(std::string(op) + ": 1.." + std::to_string(kMaxSegments) + " segments");
  }
  if (require_prepared(op) != 0) return -1;
  const void *weights[kMaxSegments];
  uint32_t rows[kMaxSegments];
  void *ys[kMaxSegments];
  for (uint32_t i = 0; i < count; ++i) {
    weights[i] = segments[i].weight;
    rows[i] = segments[i].rows;
    ys[i] = segments[i].y;
  }
  return gemm_w8a8_launch_segments<GemmDefaultConfig>(op, weights, rows, ys, count, cols, codes, scales, tokens,
                                                      y_f32 != 0 ? kYF32 : kYBf16, stream);
}

extern "C" int32_t ignis_image_gemm_w8a8_add(const void *weight, uint32_t rows, uint32_t cols, const void *codes,
                                             const float *scales, uint32_t tokens, void *residual, void *stream) {
  const char *op = "ignis_image_gemm_w8a8_add";
  if (require_prepared(op) != 0) return -1;
  return gemm_w8a8_launch_segments<GemmDefaultConfig>(op, &weight, &rows, &residual, 1, cols, codes, scales, tokens,
                                                      kYBf16Add, stream);
}
