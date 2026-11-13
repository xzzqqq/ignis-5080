// Deliberately invalid candidates for the model reload regression. Never serve.
#include "ignis_trellis_candidate.h"
#include <cuda_runtime.h>
#include <cstdint>
#ifndef TEST_MODE
#error TEST_MODE required
#endif
namespace {
int32_t prepare() { return TEST_MODE == 5 ? -1 : 0; }
uint64_t workspace(uint32_t, uint32_t, uint32_t, uint32_t) {
  return TEST_MODE == 2 ? UINT64_MAX : 1;
}
int32_t linear(const ignis_trellis_weight *w, const void *, uint32_t tokens, void *y,
               uint32_t ldy, uint32_t, const ignis_trellis_workspace *, void *stream) {
  if (TEST_MODE == 3) return -1;
  return cudaMemset2DAsync(y, ldy * 2, 0, w->out * 2, tokens,
                           static_cast<cudaStream_t>(stream)) == cudaSuccess ? 0 : -1;
}
int32_t swiglu(const ignis_trellis_weight *w, const ignis_trellis_weight *, const void *x,
               uint32_t t, void *y, uint32_t ldy, const ignis_trellis_workspace *ws, void *s) {
  return linear(w, x, t, y, ldy, 0, ws, s);
}
const char *error() { return "injected candidate failure"; }
}
#if TEST_MODE != 6
extern "C" __attribute__((visibility("default")))
const ignis_trellis_candidate_api *ignis_trellis_candidate_v1() {
  static const ignis_trellis_candidate_api api = {
    TEST_MODE == 1 ? 999u : IGNIS_TRELLIS_CANDIDATE_ABI,
    sizeof(ignis_trellis_candidate_api), sizeof(ignis_trellis_weight),
    sizeof(ignis_trellis_workspace), "test-only", prepare, workspace, linear, swiglu, error};
  return &api;
}

#endif
