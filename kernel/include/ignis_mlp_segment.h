#pragma once
#include "ignis_segment_candidate.h"
#include "ignis_trellis_linear.h"
#ifdef __cplusplus
extern "C" {
#endif
struct ignis_mlp_segment_call {
  struct ignis_segment_profile profile;
  const void *norm;
  ignis_trellis_weight gate,up,down;
  void *residual;
};
int32_t ignis_lab_mlp_segment_prepare(void);
uint64_t ignis_lab_mlp_segment_workspace(const struct ignis_segment_profile *);
int32_t ignis_lab_mlp_segment_run(const struct ignis_mlp_segment_call *,void *,uint64_t,void *);
const char *ignis_lab_mlp_segment_error(void);
#ifdef __cplusplus
}
#endif
