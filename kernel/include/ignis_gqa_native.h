#pragma once
// Bounded native C1/T8/HQ implementation; public-frame and prepared-frame entries
// share one workspace/precision/split policy. Not a dynamically loaded candidate.
// Selected by the batched C1/T8 verify path; eager prefill/readout keep their
// established routes. The decomposed verify recording uses the public frame.
#include "ignis_gqa_candidate.h"
#include "ignis_gqa_segment.h"
extern "C" {
int32_t ignis_gqa_native_prepare();
uint64_t ignis_gqa_native_workspace(uint32_t);
const char *ignis_gqa_native_error();
int32_t ignis_gqa_native_run(const ignis_gqa_candidate_call *,void *,uint64_t,void *);
int32_t ignis_gqa_native_run_hq_rotated(const ignis_gqa_candidate_call *,void *,uint64_t,void *);
int32_t ignis_gqa_native_run_hq_prepared_gate2(const ignis_gqa_candidate_call *,void *,uint64_t,void *);
int32_t ignis_gqa_native_segment_prepare();
uint64_t ignis_gqa_native_segment_workspace(const ignis_segment_profile *);
const char *ignis_gqa_native_segment_error();
int32_t ignis_gqa_native_segment_tensors_v1(const ignis_segment_profile *,const void *,uint64_t,
    ignis_lab_segment_tensor *,uint32_t);
int32_t ignis_gqa_native_segment_run(const ignis_gqa_segment_call *,void *,uint64_t,void *);
}
