/* Bounded Kernel Lab segment ABI. Exactly three finite segment kinds; no IR.
 * Calls name boundary tensors, model weights and externally consumed state /
 * records only. Candidates own ordering, fusion and every temporary layout.
 * Workspace is host-owned, zeroed on admission/replay reset, graph-stable;
 * run must allocate nothing, retain no pointer, and leave reusable counters
 * balanced. Query/prepare execute outside graph capture. C1/T8 initially.
 */
#ifndef IGNIS_SEGMENT_CANDIDATE_H
#define IGNIS_SEGMENT_CANDIDATE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define IGNIS_SEGMENT_CANDIDATE_ABI 1
enum ignis_segment_kind { IGNIS_SEGMENT_MLP=1,IGNIS_SEGMENT_GQA=2,IGNIS_SEGMENT_GDN=3 };
struct ignis_segment_profile {
  uint32_t kind,tokens,batch,hidden,intermediate,max_keys,slots,flags;
  float rms_eps;
};
struct ignis_mlp_segment_call;
struct ignis_gqa_segment_call;
struct ignis_gdn_segment_call;
struct ignis_segment_candidate_api {
  uint32_t version,size,profile_size,mlp_call_size,gqa_call_size,gdn_call_size;
  const char *identity;
  int32_t (*prepare)(void);
  uint64_t (*workspace_bytes)(const struct ignis_segment_profile *);
  int32_t (*run_mlp)(const struct ignis_mlp_segment_call *,void *,uint64_t,void *);
  int32_t (*run_gqa)(const struct ignis_gqa_segment_call *,void *,uint64_t,void *);
  int32_t (*run_gdn)(const struct ignis_gdn_segment_call *,void *,uint64_t,void *);
  const char *(*last_error)(void);
};
const struct ignis_segment_candidate_api *ignis_segment_candidate_v1(void);
/* Optional offline replay diagnostics. Called AFTER a completed run, outside
 * timing and graph capture. Describe retained contiguous workspace tensors of
 * THIS implementation; never run a different chain to reconstruct them. A
 * fused implementation may omit these symbols or return zero. Names are fixed
 * ASCII identifiers. dtype: BF16=0,U8=1,I32=2,FP32=3; ne[0] is contiguous.
 * Providers must track their implementation's workspace layout and lifetimes.
 * Legacy candidate ABI/layout is unchanged. No pointers are persisted on disk.
 */
struct ignis_lab_segment_tensor {
  const char *name; const void *data; uint64_t bytes; uint32_t dtype; int32_t ne[4];
};
typedef int32_t (*ignis_lab_segment_tensors_fn)(const struct ignis_segment_profile *,
    const void *,uint64_t,struct ignis_lab_segment_tensor *,uint32_t);
#if defined(__GNUC__)
#define IGNIS_LAB_DIAGNOSTIC_EXPORT __attribute__((visibility("default")))
#else
#define IGNIS_LAB_DIAGNOSTIC_EXPORT
#endif
IGNIS_LAB_DIAGNOSTIC_EXPORT int32_t ignis_lab_mlp_segment_tensors_v1(const struct ignis_segment_profile *,const void *,uint64_t,struct ignis_lab_segment_tensor *,uint32_t);
IGNIS_LAB_DIAGNOSTIC_EXPORT int32_t ignis_lab_gqa_segment_tensors_v1(const struct ignis_segment_profile *,const void *,uint64_t,struct ignis_lab_segment_tensor *,uint32_t);
IGNIS_LAB_DIAGNOSTIC_EXPORT int32_t ignis_lab_gdn_segment_tensors_v1(const struct ignis_segment_profile *,const void *,uint64_t,struct ignis_lab_segment_tensor *,uint32_t);
#ifdef __cplusplus
}
#endif
#endif
