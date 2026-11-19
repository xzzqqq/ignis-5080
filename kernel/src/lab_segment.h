#pragma once
#include "ignis_segment_candidate.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstdint>
struct ignis_model;
struct ignis_seq_pool;
struct ignis_lab_diagnostics;
namespace ignis { struct TrellisCandidate; }
namespace ignis {
enum SegmentFlags : uint32_t { SEGMENT_INPUT=1,SEGMENT_MUTABLE=2,SEGMENT_OUTPUT=4 };
// Stable inventory is boundary-only. INPUT is captured pre-image, MUTABLE is
// both restored pre-image and checked post-image; OUTPUT is checked post-only.
struct SegmentBinding { std::string name; ninfer::Tensor tensor; uint32_t flags; };
bool lab_segment_should_run(const ignis_model &,uint32_t kind,uint32_t layer,uint32_t tokens,uint32_t batch);
bool lab_segment_is_recording(const ignis_model &);
void *lab_segment_before(const ignis_model &,uint32_t layer,const ignis_segment_profile &,
    const std::vector<SegmentBinding> &,cudaStream_t);
void lab_segment_after(const ignis_model &,void *ticket,cudaStream_t);
// Dispatch only. Recording uses before/after around the ORIGINAL layer chain.
void lab_segment_execute(const ignis_model &,uint32_t layer,const ignis_segment_profile &,
    const std::vector<SegmentBinding> &,cudaStream_t);
void segment_prepare_runtime(ignis_model &,ignis_seq_pool &,TrellisCandidate *next);
void segment_invoke(const ignis_model &,uint32_t,const ignis_segment_profile &,const std::vector<SegmentBinding> &,
    void *workspace,uint64_t bytes,cudaStream_t);
uint64_t segment_workspace_bytes(const ignis_model &,const ignis_segment_profile &);
// Agent-owned helpers below reconstruct typed calls using model-owned weights.
// Ordered bindings must be validated by invoke before use (including shapes,
// dtypes and device index metadata in isolated replay).
ignis_segment_profile segment_gqa_profile(const ignis_model &,const ignis_seq_pool &);
ignis_segment_profile segment_gdn_profile(const ignis_model &,const ignis_seq_pool &);
ignis_segment_profile segment_mlp_profile(const ignis_model &,const ignis_seq_pool &);
uint64_t segment_gqa_workspace(const ignis_segment_profile &);
uint64_t segment_gdn_workspace(const ignis_segment_profile &);
uint64_t segment_mlp_workspace(const ignis_segment_profile &);
#define IGNIS_SEGMENT_INVOKE(name) \
void segment_##name##_invoke(const ignis_model &,uint32_t layer,const ignis_segment_profile &, \
    const std::vector<SegmentBinding> &,const ignis_segment_candidate_api *,void *workspace,uint64_t bytes,cudaStream_t)
IGNIS_SEGMENT_INVOKE(gqa);
void segment_gqa_native_invoke(const ignis_model &,uint32_t,const ignis_segment_profile &,
    const std::vector<SegmentBinding> &,void *,uint64_t,cudaStream_t);
IGNIS_SEGMENT_INVOKE(gdn);
IGNIS_SEGMENT_INVOKE(mlp);
#undef IGNIS_SEGMENT_INVOKE
// MLP helper includes fallback to the existing Trellis chain. Caller provides
// post-attention norm and layer, residual boundary; all temporary storage owned.
void segment_mlp(const ignis_model &,uint32_t layer,const ninfer::Tensor &post_norm,
    ninfer::Tensor &residual,ninfer::Tensor &post,ninfer::Tensor &fused,cudaStream_t);
}
extern "C" {
struct ignis_segment_tensor_view { const char *name; uint32_t flags,dtype,present; int32_t ne[4]; };
int32_t ignis_lab_segment_replay(ignis_model *,uint32_t layer,const ignis_segment_profile *,
    const ignis_segment_tensor_view *,uint32_t count,const void *blob,uint64_t bytes,
    uint32_t warmup,uint32_t samples,uint32_t check,float *times,uint32_t *checked,
    ignis_lab_diagnostics *,ignis_lab_diagnostics *per_binding,uint64_t *state_mismatches,uint64_t *state_bytes,
    uint64_t *binding_state_mismatches,uint64_t *binding_state_bytes);
const char *ignis_lab_segment_implementation(ignis_model *);
uint64_t ignis_lab_segment_workspace_capacity(ignis_model *,uint32_t kind);
}
