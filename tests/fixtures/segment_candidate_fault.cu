// Three-kind optional extension admission/capture failures, never serve.
// SEGMENT_TEST_MODE=1 bad ABI; 2 invalid workspace; 3 graph capture failure.
#define TEST_MODE 4
#include "trellis_candidate_fault.cu"
#include "ignis_segment_candidate.h"
#include "ignis_mlp_segment.h"
#include "ignis_gqa_segment.h"
#include "ignis_gdn_segment.h"
#ifndef SEGMENT_TEST_MODE
#error SEGMENT_TEST_MODE required
#endif
namespace {
int32_t segment_prepare(){return 0;}
uint64_t segment_workspace(const ignis_segment_profile *){return SEGMENT_TEST_MODE==2?UINT64_MAX:16u<<20;}
int32_t segment_mlp(const ignis_mlp_segment_call *,void *,uint64_t,void *){return -1;}
int32_t segment_gqa(const ignis_gqa_segment_call *,void *,uint64_t,void *){return -1;}
int32_t segment_gdn(const ignis_gdn_segment_call *,void *,uint64_t,void *){return -1;}
const char *segment_error(){return "injected finite segment capture failure";}
}
extern "C" __attribute__((visibility("default")))
const ignis_segment_candidate_api *ignis_segment_candidate_v1(){
  static const ignis_segment_candidate_api api{SEGMENT_TEST_MODE==1?999u:IGNIS_SEGMENT_CANDIDATE_ABI,
      sizeof(ignis_segment_candidate_api),sizeof(ignis_segment_profile),sizeof(ignis_mlp_segment_call),
      sizeof(ignis_gqa_segment_call),sizeof(ignis_gdn_segment_call),"segment-test-only",segment_prepare,
      segment_workspace,segment_mlp,segment_gqa,segment_gdn,segment_error};return &api;
}
