// Optional GQA extension admission/capture failures. Compile with GQA_TEST_MODE
// 1 (bad ABI), 2 (invalid query), or 3 (capture failure). Never serve.
#define TEST_MODE 4
#include "trellis_candidate_fault.cu"
#include "ignis_gqa_candidate.h"
#ifndef GQA_TEST_MODE
#error GQA_TEST_MODE required
#endif
namespace {
int32_t gqa_prepare(){return 0;}
uint64_t gqa_workspace(uint32_t){return GQA_TEST_MODE==2?UINT64_MAX:16u<<20;}
int32_t gqa_run(const ignis_gqa_candidate_call *,void *,uint64_t,void *){return -1;}
const char *gqa_error(){return "injected GQA capture failure";}
}
extern "C" __attribute__((visibility("default")))
const ignis_gqa_candidate_api *ignis_gqa_candidate_v1(){
  static const ignis_gqa_candidate_api api{GQA_TEST_MODE==1?999u:IGNIS_GQA_CANDIDATE_ABI,
      sizeof(ignis_gqa_candidate_api),sizeof(ignis_gqa_candidate_call),0,"gqa-test-only",
      gqa_prepare,gqa_workspace,gqa_run,gqa_error};return &api;
}
