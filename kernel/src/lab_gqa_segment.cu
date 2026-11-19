#ifdef IGNIS_SEGMENT_BUILTIN
#include "ignis_gqa_native.h"
extern "C" uint64_t ignis_lab_gqa_segment_workspace(const ignis_segment_profile *p){
  return ignis_gqa_native_segment_workspace(p);
}
extern "C" int32_t ignis_lab_gqa_segment_prepare(){return ignis_gqa_native_segment_prepare();}
extern "C" const char *ignis_lab_gqa_segment_error(){return ignis_gqa_native_segment_error();}
extern "C" int32_t ignis_lab_gqa_segment_run(const ignis_gqa_segment_call *c,
    void *workspace,uint64_t bytes,void *stream){
  return ignis_gqa_native_segment_run(c,workspace,bytes,stream);
}
extern "C" IGNIS_LAB_DIAGNOSTIC_EXPORT int32_t ignis_lab_gqa_segment_tensors_v1(const ignis_segment_profile *p,
    const void *workspace,uint64_t bytes,ignis_lab_segment_tensor *out,uint32_t capacity){
  return ignis_gqa_native_segment_tensors_v1(p,workspace,bytes,out,capacity);
}
#else
#define IGNIS_GQA_SEGMENT_FN(name) ignis_lab_gqa_segment_##name
#define IGNIS_GQA_ATTENTION_FN(name) ignis_lab_gqa_##name
#include "gqa_token_mixer_impl.cuh"
#undef IGNIS_GQA_ATTENTION_FN
#undef IGNIS_GQA_SEGMENT_FN
#endif
