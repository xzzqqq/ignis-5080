// Kernel Lab candidate owns the complete fixed-profile attention invocation.
// Shared upstream kernels are instantiated privately inside this immutable DSO.
#include "ignis_gqa_candidate.h"
#include "lab_gqa_candidate_policy.cuh"
// Retain the original helper under a private name, then supply the policy used
// by BOTH the partial kernel and reducer instantiated below. No runtime vendor
// code or builtin partial/reducer entry is called from this candidate.
#define gqa_small_t_active_splits ignis_lab_original_active_splits
#include "ops/kernel/gqa_attention_decode.cuh"
#undef gqa_small_t_active_splits
namespace ninfer::ops {
template <typename Geometry,bool Int8>
__device__ __forceinline__ int gqa_small_t_active_splits(int window,int capacity,int tokens){
#if IGNIS_LAB_GQA_FORCED_SPLITS > 0
  if constexpr(!Int8 && Geometry::QHeads==24){
    if(window>20160 && window<=70000)
      return IGNIS_LAB_GQA_FORCED_SPLITS<capacity?IGNIS_LAB_GQA_FORCED_SPLITS:capacity;
  }
#endif
  return ignis_lab_original_active_splits<Geometry,Int8>(window,capacity,tokens);
}
}
#define gqa_attention_small_t_tc_partial_bf16_kernel ignis_lab_gqa_partial_kernel
#include "ops/kernel/gqa_attention_decode_hq.cuh"
#define ignis_gqa_hq_reduce_kernel ignis_lab_gqa_reduce_kernel
#include "ignis_gqa_hq_reduce.cuh"
#include <algorithm>
#include <string>
#include <stdexcept>

namespace {
thread_local std::string error;
uint64_t align256(uint64_t n){return (n+255)&~255ull;}
// Conservative for the entire profile, independent of actual device positions.
// Layout and capacity are candidate-private; the host sees only the byte query.
constexpr int splits=ninfer::ops::Gqa27Geometry::DecodeSplits;
constexpr uint64_t acc_bytes=uint64_t(256)*24*8*splits*2;
constexpr uint64_t stat_bytes=uint64_t(24)*8*splits*4;
void ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
}
extern "C" int32_t ignis_lab_gqa_prepare(){
  try{
    using namespace ninfer::ops;cudaFuncAttributes attributes{};
    // Resolve candidate modules and both mask variants outside graph capture.
    ck(cudaFuncGetAttributes(&attributes,ignis_lab_gqa_partial_kernel<Gqa27Geometry,8,4,true,true,GqaAppendInput,GqaTcKVHq>));
    ck(cudaFuncGetAttributes(&attributes,ignis_lab_gqa_reduce_kernel<Gqa27Geometry,256,false,false,false,false>));
    ck(cudaFuncGetAttributes(&attributes,ignis_lab_gqa_reduce_kernel<Gqa27Geometry,256,false,false,true,false>));
    return 0;
  }catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" const char *ignis_lab_gqa_error(){return error.c_str();}
extern "C" uint64_t ignis_lab_gqa_workspace(uint32_t max_keys){
  if(!max_keys || max_keys>1048576)return 0;
  return align256(acc_bytes)+2*align256(stat_bytes);
}
extern "C" int32_t ignis_lab_gqa_run(const ignis_gqa_candidate_call *c,void *workspace,uint64_t bytes,void *raw_stream){
  try{
    if(!c || c->version!=IGNIS_GQA_CANDIDATE_ABI || c->size!=sizeof(*c) ||
       !workspace || !ignis_lab_gqa_workspace(c->max_visible_keys) || bytes<ignis_lab_gqa_workspace(c->max_visible_keys) ||
       !c->q || !c->k || !c->v || !c->gate || !c->positions || !c->table_rows || !c->block_tables ||
       !c->k_pages || !c->v_pages || !c->k_meta || !c->v_meta || !c->output || c->scale!=0.0625f ||
       !c->logical_pages || !c->physical_pages || !c->slots)
      throw std::runtime_error("invalid C1/T8/HQ candidate invocation");
    using namespace ninfer::ops;
    const auto stream=static_cast<cudaStream_t>(raw_stream);
    auto *acc=static_cast<__nv_bfloat16 *>(workspace);
    auto *m=reinterpret_cast<float *>(static_cast<char *>(workspace)+align256(acc_bytes));
    auto *l=reinterpret_cast<float *>(static_cast<char *>(workspace)+align256(acc_bytes)+align256(stat_bytes));
    const GqaAppendInput input{static_cast<const __nv_bfloat16 *>(c->k),static_cast<const __nv_bfloat16 *>(c->v)};
    const GqaTcKVHq kv{static_cast<uint8_t *>(c->k_pages),static_cast<uint8_t *>(c->v_pages),
        static_cast<uint8_t *>(c->k_meta),static_cast<uint8_t *>(c->v_meta),
        static_cast<__nv_bfloat16 *>(c->residual_k),static_cast<__nv_bfloat16 *>(c->residual_v),c->ring};
    ck(ninfer::pdl::launch_dependent({dim3(4,splits,1),dim3(kGqaHqDecodeThreads),0,stream},
        gqa_attention_small_t_tc_partial_bf16_kernel<Gqa27Geometry,8,4,true,true,GqaAppendInput,GqaTcKVHq>,
        static_cast<const __nv_bfloat16 *>(c->q),input,c->positions,kv,c->block_tables,c->valid_columns,c->table_rows,
        static_cast<int32_t>(c->logical_pages),8,8,0,static_cast<int32_t>(c->max_visible_keys),c->scale,acc,m,l));
    auto reduce=[&]<bool Masked>(){
      ck(ninfer::pdl::launch_dependent({dim3(24,1,8),dim3(kIgnisGqaHqReduceThreads),0,stream},
          ignis_gqa_hq_reduce_kernel<Gqa27Geometry,256,false,false,Masked,false>,
          static_cast<const __nv_bfloat16 *>(acc),static_cast<const float *>(m),static_cast<const float *>(l),
          c->positions,c->valid_columns,static_cast<const __nv_bfloat16 *>(c->gate),8,8,0,1,splits,
          static_cast<__nv_bfloat16 *>(c->output)));
    };
    if(c->valid_columns)reduce.template operator()<true>();else reduce.template operator()<false>();
    ck(cudaGetLastError());return 0;
  }catch(const std::exception &e){error=e.what();return -1;}
}
