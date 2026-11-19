// Native bounded C1/T8/HQ attention, shared by public-frame recording and the
// prepared-Q token mixer. Other shapes retain their established dispatcher.
#include "ignis_gqa_candidate.h"
// The private split helper is shared by this partial/reducer pair. Generic
// vendor helpers retain their original definitions in other translation units.
#include "ops/kernel/gqa_attention_decode.cuh"
namespace ninfer::ops {
template <typename Geometry,bool Int8>
__device__ __forceinline__ int IGNIS_GQA_FN(active_splits)(int window,int capacity,int tokens){
#if IGNIS_GQA_TILES_PER_SPLIT > 0
  if constexpr(!Int8 && Geometry::QHeads==24){
    if(window>0 && window<=70000){
      const int tiles=(window+31)/32;
      const int desired=tiles/IGNIS_GQA_TILES_PER_SPLIT;
      const int positive=desired>IGNIS_GQA_MIN_SPLITS?desired:IGNIS_GQA_MIN_SPLITS;
      const int bounded=positive<IGNIS_GQA_SPLIT_CAP?positive:IGNIS_GQA_SPLIT_CAP;
      return bounded<capacity?bounded:capacity;
    }
  }
#endif
  return gqa_small_t_active_splits<Geometry,Int8>(window,capacity,tokens);
}
}
#define gqa_small_t_active_splits IGNIS_GQA_FN(active_splits)
#define gqa_attention_small_t_tc_partial_bf16_kernel IGNIS_GQA_FN(partial_kernel)
#include "ops/kernel/gqa_attention_decode_hq.cuh"
#define ignis_gqa_hq_reduce_kernel IGNIS_GQA_FN(reduce_kernel)
#include "ignis_gqa_hq_reduce.cuh"
#undef gqa_small_t_active_splits
#include <algorithm>
#include <string>
#include <stdexcept>

namespace {
#if IGNIS_GQA_FP32_PARTIAL
using Partial = float;
#else
using Partial = __nv_bfloat16;
#endif
constexpr bool late_ifwht = IGNIS_GQA_LATE_IFWHT;
constexpr bool write_inactive = IGNIS_GQA_WRITE_INACTIVE_PARTIALS;
constexpr bool decode4 = IGNIS_GQA_CODEC_FOUR_LANE;
#ifndef IGNIS_GQA_REDUCER_PREFETCH4
#define IGNIS_GQA_REDUCER_PREFETCH4 0
#endif
constexpr bool prefetch4=IGNIS_GQA_REDUCER_PREFETCH4;
thread_local std::string error;
uint64_t align256(uint64_t n){return (n+255)&~255ull;}
// Conservative for the entire profile, independent of actual device positions.
// Layout and capacity are candidate-private; the host sees only the byte query.
constexpr int splits=ninfer::ops::Gqa27Geometry::DecodeSplits;
constexpr uint64_t acc_bytes=uint64_t(256)*24*8*splits*sizeof(Partial);
constexpr uint64_t stat_bytes=uint64_t(24)*8*splits*4;
void ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
}
extern "C" int32_t IGNIS_GQA_FN(prepare)(){
  try{
    using namespace ninfer::ops;cudaFuncAttributes attributes{};
    // Resolve candidate modules and both mask variants outside graph capture.
    ck(cudaFuncGetAttributes(&attributes,IGNIS_GQA_FN(partial_kernel)<Gqa27Geometry,8,4,true,true,GqaAppendInput,GqaTcKVHq,true,Partial,late_ifwht,false,write_inactive,decode4>));
    ck(cudaFuncGetAttributes(&attributes,IGNIS_GQA_FN(partial_kernel)<Gqa27Geometry,8,4,true,true,GqaAppendInput,GqaTcKVHq,true,Partial,late_ifwht,true,write_inactive,decode4>));
    ck(cudaFuncGetAttributes(&attributes,IGNIS_GQA_FN(reduce_kernel)<Gqa27Geometry,256,false,false,false,false,Partial,late_ifwht,1,prefetch4>));
    ck(cudaFuncGetAttributes(&attributes,IGNIS_GQA_FN(reduce_kernel)<Gqa27Geometry,256,false,false,false,false,Partial,late_ifwht,2,prefetch4>));
    ck(cudaFuncGetAttributes(&attributes,IGNIS_GQA_FN(reduce_kernel)<Gqa27Geometry,256,false,false,true,false,Partial,late_ifwht,1,prefetch4>));
    ck(cudaFuncGetAttributes(&attributes,IGNIS_GQA_FN(reduce_kernel)<Gqa27Geometry,256,false,false,true,false,Partial,late_ifwht,2,prefetch4>));
    return 0;
  }catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" const char *IGNIS_GQA_FN(error)(){return error.c_str();}
extern "C" uint64_t IGNIS_GQA_FN(workspace)(uint32_t max_keys){
  if(!max_keys || max_keys>1048576)return 0;
  return align256(acc_bytes)+2*align256(stat_bytes);
}
namespace {
template <bool QAlreadyRotated, int GateStride = 1>
int32_t run_attention(const ignis_gqa_candidate_call *c,void *workspace,uint64_t bytes,void *raw_stream){
  try{
    if(!c || c->version!=IGNIS_GQA_CANDIDATE_ABI || c->size!=sizeof(*c) ||
       !workspace || !IGNIS_GQA_FN(workspace)(c->max_visible_keys) || bytes<IGNIS_GQA_FN(workspace)(c->max_visible_keys) ||
       !c->q || !c->k || !c->v || !c->gate || !c->positions || !c->table_rows || !c->block_tables ||
       !c->k_pages || !c->v_pages || !c->k_meta || !c->v_meta || !c->output || c->scale!=0.0625f ||
       !c->logical_pages || !c->physical_pages || !c->slots)
      throw std::runtime_error("invalid C1/T8/HQ candidate invocation");
    using namespace ninfer::ops;
    const auto stream=static_cast<cudaStream_t>(raw_stream);
    auto *acc=static_cast<Partial *>(workspace);
    auto *m=reinterpret_cast<float *>(static_cast<char *>(workspace)+align256(acc_bytes));
    auto *l=reinterpret_cast<float *>(static_cast<char *>(workspace)+align256(acc_bytes)+align256(stat_bytes));
    const GqaAppendInput input{static_cast<const __nv_bfloat16 *>(c->k),static_cast<const __nv_bfloat16 *>(c->v)};
    const GqaTcKVHq kv{static_cast<uint8_t *>(c->k_pages),static_cast<uint8_t *>(c->v_pages),
        static_cast<uint8_t *>(c->k_meta),static_cast<uint8_t *>(c->v_meta),
        static_cast<__nv_bfloat16 *>(c->residual_k),static_cast<__nv_bfloat16 *>(c->residual_v),c->ring};
    ck(ninfer::pdl::launch_dependent({dim3(4,splits,1),dim3(kGqaHqDecodeThreads),0,stream},
        gqa_attention_small_t_tc_partial_bf16_kernel<Gqa27Geometry,8,4,true,true,GqaAppendInput,GqaTcKVHq,true,Partial,late_ifwht,QAlreadyRotated,write_inactive,decode4>,
        static_cast<const __nv_bfloat16 *>(c->q),input,c->positions,kv,c->block_tables,c->valid_columns,c->table_rows,
        static_cast<int32_t>(c->logical_pages),8,8,0,static_cast<int32_t>(c->max_visible_keys),c->scale,acc,m,l));
    auto reduce=[&]<bool Masked>(){
      ck(ninfer::pdl::launch_dependent({dim3(24,1,8),dim3(kIgnisGqaHqReduceThreads),0,stream},
          ignis_gqa_hq_reduce_kernel<Gqa27Geometry,256,false,false,Masked,false,Partial,late_ifwht,GateStride,prefetch4>,
          static_cast<const Partial *>(acc),static_cast<const float *>(m),static_cast<const float *>(l),
          c->positions,c->valid_columns,static_cast<const __nv_bfloat16 *>(c->gate),8,8,0,1,splits,
          static_cast<__nv_bfloat16 *>(c->output)));
    };
    if(c->valid_columns)reduce.template operator()<true>();else reduce.template operator()<false>();
    ck(cudaGetLastError());return 0;
  }catch(const std::exception &e){error=e.what();return -1;}
}

} // namespace
extern "C" int32_t IGNIS_GQA_FN(run)(const ignis_gqa_candidate_call *c,void *workspace,uint64_t bytes,void *stream){
  return run_attention<false,1>(c,workspace,bytes,stream);
}
// Private segment entry: Q has already passed the exact HQ BF16 FWHT boundary.
extern "C" int32_t IGNIS_GQA_FN(run_hq_rotated)(const ignis_gqa_candidate_call *c,void *workspace,uint64_t bytes,void *stream){
  return run_attention<true,1>(c,workspace,bytes,stream);
}

// Segment only: contiguous HQ-prepared Q and gate rows in [Q256,gate256] storage.
extern "C" int32_t IGNIS_GQA_FN(run_hq_prepared_gate2)(const ignis_gqa_candidate_call *c,void *workspace,uint64_t bytes,void *stream){
  return run_attention<true,2>(c,workspace,bytes,stream);
}
