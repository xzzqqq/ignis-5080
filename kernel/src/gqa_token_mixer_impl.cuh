// Shared optimized finite GQA token mixer. Public wrappers choose export and
// attention prefixes; the Q producer and Gate2 consumer remain paired.
#ifndef IGNIS_GQA_TRANSIENT_WORKSPACE
#define IGNIS_GQA_TRANSIENT_WORKSPACE 0
#endif
#include "ignis_gqa_native.h"
#include "gqa_segment_workspace.cuh"
#include "core/pdl.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/kernel/rope.cuh"

#include "ignis_gqa_candidate.h"
extern "C" int32_t IGNIS_GQA_ATTENTION_FN(prepare)();
extern "C" uint64_t IGNIS_GQA_ATTENTION_FN(workspace)(uint32_t);
extern "C" int32_t IGNIS_GQA_ATTENTION_FN(run_hq_rotated)(const ignis_gqa_candidate_call *,void *,uint64_t,void *);
extern "C" int32_t IGNIS_GQA_ATTENTION_FN(run_hq_prepared_gate2)(const ignis_gqa_candidate_call *,void *,uint64_t,void *);
extern "C" const char *IGNIS_GQA_ATTENTION_FN(error)();
extern "C" int32_t ignis_gqa_native_trellis_reset(const ignis_trellis_workspace *,void *);
extern "C" uint64_t ignis_gqa_native_q_fixed_workspace();
extern "C" int32_t ignis_gqa_native_q_fixed_splits(const ignis_trellis_weight *,
    const void *,void *,const ignis_trellis_workspace *,void *);
extern "C" uint64_t ignis_gqa_native_kv_pair_workspace();
extern "C" int32_t ignis_gqa_native_kv_pair(const ignis_trellis_weight *,
    const ignis_trellis_weight *,const void *,void *,void *,
    const ignis_trellis_workspace *,void *);

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string segment_error;
namespace gqa_workspace=ignis_gqa_segment_workspace;
constexpr uint32_t kHidden=gqa_workspace::hidden,kTokens=gqa_workspace::tokens,
    kQuery=gqa_workspace::query,kKv=gqa_workspace::kv;
constexpr gqa_workspace::Layout tensor_layout{};
constexpr auto retained_tensors=gqa_workspace::Retained::InterleavedHq;
// Preserve BF16 norm/RoPE/FWHT boundaries before the prepared-Q consumer.
constexpr bool kHqQueryTransform = true;
constexpr int kQRowStride = 2;
uint64_t aligned(uint64_t n){return (n+255)&~uint64_t(255);}
bool supported(const ignis_segment_profile *p){
  return p && p->kind==IGNIS_SEGMENT_GQA && p->tokens==kTokens && p->batch==1 &&
      p->hidden==kHidden && !p->intermediate && p->slots && p->slots<=INT32_MAX &&
      p->max_keys && p->max_keys<=1048576 &&
      !p->flags && std::isfinite(p->rms_eps) && p->rms_eps>0;
}
uint64_t linear_bytes(){
  const uint64_t pair_bytes=ignis_gqa_native_kv_pair_workspace();
  return std::max({ignis_gqa_native_q_fixed_workspace(),pair_bytes,ignis_trellis_workspace_bytes(kHidden,2*kQuery,1,kTokens),
      ignis_trellis_workspace_bytes(kHidden,kKv,1,kTokens),
      ignis_trellis_workspace_bytes(kQuery,kHidden,1,kTokens)});
}
uint64_t attention_bytes(uint32_t max_keys){
  return IGNIS_GQA_ATTENTION_FN(workspace)(max_keys);
}
uint64_t intermediate_bytes(){
  return tensor_layout.bytes;
}
void check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
void check_linear(int32_t rc){if(rc)throw std::runtime_error(ignis_trellis_last_error());}
void validate_weight(const ignis_trellis_weight &w,uint32_t in,uint32_t out){
  if(!w.trellis || !w.suh || !w.svh || w.in!=in || w.out!=out)
    throw std::runtime_error("GQA segment weight geometry mismatch");
}
}
extern "C" uint64_t IGNIS_GQA_SEGMENT_FN(workspace)(const ignis_segment_profile *p){
  if(!supported(p))return 0;
  const auto trellis=linear_bytes();
  const auto attention=attention_bytes(p->max_keys);
  return trellis && attention ? intermediate_bytes()+aligned(trellis)+aligned(attention) : 0;
}
extern "C" int32_t IGNIS_GQA_SEGMENT_FN(prepare)(){
  try{
    if(ignis_trellis_prepare())throw std::runtime_error(ignis_trellis_last_error());
    if(IGNIS_GQA_ATTENTION_FN(prepare)())throw std::runtime_error(IGNIS_GQA_ATTENTION_FN(error)());
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes,
        ninfer::ops::rmsnorm_cta_bf16x2_kernel<ninfer::ops::RmsEpilogue::Offset,512,8,true>));
    check(cudaFuncGetAttributes(&attributes,
        ninfer::ops::rope_norm_fused_kernel<ninfer::ops::RopeKernelMode::Text1D,24,4,128,kHqQueryTransform,kQRowStride>));
    return 0;
  }catch(const std::exception &e){segment_error=e.what();return -1;}
}
extern "C" const char *IGNIS_GQA_SEGMENT_FN(error)(){return segment_error.c_str();}
extern "C" int32_t IGNIS_GQA_SEGMENT_FN(run)(const ignis_gqa_segment_call *c,
    void *workspace,uint64_t bytes,void *raw_stream){
  try{
    const auto required=c?IGNIS_GQA_SEGMENT_FN(workspace)(&c->profile):0;
    if(!required || !workspace || bytes<required || !c->input || !c->input_norm ||
       !c->query_norm || !c->key_norm || !c->positions || !c->rope_positions ||
       !c->table_rows || !c->block_tables || !c->logical_pages || !c->physical_pages ||
       !c->k_pages || !c->v_pages || !c->k_meta || !c->v_meta ||
       !c->residual_k || !c->residual_v || !c->ring || !c->output ||
       !std::isfinite(c->rope_attention_factor) || c->rope_attention_factor<=0)
      throw std::runtime_error("invalid C1/T8/HQ GQA segment invocation");
    validate_weight(c->q_proj,kHidden,2*kQuery);validate_weight(c->k_proj,kHidden,kKv);
    validate_weight(c->v_proj,kHidden,kKv);validate_weight(c->output_proj,kQuery,kHidden);
    const auto stream=static_cast<cudaStream_t>(raw_stream);
    auto at=[&](uint64_t offset){return tensor_layout.at(workspace,offset);};
    void *normalized=at(tensor_layout.normalized),*q=at(tensor_layout.q),*gate=at(tensor_layout.gate);
    void *k=at(tensor_layout.k),*v=at(tensor_layout.v),*rq=at(tensor_layout.rq),*rk=at(tensor_layout.rk);
    void *attention=at(tensor_layout.attention),*interleaved=at(tensor_layout.interleaved);
    auto *cursor=static_cast<char *>(workspace)+tensor_layout.bytes;
    const ignis_trellis_workspace trellis{cursor,linear_bytes()};cursor+=aligned(trellis.bytes);
#if IGNIS_GQA_TRANSIENT_WORKSPACE
    // This reset is part of the timed/captured native chain. Other arena users
    // may have overwritten synchronization bytes since the preceding GQA layer.
    check_linear(ignis_gqa_native_trellis_reset(&trellis,stream));
#endif
    const auto attention_capacity=attention_bytes(c->profile.max_keys);
    check(ninfer::pdl::launch_dependent({dim3(kTokens),dim3(512),0,stream},
        ninfer::ops::rmsnorm_cta_bf16x2_kernel<ninfer::ops::RmsEpilogue::Offset,512,8,true>,
        static_cast<const __nv_bfloat162 *>(c->input),static_cast<const __nv_bfloat162 *>(c->input_norm),
        static_cast<const __nv_bfloat162 *>(nullptr),static_cast<__nv_bfloat162 *>(normalized),
        int32_t(kHidden),int64_t(kTokens),c->profile.rms_eps));
    check_linear(ignis_gqa_native_q_fixed_splits(&c->q_proj,normalized,interleaved,
        &trellis,stream));
    // Preserve the exact Q/gate BF16 bytes and consume the projection layout.
    q=interleaved;
    gate=static_cast<char *>(interleaved)+256*2;
    check_linear(ignis_gqa_native_kv_pair(&c->k_proj,&c->v_proj,normalized,k,v,&trellis,stream));
    ninfer::ops::RopeFrequencies rope;
    std::copy_n(c->rope_inv_frequency,128,rope.inv_frequency);
    rope.attention_factor=c->rope_attention_factor;
    check(ninfer::pdl::launch_dependent({dim3((24+4)*kTokens/4),dim3(128),0,stream},
        ninfer::ops::rope_norm_fused_kernel<ninfer::ops::RopeKernelMode::Text1D,24,4,128,kHqQueryTransform,kQRowStride>,
        static_cast<const __nv_bfloat16 *>(q),static_cast<const __nv_bfloat16 *>(k),
        static_cast<const __nv_bfloat162 *>(c->query_norm),static_cast<const __nv_bfloat162 *>(c->key_norm),
        c->profile.rms_eps,c->rope_positions,rope,static_cast<__nv_bfloat16 *>(rq),
        static_cast<__nv_bfloat16 *>(rk),int32_t(kTokens)));
    const ignis_gqa_candidate_call attention_call{IGNIS_GQA_CANDIDATE_ABI,sizeof(ignis_gqa_candidate_call),
        c->profile.max_keys,c->logical_pages,c->physical_pages,c->profile.slots,0.0625f,
        rq,rk,v,gate,c->positions,c->valid_columns,c->table_rows,c->block_tables,
        c->k_pages,c->v_pages,c->k_meta,c->v_meta,c->residual_k,c->residual_v,c->ring,attention};
    if(IGNIS_GQA_ATTENTION_FN(run_hq_prepared_gate2)(&attention_call,cursor,attention_capacity,stream))
      throw std::runtime_error(IGNIS_GQA_ATTENTION_FN(error)());
    check(cudaMemcpyAsync(c->output,c->input,uint64_t(kHidden)*kTokens*2,cudaMemcpyDeviceToDevice,stream));
    check_linear(ignis_trellis_linear(&c->output_proj,attention,kTokens,c->output,kHidden,
        IGNIS_TRELLIS_ADD,&trellis,stream));
    check(cudaGetLastError());return 0;
  }catch(const std::exception &e){segment_error=e.what();return -1;}
}

// Describe only buffers retained by this exact native/candidate launch chain.
extern "C" IGNIS_LAB_DIAGNOSTIC_EXPORT int32_t IGNIS_GQA_SEGMENT_FN(tensors_v1)(const ignis_segment_profile *p,
    const void *workspace,uint64_t bytes,ignis_lab_segment_tensor *out,uint32_t capacity){
  const auto required=IGNIS_GQA_SEGMENT_FN(workspace)(p);
  if(!required || bytes<required)return -1;
  return gqa_workspace::tensors(tensor_layout,retained_tensors,workspace,bytes,out,capacity);
}
