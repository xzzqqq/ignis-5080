// Default finite GQA segment. Compiled into both the host baseline and each
// immutable candidate DSO; all launches and intermediate layouts belong here.
#include "ignis_gqa_segment.h"
#include "core/pdl.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/kernel/rope.cuh"

#include "ignis_gqa_candidate.h"
// Baseline and DSO both link the complete existing attention implementation.
// Candidate DSOs bind these calls locally with -Bsymbolic.
extern "C" int32_t ignis_lab_gqa_prepare();
extern "C" uint64_t ignis_lab_gqa_workspace(uint32_t);
extern "C" int32_t ignis_lab_gqa_run(const ignis_gqa_candidate_call *,void *,uint64_t,void *);
extern "C" const char *ignis_lab_gqa_error();
#ifdef IGNIS_SEGMENT_BUILTIN
namespace ignis {
uint64_t segment_gqa_builtin_attention_workspace(uint32_t);
void segment_gqa_builtin_attention(const ignis_gqa_segment_call &,const void *,const void *,
    const void *,const void *,void *,void *,uint64_t,cudaStream_t);
}
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string segment_error;
constexpr uint32_t kHidden=5120,kTokens=8,kQuery=6144,kKv=1024;
uint64_t aligned(uint64_t n){return (n+255)&~uint64_t(255);}
bool supported(const ignis_segment_profile *p){
  return p && p->kind==IGNIS_SEGMENT_GQA && p->tokens==kTokens && p->batch==1 &&
      p->hidden==kHidden && !p->intermediate && p->slots && p->slots<=INT32_MAX &&
      p->max_keys && p->max_keys<=1048576 &&
      !p->flags && std::isfinite(p->rms_eps) && p->rms_eps>0;
}
uint64_t linear_bytes(){
  return std::max({ignis_trellis_workspace_bytes(kHidden,2*kQuery,1,kTokens),
      ignis_trellis_workspace_bytes(kHidden,kKv,1,kTokens),
      ignis_trellis_workspace_bytes(kQuery,kHidden,1,kTokens)});
}
uint64_t tensor_bytes(uint32_t width){return aligned(uint64_t(width)*kTokens*2);}
uint64_t attention_bytes(uint32_t max_keys){
#ifdef IGNIS_SEGMENT_BUILTIN
  return ignis::segment_gqa_builtin_attention_workspace(max_keys);
#else
  return ignis_lab_gqa_workspace(max_keys);
#endif
}
uint64_t intermediate_bytes(){
  // normalized, Q, gate, K, V, rotated Q/K, attention, interleaved Q/gate.
  return tensor_bytes(kHidden)+6*tensor_bytes(kQuery)+3*tensor_bytes(kKv);
}
void check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
void check_linear(int32_t rc){if(rc)throw std::runtime_error(ignis_trellis_last_error());}
void validate_weight(const ignis_trellis_weight &w,uint32_t in,uint32_t out){
  if(!w.trellis || !w.suh || !w.svh || w.in!=in || w.out!=out)
    throw std::runtime_error("GQA segment weight geometry mismatch");
}
}
extern "C" uint64_t ignis_lab_gqa_segment_workspace(const ignis_segment_profile *p){
  if(!supported(p))return 0;
  const auto trellis=linear_bytes();
  const auto attention=attention_bytes(p->max_keys);
  return trellis && attention ? intermediate_bytes()+aligned(trellis)+aligned(attention) : 0;
}
extern "C" int32_t ignis_lab_gqa_segment_prepare(){
  try{
    if(ignis_trellis_prepare())throw std::runtime_error(ignis_trellis_last_error());
#ifndef IGNIS_SEGMENT_BUILTIN
    if(ignis_lab_gqa_prepare())throw std::runtime_error(ignis_lab_gqa_error());
#endif
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes,
        ninfer::ops::rmsnorm_cta_bf16x2_kernel<ninfer::ops::RmsEpilogue::Offset,512,8,true>));
    check(cudaFuncGetAttributes(&attributes,
        ninfer::ops::rope_norm_fused_kernel<ninfer::ops::RopeKernelMode::Text1D,24,4,128>));
    return 0;
  }catch(const std::exception &e){segment_error=e.what();return -1;}
}
extern "C" const char *ignis_lab_gqa_segment_error(){return segment_error.c_str();}
extern "C" int32_t ignis_lab_gqa_segment_run(const ignis_gqa_segment_call *c,
    void *workspace,uint64_t bytes,void *raw_stream){
  try{
    const auto required=c?ignis_lab_gqa_segment_workspace(&c->profile):0;
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
    auto *cursor=static_cast<char *>(workspace);
    auto take=[&](uint32_t width){auto *p=cursor;cursor+=tensor_bytes(width);return p;};
    void *normalized=take(kHidden),*q=take(kQuery),*gate=take(kQuery),*k=take(kKv),*v=take(kKv);
    void *rq=take(kQuery),*rk=take(kKv),*attention=take(kQuery),*interleaved=take(2*kQuery);
    const ignis_trellis_workspace trellis{cursor,linear_bytes()};cursor+=aligned(trellis.bytes);
    const auto attention_capacity=attention_bytes(c->profile.max_keys);
    check(ninfer::pdl::launch_dependent({dim3(kTokens),dim3(512),0,stream},
        ninfer::ops::rmsnorm_cta_bf16x2_kernel<ninfer::ops::RmsEpilogue::Offset,512,8,true>,
        static_cast<const __nv_bfloat162 *>(c->input),static_cast<const __nv_bfloat162 *>(c->input_norm),
        static_cast<const __nv_bfloat162 *>(nullptr),static_cast<__nv_bfloat162 *>(normalized),
        int32_t(kHidden),int64_t(kTokens),c->profile.rms_eps));
    check_linear(ignis_trellis_linear(&c->q_proj,normalized,kTokens,interleaved,2*kQuery,
        IGNIS_TRELLIS_STORE,&trellis,stream));
    constexpr uint64_t half=256*2,rows=24*kTokens;
    check(cudaMemcpy2DAsync(q,half,interleaved,2*half,half,rows,cudaMemcpyDeviceToDevice,stream));
    check(cudaMemcpy2DAsync(gate,half,static_cast<const char *>(interleaved)+half,2*half,
        half,rows,cudaMemcpyDeviceToDevice,stream));
    check_linear(ignis_trellis_linear(&c->k_proj,normalized,kTokens,k,kKv,IGNIS_TRELLIS_STORE,&trellis,stream));
    check_linear(ignis_trellis_linear(&c->v_proj,normalized,kTokens,v,kKv,IGNIS_TRELLIS_STORE,&trellis,stream));
    ninfer::ops::RopeFrequencies rope;
    std::copy_n(c->rope_inv_frequency,128,rope.inv_frequency);
    rope.attention_factor=c->rope_attention_factor;
    check(ninfer::pdl::launch_dependent({dim3((24+4)*kTokens/4),dim3(128),0,stream},
        ninfer::ops::rope_norm_fused_kernel<ninfer::ops::RopeKernelMode::Text1D,24,4,128>,
        static_cast<const __nv_bfloat16 *>(q),static_cast<const __nv_bfloat16 *>(k),
        static_cast<const __nv_bfloat162 *>(c->query_norm),static_cast<const __nv_bfloat162 *>(c->key_norm),
        c->profile.rms_eps,c->rope_positions,rope,static_cast<__nv_bfloat16 *>(rq),
        static_cast<__nv_bfloat16 *>(rk),int32_t(kTokens)));
#ifdef IGNIS_SEGMENT_BUILTIN
    ignis::segment_gqa_builtin_attention(*c,rq,rk,v,gate,attention,cursor,attention_capacity,stream);
#else
    const ignis_gqa_candidate_call attention_call{IGNIS_GQA_CANDIDATE_ABI,sizeof(ignis_gqa_candidate_call),
        c->profile.max_keys,c->logical_pages,c->physical_pages,c->profile.slots,0.0625f,
        rq,rk,v,gate,c->positions,c->valid_columns,c->table_rows,c->block_tables,
        c->k_pages,c->v_pages,c->k_meta,c->v_meta,c->residual_k,c->residual_v,c->ring,attention};
    if(ignis_lab_gqa_run(&attention_call,cursor,attention_capacity,stream))
      throw std::runtime_error(ignis_lab_gqa_error());
#endif
    check(cudaMemcpyAsync(c->output,c->input,uint64_t(kHidden)*kTokens*2,cudaMemcpyDeviceToDevice,stream));
    check_linear(ignis_trellis_linear(&c->output_proj,attention,kTokens,c->output,kHidden,
        IGNIS_TRELLIS_ADD,&trellis,stream));
    check(cudaGetLastError());return 0;
  }catch(const std::exception &e){segment_error=e.what();return -1;}
}
