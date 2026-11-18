#include "ignis_mlp_segment.h"
#include "lab_mlp_segment_policy.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "core/pdl.cuh"
#include <algorithm>
#include <string>
#include <stdexcept>
namespace {
thread_local std::string error;
uint64_t align256(uint64_t n){return (n+255)&~255ull;}
uint64_t backend(const ignis_segment_profile &p){
  return std::max(ignis_trellis_workspace_bytes(p.hidden,p.intermediate,IGNIS_LAB_MLP_SPLIT_SWIGLU?1:2,p.tokens),
      ignis_trellis_workspace_bytes(p.intermediate,p.hidden,1,p.tokens));
}
__global__ void ignis_lab_mlp_split_swiglu(__nv_bfloat16 *gate,const __nv_bfloat16 *up,uint64_t n){
  ninfer::pdl::sync();const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n){float g=__bfloat162float(gate[i]);gate[i]=__float2bfloat16_rn((g/(1.0f+expf(-g)))*__bfloat162float(up[i]));}
  ninfer::pdl::publish();
}
void ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
}
extern "C" const char *ignis_lab_mlp_segment_error(){return error.c_str();}
extern "C" int32_t ignis_lab_mlp_segment_prepare(){
  if(ignis_trellis_prepare())return -1;
#if IGNIS_LAB_MLP_SPLIT_SWIGLU
  cudaFuncAttributes attributes{};auto rc=cudaFuncGetAttributes(&attributes,ignis_lab_mlp_split_swiglu);
  if(rc!=cudaSuccess){error=cudaGetErrorString(rc);return -1;}
#endif
  return 0;
}
extern "C" uint64_t ignis_lab_mlp_segment_workspace(const ignis_segment_profile *p){
  if(!p || p->kind!=IGNIS_SEGMENT_MLP || p->tokens!=8 || p->batch!=1 || p->hidden!=5120 || !p->intermediate)return 0;
  return align256(2ull*p->hidden*p->tokens)+align256(2ull*p->intermediate*p->tokens)*(IGNIS_LAB_MLP_SPLIT_SWIGLU?2:1)+align256(backend(*p));
}
extern "C" int32_t ignis_lab_mlp_segment_run(const ignis_mlp_segment_call *c,void *base,uint64_t bytes,void *raw_stream){
  try{
    if(!c || !base || bytes<ignis_lab_mlp_segment_workspace(&c->profile) || !ignis_lab_mlp_segment_workspace(&c->profile) ||
       !c->norm || !c->residual)throw std::runtime_error("invalid MLP segment invocation");
    const auto &p=c->profile;auto stream=static_cast<cudaStream_t>(raw_stream);
    auto *post=static_cast<char *>(base);auto *fused=post+align256(2ull*p.hidden*p.tokens);
    auto *up=fused+align256(2ull*p.intermediate*p.tokens);
    ignis_trellis_workspace ws{up+(IGNIS_LAB_MLP_SPLIT_SWIGLU?align256(2ull*p.intermediate*p.tokens):0),backend(p)};
    using namespace ninfer::ops;
    ck(ninfer::pdl::launch_dependent({dim3(p.tokens),dim3(512),0,stream},
        rmsnorm_cta_bf16x2_kernel<RmsEpilogue::Offset,512,8,true>,
        static_cast<const __nv_bfloat162 *>(c->residual),static_cast<const __nv_bfloat162 *>(c->norm),
        static_cast<const __nv_bfloat162 *>(nullptr),reinterpret_cast<__nv_bfloat162 *>(post),
        int(p.hidden),int64_t(p.tokens),p.rms_eps));
#if IGNIS_LAB_MLP_SPLIT_SWIGLU
    if(ignis_trellis_linear(&c->gate,post,p.tokens,fused,p.intermediate,IGNIS_TRELLIS_STORE,&ws,stream) ||
       ignis_trellis_linear(&c->up,post,p.tokens,up,p.intermediate,IGNIS_TRELLIS_STORE,&ws,stream))
      throw std::runtime_error(ignis_trellis_last_error());
    const uint64_t elements=uint64_t(p.intermediate)*p.tokens;
    ck(ninfer::pdl::launch_dependent({dim3((elements+255)/256),dim3(256),0,stream},
        ignis_lab_mlp_split_swiglu,reinterpret_cast<__nv_bfloat16 *>(fused),
        reinterpret_cast<const __nv_bfloat16 *>(up),elements));
#else
    if(ignis_trellis_linear_swiglu(&c->gate,&c->up,post,p.tokens,fused,p.intermediate,&ws,stream))
      throw std::runtime_error(ignis_trellis_last_error());
#endif
    if(ignis_trellis_linear(&c->down,fused,p.tokens,c->residual,p.hidden,IGNIS_TRELLIS_ADD,&ws,stream))
      throw std::runtime_error(ignis_trellis_last_error());
    return 0;
  }catch(const std::exception &e){error=e.what();return -1;}
}
