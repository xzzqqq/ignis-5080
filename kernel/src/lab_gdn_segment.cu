// Complete finite C1/T8 verify GDN launcher. This translation unit is built
// into both the builtin and immutable candidate DSO. Vendored primitives are
// instantiated in a private namespace: the candidate owns every launch and
// temporary, without calling back into the resident layer implementation.
#include "ignis_gdn_segment.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#define ninfer ignis_lab_gdn_private
#include "ops/kernel/rmsnorm.cuh"
#include "ops/gdn_input_proj/gdn_projected_conv.cu"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"
#undef ninfer
#include "trellis_decode_helpers.cuh"

namespace ignis_lab_gdn_private {
void cuda_check(cudaError_t rc,const char *,const char *,int) {
  if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));
}
}

extern "C" int32_t ignis_gdn_joint_projection(const ignis_trellis_weight *,const ignis_trellis_weight *,const void *,void *,void *,const ignis_trellis_workspace *,const ignis_trellis_workspace *,void *);
extern "C" int32_t ignis_gdn_rotation_params(const ignis_trellis_weight *,const ignis_trellis_workspace *,ignis_tl::RotateParams *);
extern "C" int32_t ignis_gdn_output_from(const ignis_trellis_weight *,const void *,const void *,void *,const ignis_trellis_workspace *,void *);
namespace {
namespace n=ignis_lab_gdn_private;
thread_local std::string error;
constexpr uint64_t align256(uint64_t n){return (n+255)&~255ull;}
void ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
bool supports(const ignis_segment_profile *p){
  return p && p->kind==IGNIS_SEGMENT_GDN && p->tokens==8 && p->batch==1 &&
    p->hidden==5120 && p->slots>0 && p->flags==0 &&
    std::isfinite(p->rms_eps) && p->rms_eps>0;
}
uint64_t trellis_bytes(){
  return std::max({ignis_trellis_workspace_bytes(5120,10240,1,8),
    ignis_trellis_workspace_bytes(5120,6144,1,8),
    ignis_trellis_workspace_bytes(6144,5120,1,8)});
}
// No Tensor member implementation is linked into this standalone candidate.
n::Tensor tensor(void *data,n::DType dtype,std::initializer_list<int32_t> shape){
  n::Tensor t;t.data=data;t.dtype=dtype;
  int i=0;for(auto d:shape)t.ne[i++]=d;
  int64_t stride=dtype==n::DType::FP32 || dtype==n::DType::I32?4:2;
  for(i=0;i<4;++i){t.nb[i]=stride;stride*=t.ne[i];}
  return t;
}
n::Tensor input(const void *data,n::DType dtype,std::initializer_list<int32_t> shape){
  return tensor(const_cast<void *>(data),dtype,shape);
}
struct Layout {
  uint64_t used=0;
  void *base;
  explicit Layout(void *p):base(p){}
  void *take(uint64_t bytes){auto offset=used;used+=align256(bytes);return base?static_cast<char *>(base)+offset:nullptr;}
};
struct Buffers {
  ignis_trellis_workspace trellis,second;
  n::Tensor h,q,k,v,z,g,beta,recurrent;
  void *partial;
  uint64_t bytes;
  explicit Buffers(void *base):trellis{},h{},q{},k{},v{},z{},g{},beta{},recurrent{} {
    Layout l(base);
    trellis.bytes=trellis_bytes();trellis.base=l.take(trellis.bytes);
    second.bytes=trellis.bytes;second.base=l.take(second.bytes);
    h=tensor(l.take(5120*8*2),n::DType::BF16,{5120,8});
    q=tensor(l.take(2048*8*2),n::DType::BF16,{2048,8,1});
    k=tensor(l.take(2048*8*2),n::DType::BF16,{2048,8,1});
    v=tensor(l.take(6144*8*2),n::DType::BF16,{6144,8,1});
    z=tensor(l.take(6144*8*2),n::DType::BF16,{6144,8});
    g=tensor(l.take(48*8*4),n::DType::FP32,{48,8});
    beta=tensor(l.take(48*8*4),n::DType::FP32,{48,8});
    recurrent=tensor(l.take(6144*8*2),n::DType::BF16,{128,48,8,1});
    partial=l.take(10*8*96*4);bytes=l.used;
  }
};
__global__ void __launch_bounds__(64) gated_rotation(const __nv_bfloat162 *x,const __nv_bfloat162 *weight,
    const __nv_bfloat162 *z,float eps,const __grid_constant__ ignis_tl::RotateParams rotation){
  n::pdl::sync();
  // Output decode may prefetch weights now; its dependency wait still
  // guards all reads of this kernel's XR/inverse-scale writes.
  n::pdl::trigger_dependents();
  const int lane=threadIdx.x&31,row=int(blockIdx.x)*2+(threadIdx.x>>5);
  if(row>=48*8)return;
  float a[4],w[4],gate[4],sum=0.0f;
#pragma unroll
  for(int k=0;k<2;++k){
    const int pair=lane+k*32;
    const float2 v=__bfloat1622float2(x[row*64+pair]);
    const float2 wt=__bfloat1622float2(weight[pair]),g=__bfloat1622float2(z[row*64+pair]);
    a[2*k]=v.x;a[2*k+1]=v.y;w[2*k]=wt.x;w[2*k+1]=wt.y;gate[2*k]=g.x;gate[2*k+1]=g.y;
    sum+=v.x*v.x+v.y*v.y;
  }
  sum=n::ops::warp_reduce_sum(sum);
  float inv=lane==0?rsqrtf(sum/128.0f+eps):0.0f;inv=__shfl_sync(0xffffffffu,inv,0);
#pragma unroll
  for(int q=0;q<4;++q)a[q]=n::ops::rmsnorm_epilogue<n::ops::RmsEpilogue::Gated>(a[q],inv,w[q],gate[q]);
  float v[4];
#pragma unroll
  for(int q=0;q<4;++q){
    const int source=(2*lane+(q>>1))&31;
    const float lo=__shfl_sync(0xffffffffu,a[q&1],source);
    const float hi=__shfl_sync(0xffffffffu,a[2+(q&1)],source);
    v[q]=lane<16?lo:hi;
  }
  ignis_tl::rotate_bf16_block(rotation,v,row%48,row/48);

}

template<bool Masked>
void recurrence(const ignis_gdn_segment_call &c,Buffers &b,cudaStream_t stream){
  namespace r=n::ops::detail::gated_delta_net;
  r::RecordAccess<Masked> access{
    static_cast<const __nv_bfloat16 *>(b.q.data),static_cast<const __nv_bfloat16 *>(b.k.data),
    static_cast<const __nv_bfloat16 *>(b.v.data),static_cast<const float *>(b.g.data),
    static_cast<const float *>(b.beta.data),static_cast<const float *>(c.recurrent_states),
    Masked?c.valid_columns:nullptr,c.lane_slots,static_cast<__nv_bfloat16 *>(c.key_record),
    static_cast<__nv_bfloat16 *>(c.value_record),static_cast<uint2 *>(c.gate_record),
    static_cast<__nv_bfloat16 *>(b.recurrent.data),r::head_map::of(16,48),8,
    int64_t(128)*128*48,1.0f/std::sqrt(128.0f)};
  ck(n::pdl::launch_dependent({dim3(48,1,128/r::kBlockDv),dim3(32,r::kNumWarps),0,stream},
      r::recurrent_record_kernel<Masked>,access));
}
}
extern "C" uint64_t ignis_lab_gdn_segment_workspace(const ignis_segment_profile *p){
  if(!supports(p) || !trellis_bytes())return 0;
  return Buffers(nullptr).bytes;
}
extern "C" const char *ignis_lab_gdn_segment_error(){return error.c_str();}
extern "C" int32_t ignis_lab_gdn_segment_prepare(){
  try{
    if(ignis_trellis_prepare())throw std::runtime_error(ignis_trellis_last_error());
    cudaFuncAttributes a{};
    ck(cudaFuncGetAttributes(&a,n::ops::rmsnorm_cta_bf16x2_kernel<n::ops::RmsEpilogue::Offset,512,8,true>));
    ck(cudaFuncGetAttributes(&a,gated_rotation));
    ck(cudaFuncGetAttributes(&a,n::ops::detail::gdn_projected_conv_kernel<10240,2048,2048,6144,0,n::ops::detail::NoHistoryPublish>));
    ck(cudaFuncGetAttributes(&a,n::ops::detail::bf16_gdn_gating_proj_small_t_partial_kernel<8,512,4>));
    ck(cudaFuncGetAttributes(&a,n::ops::detail::bf16_gdn_gating_proj_small_t_reduce_kernel));
    namespace r=n::ops::detail::gated_delta_net;
    ck(cudaFuncGetAttributes(&a,r::recurrent_record_kernel<true>));
    ck(cudaFuncGetAttributes(&a,r::recurrent_record_kernel<false>));
    return 0;
  }catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" int32_t ignis_lab_gdn_segment_run(const ignis_gdn_segment_call *c,void *workspace,uint64_t bytes,void *raw_stream){
  try{
    if(!c || !supports(&c->profile) || !workspace || !ignis_lab_gdn_segment_workspace(&c->profile) ||
       bytes<ignis_lab_gdn_segment_workspace(&c->profile) || !c->residual_in || !c->residual_out ||
       !c->input_norm || !c->convolution || !c->ab_projection || !c->a_log || !c->dt_bias || !c->gdn_norm ||
       !c->conv_states || !c->recurrent_states || !c->lane_slots || !c->conv_record ||
       !c->key_record || !c->value_record || !c->gate_record ||
       c->qkv_weight.in!=5120 || c->qkv_weight.out!=10240 || c->z_weight.in!=5120 ||
       c->z_weight.out!=6144 || c->output_weight.in!=6144 || c->output_weight.out!=5120)
      throw std::runtime_error("invalid C1/T8 verify GDN segment invocation");
    const auto stream=static_cast<cudaStream_t>(raw_stream);Buffers b(workspace);
    n::Weight a{},bw{};a.qdata=c->ab_projection;a.n=48;a.k=5120;a.shape[0]=48;a.shape[1]=5120;
    bw=a;bw.qdata=static_cast<const char *>(c->ab_projection)+48*5120*2;
    const auto alog=input(c->a_log,n::DType::FP32,{48}),bias=input(c->dt_bias,n::DType::FP32,{48});
    const auto gating = static_cast<cudaStream_t>(c->gating_stream);
    const auto fork = static_cast<cudaEvent_t>(c->gating_fork);
    const auto join = static_cast<cudaEvent_t>(c->gating_join);
    if (gating && (!fork || !join || gating == stream))
      throw std::runtime_error("invalid GDN gating branch");
    using n::ops::RmsEpilogue;
    ck(n::pdl::launch_dependent({dim3(8),dim3(512),0,stream},
        n::ops::rmsnorm_cta_bf16x2_kernel<RmsEpilogue::Offset,512,8,true>,
        static_cast<const __nv_bfloat162 *>(c->residual_in),static_cast<const __nv_bfloat162 *>(c->input_norm),
        static_cast<const __nv_bfloat162 *>(nullptr),static_cast<__nv_bfloat162 *>(b.h.data),5120,int64_t(8),c->profile.rms_eps));
    auto project_and_convolve = [&] {
      if(ignis_gdn_joint_projection(&c->qkv_weight,&c->z_weight,b.h.data,c->conv_record,b.z.data,
          &b.trellis,&b.second,stream))throw std::runtime_error(ignis_trellis_last_error());
      const auto conv=input(c->conv_record,n::DType::BF16,{10240,8,1});
      const auto conv_weight=input(c->convolution,n::DType::BF16,{10240,4});
      const auto states=input(c->conv_states,n::DType::BF16,{10240,3,int(c->profile.slots)});
      const auto valid=input(c->valid_columns,n::DType::I32,{1});
      const auto slots=input(c->lane_slots,n::DType::I32,{1});
      n::ops::detail::gdn_projected_conv_record_launch(conv,conv_weight,states,valid,slots,b.q,b.k,b.v,stream);
    };
    auto project_gates = [&](cudaStream_t target) {
      n::ops::detail::bf16_gdn_gating_proj_small_t_split10_launch(
          b.h,a,bw,alog,bias,b.partial,10*8*96*4,b.g,b.beta,target);
    };
    if (gating) {
      // Both paths read h. Only recurrence needs the outputs of both.
      ck(cudaEventRecord(fork,stream));
      ck(cudaStreamWaitEvent(gating,fork,0));
      auto rejoin = [&] {
        ck(cudaEventRecord(join,gating));
        ck(cudaStreamWaitEvent(stream,join,0));
      };
      try {
        project_gates(gating);
        project_and_convolve();
      } catch (...) {
        // Either path may have enqueued work before a launch failed.
        rejoin();
        throw;
      }
      rejoin();
    } else {
      project_and_convolve();
      project_gates(stream);
    }
    if(c->valid_columns)recurrence<true>(*c,b,stream);else recurrence<false>(*c,b,stream);
    ignis_tl::RotateParams rotation{};
    if(ignis_gdn_rotation_params(&c->output_weight,&b.trellis,&rotation))throw std::runtime_error(ignis_trellis_last_error());
    ck(n::pdl::launch_dependent({dim3(192),dim3(64),0,stream},gated_rotation,
      static_cast<const __nv_bfloat162 *>(b.recurrent.data),static_cast<const __nv_bfloat162 *>(c->gdn_norm),
      static_cast<const __nv_bfloat162 *>(b.z.data),c->profile.rms_eps,rotation));
    if(ignis_gdn_output_from(&c->output_weight,nullptr,c->residual_in,c->residual_out,&b.trellis,stream))
      throw std::runtime_error(ignis_trellis_last_error());
    ck(cudaGetLastError());return 0;
  }catch(const std::exception &e){error=e.what();return -1;}
}

extern "C" IGNIS_LAB_DIAGNOSTIC_EXPORT int32_t ignis_lab_gdn_segment_tensors_v1(const ignis_segment_profile *p,
    const void *workspace,uint64_t bytes,ignis_lab_segment_tensor *out,uint32_t capacity){
  if(!p || !workspace || !ignis_lab_gdn_segment_workspace(p) || bytes<ignis_lab_gdn_segment_workspace(p) || !out || capacity<8)return -1;
  Buffers b(const_cast<void *>(workspace));
  const char *names[]={"normalized_input","query","key","value","z","g","beta","recurrent"};
  const n::Tensor values[]={b.h,b.q,b.k,b.v,b.z,b.g,b.beta,b.recurrent};
  const uint32_t count=8;
  for(uint32_t i=0;i<count;++i){const auto &t=values[i];
    const uint64_t size=uint64_t(t.ne[0])*t.ne[1]*t.ne[2]*t.ne[3]*(t.dtype==n::DType::BF16?2:4);
    out[i]={names[i],t.data,size,t.dtype==n::DType::BF16?0u:3u,{t.ne[0],t.ne[1],t.ne[2],t.ne[3]}};
  }
  return count;
}
