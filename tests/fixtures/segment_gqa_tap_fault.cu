// Diagnostic-only provenance fixture. Link in place of lab_gqa_segment.cu in
// the normal candidate DSO, with the other candidate objects and ABI adapter.
// The retained raw Q/gate buffer changes before its normalization and
// attention; exports must therefore come from this candidate's own execution.
#include "ignis_gqa_segment.h"
#include <cuda_runtime.h>
extern "C" int32_t ignis_gqa_native_q_fixed_splits(const ignis_trellis_weight *,
    const void *,void *,const ignis_trellis_workspace *,void *);
extern "C" int32_t tap_fault_private_q(const ignis_trellis_weight *,
    const void *,void *,const ignis_trellis_workspace *,void *);
namespace {
int32_t tap_fault_linear(const ignis_trellis_weight *,const void *,uint32_t,void *,uint32_t,uint32_t,
    const ignis_trellis_workspace *,void *);
}
#define ignis_trellis_linear tap_fault_linear
#define ignis_gqa_native_q_fixed_splits tap_fault_private_q
#include "lab_gqa_segment.cu"
#undef ignis_trellis_linear
#undef ignis_gqa_native_q_fixed_splits
namespace {
__global__ void flip_raw_query(uint16_t *p){if(threadIdx.x==0)p[0]^=1;}
int32_t tap_fault_linear(const ignis_trellis_weight *w,const void *x,uint32_t tokens,void *y,uint32_t ldy,
    uint32_t epilogue,const ignis_trellis_workspace *workspace,void *stream){
  const auto rc=ignis_trellis_linear(w,x,tokens,y,ldy,epilogue,workspace,stream);
  if(!rc && w->in==5120 && w->out==12288 && tokens==8 && epilogue==IGNIS_TRELLIS_STORE){
    flip_raw_query<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(static_cast<uint16_t *>(y));
    if(cudaGetLastError()!=cudaSuccess)return -1;
  }
  return rc;
}
}

// The optimized producer uses the private S3 entry; retain the public hook
// above for split-Q/gate producers without changing either production path.
extern "C" int32_t tap_fault_private_q(const ignis_trellis_weight *w,const void *x,void *y,
    const ignis_trellis_workspace *workspace,void *stream){
  const auto rc=ignis_gqa_native_q_fixed_splits(w,x,y,workspace,stream);
  if(!rc){
    flip_raw_query<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(static_cast<uint16_t *>(y));
    if(cudaGetLastError()!=cudaSuccess)return -1;
  }
  return rc;
}
