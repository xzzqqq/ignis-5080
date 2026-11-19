// Retained BF16 workspace shared by the producer and its offline diagnostics.
// No CUDA dependency: layout/retention contracts can be checked on the CPU.
#pragma once
#include "ignis_segment_candidate.h"

namespace ignis_gqa_segment_workspace {
constexpr uint32_t hidden=5120,tokens=8,query=6144,kv=1024;
constexpr uint64_t aligned(uint64_t n){return (n+255)&~uint64_t(255);}
constexpr uint64_t tensor_bytes(uint32_t width){return aligned(uint64_t(width)*tokens*2);}
struct Layout {
  uint64_t normalized=0;
  uint64_t q=normalized+tensor_bytes(hidden);
  uint64_t gate=q+tensor_bytes(query);
  uint64_t k=gate+tensor_bytes(query);
  uint64_t v=k+tensor_bytes(kv);
  uint64_t rq=v+tensor_bytes(kv);
  uint64_t rk=rq+tensor_bytes(query);
  uint64_t attention=rk+tensor_bytes(kv);
  uint64_t interleaved=attention+tensor_bytes(query);
  uint64_t bytes=interleaved+tensor_bytes(2*query);
  void *at(void *workspace,uint64_t offset)const{
    return static_cast<char *>(workspace)+offset;
  }
};
// Select alongside the producing implementation, never by guessed offsets or
// the presence of spare allocations. InterleavedHq leaves q/gate slots unwritten
// and rq contains BF16 norm -> RoPE -> HQ FWHT, not just norm -> RoPE.
enum class Retained { SplitRope, InterleavedHq };
inline int32_t tensors(const Layout &l,Retained retained,const void *workspace,
    uint64_t bytes,ignis_lab_segment_tensor *out,uint32_t capacity){
  const uint32_t count=retained==Retained::SplitRope?9:7;
  if(!workspace || bytes<l.bytes || !out || capacity<count)return -1;
  uint32_t n=0;
  auto add=[&](const char *name,uint64_t offset,uint32_t width,int32_t d0,int32_t d1,int32_t d2){
    out[n++]={name,static_cast<const char *>(workspace)+offset,
        uint64_t(width)*tokens*2,0,{d0,d1,d2,1}};
  };
  add("normalized_input",l.normalized,hidden,hidden,tokens,1);
  if(retained==Retained::SplitRope){
    add("query",l.q,query,256,24,tokens);
    add("gate",l.gate,query,256,24,tokens);
  }
  add("key",l.k,kv,256,4,tokens);
  add("value",l.v,kv,256,4,tokens);
  add(retained==Retained::SplitRope?"rotated_query":"hq_transformed_query",l.rq,query,256,24,tokens);
  add("rotated_key",l.rk,kv,256,4,tokens);
  add("attention",l.attention,query,query,tokens,1);
  add("interleaved_query_gate",l.interleaved,2*query,512,24,tokens);
  return int32_t(n);
}
} // namespace ignis_gqa_segment_workspace
