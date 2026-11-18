// Host binding for the finite attention sublayer. No intermediate tensor is
// part of its replay schema; a candidate may replace the complete chain.
#include "lab_segment.h"
#include "ignis_gqa_segment.h"
#include "ignis_seq_internal.h"
#include "model_internal.h"
#include "ninfer/ops/gqa_attention.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ignis {
// The app baseline keeps the resident attention route. The DSO's default
// instead composes its independently editable attention candidate, including
// the pre-existing candidate-only scheduling policy.
uint64_t segment_gqa_builtin_attention_workspace(uint32_t max_keys){
  return ninfer::ops::gqa_attention_workspace_capacity_bytes(24,ninfer::DType::U8,
      {.min_visible_keys=1,.max_visible_keys=max_keys},1,8,8);
}
void segment_gqa_builtin_attention(const ignis_gqa_segment_call &c,const void *q,const void *k,
    const void *v,const void *gate,void *output,void *workspace,uint64_t bytes,cudaStream_t stream){
  const int32_t slots=static_cast<int32_t>(c.profile.slots);
  const int32_t physical=static_cast<int32_t>(c.physical_pages);
  ninfer::PagedKVBatchLayerView cache;
  cache.dtype=ninfer::DType::U8;cache.quant_group=32;cache.head_dim=256;cache.num_kv_heads=4;
  cache.k_pages=ninfer::Tensor(c.k_pages,ninfer::DType::U8,{64,64,4,physical});
  cache.v_pages=ninfer::Tensor(c.v_pages,ninfer::DType::U8,{64,64,4,physical});
  cache.k_scale_pages=ninfer::Tensor(c.k_meta,ninfer::DType::U8,{8,64,4,physical});
  cache.v_scale_pages=ninfer::Tensor(c.v_meta,ninfer::DType::U8,{8,64,4,physical});
  cache.residual_k=ninfer::Tensor(c.residual_k,ninfer::DType::BF16,{256,4,544,slots});
  cache.residual_v=ninfer::Tensor(c.residual_v,ninfer::DType::BF16,{256,4,544,slots});
  cache.ring_valid=ninfer::Tensor(c.ring,ninfer::DType::I32,{16,slots});
  cache.block_tables=ninfer::Tensor(const_cast<int32_t *>(c.block_tables),ninfer::DType::I32,
                                  {static_cast<int32_t>(c.logical_pages),slots});
  const ninfer::Tensor query(const_cast<void *>(q),ninfer::DType::BF16,{256,24,8,1});
  const ninfer::Tensor key(const_cast<void *>(k),ninfer::DType::BF16,{256,4,8,1});
  const ninfer::Tensor value(const_cast<void *>(v),ninfer::DType::BF16,{256,4,8,1});
  const ninfer::Tensor output_gate(const_cast<void *>(gate),ninfer::DType::BF16,{256,24,8,1});
  const ninfer::Tensor positions(const_cast<int32_t *>(c.positions),ninfer::DType::I32,{8,1});
  const ninfer::Tensor valid=c.valid_columns
      ? ninfer::Tensor(const_cast<int32_t *>(c.valid_columns),ninfer::DType::I32,{1}) : ninfer::Tensor{};
  const ninfer::Tensor rows(const_cast<int32_t *>(c.table_rows),ninfer::DType::I32,{1});
  ninfer::Tensor out(output,ninfer::DType::BF16,{256,24,8,1});
  ninfer::DeviceArena arena(ninfer::DeviceSpan{workspace,static_cast<size_t>(bytes)});
  ninfer::ops::gqa_attention(query,key,value,positions,valid,rows,output_gate,0.0625f,cache,
      {.min_visible_keys=1,.max_visible_keys=c.profile.max_keys},arena,out,stream);
}
ignis_segment_profile segment_gqa_profile(const ignis_model &m,const ignis_seq_pool &p){
  return {IGNIS_SEGMENT_GQA,8,1,static_cast<uint32_t>(m.hidden),0,m.max_context_tokens,
      static_cast<uint32_t>(p.kv_pool.table_row_count()),0,m.rms_norm_eps};
}
uint64_t segment_gqa_workspace(const ignis_segment_profile &p){
  return ignis_lab_gqa_segment_workspace(&p);
}
namespace {
const ninfer::Tensor &require(const std::vector<SegmentBinding> &bindings,size_t index,
    const char *name,uint32_t flags,ninfer::DType dtype,
    std::initializer_list<int32_t> shape,bool optional=false){
  const auto &b=bindings.at(index);
  if(b.name!=name || b.flags!=flags || b.tensor.dtype!=dtype)
    throw std::runtime_error("GQA segment binding metadata mismatch: "+std::string(name));
  if(optional && !b.tensor.data){
    // The only optional tensor is the dense mask, represented as null I32[1].
    if(b.tensor.ne[0]!=1 || b.tensor.ne[1]!=1 || b.tensor.ne[2]!=1 || b.tensor.ne[3]!=1)
      throw std::runtime_error("GQA segment invalid dense mask metadata");
    return b.tensor;
  }
  if(!b.tensor.data || !b.tensor.is_contiguous())
    throw std::runtime_error("GQA segment missing/noncontiguous binding: "+std::string(name));
  size_t d=0;
  for(auto n:shape)if(b.tensor.ne[d++]!=n)
    throw std::runtime_error("GQA segment binding shape mismatch: "+std::string(name));
  while(d<4)if(b.tensor.ne[d++]!=1)
    throw std::runtime_error("GQA segment binding rank mismatch: "+std::string(name));
  return b.tensor;
}
}
void segment_gqa_invoke(const ignis_model &m,uint32_t layer,const ignis_segment_profile &p,
    const std::vector<SegmentBinding> &b,const ignis_segment_candidate_api *api,
    void *workspace,uint64_t bytes,cudaStream_t stream){
  if(layer>=m.layers.size() || m.layers[layer].kind!=IGNIS_LAYER_GQA || !m.is_trellis_body ||
      !segment_gqa_workspace(p) || p.hidden!=m.hidden || p.max_keys!=m.max_context_tokens ||
      p.rms_eps!=m.rms_norm_eps || b.size()!=14)
    throw std::runtime_error("unsupported GQA segment profile/binding inventory");
  const int32_t slots=static_cast<int32_t>(p.slots);
  if(slots<=0 || b[5].tensor.ne[0]<=0 || b[6].tensor.ne[3]<=0)
    throw std::runtime_error("GQA segment invalid state extents");
  const int32_t logical=b[5].tensor.ne[0],physical=b[6].tensor.ne[3];
  if(uint64_t(logical)*64<p.max_keys)
    throw std::runtime_error("GQA segment block table does not cover the execution envelope");
  ignis_gqa_segment_call c{};c.profile=p;
  c.input=require(b,0,"input",SEGMENT_INPUT,ninfer::DType::BF16,{5120,8}).data;
  c.positions=static_cast<const int32_t *>(require(b,1,"positions",SEGMENT_INPUT,ninfer::DType::I32,{8,1}).data);
  c.rope_positions=static_cast<const int32_t *>(require(b,2,"rope_positions",SEGMENT_INPUT,ninfer::DType::I32,{8}).data);
  c.valid_columns=static_cast<const int32_t *>(require(b,3,"valid_columns",SEGMENT_INPUT,ninfer::DType::I32,{1},true).data);
  c.table_rows=static_cast<const int32_t *>(require(b,4,"table_rows",SEGMENT_INPUT,ninfer::DType::I32,{1}).data);
  c.block_tables=static_cast<const int32_t *>(require(b,5,"block_tables",SEGMENT_INPUT,ninfer::DType::I32,{logical,slots}).data);
  c.k_pages=require(b,6,"k_pages",SEGMENT_MUTABLE,ninfer::DType::U8,{64,64,4,physical}).data;
  c.v_pages=require(b,7,"v_pages",SEGMENT_MUTABLE,ninfer::DType::U8,{64,64,4,physical}).data;
  c.k_meta=require(b,8,"k_meta",SEGMENT_MUTABLE,ninfer::DType::U8,{8,64,4,physical}).data;
  c.v_meta=require(b,9,"v_meta",SEGMENT_MUTABLE,ninfer::DType::U8,{8,64,4,physical}).data;
  c.residual_k=require(b,10,"residual_k",SEGMENT_MUTABLE,ninfer::DType::BF16,{256,4,544,slots}).data;
  c.residual_v=require(b,11,"residual_v",SEGMENT_MUTABLE,ninfer::DType::BF16,{256,4,544,slots}).data;
  c.ring=static_cast<uint32_t *>(require(b,12,"ring",SEGMENT_MUTABLE,ninfer::DType::I32,{16,slots}).data);
  c.output=require(b,13,"output",SEGMENT_OUTPUT,ninfer::DType::BF16,{5120,8}).data;
  c.logical_pages=static_cast<uint32_t>(logical);c.physical_pages=static_cast<uint32_t>(physical);
  const auto &w=m.layers[layer].gqa;
  c.input_norm=w.input_norm.qdata;c.query_norm=w.query_norm.qdata;c.key_norm=w.key_norm.qdata;
  c.q_proj=w.trellis.q_proj;c.k_proj=w.trellis.k_proj;c.v_proj=w.trellis.v_proj;
  c.output_proj=w.trellis.tail.output;
  std::copy_n(m.text_rope.inv_frequency,128,c.rope_inv_frequency);
  c.rope_attention_factor=m.text_rope.attention_factor;
  if(api){
    if(!api->run_gqa || api->run_gqa(&c,workspace,bytes,stream))
      throw std::runtime_error(std::string("GQA segment candidate: ")+api->last_error());
  }else if(ignis_lab_gqa_segment_run(&c,workspace,bytes,stream)){
    throw std::runtime_error(std::string("GQA segment builtin: ")+ignis_lab_gqa_segment_error());
  }
}
}
