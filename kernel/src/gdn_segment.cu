#include "lab_segment.h"
#include "ignis_gdn_segment.h"
#include "ignis_seq_internal.h"
#include "model_internal.h"
#include <algorithm>
#include <initializer_list>
#include <stdexcept>

namespace ignis {
ignis_segment_profile segment_gdn_profile(const ignis_model &model,const ignis_seq_pool &pool){
  ignis_segment_profile p{};p.kind=IGNIS_SEGMENT_GDN;
  p.tokens=model.verify?model.verify->window+1:0;p.batch=1;p.hidden=model.hidden;
  p.slots=pool.gdn_pool.slot_count();p.rms_eps=model.rms_norm_eps;
  // A nonzero flag makes unsupported geometries fail the finite query and
  // leaves the ordinary layer implementation in charge of those paths.
  const auto &s=pool.gdn_pool.spec;
  if(!model.is_trellis_body || s.key_head_dim!=128 || s.value_head_dim!=128 ||
      s.value_heads!=48 || s.conv_channels!=10240)p.flags=1;
  return p;
}
uint64_t segment_gdn_workspace(const ignis_segment_profile &p){
  return ignis_lab_gdn_segment_workspace(&p);
}
void segment_gdn_invoke(const ignis_model &model,uint32_t layer,const ignis_segment_profile &p,
    const std::vector<SegmentBinding> &bindings,const ignis_segment_candidate_api *api,
    void *workspace,uint64_t bytes,cudaStream_t stream){
  if(layer>=model.layers.size() || model.layers[layer].kind!=IGNIS_LAYER_GDN ||
      !model.is_trellis_body || bindings.size()!=10 || p.kind!=IGNIS_SEGMENT_GDN ||
      p.tokens!=8 || p.batch!=1 || p.hidden!=5120 || !p.slots || p.flags ||
      p.rms_eps!=model.rms_norm_eps)
    throw std::runtime_error("GDN segment profile or boundary inventory mismatch");
  using D=ninfer::DType;
  auto get=[&](size_t index,const char *name,D dtype,std::initializer_list<int32_t> shape,uint32_t flags)->void *{
    const auto &b=bindings[index];const ninfer::Tensor expected(nullptr,dtype,shape);
    if(b.name!=name || b.flags!=flags || b.tensor.dtype!=dtype || !b.tensor.data || !b.tensor.is_contiguous() ||
       !std::equal(b.tensor.ne,b.tensor.ne+4,expected.ne))
      throw std::runtime_error(std::string("GDN segment invalid boundary: ")+name);
    return b.tensor.data;
  };
  ignis_gdn_segment_call c{};c.profile=p;
  c.residual_in=get(0,"residual_in",D::BF16,{5120,8},SEGMENT_INPUT);
  c.conv_states=get(1,"conv_states",D::BF16,{10240,3,int32_t(p.slots)},SEGMENT_INPUT);
  c.recurrent_states=get(2,"recurrent_states",D::FP32,{128,128,48,int32_t(p.slots)},SEGMENT_INPUT);
  c.lane_slots=static_cast<const int32_t *>(get(3,"lane_slots",D::I32,{1},SEGMENT_INPUT));
  c.valid_columns=static_cast<const int32_t *>(get(4,"valid_columns",D::I32,{1},SEGMENT_INPUT));
  c.residual_out=get(5,"residual_out",D::BF16,{5120,8},SEGMENT_OUTPUT);
  constexpr uint32_t record_flags=SEGMENT_MUTABLE|SEGMENT_OUTPUT;
  c.conv_record=get(6,"conv_record",D::BF16,{10240,8,1},record_flags);
  c.key_record=get(7,"key_record",D::BF16,{128,16,8,1},record_flags);
  c.value_record=get(8,"value_record",D::BF16,{128,48,8,1},record_flags);
  c.gate_record=get(9,"gate_record",D::FP32,{2,48,8,1},record_flags);
  const auto &w=model.layers[layer].gdn;
  c.qkv_weight=w.trellis.in_proj_qkv;c.z_weight=w.trellis.in_proj_z;c.output_weight=w.trellis.tail.output;
  c.input_norm=w.input_norm.qdata;c.convolution=w.convolution.qdata;c.ab_projection=w.a_b_projection.qdata;
  c.a_log=w.a_log.qdata;c.dt_bias=w.dt_bias.qdata;c.gdn_norm=w.norm.qdata;
  if(api){
    if(!api->run_gdn || api->run_gdn(&c,workspace,bytes,stream))
      throw std::runtime_error(api->last_error?api->last_error():"candidate GDN segment failed");
  }else if(ignis_lab_gdn_segment_run(&c,workspace,bytes,stream))
    throw std::runtime_error(ignis_lab_gdn_segment_error());
}
} // namespace ignis
