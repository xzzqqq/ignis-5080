#include "lab_segment.h"
#include "ignis_mlp_segment.h"
#include "model_internal.h"
#include "trellis_body.h"
#include "ninfer/ops/rmsnorm.h"
#include "lab_profile.h"
#include <stdexcept>
namespace {
const TrellisTail &tail(const ignis_model &m,uint32_t layer){
  if(layer>=m.layers.size())throw std::runtime_error("MLP segment layer out of range");
  return m.layers[layer].kind==IGNIS_LAYER_GQA?m.layers[layer].gqa.trellis.tail:m.layers[layer].gdn.trellis.tail;
}
const void *norm(const ignis_model &m,uint32_t layer){
  return m.layers[layer].kind==IGNIS_LAYER_GQA?m.layers[layer].gqa.post_attention_norm.qdata:m.layers[layer].gdn.post_attention_norm.qdata;
}
}
namespace ignis {
ignis_segment_profile segment_mlp_profile(const ignis_model &m,const ignis_seq_pool &){
  return {IGNIS_SEGMENT_MLP,8,1,static_cast<uint32_t>(m.hidden),tail(m,0).gate_proj.out,m.max_context_tokens,1,0,m.rms_norm_eps};
}
uint64_t segment_mlp_workspace(const ignis_segment_profile &p){return ignis_lab_mlp_segment_workspace(&p);}
void segment_mlp_invoke(const ignis_model &m,uint32_t layer,const ignis_segment_profile &p,
    const std::vector<SegmentBinding> &b,const ignis_segment_candidate_api *api,void *workspace,uint64_t bytes,cudaStream_t stream){
  if(!m.is_trellis_body || p.kind!=IGNIS_SEGMENT_MLP || p.hidden!=m.hidden || p.tokens!=8 || p.batch!=1 || p.flags || p.slots!=1 ||
      p.max_keys!=m.max_context_tokens || p.rms_eps!=m.rms_norm_eps ||
      b.size()!=1 || b[0].name!="residual" || b[0].flags!=(SEGMENT_MUTABLE|SEGMENT_OUTPUT) ||
      b[0].tensor.dtype!=ninfer::DType::BF16 || !b[0].tensor.data || !b[0].tensor.is_contiguous() ||
      b[0].tensor.ne[0]!=int(p.hidden) || b[0].tensor.ne[1]!=8 || b[0].tensor.ne[2]!=1 || b[0].tensor.ne[3]!=1)
    throw std::runtime_error("invalid MLP segment boundary");
  const auto &t=tail(m,layer);if(p.intermediate!=t.gate_proj.out)throw std::runtime_error("MLP segment intermediate mismatch");
  ignis_mlp_segment_call call{p,norm(m,layer),t.gate_proj,t.up_proj,t.down,b[0].tensor.data};
  const auto run=api?api->run_mlp:ignis_lab_mlp_segment_run;
  if(run(&call,workspace,bytes,stream))throw std::runtime_error(api?api->last_error():ignis_lab_mlp_segment_error());
}
void segment_mlp(const ignis_model &m,uint32_t layer,const ninfer::Tensor &post_norm,
    ninfer::Tensor &residual,ninfer::Tensor &post,ninfer::Tensor &fused,cudaStream_t stream){
  const auto &t=tail(m,layer);
  void *record_ticket=nullptr;
  if(lab_segment_should_run(m,IGNIS_SEGMENT_MLP,layer,uint32_t(residual.ne[1]),1)){
    ignis_segment_profile p{IGNIS_SEGMENT_MLP,8,1,static_cast<uint32_t>(m.hidden),t.gate_proj.out,m.max_context_tokens,1,0,m.rms_norm_eps};
    const std::vector<SegmentBinding> bindings{{"residual",residual,SEGMENT_MUTABLE|SEGMENT_OUTPUT}};
    if(lab_segment_is_recording(m))record_ticket=lab_segment_before(m,layer,p,bindings,stream);
    else {lab_segment_execute(m,layer,p,bindings,stream);return;}
  }
  IGNIS_PROFILE_OP("rmsnorm",ninfer::ops::rmsnorm(residual,post_norm,m.rms_norm_eps,true,post,stream));
  IGNIS_PROFILE_OP("mlp.swiglu",trellis_body::swiglu(m,t.gate_proj,t.up_proj,post,fused,stream));
  IGNIS_PROFILE_OP("linear.down",trellis_body::linear(m,t.down,fused,residual.data,m.hidden,IGNIS_TRELLIS_ADD,stream));
  lab_segment_after(m,record_ticket,stream);
}
}
