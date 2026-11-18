#include "lab_segment.h"
#include "model_internal.h"
#include "ignis_mlp_segment.h"
#include "ignis_gqa_segment.h"
#include "ignis_gdn_segment.h"
#include <stdexcept>
namespace {
void ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
uint64_t builtin_query(const ignis_segment_profile &p){
  switch(p.kind){case IGNIS_SEGMENT_MLP:return ignis::segment_mlp_workspace(p);
    case IGNIS_SEGMENT_GQA:return ignis::segment_gqa_workspace(p);
    case IGNIS_SEGMENT_GDN:return ignis::segment_gdn_workspace(p);}
  throw std::runtime_error("unknown finite segment kind");
}
void allocate(ignis::SegmentWorkspaceSet &w,const ignis_segment_profile &p,uint64_t n,cudaStream_t stream){
  if(!n || n>(1ull<<30))throw std::runtime_error("segment invalid workspace query");
  auto &buffer=w.buffers.at(p.kind-1);
  if(!buffer || buffer->bytes<n)buffer=std::make_unique<ninfer::DeviceBuffer>(n);
  ck(cudaMemsetAsync(buffer->p,0,buffer->bytes,stream));
}
}
namespace ignis {
void segment_prepare_runtime(ignis_model &m,ignis_seq_pool &pool,TrellisCandidate *next){
  if(!m.segment_builtin_workspaces)m.segment_builtin_workspaces=std::make_shared<SegmentWorkspaceSet>();
  if(ignis_lab_mlp_segment_prepare() || ignis_lab_gqa_segment_prepare() || ignis_lab_gdn_segment_prepare())
    throw std::runtime_error("builtin segment preparation failed");
  const ignis_segment_profile profiles[]={segment_mlp_profile(m,pool),segment_gqa_profile(m,pool),segment_gdn_profile(m,pool)};
  for(const auto &p:profiles){
    const auto builtin=builtin_query(p);if(!builtin)continue;
    allocate(*m.segment_builtin_workspaces,p,builtin,m.stream);
    if(next && next->segments)allocate(next->segment_workspaces,p,next->segments->workspace_bytes(&p),m.stream);
  }
  ck(cudaStreamSynchronize(m.stream));
}
uint64_t segment_workspace_bytes(const ignis_model &m,const ignis_segment_profile &p){
  const auto *api=m.trellis_candidate?m.trellis_candidate->segments:nullptr;
  const auto n=api?api->workspace_bytes(&p):builtin_query(p);
  if(!n || n>(1ull<<30))throw std::runtime_error("segment invalid replay workspace query");
  return n;
}
void segment_invoke(const ignis_model &m,uint32_t layer,const ignis_segment_profile &p,
    const std::vector<SegmentBinding> &bindings,void *workspace,uint64_t bytes,cudaStream_t stream){
  const auto *api=m.trellis_candidate?m.trellis_candidate->segments:nullptr;
  if(!workspace){
    const auto *set=api?&m.trellis_candidate->segment_workspaces:m.segment_builtin_workspaces.get();
    if(!set || p.kind<1 || p.kind>3 || !set->buffers[p.kind-1])throw std::runtime_error("segment workspace not prepared");
    workspace=set->buffers[p.kind-1]->p;bytes=set->buffers[p.kind-1]->bytes;
  }
  if(bytes<segment_workspace_bytes(m,p))throw std::runtime_error("segment exceeds host-owned workspace");
  switch(p.kind){
    case IGNIS_SEGMENT_MLP:segment_mlp_invoke(m,layer,p,bindings,api,workspace,bytes,stream);break;
    case IGNIS_SEGMENT_GQA:segment_gqa_invoke(m,layer,p,bindings,api,workspace,bytes,stream);break;
    case IGNIS_SEGMENT_GDN:segment_gdn_invoke(m,layer,p,bindings,api,workspace,bytes,stream);break;
    default:throw std::runtime_error("unknown finite segment kind");
  }
}
}
