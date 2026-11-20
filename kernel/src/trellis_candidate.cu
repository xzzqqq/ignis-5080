// Model-local code replacement for the offline experiment engine.
#include "ignis_gqa_native.h"
#include "ignis_nvfp4_kv.h"
#include "trellis_candidate.h"
#include "gqa_candidate.h"
#include "lab_segment.h"
#include "ignis_mlp_segment.h"
#include "ignis_gqa_segment.h"
#include "ignis_gdn_segment.h"
#include "model_internal.h"
#include "ignis_seq.h"
#include "ignis_step.h"
#include <dlfcn.h>
#include <algorithm>
#include <string>
#include <set>
#include <tuple>
#include <stdexcept>

namespace ignis {
TrellisCandidate::~TrellisCandidate() { if (library) dlclose(library); }
}
namespace {
thread_local std::string error;
int fail(std::string message) { error = std::move(message); return -1; }

template <class Graphs> void destroy(Graphs &graphs) {
  for (auto &g : graphs) { if (g) cudaGraphExecDestroy(g); g = nullptr; }
}

bool fits(const ignis_model &m, const ignis_trellis_candidate_api &api) {
  std::set<std::tuple<uint32_t, uint32_t, uint32_t>> shapes;
  auto linear = [&](const ignis_trellis_weight &w) { shapes.emplace(w.in, w.out, 1); };
  auto tail = [&](const TrellisTail &w) {
    linear(w.output); linear(w.down); shapes.emplace(w.gate_proj.in, w.gate_proj.out, 2);
  };
  linear(m.trellis_head);
  for (const auto &l : m.layers) {
    if (l.kind == IGNIS_LAYER_GQA) {
      const auto &w = l.gqa.trellis;
      linear(w.q_proj); linear(w.k_proj); linear(w.v_proj); tail(w.tail);
    } else {
      const auto &w = l.gdn.trellis;
      linear(w.in_proj_qkv); linear(w.in_proj_z); tail(w.tail);
    }
  }
  // Include every partial prefill chunk, decode/verify width and span readout.
  const uint32_t max_tokens = std::max(m.prefill_chunk_tokens, uint32_t(IGNIS_TRELLIS_DECODE_MAX_TOKENS));
  for (const auto &[in, out, weights] : shapes) {
    const uint32_t limit = out == m.trellis_head.out
        ? std::max(uint32_t(kSpanLogitsBlockColumns), m.decode_lanes * (m.draft_tokens + 1)) : max_tokens;
    for (uint32_t t = 1; t <= limit; ++t) {
      const uint64_t bytes = api.workspace_bytes(in, out, weights, t);
      if (!bytes || bytes > m.trellis_workspace.bytes) return false;
    }
  }
  return true;
}
}

extern "C" const char *ignis_trellis_reload_error() { return error.c_str(); }
extern "C" int32_t ignis_model_reload_trellis(ignis_model *model, ignis_seq_pool *pool, const char *path) {
  error.clear();
  if (!model || !pool || !model->is_trellis_body || model->flash_next)
    return fail("reload requires the fixed Trellis model and its pool");
  struct ignis_seq_pool_stats stats{};
  if (ignis_seq_pool_stats(pool, &stats) != 0 || stats.free_slot_count != stats.slot_count)
    return fail("reload requires an empty sequence pool");
  if (cudaStreamSynchronize(model->stream) != cudaSuccess) return fail("reload stream synchronization failed");
  std::shared_ptr<ignis::TrellisCandidate> next;
  if (path) {
    next = std::make_shared<ignis::TrellisCandidate>();
    next->library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!next->library) return fail(std::string("candidate dlopen: ") + dlerror());
    auto entry = reinterpret_cast<const ignis_trellis_candidate_api *(*)()>(dlsym(next->library, "ignis_trellis_candidate_v1"));
    if (!entry) return fail("candidate missing ignis_trellis_candidate_v1");
    next->api = entry();
    const auto *a = next->api;
    if (!a || a->version != IGNIS_TRELLIS_CANDIDATE_ABI || a->size != sizeof(*a) ||
        a->weight_size != sizeof(ignis_trellis_weight) || a->workspace_size != sizeof(ignis_trellis_workspace) ||
        !a->identity || !a->prepare || !a->workspace_bytes || !a->linear || !a->swiglu || !a->last_error)
      return fail("candidate ABI mismatch");
    if (a->prepare() != 0) return fail(std::string("candidate prepare: ") + a->last_error());
    if (!fits(*model, *a)) return fail("candidate exceeds the resident workspace or lacks a shape");
    // Optional extension: old Trellis-only libraries remain valid. Allocate
    // before touching the active generation; failed admission preserves it.
    auto gqa_entry = reinterpret_cast<const ignis_gqa_candidate_api *(*)()>(dlsym(next->library,"ignis_gqa_candidate_v1"));
    if(gqa_entry){
      next->gqa=gqa_entry();const auto *g=next->gqa;
      if(!g || g->version!=IGNIS_GQA_CANDIDATE_ABI || g->size!=sizeof(*g) ||
         g->call_size!=sizeof(ignis_gqa_candidate_call) || g->reserved || !g->identity ||
         !g->prepare || !g->workspace_bytes || !g->run || !g->last_error)
        return fail("GQA candidate ABI mismatch");
      if(g->prepare())return fail(std::string("GQA candidate prepare: ")+g->last_error());
      const auto bytes=g->workspace_bytes(model->max_context_tokens);
      if(!bytes || bytes>(1ull<<30))return fail("GQA candidate invalid workspace capacity");
      try{next->gqa_workspace=std::make_unique<ninfer::DeviceBuffer>(bytes);}
      catch(const std::exception &e){return fail(std::string("GQA candidate workspace allocation: ")+e.what());}
    }
  }
  if(next){
    auto entry=reinterpret_cast<const ignis_segment_candidate_api *(*)()>(dlsym(next->library,"ignis_segment_candidate_v1"));
    if(entry){
      next->segments=entry();const auto *s=next->segments;
      if(!s || s->version!=IGNIS_SEGMENT_CANDIDATE_ABI || s->size!=sizeof(*s) ||
         s->profile_size!=sizeof(ignis_segment_profile) || s->mlp_call_size!=sizeof(ignis_mlp_segment_call) ||
         s->gqa_call_size!=sizeof(ignis_gqa_segment_call) || s->gdn_call_size!=sizeof(ignis_gdn_segment_call) ||
         !s->identity || !s->prepare || !s->workspace_bytes || !s->run_mlp || !s->run_gqa || !s->run_gdn || !s->last_error)
        return fail("segment candidate ABI mismatch");
      if(s->prepare())return fail(std::string("segment candidate prepare: ")+s->last_error());
    }
  }
  try{ignis::segment_prepare_runtime(*model,*pool,next.get());}
  catch(const std::exception &e){return fail(std::string("segment preparation: ")+e.what());}
  // Keep old code alive until every graph referring to it has been destroyed.
  auto previous = model->trellis_candidate;
  auto old_graphs = model->decode_graph_exec;
  auto old_ready = model->decode_graph_ready;
  decltype(old_graphs) old_verify{};
  decltype(old_ready) old_verify_ready{};
  if (model->verify) {
    old_verify = model->verify->graph_exec;
    old_verify_ready = model->verify->graph_ready;
    model->verify->graph_exec.fill(nullptr);
    model->verify->graph_ready.fill(false);
  }
  model->decode_graph_exec.fill(nullptr);
  model->decode_graph_ready.fill(false);
  model->trellis_candidate = next;
  // The ABI forbids persistent workspace state. Reset all bytes between arms.
  auto rc = cudaMemsetAsync(model->trellis_workspace.base, 0, model->trellis_workspace.bytes, model->stream);
  if (rc == cudaSuccess) rc = cudaStreamSynchronize(model->stream);
  uint32_t mask = 0;
  const int captured = rc == cudaSuccess ? ignis_decode_graph_capture(model, pool, nullptr, &mask) : -1;
  bool complete = captured == 0 && mask == ((1u << model->decode_lanes) - 1);
  if (model->verify) {
    for (uint32_t i = 0; i < model->decode_lanes; ++i) complete &= model->verify->graph_ready[i];
  }
  if (!complete) {
    const std::string why = rc != cudaSuccess ? cudaGetErrorString(rc) : ignis_decode_graph_last_error();
    destroy(model->decode_graph_exec);
    if (model->verify) {
      destroy(model->verify->graph_exec);
      model->verify->graph_exec = old_verify;
      model->verify->graph_ready = old_verify_ready;
    }
    model->decode_graph_exec = old_graphs;
    model->decode_graph_ready = old_ready;
    model->trellis_candidate = previous;
    return fail("candidate graph capture failed; previous generation retained: " + why);
  }
  destroy(old_graphs);
  destroy(old_verify);
  return 0;
}

namespace ignis {
bool gqa_candidate_applies(const ignis_model &m,const ninfer::Tensor &q,const ninfer::PagedKVBatchLayerView &cache){
  return m.trellis_candidate && m.trellis_candidate->gqa && q.ne[0]==256 && q.ne[1]==24 &&
      q.ne[2]==8 && q.ne[3]==1 && cache.dtype==ninfer::DType::U8 && cache.quant_group==32;
}
bool gqa_native_attention_applies(const ignis_model &m,const ninfer::Tensor &q,
    const ninfer::PagedKVBatchLayerView &cache){
  return m.is_trellis_body && m.hidden==5120 && q.dtype==ninfer::DType::BF16 &&
      q.ne[0]==256 && q.ne[1]==24 && q.ne[2]==8 && q.ne[3]==1 && q.is_contiguous() &&
      cache.dtype==ninfer::DType::U8 && cache.head_dim==256 && cache.num_kv_heads==4 &&
      cache.quant_group==32 && ignis_gqa_native_workspace(m.max_context_tokens);
}
uint64_t gqa_candidate_workspace_bytes(const ignis_model &m,uint32_t max_keys){
  if(!m.trellis_candidate || !m.trellis_candidate->gqa)return 0;
  const auto bytes=m.trellis_candidate->gqa->workspace_bytes(max_keys);
  if(!bytes || bytes>(1ull<<30))throw std::runtime_error("GQA candidate invalid replay workspace capacity");
  return bytes;
}
void gqa_attention_dispatch(const ignis_model &m,const ninfer::Tensor &q,const ninfer::Tensor &k,
    const ninfer::Tensor &v,const ninfer::Tensor &positions,const ninfer::Tensor &valid,
    const ninfer::Tensor &rows,const ninfer::Tensor &gate,float scale,
    ninfer::PagedKVBatchLayerView cache,ninfer::ops::GqaExecutionEnvelope envelope,ninfer::DeviceArena &arena,
    ninfer::Tensor &out,cudaStream_t stream,void *replay_workspace,uint64_t replay_bytes){
  if(nvfp4_cache(cache)) {
    nvfp4_attention(q,k,v,positions,valid,rows,gate,scale,cache,envelope,arena,out,stream);
    return;
  }
  const bool extension=gqa_candidate_applies(m,q,cache);
  const bool native=!extension && gqa_native_attention_applies(m,q,cache);
  if(!extension && !native){
    ninfer::ops::gqa_attention(q,k,v,positions,valid,rows,gate,scale,cache,envelope,arena,out,stream);return;
  }
  const auto *api=extension?m.trellis_candidate->gqa:nullptr;
  const auto bytes=extension?gqa_candidate_workspace_bytes(m,envelope.max_visible_keys)
      :ignis_gqa_native_workspace(envelope.max_visible_keys);
  void *workspace=replay_workspace;
  uint64_t capacity=replay_bytes;
  if(!workspace){
    if(extension){workspace=m.trellis_candidate->gqa_workspace->p;capacity=m.trellis_candidate->gqa_workspace->bytes;}
    else {const auto storage=arena.alloc_bytes(bytes);workspace=storage.data;capacity=storage.bytes;}
  }
  if(bytes>capacity)throw std::runtime_error("GQA candidate query exceeds host-owned workspace");
  ignis_gqa_candidate_call call{IGNIS_GQA_CANDIDATE_ABI,sizeof(ignis_gqa_candidate_call),envelope.max_visible_keys,
      static_cast<uint32_t>(cache.block_tables.ne[0]),static_cast<uint32_t>(cache.k_pages.ne[3]),
      static_cast<uint32_t>(cache.block_tables.ne[1]),scale,q.data,k.data,v.data,gate.data,
      static_cast<const int32_t *>(positions.data),static_cast<const int32_t *>(valid.data),
      static_cast<const int32_t *>(rows.data),static_cast<const int32_t *>(cache.block_tables.data),
      cache.k_pages.data,cache.v_pages.data,cache.k_scale_pages.data,cache.v_scale_pages.data,
      cache.residual_k.data,cache.residual_v.data,static_cast<uint32_t *>(cache.ring_valid.data),out.data};
  const int32_t rc=extension?api->run(&call,workspace,capacity,stream)
      :ignis_gqa_native_run(&call,workspace,capacity,stream);
  if(rc)throw std::runtime_error(std::string(extension?"GQA candidate: ":"GQA native: ")+
      (extension?api->last_error():ignis_gqa_native_error()));
}
}
