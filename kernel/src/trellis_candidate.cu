// Model-local code replacement for the offline experiment engine.
#include "trellis_candidate.h"
#include "model_internal.h"
#include "ignis_seq.h"
#include "ignis_step.h"
#include <dlfcn.h>
#include <algorithm>
#include <string>
#include <set>
#include <tuple>

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
  }
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
