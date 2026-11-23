// GitHub #339 (spec 27b-16gb/05a): the trellis body's own plumbing in the layer
// programs (kernel/src/trellis_body.h, a leaf-internal header) -- OURS. The op's
// numerics are test_trellis_linear's; this holds what the layer bodies add:
//
//   split    the GQA q_proj's rows are 24 heads of [q 256 | gate 256] (transformers'
//            order), and the attention reads the query and the gate head-major: every
//            element of a [tokens][24][2][256] pattern lands in its own head and half,
//            bit for bit, so a swapped half or a wrong pitch fails here and not three
//            layers later in a logit.
//   q/gate   query_gate on a trellis q_proj is exactly one ignis_trellis_linear into the
//            interleaved rows followed by that split, on both of the op's routes.
//   head     output_head on a trellis load is the op into the logits at ldy = vocab,
//            and on the served body the vendored linear (not run here: the op tests).
//   epilogue a GDN input projection writes its columns into the conv input at its own
//            ldy, and the output projection adds into the residual: the layer bodies'
//            two uses of the op's ldy and ADD, against the op's own STORE.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "model_internal.h"
#include "moe_fixture.h"
#include "trellis_body.h"
#include "ignis_gqa_native.h"
#include "ignis_gdn_segment.h"
#include "trellis_linear_internal.cuh"

#include "core/arena.h"
#include "core/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int32_t ignis_gdn_joint_projection(const ignis_trellis_weight *,const ignis_trellis_weight *,const void *,void *,void *,const ignis_trellis_workspace *,const ignis_trellis_workspace *,void *);
extern "C" int32_t ignis_gdn_rotation_params(const ignis_trellis_weight *,const ignis_trellis_workspace *,ignis_tl::RotateParams *);
extern "C" int32_t ignis_gdn_output_from(const ignis_trellis_weight *,const void *,const void *,void *,const ignis_trellis_workspace *,void *);
using namespace moe_test;
namespace tb = ignis::trellis_body;

namespace {

#define TL_RC(expr)                                                                                \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, ignis_trellis_last_error()); \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

// The 27B's GQA heads.
constexpr int kHeads = 24, kHeadDim = 256;

void split_lands_every_half_in_its_head() {
  const int tokens = 5;
  const std::size_t n = static_cast<std::size_t>(tokens) * kHeads * 2 * kHeadDim;
  // Element (t, h, half, d) holds its own index: 5 * 24 * 2 * 256 < 65536.
  std::vector<uint16_t> pattern(n);
  for (std::size_t i = 0; i < n; ++i) pattern[i] = static_cast<uint16_t>(i);
  DeviceBytes interleaved(n * 2), query(n), gate(n);
  upload(interleaved, pattern);
  MOE_CUDA(tb::split_query_gate(interleaved.p, kHeads, kHeadDim, tokens, query.p, gate.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const auto q = download<uint16_t>(query.p, n / 2);
  const auto g = download<uint16_t>(gate.p, n / 2);
  int wrong = 0;
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < kHeads; ++h) {
      for (int d = 0; d < kHeadDim; ++d) {
        const std::size_t dense = (static_cast<std::size_t>(t) * kHeads + h) * kHeadDim + d;
        const std::size_t row = (static_cast<std::size_t>(t) * kHeads + h) * 2 * kHeadDim;
        wrong += q[dense] != pattern[row + d];
        wrong += g[dense] != pattern[row + kHeadDim + d];
      }
    }
  }
  check(wrong == 0, "split: " + std::to_string(wrong) + " elements outside their head's half");
}

// A synthetic trellis weight: counter-hash words, suh and svh of +-1 (fp16).
struct Trellis {
  DeviceBytes words, suh, svh;
  ignis_trellis_weight w{};
  Trellis(uint32_t seed, uint32_t in, uint32_t out, uint32_t k2)
      : words(static_cast<std::size_t>(in) * out / 16 * k2), suh(in * 2u), svh(out * 2u) {
    std::vector<uint16_t> v(words.bytes / 2);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint16_t>(hash_u32(seed, i));
    upload(words, v);
    const auto signs = [seed](uint32_t n, uint32_t s) {
      std::vector<uint16_t> out(n);
      for (uint32_t i = 0; i < n; ++i) out[i] = (hash_u32(seed + s, i) & 1u) ? 0x3C00u : 0xBC00u;
      return out;
    };
    upload(suh, signs(in, 1));
    upload(svh, signs(out, 2));
    w.trellis = words.p;
    w.suh = suh.p;
    w.svh = svh.p;
    w.k2 = k2;
    w.in = in;
    w.out = out;
  }
};

std::vector<uint16_t> activations(uint32_t seed, std::size_t n) {
  std::vector<uint16_t> x(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = f32_to_bf16(hash_uniform(seed, i, 0.5f));
  return x;
}

// A model handle with nothing bound but the trellis workspace (and, where an
// arm sets it, the head): the layer bodies' helpers read no more of it.
struct Model {
  ignis_model m;
  DeviceBytes workspace;
  explicit Model(uint64_t bytes) : workspace(bytes) {
    m.is_trellis_body = true;
    m.trellis_workspace.base = workspace.p;
    m.trellis_workspace.bytes = bytes;
    TL_RC(ignis_trellis_workspace_init(&m.trellis_workspace, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
  }
};

uint64_t workspace_for(uint32_t in, uint32_t out, std::initializer_list<uint32_t> tokens) {
  uint64_t bytes = 0;
  for (uint32_t t : tokens) bytes = std::max(bytes, ignis_trellis_workspace_bytes(in, out, 1, t));
  return bytes;
}

void query_gate_is_the_op_then_the_split() {
  // Two heads' worth of a smaller q_proj (4 heads of 2 x 128 over 256 inputs),
  // at a decode width and a prefill one.
  const int heads = 4, head_dim = 128;
  const uint32_t in = 256, out = 2 * heads * head_dim;
  Trellis q_proj(11, in, out, 6);
  Model model(workspace_for(in, out, {3, 80}));
  ninfer::DeviceArena arena(16u << 20);
  for (const int tokens : {3, 80}) {
    const auto xs = activations(21 + tokens, static_cast<std::size_t>(tokens) * in);
    DeviceBytes x(xs.size() * 2), full(static_cast<std::size_t>(tokens) * out * 2),
        query(static_cast<std::size_t>(tokens) * out), gate(static_cast<std::size_t>(tokens) * out);
    upload(x, xs);
    TL_RC(ignis_trellis_linear(&q_proj.w, x.p, tokens, full.p, out, IGNIS_TRELLIS_STORE, &model.m.trellis_workspace,
                               nullptr));
    const ninfer::Tensor xt(x.p, ninfer::DType::BF16, {static_cast<int32_t>(in), tokens, 1, 1});
    ninfer::Tensor qt(query.p, ninfer::DType::BF16, {heads * head_dim, tokens, 1, 1});
    ninfer::Tensor gt(gate.p, ninfer::DType::BF16, {heads * head_dim, tokens, 1, 1});
    {
      ninfer::DeviceArena::Scope scope = arena.scope();
      tb::query_gate(model.m, q_proj.w, xt, heads, head_dim, arena, qt, gt, nullptr);
    }
    MOE_CUDA(cudaDeviceSynchronize());
    const auto f = download<uint16_t>(full.p, static_cast<std::size_t>(tokens) * out);
    const auto q = download<uint16_t>(query.p, static_cast<std::size_t>(tokens) * out / 2);
    const auto g = download<uint16_t>(gate.p, static_cast<std::size_t>(tokens) * out / 2);
    int wrong = 0;
    for (int t = 0; t < tokens; ++t) {
      for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < head_dim; ++d) {
          const std::size_t dense = (static_cast<std::size_t>(t) * heads + h) * head_dim + d;
          const std::size_t row = static_cast<std::size_t>(t) * out + static_cast<std::size_t>(h) * 2 * head_dim;
          wrong += q[dense] != f[row + d];
          wrong += g[dense] != f[row + head_dim + d];
        }
      }
    }
    check(wrong == 0, "query/gate at " + std::to_string(tokens) + " tokens: " + std::to_string(wrong) +
                          " elements differ from the op's rows");
  }
}

void head_and_epilogues_are_the_op_at_the_bodies_widths() {
  const uint32_t in = 256, vocab = 1024, wide = 1536;
  Trellis head(31, in, vocab, 8);
  Model model(workspace_for(in, vocab, {1, 2, 5, 80}));
  model.m.trellis_head = head.w;
  for (const int tokens : {2, 80}) {
    const auto xs = activations(41 + tokens, static_cast<std::size_t>(tokens) * in);
    DeviceBytes x(xs.size() * 2), direct(static_cast<std::size_t>(tokens) * vocab * 2),
        logits(static_cast<std::size_t>(tokens) * vocab * 2), into(static_cast<std::size_t>(tokens) * wide * 2),
        added(static_cast<std::size_t>(tokens) * vocab * 2);
    upload(x, xs);
    const auto residual = activations(51 + tokens, static_cast<std::size_t>(tokens) * vocab);
    upload(added, residual);
    MOE_CUDA(cudaMemset(into.p, 0, into.bytes));
    TL_RC(ignis_trellis_linear(&head.w, x.p, tokens, direct.p, vocab, IGNIS_TRELLIS_STORE, &model.m.trellis_workspace,
                               nullptr));
    const ninfer::Tensor xt(x.p, ninfer::DType::BF16, {static_cast<int32_t>(in), tokens, 1, 1});
    ninfer::Tensor lt(logits.p, ninfer::DType::BF16, {static_cast<int32_t>(vocab), tokens, 1, 1});
    tb::output_head(model.m, xt, lt, nullptr);
    // A GDN input projection's columns in a wider buffer, and an output
    // projection's residual add.
    tb::linear(model.m, head.w, xt, into.as<uint16_t>() + 256, wide, IGNIS_TRELLIS_STORE, nullptr);
    tb::linear(model.m, head.w, xt, added.p, vocab, IGNIS_TRELLIS_ADD, nullptr);
    MOE_CUDA(cudaDeviceSynchronize());
    const auto d = download<uint16_t>(direct.p, static_cast<std::size_t>(tokens) * vocab);
    const auto l = download<uint16_t>(logits.p, d.size());
    const auto w = download<uint16_t>(into.p, static_cast<std::size_t>(tokens) * wide);
    const auto a = download<uint16_t>(added.p, d.size());
    int head_wrong = 0, strided_wrong = 0, outside = 0, add_wrong = 0;
    for (int t = 0; t < tokens; ++t) {
      for (uint32_t c = 0; c < wide; ++c) {
        const uint16_t got = w[static_cast<std::size_t>(t) * wide + c];
        if (c < 256 || c >= 256 + vocab) {
          outside += got != 0;
          continue;
        }
        strided_wrong += got != d[static_cast<std::size_t>(t) * vocab + (c - 256)];
      }
      for (uint32_t c = 0; c < vocab; ++c) {
        const std::size_t i = static_cast<std::size_t>(t) * vocab + c;
        head_wrong += l[i] != d[i];
        // The op rounds r + y once, y unrounded; the stored y is one BF16
        // rounding from it, so the two sums are within two.
        const float sum = bf16_to_f32(d[i]) + bf16_to_f32(residual[i]);
        const float diff = std::fabs(bf16_to_f32(a[i]) - sum);
        add_wrong += diff > (std::fabs(bf16_to_f32(d[i])) + std::fabs(sum)) * 0x1p-7f + 0x1p-12f;
      }
    }
    const std::string at = " at " + std::to_string(tokens) + " tokens";
    check(head_wrong == 0, "the trellis head is the op" + at);
    check(strided_wrong == 0 && outside == 0, "the columns land at their ldy and offset, nothing else" + at);
    check(add_wrong == 0, "the residual add" + at + ": " + std::to_string(add_wrong) + " outside one rounding");
  }
  // The head refuses logits that are not its vocab wide.
  const auto xs = activations(61, in);
  DeviceBytes x(xs.size() * 2), logits(vocab * 2);
  upload(x, xs);
  const ninfer::Tensor xt(x.p, ninfer::DType::BF16, {static_cast<int32_t>(in), 1, 1, 1});
  ninfer::Tensor short_logits(logits.p, ninfer::DType::BF16, {static_cast<int32_t>(vocab / 2), 1, 1, 1});
  bool refused = false;
  try {
    tb::output_head(model.m, xt, short_logits, nullptr);
  } catch (const std::exception &) {
    refused = true;
  }
  check(refused, "the head refuses logits narrower than its vocab");
}

// Exercise the complete native chain, whose Q / paired-KV / output projection
// shapes reuse the same transient TL counters. Attention-only/reset-only tests
// cannot detect this workspace-lifetime regression.
void native_gqa_reuses_poisoned_workspace() {
  constexpr uint32_t hidden=5120, query=6144, kv=1024, tokens=8;
  Trellis q_proj(701,hidden,2*query,6), k_proj(702,hidden,kv,6),
      v_proj(703,hidden,kv,6), output_proj(704,query,hidden,6);
  DeviceBytes input(hidden*tokens*2), input_norm(hidden*2),
      query_norm(256*2), key_norm(256*2), output(hidden*tokens*2),
      positions(tokens*4), rope_positions(tokens*4), valid(4), rows(4), tables(4),
      kc(64*64*4), vc(64*64*4), km(8*64*4), vm(8*64*4),
      kr(256*4*544*2), vr(256*4*544*2), ring(16*4);
  upload(input,activations(705,hidden*tokens));
  // RMS/rope norms use unit-offset semantics: BF16 zero means scale one.
  MOE_CUDA(cudaMemset(input_norm.p,0,input_norm.bytes));
  MOE_CUDA(cudaMemset(query_norm.p,0,query_norm.bytes));
  MOE_CUDA(cudaMemset(key_norm.p,0,key_norm.bytes));
  upload(positions,std::vector<int32_t>{0,1,2,3,4,5,6,7});
  upload(rope_positions,std::vector<int32_t>{0,1,2,3,4,5,6,7});
  upload(rows,std::vector<int32_t>{0});upload(tables,std::vector<int32_t>{0});
  ignis_gqa_segment_call c{};
  c.profile={IGNIS_SEGMENT_GQA,tokens,1,hidden,0,64,1,0,1e-6f};
  c.input=input.p;c.input_norm=input_norm.p;c.query_norm=query_norm.p;c.key_norm=key_norm.p;
  c.q_proj=q_proj.w;c.k_proj=k_proj.w;c.v_proj=v_proj.w;c.output_proj=output_proj.w;
  for(int i=0;i<128;++i)c.rope_inv_frequency[i]=std::pow(10000.0,-double(i)/128.0);
  c.rope_attention_factor=1.0f;c.positions=positions.as<int32_t>();
  c.rope_positions=rope_positions.as<int32_t>();c.valid_columns=valid.as<int32_t>();
  c.table_rows=rows.as<int32_t>();c.block_tables=tables.as<int32_t>();
  c.logical_pages=1;c.physical_pages=1;c.k_pages=kc.p;c.v_pages=vc.p;
  c.k_meta=km.p;c.v_meta=vm.p;c.residual_k=kr.p;c.residual_v=vr.p;
  c.ring=ring.as<uint32_t>();c.output=output.p;
  const auto rc=[&](int32_t status,const char *where,bool builtin=false){
    if(status){std::fprintf(stderr,"FATAL: %s: %s\n",where,builtin?
        ignis_lab_gqa_segment_error():ignis_gqa_native_segment_error());std::exit(EXIT_FAILURE);}
  };
  rc(ignis_gqa_native_segment_prepare(),"native prepare");
  rc(ignis_lab_gqa_segment_prepare(),"builtin prepare",true);
  const uint64_t bytes=ignis_gqa_native_segment_workspace(&c.profile);
  check(bytes>0,"native full-chain workspace is supported");
  check(bytes==ignis_lab_gqa_segment_workspace(&c.profile),"builtin queries native workspace");
  if(!bytes)return;
  DeviceBytes workspace(bytes);
  const std::vector<DeviceBytes*> state{&kc,&vc,&km,&vm,&kr,&vr,&ring};
  const auto reset_state=[&](){
    for(auto *p:state)MOE_CUDA(cudaMemset(p->p,0,p->bytes));
    MOE_CUDA(cudaMemset(output.p,0xFF,output.bytes));
  };
  const auto snapshot=[&](){
    std::vector<uint8_t> all;
    for(auto *p:state){const auto v=download<uint8_t>(p->p,p->bytes);all.insert(all.end(),v.begin(),v.end());}
    return all;
  };
  for(int n:{8,7,0}) {
    // Prefix masks retain strictly increasing positions on valid columns.
    auto pos=std::vector<int32_t>(tokens,n? n-1:0);
    for(int i=0;i<n;++i)pos[i]=i;
    upload(positions,pos);upload(rope_positions,pos);upload(valid,std::vector<int32_t>{n});
    reset_state();MOE_CUDA(cudaMemset(workspace.p,0,workspace.bytes));
    rc(ignis_gqa_native_segment_run(&c,workspace.p,bytes,nullptr),"zero workspace run");
    MOE_CUDA(cudaDeviceSynchronize());
    const auto expected=download<uint16_t>(output.p,hidden*tokens);
    const auto expected_state=snapshot();
    for(auto v:expected)check(std::isfinite(bf16_to_f32(v)),"native full-chain baseline finite");
    for(int poison:{0x7F,0xFF})for(bool builtin:{false,true}) {
      reset_state();MOE_CUDA(cudaMemset(workspace.p,poison,workspace.bytes));
      const int32_t status=builtin?ignis_lab_gqa_segment_run(&c,workspace.p,bytes,nullptr):
          ignis_gqa_native_segment_run(&c,workspace.p,bytes,nullptr);
      rc(status,"poisoned workspace run",builtin);MOE_CUDA(cudaDeviceSynchronize());
      const std::string label=std::string(builtin?"builtin":"native")+" poisoned workspace valid="+std::to_string(n);
      check(download<uint16_t>(output.p,hidden*tokens)==expected,label+" exact output");
      check(snapshot()==expected_state,label+" exact complete physical state");
    }
  }
}

// The native GDN chain borrows the model workspace. Only the fixed counter
// prefix is persistent: MLP/head calls may overwrite every later byte.
void gdn_reuses_trellis_payload(uint32_t qkv_bits, uint32_t z_bits) {
  Trellis qkv(810,5120,10240,qkv_bits), z(811,5120,6144,z_bits), out(812,6144,5120,6);
  DeviceBytes input(5120*8*2), norm(5120*2), conv_w(10240*4*2), ab(96*5120*2),
      alog(48*4), bias(48*4), gnorm(128*2), conv_state(10240*3*2*2),
      state(128*128*48*4*2), slots(4), valid(4), output(5120*8*2),
      cr(10240*8*2), kr(128*16*8*2), vr(128*48*8*2), gr(2*48*8*4);
  upload(input,activations(813,5120*8));upload(conv_w,activations(814,10240*4));
  upload(gnorm,std::vector<uint16_t>(128,f32_to_bf16(1.0f)));
  upload(conv_state,activations(815,10240*3*2));upload(state,std::vector<float>(128*128*48*2,0.01f));
  upload(ab,activations(816,96*5120));
  for(auto *b:{&norm,&alog,&bias})MOE_CUDA(cudaMemset(b->p,0,b->bytes));
  upload(slots,std::vector<int32_t>{1});
  ignis_gdn_segment_call c{};c.profile={IGNIS_SEGMENT_GDN,8,1,5120,0,0,2,0,1e-6f};
  c.qkv_weight=qkv.w;c.z_weight=z.w;c.output_weight=out.w;
  c.input_norm=norm.p;c.convolution=conv_w.p;c.ab_projection=ab.p;c.a_log=alog.p;c.dt_bias=bias.p;c.gdn_norm=gnorm.p;
  c.residual_in=input.p;c.residual_out=output.p;c.conv_states=conv_state.p;c.recurrent_states=state.p;
  c.lane_slots=slots.as<int32_t>();c.valid_columns=valid.as<int32_t>();
  c.conv_record=cr.p;c.key_record=kr.p;c.value_record=vr.p;c.gate_record=gr.p;
  const auto rc=[](int32_t status){if(status){std::fprintf(stderr,"GDN: %s\n",ignis_lab_gdn_segment_error());std::exit(EXIT_FAILURE);}};
  rc(ignis_lab_gdn_segment_prepare());
  DeviceBytes workspace(ignis_lab_gdn_segment_workspace(&c.profile));
  ignis_trellis_workspace full{workspace.p,workspace.bytes}, tiny{workspace.p,0};
  ignis_tl::RotateParams rotation{};
  check(ignis_gdn_joint_projection(nullptr,&z.w,input.p,cr.p,output.p,&tiny,&tiny,nullptr)!=0,"GDN rejects null weight");
  check(ignis_gdn_joint_projection(&qkv.w,&z.w,input.p,cr.p,output.p,&tiny,&tiny,nullptr)!=0,"GDN rejects undersized workspace");
  check(ignis_gdn_rotation_params(&out.w,&tiny,&rotation)!=0,"GDN rotation rejects undersized workspace");
  check(ignis_gdn_rotation_params(&out.w,&full,nullptr)!=0,"GDN rotation rejects null parameters");
  check(ignis_gdn_output_from(&out.w,nullptr,input.p,output.p,&tiny,nullptr)!=0,"GDN output rejects undersized workspace");
  check(ignis_gdn_output_from(&out.w,nullptr,static_cast<char *>(input.p)+2,output.p,&full,nullptr)!=0,"GDN rejects unaligned residual");
  const auto before_state=download<uint8_t>(state.p,state.bytes),before_conv=download<uint8_t>(conv_state.p,conv_state.bytes);
  const auto snapshot=[&](){std::vector<uint8_t> v;for(auto *b:{&output,&cr,&kr,&vr,&gr}){
    auto x=download<uint8_t>(b->p,b->bytes);v.insert(v.end(),x.begin(),x.end());}return v;};
  ignis::SideBranch branch;
  std::string branch_error;
  check(branch.create(&branch_error),"create GDN gating branch: "+branch_error);
  cudaStream_t stream{};
  MOE_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  for(int n:{8,5,0}){
    upload(valid,std::vector<int32_t>{n});
    for(auto *b:{&output,&cr,&kr,&vr,&gr})MOE_CUDA(cudaMemset(b->p,0x7F,b->bytes));
    MOE_CUDA(cudaMemset(workspace.p,0,workspace.bytes));
    c.gating_stream=c.gating_fork=c.gating_join=nullptr;
    rc(ignis_lab_gdn_segment_run(&c,workspace.p,workspace.bytes,nullptr));MOE_CUDA(cudaDeviceSynchronize());
    const auto expected=snapshot();
    c.gating_stream=branch.stream;c.gating_fork=branch.fork;c.gating_join=branch.join;
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    MOE_CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    rc(ignis_lab_gdn_segment_run(&c,workspace.p,workspace.bytes,stream));
    MOE_CUDA(cudaStreamEndCapture(stream,&graph));
    MOE_CUDA(cudaGraphInstantiate(&exec,graph,0));
    for(int poison:{0x7F,0xFF}){
      for(auto *b:{&output,&cr,&kr,&vr,&gr})MOE_CUDA(cudaMemset(b->p,0x7F,b->bytes));
      MOE_CUDA(cudaMemset(static_cast<char *>(workspace.p)+ignis_tl::kCounterBytes,poison,workspace.bytes-ignis_tl::kCounterBytes));
      MOE_CUDA(cudaDeviceSynchronize());
      MOE_CUDA(cudaGraphLaunch(exec,stream));
      MOE_CUDA(cudaStreamSynchronize(stream));
      check(snapshot()==expected,"GDN shared payload reuse qkv="+std::to_string(qkv_bits)+" z="+std::to_string(z_bits)+" valid="+std::to_string(n)+" poison="+std::to_string(poison));
      check(download<uint8_t>(state.p,state.bytes)==before_state,"GDN recurrent state readonly");
      check(download<uint8_t>(conv_state.p,conv_state.bytes)==before_conv,"GDN conv state readonly");
    }
    MOE_CUDA(cudaGraphExecDestroy(exec));
    MOE_CUDA(cudaGraphDestroy(graph));
  }
  MOE_CUDA(cudaStreamDestroy(stream));
  branch.destroy();
}

}  // namespace

int main(int argc, char **argv) {
  const bool gdn_only = argc == 2 && std::strcmp(argv[1], "--native-gdn-only") == 0;
  const bool native_only = argc == 2 && std::strcmp(argv[1], "--native-gqa-only") == 0;
  if (argc != 1 && !native_only && !gdn_only) {
    std::fprintf(stderr, "usage: %s [--native-gqa-only|--native-gdn-only]\n", argv[0]);
    return 2;
  }
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "FAIL: no CUDA device\n");
    return 1;
  }
  TL_RC(ignis_trellis_prepare());
  if(!gdn_only)native_gqa_reuses_poisoned_workspace();
  if (!native_only) {
    gdn_reuses_trellis_payload(6,8);
    gdn_reuses_trellis_payload(10,8);
    gdn_reuses_trellis_payload(6,10);
    if(!gdn_only){
    split_lands_every_half_in_its_head();
    query_gate_is_the_op_then_the_split();
    head_and_epilogues_are_the_op_at_the_bodies_widths();
    }
  }
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
