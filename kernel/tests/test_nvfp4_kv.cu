// Ignis-owned NVFP4 format and route regression. No model/artifact is required.
// The codec oracle below uses scalar host tables, not CUDA FP4/FP8 conversion.
// Numeric route cases use exactly representable decoded BF16 K/V and QK=0:
// causal attention is an analytic prefix mean. Its tolerance follows the
// BF16 normalized-split rounding bound, not BF16-vs-quantized model agreement.
// Missing GPU, launch errors and nonfinite results fail; no skip path.
#include "ignis_nvfp4_kv.h"
#include "../src/nvfp4_xqa.h"
#include "../src/nvfp4_prefill.h"
#include "ignis_seq_checkpoint_internal.h"
#include "ignis_seq_internal.h"
#include "core/arena.h"
#include "moe_fixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace moe_test;
namespace {
constexpr int D = 256, H = 4, QH = 24, Page = 64;
float sf_value(int code) {
  const int e = code >> 3, m = code & 7;
  return e == 0 ? std::ldexp(float(m), -9) : std::ldexp(float(8 + m), e - 10);
}
constexpr float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
int nearest(float value, bool scale) {
  const int end = scale ? 127 : 8;
  int best = 0;
  double distance = std::abs(double(value));
  for (int i = 1; i < end; ++i) {
    const double delta = std::abs(double(value) - (scale ? sf_value(i) : levels[i]));
    if (delta < distance || (delta == distance && !(i & 1))) { best = i; distance = delta; }
  }
  return best;
}
struct Encoded {
  std::array<std::uint8_t,128> codes{};
  std::array<std::uint8_t,16> scales{};
};
Encoded encode(const std::uint16_t *raw) {
  Encoded out;
  for (int g = 0; g < 16; ++g) {
    float amax = 0;
    for (int d = 0; d < 16; ++d) amax = std::max(amax, std::abs(bf16_to_f32(raw[g*16+d])));
    const int sf = nearest(amax / 6.0f, true);
    out.scales[g] = sf;
    for (int d = 0; d < 16; ++d) {
      const float x = bf16_to_f32(raw[g*16+d]);
      const float normalized = sf ? x / sf_value(sf) : 0.0f;
      const int nibble = nearest(std::abs(normalized), false) | (std::signbit(normalized) ? 8 : 0);
      out.codes[g*8+d/2] |= std::uint8_t(nibble << (4*(d&1)));
    }
  }
  return out;
}
std::uint16_t decoded(const Encoded &row, int d) {
  const int c = (row.codes[d/2] >> (4*(d&1))) & 15;
  return f32_to_bf16((c & 8 ? -1.f : 1.f) * levels[c & 7] * sf_value(row.scales[d/16]));
}

struct Cache {
  int logical, pages, stride;
  DeviceBytes k,v,ks,vs,tables;
  ninfer::PagedKVBatchLayerView view{};
  explicit Cache(int keys) : logical((keys+63)/64), pages(2*logical+7), stride(logical),
      k(std::size_t(pages)*H*Page*128), v(k.bytes),
      ks(std::size_t(pages)*H*Page*16), vs(ks.bytes), tables(std::size_t(2)*stride*4) {
    std::vector<std::int32_t> map(2*stride);
    // A permutation, with the two table rows interleaved and noncontiguous.
    int multiplier = 3;
    while (std::gcd(multiplier,pages) != 1) ++multiplier;
    for (int r=0;r<2;++r) for (int p=0;p<stride;++p)
      map[r*stride+p] = (multiplier*(p*2+r)+5)%pages;
    upload(tables,map);
    view.k_pages = ninfer::Tensor(k.p,ninfer::DType::U8,{128,Page,H,pages});
    view.v_pages = ninfer::Tensor(v.p,ninfer::DType::U8,{128,Page,H,pages});
    view.k_scale_pages = ninfer::Tensor(ks.p,ninfer::DType::U8,{16,Page,H,pages});
    view.v_scale_pages = ninfer::Tensor(vs.p,ninfer::DType::U8,{16,Page,H,pages});
    view.block_tables = ninfer::Tensor(tables.p,ninfer::DType::I32,{stride,2});
    view.dtype=ninfer::DType::U8;view.quant_group=16;view.head_dim=D;view.num_kv_heads=H;
  }
};

void codec_scatter_and_unselected_bytes(cudaStream_t stream) {
  Cache cache(128);
  const auto table=download<std::int32_t>(cache.tables.p,2*cache.stride);
  for (int valid_count : {0,1,7,8}) {
    for (auto *plane : {&cache.k,&cache.v,&cache.ks,&cache.vs}) MOE_CUDA(cudaMemsetAsync(plane->p,0xA5,plane->bytes,stream));
    std::vector<std::uint16_t> raw(8*H*D);
    for (std::size_t i=0;i<raw.size();++i) raw[i]=f32_to_bf16(float(int((i*17)%193)-96)/64.f);
    // Include zero-SF underflow and an exact E2M1/E4M3 midpoint group.
    for(int d=0;d<16;++d) raw[d]=f32_to_bf16(6.f*std::ldexp(1.f,-10));
    std::vector<std::int32_t> positions(8);
    for(int t=0;t<8;++t) positions[t]=t<valid_count?60+t:0x7fffffff;
    for(int t=valid_count;t<8;++t) std::fill(raw.begin()+t*H*D,raw.begin()+(t+1)*H*D,0x7FC0);
    DeviceBytes input(raw.size()*2), pos(32), valid(4), rows(4), stats(32);
    upload(input,raw);upload(pos,positions);upload(valid,std::vector<std::int32_t>{valid_count});
    upload(rows,std::vector<std::int32_t>{valid_count ? 1 : 0x7fffffff});MOE_CUDA(cudaMemsetAsync(stats.p,0,stats.bytes,stream));
    const ninfer::Tensor x(input.p,ninfer::DType::BF16,{D,H,8,1});
    ignis::nvfp4_append(x,x,ninfer::Tensor(pos.p,ninfer::DType::I32,{8,1}),
      ninfer::Tensor(valid.p,ninfer::DType::I32,{1}),ninfer::Tensor(rows.p,ninfer::DType::I32,{1}),cache.view,stream,stats.as<std::uint64_t>());
    MOE_CUDA(cudaStreamSynchronize(stream));
    std::vector<std::uint8_t> codes(cache.k.bytes,0xA5),scales(cache.ks.bytes,0xA5);
    for(int t=0;t<valid_count;++t) for(int h=0;h<H;++h) {
      const auto e=encode(&raw[(t*H+h)*D]);const int absolute=60+t;
      const auto row=(std::size_t(table[cache.stride+absolute/Page])*H+h)*Page+absolute%Page;
      std::copy(e.codes.begin(),e.codes.end(),codes.begin()+row*128);
      std::copy(e.scales.begin(),e.scales.end(),scales.begin()+row*16);
    }
    for(auto *plane : {&cache.k,&cache.v}) check(download<std::uint8_t>(plane->p,plane->bytes)==codes,"NV4 code oracle/scatter/untouched bytes");
    for(auto *plane : {&cache.ks,&cache.vs}) check(download<std::uint8_t>(plane->p,plane->bytes)==scales,"NV4 scale oracle/scatter/untouched bytes");
    const auto counters=download<std::uint64_t>(stats.p,4);
    check(counters[0]==0 && counters[1]==0,"finite in-range codec inputs");
    if(!valid_count)check(std::all_of(counters.begin(),counters.end(),[](auto x){return x==0;}),"valid0 no stats/write");
  }
}

float value_at(int slot,int head,int d) {
  const float sign=(d&8)?-1.f:1.f;
  return sign*levels[d&7]*std::ldexp(1.f,head-3-slot);
}
bool marker(int position) { return position%257==0 || position>=16384; }
int marker_count(int keys) {
  if(keys<=0)return 0;
  const int initial=std::min(keys,16384);
  return (initial-1)/257+1+std::max(0,keys-16384);
}
void initialize_history(Cache &cache,int keys,cudaStream_t stream,bool two_class=false) {
  const std::size_t elements=std::size_t(keys)*2*H*D;
  std::vector<std::uint16_t> zero(elements,0),values(elements);
  for(int slot=0;slot<2;++slot) for(int t=0;t<keys;++t) if(two_class || marker(t))
    for(int h=0;h<H;++h) for(int d=0;d<D;++d)
      values[((std::size_t(slot)*keys+t)*H+h)*D+d]=f32_to_bf16(value_at(slot,h,d)*(two_class && t>=16384 ? 2.f : 1.f));
  if(two_class) for(int slot=0;slot<2;++slot)for(int t=16384;t<keys;++t)for(int h=0;h<H;++h)
    zero[((std::size_t(slot)*keys+t)*H+h)*D]=f32_to_bf16(6.f);
  // These synthetic values are exactly reconstructed BF16 values, independently
  // checked through the host codec; this is an arithmetic, not codec-quality test.
  for(int h=0;h<H;++h) {
    const auto e=encode(&values[h*D]);
    for(int d=0;d<D;++d)check(decoded(e,d)==values[h*D+d],"analytic BF16 decoded fixture");
  }
  std::vector<std::int32_t> positions(keys*2);
  for(int slot=0;slot<2;++slot)std::iota(positions.begin()+slot*keys,positions.begin()+(slot+1)*keys,0);
  DeviceBytes k(zero.size()*2),v(values.size()*2),p(positions.size()*4),r(8);
  upload(k,zero);upload(v,values);upload(p,positions);upload(r,std::vector<std::int32_t>{0,1});
  ignis::nvfp4_append(ninfer::Tensor(k.p,ninfer::DType::BF16,{D,H,keys,2}),
      ninfer::Tensor(v.p,ninfer::DType::BF16,{D,H,keys,2}),ninfer::Tensor(p.p,ninfer::DType::I32,{keys,2}),
      ninfer::Tensor{},ninfer::Tensor(r.p,ninfer::DType::I32,{2}),cache.view,stream);
  MOE_CUDA(cudaStreamSynchronize(stream));
}

struct Inputs {
  int tokens,batch;
  DeviceBytes q,k,v,positions,valid,rows,out;
  explicit Inputs(int t,int b):tokens(t),batch(b),q(std::size_t(t)*b*QH*D*2),
    k(std::size_t(t)*b*H*D*2),v(k.bytes),positions(t*b*4),valid(b*4),rows(b*4),out(q.bytes){}
  void update(const std::vector<int>&windows,const std::vector<int>&counts,bool two_class=false) {
    std::vector<std::uint16_t> qs(q.bytes/2,0),ks(k.bytes/2,0),vs(k.bytes/2,0);
    std::vector<std::int32_t> ps(tokens*batch),rs(batch);
    for(int lane=0;lane<batch;++lane) {
      rs[lane]=counts[lane] ? 1-lane : 0x7fffffff; // Inactive lanes must not read row metadata.
      for(int t=0;t<tokens;++t) {
        const int absolute=windows[lane]-tokens+t;
        ps[lane*tokens+t]=t<counts[lane]?absolute:0x7fffffff;
        for(int h=0;h<H;++h)for(int d=0;d<D;++d)
          vs[((std::size_t(lane)*tokens+t)*H+h)*D+d]=t<counts[lane]?
              f32_to_bf16((two_class || marker(absolute))?value_at(rs[lane],h,d)*(two_class && absolute>=16384 ? 2.f : 1.f):0.f):0x7FC0;
        if(two_class && t<counts[lane]) {
          for(int h=0;h<QH;++h) qs[((std::size_t(lane)*tokens+t)*QH+h)*D]=f32_to_bf16(8.f+.5f*h);
          if(absolute>=16384)for(int h=0;h<H;++h)ks[((std::size_t(lane)*tokens+t)*H+h)*D]=f32_to_bf16(6.f);
        }
        if(t>=counts[lane]) {
          std::fill(qs.begin()+(lane*tokens+t)*QH*D,qs.begin()+(lane*tokens+t+1)*QH*D,0x7FC0);
          std::fill(ks.begin()+(lane*tokens+t)*H*D,ks.begin()+(lane*tokens+t+1)*H*D,0x7FC0);
        }
      }
    }
    upload(q,qs);upload(k,ks);upload(v,vs);upload(positions,ps);upload(rows,rs);upload(valid,std::vector<std::int32_t>(counts.begin(),counts.end()));
  }
  void run(Cache &cache,ninfer::DeviceArena &arena,int maxkeys,cudaStream_t stream) {
    arena.reset();
    ignis::nvfp4_attention(ninfer::Tensor(q.p,ninfer::DType::BF16,{D,QH,tokens,batch}),
      ninfer::Tensor(k.p,ninfer::DType::BF16,{D,H,tokens,batch}),ninfer::Tensor(v.p,ninfer::DType::BF16,{D,H,tokens,batch}),
      ninfer::Tensor(positions.p,ninfer::DType::I32,{tokens,batch}),ninfer::Tensor(valid.p,ninfer::DType::I32,{batch}),
      ninfer::Tensor(rows.p,ninfer::DType::I32,{batch}),ninfer::Tensor{},.0625f,cache.view,
      ninfer::ops::GqaExecutionEnvelope{1,std::uint32_t(maxkeys)},arena,
      // Tensor reference must remain an lvalue through the call.
      output(),stream);
  }
  ninfer::Tensor output_tensor;
  ninfer::Tensor &output(){output_tensor=ninfer::Tensor(out.p,ninfer::DType::BF16,{D,QH,tokens,batch});return output_tensor;}
  void verify(const std::vector<int>&windows,const std::vector<int>&counts,const char *label,bool two_class=false) {
    const auto host=download<std::uint16_t>(out.p,out.bytes/2);
    double maximum=0;std::size_t wrong=0;
    // uBF bounds each normalized BF16 split. Weighted split error <= uBF
    // mean|V|; final BF16 store contributes another uBF relative term.
    // gamma(8*21+32) conservatively covers 21 F32 weighted partial merges,
    // normalization and reciprocal/multiply. QK=0 makes scores/P exact.
    constexpr double uBF=1./256, uF=1./16777216;
    constexpr double gamma=(8*21+32)*uF/(1-(8*21+32)*uF);
    for(int lane=0;lane<batch;++lane)for(int t=0;t<tokens;++t)for(int h=0;h<QH;++h)for(int d=0;d<D;++d) {
      const auto index=((std::size_t(lane)*tokens+t)*QH+h)*D+d;
      const double actual=bf16_to_f32(host[index]);
      if(t>=counts[lane]){wrong+=host[index]!=0;continue;}
      const int keys=windows[lane]-tokens+t+1;
      double ratio=double(marker_count(keys))/keys;
      if(two_class) {
        const int second=std::max(0,keys-16384),first=keys-second;
        const double weight=std::exp((8.+.5*h)*6.*.0625);
        ratio=(first+2.*second*weight)/(first+second*weight);
      }
      const double expected=double(value_at(1-lane,h/6,d))*ratio;
      const double error=std::abs(actual-expected);maximum=std::max(maximum,error);
      // Nonuniform scores add BF16 P rounding. A conservative positive-sum
      // FP32 gamma(4*keys+200) covers accumulation/rescale/normalization.
      // This is an arithmetic fixture budget, never a model-quality margin.
      const double steps=4.*keys+200.;
      const double nonuniform_gamma=steps*uF/(1.-steps*uF);
      const double relative=two_class ? std::pow(1+uBF,3)*(1+nonuniform_gamma)-1 :
          (1+uBF)*(1+uBF)*(1+gamma)-1;
      const double bound=relative*std::abs(expected);
      wrong+=!std::isfinite(actual) || error>bound;
    }
    std::printf("%s: max_abs %.9g, wrong %zu\n",label,maximum,wrong);
    check(wrong==0,std::string(label)+": decoded BF16 analytic mean/rounding bound/invalid +0");
  }
};

// Exercise the private XQA query itself, not the public fallback-sized
// arena that otherwise leaves >64MiB around a roughly2MiB XQA allocation.
// The tail redzone detects missing MemSegmenter sizeof(T) alignment padding.
void direct_xqa_exact_workspace(Cache &cache,Inputs &input,int maximum,cudaStream_t stream) {
  constexpr std::size_t redzone=65536;
  // Both metadata-prefix lengths and two valid256-byte-aligned caller bases
  // must fit the query including the worst-case absolute scratch alignment.
  for(int limit:{20000,maximum})for(std::size_t offset:{std::size_t(0),std::size_t(256)}) {
    const auto exact=ignis_nv4_xqa_workspace(limit);
    DeviceBytes storage(exact+redzone+256);
    auto *workspace=static_cast<std::uint8_t*>(storage.p)+offset;
    const IgnisNv4XqaCall call{input.q.p,input.out.p,cache.k.p,cache.v.p,cache.ks.p,cache.vs.p,
      static_cast<const std::int32_t*>(cache.tables.p),input.rows.as<std::int32_t>(),
      input.positions.as<std::int32_t>(),input.valid.as<std::int32_t>(),
      std::uint32_t(cache.stride),2,std::uint32_t(cache.pages),std::uint32_t(limit),std::uint32_t(input.tokens),.0625f};
    bool rejected=false;
    try { ignis_nv4_xqa_run(call,workspace,exact-1,stream); }
    catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"private XQA rejects one-byte-short workspace");
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    MOE_CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    // The private API owns active output rows only; public gate_output handles
    // inactive rows. Initialize these rows here rather than assuming a write.
    MOE_CUDA(cudaMemsetAsync(input.out.p,0,input.out.bytes,stream));
    ignis_nv4_xqa_run(call,workspace,exact,stream);
    MOE_CUDA(cudaStreamEndCapture(stream,&graph));
    MOE_CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    auto *guard=workspace+exact;
    const std::size_t sem=256+(((limit+63)/64*4+255)/256)*256;
    for(int window:{2048,20000,2048,20000})for(int valid:{8,7,0}) {
      input.update({window},{valid});
      MOE_CUDA(cudaMemsetAsync(workspace,0xFF,exact,stream));
      MOE_CUDA(cudaMemsetAsync(guard,0x6D,redzone,stream));
      if(offset)MOE_CUDA(cudaMemsetAsync(storage.p,0x2C,offset,stream));
      MOE_CUDA(cudaStreamSynchronize(stream));
      const auto before=download<std::uint8_t>(guard,redzone);
      const auto prefix=download<std::uint8_t>(storage.p,offset);
      MOE_CUDA(cudaGraphLaunch(executable,stream));MOE_CUDA(cudaStreamSynchronize(stream));
      check(download<std::uint8_t>(guard,redzone)==before,"private exact XQA workspace tail redzone");
      check(download<std::uint8_t>(storage.p,offset)==prefix,"private offset XQA workspace prefix redzone");
      const auto counters=download<std::uint32_t>(workspace+sem,4);
      check(std::all_of(counters.begin(),counters.end(),[](auto n){return n==0;}),"private XQA semaphore reset/wrap");
      input.verify({window},{valid},"private exact XQA activeS8/21");
    }
    MOE_CUDA(cudaGraphExecDestroy(executable));MOE_CUDA(cudaGraphDestroy(graph));
  }
}

// Every small width uses the nominal8-row XQA specialization with device
// actualQ length. C2 must slice Q/output/positions/valid by actual token width
// and must reset/reuse metadata and semaphore scratch between the two lanes.
void small_width_batch_graphs(Cache &cache,int maximum,cudaStream_t stream) {
  check(ignis::nvfp4_attention_workspace_bytes(128,9,1)>=ignis_nv4_xqa_workspace(128),
        "max-width9 workspace covers the intermediate T8 XQA route");
  for(int width:{1,2,4,8})for(int batch:{1,2}) {
    Inputs input(width,batch);
    ninfer::DeviceArena arena(ignis::nvfp4_attention_workspace_bytes(maximum,width,batch));
    input.update(std::vector<int>(batch,2048),std::vector<int>(batch,width));
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    MOE_CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    input.run(cache,arena,maximum,stream);
    MOE_CUDA(cudaStreamEndCapture(stream,&graph));
    MOE_CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    std::vector<std::vector<int>> masks;
    if(batch==1)masks={{width},{width-1},{0}};
    else masks={{width,width},{width-1,width},{0,width},{width,0},{0,0}};
    // The8-key prefix has a nonzero value only at key0: using nominal8
    // instead of actualT for causal alignment changes the mean materially.
    for(int swap:{0,1,2})for(const auto &valid:masks) {
      const std::vector<int> windows=batch==1?std::vector<int>{swap==2?8:(swap?20000:2048)}:
          (swap==2?std::vector<int>{8,20000}:
           (swap?std::vector<int>{20000,2048}:std::vector<int>{2048,20000}));
      input.update(windows,valid);
      MOE_CUDA(cudaMemsetAsync(arena.base(),0xFF,arena.capacity(),stream));
      MOE_CUDA(cudaGraphLaunch(executable,stream));MOE_CUDA(cudaStreamSynchronize(stream));
      input.verify(windows,valid,"small XQA actualT/C2/tail/inactive graph");
    }
    MOE_CUDA(cudaGraphExecDestroy(executable));MOE_CUDA(cudaGraphDestroy(graph));
  }
}

void dynamic_graph_and_bands(cudaStream_t stream) {
  constexpr int maximum=32769;
  Cache cache(maximum);initialize_history(cache,maximum,stream);
  ignis::nvfp4_prepare();
  Inputs xqa(8,1);
  const auto bytes=ignis::nvfp4_attention_workspace_bytes(maximum,8,1);
  ninfer::DeviceArena arena(bytes);
  xqa.update({2048},{8});
  cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
  MOE_CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  xqa.run(cache,arena,maximum,stream);
  MOE_CUDA(cudaStreamEndCapture(stream,&graph));
  MOE_CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
  for(int window:{2048,20000,2048,20000})for(int valid:{8,7,0}) {
    xqa.update({window},{valid});MOE_CUDA(cudaMemsetAsync(arena.base(),0xFF,bytes,stream));
    MOE_CUDA(cudaGraphLaunch(executable,stream));MOE_CUDA(cudaStreamSynchronize(stream));
    xqa.verify({window},{valid},"XQA dynamic actualS and valid");
  }
  MOE_CUDA(cudaGraphExecDestroy(executable));MOE_CUDA(cudaGraphDestroy(graph));
  direct_xqa_exact_workspace(cache,xqa,maximum,stream);
  small_width_batch_graphs(cache,maximum,stream);
  for(int width:{1,65}) {
    Inputs prefill(width,2);
    const auto size=ignis::nvfp4_attention_workspace_bytes(maximum,width,2);
    ninfer::DeviceArena scratch(size);
    for(int window:{16383,16384,16385,16450,32769}) {
      const std::vector<int> counts{width,width==65?63:1},windows{window,2048};
      prefill.update(windows,counts);MOE_CUDA(cudaMemsetAsync(scratch.base(),0xFF,size,stream));
      prefill.run(cache,scratch,maximum,stream);MOE_CUDA(cudaStreamSynchronize(stream));
      prefill.verify(windows,counts,"prefill bands C2/partial/empty suffix");
    }
  }
}

void nonuniform_score_scale_and_rescale(cudaStream_t stream) {
  constexpr int maximum=32769;
  Cache cache(maximum);initialize_history(cache,maximum,stream,true);
  for(int width:{8,65}) {
    const int batch=width==8?1:2;
    Inputs input(width,batch);
    ninfer::DeviceArena arena(ignis::nvfp4_attention_workspace_bytes(maximum,width,batch));
    const std::vector<int> windows=batch==1?std::vector<int>{20000}:std::vector<int>{32769,20000};
    const std::vector<int> counts(batch,width);
    input.update(windows,counts,true);
    MOE_CUDA(cudaMemsetAsync(arena.base(),0xFF,arena.capacity(),stream));
    input.run(cache,arena,maximum,stream);MOE_CUDA(cudaStreamSynchronize(stream));
    input.verify(windows,counts,"nonuniform QK scale/new max across band",true);
  }
}

// Each KV head has16 distinct power-of-two SFs. A/B use different SF
// permutations and alternate after16K, so HND head/token strides and K32's
// odd/even SF halfwords cannot hide behind uniform scale bytes. Each query
// selects one group with two nonzero dimensions; acrossT8 every KV head
// queries all16 groups. Inverse-SF Q makes every correct B dot exactly15
// (scaled0.9375), while A dots are0: no saturated-softmax discriminator.
void all_scale_groups_have_analytic_scores(cudaStream_t stream) {
  constexpr int keys=20000, tokens=8, span=16384;
  const auto second=[](int position){return position>=16384 && (position&1);};
  const auto exponent=[](int group,int head,bool b){return ((group*5+head*3+(b?0:7))&15)-7;};
  using Row=std::array<std::uint16_t,D>;
  std::array<std::array<Row,H>,2> key_rows{},value_rows{};
  for(int b=0;b<2;++b)for(int h=0;h<H;++h) {
    std::vector<float> distinct;
    for(int g=0;g<16;++g) {
      const float sf=std::ldexp(1.f,exponent(g,h,b));distinct.push_back(sf);
      key_rows[b][h][g*16+15]=f32_to_bf16(6.f*sf); // exact scale anchor, Q=0 here
      if(b){key_rows[b][h][g*16]=f32_to_bf16(6.f*sf);key_rows[b][h][g*16+1]=f32_to_bf16(3.f*sf);}
    }
    std::sort(distinct.begin(),distinct.end());
    check(std::unique(distinct.begin(),distinct.end())==distinct.end(),"16 distinct SF groups per KVhead/class");
    for(int d=0;d<D;++d)value_rows[b][h][d]=f32_to_bf16(value_at(1,h,d)*(b?2.f:1.f));
    const auto encoded=encode(key_rows[b][h].data()),vencoded=encode(value_rows[b][h].data());
    for(int d=0;d<D;++d) {
      check(decoded(encoded,d)==key_rows[b][h][d],"multiscale K is independently NV4/BF16 exact");
      check(decoded(vencoded,d)==value_rows[b][h][d],"multiscale V is independently NV4/BF16 exact");
    }
  }
  Cache cache(keys);
  std::vector<std::uint16_t> history_k(std::size_t(keys)*H*D),history_v(history_k.size());
  std::vector<std::int32_t> positions(keys);std::iota(positions.begin(),positions.end(),0);
  for(int t=0;t<keys;++t)for(int h=0;h<H;++h) {
    const int b=second(t);const auto at=(std::size_t(t)*H+h)*D;
    std::copy(key_rows[b][h].begin(),key_rows[b][h].end(),history_k.begin()+at);
    std::copy(value_rows[b][h].begin(),value_rows[b][h].end(),history_v.begin()+at);
  }
  DeviceBytes hk(history_k.size()*2),hv(history_v.size()*2),hp(positions.size()*4),hr(4);
  upload(hk,history_k);upload(hv,history_v);upload(hp,positions);upload(hr,std::vector<std::int32_t>{1});
  ignis::nvfp4_append(ninfer::Tensor(hk.p,ninfer::DType::BF16,{D,H,keys,1}),
      ninfer::Tensor(hv.p,ninfer::DType::BF16,{D,H,keys,1}),ninfer::Tensor(hp.p,ninfer::DType::I32,{keys,1}),
      ninfer::Tensor{},ninfer::Tensor(hr.p,ninfer::DType::I32,{1}),cache.view,stream);
  MOE_CUDA(cudaStreamSynchronize(stream));
  Inputs input(tokens,1);std::vector<std::uint16_t> q(tokens*QH*D,0);
  std::array<std::array<bool,16>,H> queried{};
  for(int t=0;t<tokens;++t)for(int h=0;h<QH;++h) {
    const int g=(h+3*t)&15,e=exponent(g,h/6,true);queried[h/6][g]=true;
    const auto at=(std::size_t(t)*QH+h)*D+g*16;
    q[at]=f32_to_bf16(std::ldexp(2.f,-e));q[at+1]=f32_to_bf16(std::ldexp(1.f,-e));
  }
  for(const auto &head:queried)check(std::all_of(head.begin(),head.end(),[](bool x){return x;}),"each KVhead queries all SF groups");
  upload(input.q,q);upload(input.k,std::vector<std::uint16_t>(history_k.end()-tokens*H*D,history_k.end()));
  upload(input.v,std::vector<std::uint16_t>(history_v.end()-tokens*H*D,history_v.end()));
  upload(input.positions,std::vector<std::int32_t>(positions.end()-tokens,positions.end()));
  upload(input.valid,std::vector<std::int32_t>{tokens});upload(input.rows,std::vector<std::int32_t>{1});
  const auto verify=[&](const char *label){
    const auto actual=download<std::uint16_t>(input.out.p,input.out.bytes/2);std::size_t wrong=0;double maximum=0;
    constexpr double uBF=1./256,uF=1./16777216;const double weight=std::exp(15./16.);
    for(int t=0;t<tokens;++t) {
      const int count=keys-tokens+t+1,b=std::max(0,count/2-8192),a=count-b;
      const double ratio=(a+2.*b*weight)/(a+b*weight);
      // Same positive-column BF16 P/partial/store and conservative F32 sum
      // budget as the other nonuniform fixture; not a codec/model-quality gate.
      const double steps=4.*count+200.,gamma=steps*uF/(1.-steps*uF);
      const double relative=std::pow(1.+uBF,3)*(1.+gamma)-1.;
      for(int h=0;h<QH;++h)for(int d=0;d<D;++d) {
        const double expected=value_at(1,h/6,d)*ratio;
        const double observed=bf16_to_f32(actual[(t*QH+h)*D+d]);
        const double error=std::abs(observed-expected);maximum=std::max(maximum,error);
        wrong+=!std::isfinite(observed)||error>relative*std::abs(expected);
      }
    }
    std::printf("%s: max_abs %.9g, wrong %zu\n",label,maximum,wrong);
    check(wrong==0,std::string(label)+": independent two-class exp reference");
  };
  ninfer::DeviceArena arena(ignis::nvfp4_attention_workspace_bytes(keys,tokens,1));
  input.run(cache,arena,keys,stream);MOE_CUDA(cudaStreamSynchronize(stream));verify("XQA all16SF/odd-half/HND");
  // Run the same decoded cache and identical Q through the independent band
  // consumer, carrying FP32 state over the16K boundary, against the same formula.
  arena.reset();const auto kb=arena.alloc(ninfer::DType::BF16,{D,span,H}),vb=arena.alloc(ninfer::DType::BF16,{D,span,H});
  const auto ca=arena.alloc(ninfer::DType::FP32,{D,QH,tokens}),cm=arena.alloc(ninfer::DType::FP32,{tokens,QH}),cl=arena.alloc(ninfer::DType::FP32,{tokens,QH});
  MOE_CUDA(cudaMemsetAsync(ca.data,0xFF,ca.bytes(),stream));MOE_CUDA(cudaMemsetAsync(cm.data,0xFF,cm.bytes(),stream));MOE_CUDA(cudaMemsetAsync(cl.data,0xFF,cl.bytes(),stream));
  for(int begin=0;begin<keys;begin+=span) {
    ignis::nvfp4_decode_band(cache.view,input.rows.as<std::int32_t>(),input.positions.as<std::int32_t>(),input.valid.as<std::int32_t>(),0,tokens,begin,span,kb.data,vb.data,stream);
    const IgnisNv4PrefillBand band{input.q.p,kb.data,vb.data,input.positions.as<std::int32_t>(),input.valid.as<std::int32_t>(),input.out.p,
      static_cast<float*>(ca.data),static_cast<float*>(cm.data),static_cast<float*>(cl.data),tokens,begin,std::min(begin+span,keys),span,begin+span<keys,.0625f};
    ignis_nv4_prefill_band(band,stream);
  }
  MOE_CUDA(cudaStreamSynchronize(stream));verify("band all16SF/same Q/new maximum");
}

void checkpoint_four_plane_copy() {
  ignis_seq_pool_spec spec{};spec.num_kv_heads=4;spec.head_dim=256;spec.kv_format=IGNIS_KV_FORMAT_NVFP4;
  spec.kv_page_group_count=16;spec.max_context_tokens=192;spec.slot_count=3;
  spec.kv_num_layers=1;spec.gdn_num_layers=1;spec.gdn_conv_channels=4;spec.gdn_value_heads=1;spec.gdn_head_dim=4;
  spec.vocab=32;spec.retained_slot_count=1;
  ignis_seq_pool *pool=nullptr;ignis_seq *source=nullptr,*claim=nullptr;ignis_seq_checkpoint *checkpoint=nullptr;
  auto ok=[](int rc,const char *label){if(rc)throw std::runtime_error(std::string(label)+": "+ignis_seq_last_error());};
  ok(ignis_seq_pool_create(&spec,&pool),"NV4 checkpoint pool");
  ok(ignis_seq_alloc(pool,192,&source),"NV4 checkpoint source");
  source->position=65;for(auto &p:source->gqa_positions)p=65;for(auto &p:source->gdn_positions)p=65;
  const auto page=source->kv.page_ids()[1];
  std::vector<std::vector<std::uint8_t>> before;
  for(std::size_t i=0;i<pool->kv_pool.plane_count();++i) {
    const auto &plane=pool->kv_pool.plane(i);std::vector<std::uint8_t> pattern(plane.nb[3],std::uint8_t(31+i));
    MOE_CUDA(cudaMemcpy(static_cast<char*>(plane.data)+page*plane.nb[3],pattern.data(),pattern.size(),cudaMemcpyHostToDevice));before.push_back(pattern);
  }
  ok(ignis_seq_checkpoint_capture(pool,source,65,0,&checkpoint),"NV4 checkpoint capture");
  for(std::size_t i=0;i<pool->kv_pool.plane_count();++i){const auto &p=pool->kv_pool.plane(i);MOE_CUDA(cudaMemset(static_cast<char*>(p.data)+page*p.nb[3],0xEE,p.nb[3]));}
  ok(ignis_seq_alloc_from_checkpoint(pool,192,checkpoint,&claim),"NV4 checkpoint claim");
  const auto restored=claim->kv.page_ids()[0];
  check(restored!=page,"checkpoint tail is physically distinct");
  for(std::size_t i=0;i<pool->kv_pool.plane_count();++i){const auto &p=pool->kv_pool.plane(i);check(download<std::uint8_t>(static_cast<char*>(p.data)+restored*p.nb[3],p.nb[3])==before[i],"NV4 checkpoint copies all code/scale planes");}
  ignis_seq_release(pool,claim);ignis_seq_checkpoint_release(pool,checkpoint);ignis_seq_release(pool,source);ignis_seq_pool_free(pool);
}
} // namespace
int main() {
  try {
    MOE_CUDA(cudaSetDevice(0));cudaStream_t stream=nullptr;MOE_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    codec_scatter_and_unselected_bytes(stream);dynamic_graph_and_bands(stream);nonuniform_score_scale_and_rescale(stream);all_scale_groups_have_analytic_scores(stream);checkpoint_four_plane_copy();
    MOE_CUDA(cudaStreamDestroy(stream));
  }catch(const std::exception &error){std::fprintf(stderr,"FATAL NV4 regression: %s\n",error.what());return 1;}
  std::printf("nvfp4 KV regression: %s\n",g_failed?"FAILED":"ok");return g_failed?1:0;
}
