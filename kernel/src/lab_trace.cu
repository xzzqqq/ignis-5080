// Offline, model-owned capture/replay. No serving route enables it.
#include "ignis_gqa_native.h"
#include "lab_trace.h"
#include "gqa_candidate.h"
#include "lab_segment.h"
#include "lab_profile.h"
#include "model_internal.h"
#include "ignis_trellis_candidate.h"
#include "ignis_gqa_workspace.h"
#include "ninfer/ops/gqa_attention.h"
#include "core/arena.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include <dlfcn.h>
#include <map>
#include <memory>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Minimal toolkit installs omit cuda_profiler_api.h; these stable runtime
// entry points are provided by the same cudart already linked by the leaf.
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);

namespace {
thread_local std::string error;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
  void *p = nullptr;
  explicit Device(uint64_t n) { ck(cudaMalloc(&p,n)); }
  ~Device() { if(p) cudaFree(p); }
};
std::map<std::string,ignis_trellis_weight> weights(const ignis_model &m) {
  std::map<std::string,ignis_trellis_weight> result;
  result["head"] = m.trellis_head;
  auto tail = [&](const std::string &p, const TrellisTail &t) {
    result[p+".output"]=t.output; result[p+".gate"]=t.gate_proj;
    result[p+".up"]=t.up_proj; result[p+".down"]=t.down;
  };
  for (size_t i=0;i<m.layers.size();++i) {
    auto p="layer."+std::to_string(i);
    const auto &l=m.layers[i];
    if(l.kind==IGNIS_LAYER_GQA) {
      const auto &w=l.gqa.trellis;
      result[p+".q"]=w.q_proj; result[p+".k"]=w.k_proj; result[p+".v"]=w.v_proj; tail(p,w.tail);
    } else {
      const auto &w=l.gdn.trellis;
      result[p+".qkv"]=w.in_proj_qkv; result[p+".z"]=w.in_proj_z; tail(p,w.tail);
    }
  }
  return result;
}
struct View {
  std::string name;
  uint64_t t=0,s=0,v=0;
  ignis_trellis_weight w{};
  std::string json() const {
    std::ostringstream o;
    o<<"{\"name\":\""<<name<<"\",\"trellis_offset\":"<<t<<",\"suh_offset\":"<<s
     <<",\"svh_offset\":"<<v<<",\"k2\":"<<w.k2<<",\"in\":"<<w.in<<",\"out\":"<<w.out
     <<",\"tile_stride\":"<<w.tile_stride<<"}";
    return o.str();
  }
};
View identify(const std::map<std::string,ignis_trellis_weight> &all, const ignis_trellis_weight &w) {
  for(const auto &[name,b]:all) {
    auto t=reinterpret_cast<uintptr_t>(w.trellis), bt=reinterpret_cast<uintptr_t>(b.trellis);
    auto s=reinterpret_cast<uintptr_t>(w.suh), bs=reinterpret_cast<uintptr_t>(b.suh);
    auto v=reinterpret_cast<uintptr_t>(w.svh), bv=reinterpret_cast<uintptr_t>(b.svh);
    if(w.k2==b.k2 && w.in==b.in && w.out<=b.out && t>=bt && t<bt+uint64_t(b.in)*b.out*b.k2/16 &&
       s==bs && v>=bv && v+2ull*w.out<=bv+2ull*b.out) return {name,t-bt,s-bs,v-bv,w};
  }
  throw std::runtime_error("unregistered Trellis weight view");
}
ignis_trellis_weight resolve(const ignis_model &m,const ignis_lab_weight_view &v) {
  const auto all=weights(m);
  auto it=all.find(v.name ? v.name : "");
  if(it==all.end()) throw std::runtime_error("unknown replay weight");
  auto w=it->second;
  // Column views preserve input scales and the physical tile-row stride.
  const uint64_t column=v.svh_offset/2;
  const uint64_t expected_t=(column/16)*16ull*v.k2;
  if(v.k2!=w.k2 || v.in!=w.in || !v.out || v.out%128 || column%128 ||
     v.svh_offset%2 || column+v.out>w.out || v.suh_offset || v.trellis_offset!=expected_t ||
     (v.tile_stride ? v.tile_stride : v.out/16)!=(w.tile_stride ? w.tile_stride : w.out/16))
    throw std::runtime_error("replay weight geometry mismatch");
  w.trellis=static_cast<const char *>(w.trellis)+v.trellis_offset;
  w.svh=static_cast<const char *>(w.svh)+v.svh_offset;
  w.out=v.out;w.tile_stride=v.tile_stride;
  return w;
}
// Select on the GPU before any tensor payload crosses PCIe. Graphs remain
// reusable across the complete request; skipped calls execute only tiny guards.
// Export is an explicit offline diagnostic attached to one selected replay.
// It never substitutes recorded baseline bytes for current candidate results.
struct TensorExport {
  std::ofstream data,metadata;
  uint64_t offset=0;
  explicit TensorExport(const char *directory){
    data.open(std::string(directory)+"/candidate.bin",std::ios::binary|std::ios::trunc);
    metadata.open(std::string(directory)+"/tensors.jsonl",std::ios::trunc);
    if(!data || !metadata)throw std::runtime_error("cannot create tensor export");
  }
  void tensor(const std::string &name,const std::string &purpose,const void *source,
      uint64_t bytes,const char *dtype,const std::vector<int64_t> &shape,bool device=true){
    if(!source || !bytes)return;
    // Bounded scratch avoids allocating another full physical state image.
    std::vector<char> chunk(std::min<uint64_t>(bytes,8ull<<20));
    for(uint64_t i=0;i<bytes;i+=chunk.size()){
      auto n=std::min<uint64_t>(chunk.size(),bytes-i);
      if(device)ck(cudaMemcpy(chunk.data(),static_cast<const char *>(source)+i,n,cudaMemcpyDeviceToHost));
      else std::memcpy(chunk.data(),static_cast<const char *>(source)+i,n);
      data.write(chunk.data(),n);
    }
    metadata<<"{\"name\":\""<<name<<"\",\"purpose\":\""<<purpose<<"\",\"dtype\":\""<<dtype
      <<"\",\"shape\":[";
    for(size_t i=0;i<shape.size();++i){if(i)metadata<<",";metadata<<shape[i];}
    metadata<<"],\"file\":\"candidate.bin\",\"offset\":"<<offset<<",\"bytes\":"<<bytes<<"}\n";
    offset+=bytes;if(!data || !metadata)throw std::runtime_error("tensor export write failed");
  }
  void tensor(const std::string &name,const std::string &purpose,const ninfer::Tensor &t){
    const auto *dtype=t.dtype==ninfer::DType::BF16?"bf16":t.dtype==ninfer::DType::U8?"u8":t.dtype==ninfer::DType::I32?"i32":"f32";
    tensor(name,purpose,t.data,t.data?t.bytes():0,dtype,{t.ne[0],t.ne[1],t.ne[2],t.ne[3]});
  }
  void weight(const std::string &name,const ignis_trellis_weight &w){
    tensor(name+".trellis","model_weight_encoded",w.trellis,uint64_t(w.in)*w.out*w.k2/16,"u8",{int64_t(w.in)*w.out*w.k2/16});
    tensor(name+".suh","model_weight_scale",w.suh,uint64_t(w.in)*2,"f16",{w.in});
    tensor(name+".svh","model_weight_scale",w.svh,uint64_t(w.out)*2,"f16",{w.out});
    metadata<<"{\"weight_geometry\":\""<<name<<"\",\"in\":"<<w.in<<",\"out\":"<<w.out<<",\"k2\":"<<w.k2
      <<",\"tile_stride\":"<<w.tile_stride<<",\"effective_tile_stride\":"<<(w.tile_stride?w.tile_stride:w.out/16)
      <<",\"encoding\":\"trellis-mul1\",\"scale_dtype\":\"f16\",\"hadamard_block\":128}\n";
  }
  void weight(const std::string &name,const ninfer::Weight &w){
    if(!w.payload || !w.payload_bytes)return;
    const char *dtype=w.qtype==ninfer::QType::BF16_CTRL?"bf16":w.qtype==ninfer::QType::FP32_CTRL?"f32":w.qtype==ninfer::QType::I32_CTRL?"i32":"u8";
    std::vector<int64_t> shape;
    if(std::strcmp(dtype,"u8")==0)shape={int64_t(w.payload_bytes)};
    else {
      // Artifact weights use row-major logical axes; tensor exports use the
      // activation convention ne[0]-contiguous. Preserve both explicitly.
      if(!w.ndim || w.ndim>4)throw std::runtime_error("invalid dense weight rank");
      for(uint32_t i=w.ndim;i>0;--i)shape.push_back(w.shape[i-1]);
      while(shape.size()<4)shape.push_back(1);
    }
    tensor(name,"model_weight",w.payload,w.payload_bytes,dtype,shape);
    metadata<<"{\"weight_geometry\":\""<<name<<"\",\"logical_layout\":\"row_major\",\"logical_shape\":[";
    for(uint32_t i=0;i<w.ndim;++i){if(i)metadata<<",";metadata<<w.shape[i];}
    metadata<<"],\"qtype\":"<<uint32_t(w.qtype)<<"}\n";
  }
};
thread_local std::unique_ptr<TensorExport> tensor_export;
void export_layer_weights(const ignis_model &m,uint32_t layer,uint32_t kind){
  if(!tensor_export)return;
  const auto prefix="layer."+std::to_string(layer)+".";
  for(const auto &[name,w]:weights(m))if(name.starts_with(prefix)){
    const auto suffix=name.substr(prefix.size());
    const bool mlp=suffix=="gate" || suffix=="up" || suffix=="down";
    if((kind==IGNIS_SEGMENT_MLP)==mlp)tensor_export->weight(name,w);
  }
  tensor_export->tensor("rms_epsilon","model_parameter",&m.rms_norm_eps,4,"f32",{1},false);
  const auto &l=m.layers.at(layer);
  if(kind==IGNIS_SEGMENT_MLP){
    tensor_export->weight(prefix+"post_attention_norm",l.kind==IGNIS_LAYER_GQA?l.gqa.post_attention_norm:l.gdn.post_attention_norm);return;
  }
  if(l.kind==IGNIS_LAYER_GQA){const auto &w=l.gqa;
    tensor_export->tensor("rope_inv_frequency","model_parameter",m.text_rope.inv_frequency,128*4,"f32",{128},false);
    tensor_export->tensor("rope_attention_factor","model_parameter",&m.text_rope.attention_factor,4,"f32",{1},false);
    tensor_export->weight(prefix+"input_norm",w.input_norm);tensor_export->weight(prefix+"query_norm",w.query_norm);
    tensor_export->weight(prefix+"key_norm",w.key_norm);
  }else{const auto &w=l.gdn;
    tensor_export->weight(prefix+"input_norm",w.input_norm);tensor_export->weight(prefix+"a_log",w.a_log);
    tensor_export->weight(prefix+"dt_bias",w.dt_bias);tensor_export->weight(prefix+"convolution",w.convolution);
    tensor_export->weight(prefix+"a_b_projection",w.a_b_projection);tensor_export->weight(prefix+"norm",w.norm);
  }
}
struct CaptureControl {
  std::unique_ptr<Device> gate;
  void *mapped=nullptr;
  void allocate(void **host,uint64_t bytes){
    ck(cudaHostAlloc(host,bytes+24,cudaHostAllocMapped));
    ck(cudaHostGetDevicePointer(&mapped,*host,0));std::memset(*host,0,bytes+24);
    gate=std::make_unique<Device>(16);ck(cudaMemset(gate->p,0,16));
  }
};
__global__ void lab_select_record(unsigned long long *counter,unsigned long long *gate,
    unsigned long long start,unsigned long long limit){
  if(threadIdx.x==0){const auto ordinal=atomicAdd(counter,1ull);gate[0]=ordinal;
    gate[1]=ordinal>=start && (!limit || ordinal-start<limit);}
}
__global__ void lab_selected_copy(char *dst,const char *src,uint64_t width,uint64_t rows,
    uint64_t pitch,const unsigned long long *gate){
  if(!gate[1])return;
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<width*rows;i+=uint64_t(gridDim.x)*blockDim.x)
    dst[i]=src[(i/width)*pitch+i%width];
}
__global__ void lab_selected_finish(uint32_t *seen,uint64_t *source,const unsigned long long *gate){
  if(threadIdx.x==0 && gate[1]){*source=gate[0];++*seen;__threadfence_system();}
}
struct Call : CaptureControl {
  View w,up;
  bool swiglu=false;
  uint32_t tokens=0,ldy=0,epilogue=0;
  uint64_t nx=0,ny=0;
  void *host=nullptr;
  ~Call(){ if(host) cudaFreeHost(host); }
  char *x() const {return static_cast<char *>(host);}
  char *initial() const {return x()+nx;}
  char *output() const {return initial()+ny;}
  uint32_t *seen() const {return reinterpret_cast<uint32_t *>(output()+ny);}
  uint64_t bytes() const {return nx+2*ny;}
  uint64_t source()const{return *reinterpret_cast<const uint64_t *>(x()+((bytes()+7)&~7ull)+8);}
  void allocate(){CaptureControl::allocate(&host,bytes());}
};
// Full physical planes are deliberate: preserving physical page ids and slot
// selectors avoids rebasing errors and includes untouched bytes in diagnostics.
struct GqaCall : CaptureControl {
  std::string name;
  uint32_t layer=0,max_keys=0;
  std::vector<ninfer::Tensor> tensors;
  std::vector<uint64_t> offsets;
  uint64_t pre=0,state=0,output=0;
  void *host=nullptr;
  ~GqaCall(){if(host)cudaFreeHost(host);}
  uint64_t bytes()const{return pre+output+state;}
  char *base()const{return static_cast<char *>(host);}
  uint32_t *seen()const{return reinterpret_cast<uint32_t *>(base()+bytes());}
  uint64_t source()const{return *reinterpret_cast<const uint64_t *>(base()+((bytes()+7)&~7ull)+8);}
  void allocate(){CaptureControl::allocate(&host,bytes());}
  void add(const ninfer::Tensor &x){
    if(x.data && !x.is_contiguous())throw std::runtime_error("GQA capture requires contiguous tensors");
    offsets.push_back(pre);tensors.push_back(x);
    if(x.data)pre+=x.bytes();
  }
};
struct SegmentCall : CaptureControl {
  uint32_t layer=0;ignis_segment_profile profile{};
  std::string name;std::vector<ignis::SegmentBinding> bindings;
  std::vector<uint64_t> pre_offsets,post_offsets;
  uint64_t pre=0,post=0;void *host=nullptr;
  ~SegmentCall(){if(host)cudaFreeHost(host);}
  uint64_t bytes()const{return pre+post;}
  char *base()const{return static_cast<char *>(host);}
  uint32_t *seen()const{return reinterpret_cast<uint32_t *>(base()+bytes());}
  uint64_t source()const{return *reinterpret_cast<const uint64_t *>(base()+((bytes()+7)&~7ull)+8);}
  void allocate(){CaptureControl::allocate(&host,bytes());}
  void layout(){
    for(const auto &b:bindings){
      if(b.tensor.data && !b.tensor.is_contiguous())throw std::runtime_error("segment boundary must be contiguous");
      pre_offsets.push_back(pre);post_offsets.push_back(post);
      if(b.tensor.data){if(b.flags&(ignis::SEGMENT_INPUT|ignis::SEGMENT_MUTABLE))pre+=b.tensor.bytes();
        if(b.flags&(ignis::SEGMENT_MUTABLE|ignis::SEGMENT_OUTPUT))post+=b.tensor.bytes();}
    }
  }
};
std::string segment_name(uint32_t layer,uint32_t kind){
  return "layer."+std::to_string(layer)+(kind==1?".mlp_block":kind==2?".gqa_block":".gdn_block");
}
bool segment_selection(const ignis_model &m,const std::string &name){
  for(size_t i=0;i<m.layers.size();++i){
    if(name==segment_name(i,1) || name==segment_name(i,m.layers[i].kind==IGNIS_LAYER_GQA?2:3))return true;
  }return false;
}
uint32_t dtype_id(ninfer::DType d){
  if(d==ninfer::DType::BF16)return 0;if(d==ninfer::DType::U8)return 1;
  if(d==ninfer::DType::I32)return 2;if(d==ninfer::DType::FP32)return 3;throw std::runtime_error("unsupported GQA capture dtype");
}
}
namespace ignis {
struct LabTrace {
  enum Mode {Discover,Record} mode=Discover;
  std::map<std::string,std::unique_ptr<Call>> calls;
  std::map<std::string,std::unique_ptr<GqaCall>> gqa_calls;
  std::unique_ptr<GqaCall> gqa_eager;
  std::map<std::string,std::unique_ptr<SegmentCall>> segments;
  std::unique_ptr<SegmentCall> segment_eager;
  std::map<std::string,ignis_trellis_weight> named;
  std::unique_ptr<Call> eager;
  std::set<std::string> selection;
  std::set<uint32_t> token_selection;
  std::string graph;
  uint32_t ordinal=0,request=0,phase=0;
  uint64_t position=0,offset=0,count=0,pinned=0,pin_limit=0,disk_limit=0;
  std::ofstream data,meta;
  Device record_counter{8};
  uint64_t record_start=0,record_limit=0;
  bool selective()const{return record_start || record_limit;}
  template<class C> void begin(C &c,cudaStream_t stream){
    lab_select_record<<<1,1,0,stream>>>(static_cast<unsigned long long *>(record_counter.p),
      static_cast<unsigned long long *>(c.gate->p),record_start,record_limit);ck(cudaGetLastError());
  }
  template<class C> void copy(C &c,void *dst,const void *src,uint64_t width,uint64_t rows,uint64_t pitch,cudaStream_t stream){
    if(!selective()){ck(cudaMemcpy2DAsync(dst,width,src,pitch,width,rows,cudaMemcpyDeviceToHost,stream));return;}
    auto *target=static_cast<char *>(c.mapped)+(static_cast<char *>(dst)-static_cast<char *>(c.host));
    lab_selected_copy<<<128,256,0,stream>>>(target,static_cast<const char *>(src),width,rows,pitch,
      static_cast<const unsigned long long *>(c.gate->p));ck(cudaGetLastError());
  }
  template<class C> void finish(C &c,cudaStream_t stream){
    auto *base=static_cast<char *>(c.mapped)+c.bytes();
    lab_selected_finish<<<1,1,0,stream>>>(reinterpret_cast<uint32_t *>(base),reinterpret_cast<uint64_t *>(static_cast<char *>(c.mapped)+((c.bytes()+7)&~7ull)+8),
      static_cast<const unsigned long long *>(c.gate->p));ck(cudaGetLastError());
  }
  LabTrace(const ignis_model &m,const char *directory,uint64_t pin,uint64_t disk,const char *names,const uint32_t *token_filter,uint32_t token_filter_count,uint64_t start,uint64_t limit):named(weights(m)),pin_limit(pin),disk_limit(disk),record_start(start),record_limit(limit) {
    ck(cudaMemset(record_counter.p,0,8));
    for(uint32_t i=0;i<token_filter_count;++i)token_selection.insert(token_filter[i]);
    std::istringstream list(names ? names : "");std::string name;
    while(std::getline(list,name,',')) {
      bool attention=false;
      for(size_t i=0;i<m.layers.size();++i)
        if(m.layers[i].kind==IGNIS_LAYER_GQA && name=="layer."+std::to_string(i)+".attention")attention=true;
      if(!named.count(name) && !attention && !segment_selection(m,name))throw std::runtime_error("unknown capture operator: "+name);
      if(attention && (m.kv_format!=IGNIS_KV_FORMAT_HQ_E8_2B || token_selection!=std::set<uint32_t>{8}))
        throw std::runtime_error("GQA recording currently requires hq-e8-2b and explicit --tokens-per-call 8");
      if(name.ends_with(".gqa_block") && m.kv_format!=IGNIS_KV_FORMAT_HQ_E8_2B)
        throw std::runtime_error("GQA segment capture requires hq-e8-2b");
      if(segment_selection(m,name) && token_selection!=std::set<uint32_t>{8})
        throw std::runtime_error("finite segments require explicit --tokens-per-call 8");
      selection.insert(name);
    }
    bool segments_selected=false,legacy_selected=false;
    for(const auto &selected:selection){if(segment_selection(m,selected))segments_selected=true;else legacy_selected=true;}
    if(segments_selected && legacy_selected)throw std::runtime_error("segment and legacy operator capture scopes cannot be mixed");
    data.open(std::string(directory)+"/data.bin",std::ios::binary|std::ios::trunc);
    meta.open(std::string(directory)+"/records.jsonl",std::ios::trunc);
    if(!data || !meta) throw std::runtime_error("cannot create capture files");
  }
  void write(SegmentCall &c){
    const auto source_record=c.source();
    if(*c.seen()!=1)throw std::runtime_error("segment graph executed multiple times between flushes; corpus incomplete");
    if(c.bytes()>disk_limit-offset)throw std::runtime_error("segment capture byte budget exhausted; corpus incomplete");
    data.write(c.base(),c.bytes());
    const auto &p=c.profile;
    meta<<"{\"kind\":\"segment\",\"version\":1,\"reference\":\"original_layer_chain\",\"id\":"<<source_record<<",\"source_record\":"<<source_record<<",\"request\":"<<request
      <<",\"phase\":"<<phase<<",\"position\":"<<position<<",\"weight\":{\"name\":\""<<c.name<<"\"},\"up\":null"
      <<",\"layer\":"<<c.layer<<",\"tokens\":"<<p.tokens<<",\"offset\":"<<offset<<",\"bytes\":"<<c.bytes()
      <<",\"pre_bytes\":"<<c.pre<<",\"post_bytes\":"<<c.post<<",\"profile\":{\"kind\":"<<p.kind
      <<",\"tokens\":"<<p.tokens<<",\"batch\":"<<p.batch<<",\"hidden\":"<<p.hidden<<",\"intermediate\":"<<p.intermediate
      <<",\"max_keys\":"<<p.max_keys<<",\"slots\":"<<p.slots<<",\"flags\":"<<p.flags<<",\"rms_eps\":"<<p.rms_eps<<"},\"bindings\":[";
    for(size_t i=0;i<c.bindings.size();++i){const auto &b=c.bindings[i];if(i)meta<<",";
      meta<<"{\"name\":\""<<b.name<<"\",\"flags\":"<<b.flags<<",\"dtype\":"<<dtype_id(b.tensor.dtype)
          <<",\"present\":"<<(b.tensor.data?"true":"false")<<",\"shape\":[";
      for(int j=0;j<4;++j){if(j)meta<<",";meta<<b.tensor.ne[j];}meta<<"]}";
    }
    meta<<"]}\n";if(!data || !meta)throw std::runtime_error("segment capture disk write failed");
    offset+=c.bytes();++count;*c.seen()=0;
  }
  void write(GqaCall &c) {
    const auto source_record=c.source();
    if(*c.seen()!=1)throw std::runtime_error("GQA graph executed multiple times between capture flushes; corpus incomplete");
    if(c.bytes()>disk_limit-offset)throw std::runtime_error("capture byte budget exhausted; corpus incomplete");
    data.write(c.base(),c.bytes());
    meta<<"{\"kind\":\"gqa_attention\",\"version\":1,\"id\":"<<source_record
        <<",\"source_record\":"<<source_record<<",\"request\":"<<request<<",\"phase\":"<<phase<<",\"position\":"<<position
        <<",\"weight\":{\"name\":\""<<c.name<<"\"},\"up\":null,\"tokens\":8,\"layer\":"<<c.layer
        <<",\"max_keys\":"<<c.max_keys<<",\"scale\":0.0625,\"offset\":"<<offset
        <<",\"bytes\":"<<c.bytes()<<",\"pre_bytes\":"<<c.pre<<",\"output_bytes\":"<<c.output
        <<",\"state_bytes\":"<<c.state<<",\"tensors\":[";
    for(size_t i=0;i<c.tensors.size();++i){
      if(i)meta<<",";const auto &x=c.tensors[i];
      meta<<"{\"dtype\":"<<dtype_id(x.dtype)<<",\"shape\":[";
      for(int d=0;d<4;++d){if(d)meta<<",";meta<<(x.data?x.ne[d]:0);}
      meta<<"]}";
    }
    meta<<"]}\n";
    if(!data || !meta)throw std::runtime_error("capture disk write failed");
    offset+=c.bytes();++count;*c.seen()=0;
  }
  void write(Call &c) {
    const auto source_record=c.source();
    if(*c.seen()!=1)throw std::runtime_error("operator graph executed multiple times between flushes; corpus incomplete");
    if(c.bytes()>disk_limit-offset) throw std::runtime_error("capture byte budget exhausted; corpus incomplete");
    data.write(c.x(),c.bytes());
    meta<<"{\"id\":"<<source_record<<",\"source_record\":"<<source_record<<",\"request\":"<<request<<",\"phase\":"<<phase
        <<",\"position\":"<<position<<",\"weight\":"<<c.w.json()<<",\"up\":"
        <<(c.swiglu?c.up.json():"null")<<",\"tokens\":"<<c.tokens<<",\"ldy\":"<<c.ldy
        <<",\"epilogue\":"<<c.epilogue<<",\"offset\":"<<offset<<",\"x_bytes\":"<<c.nx
        <<",\"y_bytes\":"<<c.ny<<",\"bytes\":"<<c.bytes()<<"}\n";
    if(!data || !meta) throw std::runtime_error("capture disk write failed");
    offset+=c.bytes();++count;*c.seen()=0;
  }
};
bool lab_segment_should_run(const ignis_model &m,uint32_t kind,uint32_t layer,uint32_t tokens,uint32_t batch){
  if(!m.is_trellis_body || tokens!=8 || batch!=1 || m.hidden!=5120 || kind<1 || kind>3)return false;
  if(kind==IGNIS_SEGMENT_GQA && m.kv_format!=IGNIS_KV_FORMAT_HQ_E8_2B)return false;
  if(m.lab_trace)return m.lab_trace->selection.count(segment_name(layer,kind));
  return m.trellis_candidate && m.trellis_candidate->segments;
}
bool lab_segment_is_recording(const ignis_model &m){return bool(m.lab_trace);}
void *lab_segment_before(const ignis_model &m,uint32_t layer,const ignis_segment_profile &p,
    const std::vector<SegmentBinding> &bindings,cudaStream_t stream){
  SegmentCall *c=nullptr;
  if(m.lab_trace){
    auto &t=*m.lab_trace;
    if(!t.selection.count(segment_name(layer,p.kind)))return nullptr;
    auto make=[&](){auto x=std::make_unique<SegmentCall>();x->profile=p;x->layer=layer;
      x->name=segment_name(layer,p.kind);x->bindings=bindings;x->layout();return x;};
    cudaStreamCaptureStatus status;ck(cudaStreamIsCapturing(stream,&status));
    if(status!=cudaStreamCaptureStatusNone){
      const auto id=t.graph+":segment:"+std::to_string(p.kind)+":"+std::to_string(layer);
      if(t.mode==LabTrace::Discover)t.segments[id]=make();
      else {auto it=t.segments.find(id);if(it==t.segments.end())throw std::runtime_error("segment graph differs from discovery");
        c=it->second.get();auto next=make();
        if(next->bytes()!=c->bytes() || next->bindings.size()!=c->bindings.size())throw std::runtime_error("segment graph boundary changed");
        for(size_t i=0;i<bindings.size();++i){const auto &a=c->bindings[i],&b=bindings[i];
          if(a.name!=b.name || a.flags!=b.flags || a.tensor.dtype!=b.tensor.dtype ||
            !std::equal(a.tensor.ne,a.tensor.ne+4,b.tensor.ne) || bool(a.tensor.data)!=bool(b.tensor.data))
            throw std::runtime_error("segment graph binding identity changed");}
        c->bindings=bindings;
      }
    }else if(t.mode==LabTrace::Record){
      t.segment_eager=make();c=t.segment_eager.get();
      if(c->bytes()+24>t.pin_limit-t.pinned)throw std::runtime_error("segment pinned-memory budget exhausted");c->allocate();
    }
    if(c)t.begin(*c,stream);
    if(c)for(size_t i=0;i<bindings.size();++i){const auto &b=bindings[i];
      if(b.tensor.data && (b.flags&(SEGMENT_INPUT|SEGMENT_MUTABLE)))
        t.copy(*c,c->base()+c->pre_offsets[i],b.tensor.data,b.tensor.bytes(),1,b.tensor.bytes(),stream);}
  }
  return c;
}
void lab_segment_after(const ignis_model &m,void *ticket,cudaStream_t stream){
  if(!ticket)return;
  auto *c=static_cast<SegmentCall *>(ticket);auto &t=*m.lab_trace;
  const auto &bindings=c->bindings;
  for(size_t i=0;i<bindings.size();++i){const auto &b=bindings[i];
    if(b.tensor.data && (b.flags&(SEGMENT_MUTABLE|SEGMENT_OUTPUT)))
      t.copy(*c,c->base()+c->pre+c->post_offsets[i],b.tensor.data,b.tensor.bytes(),1,b.tensor.bytes(),stream);
  }
  t.finish(*c,stream);
  if(t.segment_eager.get()==c){ck(cudaStreamSynchronize(stream));if(*c->seen())t.write(*c);t.segment_eager.reset();}
}
void lab_segment_execute(const ignis_model &m,uint32_t layer,const ignis_segment_profile &p,
    const std::vector<SegmentBinding> &bindings,cudaStream_t stream){
  if(m.lab_trace)throw std::runtime_error("segment dispatch cannot replace the original recording chain");
  LabProfileRange profile_segment("op",p.kind==IGNIS_SEGMENT_MLP?"segment.mlp_block":
      p.kind==IGNIS_SEGMENT_GQA?"segment.gqa_block":"segment.gdn_block");
  segment_invoke(m,layer,p,bindings,nullptr,0,stream);
}

void *lab_gqa_before(const ignis_model &m,uint32_t layer,const ninfer::Tensor &q,
    const ninfer::Tensor &k,const ninfer::Tensor &v,const ninfer::Tensor &gate,
    const ninfer::Tensor &positions,const ninfer::Tensor &valid,const ninfer::Tensor &rows,
    const ninfer::PagedKVBatchLayerView &cache,uint32_t max_keys,cudaStream_t stream) {
  if(!m.lab_trace)return nullptr;
  auto &t=*m.lab_trace;const auto name="layer."+std::to_string(layer)+".attention";
  // Attention is opt-in; existing unfiltered Trellis scopes keep their meaning.
  if(!t.selection.count(name))return nullptr;
  if(!t.token_selection.count(static_cast<uint32_t>(q.ne[2])))return nullptr;
  // Graphs for unused pool widths are discovered too. Only the C1 graph is
  // instrumented; selecting attention explicitly documents this supported scope.
  if(q.ne[3]!=1)return nullptr;
  if(q.ne[2]!=8 || cache.dtype!=ninfer::DType::U8 || cache.quant_group!=32)
    throw std::runtime_error("GQA capture currently supports C1 T8 HQ attention only");
  cudaStreamCaptureStatus status;ck(cudaStreamIsCapturing(stream,&status));
  const bool graph=status!=cudaStreamCaptureStatusNone;
  auto make=[&](){
    auto c=std::make_unique<GqaCall>();c->name=name;c->layer=layer;c->max_keys=max_keys;
    for(const auto &x:{q,k,v,gate,positions,valid,rows,cache.block_tables,cache.k_pages,
        cache.v_pages,cache.k_scale_pages,cache.v_scale_pages,cache.residual_k,cache.residual_v,cache.ring_valid})c->add(x);
    c->state=c->pre-c->offsets[8];c->output=q.bytes();return c;
  };
  GqaCall *c;
  if(graph){
    const auto id=t.graph+":gqa:"+std::to_string(layer);
    if(t.mode==LabTrace::Discover){t.gqa_calls[id]=make();return nullptr;}
    auto it=t.gqa_calls.find(id);if(it==t.gqa_calls.end())throw std::runtime_error("GQA graph differs from discovered graph");
    c=it->second.get();
    auto candidate=make();
    if(candidate->pre!=c->pre || candidate->max_keys!=c->max_keys)
      throw std::runtime_error("GQA graph capture geometry changed");
    for(size_t i=0;i<c->tensors.size();++i){
      const auto &a=c->tensors[i],&b=candidate->tensors[i];
      if(a.dtype!=b.dtype || bool(a.data)!=bool(b.data) || !std::equal(a.ne,a.ne+4,b.ne))
        throw std::runtime_error("GQA graph tensor identity changed");
    }
    c->tensors=candidate->tensors;
  }else{
    if(t.mode!=LabTrace::Record)return nullptr;
    t.gqa_eager=make();c=t.gqa_eager.get();
    if(c->bytes()+24>t.pin_limit-t.pinned)throw std::runtime_error("GQA capture pinned-memory budget exhausted");
    c->allocate();
  }
  t.begin(*c,stream);
  for(size_t i=0;i<c->tensors.size();++i){const auto &x=c->tensors[i];
    if(x.data)t.copy(*c,c->base()+c->offsets[i],x.data,x.bytes(),1,x.bytes(),stream);}
  return c;
}
void lab_gqa_after(const ignis_model &m,void *ticket,const ninfer::Tensor &out,cudaStream_t stream){
  if(!ticket)return;auto &t=*m.lab_trace;auto &c=*static_cast<GqaCall *>(ticket);
  t.copy(c,c.base()+c.pre,out.data,c.output,1,c.output,stream);
  uint64_t offset=c.pre+c.output;
  for(size_t i=8;i<c.tensors.size();++i){const auto &x=c.tensors[i];
    if(x.data){t.copy(c,c.base()+offset,x.data,x.bytes(),1,x.bytes(),stream);offset+=x.bytes();}}
  t.finish(c,stream);
  if(t.gqa_eager.get()==&c){ck(cudaStreamSynchronize(stream));if(*c.seen())t.write(c);t.gqa_eager.reset();}
}

void lab_prefill_position(ignis_model *m,uint64_t p) {if(m->lab_trace)m->lab_trace->position=p;}
void lab_graph_begin(ignis_model *m,bool verify,uint32_t width) {
  if(m->lab_trace) {m->lab_trace->graph=std::string(verify?"verify":"decode")+std::to_string(width);m->lab_trace->ordinal=0;}
}
void *lab_before(const ignis_model &m,const ignis_trellis_weight &w,const ignis_trellis_weight *up,
                 const void *x,uint32_t tokens,void *y,uint32_t ldy,uint32_t epilogue,cudaStream_t stream) {
  if(!m.lab_trace) return nullptr;
  auto &t=*m.lab_trace;
  if(!t.token_selection.empty() && !t.token_selection.count(tokens))return nullptr;
  const auto identified=identify(t.named,w);
  if(!t.selection.empty() && !t.selection.count(identified.name) &&
     !(up && t.selection.count(identify(t.named,*up).name)))return nullptr;
  cudaStreamCaptureStatus status;ck(cudaStreamIsCapturing(stream,&status));
  const bool graph=status!=cudaStreamCaptureStatusNone;
  auto make=[&](){
    auto c=std::make_unique<Call>();c->w=identify(t.named,w);c->swiglu=up!=nullptr;
    if(up)c->up=identify(t.named,*up);
    c->tokens=tokens;c->ldy=ldy;c->epilogue=epilogue;
    c->nx=2ull*tokens*w.in;c->ny=2ull*tokens*w.out;return c;
  };
  Call *c;
  if(graph) {
    const auto id=t.graph+":"+std::to_string(t.ordinal++);
    if(t.mode==LabTrace::Discover) { t.calls[id]=make();return nullptr; }
    auto it=t.calls.find(id);if(it==t.calls.end()) throw std::runtime_error("record graph differs from discovered graph");
    c=it->second.get();
    if(c->tokens!=tokens || c->w.w.trellis!=w.trellis || c->ldy!=ldy || c->swiglu!=(up!=nullptr))
      throw std::runtime_error("record graph call identity changed");
  } else {
    if(t.mode!=LabTrace::Record)return nullptr;
    t.eager=make();c=t.eager.get();
    if(c->bytes()+24>t.pin_limit-t.pinned)throw std::runtime_error("capture pinned-memory budget exhausted");
    c->allocate();
  }
  t.begin(*c,stream);
  t.copy(*c,c->x(),x,c->nx,1,c->nx,stream);
  if(epilogue==IGNIS_TRELLIS_ADD && !up)
    t.copy(*c,c->initial(),y,w.out*2,tokens,ldy*2,stream);
  return c;
}
void lab_after(const ignis_model &m,void *ticket,const void *y,cudaStream_t stream) {
  if(!ticket)return;
  auto &t=*m.lab_trace;auto &c=*static_cast<Call *>(ticket);
  t.copy(c,c.output(),y,c.w.w.out*2,c.tokens,c.ldy*2,stream);
  t.finish(c,stream);
  if(t.eager.get()==&c) {ck(cudaStreamSynchronize(stream));if(*c.seen())t.write(c);t.eager.reset();}
}
}
extern "C" int32_t ignis_lab_export_directory(const char *directory){
  try{if(tensor_export){tensor_export->data.flush();tensor_export->metadata.flush();
      if(!tensor_export->data || !tensor_export->metadata)throw std::runtime_error("tensor export flush failed");}
    tensor_export.reset();if(directory)tensor_export=std::make_unique<TensorExport>(directory);return 0;}
  catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" const char *ignis_lab_trace_error(){return error.c_str();}
extern "C" const char *ignis_lab_gqa_implementation(ignis_model *m){
  return m && m->trellis_candidate && m->trellis_candidate->gqa ? m->trellis_candidate->gqa->identity : "builtin";
}
extern "C" int32_t ignis_lab_gqa_candidate_applied(ignis_model *m){
  return m && m->trellis_candidate && m->trellis_candidate->gqa ? 1 : 0;
}
extern "C" int32_t ignis_lab_profiler_start() {
  try {ck(cudaProfilerStart());return 0;}catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" int32_t ignis_lab_profiler_stop() {
  try {ck(cudaProfilerStop());return 0;}catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" int32_t ignis_lab_capture_start(ignis_model *m,ignis_seq_pool *p,const char *dir,uint64_t pin,uint64_t disk,const char *names) {
  return ignis_lab_capture_start_filtered(m,p,dir,pin,disk,names,nullptr,0);
}
extern "C" int32_t ignis_lab_capture_start_filtered(ignis_model *m,ignis_seq_pool *p,const char *dir,uint64_t pin,uint64_t disk,
    const char *names,const uint32_t *token_filter,uint32_t token_filter_count) {
  return ignis_lab_capture_start_selected(m,p,dir,pin,disk,names,token_filter,token_filter_count,0,0);
}
extern "C" int32_t ignis_lab_capture_start_selected(ignis_model *m,ignis_seq_pool *p,const char *dir,uint64_t pin,uint64_t disk,
    const char *names,const uint32_t *token_filter,uint32_t token_filter_count,uint64_t record_start,uint64_t record_limit) {
  try {
    if(!m || !p || !dir || !pin || !disk || m->lab_trace)throw std::runtime_error("invalid capture start");
    if(token_filter_count>8192 || (token_filter_count && !token_filter))throw std::runtime_error("invalid capture token filter");
    for(uint32_t i=0;i<token_filter_count;++i)
      if(!token_filter[i] || token_filter[i]>8192)throw std::runtime_error("invalid capture token filter");
    m->lab_trace=std::make_shared<ignis::LabTrace>(*m,dir,pin,disk,names,token_filter,token_filter_count,record_start,record_limit);
    if(ignis_model_reload_trellis(m,p,nullptr)!=0)throw std::runtime_error(ignis_trellis_reload_error());
    auto &t=*m->lab_trace;
    for(auto &[id,c]:t.calls) {
      if(c->bytes()+24>pin-t.pinned)throw std::runtime_error("capture graph exceeds pinned-memory budget");
      c->allocate();t.pinned+=c->bytes()+24;
    }
    for(auto &[id,c]:t.gqa_calls) {
      if(c->bytes()+24>pin-t.pinned)throw std::runtime_error("GQA capture graph exceeds pinned-memory budget");
      c->allocate();t.pinned+=c->bytes()+24;
    }
    for(auto &[id,c]:t.segments){
      if(c->bytes()+24>pin-t.pinned)throw std::runtime_error("segment graph exceeds pinned-memory budget");
      c->allocate();t.pinned+=c->bytes()+24;
    }
    t.mode=ignis::LabTrace::Record;
    if(ignis_model_reload_trellis(m,p,nullptr)!=0)throw std::runtime_error(ignis_trellis_reload_error());
    return 0;
  } catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" int32_t ignis_lab_trace_context(ignis_model *m,uint32_t request,uint32_t phase,uint64_t position) {
  if(!m)return -1;
  if(m->lab_trace){auto &t=*m->lab_trace;t.request=request;t.phase=phase;t.position=position;}
  return 0;
}
extern "C" int32_t ignis_lab_trace_flush(ignis_model *m) {
  try {
    if(m->lab_trace) {
      ck(cudaStreamSynchronize(m->stream));auto &t=*m->lab_trace;
      for(auto &[id,c]:t.calls)if(*c->seen())t.write(*c);
      for(auto &[id,c]:t.gqa_calls)if(*c->seen())t.write(*c);
      for(auto &[id,c]:t.segments)if(*c->seen())t.write(*c);
      t.data.flush();t.meta.flush();
      if(!t.data || !t.meta)throw std::runtime_error("capture flush failed");
    }return 0;
  } catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" int32_t ignis_lab_capture_stop(ignis_model *m,ignis_seq_pool *p) {
  try {
    // Keep capture buffers alive until all graphs referencing them are gone.
    auto saved=m->lab_trace;m->lab_trace.reset();
    if(ignis_model_reload_trellis(m,p,nullptr)!=0) {
      m->lab_trace=saved;throw std::runtime_error(ignis_trellis_reload_error());
    }
    return 0;
  } catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" int32_t ignis_lab_replay(ignis_model *m,const ignis_lab_weight_view *view,const ignis_lab_weight_view *upview,
    uint32_t tokens,uint32_t ldy,uint32_t epilogue,const void *x,const void *initial,const void *reference,
    uint32_t warmup,uint32_t samples,uint32_t check,float *times,uint64_t *mismatches,uint32_t *checked) {
  return ignis_lab_replay_diagnostics(m,view,upview,tokens,ldy,epilogue,x,initial,reference,
      warmup,samples,check,times,mismatches,checked,nullptr);
}
extern "C" int32_t ignis_lab_replay_diagnostics(ignis_model *m,const ignis_lab_weight_view *view,const ignis_lab_weight_view *upview,
    uint32_t tokens,uint32_t ldy,uint32_t epilogue,const void *x,const void *initial,const void *reference,
    uint32_t warmup,uint32_t samples,uint32_t check,float *times,uint64_t *mismatches,uint32_t *checked,
    ignis_lab_diagnostics *diagnostics) {
  cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;cudaEvent_t a=nullptr,b=nullptr;
  try {
    if(!m || !view || !x || !initial || !reference || !times || !mismatches || !checked || check>IGNIS_LAB_CHECK_EXPLORATION || !tokens ||
       tokens>std::max(m->prefill_chunk_tokens,uint32_t(64)) || !samples || samples>10000 || warmup>10000 || epilogue>1)
      throw std::runtime_error("invalid replay arguments");
    auto w=resolve(*m,*view);ignis_trellis_weight up{};if(upview)up=resolve(*m,*upview);
    if(ldy<w.out || ldy>IGNIS_TRELLIS_MAX_OUT)throw std::runtime_error("invalid replay stride");
    if(upview && (up.in!=w.in || up.out!=w.out))throw std::runtime_error("invalid swiglu pair");
    Device dx(2ull*tokens*w.in),dy(2ull*tokens*ldy),di(2ull*tokens*w.out);
    ck(cudaMemcpyAsync(dx.p,x,2ull*tokens*w.in,cudaMemcpyHostToDevice,m->stream));
    ck(cudaMemcpyAsync(di.p,initial,2ull*tokens*w.out,cudaMemcpyHostToDevice,m->stream));
    std::vector<uint16_t> result(uint64_t(tokens)*w.out);
    auto issue=[&](){
      const auto *api=m->trellis_candidate?m->trellis_candidate->api:nullptr;
      int rc=upview ? (api?api->swiglu:ignis_trellis_linear_swiglu)(&w,&up,dx.p,tokens,dy.p,ldy,&m->trellis_workspace,m->stream)
          : (api?api->linear:ignis_trellis_linear)(&w,dx.p,tokens,dy.p,ldy,epilogue,&m->trellis_workspace,m->stream);
      if(rc)throw std::runtime_error(api?api->last_error():ignis_trellis_last_error());
    };
    auto reset=[&](){
      ck(cudaMemsetAsync(dy.p,0,2ull*tokens*ldy,m->stream));
      ck(cudaMemcpy2DAsync(dy.p,ldy*2,di.p,w.out*2,w.out*2,tokens,cudaMemcpyDeviceToDevice,m->stream));
      ck(cudaMemsetAsync(m->trellis_workspace.base,0,m->trellis_workspace.bytes,m->stream));
      ck(cudaStreamSynchronize(m->stream));
    };
    reset();
    ck(cudaStreamBeginCapture(m->stream,cudaStreamCaptureModeThreadLocal));
    try {issue();} catch(...) {cudaStreamEndCapture(m->stream,&graph);throw;}
    ck(cudaStreamEndCapture(m->stream,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
    ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));*mismatches=0;*checked=0;
    ignis_lab_diagnostics d{};
    auto bf16=[](uint16_t bits) {uint32_t raw=uint32_t(bits)<<16;float value;std::memcpy(&value,&raw,4);return double(value);};
    for(uint32_t i=0;i<warmup+samples;++i) {
      reset();ck(cudaEventRecord(a,m->stream));ck(cudaGraphLaunch(exec,m->stream));
      ck(cudaEventRecord(b,m->stream));ck(cudaEventSynchronize(b));
      float ms;ck(cudaEventElapsedTime(&ms,a,b));if(i>=warmup)times[i-warmup]=ms;
      if(check==IGNIS_LAB_CHECK_FIRST && i!=0)continue;
      ++*checked;
      ck(cudaMemcpy2D(result.data(),w.out*2,dy.p,ldy*2,w.out*2,tokens,cudaMemcpyDeviceToHost));
      const auto *expected=static_cast<const uint16_t *>(reference);
      for(size_t j=0;j<result.size();++j) {
        const double actual=bf16(result[j]),reference_value=bf16(expected[j]);
        ++d.total_elements;
        if(result[j]!=expected[j] || !std::isfinite(actual))++d.mismatches;
        d.actual_nan+=std::isnan(actual);d.actual_inf+=std::isinf(actual);
        d.reference_nan+=std::isnan(reference_value);d.reference_inf+=std::isinf(reference_value);
        // Error norms describe only pairs where both values are finite.
        if(std::isfinite(actual) && std::isfinite(reference_value)) {
          ++d.finite_elements;const double delta=actual-reference_value;
          d.max_abs_error=std::max(d.max_abs_error,std::abs(delta));
          d.squared_error+=delta*delta;d.squared_reference+=reference_value*reference_value;
        }
      }
      *mismatches=d.mismatches;
      if(*mismatches && check!=IGNIS_LAB_CHECK_EXPLORATION)break;
    }
    if(tensor_export){
      ck(cudaMemcpy2D(result.data(),w.out*2,dy.p,ldy*2,w.out*2,tokens,cudaMemcpyDeviceToHost));
      tensor_export->tensor("output","candidate_output",result.data(),result.size()*2,"bf16",{w.out,tokens},false);
      const auto all=weights(*m);tensor_export->weight(view->name,all.at(view->name));
      if(upview && std::string(upview->name)!=view->name)tensor_export->weight(upview->name,all.at(upview->name));
    }
    if(diagnostics)*diagnostics=d;
    cudaEventDestroy(a);cudaEventDestroy(b);cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);return 0;
  } catch(const std::exception &e) {
    if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
    error=e.what();return -1;
  }
}

extern "C" int32_t ignis_lab_capture_minimum(ignis_model *m,const char *names,uint64_t tokens,uint64_t requests,uint64_t *bytes) {
  return ignis_lab_capture_minimum_filtered(m,names,tokens,requests,nullptr,0,bytes);
}
extern "C" int32_t ignis_lab_capture_minimum_filtered(ignis_model *m,const char *names,uint64_t tokens,uint64_t requests,
    const uint32_t *token_filter,uint32_t token_filter_count,uint64_t *bytes) {
  try {
    if(token_filter_count>8192 || (token_filter_count && !token_filter))throw std::runtime_error("invalid capture token filter");
    for(uint32_t i=0;i<token_filter_count;++i)
      if(!token_filter[i] || token_filter[i]>8192)throw std::runtime_error("invalid capture token filter");
    if(!m || !bytes || tokens>10000000 || requests>100000)throw std::runtime_error("invalid capture size query");
    auto all=weights(*m);std::set<std::string> selected;std::istringstream list(names?names:"");std::string name;
    while(std::getline(list,name,',')){
      bool attention=false;
      for(size_t i=0;i<m->layers.size();++i)
        if(m->layers[i].kind==IGNIS_LAYER_GQA && name=="layer."+std::to_string(i)+".attention")attention=true;
      if(!all.count(name) && !attention && !segment_selection(*m,name))throw std::runtime_error("unknown capture operator: "+name);
      if(attention && (m->kv_format!=IGNIS_KV_FORMAT_HQ_E8_2B || token_filter_count!=1 || token_filter[0]!=8))
        throw std::runtime_error("GQA recording currently requires hq-e8-2b and explicit --tokens-per-call 8");
      selected.insert(name);
    }
    uint64_t per_token=0,head=0;
    for(const auto &[id,w]:all) {
      if(id.ends_with(".up"))continue;
      const auto up=id.ends_with(".gate")?id.substr(0,id.size()-5)+".up":"";
      if(!selected.empty() && !selected.count(id) && !selected.count(up))continue;
      const uint64_t size=2ull*w.in+4ull*w.out;
      if(id=="head")head=size;else per_token+=size;
    }
    // Aggregate prompt lengths do not identify chunk sizes. With a token
    // filter the body can contribute zero; the one-row prefill head is known.
    if(token_filter_count) {
      per_token=0;
      if(std::find(token_filter,token_filter+token_filter_count,1)==token_filter+token_filter_count)head=0;
    }
    *bytes=per_token*tokens+head*requests;return 0;
  }catch(const std::exception &e){error=e.what();return -1;}
}

extern "C" int32_t ignis_lab_gqa_replay(ignis_model *m,const ignis_lab_gqa_view *view,
    const void *blob,uint64_t bytes,uint32_t warmup,uint32_t samples,uint32_t check,
    float *times,uint32_t *checked,ignis_lab_diagnostics *diagnostics,
    uint64_t *state_mismatches,uint64_t *state_bytes) {
  cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;cudaEvent_t a=nullptr,b=nullptr;
  try {
    if(!m || !view || !blob || !times || !checked || !diagnostics || !state_mismatches || !state_bytes ||
       view->version!=1 || !samples || samples>10000 || warmup>10000 || check>IGNIS_LAB_CHECK_EXPLORATION ||
       view->layer>=m->layers.size() || m->layers[view->layer].kind!=IGNIS_LAYER_GQA ||
       !view->max_keys || view->max_keys>m->max_context_tokens || view->scale!=0.0625f || bytes>(1ull<<31))
      throw std::runtime_error("invalid GQA replay arguments");
    std::vector<ninfer::Tensor> tensors;std::vector<uint64_t> offsets;
    uint64_t pre=0;
    for(size_t i=0;i<15;++i){
      const auto &v=view->tensors[i];const uint32_t expected=i<4 || i==12 || i==13 ? 0 : i>=8 && i<=11 ? 1 : 2;
      if(v.dtype!=expected && !(i==5 && v.ne[0]==0))throw std::runtime_error("invalid GQA tensor dtype");
      const auto dtype=v.dtype==0?ninfer::DType::BF16:v.dtype==1?ninfer::DType::U8:ninfer::DType::I32;
      bool empty=true;uint64_t elements=1;
      for(int dim=0;dim<4;++dim){
        if(v.ne[dim]!=0)empty=false;
        if(v.ne[dim]<0 || v.ne[dim]>1048576)throw std::runtime_error("invalid GQA tensor extent");
        elements*=uint64_t(v.ne[dim]);
        if(elements>(1ull<<30))throw std::runtime_error("GQA tensor too large");
      }
      if(empty){if(i!=5)throw std::runtime_error("missing GQA tensor");tensors.emplace_back();}
      else {
        if(!elements)throw std::runtime_error("invalid partially empty GQA tensor");
        // Temporarily bind the host blob to compute contiguous byte sizes.
        tensors.emplace_back(const_cast<char *>(static_cast<const char *>(blob))+pre,dtype,
            std::initializer_list<int32_t>{v.ne[0],v.ne[1],v.ne[2],v.ne[3]});
      }
      offsets.push_back(pre);if(!empty)pre+=tensors.back().bytes();
      if(pre>bytes)throw std::runtime_error("truncated GQA pre-state");
    }
    auto shape=[&](size_t i,int32_t d0,int32_t d1,int32_t d2,int32_t d3){
      const auto &x=tensors[i];if(x.ne[0]!=d0 || x.ne[1]!=d1 || x.ne[2]!=d2 || x.ne[3]!=d3)
        throw std::runtime_error("unsupported GQA replay geometry");
    };
    shape(0,256,kIgnisGqaQHeads,8,1);shape(1,256,4,8,1);shape(2,256,4,8,1);shape(3,256,kIgnisGqaQHeads,8,1);
    shape(4,8,1,1,1);if(tensors[5].data)shape(5,1,1,1,1);shape(6,1,1,1,1);
    const int32_t logical=tensors[7].ne[0],slots=tensors[7].ne[1],pages=tensors[8].ne[3];
    shape(7,logical,slots,1,1);
    shape(8,64,64,4,pages);shape(9,64,64,4,pages);shape(10,8,64,4,pages);shape(11,8,64,4,pages);
    shape(12,256,4,544,slots);shape(13,256,4,544,slots);shape(14,16,slots,1,1);
    const uint64_t output=tensors[0].bytes(),state=pre-offsets[8];
    if(pre+output+state!=bytes)throw std::runtime_error("GQA replay blob size mismatch");
    // Metadata is untrusted file content. Check every address the kernel may
    // form before uploading anything; unused table entries may be unallocated.
    auto read_i32=[&](size_t tensor,size_t index){int32_t result;
      std::memcpy(&result,static_cast<const char *>(blob)+offsets[tensor]+index*4,4);return result;};
    const int32_t row=read_i32(6,0),valid=tensors[5].data?read_i32(5,0):8;
    if(row<0 || row>=slots || valid<0 || valid>8)throw std::runtime_error("invalid GQA slot/mask metadata");
    int32_t previous=-1;
    for(int32_t j=0;j<8;++j){
      const int32_t pos=read_i32(4,j);
      if(pos<0 || uint32_t(pos)>=view->max_keys || (j<valid && j>0 && pos!=previous+1) ||
          (j>=valid && pos!=(valid?previous:0)))throw std::runtime_error("invalid GQA position metadata");
      if(j<valid)previous=pos;
    }
    if(valid){
      const int32_t last=read_i32(4,valid-1)/64;
      if(last>=logical)throw std::runtime_error("GQA positions exceed table capacity");
      for(int32_t j=0;j<=last;++j){const int32_t page=read_i32(7,uint64_t(row)*logical+j);
        if(page<0 || page>=pages)throw std::runtime_error("GQA table references invalid physical page");}
    }
    std::vector<std::unique_ptr<Device>> storage;
    for(size_t i=0;i<tensors.size();++i){
      storage.push_back(std::make_unique<Device>(std::max<uint64_t>(tensors[i].data?tensors[i].bytes():0,1)));
      if(tensors[i].data)tensors[i].data=storage.back()->p;
    }
    Device out_device(output);auto out=tensors[0];out.data=out_device.p;
    ninfer::PagedKVBatchLayerView cache;
    cache.block_tables=tensors[7];cache.k_pages=tensors[8];cache.v_pages=tensors[9];
    cache.k_scale_pages=tensors[10];cache.v_scale_pages=tensors[11];
    cache.residual_k=tensors[12];cache.residual_v=tensors[13];cache.ring_valid=tensors[14];
    cache.head_dim=256;cache.num_kv_heads=4;cache.dtype=ninfer::DType::U8;cache.quant_group=32;
    const ninfer::ops::GqaExecutionEnvelope envelope{1,view->max_keys};
    const auto workspace_bytes=ignis::gqa_candidate_applies(*m,tensors[0],cache)
        ? ignis::gqa_candidate_workspace_bytes(*m,envelope.max_visible_keys)
        : std::max<uint64_t>(ignis::gqa_native_attention_applies(*m,tensors[0],cache)
              ?ignis_gqa_native_workspace(envelope.max_visible_keys):0,
            ninfer::ops::gqa_attention_workspace_capacity_bytes(kIgnisGqaQHeads,cache.dtype,envelope,1,8,8));
    Device workspace(std::max<size_t>(workspace_bytes,1));
    // Immutable inputs and a pristine state image stay resident throughout
    // repetitions; only isolated mutable planes are restored between launches.
    Device pristine(state);
    ck(cudaMemcpyAsync(pristine.p,static_cast<const char *>(blob)+offsets[8],state,cudaMemcpyHostToDevice,m->stream));
    for(size_t i=0;i<8;++i)if(tensors[i].data)
      ck(cudaMemcpyAsync(tensors[i].data,static_cast<const char *>(blob)+offsets[i],tensors[i].bytes(),cudaMemcpyHostToDevice,m->stream));
    auto reset=[&](){
      for(size_t i=8;i<tensors.size();++i)
        ck(cudaMemcpyAsync(tensors[i].data,static_cast<char *>(pristine.p)+offsets[i]-offsets[8],tensors[i].bytes(),cudaMemcpyDeviceToDevice,m->stream));
      ck(cudaMemsetAsync(out.data,0,output,m->stream));
      ck(cudaMemsetAsync(workspace.p,0,std::max<size_t>(workspace_bytes,1),m->stream));
      ck(cudaStreamSynchronize(m->stream));
    };
    auto issue=[&](){
      ninfer::DeviceArena arena(ninfer::DeviceSpan{workspace.p,std::max<size_t>(workspace_bytes,1)});
      ignis::gqa_attention_dispatch(*m,tensors[0],tensors[1],tensors[2],tensors[4],tensors[5],tensors[6],tensors[3],
          view->scale,cache,envelope,arena,out,m->stream,workspace.p,workspace_bytes);
    };
    reset();ck(cudaStreamBeginCapture(m->stream,cudaStreamCaptureModeThreadLocal));
    try{issue();}catch(...){cudaStreamEndCapture(m->stream,&graph);throw;}
    ck(cudaStreamEndCapture(m->stream,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
    ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));*checked=0;*state_mismatches=0;*state_bytes=0;
    ignis_lab_diagnostics d{};std::vector<uint16_t> result(output/2);std::vector<char> actual_state(state);
    auto bf16=[](uint16_t bits){uint32_t raw=uint32_t(bits)<<16;float value;std::memcpy(&value,&raw,4);return double(value);};
    for(uint32_t i=0;i<warmup+samples;++i){
      reset();ck(cudaEventRecord(a,m->stream));ck(cudaGraphLaunch(exec,m->stream));ck(cudaEventRecord(b,m->stream));
      ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));if(i>=warmup)times[i-warmup]=ms;
      if(check==IGNIS_LAB_CHECK_FIRST && i)continue;
      ++*checked;ck(cudaMemcpy(result.data(),out.data,output,cudaMemcpyDeviceToHost));
      for(size_t j=0;j<result.size();++j){uint16_t expected;std::memcpy(&expected,static_cast<const char *>(blob)+pre+j*2,2);
        const double actual=bf16(result[j]),reference=bf16(expected);++d.total_elements;
        if(result[j]!=expected || !std::isfinite(actual))++d.mismatches;
        d.actual_nan+=std::isnan(actual);d.actual_inf+=std::isinf(actual);
        d.reference_nan+=std::isnan(reference);d.reference_inf+=std::isinf(reference);
        if(std::isfinite(actual) && std::isfinite(reference)){++d.finite_elements;double delta=actual-reference;
          d.max_abs_error=std::max(d.max_abs_error,std::abs(delta));d.squared_error+=delta*delta;d.squared_reference+=reference*reference;}
      }
      uint64_t offset=0;
      for(size_t j=8;j<tensors.size();++j){ck(cudaMemcpy(actual_state.data()+offset,tensors[j].data,tensors[j].bytes(),cudaMemcpyDeviceToHost));offset+=tensors[j].bytes();}
      const char *reference=static_cast<const char *>(blob)+pre+output;
      for(size_t j=0;j<state;++j)*state_mismatches+=actual_state[j]!=reference[j];
      *state_bytes+=state;
      if((d.mismatches || *state_mismatches) && check!=IGNIS_LAB_CHECK_EXPLORATION)break;
    }
    if(tensor_export){
      tensor_export->tensor("output","candidate_output",out);
      static constexpr const char *names[]={"k_pages","v_pages","k_meta","v_meta","residual_k","residual_v","ring"};
      for(size_t i=8;i<tensors.size();++i)tensor_export->tensor(names[i-8],"candidate_state",tensors[i]);
    }
    *diagnostics=d;
    cudaEventDestroy(a);cudaEventDestroy(b);cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);return 0;
  }catch(const std::exception &e){
    if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
    error=e.what();return -1;
  }
}

extern "C" const char *ignis_lab_segment_implementation(ignis_model *m){
  return m && m->trellis_candidate && m->trellis_candidate->segments?m->trellis_candidate->segments->identity:"builtin";
}
extern "C" uint64_t ignis_lab_segment_workspace_capacity(ignis_model *m,uint32_t kind){
  if(!m || kind<1 || kind>3)return 0;
  const auto *set=m->trellis_candidate && m->trellis_candidate->segments?&m->trellis_candidate->segment_workspaces:m->segment_builtin_workspaces.get();
  return set && set->buffers[kind-1]?set->buffers[kind-1]->bytes:0;
}
extern "C" int32_t ignis_lab_segment_replay(ignis_model *m,uint32_t layer,const ignis_segment_profile *p,
    const ignis_segment_tensor_view *views,uint32_t count,const void *blob,uint64_t bytes,
    uint32_t warmup,uint32_t samples,uint32_t check,float *times,uint32_t *checked,
    ignis_lab_diagnostics *diagnostics,ignis_lab_diagnostics *per_binding,uint64_t *state_mismatches,uint64_t *state_bytes,
    uint64_t *binding_state_mismatches,uint64_t *binding_state_bytes){
  cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;cudaEvent_t a=nullptr,b=nullptr;
  try{
    if(!m || !p || !views || !blob || !times || !checked || !diagnostics || !state_mismatches || !state_bytes ||
       layer>=m->layers.size() || p->kind<1 || p->kind>3 || p->tokens!=8 || p->batch!=1 || p->hidden!=m->hidden ||
       p->flags || p->rms_eps!=m->rms_norm_eps || count>32 || !count || bytes>(1ull<<31) ||
       !samples || samples>10000 || warmup>10000 || check>IGNIS_LAB_CHECK_EXPLORATION)
      throw std::runtime_error("invalid finite segment replay arguments");
    SegmentCall c;c.profile=*p;c.layer=layer;
    std::set<std::string> names;
    for(uint32_t i=0;i<count;++i){const auto &v=views[i];
      if(!v.name || !*v.name || std::strlen(v.name)>64 || !names.insert(v.name).second ||
         !v.flags || v.flags>7 || v.dtype>3 || v.present>1)
        throw std::runtime_error("invalid segment binding descriptor");
      uint64_t elements=1;
      for(int d=0;d<4;++d){if(v.ne[d]<=0 || v.ne[d]>1048576)throw std::runtime_error("invalid segment extent");
        elements*=uint64_t(v.ne[d]);if(elements>(1ull<<29))throw std::runtime_error("segment binding too large");}
      auto dtype=v.dtype==0?ninfer::DType::BF16:v.dtype==1?ninfer::DType::U8:v.dtype==2?ninfer::DType::I32:ninfer::DType::FP32;
      ninfer::Tensor t(v.present?const_cast<void *>(blob):nullptr,dtype,{v.ne[0],v.ne[1],v.ne[2],v.ne[3]});
      c.bindings.push_back({v.name,t,v.flags});
    }
    c.layout();if(c.bytes()!=bytes)throw std::runtime_error("segment blob size mismatch");
    // Check file metadata before a graph can form any device address.
    auto index=[&](const char *name)->size_t{
      for(size_t i=0;i<c.bindings.size();++i)if(c.bindings[i].name==name)return i;
      throw std::runtime_error(std::string("missing segment metadata binding: ")+name);
    };
    auto i32=[&](size_t i,size_t j){const auto &t=c.bindings.at(i).tensor;
      if(t.dtype!=ninfer::DType::I32 || !t.data || j>=uint64_t(t.numel()) || !(c.bindings[i].flags&ignis::SEGMENT_INPUT))
        throw std::runtime_error("invalid segment index tensor");
      int32_t v;std::memcpy(&v,static_cast<const char *>(blob)+c.pre_offsets[i]+j*4,4);return v;
    };
    int32_t gdn_valid=8;
    if(p->kind==IGNIS_SEGMENT_GQA){
      const auto rows=index("table_rows"),valid=index("valid_columns"),positions=index("positions"),rope=index("rope_positions"),tables=index("block_tables"),pages=index("k_pages");
      const int32_t slot=i32(rows,0),n=c.bindings[valid].tensor.data?i32(valid,0):8;
      if(slot<0 || uint32_t(slot)>=p->slots || n<0 || n>8)throw std::runtime_error("invalid segment GQA slot/mask");
      int32_t last=-1;
      for(int j=0;j<8;++j){const int32_t position=i32(positions,j);
        // RoPE positions are signed analytic angles, not KV addresses.
        (void)i32(rope,j);
        if(position<0 || uint32_t(position)>=p->max_keys ||
            (j<n && j && position!=last+1) || (j>=n && position!=(n?last:0)))
          throw std::runtime_error("invalid segment GQA positions");if(j<n)last=position;}
      const int32_t logical=c.bindings[tables].tensor.ne[0],physical=c.bindings[pages].tensor.ne[3];
      if(n){if(last/64>=logical)throw std::runtime_error("segment GQA history exceeds table");
        for(int j=0;j<=last/64;++j){const int32_t page=i32(tables,uint64_t(slot)*logical+j);
          if(page<0 || page>=physical)throw std::runtime_error("segment GQA invalid physical page");}}
    }else if(p->kind==IGNIS_SEGMENT_GDN){
      const int32_t slot=i32(index("lane_slots"),0),n=i32(index("valid_columns"),0);
      if(slot<0 || uint32_t(slot)>=p->slots || n<1 || n>8)throw std::runtime_error("invalid segment GDN slot/mask");
      gdn_valid=n;
    }
    std::vector<std::unique_ptr<Device>> buffers;
    for(size_t i=0;i<c.bindings.size();++i){auto &t=c.bindings[i].tensor;
      buffers.push_back(std::make_unique<Device>(t.data?t.bytes():1));if(t.data)t.data=buffers.back()->p;}
    Device pristine(std::max<uint64_t>(c.pre,1)),workspace(ignis::segment_workspace_bytes(*m,*p));
    ck(cudaMemcpyAsync(pristine.p,blob,c.pre,cudaMemcpyHostToDevice,m->stream));
    const uint64_t workspace_bytes=ignis::segment_workspace_bytes(*m,*p);
    auto reset=[&](){
      for(size_t i=0;i<c.bindings.size();++i){auto &x=c.bindings[i];if(!x.tensor.data)continue;
        if(x.flags&(ignis::SEGMENT_INPUT|ignis::SEGMENT_MUTABLE))
          ck(cudaMemcpyAsync(x.tensor.data,static_cast<char *>(pristine.p)+c.pre_offsets[i],x.tensor.bytes(),cudaMemcpyDeviceToDevice,m->stream));
        else ck(cudaMemsetAsync(x.tensor.data,0,x.tensor.bytes(),m->stream));}
      ck(cudaMemsetAsync(workspace.p,0,workspace_bytes,m->stream));ck(cudaStreamSynchronize(m->stream));
    };
    reset();ck(cudaStreamBeginCapture(m->stream,cudaStreamCaptureModeThreadLocal));
    try{ignis::segment_invoke(*m,layer,*p,c.bindings,workspace.p,workspace_bytes,m->stream);}
    catch(...){cudaStreamEndCapture(m->stream,&graph);throw;}
    ck(cudaStreamEndCapture(m->stream,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
    ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));*checked=0;*state_mismatches=0;*state_bytes=0;
    ignis_lab_diagnostics d{};if(per_binding)std::fill(per_binding,per_binding+count,ignis_lab_diagnostics{});
    if(binding_state_mismatches)std::fill(binding_state_mismatches,binding_state_mismatches+count,0);
    if(binding_state_bytes)std::fill(binding_state_bytes,binding_state_bytes+count,0);
    auto value=[](const char *base,uint32_t dtype,size_t j){float v;
      if(dtype==0){uint16_t b;std::memcpy(&b,base+j*2,2);uint32_t raw=uint32_t(b)<<16;std::memcpy(&v,&raw,4);}
      else std::memcpy(&v,base+j*4,4);return double(v);};
    for(uint32_t round=0;round<warmup+samples;++round){
      reset();ck(cudaEventRecord(a,m->stream));ck(cudaGraphLaunch(exec,m->stream));ck(cudaEventRecord(b,m->stream));ck(cudaEventSynchronize(b));
      float ms;ck(cudaEventElapsedTime(&ms,a,b));if(round>=warmup)times[round-warmup]=ms;
      if(check==IGNIS_LAB_CHECK_FIRST && round)continue;++*checked;
      for(size_t i=0;i<c.bindings.size();++i){const auto &x=c.bindings[i];if(!x.tensor.data)continue;
        std::vector<char> actual(x.tensor.bytes());ck(cudaMemcpy(actual.data(),x.tensor.data,actual.size(),cudaMemcpyDeviceToHost));
        const bool numeric=(x.flags&ignis::SEGMENT_OUTPUT) && (views[i].dtype==0 || views[i].dtype==3);
        const char *expected=static_cast<const char *>(blob)+((x.flags&(ignis::SEGMENT_MUTABLE|ignis::SEGMENT_OUTPUT))?c.pre+c.post_offsets[i]:c.pre_offsets[i]);
        if(numeric){const size_t width=views[i].dtype==0?2:4;
          for(size_t j=0;j<actual.size()/width;++j){
            const bool suffix=p->kind==IGNIS_SEGMENT_GDN && (x.name=="key_record" || x.name=="value_record" || x.name=="gate_record") &&
                j/(uint64_t(x.tensor.ne[0])*x.tensor.ne[1])>=uint32_t(gdn_valid);
            if(suffix){const char *initial=static_cast<const char *>(blob)+c.pre_offsets[i];
              for(size_t byte=0;byte<width;++byte){const bool mismatch=actual[j*width+byte]!=initial[j*width+byte];
                *state_mismatches+=mismatch;if(binding_state_mismatches)binding_state_mismatches[i]+=mismatch;}
              *state_bytes+=width;if(binding_state_bytes)binding_state_bytes[i]+=width;continue;}
            const double av=value(actual.data(),views[i].dtype,j),rv=value(expected,views[i].dtype,j);
            auto accumulate=[&](ignis_lab_diagnostics &out){
              ++out.total_elements;if(std::memcmp(actual.data()+j*width,expected+j*width,width) || !std::isfinite(av))++out.mismatches;
              out.actual_nan+=std::isnan(av);out.actual_inf+=std::isinf(av);out.reference_nan+=std::isnan(rv);out.reference_inf+=std::isinf(rv);
              if(std::isfinite(av)&&std::isfinite(rv)){++out.finite_elements;const double delta=av-rv;
                out.max_abs_error=std::max(out.max_abs_error,std::abs(delta));out.squared_error+=delta*delta;out.squared_reference+=rv*rv;}
            };
            accumulate(d);if(per_binding)accumulate(per_binding[i]);
          }
        }else{
          // INPUT bindings are immutable and must equal their captured preimage.
          if(x.flags&ignis::SEGMENT_INPUT)expected=static_cast<const char *>(blob)+c.pre_offsets[i];
          for(size_t j=0;j<actual.size();++j){const bool mismatch=actual[j]!=expected[j];*state_mismatches+=mismatch;
            if(binding_state_mismatches)binding_state_mismatches[i]+=mismatch;}
          *state_bytes+=actual.size();if(binding_state_bytes)binding_state_bytes[i]+=actual.size();
        }
      }
      if((d.mismatches || *state_mismatches) && check!=IGNIS_LAB_CHECK_EXPLORATION)break;
    }
    if(tensor_export){
      for(const auto &binding:c.bindings)if(binding.flags&(ignis::SEGMENT_OUTPUT|ignis::SEGMENT_MUTABLE))
        tensor_export->tensor(binding.name,(binding.flags&ignis::SEGMENT_OUTPUT)?"candidate_output":"candidate_state",binding.tensor);
      export_layer_weights(*m,layer,p->kind);
      ignis_lab_segment_tensor tensors[32]{};
      ignis_lab_segment_tensors_fn get=nullptr;
      if(m->trellis_candidate && m->trellis_candidate->segments){
        const char *symbol=p->kind==1?"ignis_lab_mlp_segment_tensors_v1":p->kind==2?"ignis_lab_gqa_segment_tensors_v1":"ignis_lab_gdn_segment_tensors_v1";
        get=reinterpret_cast<ignis_lab_segment_tensors_fn>(dlsym(m->trellis_candidate->library,symbol));
      }else get=p->kind==1?ignis_lab_mlp_segment_tensors_v1:p->kind==2?ignis_lab_gqa_segment_tensors_v1:ignis_lab_gdn_segment_tensors_v1;
      const int32_t n=get?get(p,workspace.p,workspace_bytes,tensors,32):0;
      if(n<0 || n>32)throw std::runtime_error("invalid candidate diagnostic tensor inventory");
      tensor_export->metadata<<"{\"intermediate_taps\":{\"available\":"<<(n?"true":"false")
        <<",\"reason\":\""<<(n?"candidate-owned retained workspace":"active implementation has no retained diagnostic tensor provider")<<"\"}}\n";
      for(int32_t i=0;i<n;++i){const auto &t=tensors[i];
        if(!t.name || !t.data || t.dtype>3 || !t.bytes)throw std::runtime_error("invalid diagnostic tensor descriptor");
        const auto begin=reinterpret_cast<uintptr_t>(workspace.p),address=reinterpret_cast<uintptr_t>(t.data);
        if(address<begin || t.bytes>workspace_bytes || address-begin>workspace_bytes-t.bytes)
          throw std::runtime_error("diagnostic tensor exceeds selected candidate workspace");
        std::vector<int64_t> shape(t.ne,t.ne+4);uint64_t elements=1;
        for(auto dim:shape){if(dim<=0 || uint64_t(dim)>(1ull<<30)/elements)throw std::runtime_error("invalid diagnostic tensor shape");elements*=dim;}
        const uint32_t width=t.dtype==0?2:t.dtype==1?1:4;
        if(elements*width!=t.bytes)throw std::runtime_error("diagnostic tensor byte count mismatch");
        tensor_export->tensor(t.name,"candidate_intermediate",t.data,t.bytes,t.dtype==0?"bf16":t.dtype==1?"u8":t.dtype==2?"i32":"f32",shape);
      }
    }
    *diagnostics=d;cudaEventDestroy(a);cudaEventDestroy(b);cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);return 0;
  }catch(const std::exception &e){if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
    error=e.what();return -1;}
}
