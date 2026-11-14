// Offline, model-owned capture/replay. No serving route enables it.
#include "lab_trace.h"
#include "model_internal.h"
#include "ignis_trellis_candidate.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

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
struct Call {
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
  void allocate(){ck(cudaHostAlloc(&host,bytes()+4,cudaHostAllocDefault));std::memset(host,0,bytes()+4);}
};
}
namespace ignis {
struct LabTrace {
  enum Mode {Discover,Record} mode=Discover;
  std::map<std::string,std::unique_ptr<Call>> calls;
  std::map<std::string,ignis_trellis_weight> named;
  std::unique_ptr<Call> eager;
  std::set<std::string> selection;
  std::string graph;
  uint32_t ordinal=0,request=0,phase=0;
  uint64_t position=0,offset=0,count=0,pinned=0,pin_limit=0,disk_limit=0;
  std::ofstream data,meta;
  Device one{4};
  LabTrace(const ignis_model &m,const char *directory,uint64_t pin,uint64_t disk,const char *names):named(weights(m)),pin_limit(pin),disk_limit(disk) {
    std::istringstream list(names ? names : "");std::string name;
    while(std::getline(list,name,',')) {
      if(!named.count(name))throw std::runtime_error("unknown capture weight: "+name);
      selection.insert(name);
    }
    data.open(std::string(directory)+"/data.bin",std::ios::binary|std::ios::trunc);
    meta.open(std::string(directory)+"/records.jsonl",std::ios::trunc);
    if(!data || !meta) throw std::runtime_error("cannot create capture files");
    uint32_t value=1;ck(cudaMemcpy(one.p,&value,4,cudaMemcpyHostToDevice));
  }
  void write(Call &c) {
    if(c.bytes()>disk_limit-offset) throw std::runtime_error("capture byte budget exhausted; corpus incomplete");
    data.write(c.x(),c.bytes());
    meta<<"{\"id\":"<<count++<<",\"request\":"<<request<<",\"phase\":"<<phase
        <<",\"position\":"<<position<<",\"weight\":"<<c.w.json()<<",\"up\":"
        <<(c.swiglu?c.up.json():"null")<<",\"tokens\":"<<c.tokens<<",\"ldy\":"<<c.ldy
        <<",\"epilogue\":"<<c.epilogue<<",\"offset\":"<<offset<<",\"x_bytes\":"<<c.nx
        <<",\"y_bytes\":"<<c.ny<<",\"bytes\":"<<c.bytes()<<"}\n";
    if(!data || !meta) throw std::runtime_error("capture disk write failed");
    offset+=c.bytes();*c.seen()=0;
  }
};
void lab_prefill_position(ignis_model *m,uint64_t p) {if(m->lab_trace)m->lab_trace->position=p;}
void lab_graph_begin(ignis_model *m,bool verify,uint32_t width) {
  if(m->lab_trace) {m->lab_trace->graph=std::string(verify?"verify":"decode")+std::to_string(width);m->lab_trace->ordinal=0;}
}
void *lab_before(const ignis_model &m,const ignis_trellis_weight &w,const ignis_trellis_weight *up,
                 const void *x,uint32_t tokens,void *y,uint32_t ldy,uint32_t epilogue,cudaStream_t stream) {
  if(!m.lab_trace) return nullptr;
  auto &t=*m.lab_trace;
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
    if(c->bytes()+4>t.pin_limit-t.pinned)throw std::runtime_error("capture pinned-memory budget exhausted");
    c->allocate();
  }
  ck(cudaMemcpyAsync(c->x(),x,c->nx,cudaMemcpyDeviceToHost,stream));
  if(epilogue==IGNIS_TRELLIS_ADD && !up)
    ck(cudaMemcpy2DAsync(c->initial(),w.out*2,y,ldy*2,w.out*2,tokens,cudaMemcpyDeviceToHost,stream));
  return c;
}
void lab_after(const ignis_model &m,void *ticket,const void *y,cudaStream_t stream) {
  if(!ticket)return;
  auto &t=*m.lab_trace;auto &c=*static_cast<Call *>(ticket);
  ck(cudaMemcpy2DAsync(c.output(),c.w.w.out*2,y,c.ldy*2,c.w.w.out*2,c.tokens,cudaMemcpyDeviceToHost,stream));
  ck(cudaMemcpyAsync(c.seen(),t.one.p,4,cudaMemcpyDeviceToHost,stream));
  if(t.eager.get()==&c) {ck(cudaStreamSynchronize(stream));t.write(c);t.eager.reset();}
}
}
extern "C" const char *ignis_lab_trace_error(){return error.c_str();}
extern "C" int32_t ignis_lab_capture_start(ignis_model *m,ignis_seq_pool *p,const char *dir,uint64_t pin,uint64_t disk,const char *names) {
  try {
    if(!m || !p || !dir || !pin || !disk || m->lab_trace)throw std::runtime_error("invalid capture start");
    m->lab_trace=std::make_shared<ignis::LabTrace>(*m,dir,pin,disk,names);
    if(ignis_model_reload_trellis(m,p,nullptr)!=0)throw std::runtime_error(ignis_trellis_reload_error());
    auto &t=*m->lab_trace;
    for(auto &[id,c]:t.calls) {
      if(c->bytes()+4>pin-t.pinned)throw std::runtime_error("capture graph exceeds pinned-memory budget");
      c->allocate();t.pinned+=c->bytes()+4;
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
  cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;cudaEvent_t a=nullptr,b=nullptr;
  try {
    if(!m || !view || !x || !initial || !reference || !times || !mismatches || !checked || check>IGNIS_LAB_CHECK_EVERY || !tokens ||
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
    for(uint32_t i=0;i<warmup+samples;++i) {
      reset();ck(cudaEventRecord(a,m->stream));ck(cudaGraphLaunch(exec,m->stream));
      ck(cudaEventRecord(b,m->stream));ck(cudaEventSynchronize(b));
      float ms;ck(cudaEventElapsedTime(&ms,a,b));if(i>=warmup)times[i-warmup]=ms;
      if(check==IGNIS_LAB_CHECK_FIRST && i!=0)continue;
      ++*checked;
      ck(cudaMemcpy2D(result.data(),w.out*2,dy.p,ldy*2,w.out*2,tokens,cudaMemcpyDeviceToHost));
      const auto *expected=static_cast<const uint16_t *>(reference);
      for(size_t j=0;j<result.size();++j) if(result[j]!=expected[j] || (result[j]&0x7f80)==0x7f80)++*mismatches;
      if(*mismatches)break;
    }
    cudaEventDestroy(a);cudaEventDestroy(b);cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);return 0;
  } catch(const std::exception &e) {
    if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
    error=e.what();return -1;
  }
}

extern "C" int32_t ignis_lab_capture_minimum(ignis_model *m,const char *names,uint64_t tokens,uint64_t requests,uint64_t *bytes) {
  try {
    if(!m || !bytes || tokens>10000000 || requests>100000)throw std::runtime_error("invalid capture size query");
    auto all=weights(*m);std::set<std::string> selected;std::istringstream list(names?names:"");std::string name;
    while(std::getline(list,name,',')){if(!all.count(name))throw std::runtime_error("unknown capture weight: "+name);selected.insert(name);}
    uint64_t per_token=0,head=0;
    for(const auto &[id,w]:all) {
      if(id.ends_with(".up"))continue;
      const auto up=id.ends_with(".gate")?id.substr(0,id.size()-5)+".up":"";
      if(!selected.empty() && !selected.count(id) && !selected.count(up))continue;
      const uint64_t size=2ull*w.in+4ull*w.out;
      if(id=="head")head=size;else per_token+=size;
    }
    *bytes=per_token*tokens+head*requests;return 0;
  }catch(const std::exception &e){error=e.what();return -1;}
}
