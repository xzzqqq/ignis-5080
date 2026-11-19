#include "gqa_segment_workspace.cuh"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

namespace gqa=ignis_gqa_segment_workspace;
constexpr gqa::Layout layout{};
static_assert(layout.bytes==gqa::tensor_bytes(5120)+6*gqa::tensor_bytes(6144)+3*gqa::tensor_bytes(1024),
    "diagnostic refactor must preserve the existing allocation");

void check(gqa::Retained retained){
  const bool split=retained==gqa::Retained::SplitRope;
  // Simulate producers writing retained allocations. Dead optimized q/gate
  // allocations deliberately keep poison rather than plausible zero outputs.
  std::vector<unsigned char> workspace(layout.bytes,0xff);
  auto fill=[&](uint64_t offset,uint32_t width,unsigned char value){
    std::memset(layout.at(workspace.data(),offset),value,uint64_t(width)*gqa::tokens*2);
  };
  fill(layout.normalized,gqa::hidden,1);
  if(split){fill(layout.q,gqa::query,2);fill(layout.gate,gqa::query,3);}
  fill(layout.k,gqa::kv,4);fill(layout.v,gqa::kv,5);
  fill(layout.rq,gqa::query,6);fill(layout.rk,gqa::kv,7);
  fill(layout.attention,gqa::query,8);fill(layout.interleaved,2*gqa::query,9);
  ignis_lab_segment_tensor out[9]{};
  const int count=gqa::tensors(layout,retained,workspace.data(),workspace.size(),out,9);
  assert(count==(split?9:7));
  const char *names[]={"normalized_input","query","gate","key","value","rotated_query",
      "rotated_key","attention","interleaved_query_gate"};
  int expected=0;
  for(int i=0;i<count;++i){
    if(!split && expected==1)expected=3;
    assert(std::strcmp(out[i].name,(!split && expected==5)?"hq_transformed_query":names[expected])==0);
    const auto *data=static_cast<const unsigned char *>(out[i].data);
    assert(data>=workspace.data() && data+out[i].bytes<=workspace.data()+workspace.size());
    assert(out[i].dtype==0);
    uint64_t elements=1;
    for(auto dim:out[i].ne){assert(dim>0);elements*=dim;}
    assert(elements*2==out[i].bytes);
    assert(std::all_of(data,data+out[i].bytes,[&](unsigned char v){return v==expected+1;}));
    ++expected;
  }
  // Explicit shape/layout assertions at the two meaningfully different taps.
  const auto &transformed=out[split?5:3];
  assert(transformed.data==layout.at(workspace.data(),layout.rq));
  assert(transformed.ne[0]==256 && transformed.ne[1]==24 && transformed.ne[2]==8);
  const auto &projection=out[count-1];
  assert(projection.data==layout.at(workspace.data(),layout.interleaved));
  assert(projection.ne[0]==512 && projection.ne[1]==24 && projection.ne[2]==8);
  const int query_row_bytes=256*2;
  auto *interleaved=static_cast<unsigned char *>(layout.at(workspace.data(),layout.interleaved));
  for(int row=0;row<24*8;++row){
    std::memset(interleaved+row*2*query_row_bytes,10,query_row_bytes);
    std::memset(interleaved+(row*2+1)*query_row_bytes,11,query_row_bytes);
  }
  const auto *projected=static_cast<const unsigned char *>(projection.data);
  assert(projected[0]==10 && projected[query_row_bytes]==11);
  assert(projected[projection.bytes-1]==11);

  // Invalid capacity/extent must reject before writing any descriptor.
  out[0].name="untouched";
  assert(gqa::tensors(layout,retained,workspace.data(),workspace.size(),out,count-1)==-1);
  assert(std::strcmp(out[0].name,"untouched")==0);
  assert(gqa::tensors(layout,retained,workspace.data(),layout.bytes-1,out,9)==-1);
  assert(gqa::tensors(layout,retained,nullptr,workspace.size(),out,9)==-1);
  assert(gqa::tensors(layout,retained,workspace.data(),workspace.size(),nullptr,9)==-1);
  assert(std::strcmp(out[0].name,"untouched")==0);
}
int main(){check(gqa::Retained::SplitRope);check(gqa::Retained::InterleavedHq);}
