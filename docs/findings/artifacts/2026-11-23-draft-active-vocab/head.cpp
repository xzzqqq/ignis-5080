#include "ignis_trellis_linear.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

int ignis_lab_active_draft(const ignis_trellis_weight &, const void *, void *,
                           const ignis_trellis_workspace &, cudaStream_t);
extern "C" int ignis_lab_set_vocab_bits(const uint32_t *);
void ck(cudaError_t err) {if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));}
void op(int err) {if (err) throw std::runtime_error(ignis_trellis_last_error());}
uint8_t *allocate(size_t bytes) {
  uint8_t *p; ck(cudaMalloc(&p, bytes)); ck(cudaMemset(p, 0, bytes)); return p;
}

int main(int argc, char **argv) {
  if (argc != 3) return 1;
  constexpr int in=5120, out=248320, tokens=7, blocks=out/128;
  constexpr size_t codes=size_t(in)*out/2, bytes=codes+2*(in+out);
  std::ifstream file(argv[1], std::ios::binary); file.seekg(std::stoull(argv[2]));
  std::vector<uint8_t> payload(bytes);
  if (!file.read(reinterpret_cast<char *>(payload.data()), bytes)) return 2;
  uint8_t *wbuf=allocate(bytes);
  ck(cudaMemcpy(wbuf,payload.data(),bytes,cudaMemcpyHostToDevice));
  ignis_trellis_weight w{wbuf,wbuf+codes,wbuf+codes+2*in,8,in,out,0};
  std::mt19937 rng(7); std::vector<uint16_t> x(in*tokens);
  for (auto &h:x) h=(rng()&0x8000) | ((123+rng()%5)<<7) | (rng()%128);
  auto *dx=allocate(2*x.size()), *dy=allocate(2*out*tokens);
  ck(cudaMemcpy(dx,x.data(),2*x.size(),cudaMemcpyHostToDevice));
  op(ignis_trellis_prepare());
  auto n=ignis_trellis_workspace_bytes(in,out,1,tokens);
  ignis_trellis_workspace ws{allocate(n),n};
  cudaStream_t stream;ck(cudaStreamCreate(&stream));
  auto full=[&]{op(ignis_trellis_linear(&w,dx,tokens,dy,out,IGNIS_TRELLIS_STORE,&ws,stream));};
  auto active=[&]{op(ignis_lab_active_draft(w,dx,dy,ws,stream));};
  full();ck(cudaStreamSynchronize(stream));
  std::vector<uint16_t> ref(out*tokens),got(ref.size());
  ck(cudaMemcpy(ref.data(),dy,2*ref.size(),cudaMemcpyDeviceToHost));
  cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
  auto time=[&](auto &func){
    func();ck(cudaEventRecord(start,stream));
    for(int i=0;i<20;++i) func();
    ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));
    float ms;ck(cudaEventElapsedTime(&ms,start,end));return ms/20;
  };
  std::array<uint32_t,61> mask;
  while(std::cin>>mask[0]){
    for(size_t j=1;j<mask.size();++j) if(!(std::cin>>mask[j])) return 3;
    ck(static_cast<cudaError_t>(ignis_lab_set_vocab_bits(mask.data())));
    active();ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(got.data(),dy,2*got.size(),cudaMemcpyDeviceToHost));
    int errors=0,count=0;
    for(int j=0;j<blocks;++j) count+=(mask[j/32]>>(j%32))&1;
    for(size_t i=0;i<got.size();++i){
      int b=i%out/128;bool kept=mask[b/32]&(1u<<(b%32));
      errors+=got[i] != (kept ? ref[i] : 0xff80u);
    }
    std::vector<float> a,b;
    for(int rep=0;rep<7;++rep){
      if(rep%2){b.push_back(time(active));a.push_back(time(full));}
      else{a.push_back(time(full));b.push_back(time(active));}
    }
    std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
    std::cout<<count<<" "<<a[3]<<" "<<b[3]<<" "<<errors<<std::endl;
    if(errors) return 4;
  }
}
