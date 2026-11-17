// Host-side emulation of the v5 MFMA path (gfx942). Runs the REAL fp16_staging.inc and
// v5_mfma.inc with a mock __builtin_amdgcn_mfma_f32_16x16x16f16 that records each lane's A/B
// fragments, rebuilds D using the hardware lane semantics (lane l: A[l%16][4*(l/16)+i],
// B[4*(l/16)+i][l%16], D[4*(l/16)+i][l%16]) and stores through the real v5_store.
// A wrong lane mapping in v5_mfma.inc makes this fail. Needs clang (ext_vector_type, _Float16).
// Written by the independent reviewer agent; wired into CTest as v5_mfma_emulation.
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#define __device__
#define __global__
struct __half { _Float16 v; };
static inline __half __float2half(float f){ return __half{(_Float16)f}; }
typedef _Float16 h4 __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
struct Rec { h4 a, b; };
static std::vector<Rec> g_rec[64]; static int g_lane;
static f4 __builtin_amdgcn_mfma_f32_16x16x16f16(h4 a, h4 b, f4 c, int, int, int) {
  g_rec[g_lane].push_back({a,b}); return c; }
#include "../../src/kernels/fp16_staging.inc"
#include "../../src/kernels/v5_mfma.inc"
constexpr int kV5WaveTile=32, kV5WavesN=2;
int run(int M,int N,int K,bool va,bool vb){
  std::mt19937 rng(M*131+N*7+K); std::uniform_real_distribution<float> U(-1,1);
  std::vector<__half> A((size_t)M*K+8), B((size_t)K*N+8); std::vector<float> C((size_t)M*N, NAN);
  for(auto&x:A) x=__float2half(U(rng)); for(auto&x:B) x=__float2half(U(rng));
  static __half As[2][kF16BM][kF16LdA], Bs[2][kF16BK][kF16LdB];
  std::vector<F16SlabRegs<256>> regs(256);
  int nsl=(K+kF16BK-1)/kF16BK;
  for(int by=0;by<(M+63)/64;++by) for(int bx=0;bx<(N+63)/64;++bx){
    int m0=by*64,n0=bx*64;
    for(int w=0;w<4;++w) { /* per wave */ }
    for(int l=0;l<64;++l) g_rec[l].clear();
    std::vector<Rec> rec[4][64];
    for(int s=0;s<nsl;++s){
      for(int t=0;t<256;++t){
        if(va&&vb) f16_load_slab<256,true,true>(regs[t],A.data(),B.data(),M,N,K,m0,n0,s*32,t);
        else if(va) f16_load_slab<256,true,false>(regs[t],A.data(),B.data(),M,N,K,m0,n0,s*32,t);
        else if(vb) f16_load_slab<256,false,true>(regs[t],A.data(),B.data(),M,N,K,m0,n0,s*32,t);
        else f16_load_slab<256,false,false>(regs[t],A.data(),B.data(),M,N,K,m0,n0,s*32,t);
        f16_store_slab<256>(regs[t],As[s&1],Bs[s&1],t);
      }
      for(int t=0;t<256;++t){ int w=t/64,l=t%64; g_lane=l; g_rec[l].clear();
        V5Accum acc; v5_acc_zero(acc);
        v5_mma_slab(acc,As[s&1],Bs[s&1],(w/2)*32,(w%2)*32,l);
        for(auto&r:g_rec[l]) rec[w][l].push_back(r); }
    }
    for(int w=0;w<4;++w){
      float D[2][2][16][16]={};
      int ncalls=rec[w][0].size();
      for(int n=0;n<ncalls;++n){ int i=(n%4)/2,j=n%2;
        for(int r=0;r<16;++r)for(int c=0;c<16;++c){ float s=0;
          for(int k=0;k<16;++k) s+=(float)rec[w][r+16*(k/4)][n].a[k%4]*(float)rec[w][c+16*(k/4)][n].b[k%4];
          D[i][j][r][c]+=s; } }
      for(int l=0;l<64;++l){ V5Accum acc;
        for(int i=0;i<2;++i)for(int j=0;j<2;++j)for(int e=0;e<4;++e) acc.c[i][j][e]=D[i][j][4*(l/16)+e][l%16];
        v5_store(acc,C.data(),M,N,m0+(w/2)*32,n0+(w%2)*32,l,nullptr); }
    }
  }
  double maxe=0; int bad=0;
  for(int i=0;i<M;++i)for(int j=0;j<N;++j){ double s=0,sa=0; for(int k=0;k<K;++k){double p=(double)(float)A[(size_t)i*K+k].v*(float)B[(size_t)k*N+j].v; s+=p; sa+=fabs(p);} 
    double e=fabs(C[(size_t)i*N+j]-s)/(sa+1e-30); if(!(e<=1e-3)) bad++; if(e>maxe||e!=e) maxe=e; }
  printf("%dx%dx%d va=%d vb=%d max_err=%.2e %s\n",M,N,K,va,vb,maxe,bad?"FAIL":"PASS"); return bad;
}
int main(){ int f=0; int S[][3]={{1,1,1},{1,7,5},{17,17,17},{127,127,127},{128,128,128},{333,257,129},{64,1000,3},{3,5,1000},{130,66,255},{64,64,32},{96,80,48}};
 for(auto&s:S){ f+=run(s[0],s[1],s[2],false,false); if(s[2]%8==0&&s[1]%8==0) f+=run(s[0],s[1],s[2],true,true);} printf("fails %d\n",f); return f ? 1 : 0; }
