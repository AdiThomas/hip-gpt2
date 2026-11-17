// STAGE: llm — embedding (token + position)
// CHANGES: First kernel of the forward pass. One block per token row; threads stride over the
//   d columns and write x[t][j] = wte[tok[t]][j] + wpe[pos0 + t][j]. pos0 lets the same kernel
//   serve prefill (pos0 = 0, T = P) and decode (pos0 = current length, T = 1).
// WHY FASTER: Pure gather + add: 2 loads and 1 store per element (12 bytes, 1 FLOP). Threads of
//   a block touch consecutive columns of one embedding row, so every load and the store is
//   fully coalesced; the token id is read once per thread from a broadcast address.
// PREDICTED BOTTLENECK: Memory bandwidth and launch latency. At d = 768 a decode step moves
//   ~9 KB, far too little to saturate HBM, so the profiler will show this kernel as a few
//   microseconds dominated by launch overhead; it is irrelevant to end-to-end speed.

#include "kernels/llm_kernels.hpp"

// 256 threads = 4 waves of 64 (or 8 of 32): enough to cover d = 768 in 3 strided iterations
// while keeping blocks small so many token rows can be in flight during prefill.
constexpr int kEmbedThreads = 256;

__global__ void embedding_kernel(const int* tok, const float* wte, const float* wpe, float* x,
                                 int T, int d, int pos0) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_embedding(const int* tok, const float* wte, const float* wpe, float* x,
                      int T, int d, int pos0, hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
