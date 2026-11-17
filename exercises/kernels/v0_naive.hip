// STAGE: v0 — naive GEMM (one thread per C element, row taken from threadIdx.x)
// CHANGES: First stage. Thread (row, col) runs the full K loop straight out of global
//   memory: C[row][col] = sum_k A[row][k] * B[k][col]. Deliberately uses the
//   "x -> row" mapping so v1 can show what fixing only the mapping is worth.
// WHY FASTER: Baseline (correctness reference for the ladder). Every FMA needs 2 global
//   loads (8 bytes) for 2 FLOPs = 4 bytes/FLOP, with no on-chip reuse except what L1/L2 catch.
// PREDICTED BOTTLENECK: Memory. Within a wave the 16 x-neighbours sit on 16 different
//   rows, so each A load touches 16 cache lines K*4 bytes apart and each C store is 16
//   scattered 16-byte pieces. Profiler: low coalescing efficiency (many requests per
//   load instruction), high L2 traffic, achieved FLOPS a tiny fraction of peak.
#include "gemm.hpp"

// 16x16 = 256 threads: a common default (4 waves on wave64, 8 on wave32).
constexpr int kV0BlockX = 16;  // along M (rows) — the "wrong" axis on purpose
constexpr int kV0BlockY = 16;  // along N (cols)

__global__ void gemm_v0_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_v0(const float* dA, const float* dB, float* dC, int M, int N, int K,
               hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
