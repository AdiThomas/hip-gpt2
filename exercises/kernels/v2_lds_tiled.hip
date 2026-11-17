// STAGE: v2 — LDS tiling
// CHANGES: A block of T x T threads owns a T x T tile of C. It walks K in steps of T:
//   each thread copies one element of the A tile and one of the B tile into LDS
//   (__shared__), the block syncs, then every thread does T FMAs from LDS. Out-of-range
//   elements are written to LDS as 0 so the inner loop has no bounds checks.
// WHY FASTER: Each global element brought on chip is reused T times (once per thread of
//   its tile row/column). Global traffic per FMA falls from 8 bytes to 8/T = 0.25 bytes
//   (T = 32): 0.125 bytes/FLOP instead of 4. The data reuse moves from caches we hope
//   for into a scratchpad we control.
// PREDICTED BOTTLENECK: LDS bandwidth and instruction issue. The inner loop does 2 LDS
//   reads per FMA (1 FLOP per LDS read), and two barriers per K-step stall the block.
//   Profiler: LDS instructions dominate, VALU utilisation well under 50%, few bank
//   conflicts (As[ty][k] is a broadcast, Bs[k][tx] is stride-1).
#include "gemm.hpp"

// T = 32: a 32-float row is one 128-byte line (coalesced tile loads), 32x32 = 1024
// threads is the HIP block maximum, and the two tiles use 8 KiB of LDS.
constexpr int kV2Tile = 32;

__global__ void gemm_v2_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_v2(const float* dA, const float* dB, float* dC, int M, int N, int K,
               hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
