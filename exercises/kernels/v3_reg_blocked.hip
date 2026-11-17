// STAGE: v3 — register blocking (each thread computes a 4x4 sub-tile of C)
// CHANGES: Block tile grows to 64x64 with a K-step of 16, computed by 256 threads; each
//   thread keeps a 4x4 accumulator array in registers. Per k, a thread loads 4 A values
//   and 4 B values from LDS into registers and does a 4x4 outer-product update. The A
//   tile is stored transposed in LDS (As[k][m]) so both operands are read along a row.
// WHY FASTER: Arithmetic intensity on the LDS side: v2 does 2 LDS reads per FMA
//   (1 FLOP/LDS read); v3 does 8 LDS reads per 16 FMAs (4 FLOPs/LDS read, 4x better).
//   Global side: a 64x64 tile reuses each loaded element 64 times (v2: 32), so global
//   traffic per FLOP halves again to 1/16 byte/FLOP.
// PREDICTED BOTTLENECK: Global load latency. Loads for the next slab are only issued
//   after the barrier, so the block idles while they are in flight; with ~2 blocks per
//   CU there is not enough other work to hide it. Profiler: VALU busy noticeably higher
//   than v2 but with stalls on vmcnt/barrier; LDS bank conflicts near zero.
#include "gemm.hpp"

// 64x64 block tile, 4x4 per thread -> 16x16 = 256 threads. 16 FMAs per 8 LDS reads is
// enough to make the inner loop VALU-heavy while 16 accumulators + 8 operands keep
// register use low (high occupancy). BK = 16: one slab is 2 x 4 KiB of LDS.
constexpr int kV3BM = 64;
constexpr int kV3BN = 64;
constexpr int kV3BK = 16;
constexpr int kV3TM = 4;                       // rows of C per thread
constexpr int kV3TN = 4;                       // cols of C per thread
constexpr int kV3ThreadsX = kV3BN / kV3TN;     // 16
constexpr int kV3ThreadsY = kV3BM / kV3TM;     // 16
constexpr int kV3Threads = kV3ThreadsX * kV3ThreadsY;  // 256
// +1 float of padding per As row: the transposing store As[k][m] has consecutive
// threads varying k, i.e. stride 65 words -> lands in different banks.
constexpr int kV3PadA = 1;
static_assert((kV3BM * kV3BK) % kV3Threads == 0, "A tile must split evenly over threads");
static_assert((kV3BK * kV3BN) % kV3Threads == 0, "B tile must split evenly over threads");

__global__ void __launch_bounds__(kV3Threads)
gemm_v3_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_v3(const float* dA, const float* dB, float* dC, int M, int N, int K,
               hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
