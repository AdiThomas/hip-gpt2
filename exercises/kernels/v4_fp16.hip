// STAGE: v4_fp16 — LDS-tiled, register-blocked, double-buffered GEMM with FP16 inputs
// CHANGES: The v4 recipe for __half inputs: 64x64 block tile, 4x4 outputs per thread,
//   256 threads, two LDS buffers with register prefetch (one barrier per slab). Slabs
//   are BK = 32 deep and staged with 16-byte (8-half) vector loads when legal, by the
//   shared code in fp16_staging.inc that v5 also uses. LDS keeps the operands as FP16;
//   they are widened to FP32 in registers and accumulated in FP32.
// WHY FASTER: vs v0_fp16, each staged element is reused 64 times from LDS (global
//   traffic 1/32 byte/FLOP with 2-byte inputs) and 4 FLOPs per LDS read as in v3/v4.
//   vs FP32 v4: half the global and LDS bytes per slab, so the same LDS budget holds a
//   2x deeper slab (fewer barriers per FLOP).
// PREDICTED BOTTLENECK: FP32 FMA throughput plus conversion overhead (8 cvt per 16 FMAs)
//   on the VALU. This is the stage that shows why matrix cores matter: the inputs are
//   already FP16, but the math is still done one FMA at a time. Profiler: VALU-bound,
//   with v_cvt_f32_f16 a visible share of instructions.
#include "gemm.hpp"
#include "fp16_staging.inc"

constexpr int kV4F16TM = 4;                                // rows of C per thread
constexpr int kV4F16TN = 4;                                // cols of C per thread
constexpr int kV4F16ThreadsX = kF16BN / kV4F16TN;          // 16
constexpr int kV4F16ThreadsY = kF16BM / kV4F16TM;          // 16
constexpr int kV4F16Threads = kV4F16ThreadsX * kV4F16ThreadsY;  // 256: one 16-byte chunk per operand per thread

template <bool kVecA, bool kVecB>
__global__ void __launch_bounds__(kV4F16Threads)
gemm_v4_fp16_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_v4_fp16(const __half* dA, const __half* dB, float* dC, int M, int N, int K,
                    hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
