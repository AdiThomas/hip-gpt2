// STAGE: v0_fp16 — naive GEMM with FP16 inputs, FP32 accumulation and output
// CHANGES: v0 with __half A and B: each product is formed from inputs converted to
//   float, accumulated in FP32, written as FP32. Same thread mapping as v0 (x -> row) so
//   it is literally "v0 with halves". Serves as the FP16 baseline/reference stage.
// WHY FASTER: Not meant to be fast. It halves input bytes (2 bytes/FLOP instead of 4),
//   which matters only because v0 is memory bound; it shows the data-type effect alone.
// PREDICTED BOTTLENECK: Same as v0: uncoalesced A loads and scattered C stores; the extra
//   half->float conversions are noise next to the memory stalls.
#include "gemm.hpp"

constexpr int kV0F16BlockX = 16;  // along M, same as v0
constexpr int kV0F16BlockY = 16;  // along N

__global__ void gemm_v0_fp16_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_v0_fp16(const __half* dA, const __half* dB, float* dC, int M, int N, int K,
                    hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
