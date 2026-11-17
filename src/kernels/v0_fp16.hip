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
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        const __half* a = A + (size_t)row * K;
        const __half* b = B + col;
        float acc = 0.f;  // FP32 accumulator: FP16 accumulation would lose ~3 digits at K = 4096
        for (int k = 0; k < K; ++k) acc += __half2float(a[k]) * __half2float(b[(size_t)k * N]);
        C[(size_t)row * N + col] = acc;
    }
    // END SOLUTION
}

void launch_v0_fp16(const __half* dA, const __half* dB, float* dC, int M, int N, int K,
                    hipStream_t stream) {
    // BEGIN SOLUTION
    const dim3 block(kV0F16BlockX, kV0F16BlockY);
    const dim3 grid((M + kV0F16BlockX - 1) / kV0F16BlockX, (N + kV0F16BlockY - 1) / kV0F16BlockY);
    hipLaunchKernelGGL(gemm_v0_fp16_kernel, grid, block, 0, stream, dA, dB, dC, M, N, K);
    // END SOLUTION
}
