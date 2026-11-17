// STAGE: llm — layernorm
// CHANGES: One block per row. Two block-wide reductions through a shared-memory tree: first
//   the mean, then the biased variance of (x - mean) (two-pass, numerically safer than
//   E[x^2] - E[x]^2). Then every thread normalizes its strided columns and applies gamma/beta.
//   eps = 1e-5 sits inside the sqrt, matching PyTorch/HF GPT-2.
// WHY FASTER: The row (3 KB at d = 768) is read from global memory twice and the second read
//   hits L2; the reductions run in LDS with log2(blockDim) synchronized steps instead of a
//   serial loop. Columns are strided by blockDim so each wave's loads are coalesced.
//   The tree uses __syncthreads only (no wave-width assumption), so it is correct for wave32
//   (RDNA, NVIDIA) and wave64 (CDNA) alike.
// PREDICTED BOTTLENECK: Latency, not bandwidth: per row only ~3 passes over d floats with
//   2 * log2(256) = 16 barriers. In decode (1 row) the GPU runs a single block, so the
//   profiler will show very low occupancy; a production engine would fuse LN into the
//   preceding residual add or the following GEMV.

#include "kernels/llm_kernels.hpp"

// Power of two (required by the halving tree). 256 threads cover d = 768 in 3 iterations.
constexpr int kLnThreads = 256;
constexpr float kLnEps   = 1e-5f;

__global__ void layernorm_kernel(const float* x, const float* gamma, const float* beta,
                                 float* y, int T, int d) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_layernorm(const float* x, const float* gamma, const float* beta, float* y,
                      int T, int d, hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
