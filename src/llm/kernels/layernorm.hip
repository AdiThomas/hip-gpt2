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
    static __shared__ float red[kLnThreads];
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row = x + static_cast<size_t>(t) * d;
    float* out = y + static_cast<size_t>(t) * d;

    // Pass 1: mean. (t < T always holds for the launched grid; no early exit so that every
    // thread reaches every barrier.)
    float s = 0.f;
    for (int j = tid; j < d; j += kLnThreads) s += row[j];
    red[tid] = s;
    __syncthreads();
    for (int stride = kLnThreads / 2; stride > 0; stride /= 2) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    const float mean = red[0] / d;
    __syncthreads();  // everyone has read red[0] before it is overwritten

    // Pass 2: biased variance of the centred values.
    float v = 0.f;
    for (int j = tid; j < d; j += kLnThreads) {
        const float c = row[j] - mean;
        v += c * c;
    }
    red[tid] = v;
    __syncthreads();
    for (int stride = kLnThreads / 2; stride > 0; stride /= 2) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    const float rstd = 1.0f / sqrtf(red[0] / d + kLnEps);

    // Pass 3: normalize. Safe when y aliases x: each element is read then written by the
    // same thread, and all reads of passes 1-2 finished before the last barrier.
    for (int j = tid; j < d; j += kLnThreads)
        out[j] = (row[j] - mean) * rstd * gamma[j] + beta[j];
    (void)T;
    // END SOLUTION
}

void launch_layernorm(const float* x, const float* gamma, const float* beta, float* y,
                      int T, int d, hipStream_t stream) {
    // BEGIN SOLUTION
    if (T <= 0) return;
    hipLaunchKernelGGL(layernorm_kernel, dim3(T), dim3(kLnThreads), 0, stream,
                       x, gamma, beta, y, T, d);
    // END SOLUTION
}
