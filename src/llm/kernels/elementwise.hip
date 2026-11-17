// STAGE: llm — elementwise (residual_add, gelu, kv_store)
// CHANGES: Grid-stride-free one-thread-per-element kernels.
//   residual_add: x += r (the two residual connections of every block).
//   gelu:         in-place tanh-approximation GELU (used when an epilogue cannot fuse it).
//   kv_store:     appends the k and v slices of the qkv projection to the KV-cache. Cache rows
//                 are indexed by ABSOLUTE position, so row pos0 + t receives token t's key and
//                 value; columns keep the head-major order of qkv (head h = [h*hd,(h+1)*hd)),
//                 which lets attention read a head's keys as a contiguous hd-float segment.
// WHY FASTER: Each kernel touches every element exactly once with consecutive threads on
//   consecutive addresses (fully coalesced). Arithmetic intensity is ~0.1 FLOP/byte, so the
//   only goal is bandwidth; nothing here can benefit from LDS.
// PREDICTED BOTTLENECK: Memory bandwidth for large T, launch latency in decode (a 768-float
//   residual add is ~6 KB of traffic). The profiler will show these as tiny kernels; the real
//   optimization is fusion (residual add into the GEMM/GEMV epilogue), not a faster kernel.

#include "kernels/llm_kernels.hpp"

// 256 threads per block: a common default that gives 4 wave64s / 8 wave32s per block.
constexpr int kEltThreads = 256;

__global__ void residual_add_kernel(float* x, const float* r, int n) {
    // BEGIN SOLUTION
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += r[i];
    // END SOLUTION
}

__global__ void gelu_kernel(float* x, int n) {
    // BEGIN SOLUTION
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = gelu_tanh(x[i]);
    // END SOLUTION
}

__global__ void kv_store_kernel(const float* qkv, float* K, float* V, int T, int d, int pos0) {
    // BEGIN SOLUTION
    // One block per token row t; threads stride over the d columns of k and v.
    const int t = blockIdx.x;
    if (t < T) {
        const float* src = qkv + static_cast<size_t>(t) * 3 * d;  // [q | k | v]
        float* krow = K + static_cast<size_t>(pos0 + t) * d;
        float* vrow = V + static_cast<size_t>(pos0 + t) * d;
        for (int j = threadIdx.x; j < d; j += blockDim.x) {
            krow[j] = src[d + j];
            vrow[j] = src[2 * d + j];
        }
    }
    // END SOLUTION
}

void launch_residual_add(float* x, const float* r, int n, hipStream_t stream) {
    // BEGIN SOLUTION
    if (n <= 0) return;
    hipLaunchKernelGGL(residual_add_kernel, dim3((n + kEltThreads - 1) / kEltThreads),
                       dim3(kEltThreads), 0, stream, x, r, n);
    // END SOLUTION
}

void launch_gelu(float* x, int n, hipStream_t stream) {
    // BEGIN SOLUTION
    if (n <= 0) return;
    hipLaunchKernelGGL(gelu_kernel, dim3((n + kEltThreads - 1) / kEltThreads),
                       dim3(kEltThreads), 0, stream, x, n);
    // END SOLUTION
}

void launch_kv_store(const float* qkv, float* K, float* V, int T, int d, int pos0,
                     hipStream_t stream) {
    // BEGIN SOLUTION
    if (T <= 0) return;
    hipLaunchKernelGGL(kv_store_kernel, dim3(T), dim3(kEltThreads), 0, stream,
                       qkv, K, V, T, d, pos0);
    // END SOLUTION
}
