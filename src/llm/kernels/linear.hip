// STAGE: llm — linear layers (gemv_xw, gemv_rowdot, bias_add, bias_gelu, fallback gemm)
// CHANGES: Decode and prefill use different kernels for the same math.
//   gemv_xw:     decode Conv1D layer, y[N] = x[K] @ W[K,N] + b (+ optional fused GELU).
//                One thread per output column; x is staged through LDS in chunks.
//   gemv_rowdot: LM head on the tied embedding, y[v] = W[v,:] . x (W is [V,d], so each
//                output is a dot product along a contiguous row): one block per (v, t),
//                shared-memory tree reduction.
//   bias_add / bias_gelu: row-broadcast epilogues run after a prefill GEMM.
//   fallback gemm: 16x16 LDS-tiled GEMM, used only if the engine is built without the ladder.
// WHY FASTER: Decode processes ONE token row, so every weight is loaded once and used for one
//   multiply-add: 2 FLOPs per 4-byte FP32 weight = 0.5 FLOP/byte. That is ~30-60x below the
//   ridge point of a modern GPU (MI300X: ~81.7 TFLOP/s vector FP32 / 5.3 TB/s = ~15 FLOP/byte;
//   with matrix cores far higher), so decode is purely memory-bound and a GEMM's tiling buys
//   nothing: the right goal is "stream W at full bandwidth", which means coalesced loads of
//   W[k][n..n+63] by consecutive threads (gemv_xw) or along contiguous rows (gemv_rowdot).
//   Prefill with T rows reuses every weight T times (intensity ~T/2 FLOP/byte from weights),
//   so beyond T of a few dozen it becomes compute-bound and the register-blocked ladder GEMM
//   is the right tool. The roofline crossover is exactly why the engine has two paths.
// PREDICTED BOTTLENECK: gemv_xw: HBM bandwidth at best, but with one thread per column GPT-2
//   small has only N = 768..3072 threads per layer, i.e. 12..48 blocks of 64 - far fewer than
//   the 304 CUs of an MI300X, so the profiler will show low occupancy and achieved bandwidth
//   well below peak. Split-K across blocks (with a second reduction pass) is the next step.
//   gemv_rowdot: 50257 rows x 3 KB = 154 MB per token, bandwidth-bound; the per-row tree
//   reduction adds 7 barriers per block. Fallback gemm: LDS bandwidth (1 FMA per 2 LDS reads).

#include "kernels/llm_kernels.hpp"

// gemv_xw: 64 threads per block = one wave64 / two wave32. Small blocks maximize the number of
// blocks (N/64) to spread the weight stream over as many CUs as possible.
constexpr int kGemvThreads = 64;
// Elements of x staged into LDS per iteration. 256 floats = 1 KiB of LDS; every thread then
// reads x[k] from LDS (broadcast, conflict-free) instead of global memory.
constexpr int kGemvXChunk = 256;
// gemv_rowdot: 128 threads per row (power of two for the tree). d = 768 -> 6 loads/thread.
constexpr int kRowdotThreads = 128;
// Elementwise epilogues.
constexpr int kEpilogueThreads = 256;
// Fallback GEMM tile edge (16x16 = 256 threads, 2 KiB of LDS for both tiles).
constexpr int kFallbackTile = 16;

__global__ void gemv_xw_kernel(const float* x, const float* W, const float* b, float* y,
                               int K, int N, int apply_gelu) {
    // BEGIN SOLUTION
    static __shared__ float xs[kGemvXChunk];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += kGemvXChunk) {
        const int len = (K - k0 < kGemvXChunk) ? (K - k0) : kGemvXChunk;
        for (int i = threadIdx.x; i < len; i += blockDim.x) xs[i] = x[k0 + i];
        __syncthreads();
        if (n < N) {
            // Consecutive threads read consecutive n of the same row k: coalesced.
            const float* wcol = W + static_cast<size_t>(k0) * N + n;
            for (int i = 0; i < len; ++i) acc += xs[i] * wcol[static_cast<size_t>(i) * N];
        }
        __syncthreads();  // xs is overwritten by the next chunk
    }
    if (n < N) {
        float v = acc + b[n];
        y[n] = apply_gelu ? gelu_tanh(v) : v;
    }
    // END SOLUTION
}

__global__ void gemv_rowdot_kernel(const float* W, const float* x, float* y, int V, int d) {
    // BEGIN SOLUTION
    static __shared__ float red[kRowdotThreads];
    const int v = blockIdx.x;   // vocabulary row
    const int t = blockIdx.y;   // token row
    const float* wr = W + static_cast<size_t>(v) * d;
    const float* xr = x + static_cast<size_t>(t) * d;
    const int tid = threadIdx.x;
    float s = 0.f;
    for (int j = tid; j < d; j += kRowdotThreads) s += wr[j] * xr[j];
    red[tid] = s;
    __syncthreads();
    for (int stride = kRowdotThreads / 2; stride > 0; stride /= 2) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[static_cast<size_t>(t) * V + v] = red[0];
    // END SOLUTION
}

__global__ void bias_act_kernel(float* Y, const float* b, int T, int N, int apply_gelu) {
    // BEGIN SOLUTION
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(T) * N;
    if (i < total) {
        const float v = Y[i] + b[i % N];
        Y[i] = apply_gelu ? gelu_tanh(v) : v;
    }
    // END SOLUTION
}

__global__ void llm_gemm_fallback_kernel(const float* A, const float* B, float* C,
                                         int M, int N, int K) {
    // BEGIN SOLUTION
    static __shared__ float As[kFallbackTile][kFallbackTile];
    static __shared__ float Bs[kFallbackTile][kFallbackTile];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * kFallbackTile + ty;
    const int col = blockIdx.x * kFallbackTile + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += kFallbackTile) {
        // Zero-fill out-of-range elements so the inner loop needs no bounds checks.
        As[ty][tx] = (row < M && k0 + tx < K) ? A[static_cast<size_t>(row) * K + k0 + tx] : 0.f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[static_cast<size_t>(k0 + ty) * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < kFallbackTile; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[static_cast<size_t>(row) * N + col] = acc;
    // END SOLUTION
}

void launch_gemv_xw(const float* x, const float* W, const float* b, float* y,
                    int K, int N, int apply_gelu, hipStream_t stream) {
    // BEGIN SOLUTION
    const int blocks = (N + kGemvThreads - 1) / kGemvThreads;
    hipLaunchKernelGGL(gemv_xw_kernel, dim3(blocks), dim3(kGemvThreads), 0, stream,
                       x, W, b, y, K, N, apply_gelu);
    // END SOLUTION
}

void launch_gemv_rowdot(const float* W, const float* x, float* y, int V, int d, int T,
                        hipStream_t stream) {
    // BEGIN SOLUTION
    if (T <= 0) return;
    hipLaunchKernelGGL(gemv_rowdot_kernel, dim3(V, T), dim3(kRowdotThreads), 0, stream,
                       W, x, y, V, d);
    // END SOLUTION
}

void launch_bias_add(float* Y, const float* b, int T, int N, hipStream_t stream) {
    // BEGIN SOLUTION
    const size_t total = static_cast<size_t>(T) * N;
    if (total == 0) return;
    const unsigned blocks = static_cast<unsigned>((total + kEpilogueThreads - 1) / kEpilogueThreads);
    hipLaunchKernelGGL(bias_act_kernel, dim3(blocks), dim3(kEpilogueThreads), 0, stream,
                       Y, b, T, N, 0);
    // END SOLUTION
}

void launch_bias_gelu(float* Y, const float* b, int T, int N, hipStream_t stream) {
    // BEGIN SOLUTION
    const size_t total = static_cast<size_t>(T) * N;
    if (total == 0) return;
    const unsigned blocks = static_cast<unsigned>((total + kEpilogueThreads - 1) / kEpilogueThreads);
    hipLaunchKernelGGL(bias_act_kernel, dim3(blocks), dim3(kEpilogueThreads), 0, stream,
                       Y, b, T, N, 1);
    // END SOLUTION
}

void launch_llm_gemm_fallback(const float* dA, const float* dB, float* dC,
                              int M, int N, int K, hipStream_t stream) {
    // BEGIN SOLUTION
    dim3 block(kFallbackTile, kFallbackTile);
    dim3 grid((N + kFallbackTile - 1) / kFallbackTile, (M + kFallbackTile - 1) / kFallbackTile);
    hipLaunchKernelGGL(llm_gemm_fallback_kernel, grid, block, 0, stream, dA, dB, dC, M, N, K);
    // END SOLUTION
}
