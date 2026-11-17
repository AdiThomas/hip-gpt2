// STAGE: llm — attention_naive (scores, causal softmax, weighted sum of V)
// CHANGES: Textbook attention as three separate launches through a global score buffer
//   scores[H][T][C] (C = cache capacity, so the buffer is allocated once at engine start):
//     1. scores: s[h][t][u] = q[t]_h . K[u]_h / sqrt(hd)  for every u <= pos0 + t
//     2. softmax: per (h, t) row over its pos0 + t + 1 valid entries: subtract the row max,
//        exponentiate, divide by the sum. Masked entries (u > pos0 + t) are never written or
//        read, so the causal mask costs nothing and no -inf sentinels are needed.
//     3. weighted sum: y[t][j] = sum_u w[h][t][u] * V[u][j], h = j / hd.
//   Query row r is at absolute position pos0 + r and attends to KV-cache rows 0..pos0+r, so the
//   same code is prefill (pos0 = 0, T = P) and decode (pos0 = len, T = 1).
// WHY FASTER: It is the baseline, not fast. Its virtue is that every step is independently
//   checkable. Kernel 3 maps threadIdx.x to the output column j so the V loads (V[u][j..])
//   and the y stores are coalesced; kernel 1 maps threadIdx.x to the key index u.
// PREDICTED BOTTLENECK: Global memory round trips of the score matrix: H*T*(pos0+T) floats are
//   written, read+written by softmax, then read again (4 passes). For long prefills this
//   O(T^2) traffic dominates, and kernel 1 reads K rows with a stride of d floats between
//   neighbouring threads (uncoalesced). The fused decode kernel and, for prefill, a
//   flash-attention style tiling remove the score buffer entirely.

#include "kernels/llm_kernels.hpp"

// Kernel 1: one thread per key. 128 = 2 wave64 / 4 wave32; keys per block row.
constexpr int kScoreThreads = 128;
// Kernel 2: tree reduction width (power of two). 256 threads cover 1024 keys in 4 steps.
constexpr int kSoftmaxThreads = 256;
// Kernel 3: one thread per output column; 128 threads cover two GPT-2 heads (hd = 64).
constexpr int kWsumThreads = 128;

__global__ void attn_scores_kernel(const float* q, int q_stride, const float* K, float* scores,
                                   int pos0, int T, int hd, int C, int d, float scale) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

__global__ void attn_softmax_kernel(float* scores, int pos0, int T, int C) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

__global__ void attn_wsum_kernel(const float* scores, const float* V, float* y,
                                 int pos0, int T, int hd, int C, int d) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_attention_naive(const float* q, int q_stride, const float* K, const float* V,
                            float* scores, float* y, int pos0, int T, int H, int hd, int C,
                            hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
