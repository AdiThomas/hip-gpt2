// STAGE: llm — attention_fused (decode, online softmax, one block per head)
// CHANGES: Decode-only (T = 1) replacement for the three naive launches. One block per head;
//   the scores of that head stay in a static LDS array (<= 1024 floats), so nothing but q, the
//   KV-cache rows and y touches global memory. The softmax statistics are computed ONLINE:
//   each thread walks its keys u = tid, tid + blockDim, ... keeping
//       m = running max of its scores,  l = running sum of exp(s - m).
//   When a new score s arrives: m' = max(m, s), l' = l * exp(m - m') + exp(s - m').
//   Two partial results (m1, l1), (m2, l2) over disjoint key sets merge with the same rule:
//       m' = max(m1, m2),   l' = l1 * exp(m1 - m') + l2 * exp(m2 - m').
//   Rescaling by exp(m_old - m') re-expresses a sum computed relative to an old max relative
//   to the new one, so the merged pair equals what one serial pass would produce (the merge
//   is associative, which is what makes a parallel tree legal). A shared-memory tree applies
//   it across the block; then w[u] = exp(s[u] - m) / l and y_h[j] = sum_u w[u] V[u][h*hd + j].
// WHY FASTER: The naive path writes the H*(pos0+1) scores to global memory, reads/writes them
//   in softmax and reads them again for the weighted sum, over three launches. Here the score
//   traffic is LDS-only and there is a single launch per layer, so the kernel's global traffic
//   is the unavoidable minimum: 2 * (pos0+1) * hd floats of K and V per head.
// PREDICTED BOTTLENECK: Parallelism: only H = 12 blocks for GPT-2 small, so a 304-CU MI300X is
//   mostly idle and the kernel is latency-bound; the score loop also reads each K row with a
//   stride of d floats between threads (uncoalesced). Split-K over the sequence (flash-decoding:
//   several blocks per head, each producing (m, l, partial y), merged with the rule above) is the
//   next step. The profiler should show low occupancy and low achieved bandwidth.

#include "kernels/llm_kernels.hpp"

// Power of two (required by the merge tree). 128 threads = 2 wave64 / 4 wave32: enough to
// cover hd = 64 output columns and to process 1024 keys in 8 strided iterations.
constexpr int kFusedThreads = 128;
constexpr float kNegLarge   = -3.402823466e+38f;  // -FLT_MAX; avoids inf - inf = NaN in merges

__global__ void attention_fused_kernel(const float* q, const float* K, const float* V, float* y,
                                       int pos0, int hd, int d, float scale) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_attention_fused(const float* q, const float* K, const float* V, float* y,
                            int pos0, int H, int hd, hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
