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
    static __shared__ float s[kFusedAttnMaxKeys];        // scores, then softmax weights
    static __shared__ float qs[kFusedAttnMaxHeadDim];    // this head's query, pre-scaled
    static __shared__ float m_sh[kFusedThreads];
    static __shared__ float l_sh[kFusedThreads];
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;
    const int n   = pos0 + 1;                            // keys 0..pos0 (causal)
    const int col0 = h * hd;

    for (int i = tid; i < hd; i += kFusedThreads) qs[i] = q[col0 + i] * scale;
    __syncthreads();

    // Phase 1: scores + per-thread online (m, l).
    float m = kNegLarge, l = 0.f;
    for (int u = tid; u < n; u += kFusedThreads) {
        const float* kr = K + static_cast<size_t>(u) * d + col0;
        float sc = 0.f;
        for (int i = 0; i < hd; ++i) sc += qs[i] * kr[i];
        s[u] = sc;
        const float m_new = fmaxf(m, sc);
        l = l * expf(m - m_new) + expf(sc - m_new);
        m = m_new;
    }
    m_sh[tid] = m;
    l_sh[tid] = l;
    __syncthreads();

    // Phase 2: tree merge of the (m, l) pairs. A thread that saw no keys contributes
    // (-FLT_MAX, 0), which the merge rule turns into a no-op.
    for (int stride = kFusedThreads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            const float m1 = m_sh[tid], m2 = m_sh[tid + stride];
            const float mm = fmaxf(m1, m2);
            l_sh[tid] = l_sh[tid] * expf(m1 - mm) + l_sh[tid + stride] * expf(m2 - mm);
            m_sh[tid] = mm;
        }
        __syncthreads();
    }
    const float m_all = m_sh[0];
    const float inv_l = 1.0f / l_sh[0];

    // Phase 3: normalized weights in LDS (computed once per key, not once per column).
    for (int u = tid; u < n; u += kFusedThreads) s[u] = expf(s[u] - m_all) * inv_l;
    __syncthreads();

    // Phase 4: weighted sum of V; consecutive threads read consecutive columns (coalesced).
    for (int j = tid; j < hd; j += kFusedThreads) {
        float acc = 0.f;
        for (int u = 0; u < n; ++u) acc += s[u] * V[static_cast<size_t>(u) * d + col0 + j];
        y[col0 + j] = acc;
    }
    // END SOLUTION
}

void launch_attention_fused(const float* q, const float* K, const float* V, float* y,
                            int pos0, int H, int hd, hipStream_t stream) {
    // BEGIN SOLUTION
    HIPGPT2_REQUIRE(pos0 + 1 <= kFusedAttnMaxKeys, "fused attention: too many keys for LDS");
    HIPGPT2_REQUIRE(hd <= kFusedAttnMaxHeadDim, "fused attention: head_dim too large");
    const float scale = 1.0f / sqrtf(static_cast<float>(hd));
    hipLaunchKernelGGL(attention_fused_kernel, dim3(H), dim3(kFusedThreads), 0, stream,
                       q, K, V, y, pos0, hd, H * hd, scale);
    // END SOLUTION
}
