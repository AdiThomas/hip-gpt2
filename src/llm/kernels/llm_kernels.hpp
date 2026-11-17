#pragma once
// Launcher declarations for every GPT-2 engine kernel (SPEC 6.3).
//
// Conventions shared by all launchers:
//   * All tensors are FP32, row-major, device pointers.
//   * Launchers are asynchronous on `stream`: no synchronization, no allocation (SPEC 3.5.7).
//   * "T" is the number of token rows processed by one call (prefill: P, decode: 1).
//   * Shapes are given as [rows, cols]; "d" = n_embd, "H" = n_head, "hd" = d / H.
//
// KV-cache layout (used by kv_store and both attention kernels):
//   Per layer there are two buffers K and V of shape [C, d] (C = n_ctx capacity).
//   Row p holds the key/value vector of the token at ABSOLUTE position p, and the columns use
//   the same head-major order as the qkv projection output: head h owns columns
//   [h*hd, (h+1)*hd). Because rows are indexed by position, decode never moves data: it
//   appends row `len` and attends to rows 0..len. Prefill and decode share one definition of
//   attention: query row r sits at absolute position pos0 + r and attends to cache rows
//   0..pos0+r (causal mask = "only rows that already exist").
#include "hip_common.hpp"

// GPT-2 GELU, tanh approximation: 0.5*z*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z^3))).
// Shared by linear.hip (fused epilogues) and elementwise.hip.
__device__ inline float gelu_tanh(float z) {
    const float kSqrt2OverPi = 0.7978845608028654f;
    return 0.5f * z * (1.0f + tanhf(kSqrt2OverPi * (z + 0.044715f * z * z * z)));
}

// ---- embedding.hip ---------------------------------------------------------------------
// x[t][:] = wte[tok[t]][:] + wpe[pos0 + t][:]       tok: device int32 [T]
void launch_embedding(const int* tok, const float* wte, const float* wpe, float* x,
                      int T, int d, int pos0, hipStream_t stream);

// ---- layernorm.hip ---------------------------------------------------------------------
// y[t] = (x[t] - mean) / sqrt(var + 1e-5) * gamma + beta, biased variance over d.
// y may alias x.
void launch_layernorm(const float* x, const float* gamma, const float* beta, float* y,
                      int T, int d, hipStream_t stream);

// ---- linear.hip ------------------------------------------------------------------------
// Decode-path linear layer (one token row): y[N] = x[K] @ W[K,N] + b[N], then GELU if
// apply_gelu != 0. W is the HF Conv1D layout [in, out], used as stored (no transpose).
void launch_gemv_xw(const float* x, const float* W, const float* b, float* y,
                    int K, int N, int apply_gelu, hipStream_t stream);
// LM head on the tied embedding: y[t][v] = dot(W[v][:], x[t][:]) for t < T, v < V.
// W: [V, d], x: [T, d] (row stride d), y: [T, V].
void launch_gemv_rowdot(const float* W, const float* x, float* y,
                        int V, int d, int T, hipStream_t stream);
// Prefill epilogues after a GEMM: Y[t][n] += b[n]  /  Y[t][n] = gelu(Y[t][n] + b[n]).
void launch_bias_add(float* Y, const float* b, int T, int N, hipStream_t stream);
void launch_bias_gelu(float* Y, const float* b, int T, int N, hipStream_t stream);
// Fallback FP32 GEMM with the include/gemm.hpp contract (row-major C[M,N] = A[M,K] B[K,N]).
// Used by the engine only when it is not linked against the GEMM ladder.
void launch_llm_gemm_fallback(const float* dA, const float* dB, float* dC,
                              int M, int N, int K, hipStream_t stream);

// ---- elementwise.hip -------------------------------------------------------------------
// x[i] += r[i] for i < n.
void launch_residual_add(float* x, const float* r, int n, hipStream_t stream);
// x[i] = gelu(x[i]) (tanh approximation) for i < n.
void launch_gelu(float* x, int n, hipStream_t stream);
// Copies k = qkv[t][d:2d) and v = qkv[t][2d:3d) into K[pos0+t], V[pos0+t] for t < T.
// qkv: [T, 3d]; K, V: [C, d].
void launch_kv_store(const float* qkv, float* K, float* V, int T, int d, int pos0,
                     hipStream_t stream);

// ---- attention_naive.hip ---------------------------------------------------------------
// Three launches (scores, causal softmax, weighted sum of V).
//   q:      query rows, row stride q_stride (the engine passes qkv with stride 3d)
//   K, V:   KV-cache of one layer, [C, d]
//   scores: global scratch of at least H * T * C floats (layout [H][T][C])
//   y:      [T, d] output, head h writes columns [h*hd, (h+1)*hd)
// Query row r is at absolute position pos0 + r and attends to cache rows 0..pos0+r.
void launch_attention_naive(const float* q, int q_stride, const float* K, const float* V,
                            float* scores, float* y, int pos0, int T, int H, int hd, int C,
                            hipStream_t stream);

// ---- attention_fused.hip ---------------------------------------------------------------
// Decode attention (T = 1) in one launch, one block per head, no global score buffer.
// q: [d] query of the row at position pos0 (attends to rows 0..pos0); y: [d].
// Requirements: pos0 + 1 <= kFusedAttnMaxKeys, hd <= kFusedAttnMaxHeadDim.
constexpr int kFusedAttnMaxKeys    = 1024;  // scores of one head live in static LDS (4 KiB)
constexpr int kFusedAttnMaxHeadDim = 256;   // query vector of one head cached in LDS
void launch_attention_fused(const float* q, const float* K, const float* V, float* y,
                            int pos0, int H, int hd, hipStream_t stream);
