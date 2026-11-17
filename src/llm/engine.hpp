#pragma once
// GPT-2 inference engine (SPEC 6): weights on the device, preallocated activations, a per-layer
// KV-cache, a GEMM-based prefill path and a GEMV-based decode path.
//
// Why two paths?
//   Prefill pushes all P prompt rows through each linear layer at once: Y[P,N] = X[P,K] W[K,N].
//   Every weight is reused P times, so the work is a real GEMM with arithmetic intensity that
//   grows with P, and the register-blocked GEMM ladder (gemm_fp32_best) is the right kernel.
//   Decode pushes ONE row: y[N] = x[K] W[K,N]. Every 4-byte weight is loaded for a single
//   multiply-add (2 FLOPs) = 0.5 FLOP/byte, far left of any GPU's roofline ridge, so the step
//   is bound by how fast the weights stream from HBM. A GEMM's tiling buys nothing there; a
//   simple coalesced GEMV (gemv_xw) is the right kernel. GPT-2 small reads ~500 MB of FP32
//   weights per decode step (124M params, the 154 MB tied LM head included), so on an
//   MI300X (5.3 TB/s) the bandwidth ceiling is roughly 10,000 tokens/s before any overhead.
//
// KV-cache: per layer K_i, V_i of shape [n_ctx, d]. Row p = key/value of the token at absolute
// position p; columns in head-major order (head h owns [h*hd, (h+1)*hd)), identical to the qkv
// projection's k and v slices, so kv_store is a plain row copy. len() rows are valid in every
// layer. Prefill writes rows 0..P-1, each decode step appends row len and attends to 0..len,
// so the cost of a decode step is O(len) instead of re-running the whole sequence (compare
// --no-kv-cache, which recomputes everything every step).
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gpt2_config.hpp"
#include "hip_common.hpp"
#include "sampler.hpp"

namespace hipgpt2 {

constexpr int kEndOfTextId = 50256;  // GPT-2 <|endoftext|>

struct GenerateOptions {
    int           max_new_tokens = 64;   // G (caller clamps to n_ctx - P + 1)
    bool          fused_attn     = true; // decode attention kernel (prefill is always naive)
    bool          use_kv_cache   = true; // false: every step re-runs prefill on the full sequence
    bool          stop_at_eot    = false;
    int           warmup         = 0;    // untimed prefill + one decode step, then cache reset
    bool          keep_logits    = false;// store the logits row used for every sampled token
    SamplerConfig sampling;
};

struct GenerateResult {
    std::vector<int>   tokens;           // generated ids, in order
    std::vector<float> logits;           // tokens.size() * vocab when keep_logits
    double prefill_ms       = 0.0;       // device part of the prefill
    double ttft_ms          = 0.0;       // prefill start -> first token sampled on the host
    double decode_seconds   = 0.0;       // wall time over all decode steps
    double decode_tok_per_s = 0.0;       // (gen - 1) / decode_seconds, 0 if gen < 2
};

class Gpt2Engine {
public:
    // Loads config.json + model.safetensors from model_dir, uploads the weights unmodified,
    // allocates every buffer at capacity n_ctx. Throws std::runtime_error on bad input.
    explicit Gpt2Engine(const std::string& model_dir);
    ~Gpt2Engine();
    Gpt2Engine(const Gpt2Engine&) = delete;
    Gpt2Engine& operator=(const Gpt2Engine&) = delete;

    const Gpt2Config& config() const { return cfg_; }
    double load_ms() const { return load_ms_; }
    int    length() const { return len_; }
    // True if the fused decode attention can be used (n_ctx <= kFusedAttnMaxKeys).
    bool   fused_attn_supported() const;
    // "v4" (the ladder's gemm_fp32_best_name()) or "internal" (fallback GEMM).
    static const char* gemm_kernel_name();

    void reset() { len_ = 0; }

    // Runs T tokens through the prefill (GEMM) path at positions len()..len()+T-1 and appends
    // them to the KV-cache. Writes the logits of the last row to logits_last (host, [V]) when it
    // is non-null, and the logits of all T rows to logits_all (host, [T, V]) when non-null.
    // Returns the device time in milliseconds (launch to stream idle, excluding the copy).
    double prefill(const int* tokens, int T, float* logits_last, float* logits_all = nullptr);

    // One token through the decode (GEMV) path at position len(); appends to the cache and
    // writes its logits to logits_host ([V]).
    void decode(int token, bool fused_attn, float* logits_host);

    // Full generation per SPEC 6.4. on_token is called for every generated id as soon as it
    // has been sampled (for streaming output). Resets the cache first.
    GenerateResult generate(const std::vector<int>& prompt, const GenerateOptions& opt,
                            const std::function<void(int)>& on_token = {});

private:
    struct Linear { const float* W; const float* b; int K; int N; };
    struct Layer {
        const float *ln1_g, *ln1_b, *ln2_g, *ln2_b;
        Linear attn_qkv, attn_proj, fc, mlp_proj;
        float *K, *V;                       // KV-cache [C, d]
    };

    // Shared forward body: T rows at positions len_..len_+T-1, result in a_ (ln_f output).
    void forward_rows(int T, bool decode_path, bool fused_attn);
    void linear(const float* X, int T, const Linear& L, float* Y, bool gelu, bool decode_path);

    Gpt2Config cfg_;
    double     load_ms_ = 0.0;
    int        len_     = 0;
    hipStream_t stream_ = nullptr;

    std::vector<float*> allocations_;      // every device buffer, freed in the destructor
    const float* wte_ = nullptr;           // [V, d], also the tied LM head
    const float* wpe_ = nullptr;           // [C, d]
    const float* lnf_g_ = nullptr;
    const float* lnf_b_ = nullptr;
    std::vector<Layer> layers_;

    // Activations at capacity C rows (SPEC 6.2).
    float *x_ = nullptr, *a_ = nullptr, *qkv_ = nullptr, *y_ = nullptr, *proj_ = nullptr,
          *mlp_ = nullptr, *logits_ = nullptr, *scores_ = nullptr, *logits_all_ = nullptr;
    int*  tok_dev_ = nullptr;              // [C] int32 token ids
    std::vector<int> tok_host_;            // staging copy of the ids being uploaded
};

}  // namespace hipgpt2
