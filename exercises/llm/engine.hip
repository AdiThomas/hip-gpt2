// GPT-2 engine: weight upload, buffer allocation, prefill/decode forward passes, generation loop.
// See engine.hpp for the prefill-vs-decode and KV-cache design notes.
#include "engine.hpp"

#include <chrono>
#include <stdexcept>

#include "kernels/llm_kernels.hpp"
#include "safetensors.hpp"
#if defined(HIPGPT2_LLM_USE_LADDER_GEMM)
#include "gemm.hpp"
#endif

namespace hipgpt2 {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Tensor lookup: "<name>", then "transformer.<name>" (HF checkpoints use the prefix).
std::string resolve(const SafeTensors& st, const std::string& name) {
    if (st.has(name)) return name;
    const std::string prefixed = "transformer." + name;
    if (st.has(prefixed)) return prefixed;
    throw std::runtime_error("model.safetensors: missing tensor '" + name + "'");
}

}  // namespace

const char* Gpt2Engine::gemm_kernel_name() {
#if defined(HIPGPT2_LLM_USE_LADDER_GEMM)
    return gemm_fp32_best_name();
#else
    return "internal";
#endif
}

bool Gpt2Engine::fused_attn_supported() const {
    return cfg_.n_ctx <= kFusedAttnMaxKeys && cfg_.head_dim() <= kFusedAttnMaxHeadDim;
}

Gpt2Engine::Gpt2Engine(const std::string& model_dir) {
    const auto t0 = Clock::now();
    cfg_ = load_gpt2_config(model_dir);
    const SafeTensors st = SafeTensors::open(model_dir + "/model.safetensors");
    const int64_t d = cfg_.n_embd, V = cfg_.vocab_size, C = cfg_.n_ctx;

    HIP_CHECK(hipStreamCreate(&stream_));

    auto alloc = [this](size_t count) {
        float* p = nullptr;
        HIP_CHECK(hipMalloc(&p, count * sizeof(float)));
        allocations_.push_back(p);
        return p;
    };
    // Every weight is validated by shape and uploaded once, unmodified (no transposes).
    auto upload = [&](const std::string& name, std::initializer_list<int64_t> shape) {
        const std::string key = resolve(st, name);
        const float* host = st.f32(key, shape);
        size_t n = 1;
        for (int64_t s : shape) n *= static_cast<size_t>(s);
        float* dev = alloc(n);
        HIP_CHECK(hipMemcpy(dev, host, n * sizeof(float), hipMemcpyHostToDevice));
        return static_cast<const float*>(dev);
    };

    wte_   = upload("wte.weight", {V, d});
    wpe_   = upload("wpe.weight", {C, d});
    lnf_g_ = upload("ln_f.weight", {d});
    lnf_b_ = upload("ln_f.bias", {d});
    const int di = static_cast<int>(d);
    for (int i = 0; i < cfg_.n_layer; ++i) {
        const std::string p = "h." + std::to_string(i) + ".";
        Layer L{};
        L.ln1_g = upload(p + "ln_1.weight", {d});
        L.ln1_b = upload(p + "ln_1.bias", {d});
        L.ln2_g = upload(p + "ln_2.weight", {d});
        L.ln2_b = upload(p + "ln_2.bias", {d});
        L.attn_qkv  = {upload(p + "attn.c_attn.weight", {d, 3 * d}), upload(p + "attn.c_attn.bias", {3 * d}), di, 3 * di};
        L.attn_proj = {upload(p + "attn.c_proj.weight", {d, d}), upload(p + "attn.c_proj.bias", {d}), di, di};
        L.fc        = {upload(p + "mlp.c_fc.weight", {d, 4 * d}), upload(p + "mlp.c_fc.bias", {4 * d}), di, 4 * di};
        L.mlp_proj  = {upload(p + "mlp.c_proj.weight", {4 * d, d}), upload(p + "mlp.c_proj.bias", {d}), 4 * di, di};
        L.K = alloc(static_cast<size_t>(C * d));
        L.V = alloc(static_cast<size_t>(C * d));
        layers_.push_back(L);
    }

    // Activations at capacity C rows: nothing is allocated during generation.
    x_      = alloc(static_cast<size_t>(C * d));
    a_      = alloc(static_cast<size_t>(C * d));
    qkv_    = alloc(static_cast<size_t>(C * 3 * d));
    y_      = alloc(static_cast<size_t>(C * d));
    proj_   = alloc(static_cast<size_t>(C * d));
    mlp_    = alloc(static_cast<size_t>(C * 4 * d));
    logits_ = alloc(static_cast<size_t>(V));
    scores_ = alloc(static_cast<size_t>(cfg_.n_head) * C * C);  // naive attention [H][T][C]
    HIP_CHECK(hipMalloc(&tok_dev_, static_cast<size_t>(C) * sizeof(int)));
    tok_host_.resize(static_cast<size_t>(C));
    HIP_CHECK(hipDeviceSynchronize());
    load_ms_ = ms_since(t0);
}

Gpt2Engine::~Gpt2Engine() {
    for (float* p : allocations_) (void)hipFree(p);
    if (tok_dev_) (void)hipFree(tok_dev_);
    if (stream_) (void)hipStreamDestroy(stream_);
}

// Y[T,N] = X[T,K] @ W[K,N] + b (optionally GELU).
//   decode (T == 1): one GEMV with the bias/GELU fused - the step is weight-bandwidth-bound.
//   prefill: GEMM (ladder or fallback) followed by a bias(+GELU) epilogue kernel.
void Gpt2Engine::linear(const float* X, int T, const Linear& L, float* Y, bool gelu,
                        bool decode_path) {
    if (decode_path) {
        launch_gemv_xw(X, L.W, L.b, Y, L.K, L.N, gelu ? 1 : 0, stream_);
        return;
    }
#if defined(HIPGPT2_LLM_USE_LADDER_GEMM)
    gemm_fp32_best(X, L.W, Y, T, L.N, L.K, stream_);
#else
    launch_llm_gemm_fallback(X, L.W, Y, T, L.N, L.K, stream_);
#endif
    if (gelu) launch_bias_gelu(Y, L.b, T, L.N, stream_);
    else      launch_bias_add(Y, L.b, T, L.N, stream_);
}

// Forward pass of SPEC 6.1 for the T token ids already in tok_dev_, at absolute positions
// pos0 = len_ .. len_ + T - 1. Leaves ln_f(x) in a_ and advances len_.
void Gpt2Engine::forward_rows(int T, bool decode_path, bool fused_attn) {
    const int d = cfg_.n_embd, H = cfg_.n_head, hd = cfg_.head_dim(), C = cfg_.n_ctx;
    const int pos0 = len_;
    HIPGPT2_REQUIRE(T >= 1 && pos0 + T <= C, "sequence exceeds n_ctx");

    launch_embedding(tok_dev_, wte_, wpe_, x_, T, d, pos0, stream_);
    for (const Layer& L : layers_) {
        // --- attention block ---
        launch_layernorm(x_, L.ln1_g, L.ln1_b, a_, T, d, stream_);
        linear(a_, T, L.attn_qkv, qkv_, false, decode_path);           // [T, 3d] = q | k | v
        // Append k, v of the new rows to the cache BEFORE attention: row pos0 + t must be
        // visible to query t itself (u <= p includes the diagonal).
        launch_kv_store(qkv_, L.K, L.V, T, d, pos0, stream_);
        if (decode_path && fused_attn)
            launch_attention_fused(qkv_, L.K, L.V, y_, pos0, H, hd, stream_);
        else
            launch_attention_naive(qkv_, 3 * d, L.K, L.V, scores_, y_, pos0, T, H, hd, C, stream_);
        linear(y_, T, L.attn_proj, proj_, false, decode_path);
        launch_residual_add(x_, proj_, T * d, stream_);
        // --- MLP block ---
        launch_layernorm(x_, L.ln2_g, L.ln2_b, a_, T, d, stream_);
        linear(a_, T, L.fc, mlp_, true, decode_path);                  // GELU fused in epilogue
        linear(mlp_, T, L.mlp_proj, proj_, false, decode_path);
        launch_residual_add(x_, proj_, T * d, stream_);
    }
    launch_layernorm(x_, lnf_g_, lnf_b_, a_, T, d, stream_);
    len_ += T;
}

double Gpt2Engine::prefill(const int* tokens, int T, float* logits_last, float* logits_all) {
    const int d = cfg_.n_embd, V = cfg_.vocab_size;
    HIPGPT2_REQUIRE(T >= 1 && len_ + T <= cfg_.n_ctx, "prefill: sequence exceeds n_ctx");
    for (int t = 0; t < T; ++t) {
        HIPGPT2_REQUIRE(tokens[t] >= 0 && tokens[t] < V, "prefill: token id out of range");
        tok_host_[t] = tokens[t];
    }
    const auto t0 = Clock::now();
    HIP_CHECK(hipMemcpyAsync(tok_dev_, tok_host_.data(), T * sizeof(int), hipMemcpyHostToDevice, stream_));
    forward_rows(T, /*decode_path=*/false, /*fused_attn=*/false);
    // LM head only on the rows that are needed (tied embedding, y = wte . xf).
    float* all_dev = nullptr;
    if (logits_all) {
        if (!logits_all_) {  // lazily allocated, only for --dump-mode all (SPEC 6.2)
            HIP_CHECK(hipMalloc(&logits_all_, static_cast<size_t>(cfg_.n_ctx) * V * sizeof(float)));
            allocations_.push_back(logits_all_);
        }
        all_dev = logits_all_;
        launch_gemv_rowdot(wte_, a_, all_dev, V, d, T, stream_);
    }
    if (logits_last)
        launch_gemv_rowdot(wte_, a_ + static_cast<size_t>(T - 1) * d, logits_, V, d, 1, stream_);
    HIP_CHECK(hipStreamSynchronize(stream_));
    const double device_ms = ms_since(t0);
    if (logits_last) {
        HIP_CHECK(hipMemcpyAsync(logits_last, logits_, V * sizeof(float), hipMemcpyDeviceToHost, stream_));
    }
    if (logits_all) {
        HIP_CHECK(hipMemcpyAsync(logits_all, all_dev, static_cast<size_t>(T) * V * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_));
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
    return device_ms;
}

void Gpt2Engine::decode(int token, bool fused_attn, float* logits_host) {
    const int d = cfg_.n_embd, V = cfg_.vocab_size;
    HIPGPT2_REQUIRE(token >= 0 && token < V, "decode: token id out of range");
    HIPGPT2_REQUIRE(len_ + 1 <= cfg_.n_ctx, "decode: KV-cache is full");
    tok_host_[0] = token;
    HIP_CHECK(hipMemcpyAsync(tok_dev_, tok_host_.data(), sizeof(int), hipMemcpyHostToDevice, stream_));
    forward_rows(1, /*decode_path=*/true, fused_attn && fused_attn_supported());
    launch_gemv_rowdot(wte_, a_, logits_, V, d, 1, stream_);
    HIP_CHECK(hipMemcpyAsync(logits_host, logits_, V * sizeof(float), hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
}

GenerateResult Gpt2Engine::generate(const std::vector<int>& prompt, const GenerateOptions& opt,
                                    const std::function<void(int)>& on_token) {
    const int V = cfg_.vocab_size, C = cfg_.n_ctx;
    const int P = static_cast<int>(prompt.size());
    HIPGPT2_REQUIRE(P >= 1 && P <= C, "generate: prompt length must be in [1, n_ctx]");
    const int G = opt.max_new_tokens;
    HIPGPT2_REQUIRE(G >= 1 && P + G - 1 <= C, "generate: max_new_tokens exceeds n_ctx - P + 1");

    std::vector<float> logits(static_cast<size_t>(V));
    GenerateResult res;

    // Warm-up: prefill + one decode step, untimed; then the cache is reset.
    for (int w = 0; w < opt.warmup; ++w) {
        reset();
        prefill(prompt.data(), P, logits.data());
        if (P < C) decode(Sampler::argmax(logits.data(), V), opt.fused_attn, logits.data());
    }
    reset();

    Sampler sampler(opt.sampling);
    std::vector<int> seq = prompt;  // used by the --no-kv-cache baseline
    auto emit = [&](int tok) {
        res.tokens.push_back(tok);
        seq.push_back(tok);
        if (opt.keep_logits) res.logits.insert(res.logits.end(), logits.begin(), logits.end());
        if (on_token) on_token(tok);
    };

    // Prefill -> first token. TTFT includes the host copy and the sampling.
    const auto t0 = Clock::now();
    res.prefill_ms = prefill(prompt.data(), P, logits.data());
    const int first = sampler.sample(logits.data(), V);
    res.ttft_ms = ms_since(t0);
    emit(first);

    // Decode: one row per step through the GEMV path, attending to the cache.
    const auto t1 = Clock::now();
    int produced = 1;
    while (produced < G && !(opt.stop_at_eot && res.tokens.back() == kEndOfTextId)) {
        if (opt.use_kv_cache) {
            decode(res.tokens.back(), opt.fused_attn, logits.data());
        } else {
            // Debug baseline: forget the cache and recompute the whole sequence every step.
            reset();
            prefill(seq.data(), static_cast<int>(seq.size()), logits.data());
        }
        emit(sampler.sample(logits.data(), V));
        ++produced;
    }
    res.decode_seconds = std::chrono::duration<double>(Clock::now() - t1).count();
    const int gen = static_cast<int>(res.tokens.size());
    res.decode_tok_per_s = (gen >= 2 && res.decode_seconds > 0) ? (gen - 1) / res.decode_seconds : 0.0;
    return res;
}

}  // namespace hipgpt2
