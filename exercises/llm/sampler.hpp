#pragma once
// Host-side token sampling (SPEC 6.4 step 6). Pure C++, no HIP.
#include <cstdint>
#include <random>

namespace hipgpt2 {

struct SamplerConfig {
    float    temperature = 0.0f;  // 0 = greedy
    int      top_k       = 40;    // 0 = keep the whole vocabulary
    uint64_t seed        = 1234;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& cfg);
    // Greedy: lowest index of the maximum logit. Otherwise: logits / temperature, keep the
    // top-k (ties broken by lower index), softmax in double, draw with mt19937_64(seed).
    int sample(const float* logits, int vocab);
    bool greedy() const { return cfg_.temperature <= 0.0f; }

    static int argmax(const float* logits, int vocab);

private:
    SamplerConfig     cfg_;
    std::mt19937_64   rng_;
};

}  // namespace hipgpt2
