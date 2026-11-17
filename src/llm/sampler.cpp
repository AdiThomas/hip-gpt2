#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace hipgpt2 {

Sampler::Sampler(const SamplerConfig& cfg) : cfg_(cfg), rng_(cfg.seed) {}

int Sampler::argmax(const float* logits, int vocab) {
    int best = 0;
    for (int i = 1; i < vocab; ++i)
        if (logits[i] > logits[best]) best = i;  // strict '>' keeps the lowest index on ties
    return best;
}

int Sampler::sample(const float* logits, int vocab) {
    if (greedy()) return argmax(logits, vocab);

    std::vector<int> idx(vocab);
    std::iota(idx.begin(), idx.end(), 0);
    const int k = (cfg_.top_k <= 0 || cfg_.top_k > vocab) ? vocab : cfg_.top_k;
    auto by_logit = [logits](int a, int b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    };
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), by_logit);

    // Softmax over the kept candidates in double, max-subtracted.
    const double inv_t = 1.0 / static_cast<double>(cfg_.temperature);
    const double m = static_cast<double>(logits[idx[0]]) * inv_t;
    std::vector<double> p(k);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        p[i] = std::exp(static_cast<double>(logits[idx[i]]) * inv_t - m);
        sum += p[i];
    }
    std::uniform_real_distribution<double> uni(0.0, sum);
    const double r = uni(rng_);
    double acc = 0.0;
    for (int i = 0; i < k; ++i) {
        acc += p[i];
        if (r < acc) return idx[i];
    }
    return idx[k - 1];  // r == sum (rounding)
}

}  // namespace hipgpt2
