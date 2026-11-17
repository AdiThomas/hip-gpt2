#pragma once
// Pure C++. Implemented by [LLM-IO] in src/io/gpt2_config.cpp.
#include <string>

namespace hipgpt2 {

struct Gpt2Config {
    int n_layer = 0, n_head = 0, n_embd = 0, n_ctx = 0, vocab_size = 0;
    int head_dim() const { return n_embd / n_head; }
};

// Reads HF-style <model_dir>/config.json. Keys: n_layer, n_head, n_embd, vocab_size, and
// n_ctx (falls back to n_positions). Throws std::runtime_error if the file or a key is
// missing, any value <= 0, or n_embd % n_head != 0.
Gpt2Config load_gpt2_config(const std::string& model_dir);

}  // namespace hipgpt2
