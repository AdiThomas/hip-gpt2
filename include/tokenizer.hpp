#pragma once
// Pure C++. Implemented by [LLM-IO] in src/io/tokenizer.cpp.
#include <memory>
#include <string>
#include <vector>

namespace hipgpt2 {

// GPT-2 byte-level BPE. Behaviour must equal OpenAI encoder.py (Encoder.encode/decode).
class Gpt2Tokenizer {
public:
    static constexpr int kEndOfText = 50256;

    // Throws std::runtime_error on missing/malformed files.
    static Gpt2Tokenizer load(const std::string& encoder_json_path,
                              const std::string& vocab_bpe_path);
    // Convenience: load(<dir>/encoder.json, <dir>/vocab.bpe).
    static Gpt2Tokenizer load_dir(const std::string& dir);

    // utf8 must be valid UTF-8. No special-token parsing ("<|endoftext|>" is plain text).
    std::vector<int> encode(const std::string& utf8) const;
    // Concatenated raw bytes of the tokens. Throws std::out_of_range on a bad id.
    std::string      decode(const std::vector<int>& ids) const;
    // Raw bytes of one token (may be a partial UTF-8 sequence; callers buffer for streaming).
    std::string      decode_token(int id) const;
    int              vocab_size() const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace hipgpt2
