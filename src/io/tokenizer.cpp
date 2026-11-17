// GPT-2 byte-level BPE tokenizer, behaviourally equal to OpenAI encoder.py.
#include "tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "json_mini.hpp"

namespace hipgpt2 {

namespace {
#include "unicode_tables.inc"

bool in_ranges(const uint32_t (*r)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}
bool is_L(uint32_t c) { return in_ranges(kUniLetter, kUniLetterCount, c); }
bool is_N(uint32_t c) { return in_ranges(kUniNumber, kUniNumberCount, c); }
bool is_S(uint32_t c) { return in_ranges(kUniSpace, kUniSpaceCount, c); }
bool is_other(uint32_t c) { return !is_S(c) && !is_L(c) && !is_N(c); }  // [^\s\p{L}\p{N}]

void put_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
    }
}

struct CodePoints {
    std::vector<uint32_t> cp;
    std::vector<size_t> off;  // byte offset of each code point; off[n] = total size
};

// Strict UTF-8 decoder (rejects overlongs, surrogates, > U+10FFFF, truncated sequences).
CodePoints decode_utf8(const std::string& s) {
    CodePoints r;
    size_t i = 0, n = s.size();
    auto bad = [&]() { throw std::runtime_error("tokenizer: input is not valid UTF-8 (byte offset " + std::to_string(i) + ")"); };
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        uint32_t cp;
        size_t len;
        if (c < 0x80) { cp = c; len = 1; }
        else if (c >= 0xC2 && c <= 0xDF) { cp = c & 0x1F; len = 2; }
        else if (c >= 0xE0 && c <= 0xEF) { cp = c & 0x0F; len = 3; }
        else if (c >= 0xF0 && c <= 0xF4) { cp = c & 0x07; len = 4; }
        else { bad(); return r; }
        if (i + len > n) bad();
        for (size_t k = 1; k < len; ++k) {
            unsigned char cc = (unsigned char)s[i + k];
            if ((cc & 0xC0) != 0x80) bad();
            cp = (cp << 6) | (cc & 0x3F);
        }
        if ((len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)) ||
            (cp >= 0xD800 && cp <= 0xDFFF))
            bad();
        r.cp.push_back(cp);
        r.off.push_back(i);
        i += len;
    }
    r.off.push_back(n);
    return r;
}

// Hand-written equivalent of
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// as used by re.findall: at each position the first alternative that matches wins.
// Returns the length (in code points) of the match starting at i (always >= 1).
size_t match_at(const std::vector<uint32_t>& c, size_t i) {
    const size_t n = c.size();
    // 1. contractions (case sensitive)
    if (c[i] == '\'' && i + 1 < n) {
        uint32_t a = c[i + 1];
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if (i + 2 < n) {
            uint32_t b = c[i + 2];
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        }
    }
    // 2-4. optional single ASCII space followed by a run of one class
    size_t j = i + ((c[i] == ' ') ? 1 : 0);
    if (j < n) {
        if (is_L(c[j])) { while (j < n && is_L(c[j])) ++j; return j - i; }
        if (is_N(c[j])) { while (j < n && is_N(c[j])) ++j; return j - i; }
        if (is_other(c[j])) { while (j < n && is_other(c[j])) ++j; return j - i; }
    }
    // (If c[i] == ' ' the optional space can also be dropped, but ' ' is \s, so no class above can match it.)
    // 5/6. whitespace: \s+(?!\S) gives up its last char when followed by a non-space; else \s+
    size_t k = i;
    while (k < n && is_S(c[k])) ++k;
    if (k == n) return k - i;
    if (k - i >= 2) return k - i - 1;
    return 1;
}

// bytes_to_unicode() from encoder.py: byte -> code point
void build_byte_tables(uint32_t b2u[256]) {
    bool keep[256] = {false};
    for (int b = '!'; b <= '~'; ++b) keep[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) keep[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) keep[b] = true;
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
        if (keep[b]) b2u[b] = (uint32_t)b;
        else b2u[b] = 256 + n++;
    }
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
}  // namespace

struct Gpt2Tokenizer::Impl {
    std::unordered_map<std::string, int> encoder;          // mapped-unicode token string -> id
    std::vector<std::string> id_bytes;                      // id -> raw bytes
    std::unordered_map<std::string, int> ranks;             // "a b" -> merge rank
    uint32_t b2u[256];
    std::unordered_map<uint32_t, int> u2b;
    mutable std::mutex mu;
    mutable std::unordered_map<std::string, std::vector<int>> cache;  // mapped word -> ids

    std::vector<int> bpe_ids(const std::string& word_mapped) const {
        {
            std::lock_guard<std::mutex> g(mu);
            auto it = cache.find(word_mapped);
            if (it != cache.end()) return it->second;
        }
        // split into symbols (one per mapped code point)
        std::vector<std::string> sym;
        for (size_t i = 0; i < word_mapped.size();) {
            unsigned char ch = (unsigned char)word_mapped[i];
            size_t len = ch < 0x80 ? 1 : (ch < 0xE0 ? 2 : (ch < 0xF0 ? 3 : 4));
            sym.push_back(word_mapped.substr(i, len));
            i += len;
        }
        while (sym.size() > 1) {
            int best = -1;
            size_t best_i = 0;
            for (size_t i = 0; i + 1 < sym.size(); ++i) {
                auto it = ranks.find(sym[i] + " " + sym[i + 1]);
                if (it != ranks.end() && (best < 0 || it->second < best)) { best = it->second; best_i = i; }
            }
            if (best < 0) break;
            const std::string first = sym[best_i], second = sym[best_i + 1];
            std::vector<std::string> out;
            for (size_t i = 0; i < sym.size();) {
                if (i + 1 < sym.size() && sym[i] == first && sym[i + 1] == second) {
                    out.push_back(first + second);
                    i += 2;
                } else {
                    out.push_back(sym[i]);
                    ++i;
                }
            }
            sym.swap(out);
        }
        std::vector<int> ids;
        for (const auto& s : sym) {
            auto it = encoder.find(s);
            if (it == encoder.end()) throw std::runtime_error("tokenizer: BPE symbol missing from encoder.json");
            ids.push_back(it->second);
        }
        std::lock_guard<std::mutex> g(mu);
        cache.emplace(word_mapped, ids);
        return ids;
    }
};

Gpt2Tokenizer Gpt2Tokenizer::load(const std::string& encoder_json_path, const std::string& vocab_bpe_path) {
    auto impl = std::make_shared<Impl>();
    build_byte_tables(impl->b2u);
    for (int b = 0; b < 256; ++b) impl->u2b[impl->b2u[b]] = b;

    json::Value root;
    try {
        root = json::parse(slurp(encoder_json_path));
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("tokenizer: ") + encoder_json_path + ": " + e.what());
    }
    if (root.type != json::Value::Obj || root.obj.empty())
        throw std::runtime_error("tokenizer: " + encoder_json_path + ": expected a non-empty JSON object");
    int64_t max_id = -1;
    for (const auto& kv : root.obj) {
        if (!kv.second.is_int() || kv.second.i < 0 || kv.second.i > 10000000)
            throw std::runtime_error("tokenizer: " + encoder_json_path + ": bad id for token");
        max_id = std::max(max_id, kv.second.i);
    }
    impl->id_bytes.assign((size_t)max_id + 1, std::string());
    std::vector<char> seen((size_t)max_id + 1, 0);
    for (const auto& kv : root.obj) {
        int id = (int)kv.second.i;
        impl->encoder[kv.first] = id;
        // token string is made of mapped code points; convert to raw bytes
        CodePoints cp = decode_utf8(kv.first);
        std::string raw;
        for (uint32_t u : cp.cp) {
            auto it = impl->u2b.find(u);
            if (it == impl->u2b.end())
                throw std::runtime_error("tokenizer: " + encoder_json_path + ": token contains unmapped character");
            raw += (char)it->second;
        }
        impl->id_bytes[(size_t)id] = raw;
        seen[(size_t)id] = 1;
    }
    for (char s : seen)
        if (!s) throw std::runtime_error("tokenizer: " + encoder_json_path + ": ids are not contiguous");

    std::string bpe = slurp(vocab_bpe_path);
    std::istringstream in(bpe);
    std::string line;
    bool first = true;
    int rank = 0;
    while (std::getline(in, line)) {
        if (first) { first = false; continue; }  // "#version: 0.2" header
        if (line.empty()) continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= line.size())
            throw std::runtime_error("tokenizer: " + vocab_bpe_path + ": malformed merge line");
        std::string b = line.substr(sp + 1);
        while (!b.empty() && (b.back() == '\r' || b.back() == ' ')) b.pop_back();
        impl->ranks.emplace(line.substr(0, sp) + " " + b, rank++);
    }
    if (impl->ranks.empty()) throw std::runtime_error("tokenizer: " + vocab_bpe_path + ": no merges");

    Gpt2Tokenizer t;
    t.impl_ = std::move(impl);
    return t;
}

Gpt2Tokenizer Gpt2Tokenizer::load_dir(const std::string& dir) {
    return load(dir + "/encoder.json", dir + "/vocab.bpe");
}

std::vector<int> Gpt2Tokenizer::encode(const std::string& utf8) const {
    CodePoints cps = decode_utf8(utf8);
    std::vector<int> ids;
    size_t i = 0, n = cps.cp.size();
    while (i < n) {
        size_t len = match_at(cps.cp, i);
        std::string mapped;
        for (size_t b = cps.off[i]; b < cps.off[i + len]; ++b)
            put_utf8(mapped, impl_->b2u[(unsigned char)utf8[b]]);
        std::vector<int> w = impl_->bpe_ids(mapped);
        ids.insert(ids.end(), w.begin(), w.end());
        i += len;
    }
    return ids;
}

std::string Gpt2Tokenizer::decode_token(int id) const {
    if (id < 0 || (size_t)id >= impl_->id_bytes.size())
        throw std::out_of_range("tokenizer: token id " + std::to_string(id) + " out of range");
    return impl_->id_bytes[(size_t)id];
}

std::string Gpt2Tokenizer::decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids) s += decode_token(id);
    return s;
}

int Gpt2Tokenizer::vocab_size() const { return (int)impl_->id_bytes.size(); }

}  // namespace hipgpt2
