// Usage: test_tokenizer <vocab_dir> <cases_file>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tokenizer.hpp"

using hipgpt2::Gpt2Tokenizer;

static std::string to_hex(const std::string& s) {
    static const char* d = "0123456789abcdef";
    std::string r;
    for (unsigned char c : s) { r += d[c >> 4]; r += d[c & 15]; }
    return r.empty() ? "-" : r;
}
static bool from_hex(const std::string& h, std::string* out) {
    out->clear();
    if (h == "-") return true;
    if (h.size() % 2) return false;
    auto v = [](char c) { return c >= '0' && c <= '9' ? c - '0' : (c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1); };
    for (size_t i = 0; i < h.size(); i += 2) {
        int a = v(h[i]), b = v(h[i + 1]);
        if (a < 0 || b < 0) return false;
        *out += (char)(a * 16 + b);
    }
    return true;
}
static std::string ids_str(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]);
    return s;
}

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s <vocab_dir> <cases_file>\n", argv[0]); return 2; }
    try {
        Gpt2Tokenizer tok = Gpt2Tokenizer::load_dir(argv[1]);
        std::ifstream f(argv[2], std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty() && line[0] == '#') continue;
            lines.push_back(line);
        }
        if (lines.size() % 2) { std::fprintf(stderr, "cases file has an odd number of lines\n"); return 2; }
        int pass = 0, fail = 0;
        for (size_t i = 0; i + 1 < lines.size(); i += 2) {
            std::string input;
            if (!from_hex(lines[i], &input)) { std::fprintf(stderr, "bad hex on case %zu\n", i / 2); return 2; }
            std::vector<int> want;
            std::istringstream is(lines[i + 1]);
            int x;
            while (is >> x) want.push_back(x);
            std::vector<int> got = tok.encode(input);
            bool ok = (got == want);
            std::string why;
            if (!ok) why = "encode mismatch";
            if (ok && tok.decode(got) != input) { ok = false; why = "decode(encode(s)) != s"; }
            if (ok) { ++pass; continue; }
            if (fail == 0) {
                std::printf("FIRST MISMATCH (case %zu): %s\n  input hex: %s\n  expected:  %s\n  got:       %s\n", i / 2,
                            why.c_str(), to_hex(input).c_str(), ids_str(want).c_str(), ids_str(got).c_str());
            }
            ++fail;
        }
        std::printf("tokenizer: %d passed, %d failed\n", pass, fail);
        return fail ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
