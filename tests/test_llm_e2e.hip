// End-to-end test of the engine on the tiny model (SPEC 7 [LLM-CORE] 2, D6).
//   test_llm_e2e <tiny model dir>
// Reference data (written by tools/make_tiny_model.py from a float64 numpy implementation):
//   prompt_ids.txt, ref_prefill_all.bin (P rows), ref_greedy.bin (G rows), ref_greedy_tokens.txt.
// Checks (each prints its measured max abs error):
//   (a) prefill logits at all P positions vs ref_prefill_all.bin          <= 1e-3
//   (b) greedy generation, fused decode attention: tokens equal, logits     <= 1e-3
//   (c) the same with naive decode attention
//   (d) prompt fed one token at a time through the decode path: final logits within 1e-3 of
//       prefill (for both attention kernels)
//   (e) --no-kv-cache semantics (re-run prefill every step): same tokens
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.hpp"

using namespace hipgpt2;

namespace {

int g_failures = 0;

struct Lgt1 {
    int rows = 0, vocab = 0;
    std::vector<float> data;
};

// LGT1: "LGT1", int32 LE rows, int32 LE vocab, rows*vocab float32 LE (SPEC 4.6).
Lgt1 read_lgt1(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4];
    int32_t hdr[2];
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
    if (!f || std::memcmp(magic, "LGT1", 4) != 0 || hdr[0] < 0 || hdr[1] <= 0)
        throw std::runtime_error("bad LGT1 header in " + path);
    Lgt1 r;
    r.rows = hdr[0];
    r.vocab = hdr[1];
    r.data.resize(static_cast<size_t>(r.rows) * r.vocab);
    f.read(reinterpret_cast<char*>(r.data.data()),
           static_cast<std::streamsize>(r.data.size() * sizeof(float)));
    if (!f) throw std::runtime_error("truncated LGT1 file " + path);
    return r;
}

std::vector<int> read_ids(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<int> ids;
    int x;
    while (f >> x) ids.push_back(x);
    return ids;
}

double max_abs(const float* a, const float* b, size_t n) {
    double e = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - b[i]);
        if (!(d <= e)) e = d;
    }
    return e;
}

void check(const std::string& what, bool ok, const std::string& detail = "") {
    std::printf("  %-58s %s%s\n", what.c_str(), ok ? "PASS" : "FAIL", detail.c_str());
    if (!ok) ++g_failures;
}

void check_err(const std::string& what, double err, double tol) {
    char buf[64];
    std::snprintf(buf, sizeof buf, " (max_err %.3e, tol %.0e)", err, tol);
    check(what, err <= tol, buf);
}

std::string ids_str(const std::vector<int>& v) {
    std::string s;
    for (int x : v) s += std::to_string(x) + " ";
    return s;
}

void check_generation(Gpt2Engine& eng, const std::vector<int>& prompt,
                      const std::vector<int>& ref_tokens, const Lgt1& ref_logits,
                      bool fused, bool kv_cache, const std::string& label) {
    GenerateOptions opt;
    opt.max_new_tokens = static_cast<int>(ref_tokens.size());
    opt.fused_attn = fused;
    opt.use_kv_cache = kv_cache;
    opt.keep_logits = true;
    const GenerateResult r = eng.generate(prompt, opt);
    const bool same = r.tokens == ref_tokens;
    check(label + ": greedy tokens equal reference", same,
          same ? "" : ("\n    got: " + ids_str(r.tokens) + "\n    ref: " + ids_str(ref_tokens)));
    const size_t n = std::min(r.logits.size(), ref_logits.data.size());
    check_err(label + ": logits rows vs ref_greedy.bin", max_abs(r.logits.data(), ref_logits.data.data(), n), 1e-3);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: test_llm_e2e <tiny model dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    try {
        const std::vector<int> prompt     = read_ids(dir + "/prompt_ids.txt");
        const std::vector<int> ref_tokens = read_ids(dir + "/ref_greedy_tokens.txt");
        const Lgt1 ref_all    = read_lgt1(dir + "/ref_prefill_all.bin");
        const Lgt1 ref_greedy = read_lgt1(dir + "/ref_greedy.bin");

        Gpt2Engine eng(dir);
        const int V = eng.config().vocab_size;
        const int P = static_cast<int>(prompt.size());
        const int G = static_cast<int>(ref_tokens.size());
        std::printf("test_llm_e2e: P=%d G=%d V=%d L=%d d=%d H=%d gemm=%s platform=%s\n", P, G, V,
                    eng.config().n_layer, eng.config().n_embd, eng.config().n_head,
                    Gpt2Engine::gemm_kernel_name(), HIPGPT2_PLATFORM_NAME);
        check("reference shapes consistent",
              ref_all.rows == P && ref_all.vocab == V && ref_greedy.rows == G && ref_greedy.vocab == V);
        if (g_failures) return 1;

        // (a) prefill logits at every prompt position.
        std::vector<float> all(static_cast<size_t>(P) * V), last(V);
        eng.reset();
        eng.prefill(prompt.data(), P, last.data(), all.data());
        check_err("(a) prefill logits, all positions", max_abs(all.data(), ref_all.data.data(), all.size()), 1e-3);
        check_err("(a) last-row logits == row P-1 of all-row logits",
                  max_abs(last.data(), all.data() + static_cast<size_t>(P - 1) * V, V), 1e-6);

        // (b), (c) greedy generation with the KV-cache, both decode attention kernels.
        check_generation(eng, prompt, ref_tokens, ref_greedy, true, true, "(b) fused");
        check_generation(eng, prompt, ref_tokens, ref_greedy, false, true, "(c) naive");

        // (d) prompt one token at a time through the decode path vs prefill's last row.
        for (bool fused : {true, false}) {
            std::vector<float> step(V);
            eng.reset();
            for (int t = 0; t < P; ++t) eng.decode(prompt[t], fused, step.data());
            const double e = max_abs(step.data(), all.data() + static_cast<size_t>(P - 1) * V, V);
            check_err(std::string("(d) token-by-token decode vs prefill, ") + (fused ? "fused" : "naive"), e, 1e-3);
        }

        // (e) no KV-cache: recompute the full sequence each step.
        check_generation(eng, prompt, ref_tokens, ref_greedy, true, false, "(e) no-kv-cache");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_llm_e2e: error: %s\n", e.what());
        return 1;
    }
    std::printf("SUMMARY: %d failed\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
