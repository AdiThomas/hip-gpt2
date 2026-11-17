// Usage: test_safetensors <tiny_model_dir>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "gpt2_config.hpp"
#include "safetensors.hpp"

using namespace hipgpt2;

static int g_fail = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s (line %d)\n", std::string(msg).c_str(), __LINE__); ++g_fail; } } while (0)

template <class F> static bool throws(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }

static std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
static void write_file(const std::string& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary);
    f.write(s.data(), (std::streamsize)s.size());
}

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <model_dir>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    try {
        Gpt2Config cfg = load_gpt2_config(dir);
        const std::string path = dir + "/model.safetensors";
        SafeTensors st = SafeTensors::open(path);
        const int64_t d = cfg.n_embd, V = cfg.vocab_size, C = cfg.n_ctx;
        std::map<std::string, std::vector<int64_t>> want;
        want["wte.weight"] = {V, d};
        want["wpe.weight"] = {C, d};
        want["ln_f.weight"] = {d};
        want["ln_f.bias"] = {d};
        for (int i = 0; i < cfg.n_layer; ++i) {
            std::string p = "h." + std::to_string(i) + ".";
            want[p + "ln_1.weight"] = {d}; want[p + "ln_1.bias"] = {d};
            want[p + "attn.c_attn.weight"] = {d, 3 * d}; want[p + "attn.c_attn.bias"] = {3 * d};
            want[p + "attn.c_proj.weight"] = {d, d}; want[p + "attn.c_proj.bias"] = {d};
            want[p + "ln_2.weight"] = {d}; want[p + "ln_2.bias"] = {d};
            want[p + "mlp.c_fc.weight"] = {d, 4 * d}; want[p + "mlp.c_fc.bias"] = {4 * d};
            want[p + "mlp.c_proj.weight"] = {4 * d, d}; want[p + "mlp.c_proj.bias"] = {d};
        }
        std::vector<std::string> names = st.names();
        CHECK(names.size() == want.size(), "tensor count matches config");
        for (size_t i = 1; i < names.size(); ++i) CHECK(names[i - 1] < names[i], "names() sorted");
        for (const auto& kv : want) {
            CHECK(st.has(kv.first), "missing tensor " + kv.first);
            if (!st.has(kv.first)) continue;
            const TensorView& tv = st.get(kv.first);
            CHECK(tv.shape == kv.second, "shape of " + kv.first);
            CHECK(tv.dtype == "F32", "dtype F32 of " + kv.first);
            CHECK(tv.nbytes == (size_t)tv.numel() * 4, "nbytes of " + kv.first);
            CHECK(reinterpret_cast<uintptr_t>(tv.data) % alignof(float) == 0, "alignment of " + kv.first);
            const float* p = nullptr;
            try {
                if (kv.second.size() == 1) p = st.f32(kv.first, {kv.second[0]});
                else p = st.f32(kv.first, {kv.second[0], kv.second[1]});
            } catch (const std::exception& e) { CHECK(false, std::string("f32 threw: ") + e.what()); }
            CHECK(p == reinterpret_cast<const float*>(tv.data), "f32 pointer");
        }

        // f32 error paths
        CHECK(throws([&] { st.f32("wte.weight", {V, d + 1}); }), "f32 wrong shape throws");
        CHECK(throws([&] { st.f32("wte.weight", {d, V}); }), "f32 transposed shape throws");
        CHECK(throws([&] { st.f32("wte.weight", {V * d}); }), "f32 wrong rank throws");
        CHECK(throws([&] { st.f32("no.such.tensor", {1}); }), "f32 missing name throws");
        CHECK(throws([&] { st.get("no.such.tensor"); }), "get missing name throws");
        CHECK(!st.has("no.such.tensor"), "has() false for missing");
        {
            SafeTensors copy = st;  // copies share the buffer
            CHECK(copy.get("wte.weight").data == st.get("wte.weight").data, "copy shares buffer");
        }
        try {
            st.f32("wte.weight", {V, d + 1});
        } catch (const std::exception& e) {
            CHECK(std::string(e.what()).find("wte.weight") != std::string::npos, "error names the tensor");
        }

        // open error paths
        CHECK(throws([&] { SafeTensors::open(dir + "/does_not_exist.safetensors"); }), "nonexistent path throws");
        std::string bytes = read_file(path);
        const std::string tmp = "/tmp/hipgpt2_trunc_test.safetensors";
        write_file(tmp, bytes.substr(0, bytes.size() - 7));
        CHECK(throws([&] { SafeTensors::open(tmp); }), "truncated data section throws");
        write_file(tmp, bytes.substr(0, 4));
        CHECK(throws([&] { SafeTensors::open(tmp); }), "file shorter than header length throws");
        write_file(tmp, bytes.substr(0, 8 + 20));
        CHECK(throws([&] { SafeTensors::open(tmp); }), "truncated header throws");
        std::string bad = bytes;
        bad[7] = (char)0x7f;  // absurd header length
        write_file(tmp, bad);
        CHECK(throws([&] { SafeTensors::open(tmp); }), "header length > file size throws");
        std::remove(tmp.c_str());
        try {
            SafeTensors::open(dir + "/does_not_exist.safetensors");
        } catch (const std::exception& e) {
            CHECK(std::string(e.what()).find("does_not_exist") != std::string::npos, "error message includes path");
        }

        // manifest: "<name> <first> <last>"
        std::ifstream mf(dir + "/manifest.txt");
        CHECK((bool)mf, "manifest.txt readable");
        std::string name;
        double first, last;
        int checked = 0;
        while (mf >> name >> first >> last) {
            const TensorView& tv = st.get(name);
            const float* p = reinterpret_cast<const float*>(tv.data);
            CHECK(p[0] == (float)first, "manifest first value of " + name);
            CHECK(p[tv.numel() - 1] == (float)last, "manifest last value of " + name);
            ++checked;
        }
        CHECK(checked == (int)want.size(), "manifest covers all tensors");

        // config error paths
        CHECK(throws([&] { load_gpt2_config("/nonexistent_dir_xyz"); }), "config: missing file throws");
        CHECK(cfg.n_layer == 2 && cfg.n_head == 4 && cfg.n_embd == 64 && cfg.vocab_size == 300 && cfg.n_ctx == 64 &&
                  cfg.head_dim() == 16,
              "config values");
        std::printf("safetensors: %d tensors checked, %d failures\n", checked, g_fail);
    } catch (const std::exception& e) {
        std::printf("FAIL: unexpected exception: %s\n", e.what());
        ++g_fail;
    }
    return g_fail ? 1 : 0;
}
