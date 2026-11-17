// gpt2 — command-line front end of the engine (SPEC 6.5).
//   gpt2 --model DIR (--prompt TEXT | --prompt-file F | --tokens 1,2,3) [options]
// Exit codes: 0 ok, 2 usage error, 1 runtime error.
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "engine.hpp"
#include "tokenizer.hpp"

using namespace hipgpt2;

namespace {

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

const char* kUsage =
    "usage: gpt2 --model DIR (--prompt TEXT | --prompt-file F | --tokens 1,2,3)\n"
    "            [--vocab-dir DIR] [--max-new-tokens N] [--temperature F] [--top-k K]\n"
    "            [--seed S] [--stop-at-eot] [--attn naive|fused] [--no-kv-cache]\n"
    "            [--warmup N] [--dump-logits FILE] [--dump-mode last|all]\n"
    "            [--bench-out DIR] [--quiet]\n";

struct Args {
    std::string model, vocab_dir, prompt, prompt_file, tokens, dump_logits, bench_out;
    std::string dump_mode = "last", attn = "fused";
    bool has_prompt = false, has_prompt_file = false, has_tokens = false;
    int max_new_tokens = 64, top_k = 40, warmup = 0;
    float temperature = 0.f;
    uint64_t seed = 1234;
    bool stop_at_eot = false, no_kv_cache = false, quiet = false;
};

int parse_int(const std::string& flag, const std::string& v) {
    try {
        size_t pos = 0;
        const int x = std::stoi(v, &pos);
        if (pos != v.size()) throw std::invalid_argument("trailing");
        return x;
    } catch (const std::exception&) {
        throw UsageError(flag + ": expected an integer, got '" + v + "'");
    }
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw UsageError(f + " needs a value");
            return argv[++i];
        };
        if (f == "--model") a.model = val();
        else if (f == "--vocab-dir") a.vocab_dir = val();
        else if (f == "--prompt") { a.prompt = val(); a.has_prompt = true; }
        else if (f == "--prompt-file") { a.prompt_file = val(); a.has_prompt_file = true; }
        else if (f == "--tokens") { a.tokens = val(); a.has_tokens = true; }
        else if (f == "--max-new-tokens") a.max_new_tokens = parse_int(f, val());
        else if (f == "--temperature") {
            const std::string v = val();
            try { a.temperature = std::stof(v); } catch (const std::exception&) { throw UsageError(f + ": bad number"); }
        }
        else if (f == "--top-k") a.top_k = parse_int(f, val());
        else if (f == "--seed") {
            const std::string v = val();
            try { a.seed = std::stoull(v); } catch (const std::exception&) { throw UsageError(f + ": bad number"); }
        }
        else if (f == "--stop-at-eot") a.stop_at_eot = true;
        else if (f == "--attn") a.attn = val();
        else if (f == "--no-kv-cache") a.no_kv_cache = true;
        else if (f == "--warmup") a.warmup = parse_int(f, val());
        else if (f == "--dump-logits") a.dump_logits = val();
        else if (f == "--dump-mode") a.dump_mode = val();
        else if (f == "--bench-out") a.bench_out = val();
        else if (f == "--quiet") a.quiet = true;
        else if (f == "--help" || f == "-h") { std::fputs(kUsage, stdout); std::exit(0); }
        else throw UsageError("unknown flag '" + f + "'");
    }
    if (a.model.empty()) throw UsageError("--model is required");
    const int sources = int(a.has_prompt) + int(a.has_prompt_file) + int(a.has_tokens);
    if (sources != 1) throw UsageError("exactly one of --prompt, --prompt-file, --tokens is required");
    if (a.attn != "naive" && a.attn != "fused") throw UsageError("--attn must be naive or fused");
    if (a.dump_mode != "last" && a.dump_mode != "all") throw UsageError("--dump-mode must be last or all");
    if (a.max_new_tokens < 1) throw UsageError("--max-new-tokens must be >= 1");
    if (a.warmup < 0) throw UsageError("--warmup must be >= 0");
    if (a.temperature < 0.f) throw UsageError("--temperature must be >= 0");
    return a;
}

bool file_exists(const std::string& p) {
    struct stat sb;
    return ::stat(p.c_str(), &sb) == 0;
}

std::vector<int> parse_token_list(const std::string& s) {
    std::vector<int> ids;
    std::string cur;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == ',') {
            if (!cur.empty()) ids.push_back(parse_int("--tokens", cur));
            cur.clear();
        } else if (!std::isspace(static_cast<unsigned char>(s[i]))) {
            cur += s[i];
        }
    }
    return ids;
}

// Streaming decode output: tokens are raw bytes and a multi-byte character may be split across
// tokens. Removes and returns the longest prefix of `pending` that consists of complete UTF-8
// sequences; a trailing incomplete sequence stays buffered unless `flush`. Bytes that can never
// form a valid sequence (stray continuation bytes, bad lead bytes) become U+FFFD.
std::string drain_utf8(std::string& pending, bool flush) {
    static const char kReplacement[] = "\xEF\xBF\xBD";
    std::string out;
    size_t i = 0;
    while (i < pending.size()) {
        const unsigned char c = static_cast<unsigned char>(pending[i]);
        size_t len = 0;
        if (c < 0x80) len = 1;
        else if (c >= 0xC2 && c <= 0xDF) len = 2;
        else if (c >= 0xE0 && c <= 0xEF) len = 3;
        else if (c >= 0xF0 && c <= 0xF4) len = 4;
        size_t have = 1;
        while (len > 1 && have < len && i + have < pending.size() &&
               (static_cast<unsigned char>(pending[i + have]) & 0xC0) == 0x80)
            ++have;
        if (len != 0 && have == len) {
            out.append(pending, i, len);
            i += len;
        } else if (len > 1 && i + have == pending.size() && !flush) {
            break;  // incomplete sequence at the end: wait for the next token
        } else {
            out += kReplacement;
            i += 1;
        }
    }
    pending.erase(0, i);
    return out;
}

std::string device_name() {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return "unknown";
    std::string n = prop.name;
    return n.empty() ? std::string("unknown") : n;
}

std::string slugify(const std::string& s) {
    std::string out = s;
    for (char& c : out)
        if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
    return out;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

double finite(double v) { return std::isfinite(v) ? v : 0.0; }

std::string utc_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&now, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string basename_of(std::string p) {
    while (p.size() > 1 && p.back() == '/') p.pop_back();
    const size_t k = p.find_last_of('/');
    return k == std::string::npos ? p : p.substr(k + 1);
}

void write_lgt1(const std::string& path, const std::vector<float>& data, int rows, int vocab) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
    const int32_t hdr[2] = {rows, vocab};
    f.write("LGT1", 4);
    f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);  // little-endian host assumed
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(static_cast<size_t>(rows) * vocab * sizeof(float)));
    if (!f) throw std::runtime_error("write failed: '" + path + "'");
}

int run(const Args& a) {
    // Tokenizer: required for text prompts, optional (for pretty output) with --tokens.
    std::unique_ptr<Gpt2Tokenizer> tok;
    {
        std::vector<std::string> dirs;
        if (!a.vocab_dir.empty()) dirs.push_back(a.vocab_dir);
        else { dirs.push_back(a.model); dirs.push_back("assets/gpt2"); }
        for (const auto& dir : dirs) {
            if (file_exists(dir + "/encoder.json") && file_exists(dir + "/vocab.bpe")) {
                tok.reset(new Gpt2Tokenizer(Gpt2Tokenizer::load_dir(dir)));
                break;
            }
        }
        if (!tok && !a.has_tokens)
            throw std::runtime_error("no encoder.json/vocab.bpe found (use --vocab-dir or --tokens)");
    }

    Gpt2Engine engine(a.model);
    const Gpt2Config& cfg = engine.config();
    const int V = cfg.vocab_size, C = cfg.n_ctx;

    std::vector<int> prompt;
    if (a.has_tokens) {
        prompt = parse_token_list(a.tokens);
    } else {
        std::string text = a.prompt;
        if (a.has_prompt_file) {
            std::ifstream f(a.prompt_file, std::ios::binary);
            if (!f) throw std::runtime_error("cannot read prompt file '" + a.prompt_file + "'");
            std::stringstream ss;
            ss << f.rdbuf();
            text = ss.str();
        }
        prompt = tok->encode(text);
    }
    if (prompt.empty()) {
        if (V > kEndOfTextId) prompt.push_back(kEndOfTextId);
        else throw std::runtime_error("empty prompt and the model has no <|endoftext|> token");
    }
    for (int id : prompt)
        if (id < 0 || id >= V) throw std::runtime_error("prompt token id " + std::to_string(id) + " out of range");
    const int P = static_cast<int>(prompt.size());
    if (P > C) throw std::runtime_error("prompt has " + std::to_string(P) + " tokens, n_ctx is " + std::to_string(C));

    int G = a.max_new_tokens;
    if (a.dump_mode == "last" && G > C - P + 1) {
        std::fprintf(stderr, "warning: --max-new-tokens %d clamped to %d (n_ctx %d, prompt %d)\n",
                     G, C - P + 1, C, P);
        G = C - P + 1;
    }
    const bool fused = (a.attn == "fused") && engine.fused_attn_supported();
    if (a.attn == "fused" && !fused)
        std::fprintf(stderr, "warning: n_ctx %d too large for fused attention, using naive\n", C);

    GenerateResult res;
    std::vector<float> dump;
    int dump_rows = 0;
    if (a.dump_mode == "all") {
        // Logits at every prompt position from one prefill; generation is skipped.
        dump.resize(static_cast<size_t>(P) * V);
        engine.reset();
        const auto t0 = std::chrono::steady_clock::now();
        res.prefill_ms = engine.prefill(prompt.data(), P, nullptr, dump.data());
        res.ttft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        dump_rows = P;
    } else {
        GenerateOptions opt;
        opt.max_new_tokens = G;
        opt.fused_attn = fused;
        opt.use_kv_cache = !a.no_kv_cache;
        opt.stop_at_eot = a.stop_at_eot;
        opt.warmup = a.warmup;
        opt.keep_logits = !a.dump_logits.empty();
        opt.sampling.temperature = a.temperature;
        opt.sampling.top_k = a.top_k;
        opt.sampling.seed = a.seed;

        std::string pending;  // bytes of an incomplete UTF-8 sequence
        auto on_token = [&](int id) {
            if (a.quiet) return;
            if (!tok) { std::printf("%d ", id); std::fflush(stdout); return; }
            try { pending += tok->decode_token(id); }
            catch (const std::exception&) { pending += "<" + std::to_string(id) + ">"; }
            const std::string ready = drain_utf8(pending, false);
            if (!ready.empty()) {
                std::fwrite(ready.data(), 1, ready.size(), stdout);
                std::fflush(stdout);
            }
        };
        res = engine.generate(prompt, opt, on_token);
        if (!a.quiet) {
            const std::string rest = drain_utf8(pending, true);
            std::fwrite(rest.data(), 1, rest.size(), stdout);
            std::printf("\n");
        }
        dump = std::move(res.logits);
        dump_rows = static_cast<int>(res.tokens.size());
    }

    if (!a.dump_logits.empty()) write_lgt1(a.dump_logits, dump, dump_rows, V);

    const int gen = static_cast<int>(res.tokens.size());
    std::printf("prompt_tokens: %d\n", P);
    std::printf("gen_tokens: %d\n", gen);
    std::printf("load_ms: %.3f\n", engine.load_ms());
    std::printf("ttft_ms: %.3f\n", res.ttft_ms);
    std::printf("decode_tok_per_s: %.3f\n", res.decode_tok_per_s);

    if (!a.bench_out.empty()) {
        ::mkdir(a.bench_out.c_str(), 0755);  // ok if it already exists
        const std::string dev = device_name();
        const std::string path = a.bench_out + "/llm_" + slugify(dev) + ".json";
        std::ofstream f(path);
        if (!f) throw std::runtime_error("cannot write '" + path + "'");
        char nums[512];
        std::snprintf(nums, sizeof nums,
                      "\"warmup_runs\": %d, \"prompt_tokens\": %d, \"gen_tokens\": %d, "
                      "\"load_ms\": %.3f, \"prefill_ms\": %.3f, \"ttft_ms\": %.3f, "
                      "\"decode_tok_per_s\": %.3f",
                      a.warmup, P, gen, finite(engine.load_ms()), finite(res.prefill_ms),
                      finite(res.ttft_ms), finite(res.decode_tok_per_s));
        f << "{\"schema_version\": 1, \"bench\": \"llm\", \"engine\": \"hip-gpt2\", "
          << "\"model\": \"" << json_escape(basename_of(a.model)) << "\", "
          << "\"platform\": \"" << HIPGPT2_PLATFORM_NAME << "\", "
          << "\"device\": \"" << json_escape(dev) << "\", "
          << "\"git_sha\": \"" << json_escape(HIPGPT2_GIT_SHA) << "\", "
          << "\"timestamp\": \"" << utc_timestamp() << "\", "
          << "\"attn\": \"" << (fused ? "fused" : "naive") << "\", "
          << "\"gemm_kernel\": \"" << Gpt2Engine::gemm_kernel_name() << "\", "
          << "\"sampling\": \"" << (a.temperature > 0.f ? "top_k" : "greedy") << "\", "
          << nums << "}\n";
        if (!f) throw std::runtime_error("write failed: '" + path + "'");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const UsageError& e) {
        std::fprintf(stderr, "gpt2: %s\n%s", e.what(), kUsage);
        return 2;
    }
    try {
        return run(args);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gpt2: error: %s\n", e.what());
        return 1;
    }
}
