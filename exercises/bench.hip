// GEMM benchmark + correctness harness. Same source builds for CPU (HIP-CPU), AMD and NVIDIA.
//
//   gemm_bench --list                             every kernel, dtype, availability
//   gemm_bench --kernel v0 --test                 correctness on awkward sizes, exit 1 on failure
//   gemm_bench --kernel v0                        benchmark default sweep, write JSON
//   gemm_bench --kernel all --sizes 1024,2048     benchmark every available kernel
//   gemm_bench --kernel v0 --sizes 4096x1024x512  non-square: MxNxK
//
// Options: --out <dir> (default results)   --iters <n> (default: adaptive)
// Exit codes: 0 ok, 1 a check failed, 2 usage error, 77 explicitly requested kernel is
// unavailable on this platform (CTest's conventional "skipped").

#include "gemm.hpp"
#include "reference.hpp"
#include "kernels/half_host.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// One row of either registry, flattened so the harness treats FP32 and FP16 alike.
struct KernelEntry {
    const char* name;
    bool fp16;                 // input dtype: false = float, true = __half
    GemmLaunchFn fn32;         // set when !fp16 and available
    GemmF16LaunchFn fn16;      // set when fp16 and available
    const char* unavailable;   // non-null => not available, human-readable reason
    double tol;                // check_gemm threshold
    const char* dtype() const { return fp16 ? "fp16" : "fp32"; }
    bool available() const { return unavailable == nullptr; }
};

static std::vector<KernelEntry> all_kernels() {
    std::vector<KernelEntry> out;
    int n32 = 0, n16 = 0;
    const GemmKernel* k32 = gemm_fp32_kernels(&n32);
    const GemmF16Kernel* k16 = gemm_fp16_kernels(&n16);
    for (int i = 0; i < n32; ++i)
        out.push_back({k32[i].name, false, k32[i].fn, nullptr, k32[i].unavailable, k32[i].tol});
    for (int i = 0; i < n16; ++i)
        out.push_back({k16[i].name, true, nullptr, k16[i].fn, k16[i].unavailable, k16[i].tol});
    return out;
}

struct Shape { int M, N, K; };

struct Result {
    Shape s;
    double ms_median = 0, ms_min = 0, gflops = 0, max_err = 0;
    bool verified = false, full_check = false;
    int iters = 0;
};

static std::vector<Shape> parse_sizes(const std::string& arg) {
    std::vector<Shape> out;
    std::stringstream ss(arg);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        int m = 0, n = 0, k = 0;
        if (std::sscanf(tok.c_str(), "%dx%dx%d", &m, &n, &k) == 3) out.push_back({m, n, k});
        else if (std::sscanf(tok.c_str(), "%d", &m) == 1) out.push_back({m, m, m});
        else { std::fprintf(stderr, "bad size '%s'\n", tok.c_str()); std::exit(2); }
        const Shape& s = out.back();
        if (s.M < 1 || s.N < 1 || s.K < 1) { std::fprintf(stderr, "bad size '%s'\n", tok.c_str()); std::exit(2); }
    }
    return out;
}

static double finite_or_zero(double x) { return std::isfinite(x) ? x : 0.0; }

static Result run_one(const KernelEntry& ke, Shape s, int forced_iters, bool time_it) {
    const size_t nA = (size_t)s.M * s.K, nB = (size_t)s.K * s.N, nC = (size_t)s.M * s.N;
    std::vector<float> hA(nA), hB(nB), hC(nC);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    // FP16 kernels: round inputs to binary16 (RNE) and give the reference the *rounded*
    // values, so the check measures only the kernel's accumulation error.
    const size_t elem = ke.fp16 ? sizeof(uint16_t) : sizeof(float);
    std::vector<uint16_t> hA16, hB16;
    if (ke.fp16) {
        hA16.resize(nA);
        hB16.resize(nB);
        for (size_t i = 0; i < nA; ++i) { hA16[i] = f32_to_f16_bits(hA[i]); hA[i] = f16_bits_to_f32(hA16[i]); }
        for (size_t i = 0; i < nB; ++i) { hB16[i] = f32_to_f16_bits(hB[i]); hB[i] = f16_bits_to_f32(hB16[i]); }
    }
    const void* srcA = ke.fp16 ? (const void*)hA16.data() : (const void*)hA.data();
    const void* srcB = ke.fp16 ? (const void*)hB16.data() : (const void*)hB.data();

    void *dA, *dB;
    float* dC;
    HIP_CHECK(hipMalloc(&dA, nA * elem));
    HIP_CHECK(hipMalloc(&dB, nB * elem));
    HIP_CHECK(hipMalloc(&dC, nC * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, srcA, nA * elem, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, srcB, nB * elem, hipMemcpyHostToDevice));
    // Poison C with NaN so a kernel that skips elements can't pass by accident.
    HIP_CHECK(hipMemset(dC, 0xFF, nC * sizeof(float)));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    auto launch = [&]() {
        if (ke.fp16) ke.fn16(static_cast<const __half*>(dA), static_cast<const __half*>(dB), dC, s.M, s.N, s.K, stream);
        else         ke.fn32(static_cast<const float*>(dA), static_cast<const float*>(dB), dC, s.M, s.N, s.K, stream);
    };

    // Correctness run.
    launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(hC.data(), dC, nC * sizeof(float), hipMemcpyDeviceToHost));

    Result r;
    r.s = s;
    CheckResult cr = check_gemm(hA.data(), hB.data(), hC.data(), s.M, s.N, s.K, ke.tol);
    r.verified = cr.ok;
    r.max_err = cr.max_err;
    r.full_check = cr.full;
    if (!cr.ok) {
        long long w = cr.worst_idx;
        std::fprintf(stderr, "  FAIL %s %dx%dx%d: max_err=%.3g at C[%lld][%lld] (got %g)\n",
                     ke.name, s.M, s.N, s.K, cr.max_err, w / s.N, w % s.N, hC[w]);
    }

    if (time_it && cr.ok) {
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        auto time_once = [&]() {
            HIP_CHECK(hipEventRecord(start, stream));
            launch();
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            return (double)ms;
        };
        // Warm-up (first launches pay for clocks ramping, caches, code loading).
        double probe = 0;
        for (int i = 0; i < 3; ++i) probe = time_once();
        // Adaptive: aim for ~1 s of measurement, between 5 and 100 runs.
        int iters = forced_iters > 0 ? forced_iters
                                     : std::clamp((int)std::ceil(1000.0 / std::max(probe, 1e-3)), 5, 100);
        std::vector<double> t(iters);
        for (int i = 0; i < iters; ++i) t[i] = time_once();
        std::sort(t.begin(), t.end());
        r.iters = iters;
        r.ms_min = t.front();
        r.ms_median = t[iters / 2];
        r.gflops = r.ms_median > 0 ? 2.0 * s.M * s.N * s.K / (r.ms_median * 1e6) : 0.0;
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    return r;
}

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

static std::string device_slug(const char* device) {
    std::string s;
    for (const char* p = device; *p; ++p) s += std::isalnum((unsigned char)*p) ? *p : '_';
    return s;
}

// Schema 8.1. Every number is finite (non-finite -> 0, and such a row is never verified).
static void write_json(const std::string& dir, const KernelEntry& ke, const char* device,
                       const std::vector<Result>& rs) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string path = dir + "/gemm_" + ke.name + "_" + device_slug(device) + ".json";
    std::ofstream f(path);
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    f.precision(9);
    char ts[32];
    std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    f << "{\n  \"schema_version\": 1,\n  \"bench\": \"gemm\",\n"
      << "  \"kernel\": \"" << json_escape(ke.name) << "\",\n  \"dtype\": \"" << ke.dtype() << "\",\n"
      << "  \"platform\": \"" << HIPGPT2_PLATFORM_NAME << "\",\n  \"device\": \"" << json_escape(device) << "\",\n"
      << "  \"git_sha\": \"" << json_escape(HIPGPT2_GIT_SHA) << "\",\n  \"timestamp\": \"" << ts << "\",\n"
      << "  \"results\": [\n";
    for (size_t i = 0; i < rs.size(); ++i) {
        const Result& r = rs[i];
        const bool finite = std::isfinite(r.ms_median) && std::isfinite(r.ms_min) &&
                            std::isfinite(r.gflops) && std::isfinite(r.max_err);
        f << "    {\"M\": " << r.s.M << ", \"N\": " << r.s.N << ", \"K\": " << r.s.K
          << ", \"ms_median\": " << finite_or_zero(r.ms_median) << ", \"ms_min\": " << finite_or_zero(r.ms_min)
          << ", \"gflops\": " << finite_or_zero(r.gflops) << ", \"iters\": " << r.iters
          << ", \"max_err\": " << finite_or_zero(r.max_err)
          << ", \"verified\": " << (r.verified && finite ? "true" : "false")
          << ", \"full_check\": " << (r.full_check ? "true" : "false") << "}"
          << (i + 1 < rs.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    std::printf("  wrote %s\n", path.c_str());
}

int main(int argc, char** argv) {
    std::string kernel = "all", sizes_arg, out_dir = "results";
    bool test_mode = false, list_mode = false;
    int iters = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--kernel") kernel = next();
        else if (a == "--sizes") sizes_arg = next();
        else if (a == "--out") out_dir = next();
        else if (a == "--iters") {
            const std::string v = next();
            char* end = nullptr;
            const long n = std::strtol(v.c_str(), &end, 10);
            if (!end || *end || n < 1) { std::fprintf(stderr, "bad --iters '%s'\n", v.c_str()); return 2; }
            iters = (int)n;
        }
        else if (a == "--test") test_mode = true;
        else if (a == "--list") list_mode = true;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }

    const std::vector<KernelEntry> kernels = all_kernels();

    if (list_mode) {
        for (const auto& ke : kernels) {
            if (ke.available()) std::printf("%s %s available\n", ke.name, ke.dtype());
            else std::printf("%s %s unavailable: %s\n", ke.name, ke.dtype(), ke.unavailable);
        }
        return 0;
    }

    if (kernel != "all") {
        bool found = false;
        for (const auto& ke : kernels) found |= (kernel == ke.name);
        if (!found) { std::fprintf(stderr, "no kernel named '%s' (see --list)\n", kernel.c_str()); return 2; }
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    // prop.warpSize is informational only (HIP-CPU reports a host value; never use it for logic).
    std::printf("device: %s  platform: %s  warpSize: %d  SMs/CUs: %d\n",
                prop.name, HIPGPT2_PLATFORM_NAME, prop.warpSize, prop.multiProcessorCount);

    // Test sizes are deliberately awkward: 1, primes, non-multiples of any tile size,
    // degenerate K and very thin matrices. Bigger shapes only where a GPU makes them cheap.
#if defined(HIPGPT2_PLATFORM_CPU)
    const bool on_gpu = false;
#else
    const bool on_gpu = true;
#endif
    std::vector<Shape> test_shapes = {{1, 1, 1},       {1, 7, 5},     {17, 17, 17},
                                      {127, 127, 127}, {128, 128, 128}, {333, 257, 129},
                                      {64, 1000, 3},   {3, 5, 1000},  {130, 66, 255}};
    if (on_gpu) {
        test_shapes.push_back({512, 512, 512});
        test_shapes.push_back({1031, 517, 263});
        test_shapes.push_back({2049, 1023, 515});
    }
    const std::vector<Shape> shapes = !sizes_arg.empty() ? parse_sizes(sizes_arg)
                                    : test_mode          ? test_shapes
                                    : parse_sizes(on_gpu ? "512,1024,2048,4096,8192" : "64,128,256");

    int passed = 0, failed = 0, skipped = 0;
    for (const auto& ke : kernels) {
        if (kernel != "all" && kernel != ke.name) continue;
        if (!ke.available()) {
            std::printf("  SKIP %s: not available on this platform (%s)\n", ke.name, ke.unavailable);
            if (kernel != "all") return 77;
            ++skipped;
            continue;
        }
        std::printf("[%s %s]\n", ke.name, ke.dtype());
        std::vector<Result> rs;
        bool kernel_ok = true;
        for (Shape s : shapes) {
            Result r = run_one(ke, s, iters, !test_mode);
            kernel_ok &= r.verified;
            if (test_mode)
                std::printf("  %dx%dx%d %s (max_err %.2e)\n", s.M, s.N, s.K,
                            r.verified ? "PASS" : "FAIL", r.max_err);
            else
                std::printf("  %5dx%-5dx%-5d %9.3f ms  %9.1f GFLOPS  %s\n", s.M, s.N, s.K,
                            r.ms_median, r.gflops, r.verified ? "ok" : "WRONG");
            rs.push_back(r);
        }
        kernel_ok ? ++passed : ++failed;
        if (!test_mode) write_json(out_dir, ke, prop.name, rs);
    }
    // Counts are per kernel: a kernel passes only if every shape passes.
    std::printf("SUMMARY: %d passed, %d failed, %d skipped\n", passed, failed, skipped);
    return failed == 0 ? 0 : 1;
}
