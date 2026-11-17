// GEMM benchmark + correctness harness. Same source builds for AMD and NVIDIA.
//
//   gemm_bench --kernel v0 --test                 correctness on awkward sizes, exit 1 on failure
//   gemm_bench --kernel v0                        benchmark default sweep 512..8192, write JSON
//   gemm_bench --kernel all --sizes 1024,2048     benchmark every registered kernel
//   gemm_bench --kernel v0 --sizes 4096x1024x512  non-square: MxNxK
//
// Options: --out <dir> (default results)   --iters <n> (default: adaptive)

#include "gemm.hpp"
#include "reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef GEMM_GIT_SHA
#define GEMM_GIT_SHA "unknown"
#endif

#if defined(__HIP_PLATFORM_AMD__)
static const char* kPlatform = "amd";
#elif defined(__HIP_PLATFORM_NVIDIA__)
static const char* kPlatform = "nvidia";
#else
static const char* kPlatform = "unknown";
#endif

struct KernelEntry { const char* name; GemmLaunchFn fn; };

// Register each new stage here.
static const KernelEntry kKernels[] = {
    {"v0", launch_v0},
};

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
    }
    return out;
}

static Result run_one(const KernelEntry& ke, Shape s, int forced_iters, bool time_it) {
    const size_t nA = (size_t)s.M * s.K, nB = (size_t)s.K * s.N, nC = (size_t)s.M * s.N;
    std::vector<float> hA(nA), hB(nB), hC(nC);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, nA * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, nB * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC, nC * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), nA * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), nB * sizeof(float), hipMemcpyHostToDevice));
    // Poison C with NaN so a kernel that skips elements can't pass by accident.
    HIP_CHECK(hipMemset(dC, 0xFF, nC * sizeof(float)));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Correctness run.
    ke.fn(dA, dB, dC, s.M, s.N, s.K, stream);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(hC.data(), dC, nC * sizeof(float), hipMemcpyDeviceToHost));

    Result r;
    r.s = s;
    CheckResult cr = check_gemm(hA.data(), hB.data(), hC.data(), s.M, s.N, s.K);
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
            ke.fn(dA, dB, dC, s.M, s.N, s.K, stream);
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
        r.gflops = 2.0 * s.M * s.N * s.K / (r.ms_median * 1e6);
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
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o;
}

static void write_json(const std::string& dir, const KernelEntry& ke, const char* device,
                       const std::vector<Result>& rs) {
    std::string safe_dev;
    for (const char* p = device; *p; ++p) safe_dev += std::isalnum((unsigned char)*p) ? *p : '_';
    std::string path = dir + "/" + ke.name + "_fp32_" + safe_dev + ".json";
    std::ofstream f(path);
    if (!f) { std::fprintf(stderr, "cannot write %s (does the directory exist?)\n", path.c_str()); return; }
    char ts[32];
    std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    f << "{\n  \"kernel\": \"" << ke.name << "\",\n  \"dtype\": \"fp32\",\n"
      << "  \"platform\": \"" << kPlatform << "\",\n  \"device\": \"" << json_escape(device) << "\",\n"
      << "  \"git_sha\": \"" << GEMM_GIT_SHA << "\",\n  \"timestamp\": \"" << ts << "\",\n"
      << "  \"results\": [\n";
    for (size_t i = 0; i < rs.size(); ++i) {
        const Result& r = rs[i];
        f << "    {\"M\": " << r.s.M << ", \"N\": " << r.s.N << ", \"K\": " << r.s.K
          << ", \"ms_median\": " << r.ms_median << ", \"ms_min\": " << r.ms_min
          << ", \"gflops\": " << r.gflops << ", \"iters\": " << r.iters
          << ", \"max_err\": " << r.max_err << ", \"verified\": " << (r.verified ? "true" : "false")
          << ", \"full_check\": " << (r.full_check ? "true" : "false") << "}"
          << (i + 1 < rs.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    std::printf("  wrote %s\n", path.c_str());
}

int main(int argc, char** argv) {
    std::string kernel = "all", sizes_arg, out_dir = "results";
    bool test_mode = false;
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
        else if (a == "--iters") iters = std::stoi(next());
        else if (a == "--test") test_mode = true;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s  platform: %s  warpSize: %d  SMs/CUs: %d\n",
                prop.name, kPlatform, prop.warpSize, prop.multiProcessorCount);

    // Test sizes are deliberately awkward: 1, primes, non-multiples of any tile size.
    std::vector<Shape> shapes = !sizes_arg.empty() ? parse_sizes(sizes_arg)
        : test_mode ? std::vector<Shape>{{1, 1, 1}, {17, 17, 17}, {127, 127, 127}, {128, 128, 128},
                                         {333, 257, 129}, {64, 1000, 3}, {512, 512, 512}, {1031, 517, 263}}
                    : parse_sizes("512,1024,2048,4096,8192");

    bool any = false, all_ok = true;
    for (const auto& ke : kKernels) {
        if (kernel != "all" && kernel != ke.name) continue;
        any = true;
        std::printf("[%s]\n", ke.name);
        std::vector<Result> rs;
        for (Shape s : shapes) {
            Result r = run_one(ke, s, iters, !test_mode);
            all_ok &= r.verified;
            if (test_mode)
                std::printf("  %5dx%-5dx%-5d %s (max_err %.2e)\n", s.M, s.N, s.K,
                            r.verified ? "PASS" : "FAIL", r.max_err);
            else
                std::printf("  %5dx%-5dx%-5d %9.3f ms  %9.1f GFLOPS  %s\n", s.M, s.N, s.K,
                            r.ms_median, r.gflops, r.verified ? "ok" : "WRONG");
            rs.push_back(r);
        }
        if (!test_mode) write_json(out_dir, ke, prop.name, rs);
    }
    if (!any) { std::fprintf(stderr, "no kernel named '%s'\n", kernel.c_str()); return 2; }
    return all_ok ? 0 : 1;
}
