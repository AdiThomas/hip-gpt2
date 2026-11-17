// Unit tests for every GPT-2 engine kernel (SPEC 6.3, acceptance 7 [LLM-CORE] 1).
// Each kernel is compared with a plain C++ (double-precision) reference on deliberately awkward
// sizes (d = 48, H = 3, T in {1, 2, 13}, V = 301, nothing a multiple of a block size).
// Tolerances: max abs error <= 1e-5, or 1e-4 for linear layers with K >= 256;
// fused vs naive attention <= 1e-5.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "kernels/llm_kernels.hpp"

namespace {

int g_failures = 0;

std::vector<float> rand_vec(size_t n, std::mt19937& rng, float lo = -1.f, float hi = 1.f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

template <typename T>
struct DevBuf {
    T* p = nullptr;
    size_t n = 0;
    explicit DevBuf(size_t count) : n(count) { HIP_CHECK(hipMalloc(&p, (count ? count : 1) * sizeof(T))); }
    explicit DevBuf(const std::vector<T>& h) : DevBuf(h.size()) { upload(h); }
    ~DevBuf() { (void)hipFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    void upload(const std::vector<T>& h) {
        HIP_CHECK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<T> h(n);
        HIP_CHECK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
        return h;
    }
};

double max_abs_err(const std::vector<float>& got, const std::vector<double>& ref) {
    double e = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
        if (!(d <= e)) e = d;  // also propagates NaN as a failure
    }
    return e;
}

void report(const std::string& name, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("  %-52s max_err %.3e (tol %.0e) %s\n", name.c_str(), err, tol, ok ? "PASS" : "FAIL");
    if (!ok) ++g_failures;
}

double gelu_ref(double z) {
    return 0.5 * z * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (z + 0.044715 * z * z * z)));
}

// ---------------------------------------------------------------------------------------------
void test_embedding(std::mt19937& rng) {
    const int V = 301, C = 40, d = 48;
    auto wte = rand_vec(static_cast<size_t>(V) * d, rng);
    auto wpe = rand_vec(static_cast<size_t>(C) * d, rng);
    DevBuf<float> dwte(wte), dwpe(wpe);
    for (int T : {1, 2, 13}) {
        for (int pos0 : {0, 5}) {
            std::vector<int> tok(T);
            for (int t = 0; t < T; ++t) tok[t] = static_cast<int>((t * 97 + 300) % V);
            DevBuf<int> dtok(tok);
            DevBuf<float> dx(static_cast<size_t>(T) * d);
            launch_embedding(dtok.p, dwte.p, dwpe.p, dx.p, T, d, pos0, 0);
            std::vector<double> ref(static_cast<size_t>(T) * d);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < d; ++j)
                    ref[t * d + j] = static_cast<double>(wte[tok[t] * d + j]) + wpe[(pos0 + t) * d + j];
            report("embedding T=" + std::to_string(T) + " pos0=" + std::to_string(pos0),
                   max_abs_err(dx.download(), ref), 1e-5);
        }
    }
}

void test_layernorm(std::mt19937& rng) {
    for (int d : {48, 769}) {
        for (int T : {1, 2, 13}) {
            auto x = rand_vec(static_cast<size_t>(T) * d, rng, -3.f, 3.f);
            auto g = rand_vec(d, rng, 0.5f, 1.5f);
            auto b = rand_vec(d, rng);
            DevBuf<float> dx(x), dg(g), db(b), dy(static_cast<size_t>(T) * d);
            launch_layernorm(dx.p, dg.p, db.p, dy.p, T, d, 0);
            std::vector<double> ref(static_cast<size_t>(T) * d);
            for (int t = 0; t < T; ++t) {
                double mean = 0, var = 0;
                for (int j = 0; j < d; ++j) mean += x[t * d + j];
                mean /= d;
                for (int j = 0; j < d; ++j) var += (x[t * d + j] - mean) * (x[t * d + j] - mean);
                var /= d;
                const double rstd = 1.0 / std::sqrt(var + 1e-5);
                for (int j = 0; j < d; ++j) ref[t * d + j] = (x[t * d + j] - mean) * rstd * g[j] + b[j];
            }
            report("layernorm d=" + std::to_string(d) + " T=" + std::to_string(T),
                   max_abs_err(dy.download(), ref), 1e-5);
            // In-place variant (the engine never relies on it, but the header allows it).
            launch_layernorm(dx.p, dg.p, db.p, dx.p, T, d, 0);
            report("layernorm in-place d=" + std::to_string(d) + " T=" + std::to_string(T),
                   max_abs_err(dx.download(), ref), 1e-5);
        }
    }
}

void test_gemv_xw(std::mt19937& rng) {
    const int shapes[][2] = {{48, 144}, {48, 191}, {300, 77}, {769, 1031}};
    for (const auto& s : shapes) {
        const int K = s[0], N = s[1];
        auto x = rand_vec(K, rng);
        auto W = rand_vec(static_cast<size_t>(K) * N, rng);
        auto b = rand_vec(N, rng);
        DevBuf<float> dx(x), dW(W), db(b), dy(N);
        for (int gelu : {0, 1}) {
            launch_gemv_xw(dx.p, dW.p, db.p, dy.p, K, N, gelu, 0);
            std::vector<double> ref(N);
            for (int n = 0; n < N; ++n) {
                double acc = b[n];
                for (int k = 0; k < K; ++k) acc += static_cast<double>(x[k]) * W[static_cast<size_t>(k) * N + n];
                ref[n] = gelu ? gelu_ref(acc) : acc;
            }
            report("gemv_xw K=" + std::to_string(K) + " N=" + std::to_string(N) + (gelu ? " +gelu" : ""),
                   max_abs_err(dy.download(), ref), K >= 256 ? 1e-4 : 1e-5);
        }
    }
}

void test_gemv_rowdot(std::mt19937& rng) {
    const int V = 301;
    for (int d : {48, 300}) {
        auto W = rand_vec(static_cast<size_t>(V) * d, rng);
        DevBuf<float> dW(W);
        for (int T : {1, 2, 13}) {
            auto x = rand_vec(static_cast<size_t>(T) * d, rng);
            DevBuf<float> dx(x), dy(static_cast<size_t>(T) * V);
            launch_gemv_rowdot(dW.p, dx.p, dy.p, V, d, T, 0);
            std::vector<double> ref(static_cast<size_t>(T) * V);
            for (int t = 0; t < T; ++t)
                for (int v = 0; v < V; ++v) {
                    double acc = 0;
                    for (int j = 0; j < d; ++j) acc += static_cast<double>(W[v * d + j]) * x[t * d + j];
                    ref[t * V + v] = acc;
                }
            report("gemv_rowdot V=301 d=" + std::to_string(d) + " T=" + std::to_string(T),
                   max_abs_err(dy.download(), ref), d >= 256 ? 1e-4 : 1e-5);
        }
    }
}

void test_bias(std::mt19937& rng) {
    const int N = 191;
    for (int T : {1, 2, 13}) {
        auto Y = rand_vec(static_cast<size_t>(T) * N, rng, -3.f, 3.f);
        auto b = rand_vec(N, rng);
        DevBuf<float> db(b);
        for (int gelu : {0, 1}) {
            DevBuf<float> dY(Y);
            if (gelu) launch_bias_gelu(dY.p, db.p, T, N, 0);
            else      launch_bias_add(dY.p, db.p, T, N, 0);
            std::vector<double> ref(Y.size());
            for (int t = 0; t < T; ++t)
                for (int n = 0; n < N; ++n) {
                    const double v = static_cast<double>(Y[t * N + n]) + b[n];
                    ref[t * N + n] = gelu ? gelu_ref(v) : v;
                }
            report(std::string(gelu ? "bias_gelu" : "bias_add") + " T=" + std::to_string(T) + " N=191",
                   max_abs_err(dY.download(), ref), 1e-5);
        }
    }
}

void test_fallback_gemm(std::mt19937& rng) {
    const int shapes[][3] = {{1, 7, 5}, {13, 144, 48}, {2, 191, 48}, {33, 65, 300}};
    for (const auto& s : shapes) {
        const int M = s[0], N = s[1], K = s[2];
        auto A = rand_vec(static_cast<size_t>(M) * K, rng);
        auto B = rand_vec(static_cast<size_t>(K) * N, rng);
        DevBuf<float> dA(A), dB(B), dC(static_cast<size_t>(M) * N);
        launch_llm_gemm_fallback(dA.p, dB.p, dC.p, M, N, K, 0);
        std::vector<double> ref(static_cast<size_t>(M) * N);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                for (int k = 0; k < K; ++k) acc += static_cast<double>(A[m * K + k]) * B[k * N + n];
                ref[m * N + n] = acc;
            }
        report("fallback_gemm " + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K),
               max_abs_err(dC.download(), ref), K >= 256 ? 1e-4 : 1e-5);
    }
}

void test_elementwise(std::mt19937& rng) {
    const int n = 1031;
    auto x = rand_vec(n, rng, -4.f, 4.f);
    auto r = rand_vec(n, rng);
    {
        DevBuf<float> dx(x), dr(r);
        launch_residual_add(dx.p, dr.p, n, 0);
        std::vector<double> ref(n);
        for (int i = 0; i < n; ++i) ref[i] = static_cast<double>(x[i]) + r[i];
        report("residual_add n=1031", max_abs_err(dx.download(), ref), 1e-5);
    }
    {
        DevBuf<float> dx(x);
        launch_gelu(dx.p, n, 0);
        std::vector<double> ref(n);
        for (int i = 0; i < n; ++i) ref[i] = gelu_ref(x[i]);
        report("gelu n=1031", max_abs_err(dx.download(), ref), 1e-5);
    }
}

void test_kv_store(std::mt19937& rng) {
    const int d = 48, C = 40;
    for (int T : {1, 2, 13}) {
        const int pos0 = (T == 1) ? 27 : 5;
        auto qkv = rand_vec(static_cast<size_t>(T) * 3 * d, rng);
        std::vector<float> sentinel(static_cast<size_t>(C) * d, 7.0f);
        DevBuf<float> dqkv(qkv), dK(sentinel), dV(sentinel);
        launch_kv_store(dqkv.p, dK.p, dV.p, T, d, pos0, 0);
        std::vector<double> refK(sentinel.begin(), sentinel.end()), refV = refK;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < d; ++j) {
                refK[(pos0 + t) * d + j] = qkv[t * 3 * d + d + j];
                refV[(pos0 + t) * d + j] = qkv[t * 3 * d + 2 * d + j];
            }
        const double e = std::max(max_abs_err(dK.download(), refK), max_abs_err(dV.download(), refV));
        report("kv_store T=" + std::to_string(T) + " pos0=" + std::to_string(pos0), e, 1e-5);
    }
}

// Reference attention against a cache: query row r at absolute position pos0 + r.
std::vector<double> attention_ref(const std::vector<float>& q, int q_stride,
                                  const std::vector<float>& K, const std::vector<float>& V,
                                  int pos0, int T, int H, int hd) {
    const int d = H * hd;
    std::vector<double> y(static_cast<size_t>(T) * d, 0.0);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            const int n = pos0 + t + 1;
            std::vector<double> s(n);
            double m = -1e300;
            for (int u = 0; u < n; ++u) {
                double acc = 0;
                for (int i = 0; i < hd; ++i)
                    acc += static_cast<double>(q[t * q_stride + h * hd + i]) * K[u * d + h * hd + i];
                s[u] = acc / std::sqrt(static_cast<double>(hd));
                m = std::max(m, s[u]);
            }
            double sum = 0;
            for (int u = 0; u < n; ++u) { s[u] = std::exp(s[u] - m); sum += s[u]; }
            for (int i = 0; i < hd; ++i) {
                double acc = 0;
                for (int u = 0; u < n; ++u) acc += s[u] / sum * V[u * d + h * hd + i];
                y[t * d + h * hd + i] = acc;
            }
        }
    return y;
}

void test_attention(std::mt19937& rng) {
    const int H = 3, hd = 16, d = H * hd;
    // Naive attention: prefill-like and decode-like calls; q rows have stride 3d like qkv.
    {
        const int C = 40;
        auto K = rand_vec(static_cast<size_t>(C) * d, rng, -2.f, 2.f);
        auto V = rand_vec(static_cast<size_t>(C) * d, rng);
        DevBuf<float> dK(K), dV(V), dscores(static_cast<size_t>(H) * 13 * C);
        const int cases[][2] = {{0, 1}, {0, 2}, {0, 13}, {5, 2}, {20, 13}, {39, 1}};  // {pos0, T}
        for (const auto& c : cases) {
            const int pos0 = c[0], T = c[1];
            auto qkv = rand_vec(static_cast<size_t>(T) * 3 * d, rng, -2.f, 2.f);
            DevBuf<float> dq(qkv), dy(static_cast<size_t>(T) * d);
            launch_attention_naive(dq.p, 3 * d, dK.p, dV.p, dscores.p, dy.p, pos0, T, H, hd, C, 0);
            auto ref = attention_ref(qkv, 3 * d, K, V, pos0, T, H, hd);
            report("attention_naive H=3 hd=16 pos0=" + std::to_string(pos0) + " T=" + std::to_string(T),
                   max_abs_err(dy.download(), ref), 1e-5);
        }
    }
    // Fused decode attention vs reference and vs naive, including > blockDim keys.
    for (int C : {40, 300}) {
        auto K = rand_vec(static_cast<size_t>(C) * d, rng, -2.f, 2.f);
        auto V = rand_vec(static_cast<size_t>(C) * d, rng);
        DevBuf<float> dK(K), dV(V), dscores(static_cast<size_t>(H) * C);
        for (int pos0 : {0, 1, 7, C / 2 + 3, C - 1}) {
            auto q = rand_vec(d, rng, -2.f, 2.f);
            DevBuf<float> dq(q), dyf(d), dyn(d);
            launch_attention_fused(dq.p, dK.p, dV.p, dyf.p, pos0, H, hd, 0);
            launch_attention_naive(dq.p, d, dK.p, dV.p, dscores.p, dyn.p, pos0, 1, H, hd, C, 0);
            auto ref = attention_ref(q, d, K, V, pos0, 1, H, hd);
            auto yf = dyf.download();
            auto yn = dyn.download();
            report("attention_fused C=" + std::to_string(C) + " pos0=" + std::to_string(pos0),
                   max_abs_err(yf, ref), 1e-5);
            std::vector<double> yn_d(yn.begin(), yn.end());
            report("fused vs naive C=" + std::to_string(C) + " pos0=" + std::to_string(pos0),
                   max_abs_err(yf, yn_d), 1e-5);
        }
    }
}

}  // namespace

int main() {
    std::printf("test_llm_kernels (platform %s)\n", HIPGPT2_PLATFORM_NAME);
    std::mt19937 rng(42);
    test_embedding(rng);
    test_layernorm(rng);
    test_gemv_xw(rng);
    test_gemv_rowdot(rng);
    test_bias(rng);
    test_fallback_gemm(rng);
    test_elementwise(rng);
    test_kv_store(rng);
    test_attention(rng);
    std::printf("SUMMARY: %d failed\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
