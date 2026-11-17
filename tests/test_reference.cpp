// Tests the checker itself: a correct C must pass, and common kernel bugs must fail.
// If the checker is wrong, every "PASS" from gemm_bench is meaningless.
#include "reference.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

static void naive(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                  int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;  // FP32 accumulation, like a GPU kernel
            for (int k = 0; k < K; ++k) acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
            C[(size_t)m * N + n] = acc;
        }
}

int main() {
    int failures = 0;
    auto expect = [&](bool cond, const char* what) {
        std::printf("  %s %s\n", cond ? "ok  " : "FAIL", what);
        if (!cond) ++failures;
    };

    const int M = 133, N = 97, K = 301;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);
    naive(A, B, C, M, N, K);

    expect(check_gemm(A.data(), B.data(), C.data(), M, N, K).ok, "correct result passes (full)");
    expect(check_gemm(A.data(), B.data(), C.data(), M, N, K, 1e-4, 0).ok,
           "correct result passes (sampled)");

    auto bad = C;  bad[(size_t)(M - 1) * N + (N - 1)] += 1.f;
    expect(!check_gemm(A.data(), B.data(), bad.data(), M, N, K).ok, "wrong last element fails");
    expect(!check_gemm(A.data(), B.data(), bad.data(), M, N, K, 1e-4, 0).ok,
           "wrong corner caught by sampled mode");

    bad = C;  bad[5] = std::numeric_limits<float>::quiet_NaN();
    expect(!check_gemm(A.data(), B.data(), bad.data(), M, N, K).ok, "NaN (unwritten element) fails");

    // Classic bug: computing A*B^T-style indexing (swapped B index). Use square to make it compile-able.
    const int S = 64;
    std::vector<float> As((size_t)S * S), Bs((size_t)S * S), Cs((size_t)S * S);
    for (auto& x : As) x = d(rng);
    for (auto& x : Bs) x = d(rng);
    for (int m = 0; m < S; ++m)
        for (int n = 0; n < S; ++n) {
            float acc = 0;
            for (int k = 0; k < S; ++k) acc += As[m * S + k] * Bs[n * S + k];  // B[n][k] instead of B[k][n]
            Cs[m * S + n] = acc;
        }
    expect(!check_gemm(As.data(), Bs.data(), Cs.data(), S, S, S).ok, "transposed-B indexing bug fails");

    std::printf("%s\n", failures ? "REFERENCE TESTS FAILED" : "all reference tests passed");
    return failures ? 1 : 0;
}
