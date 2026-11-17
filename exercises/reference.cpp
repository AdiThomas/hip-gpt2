#include "reference.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {

// Exact-ish (double accumulation) error for one element of C.
double elem_err(const float* a_row, const float* bt_row, float got, int K) {
    if (!std::isfinite(got)) return std::numeric_limits<double>::infinity();
    double acc = 0.0, bound = 0.0;
    for (int k = 0; k < K; ++k) {
        double p = static_cast<double>(a_row[k]) * bt_row[k];
        acc += p;
        bound += std::fabs(p);
    }
    double diff = std::fabs(static_cast<double>(got) - acc);
    if (bound == 0.0) return diff;  // all-zero inputs: exact answer is 0
    return diff / bound;
}

}  // namespace

CheckResult check_gemm(const float* A, const float* B, const float* C,
                       int M, int N, int K, double tol, long long full_limit,
                       int samples, unsigned seed) {
    // Transpose B so each column of B is contiguous -> fast CPU dot products.
    std::vector<float> Bt(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            Bt[static_cast<size_t>(n) * K + k] = B[static_cast<size_t>(k) * N + n];

    CheckResult r{true, 0.0, 0, -1, false};
    const long long work = static_cast<long long>(M) * N * K;

    if (work <= full_limit) {
        r.full = true;
        r.checked = static_cast<long long>(M) * N;
        const long long total = r.checked;
        double max_err = 0.0;
        long long worst = -1;
#pragma omp parallel
        {
            double local_max = 0.0;
            long long local_worst = -1;
#pragma omp for schedule(static)
            for (long long idx = 0; idx < total; ++idx) {
                int m = static_cast<int>(idx / N), n = static_cast<int>(idx % N);
                double e = elem_err(A + static_cast<size_t>(m) * K,
                                    Bt.data() + static_cast<size_t>(n) * K, C[idx], K);
                if (e > local_max || local_worst < 0) { local_max = e; local_worst = idx; }
            }
#pragma omp critical
            {
                if (local_worst >= 0 && (local_max > max_err || worst < 0)) {
                    max_err = local_max;
                    worst = local_worst;
                }
            }
        }
        r.max_err = max_err;
        r.worst_idx = worst;
    } else {
        // Too big to check fully on CPU in reasonable time: spot-check random elements.
        // A tiling/indexing bug almost always corrupts a large fraction of C,
        // so a few thousand samples catch it with overwhelming probability.
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dm(0, M - 1), dn(0, N - 1);
        // Always include the four corners: edge-handling bugs live there.
        std::vector<std::pair<int, int>> pts = {{0, 0}, {0, N - 1}, {M - 1, 0}, {M - 1, N - 1}};
        for (int i = 0; i < samples; ++i) pts.push_back({dm(rng), dn(rng)});
        for (auto [m, n] : pts) {
            long long idx = static_cast<long long>(m) * N + n;
            double e = elem_err(A + static_cast<size_t>(m) * K,
                                Bt.data() + static_cast<size_t>(n) * K, C[idx], K);
            if (e > r.max_err || r.worst_idx < 0) { r.max_err = e; r.worst_idx = idx; }
        }
        r.checked = static_cast<long long>(pts.size());
    }

    r.ok = r.max_err <= tol;
    return r;
}
