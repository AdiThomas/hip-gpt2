#pragma once
#include <cstddef>

// CPU-side correctness checking. Plain C++ (no HIP) so it can be unit-tested anywhere.
//
// Error metric per element:  |got - exact| / sum_k |a_ik * b_kj|
// Normalizing by the sum of |products| (instead of |exact|) keeps the metric sane
// when a dot product happens to land near zero. A real bug gives errors ~1;
// FP32 rounding with FP32 accumulation gives ~1e-7 .. 1e-5.
struct CheckResult {
    bool ok;
    double max_err;
    long long checked;   // number of C elements compared
    long long worst_idx; // flat index (row * N + col) of the worst element
    bool full;           // true = every element checked, false = random sample
};

// Checks every element if M*N*K <= full_limit, otherwise `samples` random elements.
CheckResult check_gemm(const float* A, const float* B, const float* C,
                       int M, int N, int K,
                       double tol = 1e-4,
                       long long full_limit = 2048LL * 2048 * 2048,
                       int samples = 4096,
                       unsigned seed = 1234);
