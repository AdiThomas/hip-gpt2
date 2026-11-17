#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if (_e != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP error '%s' at %s:%d: %s\n", #expr,       \
                         __FILE__, __LINE__, hipGetErrorString(_e));           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Contract for every kernel stage:
//   Row-major, FP32.  C[M x N] = A[M x K] * B[K x N]
//   A is M rows of K floats, B is K rows of N floats, C is M rows of N floats.
//   Must be correct for ANY M, N, K >= 1 (not just multiples of your tile size).
using GemmLaunchFn = void (*)(const float* dA, const float* dB, float* dC,
                              int M, int N, int K, hipStream_t stream);

// One launcher per stage. Add v1, v2, ... here as you write them,
// and register them in the table in src/bench.hip.
void launch_v0(const float* dA, const float* dB, float* dC, int M, int N, int K,
               hipStream_t stream);
