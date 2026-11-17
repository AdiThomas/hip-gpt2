#pragma once
#include "hip_common.hpp"

// Contract for every launcher (FP32 and FP16):
//   Row-major. C[M x N] = A[M x K] * B[K x N]. Correct for ANY M, N, K >= 1.
//   Writes every element of C, reads nothing out of bounds, C does not alias A or B.
//   Pointers are device pointers with natural alignment only (4 bytes float, 2 bytes __half);
//   they may point into the middle of a larger allocation.
//   Asynchronous on `stream`; no synchronization and no per-call allocation.
using GemmLaunchFn    = void (*)(const float*  dA, const float*  dB, float* dC,
                                 int M, int N, int K, hipStream_t stream);
// FP16 inputs, FP32 output (FP32 accumulation unless the entry's tol says otherwise).
using GemmF16LaunchFn = void (*)(const __half* dA, const __half* dB, float* dC,
                                 int M, int N, int K, hipStream_t stream);

struct GemmKernel    { const char* name; GemmLaunchFn    fn; const char* unavailable; double tol; };
struct GemmF16Kernel { const char* name; GemmF16LaunchFn fn; const char* unavailable; double tol; };
// fn == nullptr  <=>  unavailable != nullptr (human-readable reason, e.g.
// "matrix-core instructions are not available on platform cpu").
// tol: threshold for the check_gemm metric (reference.hpp).

// Registries (defined in src/kernels/registry.hip). Names are unique across both tables.
const GemmKernel*    gemm_fp32_kernels(int* count);   // v0 v1 v2 v3 v4 hipblas
const GemmF16Kernel* gemm_fp16_kernels(int* count);   // v0_fp16 v4_fp16 v5 hipblas_fp16

// Entry point other components (the LLM engine) use. Forwards to the fastest
// always-available FP32 ladder stage (v4). Same contract as GemmLaunchFn.
void        gemm_fp32_best(const float* dA, const float* dB, float* dC,
                           int M, int N, int K, hipStream_t stream);
const char* gemm_fp32_best_name();                    // e.g. "v4"

// Launchers. Always declared; defined only where available.
void launch_v0(const float* dA, const float* dB, float* dC, int M, int N, int K, hipStream_t stream);
void launch_v1(const float* dA, const float* dB, float* dC, int M, int N, int K, hipStream_t stream);
void launch_v2(const float* dA, const float* dB, float* dC, int M, int N, int K, hipStream_t stream);
void launch_v3(const float* dA, const float* dB, float* dC, int M, int N, int K, hipStream_t stream);
void launch_v4(const float* dA, const float* dB, float* dC, int M, int N, int K, hipStream_t stream);
void launch_hipblas_sgemm(const float* dA, const float* dB, float* dC, int M, int N, int K, hipStream_t stream);

void launch_v0_fp16(const __half* dA, const __half* dB, float* dC, int M, int N, int K, hipStream_t stream);
void launch_v4_fp16(const __half* dA, const __half* dB, float* dC, int M, int N, int K, hipStream_t stream);
void launch_v5(const __half* dA, const __half* dB, float* dC, int M, int N, int K, hipStream_t stream);
void launch_hipblas_hgemm(const __half* dA, const __half* dB, float* dC, int M, int N, int K, hipStream_t stream);
