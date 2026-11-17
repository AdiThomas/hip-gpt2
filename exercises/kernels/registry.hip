// Kernel registry: the two tables gemm_bench iterates over, and gemm_fp32_best (the
// entry point the LLM engine uses). Not a kernel stage itself.
//
// Availability is decided at compile time:
//   v5            needs a matrix-core path (HIPGPT2_MC_MFMA / _WMMA_RDNA3 / _WMMA_NVIDIA)
//   hipblas*      needs HIPGPT2_HAVE_HIPBLAS (set by CMake when hipBLAS is found)
// Unavailable entries keep their name so `gemm_bench --list` can say why they are missing.
#include "gemm.hpp"

namespace {

// check_gemm metric thresholds (|err| / sum_k |a*b|, see reference.hpp).
constexpr double kTolFp32 = 1e-4;         // FP32 in, FP32 accumulate
constexpr double kTolFp16In = 1e-3;       // FP16 in (rounded inputs fed to the reference), FP32 accumulate
constexpr double kTolHipblasFp16 = 5e-2;  // hipblasHgemm: FP16 output and possibly FP16 accumulation

const GemmKernel kFp32Kernels[] = {
    {"v0", launch_v0, nullptr, kTolFp32},
    {"v1", launch_v1, nullptr, kTolFp32},
    {"v2", launch_v2, nullptr, kTolFp32},
    {"v3", launch_v3, nullptr, kTolFp32},
    {"v4", launch_v4, nullptr, kTolFp32},
#if defined(HIPGPT2_HAVE_HIPBLAS)
    {"hipblas", launch_hipblas_sgemm, nullptr, kTolFp32},
#else
    {"hipblas", nullptr, "built without hipBLAS", kTolFp32},
#endif
};

const GemmF16Kernel kFp16Kernels[] = {
    {"v0_fp16", launch_v0_fp16, nullptr, kTolFp16In},
    {"v4_fp16", launch_v4_fp16, nullptr, kTolFp16In},
#if defined(HIPGPT2_MC_NONE)
    {"v5", nullptr,
     "matrix-core instructions are not available on platform " HIPGPT2_PLATFORM_NAME
     " (no MFMA/WMMA path for this build's target arch)",
     kTolFp16In},
#else
    {"v5", launch_v5, nullptr, kTolFp16In},
#endif
#if defined(HIPGPT2_HAVE_HIPBLAS)
    {"hipblas_fp16", launch_hipblas_hgemm, nullptr, kTolHipblasFp16},
#else
    {"hipblas_fp16", nullptr, "built without hipBLAS", kTolHipblasFp16},
#endif
};

}  // namespace

const GemmKernel* gemm_fp32_kernels(int* count) {
    *count = static_cast<int>(sizeof(kFp32Kernels) / sizeof(kFp32Kernels[0]));
    return kFp32Kernels;
}

const GemmF16Kernel* gemm_fp16_kernels(int* count) {
    *count = static_cast<int>(sizeof(kFp16Kernels) / sizeof(kFp16Kernels[0]));
    return kFp16Kernels;
}

// Fastest FP32 stage that exists on every platform.
void gemm_fp32_best(const float* dA, const float* dB, float* dC, int M, int N, int K,
                    hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

const char* gemm_fp32_best_name() { return "v4"; }
