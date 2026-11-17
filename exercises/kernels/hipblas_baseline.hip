// STAGE: hipblas — vendor baseline (hipBLAS -> rocBLAS/hipBLASLt on AMD, cuBLAS on NVIDIA)
// CHANGES: Not a ladder stage: the reference point every stage is measured against.
//   hipblasSgemm for FP32; hipblasHgemm (FP16 in, FP16 out) plus a small conversion
//   kernel to FP32 for the FP16 table.
// WHY FASTER: Vendor-tuned tile shapes per architecture, assembly-level scheduling,
//   matrix cores where the library chooses to use them, and large-tile pipelines.
// PREDICTED BOTTLENECK: Near-peak compute for large sizes; at small sizes launch
//   overhead and the FP16->FP32 conversion pass (one extra read/write of C) show up.
//
// Row-major via a column-major library: hipBLAS (like BLAS) is column-major. A row-major
// R x C matrix with leading dimension C is, byte for byte, its column-major transpose.
// So instead of moving data we ask for  C^T = B^T * A^T  in column-major terms:
//   gemm(OP_N, OP_N, m = N, n = M, k = K, B (ld N), A (ld K), C (ld N))
// i.e. the operands are swapped and the dimensions read (N, M, K). No transposes happen.
#include "gemm.hpp"

#if defined(HIPGPT2_HAVE_HIPBLAS)

#if __has_include(<hipblas/hipblas.h>)
#include <hipblas/hipblas.h>
#else
#include <hipblas.h>
#endif

#include <cstdint>
#include <cstring>

#define HIPBLAS_CHECK(expr)                                                         \
    do {                                                                            \
        hipblasStatus_t _s = (expr);                                                \
        if (_s != HIPBLAS_STATUS_SUCCESS) {                                         \
            std::fprintf(stderr, "hipBLAS error %d at %s:%d: %s\n", (int)_s,        \
                         __FILE__, __LINE__, #expr);                                \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

// One handle for the process, created on first use (creation is expensive and must not
// be timed per call). Not thread-safe; the bench and the engine are single-threaded.
static hipblasHandle_t hipblas_handle() {
    static hipblasHandle_t handle = nullptr;
    if (!handle) HIPBLAS_CHECK(hipblasCreate(&handle));
    return handle;
}

// hipblasHalf is uint16_t or __half depending on HIPBLAS_USE_HIP_HALF; both are two
// bytes holding IEEE binary16 bits, so build constants from raw bits.
static_assert(sizeof(hipblasHalf) == 2, "hipblasHalf must be 16-bit");
static hipblasHalf hipblas_half_from_bits(uint16_t bits) {
    hipblasHalf h;
    std::memcpy(&h, &bits, sizeof h);
    return h;
}

// FP16 C scratch for hgemm, grown lazily. The one allowed allocation exception
// (SPEC 3.5.7): it happens only when a larger M*N than ever before is requested.
// Shared across calls, so concurrent hgemm calls on different streams would race.
static __half* g_hgemm_scratch = nullptr;
static size_t g_hgemm_scratch_elems = 0;

constexpr int kCvtThreads = 256;  // plain elementwise kernel; any multiple of 64 works

__global__ void f16_to_f32_kernel(const __half* in, float* out, size_t n) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_hipblas_sgemm(const float* dA, const float* dB, float* dC, int M, int N, int K,
                          hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_hipblas_hgemm(const __half* dA, const __half* dB, float* dC, int M, int N, int K,
                          hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

#endif  // HIPGPT2_HAVE_HIPBLAS
