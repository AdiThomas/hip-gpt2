// STAGE: v4_fp16 — LDS-tiled, register-blocked, double-buffered GEMM with FP16 inputs
// CHANGES: The v4 recipe for __half inputs: 64x64 block tile, 4x4 outputs per thread,
//   256 threads, two LDS buffers with register prefetch (one barrier per slab). Slabs
//   are BK = 32 deep and staged with 16-byte (8-half) vector loads when legal, by the
//   shared code in fp16_staging.inc that v5 also uses. LDS keeps the operands as FP16;
//   they are widened to FP32 in registers and accumulated in FP32.
// WHY FASTER: vs v0_fp16, each staged element is reused 64 times from LDS (global
//   traffic 1/32 byte/FLOP with 2-byte inputs) and 4 FLOPs per LDS read as in v3/v4.
//   vs FP32 v4: half the global and LDS bytes per slab, so the same LDS budget holds a
//   2x deeper slab (fewer barriers per FLOP).
// PREDICTED BOTTLENECK: FP32 FMA throughput plus conversion overhead (8 cvt per 16 FMAs)
//   on the VALU. This is the stage that shows why matrix cores matter: the inputs are
//   already FP16, but the math is still done one FMA at a time. Profiler: VALU-bound,
//   with v_cvt_f32_f16 a visible share of instructions.
#include "gemm.hpp"
#include "fp16_staging.inc"

constexpr int kV4F16TM = 4;                                // rows of C per thread
constexpr int kV4F16TN = 4;                                // cols of C per thread
constexpr int kV4F16ThreadsX = kF16BN / kV4F16TN;          // 16
constexpr int kV4F16ThreadsY = kF16BM / kV4F16TM;          // 16
constexpr int kV4F16Threads = kV4F16ThreadsX * kV4F16ThreadsY;  // 256: one 16-byte chunk per operand per thread

template <bool kVecA, bool kVecB>
__global__ void __launch_bounds__(kV4F16Threads)
gemm_v4_fp16_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    // BEGIN SOLUTION
    alignas(32) __shared__ __half As[2][kF16BM][kF16LdA];  // As[buf][m][k]
    alignas(32) __shared__ __half Bs[2][kF16BK][kF16LdB];  // Bs[buf][k][n]

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * kV4F16ThreadsX + tx;
    const int m0 = blockIdx.y * kF16BM;
    const int n0 = blockIdx.x * kF16BN;

    // Strided output mapping (as in v3): rows ty + 16*i, cols tx + 16*j. With As[m][k]
    // layout a thread's 4 A operands are not contiguous anyway, and this way the B reads
    // and the C stores are stride-1 across x-neighbours.
    float acc[kV4F16TM][kV4F16TN] = {};
    F16SlabRegs<kV4F16Threads> regs;
    const int num_slabs = (K + kF16BK - 1) / kF16BK;

    f16_load_slab<kV4F16Threads, kVecA, kVecB>(regs, A, B, M, N, K, m0, n0, 0, tid);
    f16_store_slab<kV4F16Threads>(regs, As[0], Bs[0], tid);
    __syncthreads();

    for (int s = 0; s < num_slabs; ++s) {
        const int cur = s & 1;
        const bool has_next = (s + 1) < num_slabs;  // uniform across the block
        if (has_next)  // global loads for slab s+1 overlap the math below
            f16_load_slab<kV4F16Threads, kVecA, kVecB>(regs, A, B, M, N, K, m0, n0, (s + 1) * kF16BK, tid);

        for (int kk = 0; kk < kF16BK; ++kk) {
            float a[kV4F16TM], b[kV4F16TN];
            for (int i = 0; i < kV4F16TM; ++i) a[i] = __half2float(As[cur][ty + i * kV4F16ThreadsY][kk]);
            for (int j = 0; j < kV4F16TN; ++j) b[j] = __half2float(Bs[cur][kk][tx + j * kV4F16ThreadsX]);
            for (int i = 0; i < kV4F16TM; ++i)
                for (int j = 0; j < kV4F16TN; ++j) acc[i][j] += a[i] * b[j];
        }

        if (has_next) f16_store_slab<kV4F16Threads>(regs, As[cur ^ 1], Bs[cur ^ 1], tid);  // other buffer: free since last barrier
        __syncthreads();
    }

    for (int i = 0; i < kV4F16TM; ++i) {
        const int row = m0 + ty + i * kV4F16ThreadsY;
        if (row >= M) continue;
        for (int j = 0; j < kV4F16TN; ++j) {
            const int col = n0 + tx + j * kV4F16ThreadsX;
            if (col < N) C[(size_t)row * N + col] = acc[i][j];
        }
    }
    // END SOLUTION
}

void launch_v4_fp16(const __half* dA, const __half* dB, float* dC, int M, int N, int K,
                    hipStream_t stream) {
    // BEGIN SOLUTION
    // 16-byte chunk loads are legal only if every row starts 16-byte aligned: aligned
    // base pointer AND row stride a multiple of 8 halves.
    const bool vec_a = f16_aligned16(dA) && (K % kF16Vec == 0);
    const bool vec_b = f16_aligned16(dB) && (N % kF16Vec == 0);
    const dim3 block(kV4F16ThreadsX, kV4F16ThreadsY);
    const dim3 grid((N + kF16BN - 1) / kF16BN, (M + kF16BM - 1) / kF16BM);
    if (vec_a && vec_b) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v4_fp16_kernel<true, true>), grid, block, 0, stream, dA, dB, dC, M, N, K);
    } else if (vec_a) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v4_fp16_kernel<true, false>), grid, block, 0, stream, dA, dB, dC, M, N, K);
    } else if (vec_b) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v4_fp16_kernel<false, true>), grid, block, 0, stream, dA, dB, dC, M, N, K);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v4_fp16_kernel<false, false>), grid, block, 0, stream, dA, dB, dC, M, N, K);
    }
    // END SOLUTION
}
