// STAGE: v5 — matrix-core inner product (MFMA on Instinct, WMMA on RDNA3 / NVIDIA)
// CHANGES: Same 64x64x32 block tile, double-buffered LDS and *identical* global->LDS
//   staging as v4_fp16 (fp16_staging.inc). Only the inner product changes: the block is
//   4 waves in a 2x2 grid, each wave owns a 32x32 piece of C = 2x2 tiles of 16x16, and
//   each 16x16x16 step is one matrix instruction fed with FP16 fragments read from LDS
//   and accumulating into FP32 registers. The epilogue writes C with bounds checks;
//   edge tiles work because staging zero-pads A/B in LDS (any M, N, K).
// WHY FASTER: One MFMA does 16*16*16 = 4096 FMAs per wave; the scalar path needs 64
//   lanes x 64 FMA instructions for the same work. Matrix-core FP16 peak is several
//   times FP32 vector peak (MI300X: ~1300 dense FP16 TFLOPS vs ~160 FP32 vector), so
//   the VALU ceiling of v4_fp16 is lifted and the kernel becomes LDS/global-feed bound.
// PREDICTED BOTTLENECK: Operand feed. B fragments are 4 (MFMA) / 16 (WMMA) strided
//   2-byte LDS reads per lane (B is k-major in LDS), and a 64x64 block tile gives only
//   64 FLOPs per global byte, too little to reach matrix-core peak. Profiler: MFMA
//   utilisation well below peak, LDS read instructions and barrier stalls dominant.
//   Next steps: transpose B in LDS, larger block/wave tiles, deeper pipelining.
//
// Paths (selected by CMake via HIPGPT2_MC_*; exactly one is compiled):
//   v5_mfma.inc         __builtin_amdgcn_mfma_f32_16x16x16f16   gfx908/90a/94x, wave64, 256 threads
//   v5_wmma_rdna3.inc   __builtin_amdgcn_wmma_f32_16x16x16_f16_w32  gfx11, wave32, 128 threads (experimental)
//   v5_wmma_nvidia.inc  nvcuda::wmma 16x16x16 half->float        sm_70+, warp32, 128 threads
// HIPGPT2_MC_NONE (CPU and other archs): launch_v5 is not defined; the registry marks
// v5 unavailable.
#include "gemm.hpp"
#include "fp16_staging.inc"

#if !defined(HIPGPT2_MC_NONE)

#if defined(HIPGPT2_MC_MFMA)
#include "v5_mfma.inc"
#elif defined(HIPGPT2_MC_WMMA_RDNA3)
#include "v5_wmma_rdna3.inc"
#elif defined(HIPGPT2_MC_WMMA_NVIDIA)
#include "v5_wmma_nvidia.inc"
#else
#error "v5: unknown HIPGPT2_MC_* path"
#endif

// Contract every .inc provides:
//   kV5WaveSize, kV5Waves (= 4), kV5Threads (= kV5Waves * kV5WaveSize)
//   struct V5Accum                                        FP32 accumulators for 2x2 16x16 tiles
//   v5_acc_zero(V5Accum&)
//   v5_mma_slab(V5Accum&, As, Bs, wm0, wn0, lane)         all BK/16 k-steps of one LDS slab
//   v5_store(const V5Accum&, C, M, N, row0, col0, lane, scratch)   guarded epilogue
constexpr int kV5WaveTile = 32;                       // each wave: 32x32 of C = 2x2 16x16 tiles
constexpr int kV5WavesN = kF16BN / kV5WaveTile;       // 2 waves across N
static_assert(kV5Waves == (kF16BM / kV5WaveTile) * kV5WavesN, "2x2 waves cover the 64x64 block tile");

template <bool kVecA, bool kVecB>
__global__ void __launch_bounds__(kV5Threads)
gemm_v5_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    // BEGIN SOLUTION
    alignas(32) __shared__ __half As[2][kF16BM][kF16LdA];  // same LDS layout as v4_fp16
    alignas(32) __shared__ __half Bs[2][kF16BK][kF16LdB];
#if defined(HIPGPT2_MC_WMMA_NVIDIA)
    alignas(32) __shared__ float Cs[kV5Waves][16 * 16];    // per-warp epilogue staging (wmma layout is opaque)
#endif

    const int tid = threadIdx.x;                 // 1-D block
    const int wave = tid / kV5WaveSize;          // 0..3
    const int lane = tid % kV5WaveSize;
    const int m0 = blockIdx.y * kF16BM;
    const int n0 = blockIdx.x * kF16BN;
    const int wm0 = (wave / kV5WavesN) * kV5WaveTile;  // this wave's tile origin inside the block tile
    const int wn0 = (wave % kV5WavesN) * kV5WaveTile;

    V5Accum acc;
    v5_acc_zero(acc);
    F16SlabRegs<kV5Threads> regs;
    const int num_slabs = (K + kF16BK - 1) / kF16BK;

    // Pipeline identical to v4_fp16 (prefetch into registers, two LDS buffers, one barrier).
    f16_load_slab<kV5Threads, kVecA, kVecB>(regs, A, B, M, N, K, m0, n0, 0, tid);
    f16_store_slab<kV5Threads>(regs, As[0], Bs[0], tid);
    __syncthreads();

    for (int s = 0; s < num_slabs; ++s) {
        const int cur = s & 1;
        const bool has_next = (s + 1) < num_slabs;
        if (has_next)
            f16_load_slab<kV5Threads, kVecA, kVecB>(regs, A, B, M, N, K, m0, n0, (s + 1) * kF16BK, tid);
        v5_mma_slab(acc, As[cur], Bs[cur], wm0, wn0, lane);
        if (has_next) f16_store_slab<kV5Threads>(regs, As[cur ^ 1], Bs[cur ^ 1], tid);
        __syncthreads();
    }

#if defined(HIPGPT2_MC_WMMA_NVIDIA)
    float* scratch = Cs[wave];
#else
    float* scratch = nullptr;  // AMD paths know their accumulator layout and write C directly
#endif
    v5_store(acc, C, M, N, m0 + wm0, n0 + wn0, lane, scratch);
    // END SOLUTION
}

void launch_v5(const __half* dA, const __half* dB, float* dC, int M, int N, int K,
               hipStream_t stream) {
    // BEGIN SOLUTION
    const bool vec_a = f16_aligned16(dA) && (K % kF16Vec == 0);
    const bool vec_b = f16_aligned16(dB) && (N % kF16Vec == 0);
    const dim3 block(kV5Threads);
    const dim3 grid((N + kF16BN - 1) / kF16BN, (M + kF16BM - 1) / kF16BM);
    if (vec_a && vec_b) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v5_kernel<true, true>), grid, block, 0, stream, dA, dB, dC, M, N, K);
    } else if (vec_a) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v5_kernel<true, false>), grid, block, 0, stream, dA, dB, dC, M, N, K);
    } else if (vec_b) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v5_kernel<false, true>), grid, block, 0, stream, dA, dB, dC, M, N, K);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v5_kernel<false, false>), grid, block, 0, stream, dA, dB, dC, M, N, K);
    }
    // END SOLUTION
}

#endif  // !HIPGPT2_MC_NONE
