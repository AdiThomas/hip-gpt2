// STAGE: v5 — matrix-core inner product (MFMA on Instinct, WMMA on RDNA3 / NVIDIA)
// CHANGES: Same 64x64x32 block tile, double-buffered LDS and *identical* global->LDS
//   staging as v4_fp16 (fp16_staging.inc). Only the inner product changes: the block is
//   4 waves in a 2x2 grid, each wave owns a 32x32 piece of C = 2x2 tiles of 16x16, and
//   each 16x16x16 step is one matrix instruction fed with FP16 fragments read from LDS
//   and accumulating into FP32 registers. The epilogue writes C with bounds checks;
//   edge tiles work because staging zero-pads A/B in LDS (any M, N, K).
// WHY FASTER: One MFMA does 16*16*16 = 4096 FMAs per wave; the scalar path needs 64
//   lanes x 64 FMA instructions for the same work. Matrix-core FP16 peak is several
//   times FP32 vector peak (MI300X: ~1307 dense FP16 matrix TFLOPS vs ~81.7 FP32 vector), so
//   the VALU ceiling of v4_fp16 is lifted and the kernel becomes LDS/global-feed bound.
// PREDICTED BOTTLENECK: Operand feed. B fragments are 4 (MFMA) / 16 (WMMA) strided
//   2-byte LDS reads per lane (B is k-major in LDS), and a 64x64 FP16 block tile gives
//   only BM*BN/(BM+BN) = 32 FLOPs per global byte, too little to reach matrix-core peak.
//   The MFMA B reads may also 2-way bank conflict: with kF16PadB = 8 a Bs row is 36
//   words, so lane groups kq = 0/8 and kq = 4/12 hit the same banks (check
//   SQ_LDS_BANK_CONFLICT on MI300X). Profiler: MFMA utilisation well below peak, LDS
//   read instructions and barrier stalls dominant.
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
    // TODO: your code here
    // END SOLUTION
}

void launch_v5(const __half* dA, const __half* dB, float* dC, int M, int N, int K,
               hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

#endif  // !HIPGPT2_MC_NONE
