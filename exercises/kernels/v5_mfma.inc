// v5 inner product for AMD Instinct (CDNA2/CDNA3: gfx90a, gfx940-942) using
//   v_mfma_f32_16x16x16_f16  via  __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, cbsz, abid, blgp)
//     a: 4 x _Float16 (ext_vector_type(4))   b: 4 x _Float16   c/d: 4 x float
//     cbsz = abid = blgp = 0 (no broadcast / no lane-group permutation).
// Included only from v5_matrix_core.hip under HIPGPT2_MC_MFMA; never compiled on CPU.
//
// Lane layout (wave64; from AMD's Matrix Instruction Calculator for
// v_mfma_f32_16x16x16f16, 1 block; lane = 0..63, j = 0..3 is the element index
// inside the lane's 4-wide register):
//
//   operand   | lane holds element                     | in words
//   ----------+----------------------------------------+--------------------------------
//   A (16x16) | A[ i = lane % 16 ][ k = 4*(lane/16)+j ] | one row, 4 consecutive k
//   B (16x16) | B[ k = 4*(lane/16)+j ][ n = lane % 16 ] | one column, 4 consecutive k
//   D (16x16) | D[ m = 4*(lane/16)+j ][ n = lane % 16 ] | one column, 4 consecutive rows
//
//   lanes  0-15: k 0-3   (A/B)   rows  0-3  (D)
//   lanes 16-31: k 4-7           rows  4-7
//   lanes 32-47: k 8-11          rows  8-11
//   lanes 48-63: k 12-15         rows 12-15
//
// With As[m][k] (k contiguous) the A fragment is one 8-byte LDS read; with Bs[k][n] the
// B fragment is 4 scalar 2-byte reads 1 row apart (the cost named in the stage header).

#pragma once

// v5_matrix_core.hip derives wave/lane from kV5WaveSize; it must equal the compiled
// wavefront size (__AMDGCN_WAVEFRONT_SIZE is defined only in the device pass).
#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 64
#error "v5 MFMA path requires wave64 (CDNA default); do not build with -mwavefrontsize32"
#endif

constexpr int kV5WaveSize = 64;                    // CDNA wavefront
constexpr int kV5Waves = 4;
constexpr int kV5Threads = kV5Waves * kV5WaveSize; // 256: same as v4_fp16, 1 staging chunk per thread

typedef _Float16 v5_half4_t __attribute__((ext_vector_type(4)));
typedef float v5_float4_t __attribute__((ext_vector_type(4)));

struct V5Accum {
    v5_float4_t c[2][2];  // [tile row i][tile col j] of this wave's 2x2 grid of 16x16 tiles
};

// __half -> _Float16 without a value conversion (same 16 bits).
__device__ inline _Float16 v5_bits(const __half& h) {
    _Float16 f;
    __builtin_memcpy(&f, &h, sizeof f);
    return f;
}

__device__ inline void v5_acc_zero(V5Accum& acc) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

// All 16-deep k-steps of one LDS slab for this wave's 32x32 tile at (wm0, wn0).
__device__ inline void v5_mma_slab(V5Accum& acc, const __half (*As)[kF16LdA],
                                   const __half (*Bs)[kF16LdB], int wm0, int wn0, int lane) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

// Guarded epilogue straight from the accumulator layout (D table above).
__device__ inline void v5_store(const V5Accum& acc, float* C, int M, int N, int row0, int col0,
                                int lane, float* /*scratch: unused on MFMA*/) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
