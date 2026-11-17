// v5 inner product for RDNA3 (gfx1100-gfx1103), EXPERIMENTAL, using
//   v_wmma_f32_16x16x16_f16 (wave32) via
//   __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
//     a: 16 x _Float16 (ext_vector_type(16))   b: 16 x _Float16   c/d: 8 x float
// Included only from v5_matrix_core.hip under HIPGPT2_MC_WMMA_RDNA3 (gfx11 and
// -DHIPGPT2_ENABLE_RDNA3_WMMA=ON); never compiled on CPU. gfx12 (RDNA4) uses a different
// operand layout and builtin; this file is gfx11-only.
//
// Lane layout (wave32; per AMD GPUOpen "How to accelerate AI applications on RDNA 3
// using WMMA"; lane = 0..31, e = element index inside the lane's vector register):
//
//   operand   | lane holds element                           | notes
//   ----------+----------------------------------------------+-------------------------------
//   A (16x16) | A[ m = lane % 16 ][ k = e ],   e = 0..15     | a whole row; lanes 16-31 MUST
//             |                                              | hold a copy of lanes 0-15
//   B (16x16) | B[ k = e ][ n = lane % 16 ],   e = 0..15     | a whole column; lanes 16-31
//             |                                              | duplicate lanes 0-15
//   D (16x16) | D[ m = 2*e + lane/16 ][ n = lane % 16 ],     | e = 0..7: lanes 0-15 hold the
//             |   e = 0..7                                   | even rows, 16-31 the odd rows
//
// Loading A/B with `lane % 16` gives the required duplication automatically.

#pragma once

constexpr int kV5WaveSize = 32;                    // gfx11 wave32 mode (the default)
constexpr int kV5Waves = 4;
constexpr int kV5Threads = kV5Waves * kV5WaveSize; // 128 -> 2 staging chunks per operand per thread

typedef _Float16 v5_half16_t __attribute__((ext_vector_type(16)));
typedef float v5_float8_t __attribute__((ext_vector_type(8)));

struct V5Accum {
    v5_float8_t c[2][2];  // [tile row i][tile col j] of this wave's 2x2 grid of 16x16 tiles
};

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

__device__ inline void v5_mma_slab(V5Accum& acc, const __half (*As)[kF16LdA],
                                   const __half (*Bs)[kF16LdB], int wm0, int wn0, int lane) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

__device__ inline void v5_store(const V5Accum& acc, float* C, int M, int N, int row0, int col0,
                                int lane, float* /*scratch: unused on RDNA3*/) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
