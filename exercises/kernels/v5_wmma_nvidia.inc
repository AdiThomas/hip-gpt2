// v5 inner product for NVIDIA tensor cores (sm_70+, e.g. RTX 4060 = sm_89) through
// HIP-on-NVIDIA, using the CUDA C++ WMMA API (<mma.h>, namespace nvcuda::wmma):
//   fragment<matrix_a,    16,16,16, __half, row_major>
//   fragment<matrix_b,    16,16,16, __half, row_major>
//   fragment<accumulator, 16,16,16, float>
// Included only from v5_matrix_core.hip under HIPGPT2_MC_WMMA_NVIDIA; never compiled on CPU.
//
// Lane layout: OPAQUE. Unlike MFMA/WMMA-RDNA3, the per-lane mapping of fragment
// elements is unspecified by the API; load_matrix_sync / store_matrix_sync do the
// distribution, so this path never indexes fragments per lane:
//
//   operand   | how it is moved                                   | requirements
//   ----------+---------------------------------------------------+----------------------------------
//   A (16x16) | load_matrix_sync(a, &As[m][k], kF16LdA)  row_major | ptr 32-byte aligned, ldm % 8 == 0
//   B (16x16) | load_matrix_sync(b, &Bs[k][n], kF16LdB)  row_major | (halves): 40 and 72 both OK
//   D (16x16) | store_matrix_sync(scratch, c, 16, mem_row_major)  | scratch 32-byte aligned, ldm % 4
//             | -> per-warp LDS tile, then guarded scalar copy    | == 0 (floats)
//
// Alignment check: fragment origins are at rows that are multiples of 16 and k/n offsets
// that are multiples of 16 halves: As offset = 16*r*80 B + 32*q B, Bs offset = 16*r*144 B
// + 32*q B, all multiples of 32 given the alignas(32) LDS arrays.
// The direct store_matrix_sync to C is avoided because C tiles may be partial (any M, N)
// and C rows need not be 32-byte aligned.

#pragma once

#include <mma.h>

constexpr int kV5WaveSize = 32;                    // NVIDIA warp
constexpr int kV5Waves = 4;
constexpr int kV5Threads = kV5Waves * kV5WaveSize; // 128 -> 2 staging chunks per operand per thread

namespace v5w = nvcuda::wmma;

struct V5Accum {
    v5w::fragment<v5w::accumulator, 16, 16, 16, float> c[2][2];
};

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

// scratch: this warp's 16x16 float LDS tile (declared in the kernel).
__device__ inline void v5_store(const V5Accum& acc, float* C, int M, int N, int row0, int col0,
                                int lane, float* scratch) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
