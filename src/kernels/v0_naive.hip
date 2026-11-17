// v0 — Naive GEMM: one thread computes one element of C.
//
// YOUR TASK (write this yourself — this is the code interviewers will ask about):
//   1. Fill in gemm_v0_kernel so thread (row, col) computes
//        C[row][col] = sum over k of A[row][k] * B[k][col]
//      using the row-major layout described in include/gemm.hpp.
//   2. Fill in launch_v0: pick a block shape, compute the grid so every element
//      of C is covered, and launch on `stream`.
//   3. Run `./build/gemm_bench --kernel v0 --test` until it passes.
//
// Questions to answer in docs/stages/v0.md BEFORE you benchmark:
//   - Which index (row or col) should come from threadIdx.x? Why might it matter?
//     (Hint: think about which addresses neighbouring threads touch in B and C.)
//   - What happens to threads whose (row, col) falls outside C when M or N isn't
//     a multiple of your block size?
//   - How many bytes does this kernel read from global memory per FLOP?
//     What does that predict about its speed vs the GPU's peak?

#include "gemm.hpp"

__global__ void gemm_v0_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    // TODO(you)
}

void launch_v0(const float* dA, const float* dB, float* dC, int M, int N, int K,
               hipStream_t stream) {
    // TODO(you): choose dim3 block(...), dim3 grid(...), then launch:
    //   gemm_v0_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    (void)dA; (void)dB; (void)dC; (void)M; (void)N; (void)K; (void)stream;
}
