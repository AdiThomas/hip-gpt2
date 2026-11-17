// STAGE: v1 — coalesced global memory access
// CHANGES: Same algorithm and loop as v0; only the thread -> element mapping changes:
//   threadIdx.x now walks columns (col = blockIdx.x*32 + threadIdx.x), and the block is
//   32 wide x 8 tall so a whole wave lies along rows of C.
// WHY FASTER: Same 4 bytes/FLOP in instructions, but far fewer memory transactions.
//   v0 (x -> row): per k a wave's A loads hit 16 rows (16 cache lines), B loads are 4
//   distinct addresses, C stores are 16 scattered pieces.
//   v1 (x -> col): per k a wave's B loads are 32 consecutive floats (one 128 B line) per
//   row-group, the A load is one address broadcast to all 32 lanes of a row, and C
//   stores are full 128 B lines. Lines fetched per FMA drop by roughly 16x.
// PREDICTED BOTTLENECK: Still memory/L2 bound: no data reuse beyond caches, each B
//   element is re-read by every block row. Profiler: good coalescing, but L2/HBM
//   bandwidth near its limit and arithmetic intensity far left of the roofline ridge.
#include "gemm.hpp"

// 32 x 8 = 256 threads. 32 columns per row: a full 128-byte line of B/C per wave row
// (wave64 covers 2 rows of 32, wave32 covers 1).
constexpr int kV1BlockX = 32;  // along N (cols) — the coalesced axis
constexpr int kV1BlockY = 8;   // along M (rows)

__global__ void gemm_v1_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // BEGIN SOLUTION
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // x -> col: neighbours are adjacent in B and C
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        const float* a = A + (size_t)row * K;  // same for every lane in this row: broadcast
        const float* b = B + col;              // lane i reads b[...+i]: contiguous
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += a[k] * b[(size_t)k * N];
        C[(size_t)row * N + col] = acc;        // contiguous across lanes
    }
    // END SOLUTION
}

void launch_v1(const float* dA, const float* dB, float* dC, int M, int N, int K,
               hipStream_t stream) {
    // BEGIN SOLUTION
    const dim3 block(kV1BlockX, kV1BlockY);
    const dim3 grid((N + kV1BlockX - 1) / kV1BlockX, (M + kV1BlockY - 1) / kV1BlockY);
    hipLaunchKernelGGL(gemm_v1_kernel, grid, block, 0, stream, dA, dB, dC, M, N, K);
    // END SOLUTION
}
