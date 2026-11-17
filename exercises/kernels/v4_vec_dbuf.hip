// STAGE: v4 — vectorized loads + double buffering
// CHANGES: v3's tiling (64x64x16, 4x4 per thread, 256 threads) plus:
//   (a) each thread moves its share of a slab with one 16-byte float4 global load per
//       operand (scalar fallback when the pointer or row stride is not 16-byte aligned),
//       reads LDS operands as float4 and stores C as float4 when legal;
//   (b) two LDS buffers: the next slab is loaded into registers before the current one
//       is consumed and written to the other buffer afterwards, so the global latency
//       overlaps the FMAs and only one barrier separates slabs (v3 had two).
//   The output mapping becomes *blocked* (thread owns rows ty*4.., cols tx*4..) so each
//   thread's 4 operands are contiguous in LDS and readable as one float4.
// WHY FASTER: Same bytes/FLOP as v3 (1/16 global, 4 FLOPs per LDS word), but 4x fewer
//   load/store instructions (1 float4 instead of 4 floats; ds_read_b128 for operands),
//   and global latency is hidden behind 16 x 16 = 256 FMAs per slab instead of exposed.
// PREDICTED BOTTLENECK: Becomes compute/issue bound for large sizes: VALU utilisation
//   high, remaining stalls are the single barrier per slab and occupancy (17 KiB LDS and
//   ~40 VGPRs per thread). Beyond this point, FP32 FMA throughput itself is the ceiling;
//   matrix cores (v5) are the next step. Profiler: higher achieved FLOPS, low vmcnt
//   stalls, LDS bank conflicts near zero.
#include "gemm.hpp"

#include <cstdint>

// Same tile shape as v3 so the v3 -> v4 delta isolates vectorization + double buffering.
// BK = 16 keeps the per-thread slab share at exactly one float4 of A and one of B.
constexpr int kV4BM = 64;
constexpr int kV4BN = 64;
constexpr int kV4BK = 16;
constexpr int kV4TM = 4;
constexpr int kV4TN = 4;
constexpr int kV4ThreadsX = kV4BN / kV4TN;              // 16
constexpr int kV4ThreadsY = kV4BM / kV4TM;              // 16
constexpr int kV4Threads = kV4ThreadsX * kV4ThreadsY;   // 256
constexpr int kV4Vec = 4;                               // floats per float4
// Pad As rows by one float4: keeps every As row 16-byte aligned (float4 LDS reads) and
// shifts consecutive rows by 4 banks for the transposing scalar stores.
constexpr int kV4PadA = 4;
static_assert(kV4BM * kV4BK / kV4Vec == kV4Threads, "one float4 of A per thread per slab");
static_assert(kV4BK * kV4BN / kV4Vec == kV4Threads, "one float4 of B per thread per slab");
static_assert(kV4TM == kV4Vec && kV4TN == kV4Vec, "operand reads assume 4 = one float4");

static inline bool v4_aligned16(const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

// kVecA: A is 16-byte aligned and K % 4 == 0, so A[r][k..k+3] (k % 4 == 0) is one legal
//        float4 that lies entirely inside or entirely outside row r.
// kVecB: same for B with N % 4 == 0.
// vec_c: same for C (runtime flag, uniform across the grid, so no divergence).
template <bool kVecA, bool kVecB>
__global__ void __launch_bounds__(kV4Threads)
gemm_v4_kernel(const float* A, const float* B, float* C, int M, int N, int K, int vec_c) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}

void launch_v4(const float* dA, const float* dB, float* dC, int M, int N, int K,
               hipStream_t stream) {
    // BEGIN SOLUTION
    // TODO: your code here
    // END SOLUTION
}
