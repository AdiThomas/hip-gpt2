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
    alignas(16) __shared__ float As[2][kV4BK][kV4BM + kV4PadA];  // transposed: As[buf][k][m]
    alignas(16) __shared__ float Bs[2][kV4BK][kV4BN];            // Bs[buf][k][n]

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * kV4ThreadsX + tx;
    const int m0 = blockIdx.y * kV4BM;
    const int n0 = blockIdx.x * kV4BN;

    // This thread's slice of each slab (one 4-wide chunk of A and of B):
    //   A: row a_r (0..63), k offset a_k (0,4,8,12)   -> 4 threads cover one 16-float A row
    //   B: k row b_k (0..15), col offset b_c (0..60)  -> 16 threads cover one 64-float B row
    const int a_r = tid / (kV4BK / kV4Vec), a_k = (tid % (kV4BK / kV4Vec)) * kV4Vec;
    const int b_k = tid / (kV4BN / kV4Vec), b_c = (tid % (kV4BN / kV4Vec)) * kV4Vec;
    float a_reg[kV4Vec], b_reg[kV4Vec];  // the "in flight" slab lives here between phases

    // Global -> registers for the slab starting at k0. Never reads past A or B.
    auto load_slab = [&](int k0) {
        const int gr = m0 + a_r, gk = k0 + a_k;
        if constexpr (kVecA) {
            if (gr < M && gk < K) {  // K % 4 == 0 and gk % 4 == 0  =>  gk + 3 < K
                const float4 v = *reinterpret_cast<const float4*>(A + (size_t)gr * K + gk);
                a_reg[0] = v.x; a_reg[1] = v.y; a_reg[2] = v.z; a_reg[3] = v.w;
            } else {
                for (int i = 0; i < kV4Vec; ++i) a_reg[i] = 0.f;
            }
        } else {  // scalar path: misaligned pointer, K % 4 != 0, or K tail
            for (int i = 0; i < kV4Vec; ++i)
                a_reg[i] = (gr < M && gk + i < K) ? A[(size_t)gr * K + gk + i] : 0.f;
        }
        const int gk2 = k0 + b_k, gc = n0 + b_c;
        if constexpr (kVecB) {
            if (gk2 < K && gc < N) {  // N % 4 == 0 and gc % 4 == 0  =>  gc + 3 < N
                const float4 v = *reinterpret_cast<const float4*>(B + (size_t)gk2 * N + gc);
                b_reg[0] = v.x; b_reg[1] = v.y; b_reg[2] = v.z; b_reg[3] = v.w;
            } else {
                for (int j = 0; j < kV4Vec; ++j) b_reg[j] = 0.f;
            }
        } else {
            for (int j = 0; j < kV4Vec; ++j)
                b_reg[j] = (gk2 < K && gc + j < N) ? B[(size_t)gk2 * N + gc + j] : 0.f;
        }
    };

    // Registers -> LDS buffer `buf`. A is transposed (4 scalar stores), B is one float4.
    auto store_slab = [&](int buf) {
        for (int i = 0; i < kV4Vec; ++i) As[buf][a_k + i][a_r] = a_reg[i];
        *reinterpret_cast<float4*>(&Bs[buf][b_k][b_c]) =
            make_float4(b_reg[0], b_reg[1], b_reg[2], b_reg[3]);
    };

    float acc[kV4TM][kV4TN] = {};
    const int num_slabs = (K + kV4BK - 1) / kV4BK;

    load_slab(0);
    store_slab(0);
    __syncthreads();

    for (int s = 0; s < num_slabs; ++s) {
        const int cur = s & 1;
        const bool has_next = (s + 1) < num_slabs;  // uniform across the block
        // 1) Issue global loads for slab s+1; they complete while we compute slab s.
        if (has_next) load_slab((s + 1) * kV4BK);
        // 2) Compute slab s from LDS: per k one float4 of A and one of B, 16 FMAs.
        for (int kk = 0; kk < kV4BK; ++kk) {
            const float4 av = *reinterpret_cast<const float4*>(&As[cur][kk][ty * kV4TM]);
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[cur][kk][tx * kV4TN]);
            const float a[kV4TM] = {av.x, av.y, av.z, av.w};
            const float b[kV4TN] = {bv.x, bv.y, bv.z, bv.w};
            for (int i = 0; i < kV4TM; ++i)
                for (int j = 0; j < kV4TN; ++j) acc[i][j] += a[i] * b[j];
        }
        // 3) Park slab s+1 in the *other* buffer. Safe without an extra barrier: that
        //    buffer was last read in iteration s-1, which ended with a barrier.
        if (has_next) store_slab(cur ^ 1);
        // 4) One barrier: slab s+1 visible to all, and all reads of `cur` finished
        //    before iteration s+1 overwrites it.
        __syncthreads();
    }

    // Blocked output: thread owns rows m0+ty*4+i, cols n0+tx*4..+3. For each i, the 16
    // x-neighbours write 16 consecutive float4s = 256 contiguous bytes.
    const int col = n0 + tx * kV4TN;
    for (int i = 0; i < kV4TM; ++i) {
        const int row = m0 + ty * kV4TM + i;
        if (row >= M) continue;
        float* c = C + (size_t)row * N + col;
        if (vec_c && col < N) {  // N % 4 == 0 => all 4 columns valid
            *reinterpret_cast<float4*>(c) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        } else {
            for (int j = 0; j < kV4TN; ++j)
                if (col + j < N) c[j] = acc[i][j];
        }
    }
    // END SOLUTION
}

void launch_v4(const float* dA, const float* dB, float* dC, int M, int N, int K,
               hipStream_t stream) {
    // BEGIN SOLUTION
    // float4 access is legal only if every row starts 16-byte aligned: base pointer
    // aligned AND row stride (K for A, N for B and C) a multiple of 4 floats.
    const bool vec_a = v4_aligned16(dA) && (K % kV4Vec == 0);
    const bool vec_b = v4_aligned16(dB) && (N % kV4Vec == 0);
    const int vec_c = (v4_aligned16(dC) && (N % kV4Vec == 0)) ? 1 : 0;
    const dim3 block(kV4ThreadsX, kV4ThreadsY);
    const dim3 grid((N + kV4BN - 1) / kV4BN, (M + kV4BM - 1) / kV4BM);
    if (vec_a && vec_b) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v4_kernel<true, true>), grid, block, 0, stream, dA, dB, dC, M, N, K, vec_c);
    } else if (vec_a) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v4_kernel<true, false>), grid, block, 0, stream, dA, dB, dC, M, N, K, vec_c);
    } else if (vec_b) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v4_kernel<false, true>), grid, block, 0, stream, dA, dB, dC, M, N, K, vec_c);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_v4_kernel<false, false>), grid, block, 0, stream, dA, dB, dC, M, N, K, vec_c);
    }
    // END SOLUTION
}
