// Shared FP16 global -> LDS staging for v4_fp16 and v5 (included by both .hip files).
//
// Why a shared include: v5's matrix-core inner product cannot run on the CPU backend,
// but everything that feeds it can. Keeping the staging code literally identical means
// every CPU run of v4_fp16 tests v5's address math, zero padding and vector-load
// legality rules too; only v5's inner product and epilogue are hardware-only.
//
// Block tile: BM x BN outputs, K walked in slabs of BK. LDS layout (natural, no transpose):
//   As[m][k]  row-major A slab, k contiguous  -> an MFMA/WMMA A fragment (one row, several
//                                                consecutive k) is a contiguous LDS read
//   Bs[k][n]  row-major B slab, n contiguous
// Each thread moves 16-byte chunks (8 halves): global -> registers (load_slab), then
// registers -> LDS (store_slab), so callers can double-buffer exactly like v4.
#pragma once

#include <cstdint>

constexpr int kF16BM = 64;   // block tile rows
constexpr int kF16BN = 64;   // block tile cols
constexpr int kF16BK = 32;   // slab depth: 2 matrix-core K-steps of 16 for v5
constexpr int kF16Vec = 8;   // halves per 16-byte chunk
// +8 halves (16 bytes) of row padding: rows stay 16-byte aligned for the vector LDS
// stores (and 32-byte aligned at every 16-row/16-col fragment origin, which
// nvcuda::wmma::load_matrix_sync requires), while consecutive rows start 4 banks apart.
constexpr int kF16PadA = 8;
constexpr int kF16PadB = 8;
constexpr int kF16LdA = kF16BK + kF16PadA;   // 40 halves = 80 bytes per As row
constexpr int kF16LdB = kF16BN + kF16PadB;   // 72 halves = 144 bytes per Bs row
constexpr int kF16ChunksA = kF16BM * kF16BK / kF16Vec;   // 256 chunks per A slab
constexpr int kF16ChunksB = kF16BK * kF16BN / kF16Vec;   // 256 chunks per B slab
static_assert(kF16BK % kF16Vec == 0 && kF16BN % kF16Vec == 0, "rows must split into whole chunks");

struct alignas(16) F16Chunk { __half h[kF16Vec]; };

// Per-thread register staging for one slab. kThreads = block size (256 for v4_fp16 and
// the MFMA path, 128 for the wave32 paths -> 1 or 2 chunks per operand per thread).
template <int kThreads>
struct F16SlabRegs {
    static_assert(kF16ChunksA % kThreads == 0 && kF16ChunksB % kThreads == 0, "even split");
    F16Chunk a[kF16ChunksA / kThreads];
    F16Chunk b[kF16ChunksB / kThreads];
};

static inline bool f16_aligned16(const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

// Global -> registers for the slab starting at k0.
// kVecA: A 16-byte aligned and K % 8 == 0, so a chunk A[r][k..k+7] (k % 8 == 0) is one
//        legal 16-byte load lying entirely inside or outside the matrix. Otherwise each
//        half is loaded separately with its own bounds check. Out of range -> 0.
// kVecB: same for B with N % 8 == 0.
template <int kThreads, bool kVecA, bool kVecB>
__device__ inline void f16_load_slab(F16SlabRegs<kThreads>& r, const __half* A, const __half* B,
                                     int M, int N, int K, int m0, int n0, int k0, int tid) {
    const __half zero = __float2half(0.0f);
    // A slab: BM rows x (BK/8 = 4) chunks; consecutive threads take consecutive chunks
    // of a row, so 4 threads read one contiguous 64-byte row segment.
    for (int i = 0; i < kF16ChunksA / kThreads; ++i) {
        const int c = tid + i * kThreads;
        const int row = c / (kF16BK / kF16Vec), kk = (c % (kF16BK / kF16Vec)) * kF16Vec;
        const int gr = m0 + row, gk = k0 + kk;
        if constexpr (kVecA) {
            if (gr < M && gk < K) r.a[i] = *reinterpret_cast<const F16Chunk*>(A + (size_t)gr * K + gk);
            else for (int e = 0; e < kF16Vec; ++e) r.a[i].h[e] = zero;
        } else {
            for (int e = 0; e < kF16Vec; ++e)
                r.a[i].h[e] = (gr < M && gk + e < K) ? A[(size_t)gr * K + gk + e] : zero;
        }
    }
    // B slab: BK rows x (BN/8 = 8) chunks; 8 threads read one contiguous 128-byte row.
    for (int i = 0; i < kF16ChunksB / kThreads; ++i) {
        const int c = tid + i * kThreads;
        const int kr = c / (kF16BN / kF16Vec), cc = (c % (kF16BN / kF16Vec)) * kF16Vec;
        const int gk = k0 + kr, gc = n0 + cc;
        if constexpr (kVecB) {
            if (gk < K && gc < N) r.b[i] = *reinterpret_cast<const F16Chunk*>(B + (size_t)gk * N + gc);
            else for (int e = 0; e < kF16Vec; ++e) r.b[i].h[e] = zero;
        } else {
            for (int e = 0; e < kF16Vec; ++e)
                r.b[i].h[e] = (gk < K && gc + e < N) ? B[(size_t)gk * N + gc + e] : zero;
        }
    }
}

// Registers -> one LDS buffer. Same chunk mapping as the load, one 16-byte store each.
template <int kThreads>
__device__ inline void f16_store_slab(const F16SlabRegs<kThreads>& r, __half (*As)[kF16LdA],
                                      __half (*Bs)[kF16LdB], int tid) {
    for (int i = 0; i < kF16ChunksA / kThreads; ++i) {
        const int c = tid + i * kThreads;
        const int row = c / (kF16BK / kF16Vec), kk = (c % (kF16BK / kF16Vec)) * kF16Vec;
        *reinterpret_cast<F16Chunk*>(&As[row][kk]) = r.a[i];
    }
    for (int i = 0; i < kF16ChunksB / kThreads; ++i) {
        const int c = tid + i * kThreads;
        const int kr = c / (kF16BN / kF16Vec), cc = (c % (kF16BN / kF16Vec)) * kF16Vec;
        *reinterpret_cast<F16Chunk*>(&Bs[kr][cc]) = r.b[i];
    }
}
