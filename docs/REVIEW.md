# Independent code review (before any GPU run)

An Opus reviewer agent that had not seen the code being written audited the kernels and the engine. It could
build and run tests but not edit anything. This file records what it checked, what it found, and what was done.

## How the untestable parts were checked

The v5 matrix-core paths can't run without the right GPU, so the reviewer checked them three ways:

1. **By hand, against the instruction semantics.** For MFMA `f32_16x16x16f16` (wave64), lane `l` holds
   `A[l%16][4*(l/16)+i]`, `B[4*(l/16)+i][l%16]` and `D[4*(l/16)+i][l%16]`. For RDNA3 WMMA (wave32), A and B use `lane%16`
   (the two half-waves are replicated) and the result row is `2*e + lane/16`. `nvcuda::wmma` fragment types, leading
   dimensions (40 and 72 halves) and 32-byte alignment were also checked.
2. **Device-only compile with clang** for gfx942, gfx90a and gfx1100. This emitted real `v_mfma_f32_16x16x16_f16` /
   `v_wmma` instructions with no register spills, 19,456 B of LDS, and the expected wave sizes.
3. **CPU emulation** (`tests/emulation/v5_mfma_emulator.cpp`, now the CTest `v5_mfma_emulation`). It runs the real
   staging and MFMA code with a mock instruction that records each lane's fragments, rebuilds the result using the
   hardware lane rules, and stores it through the real store code. All test shapes pass (max error ≤ 1.1e-7). A
   deliberately wrong lane mapping fails, which proves the test can catch layout bugs.

Reviewer's confidence that v5 MFMA is correct on MI300X before running it there: **about 90%**.

## Findings and resolutions

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | risk | CMake picked the matrix-core path from the first GPU arch only. Mixed lists broke or silently disabled MFMA, and gfx950 was missing. | **Fixed.** Every listed arch is checked, mixed families fail with a clear message, and gfx950 was added. |
| 2 | risk | v5 assumes the compiled wavefront size matches the code (64 for MFMA, 32 for WMMA). | **Fixed.** A compile-time `#error` fires if `__AMDGCN_WAVEFRONT_SIZE` differs. |
| 3 | risk | Decode tokens/s timing included printing to the terminal and an extra logits copy. | **Fixed.** Only decode, the logits copy and sampling are timed. |
| 4 | nit | `prefill_ms` was host wall time, not device time. | **Fixed.** It now uses a hipEvent pair. `ttft_ms` stays wall time. |
| 5 | nit | The v5 header said 64 FLOPs per global byte; the correct figure is 32. | **Fixed.** |
| 6 | perf | Possible 2-way LDS bank conflict on v5's B-operand reads. | **Open.** Noted in the v5 header. Check `SQ_LDS_BANK_CONFLICT` with `scripts/profile_amd.sh v5 4096`. |
| 7 | nit | The hipBLAS FP16 scratch buffer grows inside the launcher (a device sync on first use at each size). | **Open, harmless.** The benchmark warms up before timing. |
| 8 | nit | `hipblasHgemm` rounds C to FP16, so it isn't a like-for-like baseline for v5 (FP32 output). | **Open.** Next step: add a `hipblasGemmEx` baseline with f16 inputs and f32 output. Good interview point. |
| 9 | perf | Fused decode attention uses only `head_dim` (64) of its 128 threads for the V sum. | **Open.** Next step: split keys across threads (flash-decoding) if it shows up in the profile. |

Also checked with no issues: v0–v4 index math and edge guards; float4 legality and tails; double-buffer hazards in
v4, v4_fp16 and v5; the hipBLAS operand swap; KV-cache written before attention; causal bounds; the online-softmax
merge; LayerNorm; GELU; GEMV against HF Conv1D `[in, out]` weights; the tied LM head; the `transformer.` prefix fallback.
