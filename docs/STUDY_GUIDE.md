# STUDY_GUIDE: a plan to own every line of hip-gpt2

Goal: in 4 to 6 weeks at 8 to 10 hours a week, be able to rebuild the important kernels from a blank file, explain every design choice, and survive a technical interview on this project. Read [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md) first (sections 1 to 3 on week 0); this guide tells you what to do with it. Background reading: [README.md](../README.md), [SETUP.md](SETUP.md), per-stage pages in [stages/](stages/) ([v0](stages/v0.md) to [v5](stages/v5.md)), and [../SPEC.md](../SPEC.md).

## Honesty note: read this before any interview

The code in this repo was written by AI agents under a design written by another AI. Say so. Something like:

> "I designed the project goals and used Claude to build it: an architect model wrote the spec, other models wrote the kernels, tests and CI. Then I spent weeks taking it apart: I rewrote the kernels from blank exercises, I can explain every stage and where the numbers come from, and I know which parts I have and have not verified on real hardware."

That sentence is only true if you do the work in this guide. Interviewers at AMD will dig into one stage until you run out of answers; being honest about the origin and then showing real understanding is far stronger than pretending, and it is the only approach that survives follow-up questions. Never present a `{{placeholder}}` as a measurement. Never claim v5, hipBLAS, or real-GPT-2 results before you have run them (see "verified vs not" in PROJECT_EXPLAINED section 7).

---

## Setup (do once, about 1 hour)

1. Create a branch so the solutions stay safe in git history:
   ```bash
   git checkout -b study
   ```
2. Configure a CPU build (no GPU needed; `HIP_CPU_DIR` is your HIP-CPU checkout, see SETUP.md):
   ```bash
   cmake -S . -B build-cpu -DGEMM_BACKEND=cpu -DHIP_CPU_DIR="$HIP_CPU_DIR"
   cmake --build build-cpu -j
   ctest --test-dir build-cpu --output-on-failure      # all 9 should pass first
   ```
3. Learn the exercise loop. `exercises/` is a copy of `src/` with every kernel body replaced by `// TODO: your code here` (made by `scripts/make_exercises.py`). The build only reads `src/`, so to work an exercise you copy it over the solution on your `study` branch:
   ```bash
   cp exercises/kernels/v2_lds_tiled.hip src/kernels/v2_lds_tiled.hip
   # ...write the bodies between BEGIN/END SOLUTION...
   cmake --build build-cpu -j
   ./build-cpu/gemm_bench --kernel v2 --test           # just this kernel
   ctest --test-dir build-cpu -R gemm                  # the CTest wrapper (gemm_correctness)
   ```
   The blanked kernel fails the test (output is NaN-poisoned, so skipped work is caught). While you work, **do not look at the solution**. When you are done or truly stuck, compare against history:
   ```bash
   git diff HEAD -- src/kernels/v2_lds_tiled.hip       # yours vs the solution
   git show HEAD:src/kernels/v2_lds_tiled.hip          # the solution itself
   git checkout HEAD -- src/kernels/v2_lds_tiled.hip   # restore the solution
   ```
   Tile-size constants and launch geometry helpers stay outside the markers, so you keep the same constants. Writing the `launch_*` function (grid and block math) is part of the exercise.
4. Tests for engine kernels: `ctest --test-dir build-cpu -R llm` (runs `llm_kernels`, `llm_tiny_e2e`, `llm_cli_smoke`). Note `llm_tiny_e2e` also depends on `gemm_fp32_best`, which is v4, so a blanked v4 breaks it too.
5. Optional but recommended: the `hip-stage-tutor` skill (in `.claude/skills/`) runs a tutor loop per stage: it asks for your predicted speedup first, reviews by questioning, then quizzes you. It refuses to hand over the solution while you work. Its prediction step writes into `docs/stages/vN.md`, so do it on the `study` branch.

Weekly rhythm that works for 8 to 10 hours: about 3 hours reading, 4 hours on the exercise, 1.5 hours teach-back (explain out loud or on paper to an imaginary interviewer), 1 hour on the question bank.

---

## Week 1: the ground floor and v0 to v2

**Read:** `include/hip_common.hpp`, `include/gemm.hpp`, `include/reference.hpp` and `src/reference.cpp`, `tests/test_reference.cpp`, `src/bench.hip` (focus on `run_one`), `src/kernels/v0_naive.hip`, `v1_coalesced.hip`, `v2_lds_tiled.hip`, and `docs/stages/v0.md` to `v2.md`. PROJECT_EXPLAINED sections 2 and 5.1 to 5.3.

**Exercises:**
1. Rewrite `v0`, then `v1`, then `v2` from the exercise files. Test each with `./build-cpu/gemm_bench --kernel vN --test`.
2. Write your own tiny CPU matmul and use `check_gemm` on it (`tests/test_reference.cpp` shows how); then break it on purpose (swap a B index) and watch it fail.
3. Mutation drill, predict first, then run: in `v2`, (a) delete the first `__syncthreads()`, (b) change the zero-fill to leave edge elements uninitialised, (c) remove `row < M` from the final store. Write down which `--test` shapes you expect to fail and why. (On HIP-CPU a missing barrier may or may not fail, which is itself a lesson about what CPU testing cannot see.)

**Concepts to master:** grid/block/thread and index math; why the grid is rounded up and guarded; row-major addressing; coalescing (v0 vs v1); what a barrier guarantees; LDS tiling and the bytes-per-FLOP argument; why the error metric normalises by the sum of absolute products; why C is NaN-poisoned.

**Checkpoint (teach-back):** on paper, draw a 4x4 tile version of v2 for an 8x8 problem and trace one output element. Predict (write in `docs/stages/vN.md` under Prediction) the v0 to v1 and v1 to v2 speedups and your reasoning. Answer questions 1 to 5 and 30 below without notes.

## Week 2: v3 and v4

**Read:** `src/kernels/v3_reg_blocked.hip`, `v4_vec_dbuf.hip`, stage pages v3 and v4, PROJECT_EXPLAINED 2.4, 2.5, 5.4, 5.5.

**Exercises:**
1. Rewrite `v3` (get correctness first with the strided mapping, then add the transposed, padded A tile).
2. Rewrite `v4` in two passes: first with scalar loads and double buffering only (all the `kVec*` flags false), then add the `float4` paths and the runtime legality check in `launch_v4`.
3. Mutation drills: set `kV3PadA` to 0 and note that tests still pass (it is a performance bug, not a correctness bug; the profiler's bank-conflict counter is how you would see it); in `v4`, call `store_slab(cur)` instead of `cur ^ 1` and predict the failure; pass a misaligned pointer (`dA + 1`) to `launch_v4` in a scratch test and confirm the scalar fallback still gives correct results.
4. Run `./build-cpu/gemm_bench --kernel v4 --test --sizes 512,1x1x1,5x3x2,257x255x254` (the alignment and tail cases from the SPEC).

**Concepts:** register blocking and arithmetic intensity per LDS read; transposed LDS layout; padding vs bank conflicts; strided versus blocked thread-to-output mapping and what each does to LDS reads and C stores; float4 legality (base alignment *and* stride divisibility *and* never reading past the matrix); the double-buffer hazard analysis; occupancy limits from VGPRs and LDS.

**Checkpoint:** draw the timeline of one loop iteration of v4 (load next, compute current, store other, barrier) and say exactly which hazard each line avoids. Questions 6 to 10.

## Week 3: FP16, matrix cores, the vendor baseline, profiling

**Read:** `src/kernels/fp16_staging.inc`, `v4_fp16.hip`, `v5_matrix_core.hip`, `v5_mfma.inc`, then skim `v5_wmma_rdna3.inc` and `v5_wmma_nvidia.inc`, `hipblas_baseline.hip`, `registry.hip`, `half_host.hpp`; `scripts/profile_amd.sh`, `profile_nvidia.sh`, `plot_results.py`; stage page v5. PROJECT_EXPLAINED 2.6, 5.6, 5.7, 6.A.

**Exercises:**
1. Rewrite `v4_fp16` (CPU-testable, and it exercises the same staging as v5).
2. **The offline v5 exercise.** You cannot run MFMA on the CPU, so emulate it. In a scratch C++ file, write `mfma_16x16x16(a, b, c)` that takes per-lane registers (64 lanes, 4 halves for A and B, 4 floats for C) laid out *exactly* per the table in `v5_mfma.inc`, computes the result, and distributes D back to lanes. Then write your own `v5_mma_slab` and `v5_store` against your emulation and check against a plain matmul. If the layout table is wrong or you misread it, you find out here, with no GPU.
3. Hand-derive, for lane 37 of a wave64, which A element, B element and D elements it holds (answer: `l % 16 = 5`, `l / 16 = 2`, so A[5][8..11], B[8..11][5], D[8..11][5]). Do the same for the WMMA RDNA3 table.
4. If you have access to an MI300X or an RTX card: `gemm_bench --kernel v5 --test`, then the benchmark, then `scripts/profile_*.sh v4 4096`. Record everything in `results/` and the stage pages. If something fails, do not "fix until green"; find out whether the layout, the alignment or the LDS indexing is wrong.

**Concepts:** why FP16 inputs with FP32 accumulation; what one MFMA computes and why lane layout is the whole difficulty; wave64 vs wave32; why `fp16_staging.inc` is shared; the row-major/column-major transpose trick for hipBLAS; roofline reading; what each profiler counter means.

**Checkpoint:** Questions 11 to 22. Be able to say, without notes, what is unverified about v5.

## Week 4: the engine's kernels

**Read:** `src/llm/kernels/llm_kernels.hpp` and each kernel file (`embedding`, `layernorm`, `linear`, `elementwise`, `attention_naive`, `attention_fused`), `tools/gpt2_numpy.py` (the whole thing; it is the clearest statement of the model, about 100 lines), `tests/test_llm_kernels.hip`. PROJECT_EXPLAINED 6.2 to 6.6.

**Exercises** (test with `ctest --test-dir build-cpu -R llm_kernels`; copy from `exercises/llm/kernels/` the same way as before):
1. `layernorm`: the two-pass block reduction with a shared-memory tree.
2. `gemv_xw` (one thread per output column, x staged through LDS) and `gemv_rowdot`.
3. `attention_naive`: three kernels.
4. `attention_fused`: online softmax with the `(m, l)` merge tree. Before coding, do the merge by hand for two partial results, and prove to yourself that a thread with no keys is a no-op.
5. Extra: in a scratch Python file, implement online softmax on a list of numbers and check it equals the ordinary softmax.

**Concepts:** the exact shape of every tensor in a layer; the Conv1D `[in, out]` layout; causal mask as "only rows that exist"; why `kv_store` runs before attention; numerically stable softmax; why GEMV differs from GEMM and why decode is memory-bound; the roofline argument with the 0.5 FLOP/byte figure.

**Checkpoint:** from a blank page, write the pseudo-code of one transformer layer with shapes. Questions 23 to 29.

## Week 5: the engine, the host side, and proof of correctness

**Read:** `src/llm/engine.hpp`, `engine.hip` (all of it, it is short), `sampler.cpp`, `main.hip`; `src/io/safetensors.cpp`, `tokenizer.cpp` (especially `match_at`, `bpe_ids`), `gpt2_config.cpp`; `tools/make_tiny_model.py`, `compare_logits.py`, `export_reference_logits.py`; `tests/test_llm_e2e.hip`, `test_tokenizer.cpp`, `test_safetensors.cpp`.

**Exercises:**
1. Trace one decode step by hand: list every kernel launch in `forward_rows` for `T = 1`, with its input and output buffers. Count them.
2. Rewrite `Sampler::sample` (top-k with temperature) in your own words and compare with `sampler.cpp`.
3. Write your own `match_at` (the GPT-2 pre-tokenizer) from the regex in the comment, in a scratch file, and test it against `tests/data/tokenizer_cases.txt` using the existing test harness structure. The tricky rule is `\s+(?!\S)`: a whitespace run followed by a non-space gives up its last character.
4. Mutation drills: swap the order of `launch_kv_store` and attention, or use `pos0 + t` vs `pos0 + t + 1` in the causal limit, and predict which of tests (a) to (e) in `llm_tiny_e2e` fail.
5. Explain why the tiny-model generator insists on a top-1 vs top-2 margin of at least 1e-2.

**Concepts:** prefill vs decode paths and why both exist; TTFT and decode tokens/s as defined in SPEC 6.4; why validation uses an independent float64 numpy implementation; what the tiny model can and cannot prove; BPE; safetensors layout.

**Checkpoint:** Questions 30 to 33. Do a 20-minute spoken walkthrough of "typed text to generated text".

## Week 6 (optional buffer, strongly recommended): hardware, CI, and mock interviews

1. **Hardware run.** Get time on an MI300X (AMD Developer Cloud credits) or your own GPU. In order: build for the GPU, `ctest`, `gemm_bench --kernel v5 --test`, the real-GPT-2 comparison (SPEC section 10, H3), benchmarks, profiles. Fill placeholders only from `results/*.json`. Any surprise is *material for the interview*, not a problem to hide.
2. **CI.** Read `ci.yml`, `bench.yml`, `check_regression.py`. Run `python3 scripts/check_regression.py --self-test`. Create a baseline with `--update-baseline`, then fake a 10% drop in a copy of the JSON and watch it fail. Measure your GPU's run-to-run noise (`{{run_to_run_noise_pct}}`) and decide whether 7% is sensible. Register a self-hosted runner if you want the full pipeline. Questions 34 to 37.
3. **Mock interviews.** Have a friend (or the tutor skill) pick random questions from the bank and push on follow-ups. Aim: every question answerable in under 90 seconds, with a number or a file path attached. Questions 38 to 40 are the "process and honesty" ones: practise them aloud.
4. Fill in the "Owner prediction" and "Explanation" sections of each stage page: *prediction versus what the profiler showed* is the story that goes in the blog.

If you only have 4 weeks, fold week 6's hardware run into week 3 and skip the optional exercises marked "extra" or "optional".

---

## Interview question bank (40)

Answers are deliberately concise. Where a number depends on hardware it is a `{{placeholder}}`; replace it with your measured value, never an invented one. Cross-reference: the file named in each answer is where to look to defend it.

### A. GEMM stages

**1. Why is the naive kernel (v0) slow?**
Each FMA loads two floats (8 bytes) for 2 FLOPs, so 4 bytes/FLOP with no reuse beyond whatever the caches catch. In this repo v0 also maps `threadIdx.x` to the row, so neighbouring lanes read A addresses K floats apart (uncoalesced) and write scattered C elements. It is memory-bound by a wide margin. (`v0_naive.hip`)

**2. What does v1 change, and why does it help when the arithmetic is identical?**
Only the thread-to-element mapping: `threadIdx.x` moves to the column, so a wave's loads of B and stores of C are consecutive addresses and merge into a few cache-line transactions, and the A load is a broadcast. Same instructions, far fewer memory transactions (the header estimates about 16x fewer lines per FMA). It is still memory-bound because nothing is reused. (`v1_coalesced.hip`)

**3. In the v2 tile loop there are two `__syncthreads()`. What does each protect against?**
The first guarantees the whole tile is in LDS before any thread reads it (read-after-write). The second guarantees every thread has finished reading the tile before the next iteration overwrites it (write-after-read). Removing the second is a data race: usually "mostly right", occasionally wrong, and it may not show up on HIP-CPU. (`v2_lds_tiled.hip`)

**4. How does tiling cut global memory traffic? Give the number for v2.**
A TxT block loads each element of its A and B tiles once from global memory but uses each T times. For T = 32, global traffic per FMA drops from 8 bytes to 8/32 = 0.25 bytes, i.e. 0.125 bytes/FLOP instead of 4. The reuse moves from hoped-for caching to a scratchpad you control.

**5. Why zero-fill out-of-range elements in LDS instead of branching in the inner loop?**
The inner loop then has no bounds checks, so it is uniform and fast, and padded zeros contribute nothing to the sum, so edge tiles are correct automatically. Bounds are checked once at load time and once at the final store. This is how every stage stays correct for any M, N, K >= 1.

**6. What is register blocking and why 4x4 per thread in v3?**
Each thread computes a small tile of C in registers. Per k step it reads 4 A and 4 B values from LDS (8 reads) and performs a 4x4 outer product (16 FMAs), so 4 FLOPs per LDS read versus 1 in v2. 4x4 is a compromise: 16 accumulators plus 8 operands keeps register use low enough for good occupancy while making the inner loop FMA-heavy. (`v3_reg_blocked.hip`)

**7. Why is `As` padded by one float in v3? Explain the bank conflict.**
The A tile is stored transposed, `As[kk][r]`, and consecutive threads vary `kk`. With 64 floats per row the stride is 64 words, and 64 mod 32 is 0, so lanes land in the same LDS bank (conflict, accesses serialised). Padding to 65 makes lane `kk` hit bank `kk`. It is a performance fix only; correctness is unaffected, so tests cannot catch its absence, but the profiler's LDS bank-conflict counter can. (Bank count is 32 x 4 bytes on GCN/CDNA; confirm in the ROCm docs for your target.)

**8. When is a `float4` load legal in v4, and what does `launch_v4` check?**
When the pointer is 16-byte aligned and the row stride is a multiple of 4 floats, so every row starts aligned and a vector never straddles the end of a row or the matrix. `launch_v4` tests `aligned16(ptr) && (K % 4 == 0)` for A, `(N % 4 == 0)` for B and C, and dispatches one of four template instantiations; tails use scalar loads. The reason it matters: misaligned vector access can fault or crawl on a GPU while HIP-CPU would happily accept it. (`v4_vec_dbuf.hip`)

**9. Walk through the v4 double-buffering and say why one barrier per slab is enough.**
Each iteration: issue global loads for slab s+1 into registers, compute slab s from LDS buffer `cur`, store the registers into `cur ^ 1`, then one barrier. The store is safe because `cur ^ 1` was last read in iteration s-1, which ended in a barrier. The barrier publishes the new buffer and guarantees everyone finished reading `cur` before iteration s+1 overwrites it. The global latency overlaps with the FMAs, *if* the compiler schedules the loads early and delays the wait (verify in the profiler, do not assume).

**10. Where does v4 sit on the roofline, and why go to matrix cores next?**
Per the header: compute-bound on FP32 vector FMAs at large sizes; the remaining stalls are the barrier and occupancy. The FP32 FMA peak is a hard ceiling for scalar code, so the next gain has to come from different hardware: matrix cores, which do 16x16x16 = 4096 FMAs per wave instruction. The realised fraction of peak for v4 is `{{v4_fp32_pct_of_peak}}`.

### B. AMD hardware

**11. Wave64 versus wave32: what changes in code?**
Instinct (CDNA) executes 64 threads per wavefront; RDNA and NVIDIA run 32. Code must never hard-code the width (this repo uses shared-memory trees or `warpSize`), block sizes should be multiples of 64 to avoid idle lanes, and matrix-instruction lane layouts differ (MFMA: wave64 with 4 elements per lane; RDNA3 WMMA: wave32 with duplicated halves). The repo's `.inc` files fix the width inside `HIPGPT2_MC_*` regions only.

**12. What is LDS, and how does it relate to CUDA shared memory?**
Local Data Share is the on-chip, software-managed scratchpad shared by the threads of one workgroup, AMD's counterpart of CUDA `__shared__`. It has banks, so access patterns matter (conflicts), and it needs barriers for cross-thread visibility. Its size per CU limits how many workgroups are resident (occupancy); v4 uses about 17 KiB per block.

**13. What does `v_mfma_f32_16x16x16_f16` do, and what is the hard part?**
One wave-wide instruction computes a 16x16x16 matrix multiply-accumulate: FP16 A and B fragments, FP32 accumulator, 4096 FMAs. The hard part is the operand layout across lanes: lane `l` holds `A[l%16][4*(l/16)+j]`, `B[4*(l/16)+j][l%16]`, and produces `D[4*(l/16)+j][l%16]` for j = 0..3. Getting any index wrong gives silently wrong numbers, which is why the repo documents the table in `v5_mfma.inc` and why it has to be tested on hardware.

**14. Why FP16 inputs but FP32 accumulation?**
FP16 halves bytes moved and is what matrix cores consume at full rate; but summing thousands of FP16 products in FP16 loses accuracy (the `v0_fp16` comment mentions roughly 3 digits at K = 4096). So products accumulate in FP32. The harness feeds the reference the same rounded inputs so the check measures only accumulation error (tolerance 1e-3).

**15. What is the MI300X, and how does it differ from an RX 7900?**
MI300X is AMD's CDNA3 datacenter accelerator (`gfx942`): wave64, MFMA matrix cores, HBM memory with multi-TB/s bandwidth, 304 compute units per the repo's comments (verify against AMD's spec sheet). RX 7900 is consumer RDNA3 (`gfx1100`): wave32, GDDR memory, WMMA instructions with a different layout, and a different ROCm support level. That is why the repo has separate `v5` paths and a CMake mapping from architecture to `HIPGPT2_MC_*`.

**16. What limits occupancy on AMD, and how do you check it?**
VGPRs per thread, LDS per workgroup, and workgroup size. More resident waves hides latency but is not automatically faster (a high-register tile like v3/v4 may win with lower occupancy). Check with `rocprof-compute` (`scripts/profile_amd.sh`), which reports achieved occupancy and the limiting resource.

**17. How does HIP relate to CUDA, and how does this repo run on both?**
HIP is a CUDA-like C++ API; most CUDA code maps one-to-one. The same source compiles for AMD via ROCm or for NVIDIA via HIP-on-CUDA, and in this repo also against HIP-CPU. Platform differences are isolated to `hip_common.hpp` macros and the `HIPGPT2_MC_*` matrix-core paths (MFMA vs `nvcuda::wmma`). Kernels launch with `hipLaunchKernelGGL`, not `<<<>>>`.

### C. Profiling and the roofline

**18. Explain the roofline model and place v0, v4 and decode on it.**
Attainable FLOP/s = min(peak compute, arithmetic intensity x memory bandwidth). v0 (0.25 FLOP/byte) and decode GEMV (0.5 FLOP/byte) are far left of the ridge: memory-bound. A large tiled GEMM moves right and can become compute-bound. The achieved point for each kernel is `{{kernel_roofline_points}}` from `rocprof-compute`'s roofline plot.

**19. How would you profile v4 on an MI300X?**
`scripts/profile_amd.sh v4 4096`, which runs `rocprof-compute profile` (with roofline) then `analyze`. Look at VALU utilisation, LDS bank conflicts, occupancy and its limiter, L1/L2 hit rates, achieved HBM bandwidth, and where the point sits on the roofline. Compare each against the header's "PREDICTED BOTTLENECK" and record which prediction was wrong. (Flags are version-dependent: check `--help`.)

**20. Your v4 reaches only `{{v4_pct_of_hipblas}}` of hipBLAS. How do you investigate?**
Establish the ceiling first (hardware peak and hipBLAS achieved). Then profile for the limiter: low VALU utilisation with stalls points to latency/occupancy; high LDS conflicts to layout; low bandwidth to access patterns. Typical gaps for a hand kernel: tile sizes not tuned per architecture, no matrix cores, imperfect instruction scheduling. Change one thing at a time and re-measure.

**21. What is the arithmetic intensity of a big square GEMM?**
2N^3 FLOPs over a minimum of about 3N^2 elements moved (read A, read B, write C), so with 4-byte floats about N/6 FLOP/byte. It grows with N, which is why large GEMMs can be compute-bound while GEMV (fixed 0.5 FLOP/byte) never is.

**22. How does the harness benchmark fairly?**
Correctness is checked first (and a kernel that fails is not timed). 3 warm-up launches, then 5 to 100 timed launches adaptively aiming for about a second, timed with device events around the launch, with the **median** reported (robust to outliers) plus the minimum. The same seeded inputs are used for every kernel, and JSON includes device, git sha and `verified`. Not controlled: GPU clock/thermal state; on shared cloud machines expect noise. (`src/bench.hip`)

### D. The engine

**23. What is the KV-cache, why use it, and how big is it?**
Per layer, the keys and values of all earlier tokens are stored so each decode step computes only one new row and attends to the cache, making a step O(length) instead of recomputing the whole sequence. Layout: `[n_ctx, d]` per layer for K and V, row = absolute position. For GPT-2 small that is 2 x 1024 x 768 x 4 bytes per layer (about 6.3 MB), roughly 75 MB over 12 layers. `--no-kv-cache` recomputes everything and is the correctness baseline.

**24. Why does decode use GEMV rather than GEMM?**
Decode feeds one row, so every weight is read once for a single multiply-add: 0.5 FLOP/byte, far left of the ridge. Tiling cannot create reuse that does not exist; the right goal is streaming the weights at full bandwidth with coalesced loads, so `gemv_xw` is simple (one thread per output column). Prefill has P rows sharing every weight, so it is a real GEMM and uses the ladder's `gemm_fp32_best`.

**25. What bounds decode speed and how could you improve it?**
Memory bandwidth: roughly `bandwidth / bytes of weights read per token` (about 500 MB of FP32 weights per step; your ceiling is `{{decode_bandwidth_ceiling_tok_s}}`, measured is `{{decode_tok_per_s}}`). Improvements: quantise weights (INT8/FP8, an unimplemented stretch goal) to cut bytes, batch several sequences to reuse weights, fuse kernels (over a hundred launches per token), split-K so small GEMVs fill the whole GPU. Known issues in the code's own comments: only N/64 blocks per layer, so a large GPU is under-occupied.

**26. Naive versus fused attention: what differs?**
Naive is three launches communicating through a global `[H][T][C]` score buffer (scores, softmax, weighted sum), with several passes of global traffic. Fused (decode only) is one launch with one block per head, scores in LDS (max 1024 keys), online softmax, so global traffic is only q, K, V and y. It is limited by launching only 12 blocks for GPT-2 small; flash-decoding (several blocks per head with an `(m, l)` merge) is the next step and is not implemented. Prefill always uses the naive path.

**27. Explain online softmax.**
Keep a running `(m, l)` = (max so far, sum of `exp(s - m)`). A new score `s` gives `m' = max(m, s)`, `l' = l*exp(m - m') + exp(s - m')`. The rescale factor re-expresses the old sum relative to the new max, so one pass yields the same result as the two-pass stable softmax. Two partial results merge by the same rule, and since the merge is associative a block's threads can each handle a subset of keys and combine in a tree. Example: scores 2, 5, 1 give `l = 1.0681` either way.

**28. What is the tied LM head and what does it cost?**
GPT-2 reuses the token-embedding matrix `wte` as the output projection: `logits[v] = dot(final_hidden, wte[v])`. `gemv_rowdot` computes one dot per vocabulary row (50257 rows, about 154 MB of weights read per generated token), which is a large share of the decode bandwidth. Only the needed rows are computed (the last prompt row in normal mode).

**29. Define TTFT and decode tokens/s as this repo measures them.**
`ttft_ms`: wall time from the start of prefill until the first generated token is available on the host (including copying logits and sampling). `decode_tok_per_s = (gen_tokens - 1) / decode_seconds`, wall time over all later steps including host copy and sampling, 0 if fewer than two tokens. The PyTorch baseline script uses the same definitions so the comparison is fair. (SPEC 6.4)

**30. How do you test kernels for awkward sizes?**
The harness uses shapes like `1x1x1`, `1x7x5`, `17x17x17`, `333x257x129`, `64x1000x3`, `3x5x1000` (primes, thin, non-multiples of every tile), poisons C with NaN so skipped elements fail, and checks against a double-precision reference. Engine kernels are tested with d = 48, H = 3, T in {1, 2, 13}, V = 301. On a GPU, larger shapes up to 2049x1023x515 are added.

**31. How did you validate the engine?**
Layered: each kernel against C++ references; the whole model on a tiny random model against an independent float64 numpy implementation (prefill logits, 16 greedy tokens, fused and naive attention, token-by-token decode vs prefill, no-cache path), all within 1e-3 with identical tokens; the generator picks a seed with a top-1/top-2 margin of at least 1e-2 so float noise cannot flip a token. Then real GPT-2 against PyTorch (`tools/compare_logits.py`, target 1e-2 plus identical greedy tokens), which `{{real_gpt2_max_abs_err}}` will record once run. The tiny model cannot expose 12-layer numerical behaviour; say so.

**32. How does the tokenizer work, and how do you know it matches OpenAI's?**
Pre-split text with the GPT-2 regex (contractions; optional space plus a run of letters, digits or punctuation; whitespace with a lookahead rule), map bytes to printable characters, then repeatedly merge the lowest-ranked adjacent pair from `vocab.bpe` and look up `encoder.json`. The regex is hand-coded using generated Unicode tables; correctness is tested by id-for-id comparison with OpenAI's `encoder.py` over a fixture (emoji, CJK, whitespace edge cases, 200 random strings) plus round-trip decoding.

**33. Why is the prompt-length limit and the fused-attention 1024-key limit not a problem for GPT-2?**
GPT-2's context is 1024, equal to the fused kernel's LDS score array (`kFusedAttnMaxKeys`), and the engine falls back to naive attention if `n_ctx` is larger. Prompt plus generation is clamped to `n_ctx`, with a warning.

### E. CI and regression design

**34. Describe the regression gate.**
`check_regression.py` compares current JSON with committed baselines per `(device, kernel, dtype, M, N, K)`, failing if GFLOPS fall below baseline x (1 - T/100) (default T = 7); for the engine it fails on decode tokens/s dropping or TTFT rising by more than T%; any `verified: false` fails; missing baseline passes with a notice unless `--require-baseline`. Limits: the threshold must sit above run-to-run noise (`{{run_to_run_noise_pct}}`), baselines are per device and only as good as the last update, and the check says nothing about *why*.

**35. Why CPU CI on every push but GPU benchmarks on a self-hosted runner?**
Hosted runners have no GPU, so they can prove correctness (HIP-CPU) and compile-check the ROCm build, never speed. A self-hosted runner on your GPU gives real numbers but is your own machine: `bench.yml` therefore triggers only on pushes to `main` and manual dispatch, not on pull requests, so untrusted code from forks never runs on it.

**36. Why report the median, publish history even when the gate fails, and require `verified: true`?**
Median resists outliers (clock ramp, interference). Publishing history even on failure means the chart shows the regression instead of hiding it; the job still ends red. A fast but wrong kernel must never become the new baseline, so unverified results fail the gate and `--update-baseline` refuses them.

**37. How would you make the gate less flaky?**
Measure noise and set T above it; use more iterations or take the best of several runs; pin clocks if the platform allows; keep one dedicated runner; compare against a rolling baseline rather than one absolute number; treat a single failure as "re-run" and a repeated one as real.

### F. HIP-CPU, process, honesty

**38. Why did you use HIP-CPU, and what can it not show?**
There was no GPU, and HIP-CPU runs the same kernels on the CPU, so logic, indexing, edge handling and the entire engine are tested everywhere and in free CI. It cannot show speed, alignment faults, wave-size behaviour, hardware races, matrix-core results (compiled out), or hipBLAS. The repo's portability rules exist to shrink that gap.

**39. What is verified and what is not?**
Verified on CPU: all 9 CTest tests (v5 MFMA lane layout via emulation, GEMM v0 to v4 and FP16 variants, tokenizer, loader, engine on the tiny model, scripts). Not verified: v5 on any matrix-core hardware, hipBLAS, real GPT-2 weights, every performance number and profiler prediction, the ROCm compile job, GPU benchmark job and Pages publishing. After your hardware run, update this answer with what changed.

**40. How was this built, what did you do, and what would you do next?**
Honest version: an architect model wrote the spec; Opus agents wrote kernels and the engine, Sonnet agents wrote tokenizer, loader, tools and CI, an Opus reviewer audited kernels, with disjoint file ownership so agents could work in parallel, and everything was CPU-verified through HIP-CPU. Your contribution: you set the goals, then worked through the code (rewrote kernels from the exercises, mutation drills, the offline MFMA emulation), ran it on real hardware, and own the measurements. Next steps: transpose B in LDS and enlarge wave tiles for v5; flash-decoding and fused LayerNorm/residual for the engine; INT8/FP8 weight-only quantisation; an FP16 engine path; tune the regression threshold from measured noise.

---

## Final self-test (the day before an interview)

- Draw the factory analogy and map it to the memory hierarchy in one minute.
- Say what each stage v0 to v5 changed, in one sentence each, with the file name.
- Derive bytes/FLOP for v0, v2 and v3, and decode's 0.5 FLOP/byte.
- Whiteboard the v4 loop timeline and its barrier reasoning.
- State lane 37's MFMA elements.
- Explain online softmax with the 2, 5, 1 example.
- Recite what is verified and what is not, including what you ran on real hardware and the measured numbers (filled from `results/`).
- Say the honesty statement out loud once.
