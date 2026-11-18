# PROJECT_EXPLAINED: everything in hip-gpt2, from zero

This document is for you (Ad). It assumes you know C/C++ and basic CS, but nothing about GPUs. After reading it you should be able to say what every folder, file and test in this repo is for, why it was built that way, and what has and has not been proven.

Companion documents: [README.md](../README.md) (overview and build steps), [SETUP.md](SETUP.md) (installing HIP, ROCm, CUDA), [stages/](stages/) (one page per GEMM stage: [v0](stages/v0.md), [v1](stages/v1.md), [v2](stages/v2.md), [v3](stages/v3.md), [v4](stages/v4.md), [v5](stages/v5.md)), [STUDY_GUIDE.md](STUDY_GUIDE.md) (the plan to own the code, plus interview questions), [BLOG.md](BLOG.md), [DEMO_VIDEO_SCRIPT.md](DEMO_VIDEO_SCRIPT.md), and the design contract [../SPEC.md](../SPEC.md).

Performance numbers in this repo were measured on an NVIDIA RTX 4060 (2026-11-18) and come from `results/*.json`. AMD MI300X numbers are still pending. Section 7 says bluntly what is verified and what is not.

---

## 1. The goal, in plain English

The project has three parts:

- **A. A GEMM ladder.** GEMM means GEneral Matrix Multiply: `C = A * B`. It is the operation that neural networks spend most of their time on. We write it six times (v0 to v5), each version faster than the last, and measure each one. The point is to show you understand *why* each trick helps.
- **B. Performance CI.** Every time code is pushed, automated tests check correctness. On a machine with a GPU, benchmarks also run, and the build fails if performance drops by more than a set percentage.
- **C. A GPT-2 inference engine.** A small C++/HIP program that loads the real GPT-2 weights and generates text with no PyTorch involved. It uses our own kernels, including our own GEMM.

Why this matters for an AMD ROCm job: those teams want someone who can write a kernel, profile it, and explain why it is fast. HIP is AMD's CUDA-like language. This repo is your evidence that you can do that.

### The factory analogy

Keep this picture in your head. Every GPU concept below maps onto it.

| Factory | GPU | Where you see it in the repo |
|---|---|---|
| A worker | A **thread** | One thread computes one output element in `src/kernels/v0_naive.hip` |
| A crew of workers who must do the same step at the same moment | A **wavefront** (AMD) or **warp** (NVIDIA): 64 threads on Instinct, 32 on RDNA/NVIDIA | The `wave64` vs `wave32` comments in `src/kernels/v5_*.inc` |
| A department with a shared bench and a foreman | A **block** (workgroup) running on one **compute unit** (CU) | `dim3 block(16, 16)` in `v0_naive.hip` |
| The warehouse: huge, far away, slow to fetch from | **Global memory** (HBM/GDDR) | Every `A[...]`, `B[...]` read in `v0` |
| Parts staged on a table beside the line, shared by the crew | **LDS** (Local Data Share; "shared memory" in CUDA), the `__shared__` arrays | `As`, `Bs` in `src/kernels/v2_lds_tiled.hip` |
| Parts in a worker's hands | **Registers** | `float acc[4][4]` in `src/kernels/v3_reg_blocked.hip` |
| The main product | **GEMM** | Everything in `src/kernels/` |
| A specialised machine that does a whole 16x16 block of multiply-adds in one go | **Matrix core** (MFMA on AMD Instinct, WMMA on RDNA3/NVIDIA) | `src/kernels/v5_mfma.inc` |

The whole ladder is about one question: *how do we stop workers waiting for the warehouse?* v0 sends every worker to the warehouse for every single part. v2 has the crew fetch parts together and stage them on the table. v3 lets each worker keep several parts in hand. v4 starts fetching the next batch before the current one is used up. v5 replaces a row of workers with one specialised machine.

---

## 2. GPU fundamentals needed for this repo

### 2.1 Execution model: grid, block, wave, thread

You launch a **kernel** (a function that runs on the GPU). The launch creates a **grid** of **blocks**; each block holds many **threads**. Every thread runs the same code but sees different built-in indices (`threadIdx`, `blockIdx`, `blockDim`).

Inside a block, threads are grouped into waves of 64 (Instinct) or 32 (RDNA, NVIDIA) that execute one instruction together ("lockstep"). If threads of a wave take different branches, the hardware runs both paths with some threads switched off. That is called **divergence**.

This repo launches with a macro, never the `<<<...>>>` syntax (SPEC rule 3.5.1; the HIP-CPU backend compiles with plain g++ which cannot parse it):

```cpp
// src/kernels/v0_naive.hip
const dim3 block(kV0BlockX, kV0BlockY);               // 16 x 16 = 256 threads per block
const dim3 grid((M + kV0BlockX - 1) / kV0BlockX, ...); // enough blocks to cover C, rounded UP
hipLaunchKernelGGL(gemm_v0_kernel, grid, block, 0, stream, dA, dB, dC, M, N, K);
```

The grid is rounded up, so edge threads may fall outside the matrix. That is why every kernel guards with `if (row < M && col < N)`. This is the "correct for any M, N, K >= 1" requirement, and half of the bugs in GPU code live there.

**Barriers.** `__syncthreads()` makes every thread in a block wait until all have arrived. Two rules from the SPEC: every thread must reach every barrier (no early `return` before one), and never hard-code the wave width. The reductions in `src/llm/kernels/layernorm.hip` use a shared-memory tree with barriers, which is correct for wave32 and wave64 alike.

### 2.2 Memory hierarchy

From slowest/biggest to fastest/smallest:

1. **Global memory** (HBM on MI300X, GDDR on consumer cards). Gigabytes, high latency (hundreds of cycles), high bandwidth only when accessed well.
2. **Caches (L2, L1)** automatically hold recently used data. Helpful but not under your control.
3. **LDS** (`__shared__`): tens of KB per compute unit, shared by the threads of one block, you control it explicitly. Fast, but needs a `__syncthreads()` between writing and reading.
4. **Registers**: private per thread, fastest. Each thread has a limited budget; using too many reduces how many threads can run at once (see occupancy).

Example in the repo: `v2_lds_tiled.hip` declares `__shared__ float As[32][32]` and `Bs[32][32]` (8 KiB of LDS). `v3_reg_blocked.hip` keeps `float acc[4][4]` in registers.

### 2.3 Coalescing

When the 64 (or 32) threads of a wave read memory in one instruction, the hardware merges the requests into as few cache-line fetches as possible. If neighbouring threads read neighbouring addresses, that is one or two fetches (**coalesced**). If they read addresses far apart, it is up to one fetch per thread.

The clearest example is the contrast between `v0` and `v1`, which have the *same algorithm* and differ only in the thread-to-element mapping:

```cpp
// v0_naive.hip: threadIdx.x -> ROW. Lane i reads A[row+i][k], K floats apart.
const int row = blockIdx.x * blockDim.x + threadIdx.x;
const int col = blockIdx.y * blockDim.y + threadIdx.y;

// v1_coalesced.hip: threadIdx.x -> COLUMN. Lane i reads B[k][col+i], adjacent.
const int col = blockIdx.x * blockDim.x + threadIdx.x;
const int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Note that v0 deliberately uses the "wrong" mapping, so the v0-to-v1 jump measures coalescing alone. Matrices are row-major (`C[row*N + col]`), so the fast-moving index in memory is the column.

### 2.4 Occupancy

A compute unit keeps many waves resident and switches between them when one stalls waiting for memory. **Occupancy** is how many waves are resident compared with the maximum. It is limited by registers per thread, LDS per block and block size. More waves means better latency hiding, but not automatically more speed.

In the repo: `__launch_bounds__(kV3Threads)` tells the compiler the block size so it can budget registers. The `v4` header notes "17 KiB LDS and ~40 VGPRs per thread" as the occupancy limiters. The 40-VGPR figure is an estimate in a comment, not a measurement; the profiler's occupancy counter is how you check it.

### 2.5 Bank conflicts

LDS is split into banks (on AMD GCN/CDNA, 32 banks of 4 bytes; confirm for your target in the ROCm docs). A wave's LDS access is fast when its lanes hit different banks (or the same address, which is a broadcast). When several lanes hit the same bank at different addresses, the accesses are serialised: a **bank conflict**.

Concrete example, `v3_reg_blocked.hip`. The A tile is stored transposed: `As[kk][r] = ...`. Consecutive threads vary `kk` (0..15), so their addresses differ by the row length. With 64 floats per row, the stride is 64 words, and `64 mod 32 = 0`, so all 16 lanes land in one bank: a 16-way conflict. The fix is padding:

```cpp
constexpr int kV3PadA = 1;
__shared__ float As[kV3BK][kV3BM + kV3PadA];   // stride 65 words: lane kk -> bank kk
```

### 2.6 Arithmetic intensity and the roofline

**Arithmetic intensity** = FLOPs performed per byte moved from memory. Every processor has two ceilings: peak compute (FLOP/s) and peak memory bandwidth (bytes/s). The **roofline model** says achievable speed is `min(peak compute, intensity x bandwidth)`. The crossover intensity ("ridge point") separates memory-bound kernels (left) from compute-bound ones (right).

Numbers you can derive yourself from the code:

- `v0`: each FMA loads 2 floats (8 bytes) for 2 FLOPs: **4 bytes/FLOP**, which is 0.25 FLOP/byte. Hopelessly memory-bound.
- `v2` (32x32 tile): each loaded element is reused 32 times: 0.125 bytes/FLOP.
- `v3` (64x64 tile): 1/16 byte/FLOP from global memory.
- Decode GEMV in the engine (`src/llm/kernels/linear.hip`): every 4-byte weight is used for one multiply-add (2 FLOPs): **0.5 FLOP/byte**. Far left of any GPU's ridge, so decode speed is set by memory bandwidth, not compute.
- A large square GEMM of size N has intensity about N/6 FLOP/byte (2N^3 FLOPs over about 3N^2 x 4 bytes), which is why big GEMMs can be compute-bound.

The profiling scripts (section 6) produce roofline plots, and `~56 FLOP/byte (FP32, datasheet)` and the achieved points are placeholders until you run them on a GPU.

---

## 3. How the project was built, and why

### 3.1 The pipeline

| Step | Who | What | Why this tier |
|---|---|---|---|
| Plan | **Fable 5.2** "architect" | Wrote [SPEC.md](../SPEC.md): file tree with one owner per file, exact header declarations, acceptance commands, JSON schemas, risk table | Planning is where the most capable (and most expensive) model pays off. Typing code is not. |
| GEMM ladder, engine core | **Opus** agents | `src/kernels/`, `src/bench.hip`, `src/llm/` (kernels, engine, sampler, CLI) | Index math, races, numerics and matrix-core lane layouts are where cheaper models make subtle mistakes. |
| Tokenizer, loader, tools, CI, scripts | **Sonnet** agents | `src/io/`, `tools/`, `.github/`, `scripts/`, `site/` | Well-specified host code and plumbing, done reliably at lower cost. |
| Audit | **Opus** reviewer (read-only) | Reviewed the kernels independently | A reviewer that did not write the code can catch what the author missed. Its findings and what was fixed are in [REVIEW.md](REVIEW.md). |
| Docs | **Sonnet** writers | README, stage pages, this document, study guide, blog and video drafts | Reads the code first; numbers stay as placeholders. |
| Table filling | **Haiku** `doc-filler` (defined, to be used once results exist) | Fills `{{placeholders}}` from `results/*.json` only | Pure copy-from-JSON work. |

The definitions of these roles are in `.claude/agents/` (section 8).

### 3.2 Why disjoint file ownership

SPEC section 2 gives every file exactly one owner (`[GEMM]`, `[LLM-CORE]`, `[LLM-IO]`, `[INFRA]`, `[DOCS]`, plus `[ORCH]` for the frozen shared headers and CMake). Agents ran in parallel, and two agents editing one file would silently overwrite each other. Shared interfaces (`include/gemm.hpp`, `safetensors.hpp`, `tokenizer.hpp`, `gpt2_config.hpp`, `hip_common.hpp`) were written first and frozen, so the GEMM, engine and tokenizer work could proceed independently and still link together. The CMake "component switches" (`HIPGPT2_BUILD_GEMM/IO/LLM`) and empty-glob skipping let each agent build while the others' parts did not exist yet.

### 3.3 Why HIP-CPU: testing without a GPU

There was no GPU in the build environment. **HIP-CPU** (github.com/ROCm/HIP-CPU) is a header-only implementation of the HIP runtime API that runs kernels on CPU threads. The same `.hip` sources compile as ordinary C++ against it, so every kernel's logic could be tested on a laptop and in free GitHub CI.

What this proves: index math, bounds guards at awkward sizes, tile bookkeeping, algorithm correctness, buffer layouts, the whole engine end to end on the tiny model.

What it cannot prove (SPEC risk R3):

- Misaligned `float4` loads that fault or slow down on a real GPU.
- Wave-size assumptions (HIP-CPU's "warp" is not 32 or 64).
- Races that happen to work under CPU scheduling.
- Out-of-bounds reads that land in mapped memory.
- Anything about speed.
- Anything inside `#if HIPGPT2_MC_*` blocks (MFMA/WMMA), which are compiled out on CPU.

To shrink that gap, the SPEC imposed portability rules on every `.hip` file: static `__shared__` only, no literal warp width, every thread reaches every barrier, never read past the end of A or B, and explicit alignment checks before using `float4`.

### 3.4 Teaching conventions baked into the code

Every kernel file starts with four labelled comment lines (`STAGE`, `CHANGES`, `WHY FASTER`, `PREDICTED BOTTLENECK`). Kernel bodies sit between `// BEGIN SOLUTION` and `// END SOLUTION` markers, which is how `scripts/make_exercises.py` produces `exercises/` (the same files with bodies replaced by `// TODO: your code here`). Tile sizes are named `constexpr int` constants with a one-line reason.

---

## 4. A guided tour of the repo

```
SPEC.md  MASTER_PROMPT.md  CMakeLists.txt  README.md
include/    shared headers (the contracts between components)
src/kernels/   the GEMM ladder             (Phase A)
src/bench.hip  GEMM test + benchmark harness
src/reference.cpp  CPU correctness checker
src/io/     tokenizer, safetensors loader, config reader   (Phase C, host side)
src/llm/    engine, sampler, CLI, and src/llm/kernels/     (Phase C)
tests/      the CTest programs and tests/data (tiny model, tokenizer cases)
tools/      Python: numpy reference model, tiny-model generator, PyTorch scripts
scripts/    benchmark runner, regression gate, history, plots, profilers, exercise maker
.github/workflows/  ci.yml, bench.yml          (Phase B)
site/       static page that charts the performance history
assets/gpt2/  GPT-2 vocabulary files (encoder.json, vocab.bpe)
exercises/  copy of src with kernel bodies blanked
perf/baselines/  committed "known good" benchmark JSON (empty until you create it)
results/    benchmark output JSON (generated)
docs/       documentation
```

### 4.1 `include/`

| File | Purpose |
|---|---|
| `hip_common.hpp` | Includes HIP headers, defines exactly one of `HIPGPT2_PLATFORM_CPU/AMD/NVIDIA`, plus `HIP_CHECK` (abort on any HIP error) and `HIPGPT2_REQUIRE` (host-side invariant check). |
| `gemm.hpp` | The contract every GEMM launcher obeys: row-major, `C[MxN] = A[MxK] * B[KxN]`, correct for any size >= 1, asynchronous on a stream. Declares the kernel registry (`gemm_fp32_kernels`, `gemm_fp16_kernels`) and `gemm_fp32_best`. |
| `reference.hpp` | `check_gemm`, the CPU checker (section 4.2). |
| `safetensors.hpp`, `tokenizer.hpp`, `gpt2_config.hpp` | Pure C++ interfaces for the loader, BPE tokenizer and model config. |

### 4.2 GEMM side

- `src/reference.cpp`: computes the exact answer in `double` and returns an error metric `|got - exact| / sum_k |a_ik * b_kj|`. Normalising by the sum of absolute products keeps it sane when a dot product happens to be near zero. A real bug gives errors near 1; honest FP32 rounding gives about 1e-7 to 1e-5. It checks every element when `M*N*K <= 2048^3`, otherwise the four corners plus 4096 random samples.
- `src/bench.hip`: the harness (`gemm_bench`). `--list`, `--test` (awkward shapes: `1x1x1`, `17x17x17`, `333x257x129`, `64x1000x3`, ...), benchmark mode (3 warm-ups, then median of 5 to 100 timed launches), JSON output. Before each correctness run it **poisons C with NaN** (`hipMemset(dC, 0xFF, ...)`), so a kernel that skips elements cannot pass by luck. FP16 kernels get inputs rounded to binary16 and the reference is fed the *rounded* values, so the check measures only the kernel's accumulation error.
- `src/kernels/registry.hip`: two tables of `{name, function, unavailable-reason, tolerance}`. Availability is decided at compile time: `v5` needs a matrix-core path; `hipblas*` needs hipBLAS. `gemm_fp32_best` forwards to `launch_v4`, which is what the engine calls.
- `src/kernels/half_host.hpp`: host-side float to binary16 conversion (round to nearest even), so the harness does not depend on a GPU for it.
- `src/kernels/hipblas_baseline.hip`: the vendor reference (section 5.7).

### 4.3 Engine side

- `src/io/safetensors.cpp`: reads the safetensors format (8-byte little-endian header length, JSON header, raw data), validates shapes, dtype and offsets, and throws on truncated or inconsistent files.
- `src/io/tokenizer.cpp` plus `unicode_tables.inc`: the GPT-2 BPE tokenizer (section 6.1). `unicode_tables.inc` is generated by `tools/gen_unicode_tables.py`.
- `src/io/json_mini.hpp`: a small JSON reader shared by the three `.cpp` files.
- `src/io/gpt2_config.cpp`: reads `config.json`.
- `src/llm/kernels/*.hip`: embedding, layernorm, linear layers (GEMV variants, bias epilogues), elementwise ops, naive and fused attention. Declared in `llm_kernels.hpp`.
- `src/llm/engine.hpp/.hip`: `Gpt2Engine`: weights on device, preallocated buffers, KV-cache, `prefill`, `decode`, `generate`.
- `src/llm/sampler.*`: greedy and top-k sampling on the host.
- `src/llm/main.hip`: the `gpt2` command-line program.

### 4.4 The build system: three backends, one CMake

`CMakeLists.txt` supports three ways to build the same source:

| Backend | Configure | What happens |
|---|---|---|
| **HIP-CPU** | `-DGEMM_BACKEND=cpu -DHIP_CPU_DIR=<checkout>` | `.hip` files are compiled as C++ (`LANGUAGE CXX`) against the HIP-CPU headers; links TBB and pthreads. No GPU needed. Matrix-core code is off. |
| **NVIDIA** | `-DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=89` | HIP on top of CUDA (RTX 4060 is sm_89). `v5` uses `nvcuda::wmma`. |
| **AMD** | `-DCMAKE_HIP_ARCHITECTURES=gfx942` (MI300X) or `gfx1100` (RX 7900) | Native ROCm. `gfx942` selects MFMA; `gfx1100` plus `-DHIPGPT2_ENABLE_RDNA3_WMMA=ON` selects the experimental WMMA path. |

CMake stamps every target with `HIPGPT2_GIT_SHA` and exactly one of `HIPGPT2_MC_NONE`, `_MFMA`, `_WMMA_RDNA3`, `_WMMA_NVIDIA`. Code may only branch on these and the `HIPGPT2_PLATFORM_*` macros. Other options: `HIPGPT2_ENABLE_HIPBLAS` (AUTO/ON/OFF), `HIPGPT2_LLM_GEMM` (`ladder` = the engine uses `gemm_fp32_best`; `internal` = its own fallback GEMM), `HIPGPT2_BUILD_GEMM/IO/LLM` switches.

Targets: `reference`, `gemm_kernels`, `gemm_bench`, `llm_io`, `llm_kernels`, `llm_engine`, `gpt2`, and one executable per test. Source lists use `CONFIGURE_DEPENDS` globs, so adding a `.hip` file to `src/kernels/` needs no CMake edit.

### 4.5 The test suite (what each CTest test proves)

Run: `ctest --test-dir <build> --output-on-failure`. There are nine tests: the eight below plus `v5_mfma_emulation` (added after the review; it compiles `tests/emulation/v5_mfma_emulator.cpp` with clang and checks the v5 MFMA lane layout on the CPU, see [REVIEW.md](REVIEW.md)). All nine passed on the HIP-CPU backend in the environment where this document was written.

| CTest name | Program | What it proves |
|---|---|---|
| `reference_checker` | `tests/test_reference.cpp` | The *checker itself* is right: a correct C passes, injected bugs (a wrong element, NaN, etc.) fail. If the checker is wrong, every "PASS" means nothing. |
| `gemm_correctness` | `gemm_bench --kernel all --test` | Every available GEMM kernel is correct on awkward shapes, with NaN-poisoned output. On CPU, `v5`, `hipblas`, `hipblas_fp16` are listed as unavailable and skipped. |
| `tokenizer` | `tests/test_tokenizer.cpp` | The C++ tokenizer's ids equal OpenAI's `encoder.py` ids for every case in `tests/data/tokenizer_cases.txt`, and `decode(encode(s)) == s`. |
| `safetensors` | `tests/test_safetensors.cpp` | The loader reads the tiny model, matches shapes and first/last values in `manifest.txt`, and throws on wrong shape, missing name, truncated file, bad header length, nonexistent path. |
| `llm_kernels` | `tests/test_llm_kernels.hip` | Each engine kernel matches a plain C++ double-precision reference on awkward sizes (d=48, H=3, T in {1,2,13}, V=301). Fused attention matches naive attention. |
| `llm_tiny_e2e` | `tests/test_llm_e2e.hip` | Whole engine vs numpy float64 on the tiny model (section 6.7). |
| `llm_cli_smoke` | `gpt2 --model tests/data/tiny --tokens 5,17,42 --max-new-tokens 4 --quiet` | The CLI runs end to end and exits 0. |
| `scripts_selftest` | `scripts/check_regression.py --self-test` | The regression gate's decision rules behave as specified. |

CI additionally runs `scripts/make_exercises.py --self-test` and `bash -n` on the shell scripts.

---

## 5. Phase A walkthrough: the GEMM ladder

All stages compute `C[M x N] = A[M x K] * B[K x N]`, row-major, FP32 unless stated. Detailed per-stage pages are in [docs/stages/](stages/).

### 5.1 v0: naive (`src/kernels/v0_naive.hip`)

**What it is.** One thread per element of C. Thread (row, col) loops over K and accumulates in a register.

```cpp
float acc = 0.f;
for (int k = 0; k < K; ++k) acc += a[k] * b[(size_t)k * N];
C[(size_t)row * N + col] = acc;
```

**Why it is slow.** 4 bytes of memory traffic per FLOP, no reuse except what caches catch. Mapping `threadIdx.x -> row` also makes neighbouring lanes read addresses K floats apart (uncoalesced).

**What can go wrong.** Forgetting the `row < M && col < N` guard; `int` overflow in `row * K` for big matrices (hence the `(size_t)` casts everywhere).

### 5.2 v1: coalesced (`v1_coalesced.hip`)

**What changed.** Only the mapping: `threadIdx.x -> col`, block is 32 wide by 8 tall. Now a wave's loads of B and stores of C touch contiguous 128-byte lines, and the load of `A[row][k]` is one address broadcast to all lanes of a row.

**Why faster.** Same instruction count, far fewer memory transactions (the header estimates roughly 16x fewer cache lines per FMA). **Still limited by:** no data reuse, so it remains memory/L2-bound.

### 5.3 v2: LDS tiling (`v2_lds_tiled.hip`)

**What changed.** A 32x32 block cooperatively copies a 32x32 tile of A and of B into `__shared__` memory, syncs, then each thread does 32 FMAs from LDS, then moves to the next K-tile.

```cpp
As[ty][tx] = (row < M && ak < K) ? A[(size_t)row * K + ak] : 0.f;   // zero-fill edges
Bs[ty][tx] = (bk < K && col < N) ? B[(size_t)bk * N + col] : 0.f;
__syncthreads();                                   // tile complete before anyone reads
for (int k = 0; k < kV2Tile; ++k) acc += As[ty][k] * Bs[k][tx];
__syncthreads();                                   // everyone done before next overwrite
```

**Why faster.** Each element fetched from global memory is reused 32 times. Global traffic falls from 4 to 0.125 bytes/FLOP. **Zero-filling** out-of-range elements means the inner loop has no bounds checks, which also makes edge tiles correct for free.

**What can go wrong.** Removing either barrier (a data race: fast but wrong, and flaky). Putting a `return` before a barrier. **Still limited by:** two LDS reads per FMA, so LDS bandwidth and instruction issue; two barriers per K-step.

### 5.4 v3: register blocking (`v3_reg_blocked.hip`)

**What changed.** Block tile grows to 64x64 with K-step 16; each of 256 threads computes a **4x4** sub-tile held in `float acc[4][4]`. Per k, a thread reads 4 A values and 4 B values from LDS into registers and does 16 FMAs (an outer product).

```cpp
for (int i = 0; i < 4; ++i) a[i] = As[kk][ty + i * 16];
for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx + j * 16];
for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
```

**Why faster.** 8 LDS reads now feed 16 FMAs (4 FLOPs per LDS read, versus 1 in v2), and a 64x64 tile reuses each global element 64 times. Two details to defend: the A tile is stored **transposed** (`As[k][m]`) so both operands are read along a row, and it is **padded by 1 float** to avoid the bank conflict from section 2.5. The output mapping is strided (thread owns rows `ty + 16*i`, cols `tx + 16*j`) so LDS reads of B are stride-1 and global writes of C are coalesced.

**What can go wrong.** Register pressure (the 16 accumulators plus operands); wrong strided index in the epilogue; the `static_assert`s guard the tile-to-thread split. **Still limited by:** global-load latency, because loads for the next slab are only issued after the barrier.

### 5.5 v4: vectorized loads and double buffering (`v4_vec_dbuf.hip`)

**What changed.** Same 64x64x16 tile, plus:

1. **`float4` loads.** One 16-byte load instead of four 4-byte loads, and `float4` LDS reads and C stores. Legal only when the address is 16-byte aligned *and* the row stride is a multiple of 4 floats. `launch_v4` checks this at runtime and picks one of four template instantiations (`kVecA`, `kVecB`), falling back to scalar loads otherwise:

```cpp
const bool vec_a = v4_aligned16(dA) && (K % kV4Vec == 0);
const bool vec_b = v4_aligned16(dB) && (N % kV4Vec == 0);
```

2. **Double buffering.** Two LDS buffers. Each loop iteration: (1) issue global loads for slab s+1 into registers, (2) compute slab s from LDS, (3) store the registers into the *other* LDS buffer, (4) one barrier.

```cpp
if (has_next) load_slab((s + 1) * kV4BK);      // global -> registers, overlaps the math below
/* ... 16 x 16 FMAs from As[cur], Bs[cur] ... */
if (has_next) store_slab(cur ^ 1);             // registers -> the other buffer
__syncthreads();                               // one barrier per slab (v3 had two)
```

Why step 3 is safe without an extra barrier: the other buffer was last *read* in iteration s-1, which ended in a barrier. Why step 4 suffices: it publishes the freshly stored buffer and guarantees everyone has finished reading `cur` before iteration s+1 overwrites it. Being able to draw this hazard analysis on a whiteboard is a classic interview test.

3. The output mapping becomes **blocked** (thread owns 4 consecutive rows and columns) so its 4 operands are contiguous in LDS and readable as one `float4`.

**What can go wrong.** A misaligned `float4` access (invisible on HIP-CPU, a fault or slowdown on GPU); reading past the end of A/B on the K tail (the vector path only triggers when `K % 4 == 0`, so a vector never straddles the edge); the compiler not actually overlapping the loads with the math (check the profiler, do not assume). **Expected limit:** FP32 FMA throughput itself.

### 5.6 FP16 stages and v5: matrix cores

`v0_fp16.hip` is "v0 with halves": FP16 inputs, **FP32 accumulation**, FP32 output (the FP16 reference stage). `v4_fp16.hip` is the v4 recipe for halves, with a 64x64 tile and K-slabs of 32, using staging code in `fp16_staging.inc`.

**Why a shared `fp16_staging.inc`.** `v5` cannot run on the CPU, but everything that feeds it can. By including the *same* global-to-LDS code in `v4_fp16` and `v5`, every CPU test of `v4_fp16` also tests v5's address math, zero padding and vector-load rules. Only v5's inner product and epilogue are hardware-only.

**What v5 changes** (`v5_matrix_core.hip` plus one of three `.inc` files). Only the inner product. The block is 4 waves in a 2x2 grid; each wave owns a 32x32 piece of C made of 2x2 tiles of 16x16; each 16x16x16 step is a single matrix instruction. On AMD Instinct (`v5_mfma.inc`):

```cpp
acc.c[i][j] = __builtin_amdgcn_mfma_f32_16x16x16f16(a[i], b[j], acc.c[i][j], 0, 0, 0);
```

One MFMA does 16x16x16 = 4096 FMAs per wave. The catch is the **lane layout**: each of the 64 lanes must hold specific elements of the operand matrices. From the table in `v5_mfma.inc` (taken from AMD's Matrix Instruction Calculator):

| Operand | Lane `l` holds (j = 0..3) |
|---|---|
| A (16x16) | `A[ l % 16 ][ 4*(l/16) + j ]`: one row, 4 consecutive k |
| B (16x16) | `B[ 4*(l/16) + j ][ l % 16 ]`: one column, 4 consecutive k |
| D (16x16) | `D[ 4*(l/16) + j ][ l % 16 ]`: one column, 4 consecutive rows |

The code reads A as one 8-byte vector from `As[m][k]`, B as four scalar reads from `Bs[k][n]`, and writes C from the D layout with bounds checks. The other two paths are `v5_wmma_rdna3.inc` (wave32, experimental, lanes 16-31 must duplicate lanes 0-15) and `v5_wmma_nvidia.inc` (`nvcuda::wmma` fragments whose per-lane layout is opaque, so results go through a per-warp LDS scratch tile).

**Why faster (expected).** Matrix-core FP16 peak is several times the FP32 vector peak (the header quotes MI300X figures; verify from AMD's spec sheet).

**What can go wrong, and this is the most likely failure in the repo.** A wrong lane layout produces wrong numbers (the `--test` would catch it, on hardware only). B fragments are strided 2-byte LDS reads, so expect a feed bottleneck and possibly a 2-way bank conflict (the current header predicts one because of the 36-word padded row and tells you to check the LDS bank-conflict counter; it names "transpose B in LDS" as a next step). The 8-byte A read needs alignment (row stride 80 bytes; `kk + kq` multiple of 4 halves). **None of this has run.** See section 7.

**Tolerance.** FP16-input kernels use tolerance 1e-3 against a reference fed the same *rounded* inputs. `hipblas_fp16` uses 5e-2 because hipBLAS's `hgemm` produces FP16 output and may accumulate in FP16.

### 5.7 The hipBLAS baseline (`hipblas_baseline.hip`)

The vendor library is the yardstick: "we reached X% of rocBLAS/cuBLAS". hipBLAS is **column-major**, ours is row-major. A row-major matrix is, byte for byte, the column-major matrix of its transpose, so we ask for `C^T = B^T * A^T` by swapping operands and dimensions. No data moves:

```cpp
hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
```

The handle is created once lazily; `hipblasSetStream` is called per launch. For FP16 there is a scratch buffer and a conversion kernel back to FP32. It compiles only with hipBLAS present, and has never been compiled here.

---

## 6. Phase B and Phase C walkthroughs

### 6.A Phase B: performance CI

**On every push** (`.github/workflows/ci.yml`, job `cpu-tests`, GitHub-hosted Ubuntu): install cmake, g++, TBB; clone HIP-CPU at a **pinned commit**; configure with `-DGEMM_BACKEND=cpu`; build; `ctest`; run the script self-tests. This proves correctness, never speed. A second job, `rocm-compile` (marked `continue-on-error`), builds inside a ROCm container for `gfx942`. It is the only automated way the MFMA path ever gets *compiled* before you have hardware (compiled, not run). The container image tag has a comment saying you must confirm it exists.

**GPU benchmark job** (`bench.yml`, `runs-on: [self-hosted, gpu]`, on push to `main` and manual dispatch): you register a runner on your GPU machine, and set repository variables `HIP_CMAKE_FLAGS`, optionally `GPT2_MODEL_DIR` and `PERF_THRESHOLD_PCT`. It builds, runs `ctest`, then `scripts/run_benchmarks.sh` (`gemm_bench --kernel all --sizes 1024,2048,4096`, and the engine on `scripts/bench_prompt.txt` if a model dir is set). Then the regression check, then history publishing, then it uploads results as an artifact. The regression step is `continue-on-error` so history still gets published, and a final step fails the job if the gate failed.

**The regression gate** (`scripts/check_regression.py`, rules in SPEC 8.3). It compares JSON files in `results/` against known-good copies in `perf/baselines/<device_slug>/`:

- GEMM: for each `(device, kernel, dtype, M, N, K)` present in both, **fail if `gflops_current < gflops_baseline * (1 - T/100)`**.
- LLM: fail if `decode_tok_per_s` drops more than T%, or `ttft_ms` rises more than T%.
- Any `verified: false` in the current results fails, even with no baseline.
- Entries in the baseline but missing now: warning (failure with `--strict`).
- No baseline for a device: notice and pass, unless `--require-baseline`.
- `T` = `--threshold`, else env `PERF_THRESHOLD_PCT`, else **7**.
- Exit codes: 0 pass, 1 regression, 2 usage error.

You create baselines with `--update-baseline` (it refuses if anything is unverified) and commit them. **A design caveat to be honest about:** 7% is a guess. Run-to-run noise on your GPU (thermals, clocks, shared cloud machines) may be bigger or smaller. Measure it (`3`), then set the threshold above it, or the gate will cry wolf.

**History and chart.** `scripts/update_history.py` appends one entry per `(git_sha, device)` to `data/history.json` (only verified GEMM rows and the hip-gpt2 LLM result), de-duplicated. The workflow publishes `site/index.html` and `site/chart.js` plus that JSON to the `gh-pages` branch. `chart.js` draws GFLOPS-per-kernel against commit at the largest common size, plus TTFT and tokens/s. You must enable Pages for the `gh-pages` branch once.

**Other scripts.** `scripts/plot_results.py` makes PNG charts (GFLOPS vs size, and a ladder bar chart with speedups). `scripts/profile_amd.sh <kernel> <size>` runs `rocprof-compute profile` then `analyze` (counters: LDS bank conflicts, occupancy, cache hit rates, bandwidth, plus the roofline). `scripts/profile_nvidia.sh` does the Nsight Compute (`ncu`) equivalent. The comments in these scripts tell you to verify the flags against your installed version because profiler CLIs change between releases; they have never been run.

### 6.B Phase C: from typed text to generated text

The engine generates GPT-2 small: 12 layers, 768-wide embeddings, 12 heads of 64, vocabulary 50257, context 1024. Numbers come from `config.json`; the code itself works for any size, which is how the 2-layer tiny model reuses it.

#### 6.1 Tokenizer: text to ids (`src/io/tokenizer.cpp`)

GPT-2 uses **byte-level BPE** (byte pair encoding):

1. **Pre-split** the text into chunks with the GPT-2 regex: contractions (`'s`, `'t`, ...), an optional space plus a run of letters, or of digits, or of punctuation, and whitespace runs. The code implements this by hand in `match_at` (no `std::regex`) using sorted Unicode range tables for letters, numbers and whitespace.
2. **Bytes to printable characters.** Each UTF-8 byte maps to a visible Unicode character (`build_byte_tables`), the same table as OpenAI's `encoder.py`.
3. **Merge.** Start with one symbol per character. Repeatedly find the adjacent pair with the lowest merge rank in `vocab.bpe` and merge it, until no listed pair remains. Look up the final pieces in `encoder.json` to get ids. A per-word cache avoids repeating work.

Correctness is tested against OpenAI's own `encoder.py` on a fixture covering emoji, CJK, contractions, runs of whitespace, `<|endoftext|>` as plain text and 200 random strings.

#### 6.2 Embeddings (`embedding.hip`)

`x[t] = wte[token[t]] + wpe[position[t]]`: one row of the token-embedding table (what the token means) plus one row of the position-embedding table (where it is). `x` is a `[T, 768]` matrix, T = number of tokens being processed.

#### 6.3 One transformer layer (12 of them), in `Gpt2Engine::forward_rows`

```
a   = LayerNorm(x)                      launch_layernorm      (ln_1)
qkv = a @ W_attn + b_attn               linear()  [T, 3*768]  q | k | v
store k, v rows into the KV-cache       launch_kv_store
y   = attention(q, cache)               naive or fused
x  += y @ W_proj + b_proj               linear() then launch_residual_add
a   = LayerNorm(x)                      (ln_2)
f   = GELU(a @ W_fc + b_fc)             linear() with GELU, [T, 4*768]
x  += f @ W_mlp_proj + b_mlp_proj
```

Points worth knowing:

- **LayerNorm** normalises each row to mean 0, variance 1, then scales and shifts (`layernorm.hip`). One block per row, two shared-memory tree reductions (mean, then variance of the centred values: the two-pass method is more numerically stable than E[x^2] - E[x]^2). eps = 1e-5 inside the square root.
- **Weights are stored `[in, out]`** (Hugging Face "Conv1D" layout), so the layer is `y = x @ W + b` directly, with no transpose. This is why `gemv_xw` has "one thread per output column": consecutive threads read consecutive columns of one row of W, which is coalesced.
- **QKV.** One matrix multiply produces queries, keys and values side by side. Head h owns columns `[h*64, (h+1)*64)` of each.
- **GELU** (tanh approximation) is the MLP's nonlinearity, `gelu_tanh` in `llm_kernels.hpp`.
- **Residual connections** (`x += ...`) add each block's output back to its input.

#### 6.4 Attention and the KV-cache

For each head and each query position p: score against every earlier position u <= p (the **causal mask**: a token may not see the future), scale by 1/sqrt(64), softmax over u, then take the weighted average of the V rows.

The **KV-cache** (`engine.hpp`) is two buffers per layer, `K_i` and `V_i`, each `[n_ctx, 768]`. Row p holds the key/value of the token at position p. Without a cache, generating token 100 would recompute the keys and values of all 99 earlier tokens, every layer, every step. With it, each step computes one new row and *appends* it (`launch_kv_store`, which runs *before* attention so the query can see itself). The `--no-kv-cache` flag exists as a debug baseline and as a correctness check (test (e) below).

Both prefill and decode share one definition: query row r sits at absolute position `pos0 + r` and attends to cache rows `0..pos0+r`.

**Naive attention** (`attention_naive.hip`) is three kernels connected through a global score buffer of shape `[H][T][C]`: (1) scores, (2) causal softmax per row (subtract the row max, exponentiate, divide by the sum), (3) weighted sum of V. Simple and independently testable, but the scores make several round trips through global memory. The buffer is `H*C*C` floats, about 50 MB at GPT-2's sizes (12 x 1024 x 1024 x 4 bytes).

**Fused attention** (`attention_fused.hip`, decode only, T = 1) is one launch, one block per head, no global score buffer: scores live in a `__shared__` array of at most 1024 floats (the engine falls back to naive if n_ctx > 1024). The trick is the **online softmax**.

*Online softmax, simply.* A normal softmax needs the maximum of all scores first (to avoid overflow in `exp`), then the sum of `exp(s - max)`. That seems to need two passes over the data. But you can do it in one pass by keeping a running pair `(m, l)`: `m` = max seen so far, `l` = sum of `exp(s - m)` so far. When a new score `s` arrives:

```
m' = max(m, s)
l' = l * exp(m - m') + exp(s - m')
```

The factor `exp(m - m')` re-expresses the old sum relative to the new maximum. Tiny example with scores 2, 5, 1: after 2: (m=2, l=1). After 5: m=5, l = 1*e^(2-5) + 1 = 1.0498. After 1: m=5, l = 1.0498 + e^(1-5) = 1.0681. Computed the long way: e^-3 + 1 + e^-4 = 0.0498 + 1 + 0.0183 = 1.0681. Same.

Two partial results `(m1, l1)`, `(m2, l2)` over *different* sets of scores merge with the same rule: `m' = max(m1, m2)`, `l' = l1*exp(m1-m') + l2*exp(m2-m')`. Because merging is associative, each of the block's 128 threads can process its own subset of keys, and a shared-memory tree combines them. A thread that saw no keys contributes `(-FLT_MAX, 0)`, which the rule turns into a no-op (using -FLT_MAX instead of -infinity avoids `inf - inf = NaN`).

Honest limits (from the file's own header): GPT-2 small has only 12 heads, so only 12 blocks launch, leaving a big GPU mostly idle; the real fix is "flash-decoding" (split the sequence across several blocks per head, merge with the same rule). That is a stretch goal, not implemented.

#### 6.5 Final LayerNorm, tied LM head, sampling

After 12 layers: `LayerNorm(x)` (`ln_f`), then **logits** = dot product of the final vector with every row of `wte` (the **tied** LM head: GPT-2 reuses the token-embedding matrix as its output matrix). `gemv_rowdot` computes one dot product per vocabulary row (50257 of them, about 154 MB of weights read per token). Normal mode computes this for the last row only.

Sampling happens on the host (`sampler.cpp`): **greedy** picks the lowest index of the maximum logit; otherwise divide by temperature, keep the top-k, softmax in `double`, and draw with `std::mt19937_64(seed)`. The chosen token is decoded back to bytes (`decode_token`). A UTF-8 character can span several tokens, so `drain_utf8` in `main.hip` holds back an incomplete character until the rest arrives.

#### 6.6 Prefill versus decode, and why decode uses GEMV

- **Prefill** processes all P prompt tokens at once. Each linear layer is `Y[P,N] = X[P,K] @ W[K,N]`: a real GEMM where every weight is reused P times. The engine calls `gemm_fp32_best` (which is v4 from the ladder), then a bias epilogue kernel (`bias_add` / `bias_gelu`). **This is where the ladder gets used.**
- **Decode** processes one new token. Each linear layer is `y[N] = x[K] @ W[K,N]`: a matrix times a *vector* (GEMV). Every 4-byte weight is read once and used for one multiply-add: 0.5 FLOP/byte, far left of the roofline. A tiled GEMM buys nothing; the job is to stream the weights from memory at full bandwidth. So `gemv_xw` is deliberately simple, with coalesced loads and bias/GELU fused in.

So an upper bound on decode speed is roughly `memory bandwidth / bytes of weights read per token`. The `engine.hpp` comment sizes the weights at about 500 MB of FP32 per step; your ceiling for your GPU is `~544 tokens/s`, and the gap between that and the measured rate is what an optimisation story would be about. Known reasons for a gap (named in `linear.hip`): too few blocks to fill a big GPU (N/64 blocks per layer), and per-kernel launch overhead (over a hundred kernel launches per decoded token: about 10 per layer times 12 layers).

#### 6.7 How correctness is proven

**Level 1, kernels** (`llm_kernels` test): each kernel vs plain C++ on awkward sizes; fused attention vs naive attention.

**Level 2, the whole model on a tiny random model.** `tools/make_tiny_model.py` creates a seed-fixed GPT-2 with L=2, d=64, H=4, V=300, C=64 (random weights, committed in `tests/data/tiny/`), and `tools/gpt2_numpy.py`, an independent float64 numpy implementation, produces reference logits and tokens. The generator searches for a seed where the top-1 vs top-2 logit gap is at least 1e-2 at every greedy step, so tiny float differences cannot flip a token and make the test flaky. `test_llm_e2e` then checks, with tolerance 1e-3 on logits:

- (a) prefill logits at all 11 prompt positions;
- (b) 16 greedy tokens with fused decode attention: identical tokens, logits within tolerance;
- (c) the same with naive attention;
- (d) feeding the prompt one token at a time through the decode path gives the same final logits as one prefill (so GEMM path and GEMV path agree);
- (e) the no-KV-cache path gives the same tokens.

On the CPU run when this was written, measured errors were around 1e-5 (tolerance 1e-3). That is a CPU number and says nothing about GPU rounding.

**Why a tiny model is not enough** (SPEC risk R6): it has 2 layers and small random activations. It cannot reveal numerical trouble that appears only with real 12-layer weights, or key-name quirks of the real checkpoint.

**Level 3, real GPT-2 against PyTorch, on your machine** (not yet done). `tools/export_reference_logits.py` downloads GPT-2 through `transformers`, writes `model.safetensors` (dropping `lm_head.weight`, since it is tied, and the `attn.bias` masks), `config.json`, and PyTorch's logits. The engine dumps its own logits (`--dump-logits ... --dump-mode all`) and `tools/compare_logits.py` checks the max absolute difference (target 1e-2 on logits, plus identical greedy tokens). Commands are in SPEC section 10 (H3) and SETUP.md.

---

## 7. What is verified and what is not (blunt version)

| Item | Status |
|---|---|
| All 9 CTest tests on HIP-CPU (incl. v5 MFMA lane-layout emulation) | **Verified** (passed in the environment where this was written) |
| GEMM v0 to v4, `v0_fp16`, `v4_fp16` logic and edge handling | **Verified on CPU only**: logic is right, nothing known about GPU behaviour |
| Engine, tokenizer, loader, sampler vs numpy/OpenAI references on the tiny model | **Verified on CPU** |
| Script logic (`check_regression.py` self-test, `make_exercises.py` self-test) | **Verified** |
| **v5 NVIDIA WMMA** (`nvcuda::wmma`) | **Verified on RTX 4060**: compiled cleanly, passes all 12 test shapes, 14.8 TFLOPS at 4096. |
| **v5 MFMA (MI300X) and RDNA3 WMMA** | **Not run on hardware.** Compiled device-only, MFMA emulated on CPU. Needs an MI300X / RX 7900. |
| **hipBLAS baselines** | **Verified on RTX 4060** (hipBLAS built with its cuBLAS backend). |
| **float4 alignment behaviour, wave-size assumptions, races** | **Verified on RTX 4060** (wave size 32) via the GPU-only larger test shapes; still unverified on AMD wave64. |
| **Real GPT-2 weights** (key names, 12-layer numerics) | **Verified on RTX 4060**: max abs logit error 3.2e-4, greedy tokens identical for 32 steps. One bug found and fixed in the export script (it dropped `c_attn.bias`). |
| **Performance numbers** | **Measured on RTX 4060** (`results/*.json`). MI300X pending. |
| "WHY FASTER" claims and profiler predictions | **Still hypotheses.** No profiler counters collected yet. The ladder ordering matches the predictions; the fused-attention speedup did not show up on the 4060 at this context length. |
| 7% regression threshold | Measured spread for v4 at 4096 is about 3% over 3 runs, so 7% leaves headroom. |
| `rocm-compile` job container tag | **Verified to exist.** `bench.yml`, the self-hosted runner and profiling scripts have not run yet. |
| The Opus review pass | Happened per the build history; the report is not in the repo. |

First things to do on real hardware, in order: build for your GPU and run `ctest` (the `gemm_correctness` test on a GPU adds 512 to 2049-sized shapes); `gemm_bench --kernel v5 --test`; the H3 GPT-2 comparison; then benchmarks and profiles; then fill placeholders (the `doc-filler` agent or by hand from `results/*.json`).

---

## 8. The master prompt, agents and skills

### 8.1 `MASTER_PROMPT.md`

A prompt you paste into Claude Code with Fable 5.2 as the main model, run on the machine that has the GPU. It makes Fable an **orchestrator** that plans and delegates but writes no code itself. It works in two situations: repo already built (skips to verification, benchmarking and filling in results) or empty folder (rebuilds everything from SPEC, then does the hardware phases). Its steps: detect state, plan (architect writes SPEC), build (parallel agents with disjoint files), real hardware (validate, benchmark, profile, GPT-2 check, fill docs), finish (final tests, `NEXT_STEPS.md`). Its core rules: delegate, give each agent a self-contained brief, never invent numbers, results only from `results/`.

### 8.2 `.claude/agents/`

| Agent | Model | Job |
|---|---|---|
| `architect` | fable | Writes or updates SPEC.md only. Never writes source. |
| `kernel-engineer` | opus | GPU kernels and numerics; must follow the portability rules. |
| `systems-engineer` | sonnet | Host C++, tokenizer, loaders, CMake, CI, Python tools. |
| `hw-validator` | sonnet | Runs build, tests, benchmarks, profiling on the real GPU. Never edits kernels; reports failures. |
| `reviewer` | opus | Read-only review: correctness, claims vs code, performance red flags. Outputs findings only. |
| `doc-writer` | sonnet | Explanations, study guide, blog, video script. Reads the code first; numbers stay placeholders. |
| `doc-filler` | haiku | Fills placeholders from `results/*.json` only; never writes prose. |

### 8.3 `.claude/skills/`

- **`hip-stage-tutor`**: a tutor loop for one stage: explain the idea with the factory analogy, ask for your predicted speedup, you write the kernel from `exercises/`, it reviews by *asking questions* rather than pasting fixes, you measure, then a five-question quiz of rising difficulty (move on at 4/5). It will not hand you the solution file while you are working an exercise. Use it with [STUDY_GUIDE.md](STUDY_GUIDE.md).
- **`gpu-results-update`**: after any kernel change or on a new GPU: build, `ctest`, benchmarks, plots, refill placeholders, commit with the device name.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **GEMM** | General Matrix Multiply, `C = A*B`. |
| **GEMV** | Matrix times vector. What decode does. |
| **Kernel** | A function that runs on the GPU, launched over many threads. |
| **Thread / block / grid** | One worker / a group sharing LDS and barriers / all blocks of a launch. |
| **Wavefront (AMD), warp (NVIDIA)** | Threads that execute in lockstep: 64 on Instinct, 32 on RDNA/NVIDIA. |
| **CU** | Compute Unit: the AMD hardware block that runs workgroups (SM on NVIDIA). |
| **LDS** | Local Data Share: AMD's on-chip, per-workgroup shared memory (`__shared__`). |
| **VGPR** | Vector general-purpose register, per thread. |
| **Occupancy** | How many waves are resident per CU compared with the maximum. |
| **Coalescing** | Neighbouring lanes reading neighbouring addresses so the hardware merges requests. |
| **Bank conflict** | Several lanes hitting the same LDS bank at different addresses, serialising the access. |
| **Divergence** | Lanes of one wave taking different branches. |
| **Arithmetic intensity** | FLOPs per byte of memory traffic. |
| **Roofline** | Plot/model: speed <= min(peak compute, intensity x bandwidth). |
| **Register blocking** | Each thread computes a small tile of outputs held in registers. |
| **Double buffering** | Loading the next chunk while computing on the current one. |
| **MFMA** | Matrix Fused Multiply-Add: AMD Instinct matrix-core instruction (`gfx90a`, `gfx94x`). |
| **WMMA** | Wave Matrix Multiply-Accumulate: RDNA3's matrix instruction; also NVIDIA's `nvcuda::wmma` API. |
| **gfx942, gfx1100** | AMD GPU architecture names: MI300X (CDNA3), RX 7900 (RDNA3). |
| **HIP** | AMD's CUDA-like C++ GPU programming interface; **ROCm** is the surrounding software stack. |
| **HIP-CPU** | Header-only HIP runtime that runs kernels on the CPU, used for testing. |
| **hipBLAS** | AMD/NVIDIA vendor BLAS wrapper (over rocBLAS or cuBLAS). |
| **FP32 / FP16** | 32-bit and 16-bit floating point. Here FP16 inputs are accumulated in FP32. |
| **BPE** | Byte pair encoding: the tokenizer's merge algorithm. |
| **Token** | The unit of text the model reads and writes (an id into a 50257-word vocabulary). |
| **Prefill / decode** | Processing the whole prompt at once / generating one token at a time. |
| **TTFT** | Time to first token: prompt start to the first generated token on the host. |
| **KV-cache** | Stored keys and values of all earlier tokens, per layer, so each step computes one new row. |
| **Causal mask** | A token may attend only to itself and earlier tokens. |
| **Online softmax** | Computing softmax in one pass with a running `(max, sum)` pair. |
| **Tied LM head** | The output projection reuses the token-embedding matrix. |
| **Conv1D layout** | GPT-2 checkpoints store linear weights as `[in, out]`. |
| **safetensors** | Simple tensor file format: JSON header plus raw bytes. |
| **LGT1** | This repo's logits file format (`LGT1`, rows, vocab, float32 data). |
| **CTest** | CMake's test runner. |
| **Regression gate** | CI check that fails when performance drops by more than T%. |
| **Baseline** | Committed known-good benchmark JSON the gate compares against. |
| **`{{placeholder}}`** | A number to be filled in from `results/*.json` after a real GPU run. |
