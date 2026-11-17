# hip-gpt2 — Implementation Spec (v1)

Single source of truth for all agents. If something is not specified here, choose the simplest option that passes the acceptance commands and note it in your final report. Never edit a file you do not own (section 2). Paths are relative to the repo root `/home/claude/hip-gpt2`.

## 1. Goal and definition of done

C++17/HIP repo with (A) a GEMM optimization ladder v0..v5, (B) performance CI, (C) a GPT-2 small inference engine built on the ladder. No GPU exists in the build environment: everything is verified with HIP-CPU; matrix-core code and all performance numbers are verified later by the owner on real hardware (section 10).

Done when all of the following hold on a clean checkout:

- [ ] D1 `cmake -S . -B build -DGEMM_BACKEND=cpu -DHIP_CPU_DIR=/home/claude/deps/HIP-CPU && cmake --build build -j` succeeds with no warnings-as-errors issues and no edits outside owned files.
- [ ] D2 `ctest --test-dir build --output-on-failure` passes: `reference_checker`, `gemm_correctness`, `tokenizer`, `safetensors`, `llm_kernels`, `llm_tiny_e2e`, `llm_cli_smoke`, `scripts_selftest`.
- [ ] D3 `./build/gemm_bench --list` shows FP32 kernels `v0 v1 v2 v3 v4 hipblas` and FP16 kernels `v0_fp16 v4_fp16 v5 hipblas_fp16`; on CPU `v5`, `hipblas`, `hipblas_fp16` are listed as unavailable with a reason, all others PASS `--test`.
- [ ] D4 `./build/gemm_bench --kernel all --sizes 64,128 --iters 3 --out /tmp/r` writes one schema-valid JSON per available kernel (section 8).
- [ ] D5 C++ tokenizer ids equal OpenAI `encoder.py` ids on every case in `tests/data/tokenizer_cases.txt`; `decode(encode(s)) == s`.
- [ ] D6 Tiny-model end-to-end: C++ logits vs numpy reference, max abs error <= 1e-3, greedy tokens identical; naive and fused attention agree; prefill and token-by-token decode agree.
- [ ] D7 `./build/gpt2 --model tests/data/tiny --tokens 5,17,42 --max-new-tokens 8 --bench-out /tmp/r` prints tokens + stats and writes a schema-valid LLM JSON.
- [ ] D8 `python3 scripts/check_regression.py --self-test` passes; a synthetic 10% drop fails with exit 1, a 3% drop passes (threshold 7%).
- [ ] D9 `.github/workflows/ci.yml` and `bench.yml` are valid YAML and reference only commands defined in this spec.
- [ ] D10 `grep -rn '<<<' src tests` returns nothing; every `__global__` kernel body has SOLUTION markers (section 5.1).
- [ ] D11 Nothing under `__HIP_CPU_RT__`-compiled code hard-codes warp width 32/64.

## 2. File tree and ownership

Owners: `[ORCH]` orchestrator (written before agents start, then frozen), `[GEMM]`, `[LLM-CORE]`, `[LLM-IO]`, `[INFRA]`, `[DOCS]` (later). A directory tag covers everything inside it. Agents create only files inside their own entries.

```
SPEC.md  CMakeLists.txt  .gitignore  results/.gitkeep          [ORCH]
include/hip_common.hpp      platform macros, HIP_CHECK          [ORCH]
include/gemm.hpp            GEMM launchers + registry API       [ORCH]
include/reference.hpp       check_gemm (existing, frozen)       [ORCH]
include/safetensors.hpp     loader API                          [ORCH]
include/tokenizer.hpp       tokenizer API                       [ORCH]
include/gpt2_config.hpp     model config API                    [ORCH]
src/reference.cpp           (existing, frozen)                  [ORCH]
tests/test_reference.cpp    (existing, frozen)                  [ORCH]
src/bench.hip               harness (existing, extend)          [GEMM]
src/kernels/                                                    [GEMM]
  registry.hip              both tables, gemm_fp32_best
  half_host.hpp             host float<->binary16 bit conversion
  v0_naive.hip  v1_coalesced.hip  v2_lds_tiled.hip  v3_reg_blocked.hip  v4_vec_dbuf.hip
  v0_fp16.hip   v4_fp16.hip       v5_matrix_core.hip
  v5_mfma.inc   v5_wmma_rdna3.inc v5_wmma_nvidia.inc   (included by v5_matrix_core.hip)
  hipblas_baseline.hip
src/io/                                                         [LLM-IO]
  safetensors.cpp  tokenizer.cpp  gpt2_config.cpp  json_mini.hpp  unicode_tables.inc
src/llm/                                                        [LLM-CORE]
  kernels/llm_kernels.hpp   launcher declarations for all llm kernels
  kernels/embedding.hip  layernorm.hip  linear.hip (gemv_xw, gemv_rowdot, bias, bias_gelu, fallback gemm)
  kernels/elementwise.hip (residual add, gelu)  attention_naive.hip  attention_fused.hip
  engine.hpp  engine.hip  sampler.hpp  sampler.cpp  main.hip (CLI `gpt2`)
tests/test_tokenizer.cpp  tests/test_safetensors.cpp            [LLM-IO]
tests/data/tokenizer_cases.txt  tests/data/tiny/                [LLM-IO]
tests/test_llm_kernels.hip  tests/test_llm_e2e.hip              [LLM-CORE]
assets/gpt2/{encoder.json,vocab.bpe,LICENSE}                    [LLM-IO]
tools/                                                          [LLM-IO]
  make_tiny_model.py  gpt2_numpy.py  export_reference_logits.py  compare_logits.py
  torch_generation_baseline.py  gen_tokenizer_cases.py  gen_unicode_tables.py
  third_party/encoder.py  requirements.txt
.github/workflows/{ci.yml,bench.yml}                            [INFRA]
scripts/                                                        [INFRA]
  check_regression.py  update_history.py  plot_results.py  make_exercises.py
  profile_amd.sh  profile_nvidia.sh  bench_prompt.txt  run_benchmarks.sh
site/{index.html,chart.js}   perf/baselines/.gitkeep            [INFRA]
README.md  docs/ (stages/vN.md, blog draft, demo-video script, study guide)   [DOCS]
```

Rules: files that include HIP headers use the `.hip` extension; pure host code uses `.cpp`. `.inc` files are never compiled directly. Build directories: each agent uses its own (`build-gemm`, `build-llm`, `build-io`, `build-infra`); `build*/` is git-ignored.

## 3. Build and platform contract

### 3.1 CMake options (top-level `CMakeLists.txt`, [ORCH])
| Option | Values (default) | Effect |
|---|---|---|
| `GEMM_BACKEND` | `gpu` (default), `cpu` | `cpu`: `project(hip_gpt2 LANGUAGES CXX)`, every `.hip` source gets `LANGUAGE CXX` (policy CMP0119 NEW adds `-x c++`), `-isystem ${HIP_CPU_DIR}/include`, link `tbb pthread`. `gpu`: additionally `enable_language(HIP)`. |
| `HIP_CPU_DIR` | path (required when `cpu`) | HIP-CPU checkout. |
| `CMAKE_HIP_PLATFORM`, `CMAKE_HIP_ARCHITECTURES` | standard CMake | NVIDIA: `-DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=89`. AMD: `-DCMAKE_HIP_ARCHITECTURES=gfx942` (or `gfx1100`). |
| `HIPGPT2_BUILD_GEMM` / `HIPGPT2_BUILD_IO` / `HIPGPT2_BUILD_LLM` | `ON` | Component switches so agents can build while other components are incomplete. Each component is also skipped (with a `message(STATUS)`) if its source glob is empty. |
| `HIPGPT2_LLM_GEMM` | `ladder` (default), `internal` | `ladder`: `llm_engine` links `gemm_kernels`, defines `HIPGPT2_LLM_USE_LADDER_GEMM=1`. `internal`: engine uses its own fallback GEMM; forced when `HIPGPT2_BUILD_GEMM=OFF`. |
| `HIPGPT2_ENABLE_HIPBLAS` | `AUTO` (default), `ON`, `OFF` | `find_package(hipblas)`; if used, link it to `gemm_kernels` and define `HIPGPT2_HAVE_HIPBLAS=1`. Always off for `cpu`. `ON` + not found = configure error. |
| `HIPGPT2_ENABLE_RDNA3_WMMA` | `OFF` | Enables the experimental gfx11 WMMA path. |
| `GEMM_USE_OPENMP` | `ON` | OpenMP for `reference` only (PRIVATE). |

Global compile definitions set by CMake on every target: `HIPGPT2_GIT_SHA="<short sha|unknown>"` and exactly one of `HIPGPT2_MC_NONE=1`, `HIPGPT2_MC_MFMA=1`, `HIPGPT2_MC_WMMA_RDNA3=1`, `HIPGPT2_MC_WMMA_NVIDIA=1`. Mapping: cpu -> NONE; nvidia -> WMMA_NVIDIA; AMD arch matching `^gfx9(08|0a|4[0-2])` -> MFMA; `^gfx11` and `HIPGPT2_ENABLE_RDNA3_WMMA` -> WMMA_RDNA3; anything else -> NONE. Build type defaults to Release; C++/HIP standard 17.

### 3.2 Targets (globs use `CONFIGURE_DEPENDS`; adding a source file never needs a CMake edit)
| Target | Sources | Links |
|---|---|---|
| `reference` (static) | `src/reference.cpp` | OpenMP (private) |
| `gemm_kernels` (static) | `src/kernels/*.hip` | hipblas (optional) |
| `gemm_bench` | `src/bench.hip` | `gemm_kernels`, `reference` |
| `llm_io` (static, pure C++) | `src/io/*.cpp` | — |
| `llm_kernels` (static) | `src/llm/kernels/*.hip` | — |
| `llm_engine` (static) | `src/llm/*.hip` except `main.hip`, `src/llm/*.cpp` | `llm_kernels`, `llm_io`, `gemm_kernels` (if ladder). Built only if `llm_io` exists. |
| `gpt2` | `src/llm/main.hip` | `llm_engine` |
| `test_reference`, `test_tokenizer`, `test_safetensors`, `test_llm_kernels`, `test_llm_e2e` | `tests/<name>.{cpp,hip}` | resp. `reference`; `llm_io`; `llm_io`; `llm_kernels`; `llm_engine` |

All targets get `include/` on the include path; `llm_*` targets also get `src/llm/`. A test target and its CTest entry are created only if its source file exists.

### 3.3 Platform macros (the only ones code may branch on)
- `HIPGPT2_PLATFORM_CPU` / `HIPGPT2_PLATFORM_AMD` / `HIPGPT2_PLATFORM_NVIDIA` (exactly one defined to 1) and `HIPGPT2_PLATFORM_NAME` (`"cpu"|"amd"|"nvidia"`) from `include/hip_common.hpp`.
- `HIPGPT2_MC_*` (section 3.1) for matrix-core paths; `HIPGPT2_HAVE_HIPBLAS`; `HIPGPT2_LLM_USE_LADDER_GEMM`.
- Do not use `__HIP_CPU_RT__`, `__HIP_PLATFORM_*__`, `__CUDACC__` directly outside `hip_common.hpp`, except the `.inc` matrix-core files which are already selected by `HIPGPT2_MC_*`.

### 3.4 CTest entries (working directory = repo root for all)
| Name | Command |
|---|---|
| `reference_checker` | `test_reference` |
| `gemm_correctness` | `gemm_bench --kernel all --test` |
| `tokenizer` | `test_tokenizer assets/gpt2 tests/data/tokenizer_cases.txt` |
| `safetensors` | `test_safetensors tests/data/tiny` |
| `llm_kernels` | `test_llm_kernels` |
| `llm_tiny_e2e` | `test_llm_e2e tests/data/tiny` |
| `llm_cli_smoke` | `gpt2 --model tests/data/tiny --tokens 5,17,42 --max-new-tokens 4 --quiet` |
| `scripts_selftest` | `python3 scripts/check_regression.py --self-test` (only if Python3 found and the script exists) |

### 3.5 HIP-CPU portability rules (apply to every `.hip` file)
1. Launch only with `hipLaunchKernelGGL(kernel, grid, block, 0, stream, args...)`. Never `<<<>>>`, not even in comments (D10 greps for it).
2. Shared memory: `static`-sized `__shared__` arrays declared inside the kernel only. No dynamic/extern shared memory; the `sharedBytes` argument is always 0.
3. No literal warp width. Block reductions use a shared-memory tree with `__syncthreads()`, or shuffle loops `for (int o = warpSize / 2; o > 0; o /= 2)`. Do not use host `hipDeviceProp_t.warpSize` for logic (HIP-CPU reports 4 on host, 64 on device).
4. `__half`, `__float2half`, `__half2float`, `float4`, `make_float4` are allowed. MFMA/WMMA builtins, `<mma.h>`, inline asm and `__builtin_amdgcn_*` are allowed only inside `#if HIPGPT2_MC_*` regions.
5. Every thread of a block must reach every `__syncthreads()` (no early `return` before a sync; guard work with `if`, not exits).
6. Keep per-thread stack arrays small (<= 256 floats) and CPU test problem sizes small (GEMM <= 512^3; engine: tiny model only).
7. Launchers are asynchronous on the given stream: no `hipStreamSynchronize`/`hipDeviceSynchronize` inside launchers, no per-call `hipMalloc`.

## 4. Shared C++ interfaces ([ORCH] writes these headers; declarations are exact, `// one #include each` lines and the launcher shorthand are expanded into ordinary code)

### 4.1 `include/hip_common.hpp`
```cpp
#pragma once
#include <hip/hip_runtime.h>, <hip/hip_fp16.h>, <cstdio>, <cstdlib>   // one #include each
// Exactly one platform macro, tested in this order:
//   __HIP_CPU_RT__          -> HIPGPT2_PLATFORM_CPU 1,    HIPGPT2_PLATFORM_NAME "cpu"
//   __HIP_PLATFORM_NVIDIA__ -> HIPGPT2_PLATFORM_NVIDIA 1, HIPGPT2_PLATFORM_NAME "nvidia"
//   __HIP_PLATFORM_AMD__    -> HIPGPT2_PLATFORM_AMD 1,    HIPGPT2_PLATFORM_NAME "amd"
//   otherwise #error. HIPGPT2_GIT_SHA defaults to "unknown" if not defined by CMake.
// Aborts with file:line and hipGetErrorString on any HIP error (existing macro, moved here).
#define HIP_CHECK(expr)  /* do { hipError_t _e = (expr); if (_e != hipSuccess) { fprintf(stderr, ...); std::exit(1); } } while (0) */
// Host-side invariant check: prints "check failed: <msg> at file:line" and std::exit(1).
#define HIPGPT2_REQUIRE(cond, msg)  /* do { if (!(cond)) { ...; std::exit(1); } } while (0) */
```

### 4.2 `include/gemm.hpp`
```cpp
#pragma once
#include "hip_common.hpp"
// Contract for every launcher (FP32 and FP16):
//   Row-major. C[M x N] = A[M x K] * B[K x N]. Correct for ANY M, N, K >= 1.
//   Writes every element of C, reads nothing out of bounds, C does not alias A or B.
//   Pointers are device pointers with natural alignment only (4 bytes float, 2 bytes __half);
//   they may point into the middle of a larger allocation.
//   Asynchronous on `stream`; no synchronization and no per-call allocation.
using GemmLaunchFn    = void (*)(const float*  dA, const float*  dB, float* dC,
                                 int M, int N, int K, hipStream_t stream);
// FP16 inputs, FP32 output (FP32 accumulation unless the entry's tol says otherwise).
using GemmF16LaunchFn = void (*)(const __half* dA, const __half* dB, float* dC,
                                 int M, int N, int K, hipStream_t stream);
struct GemmKernel    { const char* name; GemmLaunchFn    fn; const char* unavailable; double tol; };
struct GemmF16Kernel { const char* name; GemmF16LaunchFn fn; const char* unavailable; double tol; };
// fn == nullptr  <=>  unavailable != nullptr (human-readable reason, e.g.
// "matrix-core instructions are not available on platform cpu").
// tol: threshold for the check_gemm metric (reference.hpp).
// Registries (defined in src/kernels/registry.hip). Names are unique across both tables.
const GemmKernel*    gemm_fp32_kernels(int* count);   // v0 v1 v2 v3 v4 hipblas
const GemmF16Kernel* gemm_fp16_kernels(int* count);   // v0_fp16 v4_fp16 v5 hipblas_fp16
// Entry point other components (the LLM engine) use. Forwards to the fastest
// always-available FP32 ladder stage (v4). Same contract as GemmLaunchFn.
void        gemm_fp32_best(const float* dA, const float* dB, float* dC,
                           int M, int N, int K, hipStream_t stream);
const char* gemm_fp32_best_name();                    // e.g. "v4"
// Launchers. Always declared; defined only where available.
//   FP32 (GemmLaunchFn signature):    launch_v0 launch_v1 launch_v2 launch_v3 launch_v4 launch_hipblas_sgemm
//   FP16 (GemmF16LaunchFn signature): launch_v0_fp16 launch_v4_fp16 launch_v5 launch_hipblas_hgemm
// e.g. void launch_v0(const float* dA, const float* dB, float* dC, int M, int N, int K, hipStream_t stream);
```

### 4.3 `include/safetensors.hpp` (pure C++, no HIP; implemented by [LLM-IO], consumed by [LLM-CORE])
```cpp
#pragma once
#include <cstddef>, <cstdint>, <initializer_list>, <memory>, <string>, <vector>   // one #include each
namespace hipgpt2 {
struct TensorView {
    std::string          name;
    std::string          dtype;   // safetensors dtype string, e.g. "F32"
    std::vector<int64_t> shape;
    const uint8_t*       data;    // valid while the owning SafeTensors (or a copy) is alive
    size_t               nbytes;
    int64_t numel() const;        // product of shape (1 for scalars)
};
// Read-only safetensors file. Copies share one immutable buffer (cheap to copy).
class SafeTensors {
public:
    // Throws std::runtime_error (message includes the path) on I/O or format errors:
    // truncated file, header length > file size, bad JSON, data_offsets out of range,
    // nbytes != numel * sizeof(dtype).
    static SafeTensors open(const std::string& path);
    bool                     has(const std::string& name) const;
    const TensorView&        get(const std::string& name) const;   // throws if missing
    std::vector<std::string> names() const;                        // sorted, excludes "__metadata__"
    // Typed access. Throws std::runtime_error naming the tensor if it is missing, its dtype
    // is not "F32", or its shape != expect_shape. The pointer is aligned to alignof(float),
    // little-endian host assumed.
    const float* f32(const std::string& name, std::initializer_list<int64_t> expect_shape) const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};
}  // namespace hipgpt2
```

### 4.4 `include/gpt2_config.hpp` (pure C++; implemented by [LLM-IO])
```cpp
#pragma once
#include <string>
namespace hipgpt2 {
struct Gpt2Config {
    int n_layer = 0, n_head = 0, n_embd = 0, n_ctx = 0, vocab_size = 0;
    int head_dim() const { return n_embd / n_head; }
};
// Reads HF-style <model_dir>/config.json. Keys: n_layer, n_head, n_embd, vocab_size, and
// n_ctx (falls back to n_positions). Throws std::runtime_error if the file or a key is
// missing, any value <= 0, or n_embd % n_head != 0.
Gpt2Config load_gpt2_config(const std::string& model_dir);
}  // namespace hipgpt2
```

### 4.5 `include/tokenizer.hpp` (pure C++; implemented by [LLM-IO])
```cpp
#pragma once
#include <memory>, <string>, <vector>   // one #include each
namespace hipgpt2 {
// GPT-2 byte-level BPE. Behaviour must equal OpenAI encoder.py (Encoder.encode/decode).
class Gpt2Tokenizer {
public:
    static constexpr int kEndOfText = 50256;
    // Throws std::runtime_error on missing/malformed files.
    static Gpt2Tokenizer load(const std::string& encoder_json_path,
                              const std::string& vocab_bpe_path);
    // Convenience: load(<dir>/encoder.json, <dir>/vocab.bpe).
    static Gpt2Tokenizer load_dir(const std::string& dir);
    // utf8 must be valid UTF-8. No special-token parsing ("<|endoftext|>" is plain text).
    std::vector<int> encode(const std::string& utf8) const;
    // Concatenated raw bytes of the tokens. Throws std::out_of_range on a bad id.
    std::string      decode(const std::vector<int>& ids) const;
    // Raw bytes of one token (may be a partial UTF-8 sequence; callers buffer for streaming).
    std::string      decode_token(int id) const;
    int              vocab_size() const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};
}  // namespace hipgpt2
```

### 4.6 Cross-owner file formats
- **Logits file `LGT1`** (written by `gpt2 --dump-logits`, read/written by python tools, used for `tests/data/tiny/ref_*.bin`): bytes 0-3 ASCII `LGT1`; int32 LE `rows`; int32 LE `vocab`; then `rows*vocab` float32 LE, row-major.
- **Model directory**: `<dir>/model.safetensors`, `<dir>/config.json`; optionally `<dir>/encoder.json`, `<dir>/vocab.bpe`.
- **Tokenizer case file** `tests/data/tokenizer_cases.txt`: UTF-8 text; lines starting with `#` ignored; cases are line pairs: line 1 = lowercase hex of the UTF-8 input bytes (the single character `-` means empty string), line 2 = space-separated decimal ids (empty line = no ids).
- **Token id file** (`prompt_ids.txt`, `ref_greedy_tokens.txt`): space/newline-separated decimal ids.

## 5. GEMM ladder requirements [GEMM]

### 5.1 Teaching convention (applies to all kernel files in `src/kernels/` and `src/llm/kernels/`)
- File header, first lines of the file, exactly these labels:
  ```
  // STAGE: v2 — LDS tiling
  // CHANGES: <what differs from the previous stage, 2-5 lines>
  // WHY FASTER: <mechanism, with a bytes-per-FLOP or reuse argument>
  // PREDICTED BOTTLENECK: <what will limit this stage; what the profiler should show>
  ```
  LLM kernels use `// STAGE: llm — <kernel name>`.
- Inside each `__global__` function and each `launch_*` function, the whole body sits between a line containing only `// BEGIN SOLUTION` and a line containing only `// END SOLUTION` (leading whitespace allowed). No nesting. Tile-size constants and helper declarations stay outside the markers. A file must still compile after everything between the markers is deleted (unused-parameter warnings are acceptable), so nothing declared inside markers may be referenced outside them.
- Tunable constants are named `constexpr int` at file scope with a comment giving the reasoning for the default.

### 5.2 Stages (tile sizes are suggestions; correctness for any M,N,K >= 1 is mandatory for every stage)
| Kernel | File | Must demonstrate | Suggested shape |
|---|---|---|---|
| `v0` | `v0_naive.hip` | One thread per C element, K-loop over global memory, deliberately the textbook mapping. Replace the existing TODO stub. | block 16x16 |
| `v1` | `v1_coalesced.hip` | Same algorithm, thread->element mapping and loop structure chosen so neighbouring threads in a wave read neighbouring addresses of B and write neighbouring addresses of C (`threadIdx.x` -> column). Header explains which accesses are coalesced in v0 vs v1. If v0 already maps x->col, make v0 the uncoalesced mapping (x->row) so the contrast is measurable. | block 32x8 or 64x4 |
| `v2` | `v2_lds_tiled.hip` | Block computes a TxT output tile; loops over K in steps of T loading an A tile and a B tile into static `__shared__` arrays, `__syncthreads`, accumulate. Edge tiles zero-filled in LDS so the inner loop is branch-free. | T = 32 |
| `v3` | `v3_reg_blocked.hip` | Each thread computes an RxR (or RmxRn) sub-tile of outputs held in registers; block tile BM x BN with K-step BK; inner loop is an outer-product update from LDS operands loaded into registers. Header states arithmetic intensity (FLOPs per LDS read) vs v2. | BM=BN=64, BK=16, 4x4 per thread, 256 threads |
| `v4` | `v4_vec_dbuf.hip` | v3 plus (a) global->LDS loads through `float4` when legal, (b) double buffering: the next K-slab is fetched from global memory into registers (or a second LDS buffer) while the current slab is being consumed, so only one sync separates slabs. `float4` loads are legal only when the element address is 16-byte aligned; the kernel/launcher must detect misaligned pointers or row strides not divisible by 4 and use a scalar-load path, and tails are loaded scalar. Never read past the end of A or B. | as v3, BK=8..16 |
| `v0_fp16` | `v0_fp16.hip` | v0 with `__half` inputs, FP32 accumulation, FP32 output. Reference for FP16 paths. | — |
| `v4_fp16` | `v4_fp16.hip` | LDS-tiled, register-blocked FP16-input kernel with FP32 accumulation. Its global->LDS staging code is what `v5` reuses, so that part of v5 is covered by CPU tests. | as v3 |
| `v5` | `v5_matrix_core.hip` + `.inc` | Matrix-core inner product, FP16 operands from LDS, FP32 accumulators, guarded writes to C, zero-padded edge tiles (any M,N,K). Paths: `HIPGPT2_MC_MFMA`: `__builtin_amdgcn_mfma_f32_16x16x16f16`, wave64, block size a multiple of 64, one or more 16x16 output tiles per wave; operand/result lane layouts taken from AMD's Matrix Instruction Calculator and documented in a comment table. `HIPGPT2_MC_WMMA_RDNA3` (experimental): `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, wave32. `HIPGPT2_MC_WMMA_NVIDIA`: `nvcuda::wmma` fragments (`<mma.h>`, 16x16x16, half in, float accumulator), loading from LDS with explicit stride. `HIPGPT2_MC_NONE`: `launch_v5` is not defined and the registry entry is `{ "v5", nullptr, "matrix-core instructions are not available on platform <name>/<arch>", tol }`. | 4 waves/block |
| `hipblas` / `hipblas_fp16` | `hipblas_baseline.hip` | Vendor baseline via hipBLAS `hipblasSgemm` / `hipblasHgemm`. hipBLAS is column-major: compute the row-major product by calling gemm for `C^T = B^T * A^T` (swap operands, no data movement) and state this in a comment. One lazily created handle; `hipblasSetStream` per call. `hgemm` produces FP16: use a lazily grown device scratch buffer and a small conversion kernel to FP32 (one-time allocation is the only allowed exception to rule 3.5.7). Without `HIPGPT2_HAVE_HIPBLAS`: entries are unavailable with reason `"built without hipBLAS"`. | — |

Tolerances (field `tol`): FP32 kernels `1e-4`; FP16-input/FP32-accumulate kernels (`v0_fp16`, `v4_fp16`, `v5`) `1e-3`; `hipblas_fp16` `5e-2` (FP16 output and possibly FP16 accumulation; say so in the registry comment).

`gemm_fp32_best` forwards to `launch_v4`; `gemm_fp32_best_name()` returns `"v4"`.

### 5.3 Harness `src/bench.hip` (extend the existing design; keep `run_one`'s structure)
CLI (all existing flags keep their meaning):

| Flag | Meaning |
|---|---|
| `--kernel <name|all>` | Default `all`. Looks in both tables. Unknown name: exit 2. |
| `--test` | Correctness mode on awkward shapes; no timing, no JSON. |
| `--sizes a,b,MxNxK` | Overrides shapes in both modes. |
| `--iters n`, `--out dir` | As today. |
| `--list` | Prints one line per kernel: `<name> <fp32|fp16> <available|unavailable: reason>`; exit 0. |

Behaviour:

- Default test shapes: `1x1x1, 1x7x5, 17x17x17, 127x127x127, 128x128x128, 333x257x129, 64x1000x3, 3x5x1000, 130x66x255`; on GPU platforms additionally `512x512x512, 1031x517x263, 2049x1023x515`. Default benchmark sweep: GPU `512,1024,2048,4096,8192`; CPU `64,128,256`.
- FP16 path: host generates floats uniform in [-1,1] (same seed 42), converts to binary16 bits with round-to-nearest-even (`half_host.hpp`: `uint16_t f32_to_f16_bits(float)`, `float f16_bits_to_f32(uint16_t)`), uploads the 16-bit buffers as `__half`, and checks the FP32 output with `check_gemm` using the rounded values converted back to float and the entry's `tol`. C is NaN-poisoned before the correctness run, as today.
- Unavailable kernels: with `--kernel all`, print `  SKIP <name>: not available on this platform (<reason>)` and do not affect the exit code and write no JSON. With an explicit `--kernel <name>`, print the same line and exit 77.
- `--test` output per shape: `  <M>x<N>x<K> PASS|FAIL (max_err <e>)`; final line `SUMMARY: <p> passed, <f> failed, <s> skipped`; exit 1 if any FAIL.
- Benchmark mode writes `<out>/gemm_<name>_<device_slug>.json` per available kernel (schema 8.1). `device_slug` = device name with every non-alphanumeric character replaced by `_`. Timing method unchanged (3 warm-ups, adaptive iterations, median).
- Use `HIPGPT2_PLATFORM_NAME` and `HIPGPT2_GIT_SHA` (drop `kPlatform`/`GEMM_GIT_SHA`).

## 6. LLM engine design [LLM-CORE]

### 6.1 Model, tensors, numerics
All math FP32. Weights are uploaded once, unmodified (no transposes), into device buffers. Tensor lookup tries `<name>` then `transformer.<name>`; tensors named `*.attn.bias`, `*.attn.masked_bias`, `lm_head.weight` are ignored. Dimensions come from `Gpt2Config` (d = n_embd, H = n_head, hd = d/H, L = n_layer, V = vocab_size, C = n_ctx); every shape is validated through `SafeTensors::f32(name, {..})`.

| Tensor | Shape | Use |
|---|---|---|
| `wte.weight` | [V, d] | token embedding; also the tied LM head |
| `wpe.weight` | [C, d] | position embedding |
| `h.{i}.ln_1.weight`, `.bias` | [d] | pre-attention LayerNorm |
| `h.{i}.attn.c_attn.weight`, `.bias` | [d, 3d], [3d] | QKV projection (HF Conv1D: stored [in, out]) |
| `h.{i}.attn.c_proj.weight`, `.bias` | [d, d], [d] | attention output projection |
| `h.{i}.ln_2.weight`, `.bias` | [d] | pre-MLP LayerNorm |
| `h.{i}.mlp.c_fc.weight`, `.bias` | [d, 4d], [4d] | MLP up |
| `h.{i}.mlp.c_proj.weight`, `.bias` | [4d, d], [d] | MLP down |
| `ln_f.weight`, `.bias` | [d] | final LayerNorm |

Forward pass for rows at absolute positions p (T rows; prefill: p = 0..P-1, decode: one row at p = current length):

```
x[t]   = wte[token[t]] + wpe[p[t]]
for i in 0..L-1:
  a    = LayerNorm(x; ln_1)                              # eps 1e-5, biased variance over d
  qkv  = a @ c_attn.weight + c_attn.bias                 # [T, 3d]; q = cols [0,d), k = [d,2d), v = [2d,3d)
  K_i[p[t]] = k[t];  V_i[p[t]] = v[t]                    # KV-cache rows
  for head h, query t (position p):  cols J = [h*hd, (h+1)*hd)
     s[u]  = dot(q[t][J], K_i[u][J]) / sqrt(hd)   for u = 0..p      (causal: u <= p)
     w     = softmax(s)
     y[t][J] = sum_u w[u] * V_i[u][J]
  x   += y @ attn.c_proj.weight + attn.c_proj.bias
  m    = LayerNorm(x; ln_2)
  f    = gelu(m @ mlp.c_fc.weight + mlp.c_fc.bias)       # gelu(z) = 0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z^3)))
  x   += f @ mlp.c_proj.weight + mlp.c_proj.bias
xf     = LayerNorm(x; ln_f)
logits[t][v] = dot(xf[t], wte[v])                        # [V]
```

### 6.2 Data layout
- Activations: row-major `[T, width]` device buffers preallocated at capacity `C` rows at engine construction (no allocation during generation): `x [C,d]`, `a [C,d]`, `qkv [C,3d]`, `y [C,d]`, `proj [C,d]`, `mlp [C,4d]`, `logits [V]` (plus `[C,V]` allocated lazily only for `--dump-mode all`).
- KV-cache: per layer two device buffers `K_i`, `V_i` of shape `[C, d]`, row = absolute position, columns in the same head-major order as the qkv slices (head h owns columns `[h*hd, (h+1)*hd)`). `len` = number of valid rows, identical for all layers.
- Attention works only against the cache: signature concept `attention(q_rows, K_i, V_i, pos0, T, H, hd) -> y_rows`, where query row r is at absolute position `pos0 + r` and attends to cache rows `0..pos0+r`. The same definition serves prefill (`pos0 = 0, T = P`) and decode (`pos0 = len, T = 1`); the cache rows for the current rows are written before attention runs.

### 6.3 Kernels (each in `src/llm/kernels/`, declared in `llm_kernels.hpp`, convention 5.1, rules 3.5)
| Kernel | Requirement |
|---|---|
| `embedding` | `x[t] = wte[tok[t]] + wpe[pos0+t]`; token ids uploaded as int32. |
| `layernorm` | One row per block; mean and variance via shared-memory tree reduction; eps 1e-5 inside the sqrt. |
| `gemv_xw` | `y[N] = x[K] @ W[K,N] + b[N]`, optional fused GELU flag. One thread per output column so loads of `W[k][n..]` are coalesced. Decode path for all Conv1D layers. |
| `gemv_rowdot` | `y[V] = W[V,d] · x[d]` (LM head on tied `wte`). Reduction per row must follow rule 3.5.3. |
| `bias_add`, `bias_gelu` | Row-broadcast bias (optionally fused GELU) after a prefill GEMM. |
| fallback GEMM | Simple LDS-tiled FP32 GEMM with the 4.2 contract, used only when `HIPGPT2_LLM_USE_LADDER_GEMM` is not defined. |
| `residual_add`, `gelu` | Elementwise. |
| `kv_store` | Copies the k and v slices of `qkv` rows into `K_i`, `V_i` rows `pos0..pos0+T-1`. |
| `attention_naive` | Three launches: scores `[H, T, C]` into a global scratch buffer (allocated at construction), causal softmax (masked entries excluded, max-subtracted), weighted sum of V. Used for prefill always and for decode with `--attn naive`. |
| `attention_fused` | Decode only (T = 1), one launch per layer, one block per head. No score buffer in global memory. Online softmax: partial results are `(m, l)` = (running max, running sum of `exp(s - m)`) per thread, merged with `m' = max(m1, m2)`, `l' = l1*exp(m1-m') + l2*exp(m2-m')`; scores for the head may be kept in a static `__shared__` array of at most 1024 floats (engine falls back to naive if `C > 1024`). Must equal `attention_naive` within 1e-5 max abs on `test_llm_kernels` inputs. |

Linear layers: prefill uses `gemm_fp32_best(X[T,K], W[K,N], Y[T,N])` then `bias_add`/`bias_gelu` (when `HIPGPT2_LLM_USE_LADDER_GEMM`; otherwise the fallback GEMM); decode uses `gemv_xw`. The LM head always uses `gemv_rowdot` on the needed row(s) only (last prompt row in normal mode).

### 6.4 Engine flow and timing
1. Load: config, safetensors, upload weights, allocate buffers (`load_ms`).
2. Tokenize (or parse `--tokens`). Require `1 <= P <= C`; clamp `max_new_tokens` to `C - P + 1` with a warning. Empty prompt: use token 50256 if `V > 50256`, else error.
3. Warm-up: `--warmup n` untimed runs of prefill + one decode step, then the cache is reset.
4. Prefill: all P rows in one pass, logits for row P-1, copy to host, sample token 1. `ttft_ms` = wall time from prefill start to token 1 available on host (stream synchronized). `prefill_ms` = device part of that only.
5. Decode: for each further token, one row through the decode path, logits to host, sample. `decode_tok_per_s = (gen_tokens - 1) / decode_seconds`, wall time over all decode steps including the host copy and sampling; 0 if `gen_tokens < 2`.
6. Sampling on host (`sampler.cpp`, pure C++): greedy = lowest index of the maximum; otherwise logits/temperature, keep top-k (k = 0 means all), softmax in double, draw with `std::mt19937_64(seed)`.
7. Streaming output: buffer `decode_token` bytes and print only complete UTF-8 sequences.

### 6.5 CLI `gpt2` (flags are an interface for [INFRA] and [LLM-IO])
| Flag | Default | Meaning |
|---|---|---|
| `--model DIR` | required | Model directory (4.6). |
| `--vocab-dir DIR` | model dir, then `assets/gpt2` | Location of `encoder.json`, `vocab.bpe`. Not needed with `--tokens`. |
| `--prompt TEXT` / `--prompt-file F` / `--tokens 1,2,3` | one required | Prompt source. With `--tokens`, output is printed as ids if no vocab is found. |
| `--max-new-tokens N` | 64 | Number of generated tokens G. |
| `--temperature F` | 0 | 0 = greedy. |
| `--top-k K`, `--seed S` | 40, 1234 | Sampling parameters (ignored when greedy). |
| `--stop-at-eot` | off | Stop when token 50256 is sampled. |
| `--attn naive|fused` | fused | Decode attention kernel. |
| `--no-kv-cache` | off | Debug baseline: every step re-runs prefill on the whole sequence. |
| `--warmup N` | 0 | Untimed warm-up runs. |
| `--dump-logits FILE` | — | Write an `LGT1` file. |
| `--dump-mode last|all` | last | `last`: G rows, row g = logits used to sample generated token g (row 0 comes from prefill). `all`: P rows, logits at every prompt position from prefill; generation is skipped. |
| `--bench-out DIR` | — | Write `<DIR>/llm_<device_slug>.json` (schema 8.2). |
| `--quiet` | off | Suppress streamed text; print only the stats block. |

Stats block on stdout (exact keys, one per line): `prompt_tokens: <n>`, `gen_tokens: <n>`, `load_ms: <f>`, `ttft_ms: <f>`, `decode_tok_per_s: <f>`. Exit codes: 0 ok, 2 usage error, 1 runtime error (message on stderr; exceptions from the IO library are caught in `main`).

## 7. Test plan and acceptance criteria

Common configure line (substitute the agent's build dir and component switches):
`cmake -S . -B <bdir> -DGEMM_BACKEND=cpu -DHIP_CPU_DIR=/home/claude/deps/HIP-CPU [-DHIPGPT2_BUILD_*=OFF ...] && cmake --build <bdir> -j`

**[GEMM]** (`build-gemm`, `-DHIPGPT2_BUILD_LLM=OFF -DHIPGPT2_BUILD_IO=OFF`)
1. `ctest --test-dir build-gemm -R 'reference_checker|gemm_correctness' --output-on-failure` passes.
2. `./build-gemm/gemm_bench --kernel all --test` ends with `SUMMARY: ... 0 failed, 3 skipped` (v5, hipblas, hipblas_fp16) and exit 0.
3. `./build-gemm/gemm_bench --kernel v5 --test; echo $?` prints the SKIP line and `77`.
4. `./build-gemm/gemm_bench --kernel v4 --test --sizes 512,1x1x1,5x3x2,257x255x254` passes (alignment/tail paths).
5. `./build-gemm/gemm_bench --kernel all --sizes 64,128 --iters 3 --out /tmp/gemm_r` writes 7 JSON files, each parseable by `python3 -m json.tool` and matching 8.1 with `verified: true`.
6. `grep -c 'BEGIN SOLUTION' src/kernels/*.hip` >= 2 per kernel file (kernel + launcher); `grep -rn '<<<' src/` empty.
7. The three `.inc` paths are syntax-reviewed against vendor documentation; the report lists what was not compiled.

**[LLM-IO]** (`build-io`, `-DHIPGPT2_BUILD_GEMM=OFF -DHIPGPT2_BUILD_LLM=OFF`)
1. `python3 tools/gen_tokenizer_cases.py` regenerates `tests/data/tokenizer_cases.txt` byte-identically (deterministic) using `tools/third_party/encoder.py` and `assets/gpt2`. Cases must include, at minimum: empty string; `Hello world`; `Hello, world!`; leading and trailing space; runs of 2 and 3 spaces; tabs, `\n`, `\r\n`, three consecutive newlines; whitespace-only strings (`"   "`, `"\n"`, `" \n "`); contractions `I'm, you're, he's, we've, they'll, I'd, don't` and their uppercase forms `DON'T I'M`; digits `1234567890 3.14159 1,000,000 2026-11-16`; punctuation runs `!!!??? ... --- (a) [b] {c}`; the literal text `<|endoftext|>`; apostrophe edge cases `'s't're` and `''''`; identifiers `foo_bar __init__ camelCase`; accented Latin `naïve café résumé`; CJK `日本語のテキスト 中文 한국어`; emoji including a skin-tone modifier, a ZWJ family sequence and a flag; non-ASCII whitespace (U+00A0, U+2003, U+3000); non-ASCII numerics (`x²+y³=½ ①②③ ٣`); one paragraph of at least 300 characters; plus 200 seeded random strings mixing all of the above alphabets.
2. `ctest --test-dir build-io -R 'tokenizer|safetensors' --output-on-failure` passes. `test_tokenizer` checks, for every case, `encode(s) == ids` and `decode(ids) == s`, prints the first mismatching case in hex, exits non-zero on any failure. `test_safetensors` opens `tests/data/tiny/model.safetensors`, checks names/shapes against `config.json`, checks `f32` throws on wrong shape and on a missing name, checks `open` throws on a truncated copy and on a nonexistent path, and compares one tensor's first and last values with constants stored in `tests/data/tiny/manifest.txt`.
3. `python3 tools/make_tiny_model.py --out /tmp/tiny && cmp /tmp/tiny/model.safetensors tests/data/tiny/model.safetensors` (deterministic generator).
4. `python3 tools/compare_logits.py tests/data/tiny/ref_greedy.bin tests/data/tiny/ref_greedy.bin --tol 1e-3` exits 0; with a perturbed copy exits 1.
5. `python3 -m py_compile tools/*.py` passes (torch-dependent scripts import torch lazily inside `main`).

**[LLM-CORE]** (`build-llm`; kernel stage: `-DHIPGPT2_BUILD_GEMM=OFF -DHIPGPT2_BUILD_IO=OFF`; engine stage: defaults)
1. `ctest --test-dir build-llm -R llm_kernels --output-on-failure`: every kernel in 6.3 compared with a plain C++ reference written in the test, on awkward sizes (e.g. d = 48, H = 3, T in {1, 2, 13}, V = 301, non-multiples of any block size); max abs error <= 1e-5 (<= 1e-4 for linear layers with K >= 256); naive vs fused attention <= 1e-5.
2. `ctest --test-dir build-llm -R 'llm_tiny_e2e|llm_cli_smoke' --output-on-failure`. `test_llm_e2e <tiny dir>` asserts: (a) prefill logits at all P positions vs `ref_prefill_all.bin`, max abs error <= 1e-3; (b) greedy generation of G tokens: tokens equal `ref_greedy_tokens.txt` and logits rows vs `ref_greedy.bin` <= 1e-3; (c) the same with `--attn naive` semantics; (d) feeding the prompt one token at a time through the decode path gives final-position logits within 1e-3 of prefill; (e) `--no-kv-cache` semantics give the same tokens. It prints each measured max error.
3. `./build-llm/gpt2 --model tests/data/tiny --tokens 5,17,42 --max-new-tokens 8 --dump-logits /tmp/l.bin --bench-out /tmp/llm_r` exits 0; `/tmp/l.bin` is a valid `LGT1` file with rows = 8, vocab = 300; the JSON matches 8.2.
4. Final acceptance is run with `-DHIPGPT2_LLM_GEMM=ladder` once `src/kernels/registry.hip` exists; before that, `internal` is acceptable.
5. `grep -rn '<<<' src/llm tests` empty; SOLUTION markers present in every file of `src/llm/kernels/`.

**[INFRA]**
1. `python3 scripts/check_regression.py --self-test` exits 0 (covers: equal results pass; 10% GFLOPS drop fails; 3% drop passes; `verified: false` fails; LLM `decode_tok_per_s` -10% fails; LLM `ttft_ms` +10% fails; missing baseline handling; threshold override).
2. `python3 scripts/plot_results.py --results <dir with sample JSON> --out /tmp/plots` produces PNGs without a display (`matplotlib` Agg).
3. `python3 scripts/make_exercises.py --src src --out /tmp/ex` on a fixture file blanks everything between markers, inserts `// TODO: your code here`, leaves everything else byte-identical, and fails on unbalanced/nested markers (covered by `--self-test`).
4. `python3 -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]" .github/workflows/*.yml` succeeds (fall back to a manual review note if PyYAML is absent); `bash -n scripts/*.sh` passes.

**Integration ([ORCH], after all agents finish):** D1-D11 with the default configure line.

## 8. JSON schemas

All files: UTF-8, one JSON object, numbers finite (never NaN/Inf; write 0 and `verified: false` instead). `timestamp` is UTC `YYYY-MM-DDTHH:MM:SSZ`.

### 8.1 GEMM result — `results/gemm_<kernel>_<device_slug>.json` (producer [GEMM], consumer [INFRA])
```json
{"schema_version": 1, "bench": "gemm", "kernel": "v3", "dtype": "fp32", "platform": "nvidia",
 "device": "NVIDIA GeForce RTX 4060", "git_sha": "8c3e8f0", "timestamp": "2026-11-16T20:00:00Z",
 "results": [{"M": 1024, "N": 1024, "K": 1024, "ms_median": 1.92, "ms_min": 1.88, "gflops": 1118.5,
              "iters": 100, "max_err": 3.1e-7, "verified": true, "full_check": true}]}
```

`dtype`: `"fp32"` or `"fp16"` (input type). `platform`: `cpu|amd|nvidia`. Times in milliseconds per launch. `gflops = 2*M*N*K / (ms_median * 1e6)` (1e9 FLOP/s). `max_err`: `check_gemm` metric (dimensionless). `full_check`: every element checked vs sampled.

### 8.2 LLM result — `results/llm_<device_slug>.json` (producers [LLM-CORE] `gpt2`, [LLM-IO] `torch_generation_baseline.py`; consumer [INFRA])
```json
{"schema_version": 1, "bench": "llm", "engine": "hip-gpt2", "model": "gpt2", "platform": "nvidia",
 "device": "NVIDIA GeForce RTX 4060", "git_sha": "8c3e8f0", "timestamp": "2026-11-16T20:00:00Z",
 "attn": "fused", "gemm_kernel": "v4", "sampling": "greedy", "warmup_runs": 1,
 "prompt_tokens": 32, "gen_tokens": 128, "load_ms": 410.0, "prefill_ms": 9.8, "ttft_ms": 11.2,
 "decode_tok_per_s": 412.5}
```

`engine`: `"hip-gpt2"` or `"pytorch"` (the PyTorch baseline writes `llm_pytorch_<device_slug>.json` with `attn`/`gemm_kernel` = `"n/a"`). `model`: basename of the model directory. `gemm_kernel`: `gemm_fp32_best_name()` or `"internal"`. Definitions of `ttft_ms`, `prefill_ms`, `decode_tok_per_s`: section 6.4.

### 8.3 Baselines and history ([INFRA] internal, fixed here so docs can reference them)
- Baseline: `perf/baselines/<device_slug>/` holds copies of 8.1/8.2 files. Created/updated only by `scripts/check_regression.py --update-baseline` and committed by the owner.
- History (branch `gh-pages`, `data/history.json`): array of `{"git_sha", "timestamp", "device", "gemm": [{"kernel", "dtype", "M", "N", "K", "gflops"}], "llm": {"ttft_ms", "decode_tok_per_s"} | null}`; appended by `scripts/update_history.py`, newest last, de-duplicated by `(git_sha, device)`.
- Regression rule: for every `(kernel, dtype, M, N, K)` present in both baseline and current: fail if `gflops_cur < gflops_base * (1 - T/100)`. LLM: fail if `decode_tok_per_s` drops more than T% or `ttft_ms` rises more than T%. Any `verified: false` fails. Entries present in the baseline but missing now: warning (fail with `--strict`). `T` from `--threshold`, else env `PERF_THRESHOLD_PCT`, else 7. Exit 0 pass, 1 regression, 2 usage error; missing baseline directory: exit 0 with a notice unless `--require-baseline`.

## 9. Agent task cards

Every card implies: read SPEC sections 2, 3, 5.1 and the sections named in the card; use your own build directory; touch only your files; finish by running your acceptance commands (section 7) and reporting their output, deviations, and anything not verifiable on CPU.

### 9.1 [GEMM] — Opus
- **Read:** sections 3, 4.1, 4.2, 5, 7 [GEMM], 8.1, 10. Existing `src/bench.hip`, `src/kernels/v0_naive.hip`, `include/reference.hpp`.
- **Write:** `src/bench.hip`, everything in `src/kernels/`.
- **Deliver:** kernels v0..v4, `v0_fp16`, `v4_fp16`, `v5` (three `.inc` paths), hipBLAS baselines, `registry.hip` (tables, `gemm_fp32_best`), `half_host.hpp`, extended harness (5.3), JSON 8.1.
- **Order:** registry + v0 + harness changes first (so `gemm_fp32_best` links early, temporarily forwarding to the best finished stage), then v1..v4, FP16, v5, hipBLAS.
- **Accept:** section 7 [GEMM] 1-7.
- **Do not touch:** `CMakeLists.txt`, `include/`, `src/reference.cpp`, `tests/`, `src/io/`, `src/llm/`, `tools/`, `scripts/`, `.github/`, docs.

### 9.2 [LLM-CORE] — Opus
- **Read:** sections 3, 4 (all), 5.1, 6, 7 [LLM-CORE], 8.2, 10.
- **Write:** everything in `src/llm/`, `tests/test_llm_kernels.hip`, `tests/test_llm_e2e.hip`.
- **Deliver:** llm kernels (6.3), engine with KV-cache, prefill/decode paths, sampler, CLI `gpt2` (6.5), LLM JSON 8.2, `LGT1` writer and reader (test side).
- **Order:** (1) kernels + `test_llm_kernels` with `-DHIPGPT2_BUILD_GEMM=OFF -DHIPGPT2_BUILD_IO=OFF` (no external dependencies). (2) Engine + CLI coded against headers 4.3-4.5. (3) When `src/io/*.cpp` and `tests/data/tiny/ref_greedy.bin` exist, build and pass e2e; if they are not there yet, wait and re-check, do not create stubs in other owners' paths. (4) Switch to `-DHIPGPT2_LLM_GEMM=ladder` when `src/kernels/registry.hip` exists.
- **Accept:** section 7 [LLM-CORE] 1-5.
- **Do not touch:** `include/`, `src/kernels/`, `src/bench.hip`, `src/io/`, `tests/data/`, `tools/`, `assets/`, `CMakeLists.txt`, `scripts/`, `.github/`, docs.

### 9.3 [LLM-IO] — Sonnet
- **Read:** sections 3.2, 3.4, 4.3-4.6, 6.1 (tensor table and forward pass), 6.5 (`--dump-logits`), 7 [LLM-IO], 8.2. Inputs: `/home/claude/deps/gpt2vocab/{encoder.json,vocab.bpe,encoder.py}`.
- **Write:** `src/io/`, `tests/test_tokenizer.cpp`, `tests/test_safetensors.cpp`, `tests/data/`, `assets/gpt2/`, `tools/`.
- **Deliver:**
  - `safetensors.cpp`: format = 8-byte LE header length N, N bytes of JSON (`{"name": {"dtype", "shape", "data_offsets": [begin, end]}, "__metadata__": {...}}`), then the data buffer; offsets are relative to byte `8 + N`. Read the data section into a 64-byte-aligned buffer. `json_mini.hpp`: minimal JSON reader shared by the three `.cpp` files (objects, arrays, strings with `\uXXXX` escapes incl. surrogate pairs, integers).
  - `gpt2_config.cpp` (4.4).
  - `tokenizer.cpp`: byte->unicode table and BPE merge loop as in `encoder.py`; pre-tokenizer implementing `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` by hand (ordered alternation, case-sensitive contractions, negative lookahead: a whitespace run followed by a non-space gives up its last character). No `std::regex`. Character classes come from `unicode_tables.inc` (sorted code-point ranges for `\p{L}`, `\p{N}`, `\s`), generated by `tools/gen_unicode_tables.py` by brute force over all code points with the installed python `regex` module so the classes match `encoder.py` exactly. Include a per-word BPE cache.
  - `assets/gpt2/`: copies of `encoder.json`, `vocab.bpe`, plus the MIT license text of openai/gpt-2. `tools/third_party/encoder.py`: unmodified copy.
  - `tools/gpt2_numpy.py`: float64 numpy implementation of 6.1 (`load(model_dir)`, `forward(ids) -> logits [T, V]`, `greedy(ids, n)`), plus `LGT1` read/write helpers reused by the other tools.
  - `tools/make_tiny_model.py`: seed-fixed tiny model: L=2, d=64, H=4, V=300, C=64; HF tensor names without prefix; init: `wte`, `wpe` ~ N(0, 0.5); linear weights ~ N(0, 0.15); biases ~ N(0, 0.1); LayerNorm weight = 1 + 0.1*N(0,1), bias = 0.1*N(0,1). Writes into `tests/data/tiny/`: `model.safetensors`, `config.json`, `prompt_ids.txt` (P = 11 ids), `ref_prefill_all.bin` (P rows), `ref_greedy.bin` + `ref_greedy_tokens.txt` (G = 16), `manifest.txt` (`<tensor name> <first value> <last value>` lines, `%.9g`). Must assert that every greedy step has top-1 minus top-2 logit margin >= 1e-2 (change the seed until it holds) and that output is byte-deterministic.
  - `tools/export_reference_logits.py` (owner-run, needs torch + transformers): downloads `gpt2`, writes a model directory (4.6) and `LGT1` reference logits (`--dump-mode all` semantics and greedy `last` semantics) for a given prompt. `tools/compare_logits.py A B --tol T`: prints max abs error and argmax agreement, exit 1 above tolerance. `tools/torch_generation_baseline.py`: greedy generation timing with PyTorch on the available device, writes 8.2 JSON with `engine: "pytorch"`, same TTFT/decode definitions as 6.4. `tools/requirements.txt`.
- **Accept:** section 7 [LLM-IO] 1-5.
- **Do not touch:** `include/`, `src/kernels/`, `src/llm/`, `src/bench.hip`, `tests/test_llm_*.hip`, `CMakeLists.txt`, `scripts/`, `.github/`, docs.

### 9.4 [INFRA] — Sonnet
- **Read:** sections 3, 5.1 (markers), 5.3 and 6.5 (CLI flags), 7 [INFRA], 8, 10.
- **Write:** `.github/workflows/`, `scripts/`, `site/`, `perf/`.
- **Deliver:**
  - `ci.yml` (push + pull_request): job `cpu-tests` on `ubuntu-24.04`: install `cmake g++ libtbb-dev python3-numpy`, clone HIP-CPU at a pinned commit, configure with `-DGEMM_BACKEND=cpu -DHIP_CPU_DIR=...`, build, `ctest --output-on-failure`. Job `rocm-compile` (`continue-on-error: true`): container = official ROCm dev image with a pinned tag, configure `-DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=/opt/rocm`, build only.
  - `bench.yml` (`workflow_dispatch` + push to `main`, `runs-on: [self-hosted, gpu]`, configure flags from repository variable `HIP_CMAKE_FLAGS`, model dir from `GPT2_MODEL_DIR`): build, `ctest`, `scripts/run_benchmarks.sh` (runs `gemm_bench --kernel all --sizes 1024,2048,4096 --out results` and, if `GPT2_MODEL_DIR` is set, `gpt2 --model $GPT2_MODEL_DIR --prompt-file scripts/bench_prompt.txt --max-new-tokens 128 --warmup 1 --quiet --bench-out results`), `check_regression.py --baseline perf/baselines --results results` (threshold from variable `PERF_THRESHOLD_PCT`, default 7), `update_history.py`, publish `site/` + `data/history.json` to `gh-pages`, upload `results/` as an artifact. History is published even when the regression check fails; the job still ends red.
  - `scripts/check_regression.py` (rule 8.3, `--self-test`, `--update-baseline`, prints a table of base/current/delta%). `scripts/update_history.py`. `scripts/plot_results.py` (GFLOPS vs size per kernel; ladder bar chart at one size; PNG). `scripts/make_exercises.py` (5.1 blanking). `scripts/profile_amd.sh` (rocprof-compute profile + roofline for one kernel/size), `scripts/profile_nvidia.sh` (Nsight Compute `ncu` equivalent); both take `<kernel> <size>` and call `gemm_bench --kernel K --sizes S --iters 20`.
  - `site/index.html` + `site/chart.js`: static page, fetches `data/history.json`, one line chart per device of GFLOPS vs commit for each kernel at the largest common size, plus TTFT and tokens/s charts; Chart.js from a CDN; no build step.
  - Python: standard library + `matplotlib` only.
- **Accept:** section 7 [INFRA] 1-4.
- **Do not touch:** anything under `src/`, `include/`, `tests/`, `tools/`, `assets/`, `CMakeLists.txt`, docs.

### 9.5 [ORCH] — before agents start
- Rewrite `CMakeLists.txt` per section 3 (keep the existing `reference` target, git-sha stamping and OpenMP handling; replace the per-kernel `foreach` tests with 3.4). Write the six headers of section 4 (expand the shorthand, fill in the two macro bodies). Remove the `launch_v0`-only declaration style from `gemm.hpp`. Add `build*/` to `.gitignore`. Verify an empty tree (only existing files) still configures with all components skipped.

- **[DOCS], later:** `README.md`, `docs/stages/v0..v5.md` (from `docs/stages/TEMPLATE.md`), blog draft, demo-video script, study guide. Not started until integration passes.

## 10. Known risks and hardware-only verification

| # | Risk | Mitigation / who checks |
|---|---|---|
| R1 | v5 MFMA/WMMA paths never compile or run in this environment; lane layouts or builtin signatures may be wrong. | Structure v5 so only the inner product differs from `v4_fp16`; `rocm-compile` CI job catches syntax for gfx942; owner runs H1/H2. |
| R2 | hipBLAS baseline never compiled here (`hipblasHalf` type, column-major trick, stream handling). | Owner runs H1/H2 with hipBLAS installed; `gemm_bench --kernel hipblas --test`. |
| R3 | HIP-CPU hides GPU-only failures: misaligned `float4` loads, races between `__syncthreads`, out-of-bounds reads that happen to be mapped, wave-size assumptions. | Rules 3.5; v4 alignment test (7 [GEMM] 4); owner re-runs all `--test` on GPU, where shapes up to 2049 are added. |
| R4 | Real `model.safetensors` key names/prefix and extra buffers cannot be checked offline. | Prefix fallback and ignore list (6.1); owner runs H3. |
| R5 | Tokenizer equality depends on Unicode tables matching the python `regex` version. | Tables generated from the same module that `encoder.py` uses; fixture regenerated by a script. |
| R6 | Tiny random model may not exercise numerics like real weights (large activations, 12 layers). | H3 compares real GPT-2 logits; expected max abs error <= 1e-2 on logits, identical greedy tokens for 32 steps. |
| R7 | Agents building while other components are half-written. | Component switches + empty-glob skipping (3.1, 3.2); separate build dirs. |
| R8 | No performance claim can be validated here (tile sizes, double buffering benefit, fused attention speedup, 7% threshold noise level). | Owner benchmarks on GPU; tile constants are named and documented (5.1); threshold is configurable. |

Owner verification commands (real hardware):

- **H1 NVIDIA (RTX 4060):**
  `cmake -S . -B build-nv -DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=89 && cmake --build build-nv -j && ctest --test-dir build-nv --output-on-failure && ./build-nv/gemm_bench --kernel v5 --test && ./build-nv/gemm_bench --kernel all --out results`
- **H2 AMD MI300X:**
  `cmake -S . -B build-amd -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=/opt/rocm && cmake --build build-amd -j && ctest --test-dir build-amd --output-on-failure && ./build-amd/gemm_bench --kernel v5 --test && ./build-amd/gemm_bench --kernel all --out results`
  (RDNA3: `-DCMAKE_HIP_ARCHITECTURES=gfx1100 -DHIPGPT2_ENABLE_RDNA3_WMMA=ON`.)
- **H3 Real GPT-2 weights (any GPU build `<b>`):**
  `pip install -r tools/requirements.txt && python3 tools/export_reference_logits.py --out models/gpt2 --prompt "The capital of France is" --ref /tmp/ref_all.bin && ./<b>/gpt2 --model models/gpt2 --prompt "The capital of France is" --dump-logits /tmp/cpp_all.bin --dump-mode all && python3 tools/compare_logits.py /tmp/ref_all.bin /tmp/cpp_all.bin --tol 1e-2`
- **H4 End-to-end numbers:**
  `./<b>/gpt2 --model models/gpt2 --prompt-file scripts/bench_prompt.txt --max-new-tokens 128 --warmup 1 --bench-out results && python3 tools/torch_generation_baseline.py --model models/gpt2 --prompt-file scripts/bench_prompt.txt --max-new-tokens 128 --out results`
- **H5 Profiles:** `scripts/profile_amd.sh v4 4096` or `scripts/profile_nvidia.sh v4 4096`.
