# llm — the GPT-2 engine kernels

Sources: `src/llm/kernels/*.hip` (declared in `llm_kernels.hpp`), `src/llm/engine.hip`, `src/llm/main.hip` (CLI `gpt2`). The model is GPT-2 small (12 layers, d = 768, 12 heads, head size 64, vocabulary 50257, context 1024), all math in FP32, no PyTorch at inference time.

## Idea

Factory analogy: **threads are workers, global memory is the warehouse, LDS and registers are the parts bins beside the line.** Generating text has two very different phases.

- **Prefill** (reading the prompt): all P prompt tokens go through every layer at once. Each weight fetched from the warehouse is used P times. This is a real matrix multiply, and the GEMM ladder (v4) is the right tool: a big team assembling with shared bins.
- **Decode** (producing one new token at a time): one token goes through, so each weight fetched from the warehouse is used **once**. There is nothing to reuse, so tiling cannot help. The job is simply "stream all the weights past the line as fast as the trucks can run", so the engine uses simple matrix-vector kernels that read weights in coalesced order.

The KV-cache is the engine's memory of earlier tokens: each layer stores the keys and values of every token so far, so a decode step reads them back instead of recomputing the whole sequence.

## What the code does

Per layer (see `Gpt2Engine::forward_rows`): `layernorm`, QKV projection, `kv_store`, attention, output projection, `residual_add`, `layernorm`, MLP up with GELU, MLP down, `residual_add`. At the end: final `layernorm`, then the LM head.

| Kernel | What it does | Notes |
|---|---|---|
| `embedding_kernel` | `x[t] = wte[token[t]] + wpe[pos0 + t]`, one block per token row | `kEmbedThreads = 256`; pure gather, coalesced |
| `layernorm_kernel` | One block per row; mean then biased variance by shared-memory tree reduction (two-pass); eps 1e-5 inside the sqrt | `kLnThreads = 256`; no wave-width assumption |
| `gemv_xw_kernel` | Decode projection `y[N] = x[K] W[K,N] + b`, optional GELU fused; one thread per output column; `x` staged in LDS in chunks | `kGemvThreads = 64`, `kGemvXChunk = 256`; consecutive threads read consecutive `W[k][n]`: coalesced |
| `gemv_rowdot_kernel` | LM head on the tied embedding: `logits[v] = wte[v,:] . x`; one block per (vocab row, token row), tree reduction | `kRowdotThreads = 128`; reads 50257 x 768 floats (about 154 MB) per token |
| `bias_act_kernel` (`launch_bias_add` / `launch_bias_gelu`) | Row-broadcast bias (and optional GELU) after a prefill GEMM | `kEpilogueThreads = 256` |
| `llm_gemm_fallback_kernel` | 16 x 16 LDS-tiled GEMM, used only when the engine is built with `-DHIPGPT2_LLM_GEMM=internal` | `kFallbackTile = 16` |
| `residual_add`, `gelu`, `kv_store` | Elementwise; `kv_store` copies the k and v slices of the QKV rows into cache rows `pos0 + t` | `kEltThreads = 256`; head-major column order |
| `attention_naive` (3 launches) | scores `[H][T][C]` in a global scratch buffer, causal softmax (masked entries never touched, max-subtracted), weighted sum of V | Always used for prefill; for decode with `--attn naive` |
| `attention_fused_kernel` | Decode only (T = 1): one launch per layer, **one block per head** (`kFusedThreads = 128`); scores live in a static LDS array (at most `kFusedAttnMaxKeys = 1024`) | Online-softmax statistics `(m, l)` merged by a tree; the engine falls back to naive if `n_ctx > 1024` |

Engine design points to know:

- Linear layers: prefill calls `gemm_fp32_best` (v4 when built with the ladder) then a bias kernel; decode calls `gemv_xw`. The LM head always uses `gemv_rowdot`, and in normal mode only on the last prompt row.
- All weights are uploaded once, unmodified (GPT-2 stores linear weights as `[in, out]`, which is what `gemv_xw` and the GEMM expect). All activation buffers and the KV-cache are allocated at construction at full context size; nothing is allocated during generation.
- `ttft_ms` is measured from prefill start to the first sampled token on the host. `decode_tok_per_s = (gen_tokens - 1) / decode wall time`, including the host copy of the logits and sampling.
- A detail worth being honest about in an interview: in `attention_fused_kernel` the scores are already held in LDS, so the online-softmax rescaling is used mainly to merge per-thread partial results; its main win here is "one launch, no global score buffer", not an asymptotic change.

## Theory-based prediction

- **Decode is memory-bound.** One decode step streams about 124M FP32 parameters, around 500 MB (the 154 MB tied LM head included), at 2 FLOPs per 4-byte weight = 0.5 FLOP/byte. On a device with bandwidth B this gives a ceiling of about `B / 500 MB` tokens per second before any overhead (for MI300X, a datasheet bandwidth of `{{device_hbm_bandwidth_gbs}}` GB/s gives `{{decode_bandwidth_ceiling_tok_per_s}}`). Your measured decode speed divided by that ceiling is the number to report and explain.
- **The engine will be well below that ceiling**, for parallelism reasons: `gemv_xw` launches only `N / 64` blocks (12 to 48 per layer), far fewer than the compute units on a big GPU; the fused attention has only 12 blocks (one per head); layernorm runs a single block per step. Many tiny kernel launches per step (about 12 per layer) add launch latency. Expect low occupancy and low achieved bandwidth in the profiler. Next steps (not implemented): split-K across blocks for the GEMVs, flash-decoding for attention, fusing layernorm and residual adds.
- **Prefill is compute-bound for long prompts** and uses the ladder; TTFT scales with prompt length and with how good `gemm_fp32_best` is. For very short prompts the launch overhead dominates.
- **Fused vs naive attention at decode**: fused removes two kernel launches per layer and the score-buffer traffic; the gain is small at short sequence lengths and grows with length. Your guess: `{{owner: predicted fused/naive tokens/s ratio at 128 tokens}}`.

## Results

Prompt `scripts/bench_prompt.txt`, 128 new tokens, greedy, 1 warm-up run. Device `{{device_name}}`.

| Configuration | prefill_ms | TTFT (ms) | decode tokens/s | % of bandwidth ceiling |
|---|---|---|---|---|
| fused attention, v4 GEMM | `{{llm_prefill_ms}}` | `{{llm_hip_ttft_ms}}` | `{{llm_hip_tok_per_s}}` | `{{llm_pct_bw_ceiling}}` |
| naive attention, v4 GEMM | `{{llm_naive_prefill_ms}}` | `{{llm_hip_naive_ttft_ms}}` | `{{llm_hip_naive_tok_per_s}}` | `{{llm_naive_pct_bw_ceiling}}` |
| `--no-kv-cache` (debug) | `{{llm_nocache_prefill_ms}}` | `{{llm_nocache_ttft_ms}}` | `{{llm_nocache_tok_per_s}}` | n/a |
| PyTorch baseline | n/a | `{{llm_torch_ttft_ms}}` | `{{llm_torch_tok_per_s}}` | `{{llm_torch_pct_bw_ceiling}}` |

Per-kernel time share in one decode step (from the profiler): `{{owner: fill per-kernel table: gemv_xw, gemv_rowdot, attention, layernorm, other}}`.

## Profiler counters to look at

Nsight Compute (`ncu`) on NVIDIA; rocprof-compute (or `rocprofv3 --kernel-trace`) on AMD. There is no wrapper script for the engine; point the profiler at `gpt2` with a short run, for example `./<b>/gpt2 --model models/gpt2 --prompt-file scripts/bench_prompt.txt --max-new-tokens 8`.

- **Kernel time breakdown for one decode step**: which kernel dominates (predicted: `gemv_xw` summed over layers, then `gemv_rowdot`).
- **Achieved DRAM / HBM bandwidth** for `gemv_xw` and `gemv_rowdot` versus peak (predicted: well below peak for `gemv_xw`, closer for `gemv_rowdot`, which has 50257 blocks).
- **Occupancy / active waves and number of blocks launched** for `gemv_xw`, `attention_fused_kernel` and `layernorm_kernel` (predicted: very low).
- **Gaps between kernels in the timeline**: launch overhead (visible in Nsight Systems or the rocprof trace).
- **Coalescing** in `attention_fused_kernel`: its score loop reads each K row with a stride of `d` floats between neighbouring threads, which is uncoalesced.

## What actually happened

`{{owner: fill after benchmarking}}`

## Interview questions

1. **Why does the engine use a GEMM for prefill but a GEMV for decode?** In prefill each weight is reused for all P prompt rows, so it is a GEMM with growing arithmetic intensity and benefits from tiling. In decode there is one row, so each 4-byte weight does one multiply-add (0.5 FLOP/byte): memory-bound, where the goal is to stream the weights in coalesced order at full bandwidth, and tiling gives nothing.
2. **What is the KV-cache and what does it save?** For every layer it stores the keys and values of all earlier tokens. A decode step computes only the new token's q, k, v and attends to the cached rows, so each step pushes one row through the weights and reads O(current length) cached rows, instead of pushing the whole sequence through every layer again. `--no-kv-cache` demonstrates the cost of not having it.
3. **What does fused attention fuse, and what are its limits?** The three naive launches (scores, softmax, weighted sum) become one launch, one block per head, with scores in LDS instead of a global buffer. It requires the sequence to fit in 1024 scores of LDS (engine falls back otherwise), and with only 12 blocks it leaves most of a large GPU idle. Flash-decoding (several blocks per head, merging partial `(m, l, y)`) is the next step.
4. **Explain the online-softmax merge rule.** Keep a running max `m` and sum `l = sum exp(s - m)`. Two partial results over disjoint keys merge as `m' = max(m1, m2)`, `l' = l1 exp(m1 - m') + l2 exp(m2 - m')`: rescaling re-expresses each sum relative to the new max. The merge is associative, so a parallel tree gives the same answer as a serial pass, and subtracting the max keeps `exp` from overflowing.
