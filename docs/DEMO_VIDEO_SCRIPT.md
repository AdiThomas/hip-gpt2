# Demo video script (target about 2:30 to 2:40, hard limits 2:00 to 3:00)

Goal: show that the engine generates text, show the performance numbers on screen, show how the kernels got there, and show the CI chart. Record on the AMD MI300X droplet (or the RTX 4060 PC), terminal font large (at least 18 pt), dark theme, 1080p.

## Before recording (checklist)

- Build is done: `cmake -S . -B build-amd -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=/opt/rocm && cmake --build build-amd -j` (or `build-nv` on NVIDIA). Below, `B` means your build directory: `export B=build-amd`.
- Real GPT-2 exists at `models/gpt2` (see `docs/SETUP.md` section c).
- `results/` holds a full benchmark run; `docs/img/ladder_fp32.png` exists (`python3 scripts/plot_results.py --results results --out /tmp/plots`, then copy `gemm_ladder_fp32.png`).
- The Pages chart has at least a few commits of history; open `https://adithomas.github.io/hip-gpt2/` in a browser tab. Prepare a README tab showing the results table.
- Do one dry run of every command so the GPU is warm and the timings on screen are representative. Clear the terminal (`clear`) before each shot.
- If you are on a paid droplet, keep the whole recording session short and destroy the droplet afterwards.

## Shot list

| # | Time | On screen | Say (spoken line) |
|---|---|---|---|
| 1 | 0:00 to 0:12 | Title card or the README top: "hip-gpt2". | "This is hip-gpt2: a GPT-2 inference engine in C++ and HIP with no PyTorch at inference time, built on matrix-multiply kernels I wrote and optimized step by step, running on an AMD MI300X." |
| 2 | 0:12 to 0:50 | Terminal. Run the engine on real GPT-2 with streamed text and the stats block. | "First, it works. I give it a prompt and it generates text on the GPU. The tokenizer, the weight loader and every kernel are my code. The stats at the end are measured by the engine: time to first token, and decode tokens per second." (While the text streams, point at the stats block when it prints.) |
| 3 | 0:50 to 1:10 | Terminal. Run the PyTorch baseline for the same prompt (or show the prepared comparison table in the README). | "Same prompt, same weights, same measurement definitions, against PyTorch on ROCm. Here is the comparison: `295` versus `83` tokens per second." |
| 4 | 1:10 to 1:45 | Terminal: `gemm_bench --list`, then the ladder run at 4096 (table scrolls). Then the ladder chart image. | "Underneath, the engine's matrix multiplies come from a ladder of six kernels. v0 is naive. Coalesced memory access, then shared-memory tiling, then register blocking, then vector loads with double buffering, then matrix-core instructions. Each stage is checked against a CPU reference and compared with the vendor library. Here is FP32 at 4096: from `244` to `5,839` GFLOPS, and in FP16 the matrix-core kernel reaches `14,821`." |
| 5 | 1:45 to 2:05 | Briefly show the profiler output or roofline PNG for v4 or v5. | "For each stage I predicted the bottleneck from bytes moved per FLOP, then confirmed or refuted it with the profiler. For example, v5 is limited by how fast it can feed the matrix cores, not by the cores themselves: `{{owner: one concrete profiler finding}}`." |
| 6 | 2:05 to 2:30 | Browser: the GitHub Pages chart, then the Actions tab with a green bench run (or a red one that caught a regression). | "Every push to main runs on the GPU: correctness tests, benchmarks, and a regression check that fails the build if any kernel gets more than `7` percent slower. The history is published here, so performance is tracked per commit." |
| 7 | 2:30 to 2:40 | README top with the repo URL. | "Code, results and the write-up are in the repo. Thanks for watching." |

## Exact commands, in order

Shot 2, generate text with the stats block (leave out `--quiet` so the text streams):

```bash
./$B/gpt2 --model models/gpt2 --prompt "The history of GPU computing" --max-new-tokens 64 --warmup 1
```

The output ends with `prompt_tokens`, `gen_tokens`, `load_ms`, `ttft_ms` and `decode_tok_per_s`. For a reproducible run on screen add `--temperature 0.8 --top-k 40 --seed 1234`; the default is greedy.

Shot 3, PyTorch baseline (writes JSON; use `cat results/llm_pytorch_*.json` to show it) and the same measured prompt for the engine:

```bash
./$B/gpt2 --model models/gpt2 --prompt-file scripts/bench_prompt.txt --max-new-tokens 128 --warmup 1 --quiet --bench-out results
python3 tools/torch_generation_baseline.py --model models/gpt2 --prompt-file scripts/bench_prompt.txt --max-new-tokens 128 --out results
```

Run both before recording and show the results table instead if the PyTorch run is slow to start.

Shot 4, the ladder:

```bash
./$B/gemm_bench --list
./$B/gemm_bench --kernel all --sizes 4096 --iters 20
```

(`--out` defaults to `results`, so this also writes JSON there. Use `--kernel v0 --sizes 1024` instead if v0 at 4096 makes the live run too slow; or play a sped-up recording.) Then show `docs/img/ladder_fp32.png` full screen.

Shot 5, one profile (run it before recording; only show the output):

```bash
BUILD_DIR=$B scripts/profile_amd.sh v4 4096        # on NVIDIA: scripts/profile_nvidia.sh v4 4096
```

Shot 6: open `https://adithomas.github.io/hip-gpt2/` and the repository's Actions tab (workflow "bench").

## Tips

- Never show a number you have not measured on the machine you are recording on. If a number in a spoken line is still a placeholder, cut the sentence.
- The first call of the GPU pays one-time costs; use `--warmup 1` so the stats on screen are representative.
- Keep the voice-over plain: say what the kernel does before its name.
- Export at 1080p, add captions (the terminal text is small on a phone).
