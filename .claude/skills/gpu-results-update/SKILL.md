---
name: gpu-results-update
description: Run hip-gpt2 benchmarks on the GPU in this machine and refresh every results table, chart and doc placeholder from the new JSON. Use after any kernel change, on a new GPU, or before publishing the blog post.
---
# Refresh results

1. `nvidia-smi` or `rocm-smi` to identify the GPU. Build with the matching command in README.md ("Build").
2. `ctest --test-dir build --output-on-failure` — stop on any failure.
3. `./build/gemm_bench --kernel all --out results` (runs every FP32 and FP16 kernel) → `results/gemm_*.json`.
4. Engine (needs `models/gpt2/`, see SPEC.md section 10 H3): `./build/gpt2 --model models/gpt2 --prompt-file scripts/bench_prompt.txt --max-new-tokens 128 --warmup 1 --quiet --bench-out results`
   and `python3 tools/torch_generation_baseline.py --model models/gpt2 --prompt-file scripts/bench_prompt.txt --max-new-tokens 128 --out results`.
5. `python3 scripts/plot_results.py --results results --out docs/img --theme light` → charts.
6. Delegate placeholder filling in
   README.md, docs/stages/*.md and docs/BLOG_DRAFT.md to the `doc-filler` agent.
7. Commit results and charts with the device name in the message. Never hand-edit a number.
