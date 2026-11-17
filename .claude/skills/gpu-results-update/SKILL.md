---
name: gpu-results-update
description: Run hip-gpt2 benchmarks on the GPU in this machine and refresh every results table, chart and doc placeholder from the new JSON. Use after any kernel change, on a new GPU, or before publishing the blog post.
---
# Refresh results

1. `nvidia-smi` or `rocm-smi` to identify the GPU. Build with the matching command in README.md ("Build").
2. `ctest --test-dir build --output-on-failure` — stop on any failure.
3. `./build/gemm_bench --kernel all` and `./build/gemm_bench --kernel all --dtype fp16` → `results/*.json`.
4. Engine: the `gpt2_bench` command in README.md → `results/llm_*.json` (needs `models/gpt2/` — see README).
5. `python3 scripts/plot_results.py results/ docs/img/` → charts.
6. `python3 scripts/summarize_results.py results/ > docs/RESULTS.md` (if present), then delegate placeholder filling in
   README.md, docs/stages/*.md and docs/BLOG_DRAFT.md to the `doc-filler` agent.
7. Commit results and charts with the device name in the message. Never hand-edit a number.
