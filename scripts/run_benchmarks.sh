#!/usr/bin/env bash
# Run the CI benchmark suite and write result JSON (SPEC 8.1 / 8.2) into results/.
# Env: BUILD_DIR       build directory                      (default: build)
#      RESULTS_DIR     output directory                     (default: results)
#      GEMM_SIZES      comma-separated sizes                (default: 1024,2048,4096)
#      GPT2_MODEL_DIR  GPT-2 model directory; LLM bench is skipped when unset/empty
set -euo pipefail

build_dir=${BUILD_DIR:-build}
out=${RESULTS_DIR:-results}
sizes=${GEMM_SIZES:-1024,2048,4096}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$out"

"$build_dir/gemm_bench" --kernel all --sizes "$sizes" --out "$out"

if [[ -n "${GPT2_MODEL_DIR:-}" ]]; then
  "$build_dir/gpt2" --model "$GPT2_MODEL_DIR" --prompt-file "$script_dir/bench_prompt.txt" \
    --max-new-tokens 128 --warmup 1 --quiet --bench-out "$out"
else
  echo "GPT2_MODEL_DIR not set: skipping LLM benchmark"
fi
