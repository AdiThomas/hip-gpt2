#!/usr/bin/env bash
# Profile one GEMM kernel at one size with ROCm Compute Profiler
# (formerly Omniperf; CLI: `rocprof-compute`), then print the analysis.
#
# Usage:   scripts/profile_amd.sh <kernel> <size>      e.g. scripts/profile_amd.sh v4 4096
# Env:     BUILD_DIR   build directory containing gemm_bench   (default: build)
#          PROFILE_EXTRA  extra args for `rocprof-compute profile`, e.g. "--roof-only"
#
# Output:  workloads/gemm_<kernel>_<size>/<arch>/   (raw data, roofline PDFs/PNGs)
#
# Notes (verify flags with `rocprof-compute profile --help` for your ROCm version):
#   * `profile` collects the counters in several kernel replays; use a short run
#     (--iters 20 as below) because each replay re-executes the whole command.
#   * Roofline: by default `profile` also measures the roofline ceilings and
#     writes roofline plots next to the data. `--roof-only` profiles only the
#     roofline (fast); `--no-roof` skips it. To (re)print or restrict analysis:
#       rocprof-compute analyze -p workloads/<name>/<arch> --list-stats
#       rocprof-compute analyze -p workloads/<name>/<arch> -b 2        # LDS block
#       rocprof-compute analyze -p workloads/<name>/<arch> --kernel <id>
#   * Counters of interest: LDS bank conflicts, occupancy, L1/L2 hit rate,
#     achieved HBM bandwidth, VALU/MFMA utilization.
#   * MI300X (gfx942) is the supported target; consumer RDNA cards are not.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <kernel> <size>   (env BUILD_DIR, default: build)" >&2
  exit 2
fi
kernel=$1
size=$2
build_dir=${BUILD_DIR:-build}
bench="$build_dir/gemm_bench"
name="gemm_${kernel}_${size}"

[[ -x "$bench" ]] || { echo "error: $bench not found; build first (BUILD_DIR=$build_dir)" >&2; exit 2; }
command -v rocprof-compute >/dev/null || { echo "error: rocprof-compute not in PATH (is ROCm installed?)" >&2; exit 2; }

# shellcheck disable=SC2086  # PROFILE_EXTRA is intentionally word-split
rocprof-compute profile -n "$name" ${PROFILE_EXTRA:-} -- \
  "$bench" --kernel "$kernel" --sizes "$size" --iters 20

# The data lands in workloads/<name>/<arch> (e.g. MI300X_A1); pick the arch dir.
arch_dir=$(find "workloads/$name" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
[[ -n "$arch_dir" ]] || { echo "error: no output found under workloads/$name" >&2; exit 1; }
echo "analyzing $arch_dir"
rocprof-compute analyze -p "$arch_dir"
