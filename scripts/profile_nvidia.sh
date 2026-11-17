#!/usr/bin/env bash
# Profile one GEMM kernel at one size with Nsight Compute (`ncu`), the NVIDIA
# counterpart of profile_amd.sh.
#
# Usage:   scripts/profile_nvidia.sh <kernel> <size>      e.g. scripts/profile_nvidia.sh v4 4096
# Env:     BUILD_DIR   build directory containing gemm_bench   (default: build)
#          NCU_EXTRA   extra args for ncu, e.g. "--kernel-name regex:gemm"
#
# Output:  profiles/gemm_<kernel>_<size>.ncu-rep
#          Open with `ncu-ui profiles/<name>.ncu-rep`, or print in the terminal:
#          ncu --import profiles/<name>.ncu-rep --page details
#          (SpeedOfLight / Roofline / MemoryWorkloadAnalysis sections hold the
#          counters that match the AMD ones: bank conflicts, occupancy, hit rates.)
# `--set full` replays every kernel many times; the run below is kept short.
# ncu may need `sudo` or the NVreg_RestrictProfilingToAdminUsers=0 driver option.
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
command -v ncu >/dev/null || { echo "error: ncu not in PATH (install Nsight Compute)" >&2; exit 2; }

mkdir -p profiles
# shellcheck disable=SC2086  # NCU_EXTRA is intentionally word-split
ncu --set full -f -o "profiles/$name" ${NCU_EXTRA:-} \
  "$bench" --kernel "$kernel" --sizes "$size" --iters 20
echo "wrote profiles/$name.ncu-rep"
