# hip-gemm

A GEMM optimization ladder written in HIP. The same source builds for AMD (ROCm) and NVIDIA (CUDA backend).
It grows into a small LLM inference engine later in the project.

> Status: harness done, **v0 is yours to write** (`src/kernels/v0_naive.hip`).

## Layout

```
include/gemm.hpp          kernel contract + HIP_CHECK
include/reference.hpp     CPU correctness checker
src/reference.cpp         checker (double-precision dot products, OpenMP)
src/bench.hip             benchmark + test harness, writes JSON to results/
src/kernels/vN_*.hip      one file per optimization stage  <-- your code
tests/test_reference.cpp  tests the checker itself (no GPU needed)
docs/stages/vN.md         prediction -> measurement -> explanation, per stage
```

## Setup: NVIDIA (RTX 4060, Ubuntu 24.04)

1. Install the NVIDIA driver and the CUDA Toolkit (12.x) from NVIDIA's apt repo. Check that `nvidia-smi` and `nvcc --version` both work.
2. Add AMD's ROCm apt repo, following the ROCm install docs for Ubuntu 24.04. Then install the HIP layer for NVIDIA:
   ```bash
   sudo apt install hip-dev hip-runtime-nvidia
   export HIP_PLATFORM=nvidia      # add to ~/.bashrc
   /opt/rocm/bin/hipconfig --full  # should report the nvidia platform
   ```
   Package names have changed between ROCm releases. If apt can't find them, check the "Install HIP" page for your ROCm version.
3. Use CMake 3.28 or newer (`pip install cmake` if apt's is older).

## Setup: AMD (MI300X cloud / Radeon)

Install ROCm with `amdgpu-install` per the ROCm docs. AMD Developer Cloud images ship with ROCm already installed.

## Build and run

```bash
# NVIDIA
cmake -B build -DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=89
# AMD: MI300X=gfx942, RX 7900=gfx1100, RX 9070=gfx1201
cmake -B build -DCMAKE_HIP_ARCHITECTURES=gfx942

cmake --build build -j
ctest --test-dir build --output-on-failure    # checker test + per-kernel correctness

./build/gemm_bench --kernel v0 --test         # awkward sizes, exits 1 on failure
./build/gemm_bench --kernel v0                # 512..8192 sweep -> results/v0_fp32_<device>.json
./build/gemm_bench --kernel all --sizes 1024,4096x1024x512
```

## How correctness is checked

- C is filled with NaN before each kernel launch. If the kernel skips an element, the check fails.
- The error metric is `|got − exact| / Σ|aᵢₖ·bₖⱼ|`, with the exact value computed in double precision. Default tolerance is 1e-4. A real bug gives an error around 1.
- If M·N·K ≤ 2048³, every element is checked. Above that, the four corners plus 4096 random elements are checked. A full CPU check at 8192³ would take minutes.
- The test sizes are deliberately awkward (1, primes, sizes that aren't a multiple of any tile) so that edge-handling bugs show up.

## Adding a stage

1. Create `src/kernels/v1_<name>.hip` with `void launch_v1(...)`.
2. Declare it in `include/gemm.hpp` and register it in the `kKernels` table in `src/bench.hip`.
3. Add `v1` to the `foreach` in `CMakeLists.txt`.
4. Copy `docs/stages/TEMPLATE.md` to `docs/stages/v1.md`, and fill in the prediction **before** you benchmark.

## Results

_(tables + roofline charts go here as stages land)_
