# Setup guide (Ubuntu 24.04, beginner friendly)

Pick the path that matches the machine you have:

- [(a) Your RTX 4060 PC](#a-rtx-4060-pc-hip-on-nvidia): fastest way to start; tensor-core path (`nvcuda::wmma`).
- [(b) AMD Developer Cloud MI300X](#b-amd-developer-cloud-mi300x): the real AMD target; MFMA path. Costs credits while running.
- [(c) Python environment](#c-python-environment-and-real-gpt-2-validation): for real GPT-2 weights and the PyTorch baseline.
- [(d) GitHub, self-hosted runner and Pages](#d-github-repo-self-hosted-runner-and-pages): the performance CI.

Everything below was written from the repo's build files and workflows, not from running it on a GPU. Package names and web-console labels change between ROCm / CUDA releases and cloud UI updates; where that matters it says so. When a command fails, the vendor install page for your exact version wins over this file.

Terminology: **HIP** is AMD's C++ GPU programming layer. The same source compiles for AMD GPUs (through ROCm) and for NVIDIA GPUs (HIP forwards to CUDA). **ROCm** is AMD's GPU software stack. **gfx942** is the architecture name of the MI300X. **sm_89** (CMake: `89`) is the RTX 4060's.

---

## a0) Windows + NVIDIA GPU via Docker (no Linux install)

This is how the RTX 4060 numbers in this repo were produced: Windows 11, Docker Desktop (WSL2 backend), NVIDIA driver on Windows. No dual-boot and no host toolchain needed. Check GPU passthrough first:

```powershell
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

**Try GPT-2 on your GPU (one command):**

```powershell
.scriptsgpt2.ps1 "Once upon a time"            # one completion
.scriptsgpt2.ps1 -Chat                         # type prompts in a loop
.scriptsgpt2.ps1 "The capital of France is" -Tokens 20 -Temperature 0   # greedy
```

The first run builds the image (CUDA 12.6 + HIP 6.4.3 for the NVIDIA platform + hipBLAS with its cuBLAS backend), builds the engine and downloads GPT-2; later runs take a few seconds. GPT-2 small continues text; it is not a chat assistant.

**Full build, tests and benchmarks** (Git Bash; `MSYS_NO_PATHCONV=1` stops Git Bash rewriting the paths):

```bash
docker build -t hip-gpt2-cpu -f docker/Dockerfile.cpu docker
docker build -t hip-gpt2-nv  -f docker/Dockerfile.nv  docker
R="C:/path/to/hip-gpt2"   # your checkout
# CPU backend
MSYS_NO_PATHCONV=1 docker run --rm -v "$R:/work" -v hipgpt2-build-cpu:/build -w /work hip-gpt2-cpu bash -lc   "git config --global --add safe.directory /work; cmake -S . -B /build/build-cpu -DGEMM_BACKEND=cpu -DHIP_CPU_DIR=/opt/HIP-CPU && cmake --build /build/build-cpu -j && ctest --test-dir /build/build-cpu --output-on-failure"
# NVIDIA GPU
MSYS_NO_PATHCONV=1 docker run --rm --gpus all -v "$R:/work" -v hipgpt2-build-nv:/build -w /work hip-gpt2-nv bash -lc   "git config --global --add safe.directory /work; cmake -S . -B /build/build-nv -DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=89 -DHIPGPT2_ENABLE_HIPBLAS=ON && cmake --build /build/build-nv -j 8 && ctest --test-dir /build/build-nv --output-on-failure && /build/build-nv/gemm_bench --kernel all --out results"
```

Build directories live in named Docker volumes (`hipgpt2-build-*`), because building through the Windows bind mount is slow. For the real-GPT-2 check (section c), run the same commands inside the `hip-gpt2-nv` container with `-v hipgpt2-venv:/venv`; `scripts/docker_gpt2_setup.sh` shows the venv setup.

## a) RTX 4060 PC (HIP on NVIDIA)

### 1. NVIDIA driver

```bash
sudo apt update
sudo ubuntu-drivers devices          # shows the recommended driver
sudo ubuntu-drivers install          # installs it
sudo reboot
nvidia-smi                           # must show the RTX 4060
```

### 2. CUDA toolkit (12.x)

Install the CUDA Toolkit 12.x from NVIDIA's apt repository for Ubuntu 24.04 (follow NVIDIA's "CUDA Installation Guide for Linux", network-repo option). Then check:

```bash
nvcc --version                       # if not found: export PATH=/usr/local/cuda/bin:$PATH
```

Add the `PATH` line to `~/.bashrc` if you needed it. Install the toolkit only; the driver from step 1 stays.

### 3. ROCm apt repo and the HIP-on-NVIDIA packages

Add AMD's ROCm apt repository by following the "ROCm install" page for Ubuntu 24.04 (it either gives you an `amdgpu-install` .deb or a manual repo setup). You do not need AMD's kernel driver for this; you only want the HIP user-space packages. Then:

```bash
sudo apt update
sudo apt install hip-dev hip-runtime-nvidia
echo 'export HIP_PLATFORM=nvidia' >> ~/.bashrc && source ~/.bashrc
/opt/rocm/bin/hipconfig --full       # should report platform: nvidia
```

Package names have changed between ROCm releases. If `apt` cannot find them, search (`apt search hip-runtime`) and check the "Install HIP" page for your ROCm version.

Optional, for the hipBLAS baseline rows in the results tables: install the hipBLAS development package for the same ROCm release (typically `hipblas-dev`; on NVIDIA it needs a matching hipBLAS-on-cuBLAS build, which may not exist for every release). If CMake does not find hipBLAS, the build still succeeds and `hipblas` / `hipblas_fp16` show as "built without hipBLAS" in `gemm_bench --list`. To make a missing hipBLAS an error instead, add `-DHIPGPT2_ENABLE_HIPBLAS=ON`.

### 4. CMake 3.28 or newer

Ubuntu 24.04's own `cmake` package is 3.28, which is enough:

```bash
sudo apt install cmake ninja-build g++ git python3 python3-venv
cmake --version                      # >= 3.28
```

If it is older, `pip install cmake` in a virtual environment works.

### 5. Build, test, benchmark

```bash
git clone <your repo url> hip-gpt2 && cd hip-gpt2
cmake -S . -B build-nv -DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=89
cmake --build build-nv -j
ctest --test-dir build-nv --output-on-failure
./build-nv/gemm_bench --list
./build-nv/gemm_bench --kernel v5 --test
./build-nv/gemm_bench --kernel all --out results
```

What to expect: CMake prints `hip-gpt2: backend=gpu matrix-core=WMMA_NVIDIA`. `ctest` runs nine tests (eight if clang is not installed), including `gemm_correctness` on GPU-sized shapes. `v5` here is the `nvcuda::wmma` path and has never been compiled; if it fails to build, that is a bug to fix in `src/kernels/v5_wmma_nvidia.inc`, not a setup problem.

---

## b) AMD Developer Cloud MI300X

Credits are consumed for every hour a droplet exists. Do step 6 every time.

1. **Get credits.** Join the AMD AI Developer Program (free signup on AMD's developer site). Members can request AMD Developer Cloud credits. Approval and the amount change over time; read the program page.
2. **Create a droplet.** In the AMD Developer Cloud console, create a GPU droplet with a **single MI300X** and an image that already has ROCm (a ROCm / PyTorch-on-ROCm image; the exact labels vary). Add your SSH public key (`cat ~/.ssh/id_ed25519.pub`; make a key with `ssh-keygen -t ed25519` if you have none).
3. **Log in and check the GPU.**
   ```bash
   ssh root@<droplet-ip>
   rocm-smi                           # one MI300X
   rocminfo | grep -m1 gfx            # gfx942
   ls /opt/rocm/bin/hipcc
   ```
   Some images run your shell inside a container; if `/opt/rocm` is missing, look for the ROCm environment the image documents.
4. **Install build tools and clone.**
   ```bash
   apt update && apt install -y cmake ninja-build git python3 python3-venv
   cmake --version                    # >= 3.21 is enough for AMD; 3.28 only matters for NVIDIA
   git clone <your repo url> hip-gpt2 && cd hip-gpt2
   ```
   If hipBLAS headers are missing (`find_package(hipblas)` fails; the build continues without the baseline), `apt install hipblas-dev`.
5. **Build for gfx942, test, run.**
   ```bash
   cmake -S . -B build-amd -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=/opt/rocm
   cmake --build build-amd -j
   ctest --test-dir build-amd --output-on-failure
   ./build-amd/gemm_bench --kernel v5 --test
   ./build-amd/gemm_bench --kernel all --out results
   BUILD_DIR=build-amd scripts/profile_amd.sh v4 4096     # optional, needs rocprof-compute
   ```
   CMake should print `matrix-core=MFMA`. Copy what you need back before shutting down:
   ```bash
   scp -r root@<droplet-ip>:hip-gpt2/results ./results-mi300x
   ```
6. **Shut the droplet down.** Destroy it (not just "power off") from the console once your files are copied; powered-off GPU droplets can still bill, so check the console's billing text. Confirm in the billing page that no droplet is listed.

Tip: do all editing on your own machine, push to GitHub, and on the droplet only `git pull`, build and benchmark. Write down the commands you run so a session is under an hour.

---

## c) Python environment and real-GPT-2 validation

The C++ tests use a tiny random model checked into the repo, so they need no Python. Python is needed to fetch real GPT-2 weights, make reference logits, and run the PyTorch baseline.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r tools/requirements.txt      # numpy, safetensors, regex, matplotlib, torch, transformers
```

`torch` and `transformers` are only used by the owner-run scripts. On an AMD machine, install PyTorch's ROCm build instead of the default wheel (use the selector on pytorch.org; the index URL looks like `https://download.pytorch.org/whl/rocm<version>`) and install the other packages from the requirements file around it. On NVIDIA the default `pip install torch` is the CUDA build.

**Real GPT-2 correctness check (SPEC H3).** `<b>` is your build directory:

```bash
python3 tools/export_reference_logits.py --out models/gpt2 --prompt "The capital of France is" --ref /tmp/ref_all.bin
./<b>/gpt2 --model models/gpt2 --prompt "The capital of France is" --dump-logits /tmp/cpp_all.bin --dump-mode all
python3 tools/compare_logits.py /tmp/ref_all.bin /tmp/cpp_all.bin --tol 1e-2
```

The first command downloads `gpt2` from Hugging Face and writes `models/gpt2/` (`model.safetensors`, `config.json`, tokenizer files); `models/` is git-ignored. `compare_logits.py` prints the max absolute error and argmax agreement and exits 1 above the tolerance. If `gpt2` errors with a missing tensor, the key names in the real checkpoint differ from what the engine expects (it tries `<name>` and `transformer.<name>`); report the printed name.

For the greedy-token check, also pass `--greedy-ref /tmp/ref_greedy.bin --greedy-tokens-out /tmp/ref_tokens.txt --max-new-tokens 32` to the export script, then run `gpt2 ... --max-new-tokens 32 --dump-logits /tmp/cpp_greedy.bin` (default `--dump-mode last`) and compare the two files.

**End-to-end numbers (SPEC H4):**

```bash
./<b>/gpt2 --model models/gpt2 --prompt-file scripts/bench_prompt.txt --max-new-tokens 128 --warmup 1 --bench-out results
python3 tools/torch_generation_baseline.py --model models/gpt2 --prompt-file scripts/bench_prompt.txt --max-new-tokens 128 --out results
```

Both write a JSON file into `results/` (`llm_<device>.json` and `llm_pytorch_<device>.json`).

---

## d) GitHub repo, self-hosted runner and Pages

### 1. Create the repo and push

```bash
git init -b main                      # skip if already a repo
git add -A && git commit -m "Initial import"
gh repo create hip-gpt2 --public --source . --push     # or create it on github.com and `git remote add origin ...`
```

Choose a license when you create it (MIT is suggested) or add `LICENSE` later. The `ci.yml` workflow starts running on the first push.

### 2. Self-hosted runner (the machine with the GPU)

On the GitHub repo: Settings, Actions, Runners, "New self-hosted runner". Follow the commands it shows for Linux x64, and when configuring add the label `gpu`:

```bash
./config.sh --url https://github.com/<you>/hip-gpt2 --token <token-from-page> --labels gpu
sudo ./svc.sh install && sudo ./svc.sh start     # run as a service
```

`self-hosted` is added automatically; `bench.yml` needs both labels: `runs-on: [self-hosted, gpu]`. The runner machine needs `git`, `cmake`, `python3`, the GPU driver and ROCm (or CUDA + HIP), all on the runner service's `PATH`. Check in the runner's `.env` file if a tool is "not found" only inside Actions.

Security: `bench.yml` runs only on pushes to `main` and manual dispatch, never on pull requests, which is what keeps strangers' code off your runner. Do not change that on a public repo.

Cost note: a cloud MI300X droplet that you destroy after each session cannot be a permanent runner. Realistic options are the RTX 4060 PC as the runner, or registering a runner on a droplet only for the length of a session.

### 3. Repository variables

Settings, Secrets and variables, Actions, tab **Variables**:

| Variable | Value | Notes |
|---|---|---|
| `HIP_CMAKE_FLAGS` | AMD: `-DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=/opt/rocm`<br>NVIDIA: `-DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=89` | Passed word-split to `cmake -S . -B build`. Required: with it empty the configure step has no architecture. |
| `GPT2_MODEL_DIR` | absolute path on the runner, e.g. `/home/runner/models/gpt2` | Optional. If unset, the GPT-2 benchmark is skipped and only GEMM is tracked. Create the directory with step (c). |
| `PERF_THRESHOLD_PCT` | e.g. `7` | Optional; default 7. See the baseline note below before tightening it. |

### 4. Pages from `gh-pages`

`bench.yml` creates the `gh-pages` branch on its first successful benchmark run and copies everything in `site/` (the write-up `index.html`, the history chart `perf.html` + `chart.js`) and appends `data/history.json`. After that run: Settings, Pages, "Deploy from a branch", branch `gh-pages`, folder `/ (root)`. The write-up appears at `https://<you>.github.io/hip-gpt2/` and the history chart at `.../perf.html` a minute or so later. Use this URL in the README and the demo video.

### 5. Workflow permissions

`bench.yml` asks for `permissions: contents: write` so it can push `gh-pages` with the built-in token. If the push fails with a 403, go to Settings, Actions, General, Workflow permissions, and choose **Read and write permissions** (an organization policy can cap this).

### 6. Baselines (needed for the regression check to mean anything)

The first run has no baseline, so the check passes with a notice. After a run you trust, on the runner or from the downloaded artifact:

```bash
python3 scripts/check_regression.py --results results --baseline perf/baselines --update-baseline
git add perf/baselines && git commit -m "Update perf baseline" && git push
```

`--update-baseline` refuses if any result has `verified: false`. Re-run the benchmark a few times and compare before committing: if two identical runs differ by more than a few percent, raise `PERF_THRESHOLD_PCT` or the job will fail on noise. Note the CI sizes are 1024, 2048 and 4096 only (`scripts/run_benchmarks.sh`; override with the `GEMM_SIZES` environment variable); 8192 for the README table is a manual run.

### 7. Confirm the ROCm docker tag in `ci.yml`

The `rocm-compile` job uses `container: image: rocm/dev-ubuntu-24.04:6.4.1`, written without checking that the tag exists. Look it up on Docker Hub (`rocm/dev-ubuntu-24.04`) and change the tag in `.github/workflows/ci.yml` if it does not. The job is `continue-on-error`, so a wrong tag shows as a red-but-ignored job, not a failed workflow. The image is large; if the hosted runner runs out of disk, use a slimmer tag or a larger runner.
