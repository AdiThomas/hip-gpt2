# What you still need to do

Everything that could be done without a GPU, an internet-connected Python setup, or your accounts is done and passes
its tests on the CPU backend. The steps below need your hardware, your accounts or your judgement, in order.

## 1. Set up your PC (about 1–2 evenings)
Follow [docs/SETUP.md](docs/SETUP.md) part (a): Ubuntu 24.04 dual-boot, NVIDIA driver + CUDA, HIP-on-NVIDIA, CMake ≥ 3.28.
Then check the CPU build works on your machine too:
```bash
git clone --depth 1 https://github.com/ROCm/HIP-CPU ~/HIP-CPU && sudo apt install libtbb-dev clang
cmake -S . -B build-cpu -DGEMM_BACKEND=cpu -DHIP_CPU_DIR=$HOME/HIP-CPU && cmake --build build-cpu -j && ctest --test-dir build-cpu
```

## 2. First GPU run on your RTX 4060
```bash
cmake -S . -B build-nv -DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=89
cmake --build build-nv -j && ctest --test-dir build-nv --output-on-failure
./build-nv/gemm_bench --kernel all --out results
```
The v5 NVIDIA path (`nvcuda::wmma`) and the hipBLAS baselines have never been compiled. If anything fails, the
easiest fix is to paste [MASTER_PROMPT.md](MASTER_PROMPT.md) into Claude Code in this folder: it detects the GPU,
reproduces the failure and sends it to the right agent.

## 3. Validate against real GPT-2 (needs internet + Python)
```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r tools/requirements.txt
python3 tools/export_reference_logits.py --out models/gpt2 --prompt "The capital of France is" --ref /tmp/ref_all.bin
./build-nv/gpt2 --model models/gpt2 --prompt "The capital of France is" --dump-logits /tmp/cpp_all.bin --dump-mode all
python3 tools/compare_logits.py /tmp/ref_all.bin /tmp/cpp_all.bin --tol 1e-2
./build-nv/gpt2 --model models/gpt2 --prompt "Once upon a time" --max-new-tokens 64     # watch it write
```
Then run the benchmarks (SPEC.md §10 H4) and `python3 scripts/plot_results.py --results results --out docs/img`.

## 4. AMD MI300X run (the numbers that go in the write-up)
- Join the AMD AI Developer Program and claim the cloud credits **only when steps 2–3 pass** (credits expire).
- Follow [docs/SETUP.md](docs/SETUP.md) part (b). In one session: build with `-DCMAKE_HIP_ARCHITECTURES=gfx942`, `ctest`,
  `gemm_bench --kernel all --out results`, the GPT-2 steps, `scripts/profile_amd.sh v4 4096` and `scripts/profile_amd.sh v5 4096`.
  Copy `results/` and the profiles back, then **destroy the droplet** (it bills while it exists).
- This is where v5 MFMA actually runs. The reviewer was about 90% confident in it; if it fails, MASTER_PROMPT.md fixes it on the box.

## 5. Publish
1. Create a public GitHub repo `hip-gpt2`, add a top-level MIT `LICENSE` with your name, and push.
2. CI: the `cpu-tests` job runs by itself. Confirm the ROCm image tag in `.github/workflows/ci.yml` (`rocm/dev-ubuntu-24.04:6.4.1`
   is unverified) on Docker Hub.
3. Optional GPU CI: register your PC as a self-hosted runner with labels `self-hosted,gpu`. Set the repository variables
   `HIP_CMAKE_FLAGS` (e.g. `-DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=89`) and `GPT2_MODEL_DIR`. Allow
   Actions `contents: write`, enable Pages from the `gh-pages` branch, then run
   `python3 scripts/check_regression.py --update-baseline --results results --baseline perf/baselines` once and commit the baseline.
4. Fill the `{{placeholders}}` from your results (Claude Code: "use the gpu-results-update skill"), and write the
   "what actually happened" paragraph in each `docs/stages/vN.md` **yourself**.
5. Edit [docs/BLOG_DRAFT.md](docs/BLOG_DRAFT.md) into your own voice and publish it. Record the demo using
   [docs/DEMO_VIDEO_SCRIPT.md](docs/DEMO_VIDEO_SCRIPT.md).

## 6. Own the code (don't skip this one)
AI agents wrote this code. AMD interviewers will ask you to explain it line by line and may ask you to rewrite a
kernel on a whiteboard. Work through [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md) (4–6 weeks): rewrite each kernel from
`exercises/`, take the 40 interview questions, and use the `hip-stage-tutor` skill in Claude Code to get quizzed.
In interviews, say plainly that you used AI agents to build it, then show you understand every part of it.
That's a strong story; claiming you hand-wrote it is not.
