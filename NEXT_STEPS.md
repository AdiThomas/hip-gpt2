# What's left to do

Updated 2026-11-18. Steps 1–3 of the original plan are done on the RTX 4060 (via Docker on Windows). The repo is public, and the write-up is live at https://adithomas.github.io/hip-gpt2/.

## Done
- CPU build + `ctest` 9/9, in `docker/Dockerfile.cpu`.
- NVIDIA build (HIP 6.4.3 on CUDA 12.6, hipBLAS on cuBLAS), `ctest` 9/9, all 10 kernels pass `--test`, including v5 `nvcuda::wmma`.
- Benchmarks: `results/gemm_*.json`, `results/llm_*.json`, plots in `docs/img/`.
- Real GPT-2 vs PyTorch: max logit error 3.2e-4, 32/32 greedy tokens (`results/h3_*.json`). Fixed the export script dropping `c_attn.bias`.
- `{{placeholders}}` filled from results by `scripts/fill_placeholders.py`. Write-up page `site/index.html` built from `site/data/results.json` (`scripts/build_site_data.py`).
- ROCm CI image tag verified. MIT LICENSE added.

## Still to do (in order)

1. **Profile on the 4060.** `scripts/profile_nvidia.sh v4 4096` and `v5 4096` (Nsight Compute may be blocked under WSL2; if so, skip to MI300X). Use the counters to confirm or replace each "hypothesis" in `docs/BLOG.md` and `site/index.html`.
2. **AMD MI300X run** (credits expire, so do it in one session): `docs/SETUP.md` part (b). Build with `-DCMAKE_HIP_ARCHITECTURES=gfx942`, `ctest`, `gemm_bench --kernel v5 --test`, `gemm_bench --kernel all --out results`, the GPT-2 steps (SETUP part c), `scripts/profile_amd.sh v4 4096` and `v5 4096`. Copy `results/` back, then **destroy the droplet**. Then:
   `python3 scripts/fill_placeholders.py --device "<MI300X device name>"` (add its datasheet row to `DEVICE_SPECS` first), `python3 scripts/build_site_data.py`, and write part 2.
3. **Your own words.** Write the "what actually happened" paragraph in each `docs/stages/vN.md` (the `{{owner: ...}}` spots), and edit `docs/BLOG.md` / `site/index.html` into your voice.
4. **Optional GPU CI.** Register this PC as a self-hosted runner (labels `self-hosted,gpu`), set repo variables `HIP_CMAKE_FLAGS` and `GPT2_MODEL_DIR`, then `python3 scripts/check_regression.py --update-baseline --results results --baseline perf/baselines` and commit the baseline. Note: the runner job expects a Linux host with HIP installed, so on Windows it needs adapting to run through Docker.
5. **Demo video**: `docs/DEMO_VIDEO_SCRIPT.md`. `.\scripts\gpt2.ps1 -Chat` makes a good live segment.
6. **Own the code** (don't skip this one): `docs/STUDY_GUIDE.md`, 4–6 weeks. Rewrite each kernel from `exercises/`, take the 40 interview questions, and use the `hip-stage-tutor` skill. In interviews, say plainly that AI agents wrote much of the code, then show you understand every line.
