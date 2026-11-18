#!/usr/bin/env python3
"""Fill {{placeholders}} in README.md and docs/ from results/*.json for one device.

Every value is computed from a results file, except the few datasheet constants in DEVICE_SPECS,
which are labelled as datasheet values wherever they are used. Placeholders with no value
(owner prose, profiler-only numbers, other devices) are left untouched.

  python3 scripts/fill_placeholders.py --device "NVIDIA GeForce RTX 4060" [--dry-run]
  python3 scripts/fill_placeholders.py --device "..." --print-values   # dump the value table as JSON
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["README.md", "docs/PROJECT_EXPLAINED.md", "docs/STUDY_GUIDE.md", "docs/DEMO_VIDEO_SCRIPT.md"]
TARGET_GLOBS = ["docs/stages/*.md"]
SIZES = (1024, 2048, 4096, 8192)

# Datasheet constants (not measurements). Bandwidth in GB/s, peaks in TFLOPS.
DEVICE_SPECS = {
    "NVIDIA GeForce RTX 4060": {"bw_gbs": 272, "fp32_tflops": 15.1, "stack": "HIP 6.4.3 on CUDA 12.6 (NVIDIA platform)"},
}
WEIGHT_BYTES_PER_TOKEN = 500e6  # ~124M FP32 parameters streamed once per decode step (docs use "about 500 MB")
REPO_URL = "https://github.com/AdiThomas/hip-gpt2"
PAGES_URL = "https://adithomas.github.io/hip-gpt2/"
THRESHOLD_PCT = 7


def load(results, device):
    gemm, llm, extra = {}, {}, {}
    for p in sorted(Path(results).rglob("*.json")):
        try:
            o = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(o, dict) or o.get("device") != device:
            continue
        b = o.get("bench")
        if b == "gemm":
            gemm[o["kernel"]] = {r["M"]: r for r in o["results"] if r["M"] == r["N"] == r["K"]}
            extra.setdefault("git_sha", o.get("git_sha"))
            extra.setdefault("date", o.get("timestamp", "")[:10])
        elif b in ("llm", "llm_variant"):
            key = o["engine"]
            if o.get("kv_cache") is False:
                key += "_nocache"
            elif o.get("attn") == "naive":
                key += "_naive"
            llm[key] = o
        elif b in ("h3", "noise"):
            extra[b] = o
    return gemm, llm, extra


def g(x):
    return f"{x:,.0f}"


def values(device, gemm, llm, extra):
    v = {"device_name": device, "repo_url": REPO_URL, "pages_url": PAGES_URL, "threshold_pct": str(THRESHOLD_PCT)}
    if extra.get("git_sha"):
        v["git_sha"] = extra["git_sha"]
        v["bench_date"] = extra["date"]
    spec = DEVICE_SPECS.get(device, {})
    if spec:
        v["rocm_version"] = spec["stack"]
        v["device_hbm_bandwidth_gbs"] = f"{spec['bw_gbs']} (datasheet)"
        v["device_fp32_vector_peak_tflops"] = f"{spec['fp32_tflops']} (datasheet)"
        ceiling = spec["bw_gbs"] * 1e9 / WEIGHT_BYTES_PER_TOKEN
        v["decode_bandwidth_ceiling_tok_per_s"] = v["decode_bandwidth_ceiling_tok_s"] = f"~{ceiling:.0f} tokens/s"
        v["roofline_ridge_flop_per_byte"] = f"~{spec['fp32_tflops'] * 1e3 / spec['bw_gbs']:.0f} FLOP/byte (FP32, datasheet)"
    v["flops_4096"] = f"{2 * 4096 ** 3 / 1e9:.0f} GFLOP"

    def gf(k, n):
        return gemm.get(k, {}).get(n, {}).get("gflops")

    names = {"v0": "v0", "v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4", "hipblas": "hipblas"}
    for n in SIZES:
        for k in names:
            if gf(k, n):
                v[f"{k}_fp32_{n}_gflops"] = g(gf(k, n))
        for k, key in (("v0_fp16", "v0"), ("v4_fp16", "v4"), ("v5", "v5"), ("hipblas_fp16", "hipblas")):
            if gf(k, n):
                v[f"{key}_fp16_{n}_gflops"] = g(gf(k, n))
        prev = {"v1": "v0", "v2": "v1", "v3": "v2", "v4": "v3", "v5": "v4_fp16"}
        for k, p in prev.items():
            if gf(k, n) and gf(p, n):
                v[f"{k}_speedup_{n}"] = f"{gf(k, n) / gf(p, n):.1f}"
        for k, base in (("v0", "hipblas"), ("v4", "hipblas"), ("v5", "hipblas_fp16")):
            if gf(k, n) and gf(base, n):
                v[f"{k}_pct_hipblas_{n}"] = f"{100 * gf(k, n) / gf(base, n):.0f}%"
    if "v4_pct_hipblas_4096" in v:
        v["v4_pct_of_hipblas"] = v["v4_pct_hipblas_4096"]
    if spec and gf("v4", 4096):
        v["v4_fp32_pct_of_peak"] = f"{gf('v4', 4096) / (spec['fp32_tflops'] * 1e3) * 100:.0f}% of the {spec['fp32_tflops']} TFLOPS datasheet peak"
    if gf("v0", 4096):
        v["v0"] = f"{gf('v0', 4096) / 1e3:.2f}"
    if gf("v5", 4096):
        v["v5"] = f"{gf('v5', 4096) / 1e3:.1f}"

    def llm_fill(src, prefix):
        o = llm.get(src)
        if not o:
            return
        v[f"{prefix}_tok_per_s"] = f"{o['decode_tok_per_s']:.0f}"
        v[f"{prefix}_ttft_ms"] = f"{o['ttft_ms']:.1f}"
        if spec:
            v[f"{prefix}_pct_bw_ceiling"] = f"{100 * o['decode_tok_per_s'] / ceiling:.0f}%"
        return o

    o = llm_fill("hip-gpt2", "llm_hip")
    if o:
        v["llm_prefill_ms"] = f"{o['prefill_ms']:.1f}"
        v["decode_tok_per_s"] = f"{o['decode_tok_per_s']:.0f} tokens/s"
        v["gemm_kernel"] = o.get("gemm_kernel", "v4")
        if spec:
            v["llm_pct_bw_ceiling"] = v.pop("llm_hip_pct_bw_ceiling")
    o = llm_fill("hip-gpt2_naive", "llm_hip_naive")
    if o:
        v["llm_naive_prefill_ms"] = f"{o['prefill_ms']:.1f}"
        if spec:
            v["llm_naive_pct_bw_ceiling"] = v.pop("llm_hip_naive_pct_bw_ceiling")
    o = llm_fill("hip-gpt2_nocache", "llm_nocache")
    if o:
        v["llm_nocache_prefill_ms"] = f"{o['prefill_ms']:.1f}"
        v.pop("llm_nocache_pct_bw_ceiling", None)
    llm_fill("pytorch", "llm_torch")

    h3 = extra.get("h3")
    if h3:
        v["h3_max_abs_err"] = v["real_gpt2_max_abs_err"] = f"{max(h3['prefill_logits']['max_abs_err'], h3['greedy_32']['max_abs_err']):.1e}"
        v["h3_greedy_match"] = f"yes ({h3['greedy_32']['argmax_agreement']})"
    nz = extra.get("noise")
    if nz:
        runs = nz["gemm_v4_fp32_4096_gflops_runs"]
        v["run_to_run_noise_pct"] = f"{100 * (max(runs) - min(runs)) / (sum(runs) / len(runs)):.0f}"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-values", action="store_true")
    a = ap.parse_args()
    vals = values(a.device, *load(a.results, a.device))
    if a.print_values:
        json.dump(vals, sys.stdout, indent=1, sort_keys=True)
        print()
        return
    files = [ROOT / t for t in TARGETS] + [p for gl in TARGET_GLOBS for p in sorted(ROOT.glob(gl))]
    left = {}
    for f in files:
        text = f.read_text(encoding="utf-8")
        new = re.sub(r"\{\{([a-zA-Z0-9_]+)\}\}", lambda m: vals.get(m.group(1), m.group(0)), text)
        for m in re.finditer(r"\{\{([a-zA-Z0-9_]+)\}\}", new):
            left[m.group(1)] = left.get(m.group(1), 0) + 1
        if new != text:
            print(f"filled {f.relative_to(ROOT)}")
            if not a.dry_run:
                f.write_text(new, encoding="utf-8")
    print("left unfilled:", ", ".join(f"{k}({n})" for k, n in sorted(left.items())) or "none")


if __name__ == "__main__":
    main()
