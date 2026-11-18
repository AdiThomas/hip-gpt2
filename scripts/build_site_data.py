#!/usr/bin/env python3
"""Collect results/*.json into site/data/results.json for the write-up page (site/index.html).

  python3 scripts/build_site_data.py [--results results] [--out site/data/results.json]

Re-run after any new benchmark (e.g. the MI300X run); the page draws every number from this file.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = {  # datasheet constants, labelled as such on the page
    "NVIDIA GeForce RTX 4060": {"bw_gbs": 272, "fp32_tflops": 15.1, "stack": "HIP 6.4.3 on CUDA 12.6, WSL2/Docker"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "site" / "data" / "results.json"))
    a = ap.parse_args()
    devices = {}
    for p in sorted(Path(a.results).rglob("*.json")):
        try:
            o = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(o, dict) or "device" not in o:
            continue
        d = devices.setdefault(o["device"], {"spec": SPECS.get(o["device"]), "gemm": {}, "llm": {}, "h3": None, "noise": None})
        src = str(p.relative_to(ROOT)).replace("\\", "/")
        b = o.get("bench")
        if b == "gemm":
            d["gemm"][o["kernel"]] = {
                "dtype": o["dtype"], "git_sha": o.get("git_sha"), "date": o.get("timestamp", "")[:10], "source": src,
                "points": [{"n": r["M"], "gflops": round(r["gflops"], 1), "ms": round(r["ms_median"], 4)}
                           for r in o["results"] if r["M"] == r["N"] == r["K"] and r.get("verified", True)],
            }
        elif b in ("llm", "llm_variant"):
            key = o["engine"] if o["engine"] != "hip-gpt2" else (
                "hip_nocache" if o.get("kv_cache") is False else f"hip_{o.get('attn', 'fused')}")
            d["llm"][key] = {k: o.get(k) for k in ("decode_tok_per_s", "ttft_ms", "prefill_ms", "prompt_tokens", "gen_tokens")}
            d["llm"][key]["source"] = src
        elif b in ("h3", "noise"):
            d[b] = dict(o, source=src)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"devices": devices}, indent=1) + "\n")
    print(f"wrote {out} ({', '.join(devices) or 'no devices'})")


if __name__ == "__main__":
    main()
