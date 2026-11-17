#!/usr/bin/env python3
"""Append benchmark results to the performance history (SPEC section 8.3).

History file (published on branch gh-pages as data/history.json) is a JSON
array, newest last, of
  {"git_sha", "timestamp", "device",
   "gemm": [{"kernel", "dtype", "M", "N", "K", "gflops"}],
   "llm": {"ttft_ms", "decode_tok_per_s"} | null}
One entry per (git_sha, device); re-running for the same pair replaces the
old entry. Only verified GEMM rows and the hip-gpt2 engine's LLM result are
recorded. Standard library only.
"""
import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def build_entries(results_dir, sha_override=None):
    per_dev = defaultdict(lambda: {"gemm": [], "llm": None, "shas": [], "stamps": []})
    for p in sorted(Path(results_dir).rglob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"warning: skipping {p}: {exc}", file=sys.stderr)
            continue
        if not isinstance(obj, dict) or obj.get("bench") not in ("gemm", "llm"):
            continue
        dev = obj.get("device")
        if not dev:
            continue
        d = per_dev[dev]
        d["shas"].append(obj.get("git_sha", "unknown"))
        d["stamps"].append(obj.get("timestamp", ""))
        if obj["bench"] == "gemm":
            for r in obj.get("results", []):
                if r.get("verified") is True and finite(r.get("gflops")) and r["gflops"] > 0:
                    d["gemm"].append({"kernel": obj.get("kernel"), "dtype": obj.get("dtype"),
                                      "M": r["M"], "N": r["N"], "K": r["K"],
                                      "gflops": r["gflops"]})
        elif obj.get("engine") == "hip-gpt2":
            if finite(obj.get("ttft_ms")) and finite(obj.get("decode_tok_per_s")):
                d["llm"] = {"ttft_ms": obj["ttft_ms"],
                            "decode_tok_per_s": obj["decode_tok_per_s"]}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries = []
    for dev, d in sorted(per_dev.items()):
        if not d["gemm"] and d["llm"] is None:
            continue
        sha = sha_override or Counter(d["shas"]).most_common(1)[0][0]
        stamps = [s for s in d["stamps"] if s]
        entries.append({"git_sha": sha, "timestamp": max(stamps) if stamps else now,
                        "device": dev, "gemm": d["gemm"], "llm": d["llm"]})
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--history", default="data/history.json")
    ap.add_argument("--git-sha", help="override git_sha recorded in the results")
    args = ap.parse_args(argv)
    if not Path(args.results).is_dir():
        print(f"error: results directory not found: {args.results}", file=sys.stderr)
        return 2
    entries = build_entries(args.results, args.git_sha)
    if not entries:
        print("error: no usable results found", file=sys.stderr)
        return 1
    hist_path = Path(args.history)
    history = []
    if hist_path.exists():
        try:
            history = json.loads(hist_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                raise ValueError("not a JSON array")
        except (OSError, ValueError) as exc:
            print(f"error: cannot read existing history {hist_path}: {exc}", file=sys.stderr)
            return 1
    new_keys = {(e["git_sha"], e["device"]) for e in entries}
    history = [h for h in history if (h.get("git_sha"), h.get("device")) not in new_keys]
    history.extend(entries)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=hist_path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=1)
        f.write("\n")
    os.replace(tmp, hist_path)
    for e in entries:
        print(f"history: {e['git_sha']} {e['device']}: {len(e['gemm'])} gemm rows, "
              f"llm={'yes' if e['llm'] else 'no'} -> {hist_path} ({len(history)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
