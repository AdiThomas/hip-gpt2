#!/usr/bin/env python3
"""Performance regression checker (SPEC section 8.3).

Compares benchmark JSON files (SPEC 8.1 GEMM, 8.2 LLM) in --results against
baseline copies in --baseline (layout: perf/baselines/<device_slug>/*.json).

Rules
  * GEMM: for every (device, kernel, dtype, M, N, K) in both baseline and
    current: FAIL if gflops_cur < gflops_base * (1 - T/100).
  * LLM : FAIL if decode_tok_per_s drops more than T% or ttft_ms rises more
    than T% (per device and engine).
  * Any `verified: false` in the current results FAILS (even without baseline).
  * Baseline entries (same device) missing from the current results: warning,
    FAIL with --strict.
  * Baseline devices that are not in the current results are ignored.
  * T: --threshold, else env PERF_THRESHOLD_PCT (empty counts as unset), else 7.

Exit codes: 0 pass, 1 regression, 2 usage error.
Missing baseline (directory absent, or nothing for a current device): notice
and exit 0, unless --require-baseline (then exit 1).

--update-baseline copies the current JSON files into
<baseline>/<device_slug>/ (refuses when any result is unverified).
Standard library only.
"""
import argparse
import contextlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_THRESHOLD = 7.0


def slug(text):
    return re.sub(r"[^A-Za-z0-9]", "_", text)


def load_records(root):
    """Return [(path, obj)] for every GEMM/LLM result JSON under root."""
    out = []
    for p in sorted(Path(root).rglob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"warning: skipping unreadable {p}: {exc}", file=sys.stderr)
            continue
        if isinstance(obj, dict) and obj.get("bench") in ("gemm", "llm"):
            out.append((p, obj))
    return out


def num(x):
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else 0.0


def index_records(records):
    """gemm: (device,kernel,dtype,M,N,K) -> row ; llm: (device,engine) -> obj."""
    gemm, llm, unverified = {}, {}, []
    for path, obj in records:
        dev = str(obj.get("device", "unknown"))
        if obj["bench"] == "gemm":
            for r in obj.get("results", []):
                key = (dev, obj.get("kernel"), obj.get("dtype"),
                       r.get("M"), r.get("N"), r.get("K"))
                gemm[key] = r
                if r.get("verified") is False:
                    unverified.append(f"{dev} {obj.get('kernel')} "
                                      f"{r.get('M')}x{r.get('N')}x{r.get('K')} ({path.name})")
        else:
            llm[(dev, obj.get("engine", "unknown"))] = obj
    return gemm, llm, unverified


def fmt_key(key):
    if len(key) == 6:
        dev, kernel, dtype, m, n, k = key
        return f"{dev} | {kernel} {dtype} {m}x{n}x{k}"
    return f"{key[0]} | llm {key[1]}"


def resolve_threshold(arg):
    if arg is not None:
        raw = arg
    else:
        raw = os.environ.get("PERF_THRESHOLD_PCT", "").strip() or None
    if raw is None:
        return DEFAULT_THRESHOLD
    try:
        t = float(raw)
    except ValueError:
        return None
    return t if math.isfinite(t) and t >= 0 else None


def update_baseline(results_dir, baseline_dir):
    recs = load_records(results_dir)
    if not recs:
        print(f"error: no result JSON files in {results_dir}", file=sys.stderr)
        return 2
    _, _, unverified = index_records(recs)
    if unverified:
        print("refusing to update baseline, unverified results:", file=sys.stderr)
        for u in unverified:
            print(f"  {u}", file=sys.stderr)
        return 1
    for path, obj in recs:
        dest = Path(baseline_dir) / slug(str(obj.get("device", "unknown")))
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / path.name)
        print(f"baseline updated: {dest / path.name}")
    return 0


def compare(base_dir, results_dir, threshold, strict, require_baseline):
    if not Path(results_dir).is_dir():
        print(f"error: results directory not found: {results_dir}", file=sys.stderr)
        return 2
    cur_recs = load_records(results_dir)
    if not cur_recs:
        print(f"error: no result JSON files in {results_dir}", file=sys.stderr)
        return 2
    cur_gemm, cur_llm, unverified = index_records(cur_recs)
    base_recs = load_records(base_dir) if Path(base_dir).is_dir() else []
    base_gemm, base_llm, _ = index_records(base_recs)

    failures, warnings, rows = [], [], []
    for u in unverified:
        failures.append(f"unverified result: {u}")

    cur_devices = {k[0] for k in list(cur_gemm) + list(cur_llm)}
    base_devices = {k[0] for k in list(base_gemm) + list(base_llm)}
    no_base = sorted(cur_devices - base_devices)
    for dev in no_base:
        print(f"notice: no baseline for device '{dev}' in {base_dir}")
    if no_base and require_baseline:
        failures.append("baseline required but missing for: " + ", ".join(no_base))

    lo, hi = 1 - threshold / 100.0, 1 + threshold / 100.0

    for key, brow in sorted(base_gemm.items(), key=lambda kv: str(kv[0])):
        if key[0] not in cur_devices:
            continue
        crow = cur_gemm.get(key)
        if crow is None:
            warnings.append(f"missing in current results: {fmt_key(key)}")
            continue
        b, c = num(brow.get("gflops")), num(crow.get("gflops"))
        if b <= 0:
            continue
        delta = (c / b - 1) * 100
        bad = c < b * lo
        rows.append((fmt_key(key) + " gflops", b, c, delta, "FAIL" if bad else "ok"))
        if bad:
            failures.append(f"{fmt_key(key)}: gflops {b:.1f} -> {c:.1f} ({delta:+.1f}%)")

    for key, bobj in sorted(base_llm.items()):
        if key[0] not in cur_devices:
            continue
        cobj = cur_llm.get(key)
        if cobj is None:
            warnings.append(f"missing in current results: {fmt_key(key)}")
            continue
        b, c = num(bobj.get("decode_tok_per_s")), num(cobj.get("decode_tok_per_s"))
        if b > 0:
            delta = (c / b - 1) * 100
            bad = c < b * lo
            rows.append((fmt_key(key) + " decode_tok_per_s", b, c, delta, "FAIL" if bad else "ok"))
            if bad:
                failures.append(f"{fmt_key(key)}: decode_tok_per_s {b:.1f} -> {c:.1f} ({delta:+.1f}%)")
        b, c = num(bobj.get("ttft_ms")), num(cobj.get("ttft_ms"))
        if b > 0:
            delta = (c / b - 1) * 100
            bad = c > b * hi
            rows.append((fmt_key(key) + " ttft_ms", b, c, delta, "FAIL" if bad else "ok"))
            if bad:
                failures.append(f"{fmt_key(key)}: ttft_ms {b:.2f} -> {c:.2f} ({delta:+.1f}%)")

    if not base_recs:
        print(f"notice: no baseline found in {base_dir}")
    if rows:
        width = max(len(r[0]) for r in rows)
        print(f"{'metric'.ljust(width)}  {'base':>12} {'current':>12} {'delta%':>8}  status")
        for name, b, c, d, st in rows:
            print(f"{name.ljust(width)}  {b:12.2f} {c:12.2f} {d:+8.1f}  {st}")
    print(f"threshold: {threshold:g}%  compared: {len(rows)}")
    for w in warnings:
        print(f"warning: {w}")
        if strict:
            failures.append(w + " (--strict)")
    if failures:
        print("REGRESSION CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("regression check passed")
    return 0


# ----------------------------------------------------------------- self-test
def _gemm(d, device, kernel, gflops, verified=True, dtype="fp32", name=None):
    Path(d).mkdir(parents=True, exist_ok=True)
    obj = {"schema_version": 1, "bench": "gemm", "kernel": kernel, "dtype": dtype,
           "platform": "cpu", "device": device, "git_sha": "abc1234",
           "timestamp": "2026-11-16T20:00:00Z",
           "results": [{"M": 1024, "N": 1024, "K": 1024, "ms_median": 1.0, "ms_min": 0.9,
                        "gflops": gflops, "iters": 10, "max_err": 1e-7,
                        "verified": verified, "full_check": True}]}
    (Path(d) / (name or f"gemm_{kernel}_{slug(device)}.json")).write_text(json.dumps(obj))


def _llm(d, device, decode, ttft, engine="hip-gpt2"):
    Path(d).mkdir(parents=True, exist_ok=True)
    obj = {"schema_version": 1, "bench": "llm", "engine": engine, "model": "gpt2",
           "platform": "cpu", "device": device, "git_sha": "abc1234",
           "timestamp": "2026-11-16T20:00:00Z", "attn": "fused", "gemm_kernel": "v4",
           "sampling": "greedy", "warmup_runs": 1, "prompt_tokens": 32, "gen_tokens": 128,
           "load_ms": 400.0, "prefill_ms": 9.0, "ttft_ms": ttft, "decode_tok_per_s": decode}
    (Path(d) / f"llm_{slug(device)}.json").write_text(json.dumps(obj))


def self_test():
    dev = "Test GPU"
    fails = []

    def case(name, expect, build, argv=(), env=None):
        with tempfile.TemporaryDirectory() as tmp:
            base, cur = Path(tmp) / "base" / slug(dev), Path(tmp) / "cur"
            build(base, cur)
            old = os.environ.pop("PERF_THRESHOLD_PCT", None)
            if env is not None:
                os.environ["PERF_THRESHOLD_PCT"] = env
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    code = main(["--baseline", str(Path(tmp) / "base"),
                                 "--results", str(cur), *argv])
            finally:
                os.environ.pop("PERF_THRESHOLD_PCT", None)
                if old is not None:
                    os.environ["PERF_THRESHOLD_PCT"] = old
        ok = code == expect
        print(f"{'PASS' if ok else 'FAIL'}  {name} (exit {code}, expected {expect})")
        if not ok:
            fails.append(name)
            print(buf.getvalue())

    def gemm_pair(base_g, cur_g, **kw):
        def build(b, c):
            _gemm(b, dev, "v3", base_g)
            _gemm(c, dev, "v3", cur_g, **kw)
        return build

    def llm_pair(bd, bt, cd, ct):
        def build(b, c):
            _llm(b, dev, bd, bt)
            _llm(c, dev, cd, ct)
        return build

    case("equal results pass", 0, gemm_pair(1000, 1000))
    case("10% gflops drop fails", 1, gemm_pair(1000, 900))
    case("3% gflops drop passes", 0, gemm_pair(1000, 970))
    case("speedup passes", 0, gemm_pair(1000, 1500))
    case("verified false fails", 1, gemm_pair(1000, 1000, verified=False))
    case("llm decode -10% fails", 1, llm_pair(400, 10, 360, 10))
    case("llm decode -3% passes", 0, llm_pair(400, 10, 388, 10))
    case("llm ttft +10% fails", 1, llm_pair(400, 10.0, 400, 11.0))
    case("llm ttft +3% passes", 0, llm_pair(400, 10.0, 400, 10.3))
    case("llm faster passes", 0, llm_pair(400, 10.0, 500, 8.0))

    def no_base(b, c):
        _gemm(c, dev, "v3", 1000)
    case("missing baseline dir exits 0", 0, no_base)
    case("missing baseline with --require-baseline fails", 1, no_base, ["--require-baseline"])

    def no_base_unverified(b, c):
        _gemm(c, dev, "v3", 1000, verified=False)
    case("missing baseline but unverified still fails", 1, no_base_unverified)

    def missing_entry(b, c):
        _gemm(b, dev, "v3", 1000)
        _gemm(b, dev, "v4", 2000)
        _gemm(c, dev, "v3", 1000)
    case("entry missing now warns, exit 0", 0, missing_entry)
    case("entry missing now fails with --strict", 1, missing_entry, ["--strict"])

    case("--threshold 15 lets 10% drop pass", 0, gemm_pair(1000, 900), ["--threshold", "15"])
    case("--threshold 2 fails 3% drop", 1, gemm_pair(1000, 970), ["--threshold", "2"])
    case("env threshold 15 lets 10% drop pass", 0, gemm_pair(1000, 900), env="15")
    case("env threshold 2 fails 3% drop", 1, gemm_pair(1000, 970), env="2")
    case("--threshold beats env", 1, gemm_pair(1000, 900), ["--threshold", "5"], env="50")
    case("empty env falls back to 7", 1, gemm_pair(1000, 900), env="")
    case("bad threshold is a usage error", 2, gemm_pair(1000, 1000), ["--threshold", "abc"])
    case("missing results dir is a usage error", 2, lambda b, c: None)

    # --update-baseline round trip
    with tempfile.TemporaryDirectory() as tmp:
        cur, base = Path(tmp) / "cur", Path(tmp) / "base"
        _gemm(cur, dev, "v3", 1000)
        code1 = main(["--results", str(cur), "--baseline", str(base), "--update-baseline"])
        copied = (base / slug(dev) / f"gemm_v3_{slug(dev)}.json").exists()
        _gemm(cur, dev, "v3", 1000, verified=False)
        code2 = main(["--results", str(cur), "--baseline", str(base), "--update-baseline"])
    ok = code1 == 0 and copied and code2 == 1
    print(f"{'PASS' if ok else 'FAIL'}  update-baseline copies, refuses unverified")
    if not ok:
        fails.append("update-baseline")

    if fails:
        print(f"SELF-TEST FAILED: {len(fails)} case(s): {fails}")
        return 1
    print("SELF-TEST OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default="perf/baselines")
    ap.add_argument("--results", default="results")
    ap.add_argument("--threshold", help="percent; default env PERF_THRESHOLD_PCT, else 7")
    ap.add_argument("--strict", action="store_true", help="missing entries fail")
    ap.add_argument("--require-baseline", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    if args.self_test:
        return self_test()
    threshold = resolve_threshold(args.threshold)
    if threshold is None:
        print("error: invalid threshold (need a non-negative number)", file=sys.stderr)
        return 2
    if args.update_baseline:
        return update_baseline(args.results, args.baseline)
    return compare(args.baseline, args.results, threshold, args.strict, args.require_baseline)


if __name__ == "__main__":
    sys.exit(main())
