#!/usr/bin/env python3
"""Plot GEMM benchmark results (SPEC 8.1 JSON) to PNG, headless (matplotlib Agg).

Figures per device and dtype:
  gemm_gflops_<dtype>[_<device>].png   GFLOPS vs matrix size, one line per kernel
  gemm_ladder_<dtype>[_<device>].png   bars of GFLOPS at one size, speedup over
                                       the first stage (v0 / v0_fp16) labelled

Chart rules (dataviz skill): categorical colors in a fixed order that follows
the kernel, never its rank; text stays in ink colors; thin 2px-ish lines with
surface-ringed markers; hairline grid; legend always, direct end labels when
<= 4 series; one y axis (log scale only via auto/--log-y, labelled).
Only verified rows are plotted. Standard library + matplotlib.
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                          "#008300", "#6250d6", "#e34948"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
                         "#0ca30c", "#9085e9", "#e66767"]),
}

# Canonical kernel order per dtype. Color slot = position here, so a kernel keeps
# its color even when other kernels are missing from a given run.
ORDER = {
    "fp32": ["v0", "v1", "v2", "v3", "v4", "hipblas"],
    "fp16": ["v0_fp16", "v4_fp16", "v5", "hipblas_fp16"],
}
VENDOR = {"hipblas", "hipblas_fp16"}
FONT = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]


def slug(text):
    return re.sub(r"[^A-Za-z0-9]", "_", text)


def load(results_dir):
    """data[device][dtype][kernel] = {(M,N,K): gflops}; also meta[device] = platform."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    meta = {}
    for p in sorted(Path(results_dir).rglob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(obj, dict) or obj.get("bench") != "gemm":
            continue
        dev = obj.get("device", "unknown")
        meta[dev] = (obj.get("platform", "?"), obj.get("git_sha", "?"))
        for r in obj.get("results", []):
            g = r.get("gflops")
            if r.get("verified") is True and isinstance(g, (int, float)) and math.isfinite(g) and g > 0:
                data[dev][obj.get("dtype", "fp32")][obj["kernel"]][(r["M"], r["N"], r["K"])] = g
    return data, meta


def shape_label(s):
    return str(s[0]) if s[0] == s[1] == s[2] else f"{s[0]}x{s[1]}x{s[2]}"


def kernel_order(dtype, kernels):
    canon = ORDER.get(dtype, [])
    return [k for k in canon if k in kernels] + sorted(k for k in kernels if k not in canon)


def color_of(dtype, kernel, theme):
    canon = ORDER.get(dtype, [])
    idx = canon.index(kernel) if kernel in canon else len(canon) + sum(map(ord, kernel)) % 2
    return THEMES[theme]["series"][idx % len(THEMES[theme]["series"])]


def style_axes(fig, ax, t):
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(axis="both", colors=t["muted"], labelsize=9, length=0)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])


def header(fig, title, subtitle, t):
    fig.text(0.07, 0.95, title, ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.07, 0.905, subtitle, ha="left", va="top", fontsize=9, color=t["ink2"])


def line_chart(dtype, series, device, platform, sha, out_path, theme, log_y):
    t = THEMES[theme]
    shapes = sorted({s for pts in series.values() for s in pts}, key=lambda s: (s[0] * s[1] * s[2], s))
    xpos = {s: i for i, s in enumerate(shapes)}
    allv = [v for pts in series.values() for v in pts.values()]
    if log_y is None:
        log_y = max(allv) / min(allv) > 50
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    fig.subplots_adjust(left=0.10, right=0.86 if len(series) <= 4 else 0.95, top=0.76, bottom=0.14)
    style_axes(fig, ax, t)
    handles = []
    for k, pts in series.items():
        xs = [xpos[s] for s in sorted(pts, key=lambda s: xpos[s])]
        ys = [pts[s] for s in sorted(pts, key=lambda s: xpos[s])]
        c = color_of(dtype, k, theme)
        (h,) = ax.plot(xs, ys, color=c, linewidth=1.4, marker="o", markersize=4.6,
                       markeredgecolor=t["surface"], markeredgewidth=1.0, label=k, clip_on=False)
        handles.append(h)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels([shape_label(s) for s in shapes])
    ax.set_xlim(-0.25, len(shapes) - 1 + 0.25)
    ax.set_xlabel("matrix size (M = N = K)" if all(s[0] == s[1] == s[2] for s in shapes) else "matrix shape (MxNxK)")
    ax.set_ylabel("GFLOPS" + (" (log scale)" if log_y else ""))
    header(fig, f"GEMM {dtype.upper()} throughput by kernel",
           f"{device} ({platform}), commit {sha}; higher is better", t)
    # Legend always present for >= 2 series.
    if len(series) >= 2:
        leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.865),
                         ncol=len(handles), frameon=False, fontsize=9, handlelength=1.6,
                         columnspacing=1.4)
        for txt in leg.get_texts():
            txt.set_color(t["ink2"])
    # Direct end labels (<= 4 series), de-overlapped in pixel space; text in ink color.
    if len(series) <= 4:
        fig.canvas.draw()
        ends = []
        for k, pts in series.items():
            s = max(pts, key=lambda s: xpos[s])
            ends.append((k, xpos[s], pts[s]))
        px = [(k, x, y, ax.transData.transform((x, y))[1]) for k, x, y in ends]
        px.sort(key=lambda r: r[3])
        gap = 13.0 * fig.dpi / 100.0
        adj = []
        for i, (k, x, y, ypx) in enumerate(px):
            adj.append(max(ypx, adj[i - 1] + gap) if i else ypx)
        for (k, x, y, ypx), a in zip(px, adj):
            ax.annotate(k, xy=(x, y), xytext=(8, (a - ypx) * 72.0 / fig.dpi), textcoords="offset points",
                        va="center", ha="left", fontsize=9, color=t["ink"], annotation_clip=False)
    fig.savefig(out_path, facecolor=t["surface"])
    plt.close(fig)


def ladder_chart(dtype, series, device, platform, sha, size, out_path, theme):
    t = THEMES[theme]
    kernels = [k for k in kernel_order(dtype, series) if size in series[k]]
    stages = [k for k in kernels if k not in VENDOR]
    base_name = stages[0] if stages else kernels[0]
    base = series[base_name][size]
    vals = [series[k][size] for k in kernels]
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.80, bottom=0.14)
    style_axes(fig, ax, t)
    xs = range(len(kernels))
    colors = [t["muted"] if k in VENDOR else t["series"][0] for k in kernels]
    ax.bar(xs, vals, width=0.5, color=colors, edgecolor=t["surface"], linewidth=2.0, zorder=3)
    top = max(vals)
    for x, k, v in zip(xs, kernels, vals):
        speed = "1.0x (baseline)" if k == base_name else f"{v / base:.1f}x"
        ax.text(x, v + top * 0.015, f"{v:,.0f}\n{speed}", ha="center", va="bottom",
                fontsize=9, color=t["ink"], linespacing=1.3)
    ax.set_ylim(0, top * 1.18)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{k}\n(vendor ref)" if k in VENDOR else k for k in kernels])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("GFLOPS")
    header(fig, f"GEMM {dtype.upper()} ladder at {shape_label(size)}",
           f"{device} ({platform}), commit {sha}; speedup is relative to {base_name}", t)
    fig.savefig(out_path, facecolor=t["surface"])
    plt.close(fig)


def pick_ladder_size(series, dtype, wanted):
    if wanted is not None:
        return (wanted, wanted, wanted)
    kernels = kernel_order(dtype, series)
    base = next((k for k in kernels if k not in VENDOR), kernels[0])
    cands = sorted(series[base], key=lambda s: s[0] * s[1] * s[2], reverse=True)
    for s in cands:
        if sum(1 for k in kernels if s in series[k]) >= 2:
            return s
    return cands[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True, help="directory with GEMM result JSON (SPEC 8.1)")
    ap.add_argument("--out", required=True, help="output directory for PNGs")
    ap.add_argument("--size", type=int, help="square size for the ladder chart (default: largest common)")
    ap.add_argument("--theme", choices=sorted(THEMES), default="light")
    scale = ap.add_mutually_exclusive_group()
    scale.add_argument("--log-y", dest="log_y", action="store_true", default=None)
    scale.add_argument("--linear-y", dest="log_y", action="store_false")
    args = ap.parse_args(argv)
    if not Path(args.results).is_dir():
        print(f"error: results directory not found: {args.results}", file=sys.stderr)
        return 2
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT
    data, meta = load(args.results)
    if not data:
        print("error: no verified GEMM results found", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for device in sorted(data):
        suffix = f"_{slug(device)}" if len(data) > 1 else ""
        platform, sha = meta[device]
        for dtype in sorted(data[device]):
            series = {k: data[device][dtype][k] for k in kernel_order(dtype, data[device][dtype])}
            p = out / f"gemm_gflops_{dtype}{suffix}.png"
            line_chart(dtype, series, device, platform, sha, p, args.theme, args.log_y)
            written.append(p)
            size = pick_ladder_size(series, dtype, args.size)
            if any(size in pts for pts in series.values()):
                p = out / f"gemm_ladder_{dtype}{suffix}.png"
                ladder_chart(dtype, series, device, platform, sha, size, p, args.theme)
                written.append(p)
            else:
                print(f"warning: no kernel has size {shape_label(size)} for {dtype}; ladder skipped",
                      file=sys.stderr)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
