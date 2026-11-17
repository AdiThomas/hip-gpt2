#!/usr/bin/env python3
"""Compare two LGT1 logits files: prints max abs error and argmax agreement.

Exit 0 if max abs error <= --tol (and shapes match), 1 otherwise, 2 on usage/IO errors.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt2_numpy import read_lgt1  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    try:
        a, b = read_lgt1(args.a), read_lgt1(args.b)
    except (OSError, ValueError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    if a.shape != b.shape:
        print("shape mismatch: %s vs %s" % (a.shape, b.shape))
        return 1
    err = np.abs(a.astype(np.float64) - b.astype(np.float64))
    mx = float(err.max()) if err.size else 0.0
    agree = int((a.argmax(axis=1) == b.argmax(axis=1)).sum()) if a.size else 0
    print("rows=%d vocab=%d max_abs_err=%.3e argmax_agreement=%d/%d tol=%.1e"
          % (a.shape[0], a.shape[1], mx, agree, a.shape[0], args.tol))
    if not np.isfinite(mx) or mx > args.tol:
        print("FAIL")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
