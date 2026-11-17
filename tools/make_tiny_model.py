#!/usr/bin/env python3
"""Generate the deterministic tiny GPT-2 test model and its float64 reference outputs.

Model: L=2, d=64, H=4, V=300, C=64. Files written to --out (default tests/data/tiny):
model.safetensors, config.json, prompt_ids.txt (P=11), ref_prefill_all.bin,
ref_greedy.bin + ref_greedy_tokens.txt (G=16), manifest.txt.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpt2_numpy as g  # noqa: E402

L, D, H, V, C = 2, 64, 4, 300, 64
P, G = 11, 16
MIN_MARGIN = 1e-2
SEED_START = 1234


def build_tensors(seed):
    rng = np.random.default_rng(seed)
    n = lambda *s: rng.standard_normal(s)
    t = {}
    t["wte.weight"] = 0.5 * n(V, D)
    t["wpe.weight"] = 0.5 * n(C, D)
    for i in range(L):
        p = "h.%d." % i
        t[p + "ln_1.weight"] = 1.0 + 0.1 * n(D)
        t[p + "ln_1.bias"] = 0.1 * n(D)
        t[p + "attn.c_attn.weight"] = 0.15 * n(D, 3 * D)
        t[p + "attn.c_attn.bias"] = 0.1 * n(3 * D)
        t[p + "attn.c_proj.weight"] = 0.15 * n(D, D)
        t[p + "attn.c_proj.bias"] = 0.1 * n(D)
        t[p + "ln_2.weight"] = 1.0 + 0.1 * n(D)
        t[p + "ln_2.bias"] = 0.1 * n(D)
        t[p + "mlp.c_fc.weight"] = 0.15 * n(D, 4 * D)
        t[p + "mlp.c_fc.bias"] = 0.1 * n(4 * D)
        t[p + "mlp.c_proj.weight"] = 0.15 * n(4 * D, D)
        t[p + "mlp.c_proj.bias"] = 0.1 * n(D)
    t["ln_f.weight"] = 1.0 + 0.1 * n(D)
    t["ln_f.bias"] = 0.1 * n(D)
    prompt = [int(x) for x in rng.integers(0, V, size=P)]
    # float32 weights are the ground truth; the reference runs on their float64 widening.
    return {k: np.ascontiguousarray(v.astype(np.float32)) for k, v in t.items()}, prompt


def write_all(out, seed):
    from safetensors.numpy import save_file

    tensors, prompt = build_tensors(seed)
    os.makedirs(out, exist_ok=True)
    st_path = os.path.join(out, "model.safetensors")
    save_file(tensors, st_path)
    cfg = {"model_type": "gpt2", "n_layer": L, "n_head": H, "n_embd": D, "n_ctx": C,
           "n_positions": C, "vocab_size": V}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(out, "prompt_ids.txt"), "w") as f:
        f.write(" ".join(map(str, prompt)) + "\n")

    model = g.load(out)  # reload from disk: reference sees exactly what C++ sees
    toks, rows, margins = g.greedy(model, prompt, G)
    g.write_lgt1(os.path.join(out, "ref_prefill_all.bin"), g.forward(model, prompt))
    g.write_lgt1(os.path.join(out, "ref_greedy.bin"), rows)
    with open(os.path.join(out, "ref_greedy_tokens.txt"), "w") as f:
        f.write(" ".join(map(str, toks)) + "\n")
    with open(os.path.join(out, "manifest.txt"), "w") as f:
        for name in sorted(tensors):
            flat = tensors[name].reshape(-1)
            f.write("%s %s %s\n" % (name, "%.9g" % float(flat[0]), "%.9g" % float(flat[-1])))
    return min(margins), margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "tests", "data", "tiny"))
    args = ap.parse_args()
    for seed in range(SEED_START, SEED_START + 500):
        m, margins = write_all(args.out, seed)
        if m >= MIN_MARGIN:
            print("seed %d: min greedy top1-top2 margin %.4f over %d steps" % (seed, m, G))
            return 0
    print("no seed satisfied the margin requirement", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
