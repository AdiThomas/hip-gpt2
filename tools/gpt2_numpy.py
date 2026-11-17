#!/usr/bin/env python3
"""float64 numpy reference implementation of GPT-2 (SPEC 6.1) plus LGT1 helpers.

HF Conv1D weights are stored [in, out] and applied as x @ W. LayerNorm eps = 1e-5 with
biased variance, GELU is the tanh approximation, attention is causal with scale
1/sqrt(head_dim), and the LM head is tied: logits = xf @ wte.T.
"""
import json
import os
import struct
import sys

import numpy as np

LGT1_MAGIC = b"LGT1"


# ---------------------------------------------------------------- LGT1 helpers
def write_lgt1(path, arr):
    """Write a [rows, vocab] array as an LGT1 file (float32 little endian)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    if a.ndim != 2:
        raise ValueError("LGT1 array must be 2-D, got shape %r" % (a.shape,))
    with open(path, "wb") as f:
        f.write(LGT1_MAGIC)
        f.write(struct.pack("<ii", a.shape[0], a.shape[1]))
        f.write(a.tobytes())


def read_lgt1(path):
    """Read an LGT1 file, returning a float32 array [rows, vocab]."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[:4] != LGT1_MAGIC:
        raise ValueError("%s: not an LGT1 file" % path)
    rows, vocab = struct.unpack("<ii", data[4:12])
    if rows < 0 or vocab < 0:
        raise ValueError("%s: negative dimensions" % path)
    expect = 12 + rows * vocab * 4
    if len(data) != expect:
        raise ValueError("%s: size %d != expected %d for %dx%d" % (path, len(data), expect, rows, vocab))
    return np.frombuffer(data, dtype="<f4", offset=12).reshape(rows, vocab).copy()


def read_ids(path):
    """Read a whitespace-separated list of decimal ids."""
    with open(path) as f:
        return [int(t) for t in f.read().split()]


# ---------------------------------------------------------------- model
class Model:
    def __init__(self, cfg, tensors):
        self.cfg = cfg
        self.t = tensors


def _get(tensors, name):
    for key in (name, "transformer." + name):
        if key in tensors:
            return np.asarray(tensors[key], dtype=np.float64)
    raise KeyError("missing tensor " + name)


def load(model_dir):
    from safetensors.numpy import load_file

    with open(os.path.join(model_dir, "config.json")) as f:
        c = json.load(f)
    cfg = {
        "n_layer": int(c["n_layer"]),
        "n_head": int(c["n_head"]),
        "n_embd": int(c["n_embd"]),
        "vocab_size": int(c["vocab_size"]),
        "n_ctx": int(c.get("n_ctx", c.get("n_positions", 0))),
    }
    raw = load_file(os.path.join(model_dir, "model.safetensors"))
    d, L = cfg["n_embd"], cfg["n_layer"]
    t = {"wte": _get(raw, "wte.weight"), "wpe": _get(raw, "wpe.weight"),
         "lnf_w": _get(raw, "ln_f.weight"), "lnf_b": _get(raw, "ln_f.bias"), "h": []}
    for i in range(L):
        p = "h.%d." % i
        t["h"].append({k: _get(raw, p + v) for k, v in [
            ("ln1_w", "ln_1.weight"), ("ln1_b", "ln_1.bias"),
            ("attn_w", "attn.c_attn.weight"), ("attn_b", "attn.c_attn.bias"),
            ("proj_w", "attn.c_proj.weight"), ("proj_b", "attn.c_proj.bias"),
            ("ln2_w", "ln_2.weight"), ("ln2_b", "ln_2.bias"),
            ("fc_w", "mlp.c_fc.weight"), ("fc_b", "mlp.c_fc.bias"),
            ("mproj_w", "mlp.c_proj.weight"), ("mproj_b", "mlp.c_proj.bias")]})
    assert t["wte"].shape == (cfg["vocab_size"], d)
    assert t["wpe"].shape == (cfg["n_ctx"], d)
    return Model(cfg, t)


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)  # biased
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(z):
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))


def forward(model, ids):
    """Full forward pass over ids (list of ints). Returns float64 logits [T, V]."""
    cfg, t = model.cfg, model.t
    d, H = cfg["n_embd"], cfg["n_head"]
    hd = d // H
    T = len(ids)
    if T > cfg["n_ctx"]:
        raise ValueError("sequence longer than n_ctx")
    x = t["wte"][np.asarray(ids)] + t["wpe"][np.arange(T)]
    mask = np.tril(np.ones((T, T), dtype=bool))
    for lw in t["h"]:
        a = layer_norm(x, lw["ln1_w"], lw["ln1_b"])
        qkv = a @ lw["attn_w"] + lw["attn_b"]
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        y = np.zeros((T, d))
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            s = (q[:, sl] @ k[:, sl].T) / np.sqrt(hd)
            s = np.where(mask, s, -np.inf)
            s = s - s.max(axis=-1, keepdims=True)
            w = np.exp(s)
            w /= w.sum(axis=-1, keepdims=True)
            y[:, sl] = w @ v[:, sl]
        x = x + y @ lw["proj_w"] + lw["proj_b"]
        m = layer_norm(x, lw["ln2_w"], lw["ln2_b"])
        f = gelu(m @ lw["fc_w"] + lw["fc_b"])
        x = x + f @ lw["mproj_w"] + lw["mproj_b"]
    xf = layer_norm(x, t["lnf_w"], t["lnf_b"])
    return xf @ t["wte"].T


def greedy(model, ids, n):
    """Greedy generation of n tokens. Returns (tokens, logits [n, V], margins [n]).

    Row g of logits is the logit vector used to sample generated token g (row 0 comes
    from the prompt's last position), matching `gpt2 --dump-mode last`.
    """
    seq = list(ids)
    toks, rows, margins = [], [], []
    for _ in range(n):
        lg = forward(model, seq)[-1]
        order = np.argsort(-lg, kind="stable")
        nxt = int(order[0])
        margins.append(float(lg[order[0]] - lg[order[1]]))
        toks.append(nxt)
        rows.append(lg)
        seq.append(nxt)
    return toks, np.stack(rows), margins


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: gpt2_numpy.py MODEL_DIR ID[,ID...] [N]", file=sys.stderr)
        sys.exit(2)
    m = load(sys.argv[1])
    ids = [int(s) for s in sys.argv[2].split(",")]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    print(" ".join(str(i) for i in greedy(m, ids, n)[0]))
