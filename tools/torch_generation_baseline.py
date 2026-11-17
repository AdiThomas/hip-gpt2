#!/usr/bin/env python3
"""PyTorch greedy-generation timing baseline (owner-run). Writes SPEC 8.2 JSON with engine "pytorch".

Timing definitions follow SPEC 6.4: ttft_ms = wall time from prefill start until token 1 is on the
host; prefill_ms = the device part (forward pass on the prompt); decode_tok_per_s =
(gen_tokens - 1) / wall time over all later steps including the host copy.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party"))
ASSETS = os.path.join(HERE, "..", "assets", "gpt2")


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="model directory (config.json + model.safetensors)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt")
    g.add_argument("--prompt-file")
    g.add_argument("--tokens")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", required=True, help="output directory for llm_pytorch_<device_slug>.json")
    args = ap.parse_args()

    import torch  # lazy: torch is not installed in the build environment
    from transformers import GPT2LMHeadModel

    import encoder as oai

    if args.tokens:
        ids = [int(t) for t in args.tokens.split(",")]
    else:
        text = args.prompt if args.prompt is not None else open(args.prompt_file, encoding="utf-8").read()
        vdir = args.model if os.path.exists(os.path.join(args.model, "encoder.json")) else ASSETS
        with open(os.path.join(vdir, "encoder.json")) as f:
            enc = json.load(f)
        with open(os.path.join(vdir, "vocab.bpe"), encoding="utf-8") as f:
            merges = [tuple(m.split()) for m in f.read().split("\n")[1:-1]]
        ids = oai.Encoder(encoder=enc, bpe_merges=merges).encode(text)
    if not ids:
        ids = [50256]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    t0 = time.perf_counter()
    model = GPT2LMHeadModel.from_pretrained(args.model).to(dev).eval().float()
    sync()
    load_ms = (time.perf_counter() - t0) * 1e3
    n_ctx = model.config.n_positions
    gen = max(1, min(args.max_new_tokens, n_ctx - len(ids) + 1))

    def run():
        with torch.no_grad():
            x = torch.tensor([ids], device=dev)
            sync()
            t_start = time.perf_counter()
            out = model(x, use_cache=True)
            sync()
            prefill_ms = (time.perf_counter() - t_start) * 1e3
            tok = int(out.logits[0, -1].argmax().item())
            ttft_ms = (time.perf_counter() - t_start) * 1e3
            past = out.past_key_values
            toks = [tok]
            t_dec = time.perf_counter()
            for _ in range(gen - 1):
                out = model(torch.tensor([[tok]], device=dev), past_key_values=past, use_cache=True)
                past = out.past_key_values
                tok = int(out.logits[0, -1].argmax().item())
                toks.append(tok)
            dec_s = time.perf_counter() - t_dec
            tps = (gen - 1) / dec_s if gen >= 2 and dec_s > 0 else 0.0
            return prefill_ms, ttft_ms, tps, toks

    for _ in range(args.warmup):
        run()
    prefill_ms, ttft_ms, tps, toks = run()

    if dev.type == "cuda":
        device = torch.cuda.get_device_name(0)
        platform = "amd" if getattr(torch.version, "hip", None) else "nvidia"
    else:
        device, platform = "cpu", "cpu"
    slug = re.sub(r"[^A-Za-z0-9]", "_", device)
    res = {"schema_version": 1, "bench": "llm", "engine": "pytorch",
           "model": os.path.basename(os.path.normpath(args.model)), "platform": platform, "device": device,
           "git_sha": git_sha(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "attn": "n/a", "gemm_kernel": "n/a", "sampling": "greedy", "warmup_runs": args.warmup,
           "prompt_tokens": len(ids), "gen_tokens": len(toks), "load_ms": round(load_ms, 3),
           "prefill_ms": round(prefill_ms, 4), "ttft_ms": round(ttft_ms, 4),
           "decode_tok_per_s": round(tps, 3)}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "llm_pytorch_%s.json" % slug)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps(res))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
