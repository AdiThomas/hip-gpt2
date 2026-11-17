#!/usr/bin/env python3
"""Export real GPT-2 weights + reference logits (owner-run; needs torch + transformers + network).

Writes a model directory (SPEC 4.6: model.safetensors, config.json, encoder.json, vocab.bpe) and
LGT1 reference logits:
  --ref FILE         logits at every prompt position (matches `gpt2 --dump-mode all`)
  --greedy-ref FILE  greedy generation logits, one row per generated token (matches `--dump-mode last`)
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "third_party"))
ASSETS = os.path.join(HERE, "..", "assets", "gpt2")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="model directory to create")
    ap.add_argument("--model", default="gpt2", help="HF model name (default gpt2)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt")
    g.add_argument("--tokens", help="comma-separated ids")
    ap.add_argument("--ref", help="write all-position logits (LGT1) here")
    ap.add_argument("--greedy-ref", help="write greedy-generation logits (LGT1) here")
    ap.add_argument("--greedy-tokens-out", help="write greedy token ids here")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    import torch  # lazy: not installed in the build environment
    from safetensors.torch import save_file
    from transformers import GPT2LMHeadModel

    import encoder as oai
    from gpt2_numpy import write_lgt1

    os.makedirs(args.out, exist_ok=True)
    model = GPT2LMHeadModel.from_pretrained(args.model).eval().float()
    sd = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()
          if not k.endswith("lm_head.weight") and not k.endswith("attn.bias") and not k.endswith("attn.masked_bias")}
    save_file(sd, os.path.join(args.out, "model.safetensors"))
    c = model.config
    cfg = {"model_type": "gpt2", "n_layer": c.n_layer, "n_head": c.n_head, "n_embd": c.n_embd,
           "n_ctx": c.n_ctx, "n_positions": c.n_positions, "vocab_size": c.vocab_size}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")
    for fn in ("encoder.json", "vocab.bpe"):
        shutil.copyfile(os.path.join(ASSETS, fn), os.path.join(args.out, fn))

    if args.tokens:
        ids = [int(t) for t in args.tokens.split(",")]
    else:
        with open(os.path.join(ASSETS, "encoder.json")) as f:
            enc = json.load(f)
        with open(os.path.join(ASSETS, "vocab.bpe"), encoding="utf-8") as f:
            merges = [tuple(m.split()) for m in f.read().split("\n")[1:-1]]
        ids = oai.Encoder(encoder=enc, bpe_merges=merges).encode(args.prompt)
    print("prompt ids:", ids)

    with torch.no_grad():
        if args.ref:
            lg = model(torch.tensor([ids])).logits[0].float().cpu().numpy()
            write_lgt1(args.ref, lg)
            print("wrote", args.ref, lg.shape)
        if args.greedy_ref:
            seq, rows = list(ids), []
            for _ in range(args.max_new_tokens):
                lg = model(torch.tensor([seq])).logits[0, -1].float().cpu()
                rows.append(lg.numpy())
                seq.append(int(lg.argmax()))
            import numpy as np
            write_lgt1(args.greedy_ref, np.stack(rows))
            print("greedy tokens:", seq[len(ids):])
            if args.greedy_tokens_out:
                with open(args.greedy_tokens_out, "w") as f:
                    f.write(" ".join(map(str, seq[len(ids):])) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
