"""Compute the RoPE-safe QK logit upper bound from supplied checkpoints."""
import argparse
import json
import math
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    import torch
    rows = []
    for path in a.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        state = ck["model"]
        per_layer = {}
        for key, q in state.items():
            if not key.endswith(".qn.w"):
                continue
            prefix = key[:-len(".qn.w")]
            k = state[prefix + ".kn.w"]
            if q.ndim != 1 or k.shape != q.shape:
                raise ValueError("Expected shared per-head one-dimensional QK gains.")
            per_layer[prefix] = float(q.double().abs().max() * k.double().abs().max()) * math.sqrt(q.numel())
        if not per_layer:
            raise ValueError(f"No normalized QK gain vectors in {path}")
        rows.append({"checkpoint":path.name, "per_layer_bound":per_layer,
                     "model_wide_bound":max(per_layer.values()),
                     "formula":"sqrt(d_h) * max(abs(w_q)) * max(abs(w_k))"})
    text = json.dumps(rows, indent=2)
    print(text)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(text + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()

