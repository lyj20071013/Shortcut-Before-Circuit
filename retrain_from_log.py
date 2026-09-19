"""Read a synthetic-run recipe from its log; print it unless --execute is set."""
import argparse
import json
from pathlib import Path

def recipe(path, out):
    with path.open(encoding="utf-8-sig") as f:
        meta = next((json.loads(line) for line in f if line.strip()), {})
    if meta.get("kind") != "meta" or not all(k in meta for k in ("corpus", "train", "model", "spec")):
        raise ValueError("Use a train.py trajectory with corpus/train/model/spec metadata; NL/FT have separate entry points.")
    corpus = dict(meta["corpus"])
    training = dict(meta["train"])
    model = dict(meta["model"])
    spec = dict(meta["spec"])
    if not path.stem.startswith(corpus["name"]):
        raise ValueError("The log filename does not preserve the recorded corpus name.")
    training["out_dir"] = str(out)
    model.pop("vocab_size", None)
    model.pop("ctx_len", None)
    return {
        "source_log": path.name,
        "corpus": corpus, "train": training, "model": model, "spec": spec,
        "tag_suffix": path.stem[len(corpus["name"]):],
        "ckpt_steps": meta.get("ckpt_steps", [])
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="fresh directory for rerun artifacts")
    ap.add_argument("--ckpt-steps", help="optional comma-separated steps for FP32 checkpoints")
    ap.add_argument("--execute", action="store_true", help="start training instead of printing the recipe")
    a = ap.parse_args()
    r = recipe(a.log, a.out)
    if a.ckpt_steps is not None:
        r["ckpt_steps"] = sorted({int(s) for s in a.ckpt_steps.split(",") if s.strip()})
    if any(s < 1 or s > r["train"]["total_steps"] for s in r["ckpt_steps"]):
        ap.error("checkpoint steps must lie within the recorded training budget")
    print(json.dumps(r, indent=2, ensure_ascii=False))
    if not a.execute:
        print("Recipe only. Add --execute to train; no model has been loaded.")
        return
    if (a.out / (a.log.stem + ".jsonl")).exists() or (a.out / (a.log.stem + ".pt")).exists():
        raise FileExistsError("Rerun outputs already exist; choose a fresh --out directory.")
    from config import CorpusCfg, LangSpec
    from train import TrainCfg, train
    r["train"]["betas"] = tuple(r["train"]["betas"])
    train(CorpusCfg(**r["corpus"]), TrainCfg(**r["train"]),
          mc_kw=r["model"], spec=LangSpec(**r["spec"]),
          tag_suffix=r["tag_suffix"], ckpt_steps=r["ckpt_steps"])

if __name__ == "__main__":
    main()

