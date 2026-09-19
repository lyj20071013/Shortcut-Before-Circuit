"""Measure behavioral answers on generated rule-disagreement documents."""
import argparse, json, os

import torch

from config import CorpusCfg, LangSpec
from generator import generate_corpus
from go_nogo import Predictor, discover, load
from probe import r_last_value, r_rarity
from vocab import Vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_g2")
    ap.add_argument("--pattern", default="R*_D*_s*_grid")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--p-break", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    rows = []
    for tag in discover(a.out, a.pattern):
        model, spec, corpus, meta, conv, esc = load(tag, a.out, a.device)
        vocab = Vocab(spec)
        pred = Predictor(model, vocab, spec, a.device)

        # 在字典层面覆盖 p_break，其余字段与训练时逐字节一致。
        cb = CorpusCfg(**{**meta["corpus"], "p_break": a.p_break})
        docs = list(generate_corpus(vocab, cb, a.n, seed_offset=7777))
        brk = [d for d in docs if getattr(d, "is_break", False)]

        n_rec = n_rar = n_other = n_deg = 0
        for d in brk:
            v_new = r_last_value(d)
            v_old = r_rarity(d)
            if v_new is None or v_old is None or v_new == v_old:
                n_deg += 1          # 反转后仍不分歧：该篇无判别力
                continue
            p = pred.predict(d)
            if   p == v_new: n_rec += 1
            elif p == v_old: n_rar += 1
            else:            n_other += 1

        tot = n_rec + n_rar + n_other
        rows.append(dict(
            tag=tag, state=None, n_gen=len(docs), n_brk=len(brk),
            n_deg=n_deg, n=tot, acc=conv.get("acc"), copy_acc=conv.get("copy_acc"),
            pRec=(n_rec / tot) if tot else None,
            pRar=(n_rar / tot) if tot else None,
            pOther=(n_other / tot) if tot else None))

    print(f"{'tag':<24} {'nGen':>5} {'nBrk':>5} {'nDeg':>5} {'n':>5} "
          f"{'pRec':>7} {'pRar':>7} {'pOther':>7}")
    for r in sorted(rows, key=lambda x: x["tag"]):
        f = lambda v: f"{v:>7.3f}" if v is not None else f"{'—':>7}"
        print(f"{r['tag']:<24} {r['n_gen']:>5} {r['n_brk']:>5} {r['n_deg']:>5} "
              f"{r['n']:>5} {f(r['pRec'])} {f(r['pRar'])} {f(r['pOther'])}")
    with open(os.path.join(a.out, "propB.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()