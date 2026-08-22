"""平坦方向的有限差分测量。测 L 本身而不是 grad L。

一阶版在收敛点不可用：那里 grad L 由填充 token 的不可约熵地板主导，
真值 margin 也测不出对齐（cos 0.001 对地板 0.13）。这里改测：沿单位方向
走 eps，Δ 移动多少、L 移动多少。固定 batch + 固定探针集 => 无采样噪声。

三个方向：读数梯度、真值 margin 梯度、随机。关心的量是每单位 L 上升
换来多少 Δ 位移 —— 平坦方向应该便宜得多。
"""
import argparse, json, os, random
import torch
from config import CorpusCfg, LangSpec
from generator import generate_corpus
from model import LM, ModelCfg
from probe import apply_edit, fit_position_offset, r_last_value
from vocab import Vocab

KIND = "break_rarity"


def load(tag, out_dir, dev):
    with open(os.path.join(out_dir, f"{tag}.jsonl")) as f:
        meta = json.loads(f.readline())
    ck = torch.load(os.path.join(out_dir, f"{tag}.pt"), map_location=dev)
    spec, corpus = LangSpec(**meta["spec"]), CorpusCfg(**meta["corpus"])
    m = LM(ModelCfg(**ck["model_cfg"])).to(dev)
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    m.eval()
    return m, spec, corpus


def flat(model, attr="grad"):
    return torch.cat([(getattr(p, attr) if getattr(p, attr) is not None
                       else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()])


def add_(model, vec, eps):
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.add_(vec[i:i + n].view_as(p), alpha=eps)
            i += n


@torch.no_grad()
def measure(model, pairs, batch, lo, n_val, dev):
    """返回 (mean Delta, sign fraction, loss)。"""
    ds = []
    for d, ed, vs, vt in pairs:
        out = []
        for v in (d, ed):
            ids = torch.tensor([v.tokens[:v.answer_pos]], device=dev)
            lg, _ = model(ids)
            lp = torch.log_softmax(lg[0, -1, lo:lo + n_val].float(), -1)
            out.append(float(lp[vs] - lp[vt]))
        ds.append(out[1] - out[0])
    ids, lab = batch
    _, loss = model(ids, lab)
    return (sum(ds) / len(ds),
            sum(x > 0 for x in ds) / len(ds), float(loss))


def grad_of(model, pairs, lo, n_val, dev, truth=False):
    acc, n = None, 0
    for d, ed, vs, vt in pairs:
        model.zero_grad(set_to_none=True)
        ids = torch.tensor([d.tokens[:d.answer_pos]], device=dev)
        lg, _ = model(ids)
        lp_b = torch.log_softmax(lg[0, -1, lo:lo + n_val].float(), -1)
        if truth:
            q = lp_b[vt] - lp_b[vs]
        else:
            ids2 = torch.tensor([ed.tokens[:ed.answer_pos]], device=dev)
            lg2, _ = model(ids2)
            lp_e = torch.log_softmax(lg2[0, -1, lo:lo + n_val].float(), -1)
            q = (lp_e[vs] - lp_e[vt]) - (lp_b[vs] - lp_b[vt])
        q.backward()
        g = flat(model)
        acc = g.clone() if acc is None else acc.add_(g)
        n += 1
    model.zero_grad(set_to_none=True)
    return acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--loss-docs", type=int, default=256)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for tag in args.tags:
        model, spec, corpus = load(tag, args.out, dev)
        vocab = Vocab(spec)
        lo, n_val = vocab.val(0), spec.n_values

        docs = list(generate_corpus(vocab, corpus, args.docs * 3, seed_offset=1))
        off = fit_position_offset(docs)
        rng = random.Random(0)
        pairs = []
        for d in docs:
            ed = apply_edit(d, KIND, vocab, corpus, rng, off)
            if ed is None:
                continue
            pairs.append((d, ed, ed.v_star, r_last_value(d)))
            if len(pairs) >= args.docs:
                break

        tr = list(generate_corpus(vocab, corpus, args.loss_docs, seed_offset=9001))
        L = max(len(d.tokens) for d in tr)
        ids = torch.full((len(tr), L), vocab.PAD, dtype=torch.long)
        lab = torch.full((len(tr), L), -100, dtype=torch.long)
        for i, d in enumerate(tr):
            t = torch.tensor(d.tokens)
            ids[i, :len(t)] = t
            lab[i, :len(t)] = t
        batch = (ids.to(dev), lab.to(dev))

        d0, f0, l0 = measure(model, pairs, batch, lo, n_val, dev)
        gD = grad_of(model, pairs, lo, n_val, dev)
        gM = grad_of(model, pairs, lo, n_val, dev, truth=True)
        torch.manual_seed(0)
        gR = torch.randn_like(gD)
        dirs = [("readout", gD / gD.norm()), ("truth-margin", gM / gM.norm()),
                ("random", gR / gR.norm())]
        e0 = 1.0 / float(gD.norm())

        print(f"\n{tag}   n={len(pairs)}  base: Delta {d0:+.3f}  "
              f"frac+ {f0:.3f}  loss {l0:.5f}   eps0 {e0:.3e}")
        print(f"{'direction':<14}{'eps/eps0':>9}{'dDelta':>9}{'dfrac':>8}"
              f"{'dLoss':>11}{'dDelta/dLoss':>14}")
        for name, u in dirs:
            for k in (0.5, 1.0, 2.0, 4.0):
                eps = k * e0
                add_(model, u, eps)
                d1, f1, l1 = measure(model, pairs, batch, lo, n_val, dev)
                add_(model, u, -eps)
                dl = l1 - l0
                print(f"{name:<14}{k:>9.1f}{d1-d0:>+9.3f}{f1-f0:>+8.3f}"
                      f"{dl:>+11.2e}{abs(d1-d0)/abs(dl) if dl else float('nan'):>14.1f}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()