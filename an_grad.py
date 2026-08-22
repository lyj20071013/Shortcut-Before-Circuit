"""梯度正交性。cos(grad Delta, grad L) 加噪声地板与正对照。

若别名使"用哪条规则"成为目标函数的平坦方向，则读数方向的梯度应与损失梯度
正交到噪声地板量级，而目标函数直接优化的量（真值 margin）不应如此。

六个梯度，同一 checkpoint：
  gD      读数 Delta（式 2）
  gM      真值 margin，符号取"越大越好"，正对照
  gLa/gLb 全 token 损失，两个不相交 batch
  gAa/gAb 答案位专属损失，两个不相交 batch

全 token 损失里答案位只占 1/225，正对照在它上面会被稀释，故两个参照都报。
判据：cos(gD, gL) 落在地板量级 且 cos(gM, -gL) 明显高于地板。

全程 fp32：bf16 的梯度噪声会淹没余弦测量。
"""
import argparse
import json
import os
import random
import statistics as st

import torch

from config import CorpusCfg, LangSpec
from generator import generate_corpus
from model import LM, ModelCfg
from probe import apply_edit, fit_position_offset, r_last_value
from vocab import Vocab

KIND = "break_rarity"


def load(tag, out_dir, device):
    jl = os.path.join(out_dir, f"{tag}.jsonl")
    pt = os.path.join(out_dir, f"{tag}.pt")
    with open(jl) as f:
        meta = json.loads(f.readline())
    assert meta["kind"] == "meta", f"{jl} first line is not meta"
    ck = torch.load(pt, map_location=device)
    spec = LangSpec(**meta["spec"])
    corpus = CorpusCfg(**meta["corpus"])
    m = LM(ModelCfg(**ck["model_cfg"])).to(device)
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    m.eval()
    return m, spec, corpus


def flat_grad(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p))
                      .reshape(-1) for p in model.parameters()])


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def val_lp(model, tokens, answer_pos, lo, n_val, device):
    """answer_pos 处 value-token 的 log-softmax，保留计算图。
    与 go_nogo.Predictor 同约定：喂 tokens[:answer_pos]，取最后一位。"""
    ids = torch.tensor([tokens[:answer_pos]], device=device)
    logits, _ = model(ids)
    return torch.log_softmax(logits[0, -1, lo:lo + n_val].float(), -1)


def grad_readout(model, pairs, lo, n_val, device, truth_margin=False):
    """逐篇 backward 累积，避免同时持有多个计算图。"""
    acc, n = None, 0
    for d, ed, v_star, v_truth in pairs:
        model.zero_grad(set_to_none=True)
        lp_b = val_lp(model, d.tokens, d.answer_pos, lo, n_val, device)
        if truth_margin:
            q = lp_b[v_truth] - lp_b[v_star]        # 越大越好
        else:
            lp_e = val_lp(model, ed.tokens, ed.answer_pos, lo, n_val, device)
            q = ((lp_e[v_star] - lp_e[v_truth])
                 - (lp_b[v_star] - lp_b[v_truth]))  # 式 2
        q.backward()
        g = flat_grad(model)
        acc = g.clone() if acc is None else acc.add_(g)
        n += 1
    model.zero_grad(set_to_none=True)
    return acc / max(n, 1), n


def grad_loss(model, docs, pad, device, answer_only=False):
    L = max(len(d.tokens) for d in docs)
    ids = torch.full((len(docs), L), pad, dtype=torch.long)
    lab = torch.full((len(docs), L), -100, dtype=torch.long)
    for i, d in enumerate(docs):
        t = torch.tensor(d.tokens)
        ids[i, :len(t)] = t
        if answer_only:
            lab[i, d.answer_pos] = t[d.answer_pos]
        else:
            lab[i, :len(t)] = t
    model.zero_grad(set_to_none=True)
    _, loss = model(ids.to(device), lab.to(device))
    loss.backward()
    g = flat_grad(model)
    model.zero_grad(set_to_none=True)
    return g, float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--loss-batch", type=int, default=96)
    ap.add_argument("--probe-offset", type=int, default=1,
                    help="held-out pool; 1 matches the eval/probe set")
    ap.add_argument("--train-offset", type=int, default=9001,
                    help="disjoint from both the probe pool and the训练流 1000+w")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    hdr = (f"{'tag':<22}{'n':>4}{'cosDL':>9}{'floorL':>9}"
           f"{'cosML':>9}{'cosDA':>9}{'floorA':>9}{'cosMA':>9}")
    print(hdr)
    print("-" * len(hdr))

    for tag in args.tags:
        model, spec, corpus = load(tag, args.out, dev)
        for p in model.parameters():
            p.requires_grad_(True)
        vocab = Vocab(spec)
        lo, n_val = vocab.val(0), spec.n_values

        docs = list(generate_corpus(vocab, corpus, args.docs * 3,
                                    seed_offset=args.probe_offset))
        offset = fit_position_offset(docs)
        rng = random.Random(0)
        pairs = []
        for d in docs:
            ed = apply_edit(d, KIND, vocab, corpus, rng, offset)
            if ed is None:
                continue
            pairs.append((d, ed, ed.v_star, r_last_value(d)))
            if len(pairs) >= args.docs:
                break
        if not pairs:
            print(f"{tag:<22}  no edit domain, skipped")
            continue

        gD, n = grad_readout(model, pairs, lo, n_val, dev)
        gM, _ = grad_readout(model, pairs, lo, n_val, dev, truth_margin=True)

        b = args.loss_batch
        tr = list(generate_corpus(vocab, corpus, 2 * b,
                                  seed_offset=args.train_offset))
        gLa, la = grad_loss(model, tr[:b], vocab.PAD, dev)
        gLb, lb = grad_loss(model, tr[b:], vocab.PAD, dev)
        gAa, aa = grad_loss(model, tr[:b], vocab.PAD, dev, answer_only=True)
        gAb, ab = grad_loss(model, tr[b:], vocab.PAD, dev, answer_only=True)
        gL, gA = 0.5 * (gLa + gLb), 0.5 * (gAa + gAb)

        r = dict(tag=tag, n=n,
                 cos_D_L=cos(gD, gL), floor_L=cos(gLa, gLb),
                 cos_M_negL=cos(gM, -gL),
                 cos_D_A=cos(gD, gA), floor_A=cos(gAa, gAb),
                 cos_M_negA=cos(gM, -gA),
                 gD_norm=float(gD.norm()), gM_norm=float(gM.norm()),
                 gL_norm=float(gL.norm()), gA_norm=float(gA.norm()),
                 loss_all=0.5 * (la + lb), loss_ans=0.5 * (aa + ab))
        rows.append(r)
        print(f"{tag:<22}{n:>4}{r['cos_D_L']:>+9.4f}{r['floor_L']:>+9.4f}"
              f"{r['cos_M_negL']:>+9.4f}{r['cos_D_A']:>+9.4f}"
              f"{r['floor_A']:>+9.4f}{r['cos_M_negA']:>+9.4f}")

        del model, gD, gM, gL, gA, gLa, gLb, gAa, gAb
        torch.cuda.empty_cache()

    if rows:
        print(f"\n{len(rows)} runs")
        for k, lbl in [("cos_D_L", "|cos(gD, gL_all)|"),
                       ("floor_L", "|floor, all-token|"),
                       ("cos_M_negL", "|cos(gM, -gL_all)|"),
                       ("cos_D_A", "|cos(gD, gL_ans)|"),
                       ("floor_A", "|floor, answer-only|"),
                       ("cos_M_negA", "|cos(gM, -gL_ans)|")]:
            v = [abs(r[k]) for r in rows]
            print(f"  {lbl:<24} median {st.median(v):.4f}   "
                  f"range {min(v):.4f} to {max(v):.4f}")
        print("\nread: gD at or below the floor while gM is well above it means")
        print("      the readout direction is flat to first order.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()