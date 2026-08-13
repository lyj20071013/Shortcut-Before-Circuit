"""rarity 对齐的 go/no-go。不重训，直接吃已有 checkpoint。

判据（四条一起看，缺一条都不能下结论）：

  1. d_margin 跨格是否有梯度。主读数。R_old 越大，"重复⇒非当前"在语料里
     被演示得越多，若 rarity 更便宜学，高 R_old 格的 d_margin 应更大。
     单格的绝对值不说明什么，跨格差异才是信号。

  2. Δ 的符号。break_rarity 让 rarity 与 frequency 反向移动，故
     + 是 rarity 型、- 是 frequency 型。后者会是更强的结果（frequency
     在训练分布上每篇都被反驳），但先验上不该出现。

  3. 概率质量是否仍集中在 {v_old, v_new}。这是 OOD 有效性检查：编辑让
     答案值重复出现，是训练分布外的配置。若质量散掉（mass < 0.5），
     说明模型在探针输入上已不做任务，d_margin 无意义。

  4. 非 q slot 对照的 |Δ| 应远小于主读数。排除"编辑本身让模型脱离任务"
     这一泛化解释。q slot 未动时 rb==re，apply_edit 会拒绝，故这里手工
     构造视图并直接比较 margin。

R_old=1 列在此探针下无域（无副本可改写），按 N/A 报告，它是结构性的
零信号对照而非缺口。
"""
import argparse
import json
import os
import random
from typing import List, Optional, Sequence

import torch

from config import CorpusCfg, LangSpec
from generator import Doc, generate_corpus
from model import LM, ModelCfg
from probe import (EditedDoc, _answer_val, _edit_break_rarity_ctrl, _margin,
                   _order_ok, _p_final, _q_pos, _RULES, apply_edit,
                   fit_position_offset, r_last_value)
from generator import emit
from vocab import Vocab


class Predictor:
    """按前缀缓存 log-softmax。与 train.ModelPredictor 同接口。"""

    def __init__(self, model: LM, vocab: Vocab, spec: LangSpec, device):
        self.m, self.vocab, self.dev = model, vocab, device
        self.lo = vocab.val(0)
        self.n_val = spec.n_values
        self._c = {}

    @torch.no_grad()
    def _lp(self, view) -> torch.Tensor:
        key = tuple(view.tokens[:view.answer_pos])
        got = self._c.get(key)
        if got is None:
            ids = torch.tensor([key], device=self.dev)
            logits, _ = self.m(ids)
            got = torch.log_softmax(
                logits[0, -1, self.lo:self.lo + self.n_val].float(), -1).cpu()
            if len(self._c) > 8192:
                self._c.clear()
            self._c[key] = got
        return got

    def predict(self, view) -> int:
        return int(self._lp(view).argmax())

    def logp(self, view, cands: Sequence[int]) -> List[float]:
        lp = self._lp(view)
        return [float(lp[v]) for v in cands]

    def mass(self, view, cands: Sequence[int]) -> float:
        lp = self._lp(view)
        return float(sum(lp[v].exp() for v in cands))


def load(tag: str, out_dir: str, device):
    """从 jsonl 的 meta 取 spec/corpus（checkpoint 里没存 spec），
    从 .pt 取权重。两者必须来自同一个 run。"""
    jl = os.path.join(out_dir, f"{tag}.jsonl")
    pt = os.path.join(out_dir, f"{tag}.pt")
    with open(jl) as f:
        meta = json.loads(f.readline())
    assert meta["kind"] == "meta", f"{jl} 首行不是 meta"
    ck = torch.load(pt, map_location=device)
    spec = LangSpec(**meta["spec"])
    corpus = CorpusCfg(**meta["corpus"])
    mc = ModelCfg(**ck["model_cfg"])
    model = LM(mc).to(device)
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    model.eval()
    conv = ck.get("eval", {})
    return model, spec, corpus, meta, conv


def ctrl_view(d: Doc, vocab: Vocab, cfg: CorpusCfg, rng) -> Optional[EditedDoc]:
    """非 q slot 的多重性反转视图。q slot 未动，故目标规则预测不变，
    apply_edit 的守卫会拒绝 —— 这里绕过它自行构造。"""
    out = _edit_break_rarity_ctrl(d, cfg, vocab.spec, rng)
    if out is None:
        return None
    stmts, delta = out
    if not _order_ok(stmts):
        return None
    av = _answer_val(d)
    toks, apos = emit(stmts, d.q_ent, d.q_attr, d.q_hist_k, av, vocab, cfg)
    if len(toks) != len(d.tokens) or apos != d.answer_pos:
        return None
    return EditedDoc("ctrl", toks, apos, vocab.val(av), stmts, d.q_ent,
                     d.q_attr, d.q_hist_k, delta, len(stmts), d, "rarity",
                     +1, -1)


def run_one(tag: str, out_dir: str, n_docs: int, device) -> dict:
    model, spec, corpus, meta, conv = load(tag, out_dir, device)
    vocab = Vocab(spec)
    pred = Predictor(model, vocab, spec, device)
    docs = list(generate_corpus(vocab, corpus, n_docs, seed_offset=1))
    offset = fit_position_offset(docs)
    rng = random.Random(0)

    ds, masses, signs = [], [], []
    for d in docs:
        ed = apply_edit(d, "break_rarity", vocab, corpus, rng, offset)
        if ed is None:
            continue
        truth = r_last_value(d)
        v_star = ed.v_star
        dm = (_margin(pred, ed, v_star, truth)
              - _margin(pred, d, v_star, truth))
        ds.append(dm)
        signs.append(dm > 0)
        masses.append(pred.mass(ed, [v_star, truth]))

    cs = []
    rng2 = random.Random(1)
    for d in docs:
        cv = ctrl_view(d, vocab, corpus, rng2)
        if cv is None:
            continue
        truth = r_last_value(d)
        cand = [i for i in _q_pos(d) if i != _p_final(d)]
        if not cand:
            continue
        v_old = d.stmts[cand[-1]].val
        if v_old == truth:
            continue
        cs.append(_margin(pred, cv, v_old, truth)
                  - _margin(pred, d, v_old, truth))

    nan = float("nan")
    n = len(ds)
    return dict(
        tag=tag, r_old=corpus.r_old_lo, dd=corpus.delta_d_lo,
        n=n, yield_rate=n / len(docs),
        d_margin=(sum(ds) / n) if n else nan,
        frac_positive=(sum(signs) / n) if n else nan,
        mass=(sum(masses) / n) if n else nan,
        ctrl_n=len(cs),
        ctrl_margin=(sum(cs) / len(cs)) if cs else nan,
        acc=conv.get("acc", nan), tail0=conv.get("acc_tail0", nan))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="run tag，如 R3_D5_s0_big")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--txt", default="runs/go_nogo.txt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [run_one(t, a.out, a.docs, dev) for t in a.tags]

    lines = [
        f"{'tag':<22} {'R':>3} {'ΔD':>3} {'acc':>6} {'yield':>6} {'n':>5} "
        f"{'Δ':>8} {'frac+':>6} {'mass':>6} {'ctrlN':>6} {'ctrl':>8}"]
    for r in rows:
        lines.append(
            f"{r['tag']:<22} {r['r_old']:>3} {r['dd']:>3} {r['acc']:>6.3f} "
            f"{r['yield_rate']:>6.2f} {r['n']:>5} {r['d_margin']:>+8.3f} "
            f"{r['frac_positive']:>6.2f} {r['mass']:>6.2f} "
            f"{r['ctrl_n']:>6} {r['ctrl_margin']:>+8.3f}")
    lines += [
        "",
        "判读：Δ>0 rarity 型 / Δ<0 frequency 型 / |Δ|≈0 纯 recency。",
        "mass<0.5 读数无效（模型已脱离任务）。|ctrl| 应远小于 |Δ|。",
        "跨格 Δ 有梯度才算方向成立；单格绝对值不构成结论。",
        "",
        "raw: " + json.dumps(rows, ensure_ascii=False)]
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(os.path.dirname(a.txt) or ".", exist_ok=True)
    with open(a.txt, "w") as f:
        f.write(txt + "\n")
    print(f"\n已写入 {a.txt}")

if __name__ == "__main__":
    main()