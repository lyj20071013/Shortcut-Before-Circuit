"""Evaluate interpolation and shared-batch comparisons between models."""
import argparse
import json
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import go_nogo as gn
from config import CorpusCfg, LangSpec
from generator import generate_corpus
from model import LM, ModelCfg
from probe import _margin, apply_edit, fit_position_offset, r_last_value
from train import copy_diag, evaluate
from vocab import Vocab

NAN = float("nan")
MASS_FLOOR = gn.MASS_FLOOR

# ---------------- 加载与一致性 ----------------

STRUCT_FIELDS = ("p_update", "max_updates", "r_old_lo", "r_old_hi",
                 "use_marker", "delta_d_lo", "delta_d_hi", "p_hist_query",
                 "n_stmts_lo", "n_stmts_hi", "min_slots", "spread",
                 "p_break", "k_old_break", "r_new_break", "truth_rule")


def load_pair(tag_a: str, tag_b: str, out_dir: str, device):
    """两个 run 的权重 + 共用的 spec/corpus。

    corpus 只允许在 name 与 seed 上不同：其余任一字段不同就不是同一个格子，
    插值出来的东西没有意义。seed 只进生成器的文档流，不进结构。
    """
    ma, spec_a, cor_a, meta_a, conv_a, esc_a = gn.load(tag_a, out_dir, device)
    mb, spec_b, cor_b, meta_b, conv_b, esc_b = gn.load(tag_b, out_dir, device)
    assert spec_a == spec_b, f"LangSpec 不同：{spec_a} vs {spec_b}"
    bad = [f for f in STRUCT_FIELDS
           if getattr(cor_a, f) != getattr(cor_b, f)]
    assert not bad, f"corpus 结构字段不同，不是同一格子：{bad}"
    ca, cb = ma.cfg, mb.cfg
    assert (ca.d_model, ca.n_layer, ca.n_head, ca.d_mlp, ca.vocab_size) == \
           (cb.d_model, cb.n_layer, cb.n_head, cb.d_mlp, cb.vocab_size), \
        "两个 run 的 ModelCfg 形状不同"
    sa = {k: v.detach().clone().float() for k, v in ma.state_dict().items()}
    sb = {k: v.detach().clone().float() for k, v in mb.state_dict().items()}
    assert sa.keys() == sb.keys(), "state_dict 键集不同"
    info = dict(
        tag_a=tag_a, tag_b=tag_b,
        esc_a=esc_a, esc_b=esc_b,
        step_a=conv_a.get("step", NAN), step_b=conv_b.get("step", NAN),
        acc_a=conv_a.get("acc", NAN), acc_b=conv_b.get("acc", NAN),
        copy_a=conv_a.get("copy_acc", NAN), copy_b=conv_b.get("copy_acc", NAN),
        sched=meta_a.get("train", {}).get("sched", "cos"),
    )
    return sa, sb, ma.cfg, spec_a, cor_a, cor_b, info


def build(state: Dict[str, torch.Tensor], cfg: ModelCfg, device) -> LM:
    """从 state dict 造模型。strict=True：键少一个也要炸，不要静默用初始值。"""
    m = LM(cfg).to(device)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


def lerp(sa, sb, s: float) -> Dict[str, torch.Tensor]:
    """(1-s)·A + s·B，逐张量。RMSNorm 的 gain 与 embedding 一并插值 ——
    它们都是参数，排除任何一类都不再是权重空间的直线。
    RoPE 的 cos/sin 是 lazy buffer 不在 state_dict 里，由 build 重建。"""
    return {k: (1.0 - s) * sa[k] + s * sb[k] for k in sa}


# ---------------- 读数 ----------------

class Pair:
    """一篇文档的编辑对。预先算好，所有 s 复用同一批。"""
    __slots__ = ("base", "edit", "v_star", "truth")

    def __init__(self, base, edit, v_star, truth):
        self.base, self.edit = base, edit
        self.v_star, self.truth = v_star, truth


def make_pairs(vocab: Vocab, corpus: CorpusCfg, n_docs: int) -> Tuple[list, list]:
    """(编辑对, 原始文档)。与 go_nogo.run_one 同一条路径：seed_offset=1、
    剔除反转文档、fit_position_offset 后 random.Random(0) 逐篇 apply_edit。

    这里把编辑对固定下来是必须的：apply_edit 消耗 rng 且逐篇顺序相关，若在
    每个 s 上重新生成，不同 s 之间比较的就不是同一组文档。
    """
    all_docs = list(generate_corpus(vocab, corpus, n_docs, seed_offset=1))
    docs = [d for d in all_docs if not getattr(d, "is_break", False)]
    offset = fit_position_offset(docs)
    rng = random.Random(0)
    pairs = []
    for d in docs:
        ed = apply_edit(d, "break_rarity", vocab, corpus, rng, offset)
        if ed is None:
            continue
        pairs.append(Pair(d, ed, ed.v_star, r_last_value(d)))
    return pairs, docs


@torch.no_grad()
def readout(model: LM, pairs: Sequence[Pair], vocab: Vocab, spec: LangSpec,
            device) -> dict:
    """frac+ / 中位数 / mass。Predictor 每次新建：它按 token 前缀缓存
    log-softmax，跨模型复用会读到上一个 s 的输出。"""
    pred = gn.Predictor(model, vocab, spec, device)
    ds, ms = [], []
    for p in pairs:
        dm = (_margin(pred, p.edit, p.v_star, p.truth)
              - _margin(pred, p.base, p.v_star, p.truth))
        ds.append(dm)
        ms.append(pred.mass(p.edit, [p.v_star, p.truth]))
    n = len(ds)
    if not n:
        return dict(n=0, frac_positive=NAN, d_median=NAN, d_median_valid=NAN,
                    mass=NAN, sign_p=NAN)
    k_pos = sum(1 for x in ds if x > 0)
    valid = sorted(x for x, m in zip(ds, ms) if m >= MASS_FLOOR)
    return dict(
        n=n,
        frac_positive=k_pos / n,
        sign_p=gn.sign_test_p(k_pos, n),
        d_median=gn.percentile(sorted(ds), 0.50),
        d_median_valid=gn.percentile(valid, 0.50),
        mass=sum(ms) / n,
        n_valid=len(valid),
    )


def cross_check(sa, sb, cfg, spec, cor_a, cor_b, n_docs, vocab, device) -> dict:
    """2x2：两个端点模型 × 两批文档。

    cfg.seed 进文档流，所以已发表的两个 frac+ 分别在自己那批文档上算出。
    这里把两个模型都在两批上读一遍，得到

              docs(A)   docs(B)
        model A   aa        ab
        model B   ba        bb

    对角线应复现已发表值（同模型同文档）。行内差（aa vs ab）是换文档带来的
    移动，列内差（aa vs ba）是换 run 带来的移动。§5.5 断言后者远大于前者；
    若不是，跨种子极差里就混进了文档采样，整篇的方差排序要重算。

    这不是插值的一部分，是插值能不能读的前提，所以放在 sweep 之前跑。
    """
    pa, _ = make_pairs(vocab, cor_a, n_docs)
    pb, _ = make_pairs(vocab, cor_b, n_docs)
    ma = build(sa, cfg, device)
    mb = build(sb, cfg, device)
    out = {}
    for mname, m in (("A", ma), ("B", mb)):
        for dname, prs in (("A", pa), ("B", pb)):
            r = readout(m, prs, vocab, spec, device)
            out[f"{mname}{dname}"] = r["frac_positive"]
            out[f"n_{mname}{dname}"] = r["n"]
    del ma, mb
    if device == "cuda":
        torch.cuda.empty_cache()
    aa, ab = out["AA"], out["AB"]
    ba, bb = out["BA"], out["BB"]
    out["doc_move"] = max(abs(aa - ab), abs(ba - bb))
    out["run_move"] = max(abs(aa - ba), abs(ab - bb))
    # 二项标准误的上界，n 取两批的较小者
    n = min(out["n_AA"], out["n_AB"], out["n_BA"], out["n_BB"])
    out["se"] = 0.5 / math.sqrt(n) if n else NAN
    return out


@torch.no_grad()
def task_metrics(model: LM, eval_docs, copy_docs, vocab: Vocab,
                 spec: LangSpec, corpus: CorpusCfg, device) -> dict:
    """训练分布上的 loss 与答案位准确率，加 copy 诊断。

    loss 用 train.evaluate 算，与训练时完全同一个全 token CE —— dip 判据是
    关于训练目标的陈述，换任何别的量都不成立。eval_docs 全程同一批：ΔL 的
    量级在 1e-3 以下，换批的采样噪声会盖过它。
    """
    ev = evaluate(model, eval_docs, vocab, spec, device)
    cd = copy_diag(model, copy_docs, vocab, spec, corpus, device)
    model.eval()          # evaluate/copy_diag 结束时会 model.train()
    return dict(loss=ev["loss"], acc=ev["acc"], acc_tail0=ev["acc_tail0"],
                ans_nll=ev["ans_nll"], copy_acc=cd["copy_acc"])


# ---------------- 主流程 ----------------

def sweep(sa, sb, cfg, spec, corpus, pairs, eval_docs, copy_docs,
          vocab, device, n_pts: int) -> List[dict]:
    rows = []
    for i in range(n_pts):
        s = i / (n_pts - 1)
        m = build(lerp(sa, sb, s), cfg, device)
        r = dict(s=round(s, 4))
        r.update(task_metrics(m, eval_docs, copy_docs, vocab, spec, corpus,
                              device))
        r.update(readout(m, pairs, vocab, spec, device))
        rows.append(r)
        print(f"  s={s:.2f}  loss={r['loss']:.5f}  acc={r['acc']:.4f}  "
              f"copy={r['copy_acc']:.3f}  frac+={r['frac_positive']:.3f}  "
              f"med={r['d_median']:+.3f}  mass={r['mass']:.2f}", flush=True)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def verdict(rows: List[dict], xc: Optional[dict],
            published: Optional[Tuple[float, float]]) -> str:
    """判别表 + 自检。published 是 (frac+ at s=0, frac+ at s=1) 的已发表值。

    自检的容差按二项误差定，不按"应完全相等"定：cfg.seed 进文档流，所以
    B 的已发表值在 docs(B) 上算出，而路径上的 s=1 在 docs(A) 上读。对角线
    （AA / BB，同模型同文档）才是能对到小数点后两位的那一对。
    """
    L = [r["loss"] for r in rows]
    f = [r["frac_positive"] for r in rows]
    # dip 只在内部点上有意义，且参照是**更好**的那个端点。非收敛假设说两端
    # 都在通往同一个吸引子的路上，其 signature 是某个内部点比两端都好。
    # 用 max(L0,L1) 作参照时，只要两端不等高，靠近较好端点的内部点就会冒充
    # 成 dip：实测 R3_D8 的内部最小 2.5637（s=0.9）对 max 2.5786 给出
    # -0.015，而路径真正的最小值在端点 s=1.0 的 2.5502，无内部 dip。
    end_lo, end_hi = min(L[0], L[-1]), max(L[0], L[-1])
    inner = L[1:-1] if len(L) > 2 else []
    dip = (min(inner) - end_lo) if inner else float("nan")
    i_min = (1 + inner.index(min(inner))) if inner else 0
    bump = max(L) - end_hi
    i_max = L.index(max(L))
    out = ["", "=" * 68,
           f"loss 两端 {L[0]:.5f} / {L[-1]:.5f}",
           f"dip  = min_inner L - min(L0,L1) = {dip:+.5f}  @ s={rows[i_min]['s']}"
           f"bump = max_s L - max(L0,L1) = {bump:+.5f}  @ s={rows[i_max]['s']}",
           f"frac+ 两端 {f[0]:.3f} / {f[-1]:.3f}，路径 "
           + " ".join(f"{x:.2f}" for x in f)]

    if xc is not None:
        se = xc["se"]
        out += ["", "2x2 交叉检查（模型 × 文档批）",
                f"            docs(A)   docs(B)",
                f"  model A   {xc['AA']:.3f}     {xc['AB']:.3f}",
                f"  model B   {xc['BA']:.3f}     {xc['BB']:.3f}",
                f"  换文档移动 {xc['doc_move']:.3f}，换 run 移动 "
                f"{xc['run_move']:.3f}，二项 SE 约 {se:.3f}"]
        if xc["run_move"] > 3 * xc["doc_move"]:
            out += ["  换 run 的移动远大于换文档 -> 读出是 run 的属性，",
                    "  §5.5 的方差排序成立。"]
        elif xc["doc_move"] > 2 * se:
            out += ["  换文档的移动超出二项误差，且与换 run 同量级 -> 跨种子",
                    "  极差里混进了文档采样。这会动到 §5.5，先查这一项再看插值。"]
        else:
            out += ["  换文档的移动在二项误差内。"]

    if published is not None:
        pa, pb = published
        # 对角线对已发表值：同模型同文档，应当几乎相等
        if xc is not None:
            ea, eb = abs(xc["AA"] - pa), abs(xc["BB"] - pb)
            tol = 0.02
            what = "对角线 AA/BB"
        else:
            ea, eb = abs(f[0] - pa), abs(f[-1] - pb)
            tol = 3 * 0.5 / math.sqrt(max(1, rows[0].get("n", 400)))
            what = "端点（换批读数，容差放宽到 3SE）"
        ok = max(ea, eb) <= tol
        out += ["",
                f"自检：{what} 对已发表值 {pa:.3f}/{pb:.3f} "
                f"偏差 {ea:.3f}/{eb:.3f}，容差 {tol:.3f} -> "
                f"{'通过' if ok else '不通过'}"]
        if not ok:
            out += ["  读数路径与 go_nogo 不一致，路径上的数不要用。",
                    "  先查：--docs 是否 400、编辑对是否 seed_offset=1、",
                    "  是否漏了剔除 is_break 文档。"]
            return "\n".join(out)

    # 单调性：谷底坐标的说法要求 frac+ 沿路大体单调，不要求严格
    inc = sum(1 for i in range(len(f) - 1) if f[i + 1] > f[i])
    dec = sum(1 for i in range(len(f) - 1) if f[i + 1] < f[i])
    mono = max(inc, dec) / max(1, inc + dec)

    out += ["", "读法："]
    if bump > 0.5:
        out += [f"  壁垒很大（bump {bump:+.3f}，两端 loss 约 {end_hi:.3f}）。",
                "  这是两个独立初始化几乎必然落在不同置换胞腔的后果，是对称性",
                "  的产物，不是关于欠定的信息。dip 在这个量级下无法读出。",
                "  这一行的用处只有三个：读数路径自检、2x2 交叉检查、",
                "  记录朴素壁垒的基线。判别攻击 2 必须先做置换对齐（stage 2），",
                "  或改用同初始化的分叉对（--data-seed 臂）。"]
    elif dip < -0.002:
        out += [f"  dip 为负（{dip:+.5f}）。路径中段存在训练 loss 更低的点，",
                "  A lower point on this sampled path does not certify local non-optimality ",
                "  at either endpoint; interpret the interpolation descriptively."]
    elif bump > 0.01:
        out += [f"  Small sampled-path barrier (bump {bump:+.5f}) and no observed dip. ",
                "  This alone does not establish distinct basins, underspecification or convergence.",
                "  Permutation alignment can change the path and its barrier.",
                "  A barrier on one path does not exclude alternative connecting paths."]
    else:
        out += [f"  路径基本平坦（dip {dip:+.5f}，bump {bump:+.5f}），",
                f"  而 frac+ 从 {f[0]:.2f} 走到 {f[-1]:.2f}（单调度 {mono:.2f}）。",
                "  Readout variation accompanies a small sampled loss change. ",
                "  This does not prove a flat manifold or an unconstrained objective direction.",
                "  前提是端点确实同置换胞腔（分叉对，或对齐之后）。"]
    out += ["=" * 68]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_a", help="如 R3_D8_s0_grid")
    ap.add_argument("tag_b", help="如 R3_D8_s2_grid")
    ap.add_argument("--out", default="runs_g2")
    ap.add_argument("--points", type=int, default=11)
    ap.add_argument("--docs", type=int, default=400,
                    help="读数文档数，与 go_nogo 默认一致才能自检")
    ap.add_argument("--eval-docs", type=int, default=2048)
    ap.add_argument("--copy-docs", type=int, default=256)
    ap.add_argument("--published", type=float, nargs=2, default=None,
                    metavar=("FRAC_A", "FRAC_B"),
                    help="两端已发表的 frac+，用于自检")
    ap.add_argument("--no-cross", action="store_true",
                    help="跳过 2x2 交叉检查（默认跑，它是插值可读性的前提）")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sa, sb, cfg, spec, corpus, cor_b, info = load_pair(
        a.tag_a, a.tag_b, a.out, device)
    print(f"A {info['tag_a']}  acc={info['acc_a']:.4f} "
          f"copy={info['copy_a']:.3f} esc={info['esc_a']}")
    print(f"B {info['tag_b']}  acc={info['acc_b']:.4f} "
          f"copy={info['copy_b']:.3f} esc={info['esc_b']}")
    print(f"schedule={info['sched']}  corpus 结构字段已核对一致")

    vocab = Vocab(spec)

    # 2x2 先跑：它不依赖插值，且若"换文档移动"与"换 run 移动"同量级，
    # 插值路径上的 frac+ 也读不出东西，应当先停下来查。
    xc = None
    if not a.no_cross:
        print("2x2 交叉检查（模型 × 文档批）…", flush=True)
        xc = cross_check(sa, sb, cfg, spec, corpus, cor_b, a.docs, vocab,
                         device)
        print(f"  AA={xc['AA']:.3f} AB={xc['AB']:.3f} "
              f"BA={xc['BA']:.3f} BB={xc['BB']:.3f}  "
              f"docMove={xc['doc_move']:.3f} runMove={xc['run_move']:.3f}\n",
              flush=True)

    pairs, docs = make_pairs(vocab, corpus, a.docs)
    print(f"编辑对 {len(pairs)} / 文档 {len(docs)}（固定，所有 s 复用）")

    # eval/copy 用与读数同一条流（seed_offset=1），与训练流 1000+w 不相交。
    ev_docs = list(generate_corpus(vocab, corpus, a.eval_docs, seed_offset=1))
    cp_docs = ev_docs[:a.copy_docs]
    print(f"eval {len(ev_docs)} 篇，copy {len(cp_docs)} 篇（固定）\n")

    rows = sweep(sa, sb, cfg, spec, corpus, pairs, ev_docs, cp_docs,
                 vocab, device, a.points)
    print(verdict(rows, xc, tuple(a.published) if a.published else None))

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(info=info, cross=xc, n_pairs=len(pairs),
                           n_eval=len(ev_docs), rows=rows), f, indent=1)
        print(f"\n已写入 {a.json}")


if __name__ == "__main__":
    main()
