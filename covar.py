"""25 格的协变量与生成器不变量。纯 CPU，约 20 分钟，可与训练并行。

产出两块：
  1. 附录 app:gen 的五条不变量实测值。正文 §3.3 只叙述了这五条，没有数字；
     审稿人会问「不可区分是多不可区分」，这里给出比值。
  2. 正文 tab:covar 的六列协变量。这些量随轴变化且无法在构造内固定，
     只能逐格报告 —— 隐藏它们比报告它们危险得多。

判据（不满足就必须在正文说明，而不是悄悄放过）：
  bindMax  ≤3      同一 (slot,value) 对跨文档最多出现几次。大了说明可记忆
  |r_len|  <0.10   ΔD 与文档长度的相关。非零则 ΔD 轴混入长度效应
  ansLast  =0      答案等于查询前最后一个值 token 的篇数。非零则有平凡捷径
  updDens  ≈1.0    末十分位的 update 密度 / 全局密度。偏离则位置分布不均匀
  adjΔ     <0.05   q slot 与 filler slot 的同 slot 邻接率之差
  antRatio 0.9–1.2 q slot 与 filler slot 的前驱距离比。偏离 1 则 q slot 可辨认
"""
import argparse
import itertools
import json
import os
from collections import Counter

from config import CorpusCfg, LangSpec, dd_band, validate_cfg
from generator import generate_corpus
from vocab import Vocab

R_OLDS = [3, 5, 8, 12, 16]
DDS = [2, 3, 5, 8, 16]
NAN = float("nan")


def mk_cfg(r, d, seed=0, lo_st=45, hi_st=55):
    lo, hi = dd_band(d)
    return CorpusCfg(name=f"R{r}_D{d}", seed=seed, p_update=0.5,
                     max_updates=1, r_old_lo=r, r_old_hi=r,
                     use_marker=False, delta_d_lo=lo, delta_d_hi=hi,
                     p_hist_query=0.0, n_stmts_lo=lo_st, n_stmts_hi=hi_st)


def q_indices(d):
    """q slot 的语句下标。与 probe._q_pos 同义，此处独立实现以免依赖其内部。"""
    return [i for i, s in enumerate(d.stmts)
            if s.ent == d.q_ent and s.attr == d.q_attr]


def slot_key(s):
    return (s.ent, s.attr)


def measure(docs, spec):
    n = len(docs)
    out = {}

    # 不变量 1：绑定不可记忆。若同一 (slot,value) 对反复出现，模型可绕过检索
    # 不变量 1：绑定不可跨文档记忆。数同一 (slot,value) 三元组出现在几篇
# 不同文档里，而不是总出现次数 —— 后者恒 ≥ R_old（q slot 自身就有
# R_old 份副本），会把构造本身报成违反。
    pair = Counter()
    for d in docs:
        for k in {(s.ent, s.attr, s.val) for s in d.stmts}:
            pair[k] += 1
    out["bindMax"] = max(pair.values()) if pair else 0

    # 不变量 2：ΔD 精确，且与长度无关。相关非零则 ΔD 轴混入长度效应
    dds = [d.realized_delta for d in docs]
    lens = [len(d.tokens) for d in docs]
    md, ml = sum(dds) / n, sum(lens) / n
    vd = sum((x - md) ** 2 for x in dds)
    vl = sum((x - ml) ** 2 for x in lens)
    cov = sum((a - md) * (b - ml) for a, b in zip(dds, lens))
    out["r_len"] = cov / (vd * vl) ** 0.5 if vd > 0 and vl > 0 else 0.0
    out["ddReal"] = md

    # 不变量 3：答案 ≠ 查询前最后一个值 token。非零即存在平凡捷径
    bad = 0
    for d in docs:
        qi = q_indices(d)
        if not qi:
            continue
        ans = d.stmts[qi[-1]].val
        if d.stmts[-1].val == ans:
            bad += 1
    out["ansLast"] = bad

    # 不变量 4：filler slot 的 update 位置在查询前区域均匀。必须排除 q slot
# 自身的 rebinding —— 它固定在 m−ΔD−1，小 ΔD 时正落在末十分位，
# 会把「q slot 的构造」误报成「位置分布不均匀」。
    tot_u = tot_s = tail_u = tail_s = 0
    for d in docs:
        qi = set(q_indices(d))
        m = len(d.stmts)
        cut = int(m * 0.9)
        for i, s in enumerate(d.stmts):
            if i in qi:
                continue
            u = int(getattr(s, "upd", getattr(s, "is_update", False)))
            tot_u += u
            tot_s += 1
            if i >= cut:
                tail_u += u
                tail_s += 1

    g = tot_u / tot_s if tot_s else 0.0
    t = tail_u / tail_s if tail_s else 0.0
    out["updDens"] = t / g if g > 0 else NAN

    # 不变量 5：q slot 与 filler slot 结构上不可区分。两个量：
    #   adj  同 slot 相邻出现的比例
    #   ant  到最近同 slot 前驱的平均距离
    # 任一显著不同，模型就能不读查询而认出 q slot
    def slot_stats(d, want_q):
        by = {}
        for i, s in enumerate(d.stmts):
            by.setdefault(slot_key(s), []).append(i)
        qk = (d.q_ent, d.q_attr)
        adj_n = adj_d = 0
        ants = []
        for k, idx in by.items():
            if (k == qk) != want_q or len(idx) < 2:
                continue
            for a, b in zip(idx, idx[1:]):
                adj_d += 1
                adj_n += (b - a == 1)
                ants.append(b - a)
        return adj_n, adj_d, ants

    qa_n = qa_d = fa_n = fa_d = 0
    q_ant, f_ant = [], []
    for d in docs:
        a, b, c = slot_stats(d, True)
        qa_n += a
        qa_d += b
        q_ant += c
        a, b, c = slot_stats(d, False)
        fa_n += a
        fa_d += b
        f_ant += c
    q_adj = qa_n / qa_d if qa_d else NAN
    f_adj = fa_n / fa_d if fa_d else NAN
    out["adjQ"], out["adjF"] = q_adj, f_adj
    out["adjD"] = abs(q_adj - f_adj) if qa_d and fa_d else NAN
    mq = sum(q_ant) / len(q_ant) if q_ant else NAN
    mf = sum(f_ant) / len(f_ant) if f_ant else NAN
    out["antQ"], out["antF"] = mq, mf
    out["antRatio"] = mq / mf if f_ant and mf else NAN

    # tab:covar 的六列
    out["slots"] = sum(d.n_slots for d in docs) / n
    out["rreal"] = sum(d.q_kept for d in docs) / n
    out["tailUpd"] = sum(1 for d in docs if d.n_tail_updates == 0) / n
    out["nStmts"] = sum(len(d.stmts) for d in docs) / n
    out["maxTok"] = max(len(d.tokens) for d in docs)
    # edit domain：至少两份 v_old 存活。与 probe 的实际 yield 略有差异
    # （后者还要求编辑后五项不变量全部保持），故正文用 go_nogo 的 yield
    out["domain"] = sum(1 for d in docs if d.q_kept >= 2) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1500,
                    help="每格文档数。1500 与正文 §3.3 的声明一致")
    ap.add_argument("--n-values", type=int, default=512)
    ap.add_argument("--n-entities", type=int, default=200)
    ap.add_argument("--txt", default="runs_g2/covar.txt")
    ap.add_argument("--tex", default="runs_g2/covar.tex")
    a = ap.parse_args()

    spec = LangSpec(n_values=a.n_values, n_entities=a.n_entities)
    vocab = Vocab(spec)
    rows = []
    for r, d in itertools.product(R_OLDS, DDS):
        cfg = mk_cfg(r, d)
        validate_cfg(cfg, spec)
        docs = list(generate_corpus(vocab, cfg, a.docs, seed_offset=1))
        m = measure(docs, spec)
        lo, hi = dd_band(d)
        m.update(r_old=r, dd=d, dd_lo=lo, dd_hi=hi,
                 posCeil=1.0 / (hi - lo + 1))
        rows.append(m)
        print(f"  R{r:>2} D{d:>2}  slots={m['slots']:.1f} "
              f"Rreal={m['rreal']:.2f} dom={m['domain']:.2f} "
              f"tail0={m['tailUpd']:.2f} antR={m['antRatio']:.2f} "
              f"adjΔ={m['adjD']:.3f}", flush=True)

    L = ["生成器不变量（每格 %d 篇）" % a.docs, "",
         f"{'R':>3} {'ΔD':>3} {'bindMax':>8} {'r_len':>7} {'ansLast':>8} "
         f"{'updDens':>8} {'adjQ':>6} {'adjF':>6} {'adjΔ':>6} "
         f"{'antQ':>6} {'antF':>6} {'antR':>6}"]
    for x in rows:
        L.append(f"{x['r_old']:>3} {x['dd']:>3} {x['bindMax']:>8} "
                 f"{x['r_len']:>+7.3f} {x['ansLast']:>8} {x['updDens']:>8.3f} "
                 f"{x['adjQ']:>6.3f} {x['adjF']:>6.3f} {x['adjD']:>6.3f} "
                 f"{x['antQ']:>6.2f} {x['antF']:>6.2f} {x['antRatio']:>6.2f}")

    L += ["", "协变量（tab:covar）", "",
          f"{'R':>3} {'ΔD':>3} {'band':>8} {'posCeil':>8} {'slots':>7} "
          f"{'Rreal':>6} {'domain':>7} {'tail0':>6} {'nStmts':>7} {'maxTok':>7}"]
    for x in rows:
        band = f"[{x['dd_lo']},{x['dd_hi']}]"
        L.append(f"{x['r_old']:>3} {x['dd']:>3} {band:>8} "
                 f"{x['posCeil']:>8.3f} {x['slots']:>7.1f} {x['rreal']:>6.2f} "
                 f"{x['domain']:>7.2f} {x['tailUpd']:>6.2f} "
                 f"{x['nStmts']:>7.1f} {x['maxTok']:>7}")

    bad = []
    for x in rows:
        t = f"R{x['r_old']}_D{x['dd']}"
        if x["bindMax"] > 3:
            bad.append(f"{t} bindMax={x['bindMax']} >3：绑定可能可记忆")
        if abs(x["r_len"]) > 0.10:
            bad.append(f"{t} r_len={x['r_len']:+.3f}：ΔD 与长度相关")
        if x["ansLast"]:
            bad.append(f"{t} ansLast={x['ansLast']}：存在平凡捷径")
        if x["updDens"] == x["updDens"] and not 0.85 < x["updDens"] < 1.15:
            bad.append(f"{t} updDens={x['updDens']:.2f}：位置分布不均匀")
        if x["adjD"] == x["adjD"] and x["adjD"] > 0.05:
            bad.append(f"{t} adjΔ={x['adjD']:.3f}：q slot 邻接率可辨认")
        if x["antRatio"] == x["antRatio"] and not 0.9 < x["antRatio"] < 1.2:
            bad.append(f"{t} antRatio={x['antRatio']:.2f}：q slot 前驱距离可辨认")
    L += ["", "不变量判据：bindMax≤3 / |r_len|<0.10 / ansLast=0 / "
              "updDens∈(0.85,1.15) / adjΔ<0.05 / antRatio∈(0.9,1.2)"]
    L += ([""] + ["违反：" + b for b in bad]) if bad else ["", "全部通过。"]

    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(os.path.dirname(a.txt) or ".", exist_ok=True)
    with open(a.txt, "w") as f:
        f.write(txt + "\n\nraw: " + json.dumps(rows, ensure_ascii=False) + "\n")

    # tab:covar 的 LaTeX，直接贴进正文
    T = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
         r"$R_{\mathrm{old}}$ & $\Delta D$ & support & posCeil & slots & "
         r"$R_{\mathrm{real}}$ & domain & tail-0 \\", r"\midrule"]
    for x in rows:
        T.append(f"{x['r_old']} & {x['dd']} & "
                 f"$[{x['dd_lo']},{x['dd_hi']}]$ & {x['posCeil']:.3f} & "
                 f"{x['slots']:.1f} & {x['rreal']:.2f} & "
                 f"{x['domain']:.2f} & {x['tailUpd']:.2f} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    with open(a.tex, "w") as f:
        f.write("\n".join(T) + "\n")
    print(f"\n已写入 {a.txt} 与 {a.tex}")


if __name__ == "__main__":
    main()