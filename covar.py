"""Measure corpus covariates and generator invariants."""
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

def band_of(d, fixw=0):
    """fixw>0: 固定宽度带 [d, d+fixw]，切断 posCeil 与 ΔD 的共线。
    fixw=0: 主网格的 dd_band(d)，宽度随 d 增长。"""
    return (d, d + fixw) if fixw else dd_band(d)

NAN = float("nan")


def mk_cfg(r, d, seed=0, lo_st=45, hi_st=55, fixw=0,
           p_break=0.0, k_old=1, r_new=0):
    lo, hi = band_of(d, fixw)
    return CorpusCfg(name=f"R{r}_D{d}", seed=seed, p_update=0.5,
                     max_updates=1, r_old_lo=r, r_old_hi=r,
                     use_marker=False, delta_d_lo=lo, delta_d_hi=hi,
                     p_hist_query=0.0, n_stmts_lo=lo_st, n_stmts_hi=hi_st,
                     p_break=p_break, k_old_break=k_old, r_new_break=r_new,
                     truth_rule="recency")


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
    # 不变量 5 的 break 内版本。池化的 antRatio 会被 90% 的非反转文档稀释，
    # 掩盖反转文档内部的可辨识性：那才是模型能否绕过 query 定位 q slot 的
    # 实际条件。反转文档的 q_old 若走 clamp 回退（老值被压进截短窗口），
    # 其 ant 会系统偏小而填充侧不受影响，比值显著偏离 1 即存在旁路 —— 那条
    # 旁路会让 N− 对 rarity 的惩罚落在独立回路上，主回路仍欠定，读数呈假阴性。
    q_ant_bk, f_ant_bk = [], []
    for d in docs:
        a, b, c = slot_stats(d, True)
        qa_n += a
        qa_d += b
        q_ant += c
        if getattr(d, "is_break", False):
            q_ant_bk += c
        a, b, c = slot_stats(d, False)
        fa_n += a
        fa_d += b
        f_ant += c
        if getattr(d, "is_break", False):
            f_ant_bk += c
    q_adj = qa_n / qa_d if qa_d else NAN
    f_adj = fa_n / fa_d if fa_d else NAN
    out["adjQ"], out["adjF"] = q_adj, f_adj
    out["adjD"] = abs(q_adj - f_adj) if qa_d and fa_d else NAN
    mq = sum(q_ant) / len(q_ant) if q_ant else NAN
    mf = sum(f_ant) / len(f_ant) if f_ant else NAN
    out["antQ"], out["antF"] = mq, mf
    out["antRatio"] = mq / mf if f_ant and mf else NAN
    # break 内的同一比值。这是旁路泄漏的直接度量：反转文档里被查询 slot 的
    # 前驱距离若显著小于同篇填充 slot，"最后两次提及挨得最近的 slot"就是
    # 被查询 slot，模型可不读 query 定位它。池化的 antRatio 被 90% 的非反转
    # 文档稀释，看不出这件事（R16_D16 池化 0.98 而 gapBk/gapNb 是 0.32）。
    mqb = sum(q_ant_bk) / len(q_ant_bk) if q_ant_bk else NAN
    mfb = sum(f_ant_bk) / len(f_ant_bk) if f_ant_bk else NAN
    out["antQbk"], out["antFbk"] = mqb, mfb
    out["antRbk"] = mqb / mfb if f_ant_bk and mfb else NAN

    # tab:covar 的六列。旋钮 6 下 rreal 与 domain 必须按 is_break 分层：
    # q_kept 数的是 q_old 的存活者，而反转文档的 q_old 是
    # [老值×k, 末代值×(rb−1)]，两代值混在一个计数里，与非反转文档的
    # 「老值存活份数」不是同一个量。domain 用 q_kept≥2 估 break_rarity 的
    # 编辑域，而反转文档在该编辑上根本没有域（probe 的守卫直接返回 None）。
    # 池化算两者都会给出无法解释的中间值。
    nb = [d for d in docs if not getattr(d, "is_break", False)]
    bk = [d for d in docs if getattr(d, "is_break", False)]
    out["slots"] = sum(d.n_slots for d in docs) / n
    out["tailUpd"] = sum(1 for d in docs if d.n_tail_updates == 0) / n
    out["nStmts"] = sum(len(d.stmts) for d in docs) / n
    out["maxTok"] = max(len(d.tokens) for d in docs)
    out["fracBreak"] = len(bk) / n
    # 这两列只在非反转文档上算 —— 与 go_nogo 的读数域一致
    out["rreal"] = sum(d.q_kept for d in nb) / len(nb) if nb else NAN
    out["domain"] = sum(1 for d in nb if d.q_kept >= 2) / len(nb) if nb else NAN
    # 反转文档的两代值存活份数。老值份数须 ≥1，否则 N+ 的标签不在文档内
    # （生成器有断言），此处给均值供核对。
    out["brkOld"] = sum(d.n_old_kept for d in bk) / len(bk) if bk else NAN
    out["brkNew"] = sum(d.n_new_kept for d in bk) / len(bk) if bk else NAN
    # q_gap 与 clamp 率也分层。反转文档的 want 集合有两个元素（老值恰 1 份且
    # 排在 q_old[0]，拿最小位置键最易越界），重试循环的条件比非反转文档更紧，
    # 故其 q_old 位置分布被条件化。影响 q_gap 多少是实测问题，不能推断。
    out["qGapNb"] = sum(d.q_gap for d in nb) / len(nb) if nb else NAN
    out["qGapBk"] = sum(d.q_gap for d in bk) / len(bk) if bk else NAN
    # 机制模型的可验关系：q_old 的位置从 range(p_final-w_st, p_final) 均匀抽
    # len(q_old) 个、越界丢弃，故 q_gap = p_final - max(存活位置) 的期望
    # ≈ w_st / (存活数 + 1)。把三个输入量也分层印出来，这条关系式就能逐格
    # 直接验，不必靠推断。
    # 判读：若某行的 gap 与 w_st/(kept+1) 差一倍以上，说明该行走的不是均匀
    # 抽样路径（clamp 回退是唯一的另一条，它从 range(max(0,p_final-w_st),
    # p_final) 抽，等效窗口被截短到 min(w_st, p_final)）。
    out["wStNb"] = sum(d.w_st for d in nb) / len(nb) if nb else NAN
    out["wStBk"] = sum(d.w_st for d in bk) / len(bk) if bk else NAN
    out["pFinNb"] = (sum(d.n_stmts - 1 - d.realized_delta for d in nb)
                     / len(nb)) if nb else NAN
    out["pFinBk"] = (sum(d.n_stmts - 1 - d.realized_delta for d in bk)
                     / len(bk)) if bk else NAN
    # 存活数。非反转文档是 q_kept（全是老值），反转文档要用两代之和。
    out["keptNb"] = sum(d.q_kept for d in nb) / len(nb) if nb else NAN
    out["keptBk"] = (sum(d.n_old_kept + d.n_new_kept for d in bk)
                     / len(bk)) if bk else NAN
    out["clampNb"] = sum(d.q_clamped for d in nb) / len(nb) if nb else NAN
    out["clampBk"] = sum(d.q_clamped for d in bk) / len(bk) if bk else NAN

    # 不变量 6：「有 update 的 slot 中被反转的比例」在被查询侧与填充侧匹配。
    # 不匹配则「末代值被重复」本身就是定位被查询 slot 的完美判别式，模型
    # 无需读 query。无 update 的 slot 只有一代值、反转对它是空操作，不进分母；
    # 份数相等的（窗口越界丢尽副本，或 R_old=1）判读不出方向，也不进。
    q_rev = q_norm = f_rev = f_norm = 0
    for d in docs:
        by = {}
        for s in d.stmts:
            by.setdefault(slot_key(s), []).append(s)
        qk = (d.q_ent, d.q_attr)
        for k, grp in by.items():
            vals = [s.val for s in grp]
            if len(set(vals)) < 2:
                continue
            nf = vals.count(vals[-1])
            no = len(vals) - nf
            if nf == no:
                continue
            if k == qk:
                q_rev += nf > no
                q_norm += nf < no
            else:
                f_rev += nf > no
                f_norm += nf < no
    out["revQ"] = q_rev / (q_rev + q_norm) if q_rev + q_norm else NAN
    out["revF"] = f_rev / (f_rev + f_norm) if f_rev + f_norm else NAN
    out["revD"] = (abs(out["revQ"] - out["revF"])
                   if out["revQ"] == out["revQ"] and out["revF"] == out["revF"]
                   else NAN)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1500,
                    help="每格文档数。1500 与正文 §3.3 的声明一致")
    ap.add_argument("--n-values", type=int, default=512)
    ap.add_argument("--n-entities", type=int, default=200)
    ap.add_argument("--txt", default="runs_g2/covar.txt")
    ap.add_argument("--tex", default="runs_g2/covar.tex")
    ap.add_argument("--fixband", type=int, default=0,
                    help="固定带宽 W：ΔD ~ U[d, d+W]。0 = 用 dd_band(d)")
    ap.add_argument("--rows", type=int, nargs="+", default=R_OLDS)
    ap.add_argument("--cols", type=int, nargs="+", default=DDS)
    # 旋钮 6。p_break>0 时多出不变量 6（反转率在被查询侧与填充侧匹配），
    # 且 rreal / domain 两列必须按 is_break 分层 —— 见 measure() 内的说明。
    ap.add_argument("--p-break", type=float, default=0.0)
    ap.add_argument("--k-old-break", type=int, default=1)
    ap.add_argument("--r-new-break", type=int, default=0)
    a = ap.parse_args()

    spec = LangSpec(n_values=a.n_values, n_entities=a.n_entities)
    vocab = Vocab(spec)
    rows = []
    for r, d in itertools.product(a.rows, a.cols):
        cfg = mk_cfg(r, d, fixw=a.fixband, p_break=a.p_break,
                     k_old=a.k_old_break, r_new=a.r_new_break)
        validate_cfg(cfg, spec)
        docs = list(generate_corpus(vocab, cfg, a.docs, seed_offset=1))
        m = measure(docs, spec)
        lo, hi = band_of(d, a.fixband)
        m.update(r_old=r, dd=d, dd_lo=lo, dd_hi=hi,
                 posCeil=1.0 / (hi - lo + 1))
        rows.append(m)
        extra = ""
        if a.p_break > 0.0:
            extra = (f" brk={m['fracBreak']:.3f} revQ={m['revQ']:.3f} "
                     f"revF={m['revF']:.3f} revΔ={m['revD']:.3f}")
        print(f"  R{r:>2} D{d:>2}  slots={m['slots']:.1f} "
              f"Rreal={m['rreal']:.2f} dom={m['domain']:.2f} "
              f"tail0={m['tailUpd']:.2f} antR={m['antRatio']:.2f} "
              f"adjΔ={m['adjD']:.3f}" + extra, flush=True)

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

    if a.p_break > 0.0:
        L += ["", f"旋钮 6（p_break={a.p_break}）", "",
              f"{'R':>3} {'ΔD':>3} {'brk':>6} {'revQ':>6} {'revF':>6} "
              f"{'revΔ':>6} {'brkOld':>7} {'brkNew':>7} {'gapNb':>6} "
              f"{'gapBk':>6} {'clmpNb':>7} {'clmpBk':>7}"]
        for x in rows:
            L.append(f"{x['r_old']:>3} {x['dd']:>3} {x['fracBreak']:>6.3f} "
                     f"{x['revQ']:>6.3f} {x['revF']:>6.3f} {x['revD']:>6.3f} "
                     f"{x['brkOld']:>7.2f} {x['brkNew']:>7.2f} "
                     f"{x['qGapNb']:>6.2f} {x['qGapBk']:>6.2f} "
                     f"{x['clampNb']:>7.3f} {x['clampBk']:>7.3f}")
        L += ["", "位置法则的机制核对（gap 应 ≈ wSt/(kept+1)）", "",
              f"{'R':>3} {'ΔD':>3} {'wStNb':>6} {'pFinNb':>7} {'keptNb':>7} "
              f"{'gapNb':>6} {'预测':>6} {'wStBk':>6} {'pFinBk':>7} "
              f"{'keptBk':>7} {'gapBk':>6} {'预测':>6}"]
        for x in rows:
            # 分子是 p_final 而非 wSt：存活者是条件在「位置非负」之后的那些，
            # 它们均匀落在 [0, p_final) 而不是整个窗口 [p_final−wSt, p_final)。
            # 用 wSt 会系统性高 30%（R3_D5 实测 15.11 而 wSt 式给 16.84）。
            # 这仍是近似 —— 离散无放回抽样 + 条件化与 wSt 有二阶耦合，实测/预测
            # 在 D5 上是 1.18/1.18、在 D8 上是 1.08/0.93。用途是抓「差一倍以上」
            # 的路径切换（clamp 回退），不是精确预测。
            pn = (x["pFinNb"] / (x["keptNb"] + 1)
                  if x["keptNb"] == x["keptNb"] else NAN)
            pb = (x["pFinBk"] / (x["keptBk"] + 1)
                  if x["keptBk"] == x["keptBk"] else NAN)
            L.append(f"{x['r_old']:>3} {x['dd']:>3} {x['wStNb']:>6.1f} "
                     f"{x['pFinNb']:>7.1f} {x['keptNb']:>7.2f} "
                     f"{x['qGapNb']:>6.2f} {pn:>6.2f} {x['wStBk']:>6.1f} "
                     f"{x['pFinBk']:>7.1f} {x['keptBk']:>7.2f} "
                     f"{x['qGapBk']:>6.2f} {pb:>6.2f}")
        L += ["",
              "q_old 的位置从 range(p_final−wSt, p_final) 均匀抽 len(q_old) 个、",
              "越界丢弃，故 q_gap = p_final − max(存活位置) 的期望 ≈ wSt/(kept+1)。",
              "实测与预测差一倍以上 ⇒ 该行走的不是均匀抽样路径。唯一的另一条是",
              "clamp 回退（从 range(max(0,p_final−wSt), p_final) 抽，等效窗口被截短",
              "到 min(wSt, p_final)），核对 clmpNb/clmpBk 即可确认。",
              "这张表存在的理由：上一轮 R16_D16 的 gapNb=2.90/gapBk=5.19 与该关系式",
              "矛盾（按式子应当反过来），而 R3_D8 的 13.58/9.56 完全吻合。",
              "泄漏判定依赖这两个数，所以先把输入量摊开，不靠推断定案。",
              "",
              "旁路泄漏（不变量 5 的 break 内版本）", "",
              f"{'R':>3} {'ΔD':>3} {'antQbk':>7} {'antFbk':>7} {'antRbk':>7} "
              f"{'antR池化':>9}"]
        for x in rows:
            L.append(f"{x['r_old']:>3} {x['dd']:>3} {x['antQbk']:>7.2f} "
                     f"{x['antFbk']:>7.2f} {x['antRbk']:>7.3f} "
                     f"{x['antRatio']:>9.2f}")
        L += ["",
              "antRbk 是反转文档内部「被查询 slot 前驱距离 / 同篇填充 slot」。",
              "显著 <1 即存在旁路：「最后两次提及挨得最近的 slot」就是被查询",
              "slot，模型可不读 query 定位它。后果不是定位本身，而是反转文档可由",
              "一条独立回路处理，于是 N− 对 rarity 的惩罚落不到主回路上，主回路",
              "仍欠定 —— 表现为 N− 极差不收缩，会被判据表误读成「读数是噪声」。",
              "池化的 antRatio 被 90% 的非反转文档稀释，看不出这件事。",
              "根因是 clamp 回退：反转文档的 want 有两个元素，老值在 q_old[0] 拿",
              "最小位置，故「老值存活」等价于「q_old 全部位置非负」，概率",
              "(p_final/wSt)^R 随 R 指数衰减。R16 上 (33/91)^16≈9e-8，64 次重试",
              "必失败 -> clmpBk=1.000。处置是把反转改成 k+rb=R+1 的均分",
              "（R16 取 k=8,rb=9），把指数从 R 降到约 R/2。",
              "",
              "revQ/revF 是「有 update 的 slot 中被反转的比例」，被查询侧 vs",
              "填充侧。不变量 6 要求两者匹配（revΔ<0.05）：不匹配则「末代值被",
              "重复」本身就是定位被查询 slot 的完美判别式，模型无需读 query。",
              "brkOld 须 ≥1，否则 N+ 的标签不在文档内（生成器有断言拦截）。",
              "gapNb/gapBk 与 clmpNb/clmpBk 分层报告：反转文档的 want 集合有两",
              "个元素（老值恰 1 份且排在 q_old[0]，拿最小位置键最易越界），重试",
              "循环的条件比非反转文档更紧，其 q_old 位置分布被条件化。两者是否",
              "同分布是实测问题，不能由构造推断 —— 差异显著则须报告为协变量。",
              "Rreal 与 domain 两列只在非反转文档上算，与 go_nogo 的读数域一致。"]

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
        # 不变量 6。只在 p_break>0 时有定义（否则 revQ/revF 恒为 0，revD 为 0）
        if a.p_break > 0.0:
            if x["revD"] == x["revD"] and x["revD"] > 0.05:
                bad.append(f"{t} revΔ={x['revD']:.3f}：反转率在被查询侧与填充侧"
                           f"失配，「末代值被重复」是定位 q slot 的判别式")
            if x["brkOld"] == x["brkOld"] and x["brkOld"] < 1.0:
                bad.append(f"{t} brkOld={x['brkOld']:.2f} <1：反转文档的老值被"
                           f"越界丢弃，N+ 的标签不在文档内")
            # 旁路泄漏。阈值与 selfcheck 的 gapNear 判据同口径 (0.80,1.25)。
            if x["antRbk"] == x["antRbk"] and not 0.80 < x["antRbk"] < 1.25:
                bad.append(
                    f"{t} antRbk={x['antRbk']:.3f} 越界：反转文档内被查询 slot "
                    f"的前驱距离与填充 slot 不同分布，可绕过 query 定位。"
                    f"clmpBk={x['clampBk']:.3f} —— 若接近 1 则根因是 clamp 回退，"
                    f"改用 k+rb=R+1 的均分（R16 取 k=8,rb=9）")
            if x["clampBk"] == x["clampBk"] and x["clampBk"] > 0.20:
                bad.append(
                    f"{t} clmpBk={x['clampBk']:.3f} >0.20：反转文档大量走 clamp "
                    f"回退，q_old 被压进截短窗口，q_gap 系统偏小（生成器 "
                    f"docstring 点名要避免的失效模式）")
    L += ["", "不变量判据：bindMax≤3 / |r_len|<0.10 / ansLast=0 / "
              "updDens∈(0.85,1.15) / adjΔ<0.05 / antRatio∈(0.9,1.2)"]
    L += ([""] + ["违反：" + b for b in bad]) if bad else ["", "全部通过。"]

    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(os.path.dirname(a.txt) or ".", exist_ok=True)
    with open(a.txt, "w") as f:
        f.write(txt + "\n\nraw: " + json.dumps(rows, ensure_ascii=False) + "\n")

    # Render the covariate summary as LaTeX.
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