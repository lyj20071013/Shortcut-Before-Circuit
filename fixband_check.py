"""固定带宽 arm 的上卡前预检。只用 CPU，不碰模型。

拦三类问题：
  1 结构退化。ΔD 上界抬高会压缩 q_old 可用窗口，2R_old/W 超过主网格最坏格
    （0.78）就意味着 q slot 可由离散度定位，整条臂的结论作废。这是本臂特有的
    风险，主网格的 dd_band 下不会出现。
  2 App A 的五条生成器不变量在新带宽下是否仍然成立。它们在主网格上验过，
    换了 ΔD 支撑就得重验，不能假定继承。
  3 编辑域。ΔD 抬高会丢弃更多越界副本，域跌破 0.5 时该格是在不到一半的
    文档上取读数，与主网格不可比。

每格给一行 OK / WARN / FAIL。有 FAIL 时退出码非零，可直接作为 runner 的门。
"""
import argparse
import math
import random
import sys
from collections import Counter, defaultdict

from config import CorpusCfg, LangSpec, validate_cfg
from generator import generate_corpus
from probe import apply_edit, fit_position_offset
from vocab import Vocab

# 阈值。括号里是主网格实测值，门限留出 1500 篇的采样噪声。
GATE = dict(
    rlen=0.06,          # App A |r| ≤ 0.043，se≈0.026
    adj_gap=0.05,       # App A 匹配到 0.042
    ant_ratio=1.35,     # App A 最大 1.22，本臂预期更高，超 1.35 视为失控
    clamp=0.01,         # q_clamped 应 ≪1%
    degen=0.90,         # 2R/W，主网格最坏格 0.78
    domain=0.50,        # 主网格最低 0.66
    dd_ratio=1.70,      # ΔD 支撑均匀性，9 桶 1500 篇的 3σ 约 1.6
    bindmax=6,          # App A 实测 ≤4
)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def check_cell(r, lo, hi, spec, n_docs, stmts_lo, stmts_hi, seed=0):
    cfg = CorpusCfg(name=f"R{r}_D{lo}_fb", seed=seed, p_update=0.5,
                    max_updates=1, r_old_lo=r, r_old_hi=r, use_marker=False,
                    delta_d_lo=lo, delta_d_hi=hi, p_hist_query=0.0,
                    n_stmts_lo=stmts_lo, n_stmts_hi=stmts_hi)
    validate_cfg(cfg, spec)                 # 配置期硬校验，不过就直接炸
    vocab = Vocab(spec)
    docs = list(generate_corpus(vocab, cfg, n_docs, seed_offset=1))
    offset = fit_position_offset(docs)
    rng = random.Random(0)

    # --- 结构退化：最坏情形的解析值 + 实测窗口 ---
    p_min = stmts_lo - 1 - hi
    degen = 2.0 * r / p_min if p_min > 0 else float("inf")
    win = [min(d.n_stmts - 1 - d.realized_delta, d.w_st) for d in docs]

    # --- App A 的不变量 ---
    rlen = pearson([d.realized_delta for d in docs],
                   [len(d.tokens) for d in docs])
    ansLast = sum(1 for d in docs
                  if [t for t in d.tokens[:d.answer_pos] if vocab.is_val(t)][-1]
                  == d.answer)
    adj_q = sum(d.adj_q for d in docs) / len(docs)
    adj_f = [d.adj_fill for d in docs if d.adj_fill == d.adj_fill]
    adj_f = sum(adj_f) / len(adj_f) if adj_f else float("nan")
    ant_q = sum(d.q_gap for d in docs) / len(docs)
    ant_f = [d.fill_gap_late for d in docs if d.fill_gap_late == d.fill_gap_late]
    ant_f = sum(ant_f) / len(ant_f) if ant_f else float("nan")
    ratio = ant_q / ant_f if ant_f == ant_f and ant_f > 0 else float("nan")

    trip = defaultdict(set)
    for i, d in enumerate(docs):
        for s in d.stmts:
            trip[(s.ent, s.attr, s.val)].add(i)
    bindmax = max(len(v) for v in trip.values())

    # --- ΔD 支撑均匀性（App D 的第二个等式靠它精确成立）---
    cnt = Counter(d.realized_delta for d in docs)
    exp = set(range(lo, hi + 1))
    missing = exp - set(cnt)
    hits = [cnt.get(k, 0) for k in sorted(exp)]
    dd_ratio = max(hits) / min(hits) if min(hits) > 0 else float("inf")

    # --- 编辑域：apply_edit 全部不变量重查后真正可用的比例 ---
    dom = sum(1 for d in docs
              if apply_edit(d, "break_rarity", vocab, cfg, rng, offset) is not None)

    row = dict(
        r=r, band=(lo, hi), width=hi - lo + 1, posCeil=1.0 / (hi - lo + 1),
        p_final_min=p_min, degen=degen,
        win_mean=sum(win) / len(win),
        q_kept=sum(d.q_kept for d in docs) / len(docs),
        clamp=sum(d.q_clamped for d in docs) / len(docs),
        slots=sum(d.n_slots for d in docs) / len(docs),
        tokens=sum(len(d.tokens) for d in docs) / len(docs),
        rlen=rlen, ansLast=ansLast, adj_q=adj_q, adj_f=adj_f,
        ant_q=ant_q, ant_f=ant_f, ant_ratio=ratio, bindmax=bindmax,
        dd_ratio=dd_ratio, dd_missing=sorted(missing),
        domain=dom / len(docs),
        tail0=sum(1 for d in docs if d.n_tail_updates == 0) / len(docs),
    )

    fail, warn = [], []
    if row["degen"] > GATE["degen"]:
        fail.append(f"退化 2R/W={row['degen']:.2f}>{GATE['degen']}")
    elif row["degen"] > 0.78:
        warn.append(f"2R/W={row['degen']:.2f} 已达主网格最坏格")
    if row["ansLast"]:
        fail.append(f"ansLast={row['ansLast']}≠0")
    if abs(row["rlen"]) > GATE["rlen"]:
        fail.append(f"rlen={row['rlen']:+.3f}")
    if abs(row["adj_q"] - row["adj_f"]) > GATE["adj_gap"]:
        fail.append(f"adj 差 {row['adj_q'] - row['adj_f']:+.3f}")
    if row["ant_ratio"] > GATE["ant_ratio"]:
        fail.append(f"ant 比 {row['ant_ratio']:.2f}")
    elif row["ant_ratio"] > 1.25:
        warn.append(f"ant 比 {row['ant_ratio']:.2f} 超主网格 1.22")
    if row["clamp"] > GATE["clamp"]:
        fail.append(f"clamp={row['clamp']:.3f}")
    if row["domain"] < GATE["domain"]:
        fail.append(f"域={row['domain']:.2f}")
    if row["dd_missing"]:
        fail.append(f"ΔD 缺值 {row['dd_missing']}")
    if row["dd_ratio"] > GATE["dd_ratio"]:
        warn.append(f"ΔD 不均匀 {row['dd_ratio']:.2f}")
    if row["bindmax"] > GATE["bindmax"]:
        fail.append(f"bindMax={row['bindmax']}")
    row["fail"], row["warn"] = fail, warn
    return row


def main():
    ap = argparse.ArgumentParser(description="固定带宽 arm 预检（CPU）")
    ap.add_argument("--r", type=int, nargs="+", default=[3])
    ap.add_argument("--lo", type=int, nargs="+", default=[1, 4, 8, 16],
                    help="各带的下界，宽度由 --width 决定")
    ap.add_argument("--width", type=int, default=9,
                    help="恒定支撑宽度，posCeil=1/width")
    ap.add_argument("--docs", type=int, default=1500, help="与 App A 同量")
    ap.add_argument("--stmts-lo", type=int, default=45)
    ap.add_argument("--stmts-hi", type=int, default=55)
    ap.add_argument("--n-values", type=int, default=512)
    ap.add_argument("--n-entities", type=int, default=200)
    a = ap.parse_args()

    spec = LangSpec(n_values=a.n_values, n_entities=a.n_entities)
    print(f"posCeil 恒为 {1.0 / a.width:.3f}（宽 {a.width}）  "
          f"n_stmts~U[{a.stmts_lo},{a.stmts_hi}]  {a.docs} 篇/格\n")
    hdr = (f"{'R':>3} {'band':>9} {'W':>4} {'2R/W':>6} {'qkept':>6} "
           f"{'slots':>6} {'tok':>5} {'域':>5} {'rlen':>7} "
           f"{'adjq/f':>12} {'antq/f':>12} {'比':>5} {'tail0':>6}")
    print(hdr)
    print("-" * len(hdr))
    bad = 0
    for r in a.r:
        for lo in a.lo:
            row = check_cell(r, lo, lo + a.width - 1, spec, a.docs,
                             a.stmts_lo, a.stmts_hi)
            print(f"{row['r']:>3} {str(row['band']):>9} "
                  f"{row['win_mean']:>4.0f} {row['degen']:>6.2f} "
                  f"{row['q_kept']:>6.2f} {row['slots']:>6.1f} "
                  f"{row['tokens']:>5.0f} {row['domain']:>5.2f} "
                  f"{row['rlen']:>+7.3f} "
                  f"{row['adj_q']:>5.3f}/{row['adj_f']:<6.3f} "
                  f"{row['ant_q']:>5.2f}/{row['ant_f']:<6.2f} "
                  f"{row['ant_ratio']:>5.2f} {row['tail0']:>6.2f}")
            for m in row["fail"]:
                print(f"      FAIL {m}")
                bad += 1
            for m in row["warn"]:
                print(f"      WARN {m}")
    print()
    if bad:
        print(f"{bad} 处 FAIL，不要上卡。")
        sys.exit(1)
    print("全部通过。")


if __name__ == "__main__":
    main()