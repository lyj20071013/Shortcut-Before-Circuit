"""Summarize constant-learning-rate trajectories from saved records."""
import argparse
import json
import os
from typing import List, Optional, Sequence

import numpy as np

from figs2 import read_run

# 已发表参照。cos 16000 的 go_nogo 终态值（Table tab:spread）
PUB_COS = {0: 0.098, 1: 0.477, 2: 0.977}
# App J.6：R3_D8, p_break=0.03, const, 16000 步, 三种子
J6_CONST_RELABEL = (0.653, 0.500, 0.486)
# App F：post-formation within-run sd 的中位数与最大值，69 run
APPF_BAND_MED, APPF_BAND_MAX = 0.069, 0.303

def at_step(probes: Sequence[tuple], step: int,
            tol: int = 0) -> Optional[tuple]:
    """恰好 step 处的探针点。tol>0 时取最近的一个并返回其真实步号。

    默认 tol=0：要的是"逐比特等同于一个 16000 步 const run"那个点，取近似
    会悄悄换成另一个模型状态。找不到就返回 None，由调用方报缺。
    """
    for s, f, m in probes:
        if s == step:
            return (s, f, m)
    if tol <= 0:
        return None
    near = [(abs(s - step), s, f, m) for s, f, m in probes
            if abs(s - step) <= tol]
    if not near:
        return None
    _, s, f, m = min(near)
    return (s, f, m)


def band(probes: Sequence[tuple], esc: Optional[int],
         lo: Optional[int] = None, hi=None) -> dict:
    """within-run band。

    post: 逃逸后全部探针点的 sd，与 App F 的 post_sd 同口径（ddof=1，<5 点
          不报）。这是唯一能与 0.069/0.303 直接比的量。
    late: 再限制到 step >= lo 的版本。逃逸后紧邻的几个点仍在快速移动，
          把它们算进去会把 band 撑大，于是"极差落在 band 内"这个判据变得
          过于容易通过。两个都报，别只报有利的那个。
    """
    post = [f for s, f, _ in probes if esc is not None and s >= esc]
    late = [f for s, f, _ in probes
            if esc is not None and s >= esc
            and (lo is None or s >= lo) and (hi is None or s <= hi)]
    def sd(xs):
        return float(np.std(xs, ddof=1)) if len(xs) >= 5 else None

    return dict(n_post=len(post), sd_post=sd(post),
                n_late=len(late), sd_late=sd(late),
                rng_post=(max(post) - min(post)) if post else None,
                rng_late=(max(late) - min(late)) if late else None)


def read_one(path: str, at: int, late_lo: int, late_hi: int) -> dict:
    R = read_run(path)
    pr = R["probes"]
    hit = at_step(pr, at)
    return dict(path=path, esc=R["esc"], n_probe=len(pr),
                steps=[s for s, _, _ in pr],
                frac_at=(hit[1] if hit else None),
                marg_at=(hit[2] if hit else None),
                frac_term=(pr[-1][1] if pr else None),
                step_term=(pr[-1][0] if pr else None),
                **band(pr, R["esc"], late_lo, late_hi))


def rng(xs: Sequence[Optional[float]]) -> Optional[float]:
    v = [x for x in xs if x is not None]
    return (max(v) - min(v)) if len(v) >= 2 else None


def band_of(o: dict) -> Optional[float]:
    """一个 run 的 within-run band。late 优先，点数不足时退回 post。

    late 要求 >=5 个探针点（band 的 sd 门），16000 步的新 seed 恰好卡在 5，
    所以退回逻辑必须存在，否则新 seed 的 band 全是 None。
    """
    return o.get("sd_late") if o.get("sd_late") is not None else o.get("sd_post")


def paired(C: dict, S: dict) -> dict:
    """配对差 const - cos，同 seed 同 16000 步。

    这是本臂的主要收获。跨 seed 的 range 只说明「换 schedule 后离散度还在」；
    配对差说明「schedule 对每个 run 做了什么」，而 §5.5 阶梯第三级现在只有
    三点（+0.749/+0.013/-0.188），靠它从轶事变成一个分布。

    每个差都对自己那对 run 的 band 取最大值作阈：一个小于两个 run 各自晚期
    漂移的差，与漂移不可区分。这是逐 seed 的判据，不是池化的。
    """
    rows = []
    for s in sorted(set(C) & set(S)):
        c, o = C[s], S[s]
        cf, of = c.get("frac_at"), o.get("frac_at")
        if cf is None or of is None:
            continue
        bc, bo = band_of(c), band_of(o)
        b = max([x for x in (bc, bo) if x is not None], default=None)
        d = cf - of
        rows.append(dict(seed=s, const=cf, cos=of, diff=d, band=b,
                         over=(b is not None and abs(d) > b)))
    ds = [r["diff"] for r in rows]
    n = len(ds)
    out = dict(rows=rows, n=n)
    if n:
        m = sum(ds) / n
        out["mean"] = m
        out["median"] = sorted(ds)[n // 2] if n % 2 else \
            0.5 * (sorted(ds)[n // 2 - 1] + sorted(ds)[n // 2])
        out["sd"] = (sum((x - m) ** 2 for x in ds) / (n - 1)) ** 0.5 if n > 1 else None
        k = sum(x > 0 for x in ds)
        out["n_pos"] = k
        out["frac_pos"] = k / n
        out["n_over"] = sum(r["over"] for r in rows)
        # 双侧符号检验：schedule 是否有一致方向的效应
        from math import comb
        tail = sum(comb(n, i) for i in range(n + 1)
                   if abs(i - n / 2) >= abs(k - n / 2))
        out["sign_p"] = min(1.0, tail / (2 ** n))
    return out


def fmt(v, p=3, w=7):
    return "-".rjust(w) if v is None else f"{v:{w}.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const-dir", default="runs_constlr")
    ap.add_argument("--const-suffix", default="_constlr")
    ap.add_argument("--cos-dir", default="runs_g2",
                    help="已发表主网格所在目录，用于在线探针对在线探针")
    ap.add_argument("--cos-suffix", default="_grid")
    ap.add_argument("--cos-dir2", default="runs_g2_seeds",
                    help="cos 侧的备用目录。十 seed 扩展（App tenseed）的 "
                         "seed 3-9 在这里，且文件名无后缀")
    ap.add_argument("--cos-suffix2", default="",
                    help="备用目录的后缀，默认空")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--at", type=int, default=16000,
                    help="预算匹配的比较点。const 下该点等同独立 run")
    ap.add_argument("--late-lo", type=int, default=8000,
                    help="late band 的下界")
    ap.add_argument("--late-hi", type=int, default=None,
                    help="late band 上界")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    def path_of(d, suf, s):
        return os.path.join(d, f"R{a.r}_D{a.d}_s{s}{suf}.jsonl")

    def cos_path(s):
        """cos 侧的十个 seed 分两处：seed 0-2 是主网格 runs_g2/*_grid.jsonl，
        seed 3-9 是十 seed 扩展 runs_g2_seeds/*.jsonl（无后缀）。两处的 meta
        已核为同配置（sched=cos, 16000 步, d_model 512, n_layer 8, 26.05M）。
        主路径优先，找不到就试备用，都没有才报缺 —— 否则配对差会静默地只有
        三个，而那正是本臂要补的东西。"""
        p = path_of(a.cos_dir, a.cos_suffix, s)
        if os.path.exists(p):
            return p
        alt = path_of(a.cos_dir2, a.cos_suffix2, s)
        return alt if os.path.exists(alt) else p

    C, S = {}, {}
    for s in a.seeds:
        pc = path_of(a.const_dir, a.const_suffix, s)
        if os.path.exists(pc):
            C[s] = read_one(pc, a.at, a.late_lo, a.late_hi)
        else:
            print(f"MISSING {pc}")
        ps = cos_path(s)
        if os.path.exists(ps):
            S[s] = read_one(ps, a.at, a.late_lo, a.late_hi)
        else:
            print(f"MISSING cos {ps}")

    print(f"\ncell R{a.r}_D{a.d}   const={a.const_dir}{a.const_suffix}  "
          f"cos={a.cos_dir}{a.cos_suffix}")

    print(f"\n--- 探针轨迹（const）---")
    for s in sorted(C):
        o = C[s]
        pr = read_run(o["path"])["probes"]
        late = [(st, f) for st, f, _ in pr if st >= a.late_lo]
        print(f"seed {s}  esc={o['esc']}  {o['n_probe']} 个探针点")
        print("  step>=%d: " % a.late_lo
              + " ".join(f"{st//1000}k:{f:.2f}" for st, f in late))

    print(f"\n--- 1. 预算匹配点 step {a.at}（const 下等同独立 16000 步 run）---")
    print(f"{'seed':>5} {'const@16k':>10} {'cos@16k(online)':>16} "
          f"{'cos published':>14} {'const@term':>11}")
    for s in sorted(set(C) | set(S)):
        c = C.get(s, {})
        o = S.get(s, {})
        print(f"{s:>5} {fmt(c.get('frac_at'),3,10)} "
              f"{fmt(o.get('frac_at'),3,16)} "
              f"{fmt(PUB_COS.get(s),3,14)} {fmt(c.get('frac_term'),3,11)}")

    r_c16 = rng([C[s].get("frac_at") for s in sorted(C)])
    r_s16 = rng([S[s].get("frac_at") for s in sorted(S)])
    r_pub = rng([PUB_COS[s] for s in sorted(C) if s in PUB_COS])
    r_cte = rng([C[s].get("frac_term") for s in sorted(C)])
    n = len(C)
    print(f"\n  range over {n} seeds, 在线探针对在线探针：")
    print(f"    const @ {a.at}      {fmt(r_c16)}")
    print(f"    cos   @ {a.at}      {fmt(r_s16)}   <- 同口径对照")
    print(f"    cos published       {fmt(r_pub)}   (go_nogo n=400)")
    print(f"    const @ terminal    {fmt(r_cte)}")

    # 直算 SD：与论文新口径一致，不再用 range/d_n 转换
    def sd_of(xs):
        v = [x for x in xs if x is not None]
        if len(v) < 2:
            return None
        m = sum(v) / len(v)
        return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5

    s_c16 = sd_of([C[s].get("frac_at") for s in sorted(C)])
    s_s16 = sd_of([S[s].get("frac_at") for s in sorted(S)])
    print(f"\n  直算 SD（n 匹配才可比）：")
    print(f"    const @ {a.at}  n={len(C):>2}  {fmt(s_c16)}")
    print(f"    cos   @ {a.at}  n={len(S):>2}  {fmt(s_s16)}")
    if s_c16 and s_s16:
        print(f"    比值 const/cos  {s_c16 / s_s16:.3f}"
              f"   <- schedule 解释掉的份额 = 1 - 比值")

    P = paired(C, S)
    print(f"\n--- 1b. 配对差 const - cos（同 seed，只差 schedule）---")
    if not P["n"]:
        print("  无可配对的 seed。cos 侧缺 16000 探针点或文件未找到。")
    else:
        print(f"{'seed':>5} {'const':>8} {'cos':>8} {'diff':>8} "
              f"{'band':>8} {'超band':>7}")
        for r in P["rows"]:
            print(f"{r['seed']:>5} {r['const']:>8.3f} {r['cos']:>8.3f} "
                  f"{r['diff']:>+8.3f} {fmt(r['band'],3,8)} "
                  f"{'是' if r['over'] else '否':>7}")
        print(f"\n  n={P['n']}  中位 {P['median']:+.3f}  均值 {P['mean']:+.3f}"
              + (f" ± {P['sd']:.3f}" if P.get("sd") else ""))
        print(f"  正 {P['n_pos']}/{P['n']}（{P['frac_pos']:.2f}）"
              f"  双侧符号检验 p={P['sign_p']:.3f}")
        print(f"  超出自身 band 的 {P['n_over']}/{P['n']}")
        print(f"  Archived seed 0-2 reference differences: +0.749 / +0.013 / -0.188.")
        if P["sign_p"] > 0.05:
            print(f"  p>0.05: the sign test does not reject its null; "
                  f"this does not establish absence of a directional effect (n={P['n']}).")
        else:
            print(f"  p<=0.05: nominal sign-test evidence under its assumptions; "
                  f"cell selection and multiplicity are not addressed here.")

    print(f"\n--- 2. within-run band ---")
    print(f"{'seed':>5} {'nPost':>6} {'sdPost':>8} {'rngPost':>8} "
          f"{'nLate':>6} {'sdLate':>8} {'rngLate':>8}")
    for s in sorted(C):
        o = C[s]
        print(f"{s:>5} {o['n_post']:>6} {fmt(o['sd_post'],3,8)} "
              f"{fmt(o['rng_post'],3,8)} {o['n_late']:>6} "
              f"{fmt(o['sd_late'],3,8)} {fmt(o['rng_late'],3,8)}")
    print(f"  App F 参照（cos, 69 run）：sd 中位 {APPF_BAND_MED:.3f}，"
          f"最大 {APPF_BAND_MAX:.3f}")

    # Distinguish late/post-formation SD from range. The printed
    # reference matches identify the statistic used by each archived value.
    def med(xs):
        s = sorted(xs)
        m = len(s)
        return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])

    print(f"\nE4/E5  Archived 0.098 / 0.182 references versus SD/range columns")
    print(f"{'column':>10}{'median':>9}{'max':>9}   verdict")
    for col in ("sd_post", "rng_post", "sd_late", "rng_late"):
        vs = [C[s][col] for s in sorted(C) if C[s].get(col) is not None]
        if not vs:
            print(f"{col:>10}{'-':>9}{'-':>9}   (无值)")
            continue
        m_, x_ = med(vs), max(vs)
        hit = (abs(m_ - 0.098) < 5e-4, abs(x_ - 0.182) < 5e-4)
        tag = ("中位命中 0.098 " if hit[0] else "") + \
              ("最大命中 0.182" if hit[1] else "")
        print(f"{col:>10}{m_:>9.3f}{x_:>9.3f}   {tag or '—'}")
    print(f"  ^ Compare SD with SD and range with range.")
    print(f"    The label band alone does not identify the statistic.")
    print(f"    Interpret a ratio only after identifying both of its statistics.")

    # APPF_BAND_MAX is a saved reference, not a computed quantity.
    # The current main-grid SD is available from paper_numbers.py.
    print(f"\nX1  本文件的 APPF_BAND_MAX = {APPF_BAND_MAX:.3f}（硬编码）")
    print(f"  paper_numbers.py reports the main-grid maximum within-run SD.")
    print(f"  The saved reference constant should be compared with that output.")
    print(f"  注意 ddof：本文件 sd() 用 ddof=1，paper_numbers.py 已同步为 1。")

    print(f"\n--- 3. 与 App J.6 的 const-16000 relabelled 臂对比 ---")
    r_j6 = max(J6_CONST_RELABEL) - min(J6_CONST_RELABEL)
    print(f"  relabelled (p_break=0.03, const, 16000, 3 seeds): "
          f"{J6_CONST_RELABEL} range {r_j6:.3f}")
    print(f"  aliased    (p_break=0,    const, {a.at}, {n} seeds): "
          f"range {fmt(r_c16)}")
    print(f"  注意样本量不同：range 随种子数单调不减，两种子对三种子偏小。")

    # ---- 判读 ----
    print("\n" + "=" * 66)
    sds = [C[s]["sd_late"] for s in sorted(C)
           if C[s].get("sd_late") is not None]
    b = max(sds) if sds else None
    if r_c16 is None:
        print(f"缺 step {a.at} 的探针点。探针点集见上方轨迹；"
              f"若 {a.at} 不在其中，用 --at 换一个在集合里的步号，"
              f"或用 --at 配合 tol 修改 at_step 调用。")
    elif b is None:
        print("晚期探针点不足 5 个，band 不可估。先确认 --late-lo 是否过高。")
    else:
        print(f"const @ {a.at}: range {r_c16:.3f}，late band 最大 {b:.3f}")
        if r_c16 <= b:
            print("  极差落在单 run 自身的晚期漂移带内 -> 这个臂测不出跨 run")
            print("  差异，因为 run 自己就在这个范围里动。极差收窄不能读成收敛。")
            print("  Interpret these readouts together with checkpoint and schedule; ")
            print("  this comparison does not establish convergence.")
        elif r_pub is not None and r_c16 < 0.5 * r_pub:
            print(f"  Range {r_c16:.3f} exceeds band {b:.3f} and is below half the ")
            print(f"  paired-seed reference range {r_pub:.3f}; this is descriptive.")
            print("  The schedule changes the observed range in this comparison. ")
            print("  It does not establish optimizer convergence or an ordering of variance sources.")
        else:
            print(f"  极差 {r_c16:.3f} 超出 band {b:.3f} 且与已发表同量级")
            print(f"  ({r_pub if r_pub else float('nan'):.3f}) -> 换 schedule、")
            print("  The range persists in this comparison without annealing. ")
            print("  This does not exclude all optimization-based explanations.")
    print("=" * 66)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(cell=[a.r, a.d], at=a.at, late_lo=a.late_lo,
                           const={str(k): v for k, v in C.items()},
                           cos={str(k): v for k, v in S.items()},
                           range_const_at=r_c16, range_cos_at=r_s16,
                           range_published=r_pub,
                           range_const_term=r_cte,
                           sd_const_at=s_c16, sd_cos_at=s_s16,
                           paired=P), f, indent=1)
        print(f"\n已写入 {a.json}")


if __name__ == "__main__":
    main()
