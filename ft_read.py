"""Summarize fine-tuning trajectories from saved records."""
import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

D3 = 1.693
ACC_FLOOR = 0.99
MASS_FLOOR = 0.5


def load(path: str) -> dict:
    meta, pts = None, []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("kind") == "meta":
                meta = o
            elif o.get("kind") == "probe":
                rd = (o.get("causal") or {}).get("break_rarity") or {}
                pts.append(dict(step=o["step"], zeroshot=o.get("zeroshot", False),
                                loss=o.get("loss"), copy=o.get("copy_acc"),
                                n_copy=o.get("n_copy"), **rd))
    if meta is None:
        raise ValueError(f"{path} 无 meta 记录")
    pts.sort(key=lambda x: x["step"])
    return dict(meta=meta, pts=pts, path=os.path.basename(path))


def monotone_frac(xs: Sequence[float]) -> float:
    """相邻差同号的比例。1.0 = 单调，0.5 = 随机游走，接近 0 = 锯齿振荡。

    这是区分"漂移"与"振荡"的最简判据，不需要假设趋势形状。
    """
    d = [b - a for a, b in zip(xs, xs[1:]) if a == a and b == b]
    if len(d) < 2:
        return float("nan")
    same = sum(1 for a, b in zip(d, d[1:]) if a * b > 0)
    return same / (len(d) - 1)


def stable_subset(pts: Sequence[dict]) -> Optional[List[int]]:
    """每个探针点都过 mass 门的文档下标。

    用它算 frac+ 就去掉了"门控子集成分在变"这个混淆：n_gated 从 12 涨到 123
    时，相邻点的 frac+ 差里混着"哪些文档在分母里"的变化。
    """
    keep = None
    for p in pts:
        ma = p.get("mass_all")
        if not ma:
            return None
        s = {i for i, m in enumerate(ma) if m >= MASS_FLOOR}
        keep = s if keep is None else (keep & s)
    return sorted(keep) if keep else []


def frac_on(p: dict, idx: Sequence[int]) -> float:
    d = p.get("d_all")
    if not d or not idx:
        return float("nan")
    v = [d[i] for i in idx]
    return sum(1 for x in v if x > 0) / len(v)


def sd(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mu = sum(xs) / n
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def summarize(run: dict, win: Optional[Tuple[int, int]]) -> dict:
    pts = run["pts"]
    zs = next((p for p in pts if p["zeroshot"]), None)
    tr = [p for p in pts if not p["zeroshot"]]
    form = next((p["step"] for p in tr if p.get("acc_base", 0) >= ACC_FLOOR),
                None)
    post = [p for p in tr if form is not None and p["step"] >= form]

    lo, hi = win if win else (
        (post[len(post) // 2]["step"], post[-1]["step"]) if post else (0, 0))
    wp = [p for p in post if lo <= p["step"] <= hi]

    fg = [p["frac_expected"] for p in wp
          if p.get("frac_expected") == p.get("frac_expected")]
    # 门控 Δ 中位在读窗上的中位。app:surface 的表要它 —— 先前 summarize 没
    # 输出它，于是那三格只能填 ---，而 rendered 臂的 +4.69/+5.63/+7.26 对主网格
    # 同格的 -0.02 是这张表里信息量最大的一列。
    dmw = [p["d_median"] for p in wp
           if p.get("d_median") == p.get("d_median")]
    # 固定子集在**读窗**上取交，不在全部形成后点上取。
    #
    # 交集受最小的那个 n_gated 约束，而 n_gated 在形成初期还在长（实测 12 ->
    # 28 -> 110 -> 123）。在全部形成后点上取交会被最早那个点拖到很小，而那个
    # 点并不在读窗里 —— 子集应当是"读窗内始终有定义的文档"。
    idx = stable_subset(wp) if wp else None
    fs = [frac_on(p, idx) for p in wp] if idx else []
    fs = [x for x in fs if x == x]

    return dict(
        path=run["path"], seed=run["meta"]["train"]["seed"],
        lr=run["meta"]["train"]["lr"], form=form,
        n_post=len(post), win=(lo, hi), n_win=len(wp),
        zs_frac_ungated=zs.get("frac_ungated") if zs else None,
        zs_acc=zs.get("acc_base") if zs else None,
        zs_n_gated=zs.get("n_gated") if zs else None,
        acc_end=tr[-1].get("acc_base") if tr else None,
        copy_end=tr[-1].get("copy") if tr else None,
        mass_ok_end=tr[-1].get("mass_ok") if tr else None,
        n_gated_end=tr[-1].get("n_gated") if tr else None,
        # 窗口内的门控 frac+：中位作点估计，极差与 sigma 都报。
        #
        # Report sample SD as well as range. A range depends on the
        # number of checkpoints; compare matching statistics and windows.
        f_med=(sorted(fg)[len(fg) // 2] if fg else float("nan")),
        f_band=(max(fg) - min(fg) if len(fg) > 1 else float("nan")),
        f_sd=sd(fg),
        f_end=(fg[-1] if fg else float("nan")),
        d_med=(sorted(dmw)[len(dmw) // 2] if dmw else float("nan")),
        d_band=(max(dmw) - min(dmw) if len(dmw) > 1 else float("nan")),
        f_mono=monotone_frac([p.get("frac_expected") for p in post]),
        # 固定子集版本
        n_stable=(len(idx) if idx is not None else None),
        s_med=(sorted(fs)[len(fs) // 2] if fs else float("nan")),
        s_band=(max(fs) - min(fs) if len(fs) > 1 else float("nan")),
        s_sd=sd(fs),
        # 单调性也在读窗上算，与 idx 的定义域一致
        s_mono=(monotone_frac([frac_on(p, idx) for p in wp])
                if idx else float("nan")),
        traj=[(p["step"], p.get("acc_base"), p.get("mass_ok"),
               p.get("n_gated"), p.get("frac_expected"),
               p.get("frac_ungated"), p.get("d_median")) for p in tr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs_ft")
    ap.add_argument("--win", type=int, nargs=2, default=None,
                    help="读点窗口，如 1000 2000。默认取形成后的后半段")
    ap.add_argument("--traj", action="store_true", help="打印完整轨迹")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.dir, "*.jsonl")))
    if not paths:
        raise SystemExit(f"{a.dir} 下没有 jsonl")
    runs = [summarize(load(p), tuple(a.win) if a.win else None) for p in paths]

    for r in runs:
        print(f"\n=== {r['path']}  seed {r['seed']}  lr {r['lr']} ===")
        print(f"  零样本  acc {r['zs_acc']:.4f}  frac+ (未门控) "
              f"{r['zs_frac_ungated']:.4f}  n_gated {r['zs_n_gated']}")
        print(f"  末点    acc {r['acc_end']:.4f}  copy {r['copy_end']:.4f}"
              f"  massOK {r['mass_ok_end']:.3f}  n_gated {r['n_gated_end']}")
        print(f"  形成在 step {r['form']}，形成后 {r['n_post']} 个探针点")
        print(f"  读窗 {r['win'][0]}-{r['win'][1]}（{r['n_win']} 点）")
        # sigma 与 band 并列。sigma 是与 App.~drift 的 0.069 中位可比的那个；
        # band 随点数增长，跨 run 比较时点数必须相同才有意义。
        print(f"    门控 frac+   中位 {r['f_med']:.4f}   sigma {r['f_sd']:.4f}"
              f"   band {r['f_band']:.4f}   末点 {r['f_end']:.4f}")
        print(f"    门控 Δ 中位  {r['d_med']:+.3f}   band {r['d_band']:.3f}"
              f"   （主网格同格 -0.02，见 tab:stats）")
        print(f"    单调性 {r['f_mono']:.2f}"
              f"   （1.0=单调漂移，0.5=随机游走，0=锯齿）")
        if r["n_stable"] is not None:
            print(f"    固定子集 n={r['n_stable']}（读窗内每点都过门）")
            print(f"      frac+ 中位 {r['s_med']:.4f}   sigma {r['s_sd']:.4f}"
                  f"   band {r['s_band']:.4f}   单调性 {r['s_mono']:.2f}")
            # 二项 se 是**探针集项**，不是这个 sigma 的零假设。
            #
            # 固定子集在每个探针点上是同一批文档，所以模型若真的冻住，frac+ 会
            # 逐点完全相同、sigma 恰为 0。二项 se 回答的是另一个问题："换一批
            # 同样大小的文档，frac+ 会移动多少" —— 那正是 App.~cross 量的 batch
            # 项（<=0.033）。
            #
            # 所以这个比值把方差排序里的两项并列：within-run 漂移 对 探针集
            # 有限性。两者同量级说明这个臂的形成后不稳定性已经压到探针集
            # 分辨率附近，再密的采样也读不出更多。
            if r["s_med"] == r["s_med"] and r["n_stable"]:
                p = r["s_med"]
                se = (p * (1 - p) / r["n_stable"]) ** 0.5
                ratio = r["s_sd"] / se if se else float("nan")
                print(f"      二项 se（n={r['n_stable']}, p={p:.3f}）= "
                      f"{se:.4f}   sigma/se = {ratio:.2f}")
        if a.traj:
            print(f"  {'step':>6} {'acc':>7} {'mOK':>6} {'nG':>5} "
                  f"{'frac+G':>8} {'frac+U':>8} {'ΔmedG':>9}")
            for t in r["traj"]:
                f = lambda x, w, p: (f"{x:>{w}.{p}f}" if isinstance(x, float)
                                     and x == x else f"{'-':>{w}}")
                print(f"  {t[0]:>6} {f(t[1],7,4)} {f(t[2],6,3)} "
                      f"{t[3] if t[3] is not None else '-':>5} "
                      f"{f(t[4],8,4)} {f(t[5],8,4)} {f(t[6],9,3)}")

    # 跨 seed：只在同 lr 的 run 之间比
    by_lr: Dict[float, List[dict]] = {}
    for r in runs:
        by_lr.setdefault(r["lr"], []).append(r)
    per_lr: Dict[float, dict] = {}      # 跨 lr 比较要用

    for lr, rs in sorted(by_lr.items()):
        ok = [r for r in rs if r["form"] is not None
              and r["f_med"] == r["f_med"]]
        print(f"\n--- lr {lr}：{len(ok)}/{len(rs)} 个 run 可解释"
              f"（形成且窗口内有门控读数）---")
        if len(rs) > len(ok):
            for r in rs:
                if r["form"] is None:
                    print(f"  seed {r['seed']} 丢弃：acc 从未达到 {ACC_FLOOR}"
                          f"（末点 {r['acc_end']:.4f}）—— 分不清'先验决定'与"
                          f"'预算太短'")
        # 先登记，再 continue。单 run 的组（lr 消融）也要进 per_lr，否则跨 lr
        # 比较拿不到它 —— 而那个比较是这个消融存在的全部理由。
        fm = [r["f_med"] for r in ok]
        bands = [r["f_band"] for r in ok if r["f_band"] == r["f_band"]]
        per_lr[lr] = dict(n=len(ok), meds=fm, seeds=[r["seed"] for r in ok],
                          bands=bands,
                          rng=(max(fm) - min(fm) if len(fm) > 1
                               else float("nan")))
        if len(ok) < 2:
            print(f"  单个 run（seed {[r['seed'] for r in ok]}），组内无极差可算。"
                  f"读窗中位 {fm[0]:.4f}" if ok else "  无可解释的 run")
            continue
        rng = max(fm) - min(fm)
        print(f"  跨 seed 极差 {rng:.4f}"
              f"   sigma-hat {rng / D3 if len(fm) == 3 else sd(fm):.4f}")
        if bands:
            mb = sorted(bands)[len(bands) // 2]
            print(f"  within-run band 中位 {mb:.4f}   最大 {max(bands):.4f}")
            print(f"  比值：极差 / band 中位 = "
                  f"{rng / mb if mb else float('inf'):.2f}"
                  f"，/ band 最大 = {rng / max(bands) if max(bands) else 0:.2f}")
            print("\n  判读（App.~constlr 第 1968-1971 行的结构：残差对 band")
            print("  中位 2.0 倍但对最大只有 1.10 倍，故'在典型 run 上分开、在")
            print("  最不利的 run 上没有'）")
            # 括号补全：先前写成 `if A >= 2 if B else False`，那个三元表达式的
            # 结合方式不是我想要的（它读成 `A >= (2 if B else False)`，而
            # 2 if B else False 在 B 为真时是 2、假时是 False==0，于是 band 全
            # 为零时判据变成 rng >= 0 恒真）。
            if max(bands) > 0 and rng / max(bands) >= 2:
                print("  跨 seed 分散在最不利 run 上也与漂移分开。这个臂支持")
                print("  '欠定性外推到 0.5B 微调'。")
            elif mb and rng / mb >= 2:
                print("  在典型 run 上分开、在最不利的 run 上没有。要照 constlr")
                print("  的措辞两个比值都报。")
            else:
                print("  跨 seed 极差与形成后漂移同量级 —— 这个臂分不开两者。")
                print("  不能声称 sigma 反映 seed；要如实写成'读数在这个臂上")
                print("  的不稳定性由漂移主导'。")

    # ---- 跨 lr：这个消融存在的全部理由 ----
    #
    # 两臂的 lr 无法匹配（from-scratch 用 1e-3，那会毁掉预训练权重），而
    # Compare within-seed changes across learning rates with the
    # cross-seed range at a fixed learning rate. This is descriptive
    # evidence and does not by itself identify a causal explanation.
    if len(per_lr) >= 2:
        # 主设置 = run 数最多的那组（消融通常只跑一个 seed），并列时取小的 lr。
        # 先前写的是 min(per_lr)，那假设了"主设置的 lr 更小" —— 对 2e-5 vs 5e-5
        # 成立，但若之后加一个 1e-5 的消融就反了，而错的那一侧会被当成基线、
        # 位移的符号和分母都跟着错。
        base_lr = max(per_lr, key=lambda k: (per_lr[k]["n"], -k))
        base = per_lr[base_lr]
        print(f"\n=== 跨 lr 比较（主设置 lr {base_lr}）===")
        for lr, g in sorted(per_lr.items()):
            if lr == base_lr:
                continue
            # 只在两组都有的 seed 上配对；lr 消融通常只跑 seed 0
            shared = sorted(set(base["seeds"]) & set(g["seeds"]))
            print(f"\n  lr {lr} 对 {base_lr}：共有 seed {shared}")
            for s in shared:
                a_ = base["meds"][base["seeds"].index(s)]
                b_ = g["meds"][g["seeds"].index(s)]
                print(f"    seed {s}  {a_:.4f} -> {b_:.4f}"
                      f"   位移 {b_ - a_:+.4f}")
            if not shared:
                print("    无共有 seed，无法配对 —— 消融要跑主设置里已有的 seed")
                continue
            disp = max(abs(g["meds"][g["seeds"].index(s)]
                           - base["meds"][base["seeds"].index(s)])
                       for s in shared)
            rng = base["rng"]
            if rng == rng:
                print(f"\n    最大 lr 位移 {disp:.4f}   对 lr {base_lr} 的"
                      f"跨 seed 极差 {rng:.4f}   比 "
                      f"{disp / rng if rng else float('inf'):.2f}")
                if disp > rng:
                    print("    The largest paired learning-rate displacement exceeds this seed range.")
                    print("    Report both quantities and the sampled learning rates. ")
                    print("    This descriptive comparison does not identify ")
                    print("    directions unconstrained by the training objective.")
                else:
                    print("    The paired learning-rate displacement does not exceed this seed range. ")
                    print("    This does not rule out a learning-rate contribution to the readout.")
            # 形成步数也要比：lr 大则形成早，而读窗必须在形成之后
            print(f"\n    形成步数：", end="")
            for lr2, g2 in sorted(per_lr.items()):
                fs = [r["form"] for r in by_lr[lr2] if r["form"] is not None]
                print(f"lr {lr2} -> {fs}  ", end="")
            print("\n    读窗若落在某个 run 的形成之前，那个 run 的读数无效。")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(
                runs=[{k: v for k, v in r.items() if k != "traj"}
                      for r in runs],
                per_lr={str(k): v for k, v in per_lr.items()}), f, indent=1)
        print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
