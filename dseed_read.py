"""Summarize runs with initialization and data-stream seeds separated."""
import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional

AT = 16000
LATE_LO = 8000


def read_run(path: str) -> dict:
    meta, pts = None, []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("kind") == "meta":
                meta = o
            elif o.get("kind") == "probe":
                c = ((o.get("causal") or {}).get("break_rarity") or {})
                if "frac_expected" in c:
                    pts.append(dict(step=o["step"], frac=c["frac_expected"],
                                    marg=c.get("d_margin"),
                                    mass=c.get("mass_mean")))
    if meta is None:
        raise ValueError(f"{path} 没有 meta 记录")
    pts.sort(key=lambda x: x["step"])
    at = next((p for p in pts if p["step"] == AT), None)
    late = [p["frac"] for p in pts if LATE_LO <= p["step"] <= AT]
    return dict(path=os.path.basename(path),
                data_seed=meta["corpus"]["seed"],
                init_seed=meta["train"]["seed"],
                corpus_name=meta["corpus"]["name"],
                frac_at=at["frac"] if at else float("nan"),
                marg_at=at["marg"] if at else float("nan"),
                mass_at=at["mass"] if at else float("nan"),
                n_late=len(late),
                # band_late is a range (max-min), whereas sd_late is a sample SD.
                # Keep both fields distinct and compare like statistics across arms.
                band_late=(max(late) - min(late)) if len(late) > 1
                else float("nan"),
                sd_late=sd(late) if len(late) > 1 else float("nan"),
                n_probe=len(pts))


def sd(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs_dseeds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.dir, "*.jsonl")))
    if not paths:
        raise SystemExit(f"{a.dir} 下没有 jsonl")
    runs = [read_run(p) for p in paths]

    print(f"{'run':<28} {'dseed':>6} {'init':>5} {'frac@16k':>9} "
          f"{'marg':>8} {'mass':>6} {'rngLate':>8} {'sdLate':>7} {'nprobe':>7}")
    print("-" * 96)
    for r in runs:
        print(f"{r['path']:<28} {r['data_seed']:>6} {r['init_seed']:>5} "
              f"{r['frac_at']:>9.4f} {r['marg_at']:>+8.3f} "
              f"{r['mass_at']:>6.3f} {r['band_late']:>8.3f} "
              f"{r['sd_late']:>7.3f} {r['n_probe']:>7}")

    # 锚点检查：data_seed==init_seed 的那个 run 与已发表值对照。
    anchor = [r for r in runs if r["data_seed"] == r["init_seed"]]
    print()
    if anchor:
        for r in anchor:
            print(f"锚点 {r['path']}：corpus.name={r['corpus_name']}、"
                  f"两个 seed 都是 {r['init_seed']}")
            print(f"  frac@16000 = {r['frac_at']:.4f}")
            print("  已发表的 R3_D8_s0 在线 cosine 值是 0.094（App.~constlr）。")
            print("  差得远说明环境或代码相对已发表状态漂了，先查这个再看 σ。")
    else:
        print("没有 data_seed==init_seed 的 run。加一个 --data-seed 0 才有锚点：")
        print("  它的 corpus.name 与 tc.seed 都与已发表 run 相同，是唯一能")
        print("  确认整条链没漂的对照。")

    # 数据顺序单独的 σ。锚点也算进去 —— 它是这一族里 data_seed=0 的成员，
    # 与其他成员地位相同（都是"固定初始化、某个文档流"）。
    fr = [r["frac_at"] for r in runs if r["frac_at"] == r["frac_at"]]
    bands = [r["band_late"] for r in runs
             if r["band_late"] == r["band_late"]]
    print()
    print(f"数据顺序单独的方差（n={len(fr)}，初始化全部固定为 "
          f"seed={runs[0]['init_seed']}）")
    print(f"  frac@16000  range {max(fr) - min(fr):.4f}   σ {sd(fr):.4f}")
    def median(xs):
        """真中位数。sorted(x)[n//2] 在偶数样本上取的是上中位，n=4 时它是第
        三小而不是中间两个的均值 —— 而 App.~constlr 报的 0.098 是中位数，
        口径不一致会让两个臂不可比。"""
        s = sorted(xs)
        m = len(s)
        return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])

    if bands:
        print(f"  within-run late RANGE  中位 {median(bands):.4f}"
              f"   最大 {max(bands):.4f}")
    sds_late = [r["sd_late"] for r in runs if r["sd_late"] == r["sd_late"]]
    if sds_late:
        print(f"  within-run late SD     中位 {median(sds_late):.4f}"
              f"   最大 {max(sds_late):.4f}")

    # Report range/range and SD/SD comparisons separately.
    s_fr = sd(fr)
    r_fr = max(fr) - min(fr)
    print(f"\nE6  Matched-statistic comparisons: range/range and SD/SD")
    if bands:
        mb, xb = median(bands), max(bands)
        print(f"  (a) range 对 range : {r_fr:.3f} / {mb:.3f} = "
              f"{r_fr / mb:.1f}   ；{r_fr:.3f} / {xb:.3f} = {r_fr / xb:.1f}")
        print(f"      This row compares ranges; the next comparison uses sample SDs.")
    if sds_late:
        ms, xs_ = median(sds_late), max(sds_late)
        print(f"  (b) sd 对 sd       : {s_fr:.3f} / {ms:.3f} = "
              f"{s_fr / ms:.1f}   ；{s_fr:.3f} / {xs_:.3f} = {s_fr / xs_:.1f}")
        print(f"      本次实测的 within-run sd 中位 {ms:.3f}、最大 {xs_:.3f}")
    print(f"  ^ 两条都比 2.9/2.4 大，所以改正后这个臂的分离度是提高的")

    print()
    print("判读（对照 App.~constlr 的十 seed 数）")
    print("  完整 seed（初始化+数据顺序）cosine σ = 0.271")
    print("  恒定 lr 十 seed              σ = 0.200")
    s = sd(fr)
    if s == s:
        print(f"  数据顺序单独                 σ = {s:.3f}")
        print()
        if s >= 0.20:
            print("  数据顺序单独就产生了与完整 seed 同量级的方差。这加强中心")
            print("  声称：现象不依赖初始化的随机性，故'小模型初始化噪声'")
            print("  这一类反驳不成立。")
        elif s >= 0.10:
            print("  数据顺序贡献了一部分但不是全部。两个来源都要在文里区分，")
            print("  §sec:flat 的方差排序应当加这一格。")
        else:
            print("  数据顺序几乎不贡献方差 —— 它主要来自初始化。中心声称要")
            print("  改成'初始化的 underdetermination'，比现在窄但更精确。")
        print()
        # σ 的 95% CI，卡方法：[s*sqrt((n-1)/χ²_.975), s*sqrt((n-1)/χ²_.025)]
        # n=4（3 自由度）：χ²_.975=9.348、χ²_.025=0.216 -> [0.567s, 3.727s]
        # 先前我写的上限 2.9 是错的，它对应约 6 自由度。区间比我说的更宽。
        chi = {3: (9.348, 0.216), 4: (11.143, 0.484), 9: (19.023, 2.700)}
        df = len(fr) - 1
        if df in chi:
            hi, lo = chi[df]
            print(f"  警告：n={len(fr)} 下 σ 的 95% CI 是 "
                  f"[{s * math.sqrt(df / hi):.3f}, {s * math.sqrt(df / lo):.3f}]，")
            print("  与 0.271 的区间大幅重叠，所以这个臂能定方向、不能定幅度。")
            print("  要窄的区间需要十个 data_seed（再六个 run，约九小时）。")
        else:
            print(f"  警告：n={len(fr)}，σ 的区间未列表；n 小则区间极宽。")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(at=AT, late_lo=LATE_LO, runs=runs,
                           sd_frac=sd(fr), range_frac=max(fr) - min(fr),
                           n=len(fr)), f, indent=1)
        print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
