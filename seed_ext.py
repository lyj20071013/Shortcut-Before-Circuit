"""Summarize seed-extension terminal records."""
import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# E[range]/sigma，正态样本（质量控制里的 d2）。n=3 是 Table 18 用的 1.693。
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
      7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}

def read_traj(path: str) -> List[Tuple[int, float, Optional[int]]]:
    """(step, frac_positive, escape_step) 逐探针点。jsonl 逐行 JSON。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "frac_positive" not in o:
                continue
            st = o.get("step")
            if st is None:
                continue
            out.append((int(st), float(o["frac_positive"]),
                        o.get("escape_step")))
    out.sort(key=lambda t: t[0])
    return out


def terminal_and_band(path: str, at: int, lo: int, hi: int
                      ) -> Tuple[Optional[float], Optional[float], int]:
    """终态读数、[lo,hi] 窗口内的 within-run sd、该窗口点数。

    地板用同一批 run 自己的 post-escape 漂移，而不是 App F 的 grid-wide
    最大值 —— 后者是 69 次抽样的最大值，用它做阈值等于把结论交给选择效应。
    """
    tr = read_traj(path)
    if not tr:
        return None, None, 0
    esc = next((e for (_, _, e) in tr if e), None)
    at_v = None
    for (st, fp, _) in tr:
        if st == at:
            at_v = fp
    if at_v is None:                      # 没有正好落在 at 的点，取最近的
        st, at_v, _ = min(tr, key=lambda t: abs(t[0] - at))
    win = [fp for (st, fp, _) in tr
           if lo <= st <= hi and (esc is None or st >= esc)]
    sd = float(np.std(win, ddof=1)) if len(win) >= 2 else None
    return at_v, sd, len(win)


def summarize(vals: Sequence[float], label: str) -> dict:
    """sigma 用样本 sd（n>=5）或 range/d2（n<5，兼容 Table 18 的口径）。"""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    rng = float(np.ptp(v)) if n >= 2 else float("nan")
    sd = float(np.std(v, ddof=1)) if n >= 2 else float("nan")
    d2 = D2.get(n)
    rng_sigma = (rng / d2) if d2 else float("nan")
    use_sd = n >= 5
    return dict(label=label, n=n, vals=[round(x, 4) for x in v],
                range=rng, sd=sd, range_over_d2=rng_sigma, d2=d2,
                sigma=(sd if use_sd else rng_sigma),
                sigma_from=("sample sd" if use_sd else f"range/d2(n={n})"))


def main():
    ap = argparse.ArgumentParser(
        description="种子扩展的聚合。每个 --arm 是一个条件，给若干 jsonl。")
    ap.add_argument("--arm", action="append", nargs="+", required=True,
                    metavar=("LABEL", "JSONL"),
                    help="条件名后跟该条件下每个种子的 jsonl，可重复")
    ap.add_argument("--at", type=int, default=16000, help="终态读数的步")
    ap.add_argument("--late-lo", type=int, default=8000)
    ap.add_argument("--late-hi", type=int, default=16000)
    ap.add_argument("--floor", type=float, default=None,
                    help="外部指定地板；缺省时从各臂的 within-run sd 取最大")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    arms = []
    for spec in a.arm:
        if len(spec) < 2:
            raise SystemExit(f"臂 {spec[0]} 没有给 jsonl")
        label, paths = spec[0], spec[1:]
        vals, sds = [], []
        for p in paths:
            if not os.path.exists(p):
                print(f"  ! 缺文件，跳过：{p}")
                continue
            v, sd, npts = terminal_and_band(p, a.at, a.late_lo, a.late_hi)
            if v is None:
                print(f"  ! 无读数，跳过：{p}")
                continue
            vals.append(v)
            if sd is not None:
                sds.append(sd)
        if not vals:
            print(f"  ! 臂 {label} 无可用 run")
            continue
        s = summarize(vals, label)
        s["within_run_sd"] = sds
        s["floor_self"] = max(sds) if sds else None
        arms.append(s)

    if not arms:
        raise SystemExit("没有可用数据")

    # 地板：外部指定优先；否则用所有臂 within-run sd 的最大值（cell 匹配）
    floor = a.floor
    if floor is None:
        cands = [x["floor_self"] for x in arms if x["floor_self"]]
        floor = max(cands) if cands else None

    print(f"\n终态步 {a.at}，within-run 窗口 [{a.late_lo},{a.late_hi}]")
    if floor:
        print(f"地板 {floor:.4f}（cell 匹配，取各臂 post-escape sd 的最大）")
    print(f"\n{'arm':<22} {'n':>3} {'range':>7} {'sigma':>7} {'from':>14} "
          f"{'/floor':>7}")
    for s in arms:
        ratio = (s["sigma"] / floor) if floor else float("nan")
        print(f"{s['label']:<22} {s['n']:>3} {s['range']:>7.3f} "
              f"{s['sigma']:>7.3f} {s['sigma_from']:>14} {ratio:>7.2f}")

    print("\n逐 run 值")
    for s in arms:
        print(f"  {s['label']}: {s['vals']}")

    # d2 陷阱的显式检查：若某臂 n>=5 而仍按 n=3 换算，会虚高多少
    big = [s for s in arms if s["n"] >= 5]
    if big:
        print("\nd2 检查（为什么 n>=5 不能沿用 Table 18 的 /1.693）")
        for s in big:
            wrong = s["range"] / 1.693
            print(f"  {s['label']}: 正确 {s['sigma']:.3f}（{s['sigma_from']}），"
                  f"若沿用 /1.693 则 {wrong:.3f}，虚高 "
                  f"{wrong / s['sigma']:.2f}x")

    # range 的单调性警告
    ns = {s["n"] for s in arms}
    if len(ns) > 1:
        print(f"\n各臂 n 不同（{sorted(ns)}）。range 是最大值统计量，不随采样"
              f"缩小，所以跨臂比较 range 有偏；sigma 列已换成 n 稳健的口径，"
              f"跨臂比较用它。")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(at=a.at, late=[a.late_lo, a.late_hi],
                           floor=floor, arms=arms), f, indent=1)
        print(f"\n已写入 {a.json}")


if __name__ == "__main__":
    main()
