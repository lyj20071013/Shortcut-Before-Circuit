
"""逐篇 Δ 的分布形状。中位数与符号比例会藏双峰：若某格是「少数强正 +
多数弱负」两簇，结论从「弱 frequency 型」变成「同格内不同文档用不同规则」，
摘要措辞要跟着改。所以下笔前必须先看这张图。"""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--bins", type=int, default=25)
a = ap.parse_args()

for line in open(a.path):
    o = json.loads(line)
    xs = sorted(o["d_all"])
    n = len(xs)
    if not n:
        continue
    lo, hi = xs[0], xs[-1]
    w = (hi - lo) / a.bins or 1.0
    cnt = [0] * a.bins
    for x in xs:
        cnt[min(a.bins - 1, int((x - lo) / w))] += 1
    mx = max(cnt)
    npos = sum(1 for x in xs if x > 0)
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    print(f"\n{o['tag']}  n={n}  median={med:+.3f}  "
          f"frac+={npos / n:.3f}  mean={sum(xs) / n:+.3f}")
    print(f"  range [{lo:+.2f}, {hi:+.2f}]")
    for i, c in enumerate(cnt):
        a0, a1 = lo + i * w, lo + (i + 1) * w
        bar = "#" * int(40 * c / mx) if mx else ""
        zero = " <0" if a0 <= 0 < a1 else ""
        print(f"  [{a0:+6.2f},{a1:+6.2f}) {c:>4} {bar}{zero}")
    # 峰间谷：若最大与次大峰之间存在明显低谷，中位数在掩盖两个机制
    peaks = [i for i in range(1, a.bins - 1)
             if cnt[i] > cnt[i - 1] and cnt[i] > cnt[i + 1] and cnt[i] > n / 50]
    print(f"  局部峰 {len(peaks)} 个" + ("  ← 可能双峰，看图确认" if len(peaks) > 1 else ""))
