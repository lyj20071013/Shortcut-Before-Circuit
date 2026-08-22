"""逐 run 轨迹派生量，多 seed。定义全部从 figs2.py 导入，与图 2 和
Table tab:escape 同口径。

  校准（必做一次）: python traj.py runs_g2 --seeds 0 --suffix _grid
      peak/h 两列必须复现论文表 tab:escape 的 seed 0 列
  固定带宽:        python traj.py <fixband目录> --seeds 0 1 2 --suffix _fixband

边界伪影：cosine 末段单调爬升时，峰值搜索会撞在 tail_frac 窗口右边界上，
返回衰减段内的位置而非环路形成处。主网格 seed 2 有三格如此（peak=12800
=0.8x16000）。两个独立判据都标出来，不自动删——由你看过再决定剔除哪些。
"""
import argparse, os
import numpy as np
from figs2 import read_run, deriv, escape_peak, R_ORD, D_ORD
import glob

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+", help="一个目录，或按 --seeds 顺序一个个给")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--suffix", default="_grid",
                help="tag 后缀，可含通配符。固定带宽用 '_fb*'")
ap.add_argument("--rows", type=int, nargs="+", default=R_ORD)
ap.add_argument("--cols", type=int, nargs="+", default=D_ORD)
ap.add_argument("--tail-frac", type=float, default=0.8)
ap.add_argument("--nbr", type=int, default=1000)
ap.add_argument("--interval", type=int, default=100, help="loss 记录间隔")
a = ap.parse_args()
dirs = a.dirs if len(a.dirs) == len(a.seeds) else a.dirs * len(a.seeds)

rows = []
for s, dirp in zip(a.seeds, dirs):
    for r in a.rows:
        for d in a.cols:
            hits = sorted(glob.glob(
                os.path.join(dirp, f"R{r}_D{d}_s{s}{a.suffix}.jsonl")))
            if not hits:
                print(f"MISSING R{r}_D{d}_s{s}{a.suffix}")
                continue
            if len(hits) > 1:
                print(f"AMBIGUOUS {hits}")
            p = hits[0]
            R = read_run(p)
            if not R["loss"]:
                print(f"NO LOSS {p}")
                continue
            total = R["loss"][-1][0]
            dv = deriv(R["loss"])
            cut = total * a.tail_frac
            head = [t for t in dv if t[0] <= cut]
            tail = [t for t in dv if t[0] > cut]
            pk = max(head, key=lambda t: t[1]) if head else (None, None)
            runner = (max((t[1] for t in head if abs(t[0] - pk[0]) > a.nbr),
                          default=None) if pk[0] is not None else None)
            decay = max((t[1] for t in tail), default=None)
            # 判据 1：峰位贴在窗口右边界（2 个记录间隔内）
            at_edge = pk[0] is not None and pk[0] >= cut - 2 * a.interval
            # 判据 2：末段衰减超过峰高
            decay_wins = decay is not None and pk[1] is not None and decay > pk[1]
            pre = [t for t in R["probes"] if R["esc"] and t[0] < R["esc"]]
            post = [t[1] for t in R["probes"] if R["esc"] and t[0] >= R["esc"]]
            rows.append(dict(s=s, r=r, d=d, gate=R["esc"], peak=pk[0], h=pk[1],
                             runner=runner, decay=decay, edge=at_edge,
                             dwin=decay_wins,
                             pre_frac=pre[-1][1] if pre else None,
                             term=R["probes"][-1][1] if R["probes"] else None,
                             n_post=len(post),
                             post_sd=(float(np.std(post, ddof=1))
                                      if len(post) >= 5 else None)))

def f(v, w, p=2):
    return "-".rjust(w) if v is None else f"{v:{w}.{p}f}"

print("\ns  R  dD  gate   peak      h  runner  decay  flag  preFrac"
      "   term  nPost  postSD")
for o in rows:
    flag = ("EDGE" if o["edge"] else "") + ("*" if o["dwin"] else "")
    print(f"{o['s']:<3}{o['r']:<3}{o['d']:<4}{str(o['gate']):>5}"
          f"{str(o['peak']):>7}{f(o['h'],7)}{f(o['runner'],8)}"
          f"{f(o['decay'],7)}{flag:>6}{f(o['pre_frac'],9)}"
          f"{f(o['term'],7)}{o['n_post']:>7}{f(o['post_sd'],8,3)}")
print("EDGE = 峰位贴窗口右边界（伪影）; * = 末段衰减 > 峰高（该格改用 gate）")

print("\n=== 行均值：全部 / 剔除 EDGE ===")
for s in a.seeds:
    print(f"-- seed {s}")
    for r in a.rows:
        v = [o for o in rows if o["s"] == s and o["r"] == r and o["peak"]]
        cl = [o for o in v if not o["edge"]]
        pk = f"{np.mean([o['peak'] for o in v]):.0f}" if v else "-"
        pc = f"{np.mean([o['peak'] for o in cl]):.0f}" if cl else "-"
        gt = ([o["gate"] for o in v if o["gate"]])
        print(f"  R{r:<3} peak {pk:>7} (clean {pc:>7}, n={len(cl)})"
              f"   gate {np.mean(gt):.0f}" if gt else "")

print("\n=== 跨 seed 的行区间（剔除 EDGE 后），检验层级是否重叠 ===")
for r in a.rows:
    v = [o["peak"] for o in rows if o["r"] == r and o["peak"] and not o["edge"]]
    if v:
        print(f"  R{r:<3} peak [{min(v)}, {max(v)}]  n={len(v)}")
print("  相邻行区间不重叠 => 层级跨 seed 可分")

print("\n=== 平台期伪归因 ===")
pf = [o for o in rows if o["pre_frac"] is not None]
lo20 = [o for o in pf if o["pre_frac"] < 0.20]
print(f"{len(pf)} runs with a pre-escape probe; "
      f"{len(lo20)} below 0.20, {sum(1 for o in pf if o['pre_frac'] < 0.05)} below 0.05")
print(f"of the {len(lo20)} below 0.20, "
      f"{sum(1 for o in lo20 if o['term'] and o['term'] > 0.80)} end above 0.80")
sd = [o["post_sd"] for o in rows if o["post_sd"] is not None]
if sd:
    print(f"within-run sd over {len(sd)} runs: {min(sd):.3f}-{max(sd):.3f}, "
          f"median {np.median(sd):.3f}")