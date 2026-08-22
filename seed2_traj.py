"""seed 2 的轨迹派生量。定义全部从 figs2.py 导入，保证与图 2 和
Table tab:escape 同一口径。

  先校准:  python seed2_traj.py runs_g2 0     # 应复现表 17 的 seed 0 peak/h 两列
  再用:    python seed2_traj.py /root/autodl-tmp/runs_g2_s2 2
"""
import os, sys
import numpy as np
from figs2 import read_run, deriv, escape_peak, R_ORD, D_ORD

DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/runs_g2_s2"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 2
TAIL, NBR = 0.8, 1000

rows = []
for r in R_ORD:
    for d in D_ORD:
        p = os.path.join(DIR, f"R{r}_D{d}_s{SEED}_grid.jsonl")
        if not os.path.exists(p):
            print(f"MISSING {p}")
            continue
        R = read_run(p)
        if not R["loss"]:
            print(f"NO LOSS {p}")
            continue
        total = R["loss"][-1][0]
        dv = deriv(R["loss"])
        head = [t for t in dv if t[0] <= total * TAIL]
        tail = [t for t in dv if t[0] > total * TAIL]
        pk = max(head, key=lambda t: t[1]) if head else (None, None)
        # 次高值：排除峰位 +/-NBR 的邻域。用来核 "exactly one interior peak"。
        runner = (max((t[1] for t in head if abs(t[0] - pk[0]) > NBR),
                      default=None) if pk[0] is not None else None)
        decay = max((t[1] for t in tail), default=None)
        pre = [t for t in R["probes"] if R["esc"] and t[0] < R["esc"]]
        post = [t[1] for t in R["probes"] if R["esc"] and t[0] >= R["esc"]]
        rows.append(dict(r=r, d=d, gate=R["esc"], peak=pk[0], h=pk[1],
                         runner=runner, decay=decay,
                         pre_step=pre[-1][0] if pre else None,
                         pre_frac=pre[-1][1] if pre else None,
                         n_post=len(post),
                         post_sd=(float(np.std(post, ddof=1))
                                  if len(post) >= 5 else None)))

def f(v, w, p=2):
    return "-".rjust(w) if v is None else f"{v:{w}.{p}f}"

print("R   dD  gate   peak      h  runner  decay  preStep  preFrac  nPost  postSD")
for o in rows:
    print(f"{o['r']:<4}{o['d']:<4}{str(o['gate']):>5}{str(o['peak']):>7}"
          f"{f(o['h'],7)}{f(o['runner'],8)}{f(o['decay'],7)}"
          f"{str(o['pre_step']):>9}{f(o['pre_frac'],9)}"
          f"{o['n_post']:>7}{f(o['post_sd'],8,3)}")

print()
for r in R_ORD:
    pv = [o["peak"] for o in rows if o["r"] == r and o["peak"]]
    gv = [o["gate"] for o in rows if o["r"] == r and o["gate"]]
    if pv:
        print(f"R{r:<3} peak row mean {np.mean(pv):7.0f}"
              f"   gate row mean {np.mean(gv):7.0f}")

pf = [o["pre_frac"] for o in rows if o["pre_frac"] is not None]
print(f"\npre-escape frac+: {len(pf)} runs, "
      f"{sum(1 for v in pf if v < 0.20)} below 0.20, "
      f"{sum(1 for v in pf if v < 0.05)} below 0.05")
sd = [o["post_sd"] for o in rows if o["post_sd"] is not None]
if sd:
    print(f"post-formation within-run sd over {len(sd)} runs: "
          f"{min(sd):.3f}-{max(sd):.3f}, median {np.median(sd):.3f}")
ok = [o for o in rows if o["h"] and o["runner"] and o["h"] > 2 * o["runner"]]
print(f"peak > 2x next local max outside +/-{NBR} steps in {len(ok)}/{len(rows)} runs")