
"""Fig 2 替代：逃逸的连续轨迹。
逃逸判定是阈值 + 1k 栅格，正是 Schaeffer et al. 2023 批评的配方。
train.py 每 100 步记 loss，用它画 d(loss)/d(log step) 说明陡峭程度不是
离散化产物。左panel：R3 五格 copy_acc 轨迹（逃逸步分散）；
右panel：R3/D5 的 loss 与其对数导数（连续、无阈值）。"""
import json, math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D_ORD = [2, 3, 5, 8, 16]

def read(path):
    tr, ev = [], []
    for line in open(path):
        o = json.loads(line)
        k = o.get("kind")
        if k == "train" and "loss" in o:
            tr.append((o["step"], o["loss"]))
        elif k == "eval" and o.get("copy_acc") is not None:
            ev.append((o["step"], o["copy_acc"], o.get("acc")))
    return sorted(tr), sorted(ev)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.6))
cmap = plt.get_cmap("viridis")

for j, d in enumerate(D_ORD):
    try:
        _, ev = read(f"runs_g2/R3_D{d}_s0_grid.jsonl")
    except FileNotFoundError:
        continue
    xs = [s for s, _, _ in ev]
    ys = [c for _, c, _ in ev]
    ax1.plot(xs, ys, "o-", color=cmap(j / 4), ms=3, lw=1.1,
             label=rf"$\Delta D\!=\!{d}$")
ax1.axhline(0.95, color="crimson", ls=":", lw=1)
ax1.text(15500, 0.90, "gate", fontsize=7, color="crimson", ha="right")
ax1.set_xlabel("step")
ax1.set_ylabel("copy diagnostic")
ax1.set_title(r"$R_{\mathrm{old}}=3$: circuit formation", fontsize=9)
ax1.legend(fontsize=7, frameon=False, loc="center right")

tr, ev = read("runs_g2/R3_D5_s0_grid.jsonl")
st = np.array([s for s, _ in tr], dtype=float)
ls = np.array([l for _, l in tr], dtype=float)
m = st > 0
ax2.plot(st[m], ls[m], color="0.25", lw=1.0)
ax2.set_xlabel("step")
ax2.set_ylabel("training loss", color="0.25")
ax2.set_xscale("log")

ax3 = ax2.twinx()
lg = np.log(st[m])
dl = np.gradient(ls[m], lg)
ax3.plot(st[m], -dl, color="steelblue", lw=1.0, alpha=0.8)
ax3.set_ylabel(r"$-\mathrm{d}\,\mathrm{loss}/\mathrm{d}\log\,\mathrm{step}$",
               color="steelblue", fontsize=8)
ax3.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

esc = next((s for s, c, _ in ev if c and c >= 0.95), None)
if esc:
    ax2.axvline(esc, color="crimson", ls="--", lw=1)
    ax2.text(esc * 1.1, ls[m].max() * 0.9, f"gate at {esc}",
             fontsize=7, color="crimson")
ax2.set_title(r"$R_{\mathrm{old}}=3$, $\Delta D=5$: unthresholded", fontsize=9)

fig.tight_layout()
fig.savefig("fig_traj.pdf", dpi=200, bbox_inches="tight")
print("wrote fig_traj.pdf")
