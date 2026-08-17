"""画六格的 per-doc Δ 分布（从 go_nogo 的 raw JSON 读取）。
用法：python plot_perdoc.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

RAW_TXT = "runs_g2/ctrl_table2.txt"
OUT_DIR = "runs_g2"

ORDER = ["R3/D2", "R3/D16", "R5/D5", "R16/D2", "R16/D5", "R16/D16"]
COLORS = {
    "R3/D2":   "#2166ac", "R3/D16":  "#4393c3",
    "R5/D5":   "#92c5de",
    "R16/D2":  "#d6604d", "R16/D5":  "#f4a582", "R16/D16": "#fddbc7",
}

# raw 那行在文件末尾
rows = None
with open(RAW_TXT) as f:
    for line in f:
        if line.startswith("raw: "):
            rows = json.loads(line[5:])
            break

if rows is None:
    raise RuntimeError(f"{RAW_TXT} 里没有 raw: 行，先跑 go_nogo.py")

by_label = {}
for r in rows:
    label = f"R{r['r_old']}/D{r['dd']}"
    by_label[label] = r

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()

for ax, label in zip(axes, ORDER):
    r = by_label.get(label)
    if r is None or "d_all" not in r or not r["d_all"]:
        ax.set_title(f"{label}\n(d_all 缺失)", fontsize=10)
        ax.text(0.5, 0.5, "需要在 go_nogo.py 加 d_all", ha="center",
                va="center", transform=ax.transAxes)
        continue

    ds = np.array(r["d_all"])
    med = np.median(ds)
    q25, q75 = np.percentile(ds, 25), np.percentile(ds, 75)
    frac_pos = float((ds > 0).mean())

    lo_c, hi_c = np.percentile(ds, 1), np.percentile(ds, 99)
    ds_p = ds[(ds >= lo_c) & (ds <= hi_c)]

    ax.hist(ds_p, bins=40, color=COLORS.get(label, "steelblue"),
            alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax.axvline(med, color="darkred", lw=1.5, ls="-",
               label=f"med={med:+.2f}")
    ax.set_title(
        f"{label}  n={len(ds)}\n"
        f"μ={ds.mean():+.2f}  med={med:+.2f}  "
        f"IQR=[{q25:+.2f},{q75:+.2f}]  frac+={frac_pos:.2f}",
        fontsize=9)
    ax.set_xlabel("per-doc Δ (nats)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.legend(fontsize=8)

fig.suptitle("Per-doc Δ 分布  (break_rarity，400 篇)\n"
             "虚线=0，红线=中位数", fontsize=11)
plt.tight_layout()
path = os.path.join(OUT_DIR, "perdoc.png")
plt.savefig(path, dpi=150)
print(f"已写入 {path}")