
"""生成正文三张图。相图用 frac+ 而非中位数：后者受 mass 天花板压缩
（R12/D5 的 mass=0.66，逐篇门把中位数移了 33%），而 frac+ 不受影响
且跨预算跨 seed 可比。"""
import argparse, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R_ORD = [3, 5, 8, 12, 16]
D_ORD = [2, 3, 5, 8, 16]

def load(paths):
    """go_nogo 的 .txt 是人读报表，机器数据在末尾 `raw: [...]` 一行；
    .jsonl 缓存是逐行 JSON。两种都吃。"""
    out = {}
    for p in paths:
        objs = []
        for line in open(p):
            line = line.strip()
            if line.startswith("raw:"):
                objs.extend(json.loads(line[4:]))
            elif line.startswith("{"):
                objs.append(json.loads(line))
        for o in objs:
            if o.get("state") != "retr":
                continue
            if o.get("total_steps") != 16000:
                continue          # 不同预算不可混入同一张图
            out[(o["r_old"], o["dd"], o.get("seed", 0))] = o
    return out

def fig_phase(rows, path):
    """Fig 1：相图。格内标 frac+，颜色以 0.5 为中点。
    R3 行离 chance 只有 0.15 而 R16 行有 0.50，纯颜色会让读者以为低端无效应；
    实际那五格 p 从 2e-3 到 1e-54。所以显著格加边框，让符号翻转在视觉上对称。"""
    n_seed = len({k[2] for k in rows})
    grid = np.full((len(R_ORD), len(D_ORD)), np.nan)
    text = [["" for _ in D_ORD] for _ in R_ORD]
    sig = np.zeros((len(R_ORD), len(D_ORD)), dtype=bool)
    for i, r in enumerate(R_ORD):
        for j, d in enumerate(D_ORD):
            cs = [rows[(r, d, s)] for s in range(n_seed) if (r, d, s) in rows]
            if not cs:
                continue
            fs = [c["frac_positive"] for c in cs]
            grid[i, j] = float(np.mean(fs))
            text[i][j] = (f"{np.mean(fs):.2f}" if len(fs) == 1
                          else f"{np.mean(fs):.2f}\n$\\pm${np.ptp(fs) / 2:.2f}")
            sig[i, j] = all(c["sign_p"] < 0.05 for c in cs)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=0.0, vmax=1.0,
                   origin="lower", aspect="auto")
    for i in range(len(R_ORD)):
        for j in range(len(D_ORD)):
            if not text[i][j]:
                continue
            v = grid[i, j]
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=8,
                    color="white" if abs(v - 0.5) > 0.32 else "black")
            if sig[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, lw=1.6,
                                           edgecolor="0.15", zorder=3))
    ax.set_xticks(range(len(D_ORD)))
    ax.set_xticklabels([f"{d}" for d in D_ORD])
    ax.set_yticks(range(len(R_ORD)))
    ax.set_yticklabels([f"{r}" for r in R_ORD])
    ax.set_xlabel(r"update-to-query distance $\Delta D$ (nominal)")
    ax.set_ylabel(r"redundancy $R_{\mathrm{old}}$")
    cb = fig.colorbar(im, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("fraction of documents rarity-type", fontsize=9)
    cb.ax.axhline(0.5, color="k", lw=1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")

def fig_escape(rows, path):
    """Fig 2：逃逸步。横轴 R_old，颜色区分带宽。1k 是评估栅格下界，标出来。"""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    cmap = plt.get_cmap("viridis")
    for j, d in enumerate(D_ORD):
        xs, ys = [], []
        for i, r in enumerate(R_ORD):
            es = [rows[(r, d, s)]["escape_step"]
                  for s in {k[2] for k in rows} if (r, d, s) in rows]
            es = [e for e in es if e]
            if not es:
                continue
            xs.append(r + (j - 2) * 0.18)
            ys.append(float(np.mean(es)))
        ax.plot(xs, ys, "o-", color=cmap(j / 4), ms=5, lw=1.2,
                label=rf"$\Delta D\!=\!{d}$")
    ax.axhline(1000, color="gray", ls=":", lw=1)
    ax.text(15.4, 1080, "evaluation grid", fontsize=7, color="gray", ha="right")
    ax.set_yscale("log")
    ax.set_xticks(R_ORD)
    ax.set_xlabel(r"redundancy $R_{\mathrm{old}}$")
    ax.set_ylabel("escape step")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")

def fig_dist(perdoc, path, cells=((3, 5), (8, 5), (16, 5))):
    """Fig 3：三格逐篇分布。同一个编辑在两侧的响应形状不同 ——
    负侧薄尾、正侧厚尾，这是中位数看不出来的。"""
    by = {}
    for line in open(perdoc):
        o = json.loads(line)
        t = o["tag"]
        for r in R_ORD:
            for d in D_ORD:
                if t.startswith(f"R{r}_D{d}_"):
                    by[(r, d)] = o["d_all"]
    fig, axes = plt.subplots(1, len(cells), figsize=(7.6, 2.4))

    for ax, (r, d) in zip(axes, cells):
        xs = by.get((r, d))
        if not xs:
            ax.set_visible(False)
            continue
        lo, hi = np.percentile(xs, [0.5, 99.5])
        ax.hist(xs, bins=40, range=(min(lo, -0.5), max(hi, 0.5)),
                color="0.35", edgecolor="none")
        ax.axvline(0, color="crimson", lw=1)
        med = float(np.median(xs))
        ax.axvline(med, color="steelblue", lw=1, ls="--")
        fp = sum(1 for x in xs if x > 0) / len(xs)
        ax.set_title(rf"$R_{{\mathrm{{old}}}}\!=\!{r}$, $\Delta D\!=\!{d}$"
                     "\n" rf"median {med:+.2f}, frac$+$ {fp:.2f}", fontsize=8)
        ax.set_xlabel(r"$\Delta$ (nats)", fontsize=8)
        ax.set_ylabel("documents", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("documents", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gonogo", nargs="+")
    ap.add_argument("--perdoc", default=None)
    ap.add_argument("--prefix", default="fig")
    a = ap.parse_args()
    rows = load(a.gonogo)
    print(f"{len(rows)} cells, seeds {sorted({k[2] for k in rows})}")
    fig_phase(rows, f"{a.prefix}_phase.pdf")
    fig_escape(rows, f"{a.prefix}_escape.pdf")
    if a.perdoc:
        fig_dist(a.perdoc, f"{a.prefix}_dist.pdf")


# python figs.py runs_g2/gonogo_s0.txt runs_g2/gonogo_strat.txt \
# --perdoc runs_g2/gonogo_s0.txt.perdoc.jsonl