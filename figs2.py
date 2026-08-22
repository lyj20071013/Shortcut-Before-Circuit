"""Fig 1 与 Fig 2 的替换版。

Fig 1 原来是 5x5 热图，对 seed 取均值。R3/D5 两 seed 是 0.13 和 0.97，
均值 0.55 恰好把要报的结论平均掉，所以整张图换掉：
  (a) 25 格配对点图，连线长度就是跨 seed 落差
  (b) 单 run 的 loss 导数与 frac+ 并置，显示符号翻转处 loss 无特征

Fig 2 左联原来用 escape_step（1000 步栅格，高 R_old 全部压在下界），
换成 loss 导数峰位（100 步分辨率）。

用法:
  python figs2.py --s0-dir runs_g2 --s1-dir /root/autodl-tmp/runs_g2_s1 \
                  --gonogo runs_g2/gonogo_s0.txt /root/autodl-tmp/runs_g2_s1/*.txt \
                  --flip-run 3 3 1 --prefix fig
"""
import argparse, glob, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R_ORD = [3, 5, 8, 12, 16]
D_ORD = [2, 3, 5, 8, 16]
RCOL = {3: "#4C72B0", 5: "#55A868", 8: "#C44E52", 12: "#8172B2", 16: "#CCB974"}


def read_run(path):
    loss, probes, esc, accs = [], [], None, []
    with open(path) as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = o.get("kind")
            if k == "train" and "loss" in o:
                loss.append((o["step"], o["loss"]))
            elif k == "eval":
                accs.append((o["step"], o.get("acc"), o.get("copy_acc")))
                if esc is None and o.get("copy_acc", 0) >= 0.95:
                    esc = o["step"]
            elif k == "probe":
                c = (o.get("causal") or {}).get("break_rarity")
                if c and c.get("frac_expected") is not None:
                    probes.append((o["step"], c["frac_expected"], c.get("d_margin")))
    loss.sort()
    probes.sort()
    accs.sort()
    return dict(loss=loss, probes=probes, esc=esc, accs=accs)


def deriv(loss):
    out = []
    for i in range(1, len(loss) - 1):
        s0, l0 = loss[i - 1]
        s1, _ = loss[i]
        s2, l2 = loss[i + 1]
        if s0 <= 0 or s2 <= 0:
            continue
        out.append((s1, (l0 - l2) / (math.log(s2) - math.log(s0))))
    return out


def escape_peak(loss, tail_frac=0.8):
    d = deriv(loss)
    if not d:
        return None
    head = [t for t in d if t[0] <= loss[-1][0] * tail_frac]
    return max(head, key=lambda t: t[1]) if head else None


def load_gonogo(paths):
    out = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                objs = []
                if line.startswith("raw:"):
                    objs = json.loads(line[4:])
                elif line.startswith("{"):
                    objs = [json.loads(line)]
                for o in objs:
                    if o.get("state") != "retr" or o.get("total_steps") != 16000:
                        continue
                    out[(o["r_old"], o["dd"], o.get("seed", 0))] = o
    return out


def fig1(gg, runs, flip_key, path):
    fig = plt.figure(figsize=(7.4, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    # ---- (a) 三 seed 点图 -----------------------------------------------
    # 连线长度是三 seed 极差（不再是两 seed 落差）。加 seed 2 之后最大极差
    # 从 0.845 (3,5) 变成 0.879 (3,8)，且 (3,5) 的三个点是 0.13/0.97/0.27,
    # 中间那个孤点靠 marker 区分，不靠连线。
    x, ticks, labels, group_edges = 0, [], [], []
    for r in R_ORD:
        start = x
        for d in D_ORD:
            cs = [gg.get((r, d, s)) for s in (0, 1, 2)]
            vs = [(s, c["frac_positive"]) for s, c in enumerate(cs) if c]
            if len(vs) < 2:
                continue
            ys = [v for _, v in vs]
            ax.plot([x, x], [min(ys), max(ys)], "-", color=RCOL[r], lw=1.4,
                    alpha=0.55, zorder=1)
            for s, v in vs:
                if s == 0:
                    ax.plot(x, v, "o", color=RCOL[r], ms=4.5, mec="none",
                            zorder=2)
                elif s == 1:
                    ax.plot(x, v, "s", color="white", ms=4.5, mec=RCOL[r],
                            mew=1.3, zorder=2)
                else:
                    ax.plot(x, v, "^", color="white", ms=5.0, mec=RCOL[r],
                            mew=1.3, zorder=2)
            ticks.append(x)
            labels.append(str(d))
            x += 1
        if x > start:
            group_edges.append((start, x - 1, r))
            x += 0.8

    ax.axhline(0.5, color="0.4", ls="--", lw=0.9, zorder=0)
    ax.set_ylim(-0.04, 1.09)
    ax.set_xlim(-0.9, x - 0.7)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("fraction of documents\nrarity-type", fontsize=8)
    ax.set_xlabel(r"$\Delta D$ within each $R_{\mathrm{old}}$ block", fontsize=8)
    ax.tick_params(labelsize=7)
    for lo, hi, r in group_edges:
        ax.text((lo + hi) / 2, 1.045, rf"$R_{{\mathrm{{old}}}}{{=}}{r}$",
                ha="center", fontsize=7, color=RCOL[r])
    h = [plt.Line2D([], [], marker="o", ls="none", color="0.3", ms=4.5,
                    label="seed 0"),
         plt.Line2D([], [], marker="s", ls="none", mfc="white", mec="0.3",
                    mew=1.3, ms=4.5, label="seed 1"),
         plt.Line2D([], [], marker="^", ls="none", mfc="white", mec="0.3",
                    mew=1.3, ms=5.0, label="seed 2")]
    ax.legend(handles=h, fontsize=6.5, frameon=False, loc="lower right",
              handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a) the readout does not replicate across three seeds",
                 fontsize=8, loc="left")

    # ---- (b) 单 run：翻转处 loss 无特征 --------------------------------
    ax = fig.add_subplot(gs[0, 1])
    R = runs[flip_key]
    d = deriv(R["loss"])
    ds = np.array([t[0] for t in d], float)
    dv = np.array([t[1] for t in d], float)
    ax.plot(ds, dv, "-", color="0.25", lw=1.0)
    ax.set_xscale("log")
    ax.set_ylabel(r"$d\,\mathrm{loss}\,/\,d\log\mathrm{step}$", fontsize=8)
    ax.set_xlabel("step", fontsize=8)
    ax.tick_params(labelsize=7)

    ax2 = ax.twinx()
    ps = [t[0] for t in R["probes"]]
    pf = [t[1] for t in R["probes"]]
    ax2.plot(ps, pf, "o-", color="#C44E52", ms=3.2, lw=1.1)
    ax2.axhline(0.5, color="#C44E52", ls=":", lw=0.8)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_ylabel("fraction rarity-type", fontsize=8, color="#C44E52")
    ax2.tick_params(labelsize=7, colors="#C44E52")

    if R["esc"]:
        ax.axvline(R["esc"], color="#4C72B0", lw=1.1)
        ax.text(R["esc"] * 1.12, ax.get_ylim()[1] * 0.92, "circuit\nforms",
                fontsize=6.5, color="#4C72B0", va="top")
    cross = None
    for i in range(1, len(R["probes"])):
        if R["probes"][i - 1][1] < 0.5 <= R["probes"][i][1]:
            cross = R["probes"][i][0]
    if cross:
        ax.axvline(cross, color="#C44E52", lw=1.1, ls="--")
        ax.text(cross * 1.12, ax.get_ylim()[1] * 0.45, "readout\ncrosses",
                fontsize=6.5, color="#C44E52", va="top")
    r, dd, s = flip_key
    ax.set_title(rf"(b) $R_{{\mathrm{{old}}}}{{=}}{r}$, $\Delta D{{=}}{dd}$, "
                 rf"seed {s}", fontsize=8, loc="left")

    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


def fig2(runs, path, tail_frac=0.8):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))

    # ---- 左：导数峰位 vs R_old -----------------------------------------
    ax = axes[0]
    for s, mk in ((0, "o"), (1, "s")):
        for r in R_ORD:
            for d in D_ORD:
                R = runs.get((r, d, s))
                if not R:
                    continue
                pk = escape_peak(R["loss"], tail_frac)
                if not pk:
                    continue
                jit = (D_ORD.index(d) - 2) * 0.13
                ax.plot(r + jit, pk[0], mk, color=RCOL[r], ms=4.5,
                        mfc=RCOL[r] if s == 0 else "white",
                        mec=RCOL[r], mew=1.2)
        xs, ys = [], []
        for r in R_ORD:
            v = [escape_peak(runs[(r, d, s)]["loss"], tail_frac)[0]
                 for d in D_ORD if (r, d, s) in runs
                 and escape_peak(runs[(r, d, s)]["loss"], tail_frac)]
            if v:
                xs.append(r)
                ys.append(float(np.mean(v)))
        ax.plot(xs, ys, "-", color="0.3", lw=1.1,
                ls="-" if s == 0 else "--", zorder=0)
    ax.axhline(1000, color="gray", ls=":", lw=1)
    ax.text(16.4, 1080, "old evaluation grid", fontsize=6.5, color="gray",
            ha="right")
    ax.set_yscale("log")
    ax.set_xticks(R_ORD)
    ax.set_xlabel(r"redundancy $R_{\mathrm{old}}$", fontsize=8)
    ax.set_ylabel("escape step (loss-derivative peak)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("(a) timing reproduces", fontsize=8, loc="left")

    # ---- 右：峰高 vs 峰位 ----------------------------------------------
    ax = axes[1]
    for s, mk in ((0, "o"), (1, "s")):
        for r in R_ORD:
            for d in D_ORD:
                R = runs.get((r, d, s))
                if not R:
                    continue
                pk = escape_peak(R["loss"], tail_frac)
                if not pk:
                    continue
                ax.plot(pk[0], pk[1], mk, color=RCOL[r], ms=4.5,
                        mfc=RCOL[r] if s == 0 else "white",
                        mec=RCOL[r], mew=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("escape step", fontsize=8)
    ax.set_ylabel("peak height", fontsize=8)
    ax.tick_params(labelsize=7)
    h = [plt.Line2D([], [], marker="o", ls="none", color=RCOL[r], ms=4.5,
                    label=rf"$R_{{\mathrm{{old}}}}{{=}}{r}$") for r in R_ORD]
    h += [plt.Line2D([], [], marker="s", ls="none", mfc="white", mec="0.3",
                     mew=1.2, ms=4.5, label="seed 1")]
    ax.legend(handles=h, fontsize=6.2, frameon=False, ncol=2,
              handletextpad=0.3, borderaxespad=0.2, loc="upper left")
    ax.set_title("(b) later escape is sharper", fontsize=8, loc="left")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0-dir", default="runs_g2")
    ap.add_argument("--s1-dir", default="/root/autodl-tmp/runs_g2_s1")
    ap.add_argument("--gonogo", nargs="*", default=[])
    ap.add_argument("--flip-run", nargs=3, type=int, default=[3, 3, 1],
                    metavar=("R", "DD", "SEED"))
    ap.add_argument("--tail-frac", type=float, default=0.8)
    ap.add_argument("--prefix", default="fig")
    a = ap.parse_args()

    runs = {}
    for s, dirp in ((0, a.s0_dir), (1, a.s1_dir)):
        for r in R_ORD:
            for d in D_ORD:
                p = os.path.join(dirp, f"R{r}_D{d}_s{s}_grid.jsonl")
                if os.path.exists(p):
                    runs[(r, d, s)] = read_run(p)
    gg = {}
    for pat in a.gonogo:
        gg.update(load_gonogo(sorted(glob.glob(pat)) or [pat]))
    print(f"{len(runs)} runs, {len(gg)} go_nogo rows")

    for n in (2, 3):
        c = sum(1 for r in R_ORD for d in D_ORD
                if sum((r, d, s) in gg for s in (0, 1, 2)) >= n)
        print(f"{c} cells have at least {n} seeds")
    rng = sorted(
        ((max(v) - min(v), r, d) for r in R_ORD for d in D_ORD
         for v in [[gg[(r, d, s)]["frac_positive"]
                    for s in (0, 1, 2) if (r, d, s) in gg]] if len(v) >= 2),
        reverse=True)
    print(f"{sum(1 for g, _, _ in rng if g > 0.3)} cells span > 0.3; "
          f"max {rng[0][0]:.3f} at R{rng[0][1]}/D{rng[0][2]}")
    
    fig1(gg, runs, tuple(a.flip_run), f"{a.prefix}1_replication.pdf")
    fig2(runs, f"{a.prefix}2_escape.pdf", a.tail_frac)


