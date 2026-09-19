"""Render the final paper figures from cached main-grid records."""
import argparse, glob, json, math, os, re, sys
import matplotlib
matplotlib.use("Agg")
# Use Matplotlib's built-in mathtext. Labels such as $\Delta > 0$ are therefore
# rendered in the PDF without requiring a system LaTeX installation.
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import numpy as np

R_ORD = [3, 5, 8, 12, 16]
D_ORD = [2, 3, 5, 8, 16]
RCOL = {3: "#4C72B0", 5: "#55A868", 8: "#C44E52", 12: "#8172B2", 16: "#CCB974"}
SEED_MARKERS = {0: "o", 1: "s", 2: "^"}
SEED_LINESTYLES = {0: "-", 1: "--", 2: ":"}
GRID_TAG_RE = re.compile(r"^R(\d+)_D(\d+)_s([012])_grid$")
EXPECTED_FLAGGED = {
    (12, 8, 2), (12, 16, 2),
    (16, 2, 0), (16, 3, 0), (16, 5, 0),
    (16, 5, 2), (16, 16, 2),
}


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
                if esc is None and o.get("copy_acc", 0) > 0.95:
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
    """Centered estimate of the loss-decrease rate -d loss / d log(step)."""
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
    """Return the early candidate and whether the final-fifth maximum is higher."""
    d = deriv(loss)
    if not d:
        return None
    cut = loss[-1][0] * tail_frac
    head = [t for t in d if t[0] <= cut]
    tail = [t for t in d if t[0] > cut]
    if not head:
        return None
    peak_step, peak_h = max(head, key=lambda t: t[1])
    tail_step, tail_h = (max(tail, key=lambda t: t[1])
                         if tail else (None, float("nan")))
    return dict(peak_step=peak_step, peak_h=peak_h,
                tail_step=tail_step, tail_h=tail_h,
                flagged=bool(tail and tail_h > peak_h))


def load_gonogo(paths):
    """Load only canonical 75-run main-grid terminal rows.

    Some unrelated arms reuse ``r_old``/``dd`` fields, and legacy reports can
    carry an incorrect ``seed`` field. The canonical tag is therefore the run
    identity and also supplies the seed.
    """
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
                    match = GRID_TAG_RE.fullmatch(o.get("tag", ""))
                    if not match:
                        continue
                    r, d, s = map(int, match.groups())
                    if (o.get("state") != "retr"
                            or o.get("total_steps") != 16000
                            or o.get("step") != 16000
                            or o.get("r_old") != r or o.get("dd") != d):
                        continue
                    out[(r, d, s)] = o
    return out


def fig1(gg, runs, flip_key, path):
    fig = plt.figure(figsize=(7.4, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    # (a) Each line spans the three seeds; markers identify individual runs.
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
    ax.set_ylabel("fraction of valid edit pairs\n" r"with $\Delta>0$", fontsize=8)
    ax.set_xlabel(r"$\Delta D$ within each $R_{\mathrm{old}}$ block", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
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
    ax.set_title("(a) cross-seed variation within cells",
                fontsize=8, loc="left")

    # ---- (b) 单 run：copy gate 与后续 sign-fraction crossing -------------
    ax = fig.add_subplot(gs[0, 1])
    R = runs[flip_key]
    d = deriv(R["loss"])
    ds = np.array([t[0] for t in d], float)
    dv = np.array([t[1] for t in d], float)
    # Focus on the interval containing copy-gate passage and readout reversal.
    # Filtering before plotting also rescales the y-axis to the visible interval.
    visible = (ds >= 3000) & (ds <= 16000)
    ax.plot(ds[visible], dv[visible], "-", color="0.25", lw=1.0)
    ax.set_xlim(3000, 16000)
    ax.set_ylabel(r"$g(t)=-\mathrm{d}\ell/\mathrm{d}\log t$", fontsize=8)
    ax.set_xlabel("step", fontsize=8)
    ax.set_xticks([4000, 8000, 12000, 16000], ["4k", "8k", "12k", "16k"])
    ax.tick_params(labelsize=7)

    ax2 = ax.twinx()
    ps = [t[0] for t in R["probes"]]
    pf = [t[1] for t in R["probes"]]
    ax2.plot(ps, pf, "o-", color="#C44E52", ms=3.2, lw=1.1)
    ax2.axhline(0.5, color="#C44E52", ls=":", lw=0.8)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_ylabel(r"fraction with $\Delta > 0$", fontsize=8, color="#C44E52")
    ax2.tick_params(labelsize=7, colors="#C44E52")
    ax2.set_xlim(3000, 16000)

    if R["esc"]:
        ax.axvline(R["esc"], color="#4C72B0", lw=1.1)
        ax.text(R["esc"] * 1.03, ax.get_ylim()[1] * 0.92,
                "copy gate\ncrossed",
                fontsize=6.5, color="#4C72B0", va="top")
    cross = None
    for i in range(1, len(R["probes"])):
        if R["probes"][i - 1][1] < 0.5 <= R["probes"][i][1]:
            cross = R["probes"][i][0]
    if cross:
        ax.axvline(cross, color="#C44E52", lw=1.1, ls="--")
        ax.text(cross + 400, ax.get_ylim()[1] * 0.65,
                "sign fraction\ncrosses 0.5",
                fontsize=6.5, color="#C44E52", va="top")
    r, dd, s = flip_key
    ax.set_title(rf"(b) $R_{{\mathrm{{old}}}}{{=}}{r}$, $\Delta D{{=}}{dd}$, "
                 rf"seed {s}", fontsize=8, loc="left")

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig2(runs, path, tail_frac=0.8):
    # Reserve a shallow top band for the seed/flag legend so it cannot cover data.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05))
    peaks = {k: escape_peak(v["loss"], tail_frac) for k, v in runs.items()}
    missing_peaks = sorted(k for k, p in peaks.items() if p is None)
    if missing_peaks:
        raise ValueError(
            "training trajectories without a candidate peak: "
            + ", ".join(f"R{r}D{d}s{s}" for r, d, s in missing_peaks))
    flagged = sorted(k for k, p in peaks.items() if p["flagged"])

    # ---- 左：三 seed 的候选峰位 vs R_old -------------------------------
    ax = axes[0]
    for s, mk in SEED_MARKERS.items():
        for r in R_ORD:
            for d in D_ORD:
                pk = peaks.get((r, d, s))
                if not pk:
                    continue
                jit = (D_ORD.index(d) - 2) * 0.13
                xpos = r + jit + (s - 1) * 0.035
                ax.plot(xpos, pk["peak_step"], mk, color=RCOL[r], ms=4.5,
                        mfc=RCOL[r] if s == 0 else "white",
                        mec=RCOL[r], mew=1.2)
                if pk["flagged"]:
                    ax.plot(xpos, pk["peak_step"], "x", color="black",
                            ms=5.2, mew=1.0)
        xs, ys = [], []
        for r in R_ORD:
            v = [peaks[(r, d, s)]["peak_step"] for d in D_ORD
                 if peaks.get((r, d, s)) and not peaks[(r, d, s)]["flagged"]]
            if v:
                xs.append(r)
                ys.append(float(np.mean(v)))
        ax.plot(xs, ys, color="0.3", lw=1.1,
                ls=SEED_LINESTYLES[s], zorder=0)
    ax.axhline(1000, color="gray", ls=":", lw=1)
    ax.text(16.4, 1080, "1000-step gate grid", fontsize=6.5, color="gray",
            ha="right")
    ax.set_yscale("log")
    ax.set_xticks(R_ORD)
    ax.set_xlabel(r"nominal redundancy $R_{\mathrm{old}}$", fontsize=8)
    ax.set_ylabel("candidate peak step", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("(a) candidate-peak timing by seed", fontsize=8, loc="left")
    seed_handles = [
        plt.Line2D([], [], marker=SEED_MARKERS[s], ls=SEED_LINESTYLES[s],
                   color="0.3", mfc="0.3" if s == 0 else "white",
                   mec="0.3", mew=1.2, ms=4.5, label=f"seed {s}")
        for s in SEED_MARKERS
    ]
    seed_handles.append(
        plt.Line2D([], [], marker="x", ls="none", color="black", ms=5.2,
                   mew=1.0, label="final fifth higher"))
    fig.legend(handles=seed_handles, fontsize=5.8, frameon=False, ncol=4,
               handletextpad=0.3, columnspacing=0.8,
               loc="upper left", bbox_to_anchor=(0.075, 0.995))

    # ---- 右：三 seed 的候选峰高 vs 峰位 -------------------------------
    ax = axes[1]
    for s, mk in SEED_MARKERS.items():
        for r in R_ORD:
            for d in D_ORD:
                pk = peaks.get((r, d, s))
                if not pk:
                    continue
                ax.plot(pk["peak_step"], pk["peak_h"], mk, color=RCOL[r],
                        ms=4.5, mfc=RCOL[r] if s == 0 else "white",
                        mec=RCOL[r], mew=1.2)
                if pk["flagged"]:
                    ax.plot(pk["peak_step"], pk["peak_h"], "x", color="black",
                            ms=5.2, mew=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("candidate peak step", fontsize=8)
    ax.set_ylabel(r"candidate peak height, $g(t)$", fontsize=8)
    ax.tick_params(labelsize=7)
    color_handles = [
        plt.Line2D([], [], marker="o", ls="none", color=RCOL[r], ms=4.5,
                   label=rf"$R_{{\mathrm{{old}}}}{{=}}{r}$")
        for r in R_ORD
    ]
    ax.legend(handles=color_handles, fontsize=6.0, frameon=False, ncol=2,
              handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2,
              loc="upper left")
    ax.set_title("(b) candidate height versus timing", fontsize=8, loc="left")

    if math.isclose(tail_frac, 0.8) and set(flagged) != EXPECTED_FLAGGED:
        got = ", ".join(f"R{r}D{d}s{s}" for r, d, s in flagged)
        want = ", ".join(f"R{r}D{d}s{s}" for r, d, s in sorted(EXPECTED_FLAGGED))
        raise ValueError(f"flagged candidate mismatch: got [{got}], expected [{want}]")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("flagged candidates: "
          + ", ".join(f"R{r}D{d}s{s}" for r, d, s in flagged))
    print(f"wrote {path}")


if __name__ == "__main__":
    # Windows consoles may use a locale that cannot encode Chinese output paths.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0-dir", default="runs_g2")
    ap.add_argument("--s1-dir", default="runs_g2")
    ap.add_argument("--s2-dir", default="runs_g2")
    ap.add_argument("--gonogo", nargs="*",
                    default=["runs_g2/go_nogo.txt.jsonl"])
    ap.add_argument("--flip-run", nargs=3, type=int, default=[3, 3, 1],
                    metavar=("R", "DD", "SEED"))
    ap.add_argument("--tail-frac", type=float, default=0.8)
    ap.add_argument("--prefix", default="fig")
    a = ap.parse_args()

    runs = {}
    for s, dirp in ((0, a.s0_dir), (1, a.s1_dir), (2, a.s2_dir)):
        for r in R_ORD:
            for d in D_ORD:
                p = os.path.join(dirp, f"R{r}_D{d}_s{s}_grid.jsonl")
                if os.path.exists(p):
                    runs[(r, d, s)] = read_run(p)
    expected_runs = {(r, d, s) for r in R_ORD for d in D_ORD
                     for s in SEED_MARKERS}
    missing_runs = sorted(expected_runs - set(runs))
    if missing_runs:
        raise FileNotFoundError(
            "missing main-grid training trajectories: "
            + ", ".join(f"R{r}D{d}s{s}" for r, d, s in missing_runs))

    gg = {}
    for pat in a.gonogo:
        gg.update(load_gonogo(sorted(glob.glob(pat)) or [pat]))
    expected_gg = {(r, d, s) for r in R_ORD for d in D_ORD for s in (0, 1, 2)}
    missing_gg = sorted(expected_gg - set(gg))
    extra_gg = sorted(set(gg) - expected_gg)
    if missing_gg or extra_gg:
        missing = ", ".join(f"R{r}D{d}s{s}" for r, d, s in missing_gg)
        extra = ", ".join(f"R{r}D{d}s{s}" for r, d, s in extra_gg)
        raise ValueError(
            f"terminal grid mismatch; missing [{missing}], extra [{extra}]")
    flip_key = tuple(a.flip_run)
    if flip_key not in runs:
        raise ValueError(f"--flip-run {flip_key} has no training trajectory")
    print(f"{len(runs)} complete three-seed trajectories, "
          f"{len(expected_gg)} canonical terminal grid records")

    for n in (2, 3):
        c = sum(1 for r in R_ORD for d in D_ORD
                if sum((r, d, s) in gg for s in (0, 1, 2)) >= n)
        print(f"{c} cells have at least {n} seeds")
    rng = sorted(
        ((max(v) - min(v), r, d) for r in R_ORD for d in D_ORD
         for v in [[gg[(r, d, s)]["frac_positive"] for s in (0, 1, 2)]]),
        reverse=True)
    print(f"{sum(1 for g, _, _ in rng if g > 0.3)} cells span > 0.3; "
          f"max {rng[0][0]:.3f} at R{rng[0][1]}/D{rng[0][2]}")

    out_dir = os.path.dirname(a.prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig1(gg, runs, flip_key, f"{a.prefix}1_replication.pdf")
    fig2(runs, f"{a.prefix}2_escape.pdf", a.tail_frac)
