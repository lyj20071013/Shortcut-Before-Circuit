#!/usr/bin/env python3
"""
plot_continuous.py -- continuous-metric escape trajectories.

Answers the "is the transition a metric artifact?" objection
(Schaeffer et al. 2023) by showing the escape is sharp in continuous,
unthresholded metrics (loss and d loss / d log step), not only in the
thresholded copy_acc gate.

  python plot_continuous.py --runs runs_g2 --cells R3_D2,R16_D2 --seeds 1
  python plot_continuous.py --runs runs_g2 --schema-only   # dump keys, exit
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_STEP = ["step", "global_step", "it", "iter", "iteration"]
K_LOSS = ["loss", "train_loss", "tr_loss", "loss_train"]
K_COPY = ["copy_acc", "copyAcc", "copy", "copy_accuracy"]
K_ACC  = ["acc", "seq_acc", "accuracy", "exact_acc"]


def pick(d, names):
    for n in names:
        v = d.get(n)
        if isinstance(v, (int, float)) and not isinstance(v, bool) \
           and not (isinstance(v, float) and np.isnan(v)):
            return n
    return None


def load_run(jsonl):
    rows = []
    with open(jsonl) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue                        # tolerate truncated tail
    if not rows:
        return None

    kstep = next((pick(r, K_STEP) for r in rows if pick(r, K_STEP)), None)
    if kstep is None:
        return None

    keys = {"step": kstep, "loss": None, "acc": None, "copy": None}
    tr_s, tr_l, ev_s, ev_a, ev_c = [], [], [], [], []
    for r in rows:
        if kstep not in r:
            continue
        s = float(r[kstep])
        kl = pick(r, K_LOSS)
        if kl:
            keys["loss"] = keys["loss"] or kl
            tr_s.append(s); tr_l.append(float(r[kl]))
        ka, kc = pick(r, K_ACC), pick(r, K_COPY)
        if ka or kc:
            keys["acc"] = keys["acc"] or ka
            keys["copy"] = keys["copy"] or kc
            ev_s.append(s)
            ev_a.append(float(r[ka]) if ka else np.nan)
            ev_c.append(float(r[kc]) if kc else np.nan)

    def srt(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.size == 0:
            return x, y
        o = np.argsort(x, kind="stable")
        x, y = x[o], y[o]
        keep = np.ones(x.size, bool)          # FIX 3: drop duplicate steps
        keep[1:] = np.diff(x) > 0
        return x[keep], y[keep]

    tr_s, tr_l = srt(tr_s, tr_l)
    o = np.argsort(np.asarray(ev_s, float), kind="stable") if ev_s else []
    ev_s = np.asarray(ev_s, float)[o]
    ev_a = np.asarray(ev_a, float)[o]
    ev_c = np.asarray(ev_c, float)[o]
    return dict(keys=keys, tr_s=tr_s, tr_l=tr_l, ev_s=ev_s, ev_a=ev_a, ev_c=ev_c)


def label_of(path):
    """(R, D, seed, label). Regex first; then any *.json in the run dir."""
    for name in (path.parent.name, path.name):
        m = re.search(r"[Rr](?:old)?[_-]?(\d+)[_-]*[Dd]{1,2}[_-]?(\d+)", name)
        if m:
            ms = re.search(r"(?:seed|s)[_-]?(\d+)", name)
            R, D = int(m.group(1)), int(m.group(2))
            sd = int(ms.group(1)) if ms else None
            return R, D, sd, f"R{R}/D{D}"
    for cand in sorted(path.parent.glob("*.json")):      # FIX 6: config fallback
        try:
            cfg = json.loads(cand.read_text())
        except Exception:
            continue
        R = cfg.get("r_old", cfg.get("R_old", cfg.get("rold")))
        D = cfg.get("dd_lo", cfg.get("delta_d", cfg.get("dd")))
        if R is not None and D is not None:
            return int(R), int(D), cfg.get("seed"), f"R{int(R)}/D{int(D)}"
    return None, None, None, path.parent.name


def escape_step(ev_s, ev_c, thr):
    idx = np.where(ev_c >= thr)[0]
    if not len(idx):
        return None, None
    i = idx[0]
    return float(ev_s[i]), (float(ev_s[i - 1]) if i > 0 else None)


def dloss_dlogstep(s, l, win=9):
    m = s > 0
    s, l = s[m], l[m]
    if s.size < win + 2:
        return s, np.full(s.shape, np.nan)
    ls = np.convolve(l, np.ones(win) / win, mode="same")
    dv = np.gradient(ls, np.log10(s))
    e = win // 2
    dv[:e] = np.nan; dv[-e:] = np.nan
    return s, dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_g2")
    ap.add_argument("--out", default="fig_escape")
    ap.add_argument("--glob", default="**/*.jsonl")
    ap.add_argument("--cells", default="", help="substring filter, comma-separated")
    ap.add_argument("--seeds", default="", help="keep only these seeds, e.g. 1")
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--schema-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.runs)
    if not root.exists():
        sys.exit(f"no such dir: {root}")
    files = sorted(root.glob(args.glob))
    if args.cells:
        want = [c.strip() for c in args.cells.split(",") if c.strip()]
        files = [f for f in files if any(w in str(f) for w in want)]
    if not files:
        sys.exit(f"no jsonl under {root} matching {args.glob}")

    keep_seeds = {int(x) for x in args.seeds.split(",") if x.strip()} if args.seeds else None

    runs = []
    for f in files:
        d = load_run(f)
        if d is None or d["ev_s"].size == 0:
            print(f"  skip (no step key / no eval rows): {f}")
            continue
        R, D, sd, lab = label_of(f)
        if keep_seeds and sd is not None and sd not in keep_seeds:
            continue
        d.update(path=f, R=R, D=D, seed=sd, lab=lab)
        runs.append(d)
    if not runs:
        sys.exit("nothing loaded -- check --glob, or edit the K_* aliases at top")

    print(f"loaded {len(runs)} runs")
    print(f"detected keys: {runs[0]['keys']}")
    if args.schema_only:
        return
    if runs[0]["keys"]["loss"] is None:                    # FIX 5: loud, not silent
        print("!! no loss key found -- panels (c)/(d) will be EMPTY, and those "
              "are the metric-artifact rebuttal. Add your key to K_LOSS.")
    if runs[0]["keys"]["copy"] is None:
        print("!! no copy_acc key found -- no escape steps will be marked. "
              "Add your key to K_COPY.")

    rolds = sorted({r["R"] for r in runs if r["R"] is not None})
    cmap = plt.get_cmap("viridis")
    col = {R: cmap(i / max(1, len(rolds) - 1)) for i, R in enumerate(rolds)}

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    ax_acc, ax_cp, ax_ls, ax_dv = axes.ravel()        # FIX 4: no `d` shadowing
    seen = set()                                      # FIX 1/2: dedupe legend

    hdr = f"{'cell':>10} {'seed':>5} {'esc':>8} {'prev':>8} {'cp@prev':>8} {'cp@esc':>8} {'acc_f':>7}"
    print("\n" + hdr)
    for r in runs:
        c = col.get(r["R"], "0.5")
        lab = r["lab"] if r["lab"] not in seen else None
        seen.add(r["lab"])

        ax_acc.plot(r["ev_s"], r["ev_a"], "-o", ms=2.5, lw=1.0, color=c, alpha=.85)
        ax_cp.plot(r["ev_s"], r["ev_c"], "-o", ms=2.5, lw=1.0, color=c, alpha=.85, label=lab)
        if r["tr_s"].size:
            ax_ls.plot(r["tr_s"], r["tr_l"], lw=1.0, color=c, alpha=.85)
            xs, dv = dloss_dlogstep(r["tr_s"], r["tr_l"])
            ax_dv.plot(xs, dv, lw=1.0, color=c, alpha=.85)

        esc, prev = escape_step(r["ev_s"], r["ev_c"], args.thr)
        if esc is not None:
            for a in axes.ravel():
                a.axvline(esc, color=c, ls=":", lw=.7, alpha=.55)

        def at(s):
            if s is None:
                return float("nan")
            i = np.where(r["ev_s"] == s)[0]
            return r["ev_c"][i[0]] if i.size else float("nan")

        print(f"{r['lab']:>10} {str(r['seed']):>5} {str(esc):>8} {str(prev):>8} "
              f"{at(prev):>8.3f} {at(esc):>8.3f} {r['ev_a'][-1]:>7.3f}")

    for a, t in ((ax_acc, "(a) sequence accuracy"),
                 (ax_cp, f"(b) copy diagnostic (gate {args.thr})"),
                 (ax_ls, "(c) training loss  [continuous, unthresholded]"),
                 (ax_dv, r"(d) $d\,\mathrm{loss}/d\log_{10}\mathrm{step}$")):
        a.set_title(t, fontsize=9)
        a.set_xscale("log")
        a.set_xlabel("step", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=.25, lw=.4)
    ax_cp.axhline(args.thr, color="k", ls="--", lw=.6)
    if seen:
        ax_cp.legend(fontsize=6, ncol=2, loc="lower right", framealpha=.85)

    fig.suptitle("Escape is sharp in continuous metrics, not only in the thresholded gate",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=180 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()