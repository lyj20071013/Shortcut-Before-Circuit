"""论文里每个行均值与占位数字的唯一来源，按论文位置分节输出。

截图读数不进论文。所有 \PL{} 标记的数从这里取。

用法:
  python numbers.py \
    --s0-dir runs_g2 \
    --s1-dir /root/autodl-tmp/runs_g2_s1 \
    --gonogo runs_g2/gonogo_s0.txt /root/autodl-tmp/runs_g2_s1/*.txt

口径说明:
  终态 frac+ 取自 go_nogo（n=400，权威）。
  逃逸前 frac+ 取自训练内嵌探针（n=200，是训练时唯一存在的读数）。
  两者在同格上差 <0.03（见 --check-consistency）。
"""
import argparse, glob, json, math, os
import numpy as np

R_ORD = [3, 5, 8, 12, 16]
D_ORD = [2, 3, 5, 8, 16]


def read_run(path):
    """训练 jsonl -> (loss 序列, 探针序列, 门判逃逸步)。"""
    loss, probes, esc = [], [], None
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
                if esc is None and o.get("copy_acc", 0) >= 0.95:
                    esc = o["step"]
            elif k == "probe":
                c = (o.get("causal") or {}).get("break_rarity")
                if c and c.get("frac_expected") is not None:
                    probes.append((o["step"], c["frac_expected"], c.get("d_margin")))
    loss.sort()
    probes.sort()
    return loss, probes, esc


def deriv(loss):
    """d loss / d log step，中心差分。返回 [(step, value)]。"""
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
    """内部极大值，排除余弦末段衰减。

    tail_frac=0.8 是因为高 R_old 格的最大导数落在 14800-15900，
    那是调度尾巴不是相变（见 §6.2 的限制段）。
    同时返回被排除区间的最大值，用来量化那条限制。
    """
    d = deriv(loss)
    if not d:
        return None
    cut = loss[-1][0] * tail_frac
    head = [t for t in d if t[0] <= cut]
    tail = [t for t in d if t[0] > cut]
    if not head:
        return None
    pk = max(head, key=lambda t: t[1])
    tl = max(tail, key=lambda t: t[1]) if tail else (None, float("nan"))
    others = [v for s, v in head if abs(s - pk[0]) > 500]
    return {
        "peak_step": pk[0],
        "peak_h": pk[1],
        "max_elsewhere": max(others) if others else float("nan"),
        "tail_step": tl[0],
        "tail_h": tl[1],
    }


def load_gonogo(paths):
    """go_nogo 报表 -> {(r, d, seed): row}。只取 16k 主网格。"""
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
                    if o.get("state") != "retr":
                        continue
                    if o.get("total_steps") != 16000:
                        continue
                    out[(o["r_old"], o["dd"], o.get("seed", 0))] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0-dir", default="runs_g2")
    ap.add_argument("--s1-dir", default="/root/autodl-tmp/runs_g2_s1")
    ap.add_argument("--gonogo", nargs="*", default=[])
    ap.add_argument("--tail-frac", type=float, default=0.8)
    a = ap.parse_args()

    src = {0: a.s0_dir, 1: a.s1_dir}
    runs = {}
    for s, d in src.items():
        for r in R_ORD:
            for dd in D_ORD:
                p = os.path.join(d, f"R{r}_D{dd}_s{s}_grid.jsonl")
                if os.path.exists(p):
                    runs[(r, dd, s)] = read_run(p)

    gg = {}
    for pat in a.gonogo:
        gg.update(load_gonogo(sorted(glob.glob(pat)) or [pat]))

    print(f"# {len(runs)} training runs, {len(gg)} go_nogo rows\n")

    # ---------------------------------------------------------------- §5.2
    print("=" * 68)
    print("§5.2  frac+ row means (go_nogo, n=400)")
    print("=" * 68)
    if gg:
        print(f"{'R_old':>6} {'seed 0':>9} {'seed 1':>9} {'|diff|':>8}  n_cells")
        for r in R_ORD:
            m = {}
            for s in (0, 1):
                v = [gg[(r, d, s)]["frac_positive"] for d in D_ORD if (r, d, s) in gg]
                m[s] = float(np.mean(v)) if v else float("nan")
            print(f"{r:>6} {m[0]:>9.3f} {m[1]:>9.3f} {abs(m[0]-m[1]):>8.3f}"
                  f"  {sum(1 for d in D_ORD if (r,d,0) in gg)}")
        diffs = []
        for r in R_ORD:
            for d in D_ORD:
                if (r, d, 0) in gg and (r, d, 1) in gg:
                    diffs.append(abs(gg[(r, d, 0)]["frac_positive"]
                                     - gg[(r, d, 1)]["frac_positive"]))
        if diffs:
            print(f"\ncells with |diff| > 0.3 : {sum(1 for x in diffs if x > 0.3)}"
                  f" of {len(diffs)}")
            print(f"max |diff|              : {max(diffs):.3f}")
            print(f"binomial SE at n=400    : {0.5/math.sqrt(400):.4f}")
            print(f"max |diff| in SE units  : {max(diffs)/(0.5/math.sqrt(400)):.1f}")
    else:
        print("  (pass --gonogo)")

    # ---------------------------------------------------------------- §5.3
    print("\n" + "=" * 68)
    print("§5.3  median row means, gated (d_median_valid)")
    print("=" * 68)
    if gg:
        print(f"{'R_old':>6} {'s0 med':>9} {'s1 med':>9} "
              f"{'s0 neg':>8} {'s1 neg':>8}")
        for r in R_ORD:
            cells = {s: [gg[(r, d, s)] for d in D_ORD if (r, d, s) in gg]
                     for s in (0, 1)}
            row = [r]
            for s in (0, 1):
                v = [c.get("d_median_valid", c["d_median"]) for c in cells[s]]
                row.append(float(np.mean(v)) if v else float("nan"))
            neg = [sum(1 for c in cells[s]
                       if c.get("d_median_valid", c["d_median"]) < 0) for s in (0, 1)]
            print(f"{row[0]:>6} {row[1]:>+9.3f} {row[2]:>+9.3f} "
                  f"{neg[0]:>8} {neg[1]:>8}")
        print("\ncells with negative gated median at R_old >= 8:")
        for r in R_ORD:
            for d in D_ORD:
                for s in (0, 1):
                    c = gg.get((r, d, s))
                    if c and c.get("d_median_valid", c["d_median"]) < 0:
                        print(f"  R{r} D{d} s{s}: "
                              f"{c.get('d_median_valid', c['d_median']):+.4f}")

    # ---------------------------------------------------------------- §5.5
    print("\n" + "=" * 68)
    print("§5.5  gate effect on median (ungated -> gated)")
    print("=" * 68)
    rows = []
    for (r, d, s), c in gg.items():
        u = c["d_median"]
        v = c.get("d_median_valid", u)
        if abs(u) > 1e-9:
            rows.append((abs((v - u) / u), r, d, s, u, v, c.get("mass"),
                         c.get("frac_mass_ok")))
    rows.sort(reverse=True)
    print(f"{'cell':<12}{'ungated':>9}{'gated':>9}{'shift':>8}"
          f"{'mass':>7}{'mOK':>7}")
    for sh, r, d, s, u, v, mass, mok in rows[:8]:
        print(f"R{r}D{d}s{s:<7}{u:>+9.3f}{v:>+9.3f}{sh*100:>7.0f}%"
              f"{mass:>7.2f}{mok:>7.2f}")

    # ---------------------------------------------------------------- §6.2
    print("\n" + "=" * 68)
    print(f"§6.2 / App E  escape: gate vs loss-derivative peak "
          f"(tail_frac={a.tail_frac})")
    print("=" * 68)
    pk = {}
    for k, (loss, probes, esc) in runs.items():
        p = escape_peak(loss, a.tail_frac)
        if p:
            p["gate"] = esc
            pk[k] = p
    print(f"{'R_old':>6}  {'gate s0':>8} {'gate s1':>8}   "
          f"{'peak s0':>8} {'peak s1':>8}")
    for r in R_ORD:
        g = {s: [pk[(r, d, s)]["gate"] for d in D_ORD
                 if (r, d, s) in pk and pk[(r, d, s)]["gate"]] for s in (0, 1)}
        q = {s: [pk[(r, d, s)]["peak_step"] for d in D_ORD
                 if (r, d, s) in pk] for s in (0, 1)}
        print(f"{r:>6}  {np.mean(g[0]):>8.0f} {np.mean(g[1]):>8.0f}   "
              f"{np.mean(q[0]):>8.0f} {np.mean(q[1]):>8.0f}")

    print("\nper-run detail (LaTeX rows for tab:escape):")
    print("% R & dD & gate0 & peak0 & h0 & gate1 & peak1 & h1")
    for r in R_ORD:
        for d in D_ORD:
            A, B = pk.get((r, d, 0)), pk.get((r, d, 1))
            if not (A and B):
                continue
            print(f"{r} & {d} & {A['gate']} & {A['peak_step']} & {A['peak_h']:.2f}"
                  f" & {B['gate']} & {B['peak_step']} & {B['peak_h']:.2f} \\\\")
        print(r"\addlinespace")

    print("\npeak height vs escape step (for the 'sharper when later' claim):")
    pts = [(v["peak_step"], v["peak_h"], k) for k, v in pk.items()]
    pts.sort()
    print(f"  earliest: {pts[0][2]} step {pts[0][0]} h {pts[0][1]:.2f}")
    print(f"  latest  : {pts[-1][2]} step {pts[-1][0]} h {pts[-1][1]:.2f}")
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    print(f"  Spearman rho = "
          f"{np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0,1]:.3f}"
          f"  (n={len(pts)})")

    print("\nruns where the cosine tail exceeds the escape peak "
          "(§6.2 qualification):")
    bad = [(k, v) for k, v in pk.items() if v["tail_h"] > v["peak_h"]]
    for k, v in sorted(bad):
        print(f"  R{k[0]}D{k[1]}s{k[2]}: escape {v['peak_h']:.2f} @{v['peak_step']}"
              f"  tail {v['tail_h']:.2f} @{v['tail_step']}")
    print(f"  total {len(bad)} of {len(pk)}")

    # ---------------------------------------------------------------- §6.3
    print("\n" + "=" * 68)
    print("§6.3  last probe before escape (plateau attribution)")
    print("=" * 68)
    pre = []
    for k, (loss, probes, esc) in runs.items():
        if not esc:
            continue
        cand = [p for p in probes if p[0] < esc]
        if cand:
            pre.append((k, esc) + cand[-1])
    pre.sort(key=lambda t: t[3])
    print(f"{'cell':<12}{'esc':>6}{'probe':>7}{'frac+':>7}{'mean':>8}")
    for k, esc, st, fx, dm in pre:
        flag = "  <- step<200, untrained" if st < 200 else ""
        print(f"R{k[0]}D{k[1]}s{k[2]:<6}{esc:>6}{st:>7}{fx:>7.2f}"
              f"{(dm if dm is not None else float('nan')):>+8.2f}{flag}")
    usable = [t for t in pre if t[2] >= 200]
    print(f"\nruns with a pre-escape probe          : {len(pre)} of {len(runs)}")
    print(f"  of which probe step >= 200          : {len(usable)}")
    print(f"  frac+ < 0.20                        : "
          f"{sum(1 for t in usable if t[3] < 0.20)}")
    print(f"  frac+ < 0.05                        : "
          f"{sum(1 for t in usable if t[3] < 0.05)}")
    print("  ^ these two go into the \\PL{} slots in §6.3")

    # ---------------------------------------------------------------- §5.6
    print("\n" + "=" * 68)
    print("§5.6 / App F  within-run spread after formation")
    print("=" * 68)
    sds = []
    for k, (loss, probes, esc) in runs.items():
        if not esc:
            continue
        post = [p[1] for p in probes if p[0] > esc]
        if len(post) >= 5:
            sds.append((float(np.std(post)), k, len(post), min(post), max(post)))
    sds.sort(reverse=True)
    print(f"{'cell':<12}{'n':>4}{'sd':>7}{'min':>7}{'max':>7}")
    for sd, k, n, lo, hi in sds[:8]:
        print(f"R{k[0]}D{k[1]}s{k[2]:<6}{n:>4}{sd:>7.3f}{lo:>7.2f}{hi:>7.2f}")
    print(f"...\nmedian within-run sd over {len(sds)} runs: "
          f"{np.median([s[0] for s in sds]):.3f}")
    print(f"range                                : "
          f"{min(s[0] for s in sds):.3f} to {max(s[0] for s in sds):.3f}")

    # ---------------------------------------------------- consistency check
    print("\n" + "=" * 68)
    print("check  terminal frac+: trajectory (n=200) vs go_nogo (n=400)")
    print("=" * 68)
    ds = []
    for k, (loss, probes, esc) in runs.items():
        if k in gg and probes:
            ds.append(abs(probes[-1][1] - gg[k]["frac_positive"]))
    if ds:
        print(f"  n={len(ds)}  mean |diff| {np.mean(ds):.3f}  max {max(ds):.3f}")
        print("  (both ungated, computed over all documents with edit domain)")


if __name__ == "__main__":
    main()



"""python paper_numbers.py \
  --s0-dir runs_g2 \
  --s1-dir /root/autodl-tmp/runs_g2_s1 \
  --gonogo runs_g2/gonogo_s0.txt "/root/autodl-tmp/runs_g2_s1/*.txt" \
  2>&1 | tee numbers_out.txt"""