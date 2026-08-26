r"""论文里每个行均值与占位数字的唯一来源，按论文位置分节输出。

截图读数不进论文。所有 \PL{} 标记的数从这里取。

用法:
  python paper_numbers.py \
    --s0-dir runs_g2 \
    --s1-dir /root/autodl-tmp/runs_g2_s1 \
    --s2-dir /root/autodl-tmp/runs_g2_s2 \
    --gonogo runs_g2/gonogo_s0.txt /root/autodl-tmp/runs_g2_s1/*.txt \
             /root/autodl-tmp/runs_g2_s2/*.txt

口径说明:
  终态 frac+ 取自 go_nogo（n=400，权威）。
  逃逸前 frac+ 取自训练内嵌探针（n=200，是训练时唯一存在的读数）。
  两者在同格上差 <0.03（见 --check-consistency）。
"""
import argparse, glob, json, math, os, re
import numpy as np

R_ORD = [3, 5, 8, 12, 16]
D_ORD = [2, 3, 5, 8, 16]


def mean_or_nan(v):
    """空列表返回 nan，不让 np.mean 抛 RuntimeWarning。

    种子目录缺失时下游整片输出会变成空列表，之前靠 (0, 1) 写死掩盖了。
    """
    return float(np.mean(v)) if len(v) else float("nan")


def rank_avg(x):
    """平均秩，并列取组内均值。

    §6.2 明说 Spearman 用平均秩，而 argsort(argsort(x)) 给的是序数秩，
    并列时按出现顺序任意打断。tab:escape 里有 19 个 run 的 peak 都在
    400 步，这个区别不是小数点后的事。
    """
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(xs, ys):
    """平均秩上的 Pearson。"""
    if len(xs) < 3:
        return float("nan")
    return float(np.corrcoef(rank_avg(xs), rank_avg(ys))[0, 1])


def gated(c):
    """门后中位数，缺列回落到未门。"""
    return c.get("d_median_valid", c["d_median"])


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
    ap.add_argument("--s2-dir", default="/root/autodl-tmp/runs_g2_s2")
    ap.add_argument("--gonogo", nargs="*", default=[])
    ap.add_argument("--tail-frac", type=float, default=0.8)
    ap.add_argument("--extra-dir", nargs="*", default=[],
                    help="任意命名的训练 jsonl 目录（深度臂等）。主网格按 "
                         "R{r}_D{d}_s{s}_grid.jsonl 匹配，这些臂的 tag 不合该式，"
                         "所以按 tag 逐个报峰位与逃逸后漂移。")
    a = ap.parse_args()

    src = {0: a.s0_dir, 1: a.s1_dir, 2: a.s2_dir}
    runs = {}
    for s, d in src.items():
        if not d:
            continue
        if not os.path.isdir(d):
            print(f"# WARNING seed {s}: no such dir {d}")
            continue
        for r in R_ORD:
            for dd in D_ORD:
                p = os.path.join(d, f"R{r}_D{dd}_s{s}_grid.jsonl")
                if os.path.exists(p):
                    runs[(r, dd, s)] = read_run(p)

    gg = {}
    for pat in a.gonogo:
        hits = sorted(glob.glob(pat))
        if not hits:
            print(f"# WARNING gonogo pattern matched nothing: {pat}")
            continue
        gg.update(load_gonogo(hits))

    # 种子集合从数据里推，不写死。缺哪个种子下面每一节都会自己少一列，
    # 而不是静默按两列算完再骗人。
    seeds_gg = sorted({k[2] for k in gg})
    seeds_run = sorted({k[2] for k in runs})
    print(f"# {len(runs)} training runs (seeds {seeds_run}), "
          f"{len(gg)} go_nogo rows (seeds {seeds_gg})")
    for s in seeds_run:
        print(f"#   seed {s}: {sum(1 for k in runs if k[2] == s)} runs, "
              f"{sum(1 for k in gg if k[2] == s)} go_nogo cells")
    print()

    # ---------------------------------------------------------------- §5.2
    print("=" * 68)
    print("§5.2  frac+ row means (go_nogo, n=400)")
    print("=" * 68)
    SE400 = 0.5 / math.sqrt(400)
    if gg:
        hdr = "".join(f"{'s' + str(s):>9}" for s in seeds_gg)
        print(f"{'R_old':>6}{hdr}{'pooled':>9}{'rowmin':>9}{'rowmax':>9}  n")
        for r in R_ORD:
            per = {s: mean_or_nan([gg[(r, d, s)]["frac_positive"]
                                   for d in D_ORD if (r, d, s) in gg])
                   for s in seeds_gg}
            allv = [gg[(r, d, s)]["frac_positive"]
                    for d in D_ORD for s in seeds_gg if (r, d, s) in gg]
            cols = "".join(f"{per[s]:>9.3f}" for s in seeds_gg)
            lo = min(allv) if allv else float("nan")
            hi = max(allv) if allv else float("nan")
            print(f"{r:>6}{cols}{mean_or_nan(allv):>9.3f}"
                  f"{lo:>9.3f}{hi:>9.3f}  {len(allv)}")
        print("  pooled  = mean over all seeds x cells in the row (§5.2 row means)")
        print("  rowmin/rowmax = extremes over that same set")
        print("  ^ §sec:flip 'spanning X to Y across the fifteen runs' 取这两列")

        cr = []
        for r in R_ORD:
            for d in D_ORD:
                v = [gg[(r, d, s)]["frac_positive"]
                     for s in seeds_gg if (r, d, s) in gg]
                if len(v) >= 2:
                    cr.append((max(v) - min(v), r, d, v))
        cr.sort(reverse=True)
        if cr:
            print(f"\nper-cell range across seeds {seeds_gg}, descending "
                  f"(tab:spread):")
            for rg, r, d, v in cr:
                vs = " ".join(f"{x:.3f}" for x in v)
                side = "  <- straddles 0.5" if min(v) < 0.5 <= max(v) else ""
                print(f"  R{r:<3}D{d:<3}{vs:>26}   range {rg:.3f}{side}")
            print(f"\ncells with range > 0.3  : "
                  f"{sum(1 for t in cr if t[0] > 0.3)} of {len(cr)}")
            print(f"cells straddling 0.5    : "
                  f"{sum(1 for t in cr if min(t[3]) < 0.5 <= max(t[3]))}")
            print(f"max range               : {cr[0][0]:.3f} "
                  f"at R{cr[0][1]} D{cr[0][2]}")
            print(f"binomial SE at n=400    : {SE400:.4f}")
            print(f"max range in SE units   : {cr[0][0] / SE400:.1f}")
    else:
        print("  (pass --gonogo)")

    # ---------------------------------------------------------------- §5.3
    print("\n" + "=" * 68)
    print("§5.3  median row means, gated (d_median_valid)")
    print("=" * 68)
    if gg:
        hdr = "".join(f"{'s' + str(s) + ' med':>10}" for s in seeds_gg)
        neghdr = "".join(f"{'s' + str(s) + ' neg':>8}" for s in seeds_gg)
        print(f"{'R_old':>6}{hdr}{neghdr}")
        for r in R_ORD:
            cells = {s: [gg[(r, d, s)] for d in D_ORD if (r, d, s) in gg]
                     for s in seeds_gg}
            med = "".join(f"{mean_or_nan([gated(c) for c in cells[s]]):>+10.3f}"
                          for s in seeds_gg)
            neg = "".join(f"{sum(1 for c in cells[s] if gated(c) < 0):>8}"
                          for s in seeds_gg)
            print(f"{r:>6}{med}{neg}")

        # §sec:sign 说逐格 gated median 只在四格为负、且都在 0 的 0.031 nats 内。
        # 这里把全部负格连值列出来，四个和 0.031 都能直接数出来。
        print("\nevery cell with negative gated median, any R_old:")
        negs = []
        for r in R_ORD:
            for d in D_ORD:
                for s in seeds_gg:
                    c = gg.get((r, d, s))
                    if c and gated(c) < 0:
                        negs.append((gated(c), r, d, s))
        for v, r, d, s in sorted(negs):
            print(f"  R{r:<3}D{d:<3}s{s}: {v:+.4f}")
        print(f"  count {len(negs)}")
        if negs:
            print(f"  largest |value| : {max(abs(v) for v, *_ in negs):.4f}"
                  f"   <- §sec:sign 的 0.031 nats")
            r8 = [t for t in negs if t[1] >= 5]
            print(f"  of which R_old >= 5 : {len(r8)}"
                  f"  {[f'R{r}D{d}s{s}' for _, r, d, s in sorted(r8)]}")

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
    ghdr = "".join(f"{'gate s' + str(s):>9}" for s in seeds_run)
    qhdr = "".join(f"{'peak s' + str(s):>9}" for s in seeds_run)
    print(f"{'R_old':>6}  {ghdr}   {qhdr}")
    for r in R_ORD:
        g = {s: [pk[(r, d, s)]["gate"] for d in D_ORD
                 if (r, d, s) in pk and pk[(r, d, s)]["gate"]] for s in seeds_run}
        q = {s: [pk[(r, d, s)]["peak_step"] for d in D_ORD
                 if (r, d, s) in pk] for s in seeds_run}
        gs = "".join(f"{mean_or_nan(g[s]):>9.0f}" for s in seeds_run)
        qs = "".join(f"{mean_or_nan(q[s]):>9.0f}" for s in seeds_run)
        print(f"{r:>6}  {gs}   {qs}")

    # §sec:inv 的 "16000 steps leaves every analyzed cell at least N
    # post-formation steps" 只能由最晚的 gate 定。逐格打出来，三个种子都在，
    # 免得再用行均值去猜某个种子的最大值。
    print("\nper-cell gate, all seeds (§sec:inv post-formation floor):")
    print(f"{'R_old':>6} {'dD':>4}" + "".join(f"{'s' + str(s):>8}"
                                              for s in seeds_run))
    for r in R_ORD:
        for d in D_ORD:
            cells = "".join(
                f"{(pk[(r, d, s)]['gate'] if (r, d, s) in pk and pk[(r, d, s)]['gate'] else float('nan')):>8.0f}"
                for s in seeds_run)
            print(f"{r:>6} {d:>4}{cells}")
    allg = [v["gate"] for v in pk.values() if v["gate"]]
    never = [k for k, v in pk.items() if not v["gate"]]
    if allg:
        BUD = 16000
        worst = max(allg)
        wk = [k for k, v in pk.items() if v["gate"] == worst]
        print(f"\n  latest gate over {len(allg)} runs : {worst}"
              f"  at {['R%dD%ds%d' % k for k in sorted(wk)]}")
        print(f"  budget - latest gate        : {BUD - worst}"
              f"   <- §sec:inv 该写这个数")
        below9k = sorted(k for k, v in pk.items()
                         if v["gate"] and BUD - v["gate"] < 9000)
        print(f"  runs with < 9000 post-gate : {len(below9k)}"
              f"  {['R%dD%ds%d' % k for k in below9k]}")
    if never:
        print(f"  runs that never cross the gate: "
              f"{['R%dD%ds%d' % k for k in sorted(never)]}")

    # §sec:escape 报 "at 3 escape spans 1900--9600 steps across the three seeds"，
    # 下界只可能来自 seed 2，而 tab:escape 只印 seeds 0/1。逐格 peak 三种子全打。
    print("\nper-cell loss-derivative peak, all seeds:")
    print(f"{'R_old':>6} {'dD':>4}" + "".join(f"{'s' + str(s):>8}"
                                              for s in seeds_run))
    for r in R_ORD:
        for d in D_ORD:
            cells = "".join(
                f"{pk[(r, d, s)]['peak_step']:>8}" if (r, d, s) in pk
                else f"{'---':>8}" for s in seeds_run)
            print(f"{r:>6} {d:>4}{cells}")
    for r in R_ORD:
        v = [pk[(r, d, s)]["peak_step"] for d in D_ORD for s in seeds_run
             if (r, d, s) in pk]
        if v:
            print(f"  R_old={r:<3} peak span over all seeds: {min(v)}--{max(v)}")

    print("\nper-run detail (LaTeX rows for tab:escape, seeds 0 and 1):")
    print("% R & dD & gate0 & peak0 & h0 & gate1 & peak1 & h1")
    for r in R_ORD:
        for d in D_ORD:
            A, B = pk.get((r, d, 0)), pk.get((r, d, 1))
            if not (A and B):
                continue
            ga = A["gate"] if A["gate"] else "---"
            gb = B["gate"] if B["gate"] else "---"
            print(f"{r} & {d} & {ga} & {A['peak_step']} & {A['peak_h']:.2f}"
                  f" & {gb} & {B['peak_step']} & {B['peak_h']:.2f} \\\\")
        print(r"\addlinespace")

    print("\npeak height vs escape step (for the 'sharper when later' claim):")
    pts = sorted((v["peak_step"], v["peak_h"], k) for k, v in pk.items())
    if not pts:
        print("  (no training runs loaded; check --s0-dir/--s1-dir/--s2-dir)")
    else:
        print(f"  earliest: {pts[0][2]} step {pts[0][0]} h {pts[0][1]:.2f}")
        print(f"  latest  : {pts[-1][2]} step {pts[-1][0]} h {pts[-1][1]:.2f}")
        # 论文 §sec:escape 报的 rho=0.972 是 seeds 0+1 的 50 个 run，且明说用平均秩。
        # 两个口径都打，免得换了样本还沿用旧数字。
        for label, sub in (("seeds 0+1 (paper's 50 runs)", [0, 1]),
                           (f"all seeds {seeds_run}", seeds_run)):
            q = [p for p in pts if p[2][2] in sub]
            if len(q) >= 3:
                rho = spearman([p[0] for p in q], [p[1] for p in q])
                print(f"  Spearman rho, {label:<28}: {rho:+.3f}  (n={len(q)})")
        ties = {}
        for p in pts:
            ties[p[0]] = ties.get(p[0], 0) + 1
        big = max(ties.items(), key=lambda t: t[1])
        print(f"  largest tie group: {big[1]} runs at peak step {big[0]}"
              f"  (average ranks used)")

        print("\nruns where the cosine tail exceeds the escape peak "
              "(§6.2 qualification):")
        bad = [(k, v) for k, v in pk.items() if v["tail_h"] > v["peak_h"]]
        for k, v in sorted(bad):
            print(f"  R{k[0]}D{k[1]}s{k[2]}: escape {v['peak_h']:.2f} "
                  f"@{v['peak_step']}  tail {v['tail_h']:.2f} @{v['tail_step']}")
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
    if not sds:
        print("  (no run has >= 5 post-formation probe points)")
    else:
        print(f"{'cell':<12}{'n':>4}{'sd':>7}{'min':>7}{'max':>7}")
        for sd, k, n, lo, hi in sds[:8]:
            print(f"R{k[0]}D{k[1]}s{k[2]:<6}{n:>4}{sd:>7.3f}{lo:>7.2f}{hi:>7.2f}")
        print(f"...\nmedian within-run sd over {len(sds)} runs: "
              f"{np.median([s[0] for s in sds]):.3f}")
        print(f"range                                : "
              f"{min(s[0] for s in sds):.3f} to {max(s[0] for s in sds):.3f}")
        # §sec:flat 的 "reaches 0.332 at its largest over 69 runs" 取 len(sds) 和 max
        print(f"  ^ §sec:flat 的 N runs = {len(sds)}, 最大 sd = "
              f"{max(s[0] for s in sds):.3f}")
        per_seed = {}
        for sd, k, *_ in sds:
            per_seed.setdefault(k[2], []).append(sd)
        for s in sorted(per_seed):
            print(f"  seed {s} median sd: {np.median(per_seed[s]):.3f}"
                  f"  (n={len(per_seed[s])})")

    # ------------------------------------------------- per-cell dump, all seeds
    print("\n" + "=" * 68)
    print("App D / tab:stats  per-cell median, control, mass -- ALL seeds")
    print("=" * 68)
    if gg:
        # ctrl 的键名不确定（go_nogo 报表由另一个脚本写），先把一行的键全打出来，
        # 下次运行就能确认；同时按几个候选名去取。
        sample = gg[sorted(gg)[0]]
        print("keys available in a go_nogo row:")
        print("  " + ", ".join(sorted(sample.keys())))
        CTRL_KEYS = ("ctrl_median", "d_median_ctrl", "ctrl", "control_median",
                     "d_median_filler", "filler_median")
        ck = next((k for k in CTRL_KEYS if k in sample), None)
        print(f"ctrl column resolved to: {ck}"
              f"{'' if ck else '  <- NOT FOUND, add its name to CTRL_KEYS'}")

        def fmt(x, w=9, p=3):
            """数值右对齐带符号，None/缺列打 ---，不让格式化抛 TypeError。"""
            return (f"{x:>+{w}.{p}f}" if isinstance(x, (int, float))
                    else f"{'---':>{w}}")

        print(f"\n{'R':>3}{'dD':>4}{'s':>3}{'gated':>9}{'ungated':>9}"
              f"{'ctrl':>9}{'mass':>7}{'mOK':>7}{'frac+':>7}")
        ctrls = []
        for r in R_ORD:
            for d in D_ORD:
                for s in seeds_gg:
                    c = gg.get((r, d, s))
                    if not c:
                        continue
                    cv = c.get(ck) if ck else None
                    if cv is not None:
                        ctrls.append((cv, r, d, s))
                    print(f"{r:>3}{d:>4}{s:>3}{fmt(gated(c))}{fmt(c['d_median'])}"
                          f"{fmt(cv)}{fmt(c.get('mass'), 7, 2)}"
                          f"{fmt(c.get('frac_mass_ok'), 7, 2)}"
                          f"{fmt(c.get('frac_positive'), 7, 2)}")
            print()
        if ctrls:
            lo = min(ctrls)
            hi = max(ctrls)
            print(f"ctrl range over {len(ctrls)} cells: "
                  f"{lo[0]:+.4f} (R{lo[1]}D{lo[2]}s{lo[3]}) to "
                  f"{hi[0]:+.4f} (R{hi[1]}D{hi[2]}s{hi[3]})")
            print("  ^ App:oracle 的 '-0.015 to +0.017 across the grid' 取这一行")
            print(f"largest |ctrl|                : "
                  f"{max(abs(v) for v, *_ in ctrls):.4f}")
            # ctrl 与 readout 同号且量级可比的格子，是 App:oracle 论证的例外
            same = [(r, d, s, v, gated(gg[(r, d, s)]))
                    for v, r, d, s in ctrls
                    if v * gated(gg[(r, d, s)]) > 0
                    and abs(gated(gg[(r, d, s)])) < 0.15]
            print(f"low-median cells where ctrl shares the readout's sign: "
                  f"{len(same)}")
            for r, d, s, v, m in sorted(same):
                print(f"  R{r:<3}D{d:<3}s{s}: ctrl {v:+.4f}  median {m:+.4f}"
                      f"  ratio {abs(v / m) if m else float('nan'):.2f}")

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



r"""python paper_numbers.py \
  --s0-dir runs_g2 \
  --s1-dir /root/autodl-tmp/runs_g2_s1 \
  --s2-dir /root/autodl-tmp/runs_g2_s2 \
  --gonogo runs_g2/gonogo_s0.txt "/root/autodl-tmp/runs_g2_s1/*.txt" \
           "/root/autodl-tmp/runs_g2_s2/*.txt" \
  2>&1 | tee numbers_out.txt

种子目录用 --sN-dir，go_nogo 报表用 --gonogo（glob 要加引号，让脚本自己展开）。
缺目录或 glob 空匹配会打 WARNING 而不是静默按两种子算完。"""