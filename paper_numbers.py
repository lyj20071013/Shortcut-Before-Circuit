"""Summarize manuscript-related quantities from main-grid trajectories and caches."""
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


GATED_MISS = [0]   # d_median_valid 缺列的查询次数，main() 末尾报告


def gated(c):
    """Return the mass-restricted median when that field is present.

    Legacy records without d_median_valid fall back to d_median and
    increment GATED_MISS. Such fallback values are UNRESTRICTED and
    must not be described or pooled as mass-restricted observations.
    """
    if "d_median_valid" not in c:
        GATED_MISS[0] += 1
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
                    # mass_mean 一并存：§sec:plateau 的 "In 17 of the 32 the
                    # contrast pair carries at least half the mass" 没有它就
                    # 核不了。附在元组末位，前三位索引不变，下游按位取值的
                    # 地方（§5.6 的 p[1]、§6.3 的 p[0]）不受影响。
                    probes.append((o["step"], c["frac_expected"],
                                   c.get("d_margin"), c.get("mass_mean")))
    loss.sort()
    probes.sort()
    return loss, probes, esc


def deriv(loss):
    """Loss-decrease rate -d loss / d log(step)，中心差分。"""
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
    ap.add_argument("--s1-dir", default="runs_g2")
    ap.add_argument("--s2-dir", default="runs_g2")
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
    # 每格的分母是 edit domain（App D 的 n 列，274–400），不是 400。
    # frac+ 按 App:stats 的定义就是"在有 edit domain 的文档上"的比例，
    # 所以固定用 400 会把 SE 系统性低估，低 R_old 高 ΔD 的格最严重。
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

            # 门后版本。tab:spread、摘要的 0.879 与 13/25 都取未门的 frac+，
            # 而 App:oracle 的 mass 门是逐篇的：mass<0.5 的文档上读数无效。
            # d_median_valid 服从那个门，frac_positive 不服从。两个版本并排，
            # 差值大的格就是主 DV 含无效读数的格（mOK 低处）。
            # 需要 go_nogo.py 打过 frac_positive_valid 补丁后重跑；旧的 go_nogo
            # 文件没有该字段，此节自动跳过。
            has_g = [c for c in gg.values()
                     if c.get("frac_positive_valid") is not None
                     and c["frac_positive_valid"] == c["frac_positive_valid"]]
            if not has_g:
                print("\n  (门后 frac+ 缺字段：go_nogo 需打补丁后 --force 重跑)")
            elif len(has_g) < len(gg):
                print(f"\n  (门后 frac+ 只有 {len(has_g)}/{len(gg)} 格有，"
                      f"部分 go_nogo 文件是旧的，不做门后比较)")
            else:
                crg = []
                for r in R_ORD:
                    for d in D_ORD:
                        u = [gg[(r, d, s)]["frac_positive"]
                             for s in seeds_gg if (r, d, s) in gg]
                        g = [gg[(r, d, s)]["frac_positive_valid"]
                             for s in seeds_gg if (r, d, s) in gg]
                        if len(u) >= 2:
                            crg.append((max(u) - min(u), max(g) - min(g),
                                        r, d, u, g))
                crg.sort(reverse=True)
                print(f"\nper-cell range, ungated vs gated (mass>=0.5 subset):")
                print(f"  {'cell':<10}{'rangeU':>8}{'rangeG':>8}{'Δ':>8}"
                      f"{'mOKmin':>8}")
                for ru, rg, r, d, u, g in crg:
                    mok = min(gg[(r, d, s)]["frac_mass_ok"]
                              for s in seeds_gg if (r, d, s) in gg)
                    print(f"  R{r:<3}D{d:<5}{ru:>8.3f}{rg:>8.3f}"
                          f"{rg - ru:>+8.3f}{mok:>8.2f}")
                print(f"  cells >0.3 ungated : "
                      f"{sum(1 for t in crg if t[0] > 0.3)}"
                      f"   <- tab:spread 与摘要写 13")
                print(f"  cells >0.3 gated   : "
                      f"{sum(1 for t in crg if t[1] > 0.3)}")
                print(f"  straddling 0.5 ungated : "
                      f"{sum(1 for t in crg if min(t[4]) < 0.5 <= max(t[4]))}"
                      f"   <- 摘要写 8")
                print(f"  straddling 0.5 gated   : "
                      f"{sum(1 for t in crg if min(t[5]) < 0.5 <= max(t[5]))}")
                print(f"  max range ungated  : {crg[0][0]:.3f} "
                      f"at R{crg[0][2]} D{crg[0][3]}   <- 摘要写 0.879")
                bg = max(crg, key=lambda t: t[1])
                print(f"  max range gated    : {bg[1]:.3f} "
                      f"at R{bg[2]} D{bg[3]}")
                dd_ = sorted(abs(t[1] - t[0]) for t in crg)
                print(f"  |Δrange| 中位 {dd_[len(dd_) // 2]:.3f}，"
                      f"最大 {dd_[-1]:.3f}")
                print(f"  ^ 这四对数进 App:stats 新增段。若门后与未门的 13/8/0.879"
                      f"都基本不动，那一段就是加固；若动了，须改 §sec:flip。")

            # --- P2: SE 用真实 edit domain，不用固定 400 ---
            ns = [c.get("n") for c in gg.values() if c.get("n")]
            print(f"\nbinomial SE, tab:spread 图注该用这一组:")
            print(f"  edit domain n over {len(ns)} cells: "
                  f"{min(ns)}--{max(ns)}")
            print(f"  SE at n=400 (现在图注写的)       : {SE400:.4f}")
            SEMAX = 0.5 / math.sqrt(min(ns))
            print(f"  SE at n={min(ns)} (最小 domain，保守): {SEMAX:.4f}")
            print(f"  max range in SE units, n=400     : "
                  f"{cr[0][0] / SE400:.1f}   <- 现在写的 35")
            print(f"  max range in SE units, n={min(ns)}     : "
                  f"{cr[0][0] / SEMAX:.1f}   <- 应改成这个")

            # 最宽格两端各自的真实 n 与差值 SE。极差是两个比例之差，
            # 它的 SE 不是单个比例的 SE，而是两者的平方和根。
            rg0, r0, d0, v0 = cr[0]
            ns0 = [(s, gg[(r0, d0, s)].get("n"),
                    gg[(r0, d0, s)]["frac_positive"])
                   for s in seeds_gg if (r0, d0, s) in gg]
            lo_s = min(ns0, key=lambda t: t[2])
            hi_s = max(ns0, key=lambda t: t[2])
            def se_p(p, n):
                return math.sqrt(p * (1 - p) / n) if n else float("nan")
            s_lo = se_p(lo_s[2], lo_s[1])
            s_hi = se_p(hi_s[2], hi_s[1])
            se_diff = math.sqrt(s_lo ** 2 + s_hi ** 2)
            print(f"  widest cell R{r0}D{d0}: "
                  f"s{lo_s[0]} p={lo_s[2]:.3f} n={lo_s[1]} SE={s_lo:.4f}; "
                  f"s{hi_s[0]} p={hi_s[2]:.3f} n={hi_s[1]} SE={s_hi:.4f}")
            print(f"  SE of the difference             : {se_diff:.4f}")
            print(f"  max range in SE-of-difference    : "
                  f"{rg0 / se_diff:.1f}")
            print(f"  ^ App:cross 的 2.3sigma/2.7sigma 也该用逐格 n 重算，"
                  f"该格 n={lo_s[1]} 而非 400")
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
    # P3: 从 8 行放到 20 行。§sec:degrade 的 "shifts the cell median by up to
    # 63% wherever mass drops below 0.85" 在论文里没有出处表，63% 那格必须能
    # 在这里数出来；只打前 8 行时它可能不在里面。
    print(f"{'cell':<12}{'ungated':>9}{'gated':>9}{'shift':>8}"
          f"{'mass':>7}{'mOK':>7}")
    for sh, r, d, s, u, v, mass, mok in rows[:20]:
        flag = "  <- mass<0.85" if (mass == mass and mass < 0.85) else ""
        print(f"R{r}D{d}s{s:<7}{u:>+9.3f}{v:>+9.3f}{sh*100:>7.0f}%"
              f"{mass:>7.2f}{mok:>7.2f}{flag}")

    # §sec:degrade 的声称限定在 mass<0.85 的格上，所以最大位移要在该子集上取。
    sub = [t for t in rows if t[6] == t[6] and t[6] < 0.85]
    print(f"\ncells with mass < 0.85 : {len(sub)}"
          f"   (§sec:stats 说是 ten cells across the three seeds)")
    if sub:
        top = sub[0]
        print(f"largest shift among them: {top[0]*100:.0f}% "
              f"at R{top[1]}D{top[2]}s{top[3]} "
              f"(ungated {top[4]:+.3f} -> gated {top[5]:+.3f}, "
              f"mass {top[6]:.2f})")
        print(f"  ^ D2: §sec:degrade 的 63% 该指这一格；若这里不是 63%，"
              f"那句话的出处要重新定")
    allsh = [t[0] for t in rows]
    if allsh:
        print(f"largest shift over all {len(rows)} cells: "
              f"{max(allsh)*100:.0f}%")

    # 同时把"每个 >2 nats 的格是否真的 gated"数出来，供 E18 用。
    big = [(r, d, s, u, v) for sh, r, d, s, u, v, _m, _k in rows
           if abs(u) > 2.0]
    print(f"\ncells with |ungated median| > 2 nats: {len(big)}")
    same = sum(1 for *_x, u, v in big if abs(u - v) < 1e-9)
    print(f"  of which gated == ungated (门无效果): {same}")
    print(f"  ^ E18: tab:stats 的 median 列是未门的，§sec:degrade 说主文引用的"
          f">2 nats 的中位数都是门后的；这两句只有在上面这行为 0 时才无歧义")

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

    print("\npeak height vs candidate peak step (for the 'higher when later' claim):")
    pts = sorted((v["peak_step"], v["peak_h"], k) for k, v in pk.items())
    if not pts:
        print("  (no training runs loaded; check --s0-dir/--s1-dir/--s2-dir)")
    else:
        print(f"  earliest: {pts[0][2]} step {pts[0][0]} h {pts[0][1]:.2f}")
        print(f"  latest  : {pts[-1][2]} step {pts[-1][0]} h {pts[-1][1]:.2f}")
        # Use average ranks for ties; report each seed subset explicitly.
        rho01 = None
        for label, sub in (("seeds 0+1 (paper's 50 runs)", [0, 1]),
                           (f"all seeds {seeds_run}", seeds_run)):
            q = [p for p in pts if p[2][2] in sub]
            if len(q) >= 3:
                rho = spearman([p[0] for p in q], [p[1] for p in q])
                print(f"  Spearman rho, {label:<28}: {rho:+.3f}  (n={len(q)})")
                if sub == [0, 1]:
                    rho01 = rho
        if rho01 is not None:
            print(f"\nX2  peak height vs escape step 的 Spearman rho:")
            print(f"  本次实测 (seeds 0+1)            {rho01:+.4f}")
            print(f"  论文 §sec:escape / tab:escape    +0.9330"
                  f"  {'一致' if abs(rho01 - 0.933) < 5e-4 else '不一致'}")
            print(f"  本文件旧注释                     +0.9720"
                  f"  {'一致' if abs(rho01 - 0.972) < 5e-4 else '不一致'}")
            print(f"  ^ 以本行为准，改 tex 或改注释，别让两个 artifact 各留一个")
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
    print(f"{'cell':<12}{'esc':>6}{'probe':>7}{'frac+':>7}{'mean':>8}{'mass':>7}")
    for k, esc, st, fx, dm, ms in pre:
        flag = "  <- step<200, untrained" if st < 200 else ""
        print(f"R{k[0]}D{k[1]}s{k[2]:<6}{esc:>6}{st:>7}{fx:>7.2f}"
              f"{(dm if dm is not None else float('nan')):>+8.2f}"
              f"{(ms if ms is not None else float('nan')):>7.2f}{flag}")
    usable = [t for t in pre if t[2] >= 200]
    n_esc = sum(1 for _l, _p, e in runs.values() if e)
    print(f"\n--- 分母，逐层收窄（摘要与 §sec:disc 现在都写 'of 75'）---")
    print(f"  runs loaded                         : {len(runs)}")
    print(f"  of which cross the gate             : {n_esc}")
    print(f"  of which have a pre-escape probe    : {len(pre)}")
    print(f"  of which probe step >= 200          : {len(usable)}")
    print(f"  frac+ < 0.20                        : "
          f"{sum(1 for t in usable if t[3] < 0.20)}")
    print(f"  frac+ < 0.05                        : "
          f"{sum(1 for t in usable if t[3] < 0.05)}")
    print(f"\n  ^ 计数是在 usable 上做的，不是在 {len(runs)} 上。一个在首次评估"
          f"（gate=1000）逃逸的 run 没有逃逸前探针点，根本不进 pre。")
    print(f"  ^ 所以 '32 of the 75 runs' 的分母应是 {len(usable)}，"
          f"方向上这比 /75 更强，但现在的写法不成立。")

    # §sec:plateau 说 "spanning every row across the three seeds"。若高 R_old
    # 行逃逸太早、没有逃逸前探针点，该行就不可能进入统计。逐行数出来。
    print(f"\n  usable 的逐 (R_old, seed) 分布，"
          f"供核对 'spanning every row across the three seeds':")
    print(f"{'R_old':>6}" + "".join(f"{'s' + str(s):>8}" for s in seeds_run)
          + f"{'row':>6}")
    for r in R_ORD:
        per = {s: sum(1 for t in usable if t[0][0] == r and t[0][2] == s)
               for s in seeds_run}
        tot = sum(per.values())
        print(f"{r:>6}" + "".join(f"{per[s]:>8}" for s in seeds_run)
              + f"{tot:>6}")
    print(f"  ^ 任何一行为 0 就说明该行没有逃逸前读数，'every row' 要改写")

    # 反号计数也按行打，供 §sec:plateau 的 "spanning every row" 用。
    print(f"\n  frac+ < 0.20 的逐 (R_old, seed) 分布:")
    print(f"{'R_old':>6}" + "".join(f"{'s' + str(s):>8}" for s in seeds_run)
          + f"{'row':>6}")
    for r in R_ORD:
        per = {s: sum(1 for t in usable
                      if t[0][0] == r and t[0][2] == s and t[3] < 0.20)
               for s in seeds_run}
        tot = sum(per.values())
        print(f"{r:>6}" + "".join(f"{per[s]:>8}" for s in seeds_run)
              + f"{tot:>6}")

    # §sec:plateau 的 "In 17 of the 32 the contrast pair carries at least half
    # the mass"。read_run 现在存了 mass_mean，可以直接数。
    rev = [t for t in usable if t[3] < 0.20]
    withm = [t for t in rev if t[5] is not None]
    print(f"\n  §sec:plateau 的 '17 of the 32 carry at least half the mass':")
    print(f"    frac+ < 0.20 的 run              : {len(rev)}")
    print(f"    其中带 mass 记录的               : {len(withm)}")
    if withm:
        print(f"    其中 mass >= 0.5               : "
              f"{sum(1 for t in withm if t[5] >= 0.5)}"
              f"   <- 论文写 17")
        print(f"    mass 范围                      : "
              f"{min(t[5] for t in withm):.2f} to "
              f"{max(t[5] for t in withm):.2f}")
    if len(withm) < len(rev):
        print(f"    ！{len(rev) - len(withm)} 个 run 的探针点没有 mass_mean，"
              f"这些格算不进来")

    # §sec:plateau 还说 "In five runs the pre-escape sign fraction exceeds 0.6,
    # but all five sit within one evaluation interval of escape"。一并数。
    hi6 = [t for t in usable if t[3] > 0.6]
    print(f"\n  §sec:plateau 的 'In five runs the pre-escape sign fraction "
          f"exceeds 0.6':")
    print(f"    frac+ > 0.60 的 run              : {len(hi6)}   <- 论文写 5")
    for t in sorted(hi6, key=lambda x: -x[3]):
        gap = t[1] - t[2]
        print(f"      R{t[0][0]}D{t[0][1]}s{t[0][2]}: frac+ {t[3]:.2f} "
              f"probe {t[2]} esc {t[1]} gap {gap}"
              f"{'  <= 一个评估间隔' if gap <= 1000 else '  > 一个评估间隔 ！'}")

    # 摘要与 §sec:disc 的 "reverses in 27 of 75 runs"。全文此前没有这个数的
    # 推导，tab:preescape 只印 32/22/28/17，而 "reverse" 也从未定义：相对 0.5？
    # 相对终端符号？这里按「逃逸前探针点与 16000 步终端读数落在 0.5 两侧」定义，
    # 两个方向都数。
    #
    # 口径警告：两端来自不同仪器。逃逸前是训练内嵌探针（n=200，训练时唯一存在
    # 的读数），终端是 go_nogo（n=400）。两者在同格上差 <0.03（见文件头），所以
    # 阈值附近的 run 归哪一侧可能取决于用哪个仪器 —— 因此在 0.45/0.50/0.55
    # 三个阈值上都报，计数对阈值敏感就说明这个数不能只报一个。
    print(f"\n  '27 of 75 runs reverse' 的推导（摘要与 §sec:disc）:")
    if not gg:
        print("    (需要 --gonogo 才能取终端读数)")
    else:
        for thr in (0.50, 0.45, 0.55):
            n_pair = n_lo_hi = n_hi_lo = 0
            det = []
            for k, esc, st, fx, dm, ms in usable:
                t = gg.get(k)
                if t is None:
                    continue
                n_pair += 1
                tf = t["frac_positive"]
                if fx < thr <= tf:
                    n_lo_hi += 1
                    det.append((k, st, fx, tf))
                elif tf < thr <= fx:
                    n_hi_lo += 1
            main_thr = abs(thr - 0.50) < 1e-9
            tag = "   <- 论文写 27" if main_thr else ""
            print(f"    thr={thr:.2f}  可配对 {n_pair}"
                  f"   pre<thr 且 terminal>=thr: {n_lo_hi}{tag}")
            print(f"                            反向 pre>=thr 且 terminal<thr: "
                  f"{n_hi_lo}")
            if main_thr:
                n1 = sum(1 for t in usable if t[3] < 0.50)
                print(f"      N1 = 逃逸前 frac+ < 0.5      : {n1}")
                print(f"      N2 = 其中终端 > 0.5          : {n_lo_hi}")
                print(f"      ^ tab:preescape 新增两行取这两个数。"
                      f"若 N2 != 27，改摘要而不是改表。")
                print(f"      逐 run 明细（按逃逸前 frac+ 升序）:")
                for k, st, fx, tf in sorted(det, key=lambda x: x[2]):
                    print(f"        R{k[0]}D{k[1]}s{k[2]}: probe {st} "
                          f"frac+ {fx:.3f} -> terminal {tf:.3f}")
        print(f"    可配对数少于 usable({len(usable)}) 时，差额是终端 go_nogo 里"
              f" state!=retr 或 total_steps!=16000 的 run")
        print(f"    （load_gonogo 只收这两条都满足的行）")

    # ---------------------------------------------------------------- §5.6
    print("\n" + "=" * 68)
    print("§5.6 / App F  within-run spread after formation")
    print("=" * 68)
    # Report sample SD (ddof=1) and range separately.
    # Historical readers use band for different statistics; comparisons
    # require matching the statistic, population and checkpoint window.
    sds = []
    for k, (loss, probes, esc) in runs.items():
        if not esc:
            continue
        post = [p[1] for p in probes if p[0] > esc]
        if len(post) >= 5:
            sds.append((float(np.std(post, ddof=1)), k, len(post),
                        min(post), max(post)))
    sds.sort(reverse=True)
    if not sds:
        print("  (no run has >= 5 post-formation probe points)")
    else:
        print(f"{'cell':<12}{'n':>4}{'sd':>7}{'min':>7}{'max':>7}")
        for sd, k, n, lo, hi in sds[:8]:
            print(f"R{k[0]}D{k[1]}s{k[2]:<6}{n:>4}{sd:>7.3f}{lo:>7.2f}{hi:>7.2f}")
        print(f"...\nmedian within-run sd over {len(sds)} runs: "
              f"{np.median([s[0] for s in sds]):.3f}")
        print(f"sd range                             : "
              f"{min(s[0] for s in sds):.3f} to {max(s[0] for s in sds):.3f}")

        # --- D1: within-run 极差，App:drift 现在只报了 sd ---
        rngs = [hi - lo for _, _, _, lo, hi in sds]
        print(f"\nD1  within-run RANGE over the same {len(sds)} runs:")
        print(f"  median {np.median(rngs):.3f}   max {max(rngs):.3f}"
              f"   min {min(rngs):.3f}")
        print(f"  ^ App:width 的 0.59、App:nonalias 的 0.291 都在跟"
              f"'within-run band' 比，若那个 band 指极差就用这里的 max")
        print(f"  ^ App:drift 建议两列都报：sd (median "
              f"{np.median([s[0] for s in sds]):.3f}, max "
              f"{max(s[0] for s in sds):.3f}) 与 range (median "
              f"{np.median(rngs):.3f}, max {max(rngs):.3f})")

        # Compare against the recorded reference constants below.
        # Historical references are labels, not newly computed observations.
        mx = max(s[0] for s in sds)
        print(f"\nX1  最大 within-run sd = {mx:.4f}")
        for lbl, val in (("论文 §sec:flat / App:drift", 0.303),
                         ("本文件旧注释", 0.332),
                         ("constlr_read.py:35 的 APPF_BAND_MAX", 0.303)):
            print(f"  {lbl:<38} {val:.3f}"
                  f"  {'一致' if abs(mx - val) < 5e-4 else '不一致'}")
        print(f"  ^ Compare this computed value with the saved APPF_BAND_MAX reference "
              f"using the same run set, population and SD definition.")
        worst = sds[0]
        print(f"  最大 sd 那个 run: R{worst[1][0]}D{worst[1][1]}s{worst[1][2]}"
              f"  n={worst[2]}  sd={worst[0]:.3f}"
              f"  min={worst[3]:.3f} max={worst[4]:.3f}"
              f"  range={worst[4]-worst[3]:.3f}")
        print(f"  ^ §sec:plateau 说这个 run 的 frac+ 从 0.23 动到 0.94（0.71）。"
              f"若上面 range 不是 0.71，两处口径不同（探针窗口或 ddof）")
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
        # ctrl_margin 必须在列表里：go_nogo 每格同时写 ctrl_median 与
        # ctrl_margin（报表里的 ctrlMed / ctrlAvg），两者可以反号 —— s0 的
        # R3_D3 是 ctrlMed=+0.001 而 ctrlAvg=-0.015。论文 App:oracle 报的
        # -0.009..+0.015 是 ctrl_median 的范围，而本文件旧注释的
        # -0.015..+0.017 极可能是 ctrl_margin 的范围。不把它列进来，下面的
        # 「多于一个候选键」诊断就不会触发，X3 也就查不出来。
        # ctrl_median 放第一位，与 tab:stats 的 ctrl 列一致。
        CTRL_KEYS = ("ctrl_median", "ctrl_margin", "d_median_ctrl", "ctrl",
                     "control_median", "d_median_filler", "filler_median")
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
            print(f"largest |ctrl|                : "
                  f"{max(abs(v) for v, *_ in ctrls):.4f}")

            # --- X3: 三处不一致 ---
            print(f"\nX3  filler-slot 控制的范围，三个 artifact 对不上:")
            print(f"  本次实测                       "
                  f"{lo[0]:+.4f} to {hi[0]:+.4f}")
            print(f"  论文 App:oracle / 引言          -0.0090 to +0.0150")
            print(f"  本文件旧注释                    -0.0150 to +0.0170")
            ok = abs(lo[0] - (-0.009)) < 5e-4 and abs(hi[0] - 0.015) < 5e-4
            print(f"  与论文{'一致' if ok else '不一致'}")
            print(f"  ^ 引言写 'stays within 0.015 nats'，若上界 > 0.015 那句不成立")
            print(f"  ^ 这一条支撑'编辑没有副作用'这个仪器级声明，优先定")

            # ctrl 列名歧义：多个候选键同时存在时，取哪个会改变上面的范围。
            present = [k for k in CTRL_KEYS if k in sample]
            print(f"\n  ctrl 候选键在数据里出现的: {present}")
            print(f"  实际取用: {ck}")
            if len(present) > 1:
                print(f"  ！多于一个候选键存在。若它们语义不同"
                      f"（median vs margin），上面的范围可能取错了列。")
                for k2 in present:
                    vals = [c[k2] for c in gg.values()
                            if isinstance(c.get(k2), (int, float))]
                    if vals:
                        print(f"    {k2:<18} range {min(vals):+.4f} "
                              f"to {max(vals):+.4f}  (n={len(vals)})")

            # App:oracle 的两个比例声明，逐格核。
            big1 = [(r, d, s, v, gated(gg[(r, d, s)]))
                    for v, r, d, s in ctrls
                    if abs(gated(gg[(r, d, s)])) > 1.0]
            if big1:
                worst1 = max(big1, key=lambda t: abs(t[3] / t[4]))
                print(f"\n  App:oracle 'below 1% of the targeted readout "
                      f"wherever that readout exceeds 1 nat':")
                print(f"    {len(big1)} 格 |median|>1 nat，"
                      f"最大比值 {abs(worst1[3]/worst1[4])*100:.2f}% "
                      f"at R{worst1[0]}D{worst1[1]}s{worst1[2]}")
            r3s0 = [(d, v, gated(gg[(3, d, 0)])) for v, r, d, s in ctrls
                    if r == 3 and s == 0]
            if r3s0:
                worst2 = max(r3s0, key=lambda t: abs(t[1] / t[2]) if t[2] else 0)
                print(f"  App:oracle 'at most 5% of the median' "
                      f"(五个 R=3 seed-0 格):")
                print(f"    最大比值 {abs(worst2[1]/worst2[2])*100:.1f}% "
                      f"at D{worst2[0]}")
                opp = sum(1 for _d, v, m in r3s0 if v * m < 0)
                print(f"    与 median 反号的: {opp}/{len(r3s0)}"
                      f"   (论文说 all five)")
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
        print(f"  ^ App:oracle 报 'agree to 0.012 on average and 0.060 at "
              f"worst'，以本行为准")
        print(f"  ^ D4/E21: tab:surface 的对照行标 n=200 => 口径 B（在线探针），"
              f"rendered 与 fine-tuned 两臂也是 n=200 => 同口径，")
        print(f"     所以 0.580 与位移 0.40 不用改；要改的是该行 acc 单元格"
              f"（含 seed 8 的 0.985）与图注的门控说明。")

    # ---------------------------------------------------------- P5 报告
    print("\n" + "=" * 68)
    print("P5  gated() 静默回落计数")
    print("=" * 68)
    print(f"  d_median_valid 缺列的查询次数: {GATED_MISS[0]}")
    if GATED_MISS[0]:
        print(f"  ！非零。说明某些标为 gated 的数其实是未门中位数。")
        print(f"  这是 §sec:degrade 与 tab:stats 那处矛盾（E18）、以及")
        print(f"  tab:slotmatch 的 -0.005 与 App:oracle 的 -0.006（A4/E19）")
        print(f"  最可能的来源。逐格确认哪些 go_nogo 行缺这一列。")
        miss = [(r, d, s) for (r, d, s), c in gg.items()
                if "d_median_valid" not in c]
        print(f"  缺列的格: {sorted(miss)}")
    else:
        print(f"  零。所有 go_nogo 行都带 d_median_valid，"
              f"E18 可以安全写成 'the two coincide wherever mOK = 1.00'。")


if __name__ == "__main__":
    main()



r"""python paper_numbers.py \
  --s0-dir runs_g2 \
  --s1-dir runs_g2 \
  --s2-dir runs_g2 \
  --gonogo runs_g2/gonogo_s0.txt "runs_g2/*.txt" \
           "runs_g2/*.txt" \
  2>&1 | tee numbers_out.txt

种子目录用 --sN-dir，go_nogo 报表用 --gonogo（glob 要加引号，让脚本自己展开）。
缺目录或 glob 空匹配会打 WARNING 而不是静默按两种子算完。"""