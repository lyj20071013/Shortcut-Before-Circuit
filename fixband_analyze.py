"""固定带宽 arm 的读表。posCeil 恒定，所以三件事各自变成干净的对照：

  逃逸时刻   §6.2 说 R3 行的逃逸跨 3400–9600 步、"broadly inverse with band
             width but not monotone in it"。带宽现在恒定，若逃逸仍随 ΔD 变，
             驱动量是距离；若不变，驱动量是带宽。这是本臂唯一能跨种子复现的
             DV，也是主要产出。
  平台高度   四格同为 posCeil=1/9，主文 Table 2 只有两格同 |supp|。
             四点同一天花板是 Eq. 3 强得多的内部复制，且从现有 eval 日志直接读。
  终态读数   带跨种子散布，只能报"是否落在 Table 11 的散布内"。

导数符号约定与产 Table 4 的脚本一致：算的是 -d loss/d log step（下降越陡
峰越高）。尾部 25% 排除：cosine 末段衰减本身产生正峰，高 R_old 上会超过
逃逸峰。
"""
import argparse
import json
import math
import os
import re
from collections import defaultdict


def read(path):
    recs = defaultdict(list)
    with open(path) as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[o.get("kind")].append(o)
    return recs


def descent_peak(train, total, tail_frac=0.75):
    pts = sorted((r["step"], r["loss"]) for r in train if r["step"] > 0)
    if len(pts) < 3:
        return None
    best = best_all = (float("-inf"), None)
    for i in range(1, len(pts) - 1):
        (s0, y0), (s1, _), (s2, y2) = pts[i - 1], pts[i], pts[i + 1]
        dx = math.log(s2) - math.log(s0)
        if dx <= 0:
            continue
        d = -(y2 - y0) / dx
        if d > best_all[0]:
            best_all = (d, s1)
        if s1 <= tail_frac * total and d > best[0]:
            best = (d, s1)
    return dict(peak=best[1], height=best[0],
                peak_all=best_all[1], decay_dominates=best_all[1] != best[1])


def escape_gate(ev, floor=0.95):
    for r in sorted(ev, key=lambda x: x["step"]):
        if r.get("copy_acc", 0.0) >= floor:
            return r["step"]
    return None


def plateau(ev, gate, copy_floor=0.10):
    cand = [r for r in sorted(ev, key=lambda x: x["step"])
            if (gate is None or r["step"] < gate)
            and r.get("copy_acc", 1.0) < copy_floor]
    return cand[-1] if cand else None


def last_probe(pb):
    for r in sorted(pb, key=lambda x: -x["step"]):
        c = (r.get("causal") or {}).get("break_rarity")
        if c and c.get("frac_expected") == c.get("frac_expected"):
            return r["step"], c["frac_expected"], c.get("mass_mean")
    return None, float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--pattern", default=r"_fb(\d+)-(\d+)$")
    ap.add_argument("--steps", type=int, default=16000)
    a = ap.parse_args()

    rows = []
    for fn in sorted(os.listdir(a.dir)):
        if not fn.endswith(".jsonl"):
            continue
        tag = fn[:-6]
        m = re.search(a.pattern, tag)
        if not m:
            continue
        lo, hi = int(m.group(1)), int(m.group(2))
        rc = read(os.path.join(a.dir, fn))
        if not rc["meta"]:
            continue
        cor = rc["meta"][0].get("corpus", {})
        gate = escape_gate(rc["eval"])
        pk = descent_peak(rc["train"], a.steps) or {}
        pl = plateau(rc["eval"], gate)
        ps, frac, mass = last_probe(rc["probe"])
        done = rc["done"][0] if rc["done"] else {}
        rows.append(dict(
            tag=tag, r=cor.get("r_old_hi"), lo=lo, hi=hi, seed=cor.get("seed"),
            posCeil=1.0 / (hi - lo + 1), gate=gate,
            peak=pk.get("peak"), height=pk.get("height"),
            decay=pk.get("decay_dominates"),
            pl_acc=pl["acc"] if pl else float("nan"),
            frac=frac, mass=mass,
            fin_acc=done.get("acc", float("nan")),
            fin_copy=done.get("copy_acc", float("nan"))))

    if not rows:
        print("没有匹配的 jsonl")
        return

    print(f"{'tag':<26} {'band':>9} {'ceil':>6} {'gate':>6} {'peak':>6} "
          f"{'h':>6} {'plat':>8} {'/ceil':>6} {'frac+':>6} {'acc':>7} {'copy':>6}")
    for r in sorted(rows, key=lambda x: (x["lo"], x["seed"])):
        ratio = (r["pl_acc"] / r["posCeil"]) if r["pl_acc"] == r["pl_acc"] else float("nan")
        print(f"{r['tag']:<26} {str((r['lo'], r['hi'])):>9} {r['posCeil']:>6.3f} "
              f"{str(r['gate']):>6} {str(r['peak']):>6} {r['height'] or 0:>6.2f} "
              f"{r['pl_acc']:>8.4f} {ratio:>6.0%} {r['frac']:>6.3f} "
              f"{r['fin_acc']:>7.4f} {r['fin_copy']:>6.3f}"
              + ("  [末段衰减压过内峰]" if r["decay"] else ""))

    print("\n按带聚合（跨种子）")
    by = defaultdict(list)
    for r in rows:
        by[(r["lo"], r["hi"])].append(r)
    print(f"{'band':>9} {'n':>3} {'peak 均值':>10} {'peak 范围':>14} "
          f"{'gate 范围':>14} {'plat/ceil':>10}")
    for k in sorted(by):
        g = by[k]
        pks = [x["peak"] for x in g if x["peak"]]
        gts = [x["gate"] for x in g if x["gate"]]
        rts = [x["pl_acc"] / x["posCeil"] for x in g
               if x["pl_acc"] == x["pl_acc"]]
        print(f"{str(k):>9} {len(g):>3} "
              f"{(sum(pks) / len(pks)) if pks else float('nan'):>10.0f} "
              f"{f'{min(pks)}-{max(pks)}' if pks else '-':>14} "
              f"{f'{min(gts)}-{max(gts)}' if gts else '-':>14} "
              f"{f'{min(rts):.0%}-{max(rts):.0%}' if rts else '-':>10}")
    print("\n带宽恒定下逃逸不再随 ΔD 变 => 带宽设定逃逸，距离不设定。")
    print("四格 plat/ceil 一致 => Eq. 3 的四点内部复制（现在只有两点）。")


if __name__ == "__main__":
    main()