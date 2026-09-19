"""Compare saved exploratory geometry measurements across seeds."""
import json
import os
import sys


def load(path):
    recs = []
    for i, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "kind" in o and "fd" not in o:
            print(f"{path} 是 train.py 的训练日志（首条记录 kind="
                  f"{o['kind']!r}），不是 flatdir 的输出。\n"
                  f"flatdir 的输出由 --out 指定，记录含 step / fd / cos 三个字段。",
                  file=sys.stderr)
            sys.exit(2)
        missing = {"step", "fd", "cos", "readout", "gnorm"} - set(o)
        if missing:
            print(f"{path} 第 {i} 行缺字段 {sorted(missing)}，不像 flatdir 输出。",
                  file=sys.stderr)
            sys.exit(2)
        recs.append(o)
    if not recs:
        print(f"{path} 里没有可用记录。", file=sys.stderr)
        sys.exit(2)
    return {r["step"]: r for r in recs}

def ratio(x):
    return (abs(x["dD_plus"]) / x["dL_plus"]
            if x["dL_plus"] > 0 else float("nan"))


def main():
    if len(sys.argv) != 3:
        print("用法: python flat_compare.py <s0.jsonl> <s1.jsonl>", file=sys.stderr)
        sys.exit(2)
    for p in sys.argv[1:]:
        if not os.path.exists(p):
            print(f"找不到 {p}", file=sys.stderr)
            sys.exit(2)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    steps = sorted(set(a) & set(b))
    print(f"共同 checkpoint {len(steps)} 个: {steps}")
    print(f"仅 s0: {sorted(set(a) - set(b))}   仅 s1: {sorted(set(b) - set(a))}\n")

    print(f"{'step':>6} {'seed':>4} {'mass':>6} {'frac+':>6} {'med':>8} "
          f"{'|gD|':>9} {'nats_perp':>10} {'nats_L':>9} "
          f"{'cv_perp':>9} {'cv_L':>9} {'cv_R':>9} {'cos':>8}")
    ok = []
    for s in steps:
        for tag, src in (("s0", a), ("s1", b)):
            r = src[s]
            ro = r["readout"]
            dp, ld, rd = (r["fd"][k][1] for k in
                          ("delta_perp", "loss_dir", "random"))
            mark = "*" if ro["mass_mean"] < 0.5 else " "
            print(f"{s:>6} {tag:>4} {ro['mass_mean']:>5.3f}{mark} "
                  f"{ro['frac_pos']:>6.3f} {ro['median']:>+8.3f} "
                  f"{r['gnorm']['D']:>9.2e} {ratio(dp):>10.2e} "
                  f"{ratio(ld):>9.2e} {dp['curv_L']:>9.2e} "
                  f"{ld['curv_L']:>9.2e} {rd['curv_L']:>9.2e} "
                  f"{r['cos']['L_D']:>+8.4f}")
        ra, rb = a[s], b[s]
        if min(ra["readout"]["mass_mean"], rb["readout"]["mass_mean"]) >= 0.5:
            pa = ratio(ra["fd"]["delta_perp"][1])
            pb = ratio(rb["fd"]["delta_perp"][1])
            if pa > 0 and pb > 0:
                ok.append((s, pa, pb, pa / pb))
        print()

    print("post-escape（两侧 mass≥0.5）的 nats_perp 比较")
    print(f"{'step':>6} {'s0':>10} {'s1':>10} {'s0/s1':>8}")
    for s, pa, pb, q in ok:
        print(f"{s:>6} {pa:>10.2e} {pb:>10.2e} {q:>8.2f}")
    if ok:
        qs = [q for _, _, _, q in ok]
        print(f"\n比值范围 {min(qs):.2f}–{max(qs):.2f}（{len(qs)} 个 checkpoint）")
        print("Ratios within 0.1-10 indicate the same order of magnitude for these endpoints.")
        print("Larger ratios indicate endpoint sensitivity here; neither case establishes generality.")


if __name__ == "__main__":
    main()