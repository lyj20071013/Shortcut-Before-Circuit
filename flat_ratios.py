
"""两个种子的曲率比与 nats 比，只在 post-escape（mass≥0.5）段。

cv⊥/cv_L  主数字。无标度，不含读数量纲。这是进论文的那个。
cv⊥/cv_R  证明不是绝对平坦。
nats⊥/nats_L  |g_Δ| 在比里约掉，但含 cos(g_Δ,g_L) 在分母，噪声大，仅辅助。

取索引 1 的 eps（第二小档）：最小档 dL 落在 fp32 ULP 附近，最大两档已离开线性区。
"""
import json
import os
import sys


def recs(p):
    for line in open(p):
        if line.strip():
            yield json.loads(line)


def main():
    if len(sys.argv) != 3:
        print("用法: python flat_ratios.py <s0.jsonl> <s1.jsonl>", file=sys.stderr)
        sys.exit(2)
    for p in sys.argv[1:]:
        if not os.path.exists(p):
            print(f"找不到 {p}", file=sys.stderr)
            sys.exit(2)

    for tag, path in (("s0", sys.argv[1]), ("s1", sys.argv[2])):
        out = []
        for r in sorted(recs(path), key=lambda x: x["step"]):
            if r["readout"]["mass_mean"] < 0.5:
                continue
            dp, ld, rd = (r["fd"][k][1] for k in
                          ("delta_perp", "loss_dir", "random"))

            def nats(x):
                return (abs(x["dD_plus"]) / x["dL_plus"]
                        if x["dL_plus"] > 0 else float("nan"))

            np_, nl = nats(dp), nats(ld)
            out.append((r["step"],
                        dp["curv_L"] / ld["curv_L"] if ld["curv_L"] else float("nan"),
                        dp["curv_L"] / rd["curv_L"] if rd["curv_L"] > 0 else float("nan"),
                        np_ / nl if nl > 0 else float("nan"),
                        r["readout"]["mass_mean"],
                        r["readout"]["frac_pos"]))
        print(f"\n{tag}  post-escape {len(out)} 个 checkpoint")
        print(f"  {'step':>6} {'cv_perp/cv_L':>13} {'1/x':>8} "
              f"{'cv_perp/cv_R':>13} {'nats ratio':>11} {'mass':>6} {'frac+':>6}")
        for s, a, b, c, m, f in out:
            inv = f"1/{1/a:.0f}" if a == a and a > 0 else "-"
            print(f"  {s:>6} {a:>13.3e} {inv:>8} {b:>13.1f} "
                  f"{c:>11.2e} {m:>6.3f} {f:>6.3f}")
        if out:
            for i, name in ((1, "cv_perp/cv_L"), (2, "cv_perp/cv_R"),
                            (3, "nats ratio")):
                v = [x[i] for x in out if x[i] == x[i]]
                if v:
                    extra = (f"   即 1/{1/max(v):.0f} – 1/{1/min(v):.0f}"
                             if i == 1 and min(v) > 0 else "")
                    print(f"  {name:>14} 范围 {min(v):.3e} – {max(v):.3e}{extra}")


if __name__ == "__main__":
    main()
