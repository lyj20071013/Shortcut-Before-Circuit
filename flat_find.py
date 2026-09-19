"""Extract rows from saved exploratory geometry measurements."""
import json
import os
import sys


def emit(path, cell):
    recs = sorted((json.loads(l) for l in open(path) if l.strip()),
                  key=lambda r: r["step"])
    for r in recs:
        ro, gn, c = r["readout"], r["gnorm"], r["cos"]
        dp, ld, rd = (r["fd"][k][1] for k in ("delta_perp", "loss_dir", "random"))

        def ratio(x):
            return (abs(x["dD_plus"]) / x["dL_plus"]
                    if x["dL_plus"] > 0 else float("nan"))

        mark = "" if ro["mass_mean"] >= 0.5 else r"\rlap{$^{a}$}"
        print(f"{cell} & {r['step']} & {ro['mass_mean']:.3f}{mark} & "
              f"{ro['frac_pos']:.3f} & {c['L_D']:+.4f} & {c['half_half']:.2f} & "
              f"{ratio(dp):.1e} & {ratio(ld):.1e} & "
              f"{dp['curv_L']:.1e} & {ld['curv_L']:.1e} & {rd['curv_L']:.1e} \\\\")


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        print(f"\n收到 {len(sys.argv) - 1} 个参数，需要 2 个。", file=sys.stderr)
        sys.exit(2)
    for p in sys.argv[1:]:
        if not os.path.exists(p):
            print(f"找不到 {p}", file=sys.stderr)
            sys.exit(2)
    emit(sys.argv[1], r"$3,5$")
    print(r"\addlinespace")
    emit(sys.argv[2], r"$16,2$")


if __name__ == "__main__":
    main()