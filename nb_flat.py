"""Compare saved geometry measurements between supervision arms."""
import argparse
import json

NAN = float("nan")


def load(path):
    """step -> 记录。同 step 重复时后者覆盖（flatdir 是追加写的）。"""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                out[o["step"]] = o
    return out


def fd_at(rec, direction, eps_idx):
    """取某方向某步长的有限差分记录。eps_idx 是 --eps-frac 里的下标。"""
    xs = rec.get("fd", {}).get(direction) or []
    return xs[eps_idx] if eps_idx < len(xs) else {}


def fmt(v, w=10, p=3, sign=False):
    if v is None or v != v:
        return f"{'—':>{w}}"
    s = "+" if sign else ""
    return f"{v:>{w}.{p}f}" if abs(v) >= 1e-3 or v == 0 else f"{v:>{w}.{p}e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plus", help="N+ 的 flatdir 输出（truth_rule=rarity）")
    ap.add_argument("minus", help="N− 的 flatdir 输出（truth_rule=recency）")
    ap.add_argument("--main", default=None,
                    help="主网格的 flatdir 输出（App I），可选，并列第三列")
    ap.add_argument("--eps-idx", type=int, default=2,
                    help="用第几个步长做头条，默认 2（--eps-frac 的 1e-3）")
    a = ap.parse_args()

    arms = [("N+ rarity", load(a.plus)), ("N- recency", load(a.minus))]
    if a.main:
        arms.append(("main grid", load(a.main)))

    steps = sorted(set().union(*(set(d) for _, d in arms)))
    print(f"步长下标 {a.eps_idx}；步数 {steps}")

    # ---- 配对不变量。不相等则后面的比较无意义 ----
    print("\n配对检查（两条 arm 的编辑域必须逐篇相同）")
    bad = 0
    for s in steps:
        r0, r1 = arms[0][1].get(s), arms[1][1].get(s)
        if not r0 or not r1:
            print(f"  step {s:>6}  缺一侧，跳过")
            continue
        n0, n1 = r0.get("pair_n"), r1.get("pair_n")
        ok = n0 == n1
        bad += not ok
        print(f"  step {s:>6}  pair_n {n0} vs {n1}"
              f"  yield {r0.get('pair_yield', NAN):.3f} vs "
              f"{r1.get('pair_yield', NAN):.3f}  {'✓' if ok else '✗'}")
    if bad:
        print(f"  ⚠ {bad} 步的编辑域不同。两条 arm 的 corpus 只应差标签 token，")
        print("    且反转文档已被 break_rarity 的域排除 —— 不相等说明配对假设破了。")
    else:
        print("  编辑域逐步相同 ✓（pair_yield≈0.58 是对的：0.83 的编辑域 ×")
        print("   0.70 的非反转比例，反转文档被 probe 的守卫排除）")

    # ---- 第 1 层：一阶角度与噪声地板 ----
    print("\n一阶角度（预期三组都落在偶然水平附近 —— 这是 App I 已失败的那层）")
    print(f"  {'step':>6} {'arm':<11} {'cos(gL,gD)':>12} {'去衰减':>10} "
          f"{'cos(gL,rand)':>13} {'half_half':>10} {'偶然':>9}")
    for s in steps:
        for name, d in arms:
            r = d.get(s)
            if not r:
                continue
            c = r["cos"]
            print(f"  {s:>6} {name:<11} {fmt(c.get('L_D'), 12, 5, True)} "
                  f"{fmt(c.get('L_D_corrected'), 10, 5, True)} "
                  f"{fmt(c.get('L_R'), 13, 5, True)} "
                  f"{fmt(c.get('half_half'), 10, 4)} "
                  f"{r.get('chance_cos', NAN):>9.1e}")

    # ---- 第 3 层：有限差分。头条在这里 ----
    print("\n有限差分沿 û_Δ⊥（目标函数不同 ⇒ 这一层该分开）")
    print(f"  {'step':>6} {'arm':<11} {'dL(+)':>12} {'dL(-)':>12} "
          f"{'dD(+)':>10} {'nats/loss':>11} {'uᵀH_Lu':>12}")
    for s in steps:
        for name, d in arms:
            r = d.get(s)
            if not r:
                continue
            x = fd_at(r, "delta_perp", a.eps_idx)
            if not x:
                continue
            # dL_minus 不在 flatdir 的输出里；dL_central 是对称差分，
            # 它与 dL_plus 的关系给出不对称性：central≈(plus−minus)/2eps。
            print(f"  {s:>6} {name:<11} {fmt(x.get('dL_plus'), 12, 3, True)} "
                  f"{fmt(x.get('dL_central'), 12, 3, True)} "
                  f"{fmt(x.get('dD_plus'), 10, 4, True)} "
                  f"{fmt(x.get('nats_per_loss'), 11, 1)} "
                  f"{fmt(x.get('curv_L'), 12, 3, True)}")

    print("\n对照方向（û_L 与随机）")
    for direction in ("loss_dir", "random"):
        print(f"  -- {direction}")
        for s in steps:
            for name, d in arms:
                r = d.get(s)
                if not r:
                    continue
                x = fd_at(r, direction, a.eps_idx)
                if not x:
                    continue
                print(f"     {s:>6} {name:<11} "
                      f"dL(+) {fmt(x.get('dL_plus'), 12, 3, True)} "
                      f"dD(+) {fmt(x.get('dD_plus'), 10, 4, True)} "
                      f"nats/loss {fmt(x.get('nats_per_loss'), 9, 1)}")

    # ---- 终态读数，确认测的是同一个模型 ----
    print("\n终态读数（应与 go_nogo 的 med/frac+ 一致）")
    for name, d in arms:
        r = d.get(max(d)) if d else None
        if not r:
            continue
        ro = r["readout"]
        print(f"  {name:<11} step {r['step']:>6}  L={r['loss']:.6f}  "
              f"Δ mean {ro['mean']:+.3f}  median {ro['median']:+.3f}  "
              f"frac+ {ro['frac_pos']:.3f}  mass {ro['mass_mean']:.3f}")

    print("\nExploratory interpretation guide (conditional, not a computed verdict)")
    print("  If first-order angles overlap across all arms, ")
    print("    this instrument may fail to separate the supervision conditions; ")
    print("    the cause of a null result remains unidentified.")
    print("  If dL(+) has opposite signs in the two arms, compare the specified ")
    print("    directions, step sizes and objective populations before interpreting ")
    print("    the difference; the sign alone does not identify its cause.")
    print("  Compare nats/loss descriptively using the same measurement definition.")
    print("  Overlapping measurements do not establish equivalence of mechanisms.")


if __name__ == "__main__":
    main()
