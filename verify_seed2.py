"""核对 seed 2 相关的全部论文数字。
seeds 0/1 的 frac+ 从 tab:spread 硬编码（论文既有值），seed 2 从 go_nogo 现算，
两者交叉比对。任何 MISMATCH 都说明我给的替换文本有错，别改论文，把输出发我。

  python verify_seed2.py /root/autodl-tmp/runs_g2_s2/go_nogo.txt
"""
import json, sys
import numpy as np

R_ORD, D_ORD = [3, 5, 8, 12, 16], [2, 3, 5, 8, 16]

# 论文 tab:spread 的 s0/s1 两列，作为参照基准
PAPER = {
    (3,2):(0.354,0.138), (3,3):(0.365,0.934), (3,5):(0.126,0.972),
    (3,8):(0.098,0.477), (3,16):(0.405,0.167),
    (5,2):(0.781,0.175), (5,3):(0.613,0.997), (5,5):(0.519,0.675),
    (5,8):(0.800,0.798), (5,16):(0.796,0.844),
    (8,2):(0.662,0.450), (8,3):(0.992,0.997), (8,5):(0.932,1.000),
    (8,8):(0.970,0.997), (8,16):(0.895,0.916),
    (12,2):(1.000,1.000), (12,3):(0.990,0.570), (12,5):(0.997,0.995),
    (12,8):(1.000,0.995), (12,16):(0.980,0.992),
    (16,2):(0.998,0.736), (16,3):(1.000,0.965), (16,5):(0.995,0.584),
    (16,8):(0.997,0.670), (16,16):(0.977,0.469),
}
# 我给你的 tab:within seed 2 两列，逐值核对
MY_WITHIN = {
    (3,2):(+0.029,+0.002), (3,3):(+1.472,+0.467), (3,5):(-0.017,-0.011),
    (3,8):(+1.486,+0.870), (3,16):(-0.032,-0.007),
    (5,2):(+0.481,+0.322), (5,3):(+0.242,+0.199), (5,5):(+0.300,+0.175),
    (5,8):(+0.596,+0.158), (5,16):(+0.172,+0.025),
    (8,2):(+0.113,+0.081), (8,3):(+1.022,+0.936), (8,5):(+0.997,+0.974),
    (8,8):(+2.054,+0.327), (8,16):(-0.016,-0.008),
    (12,2):(+0.833,+0.318), (12,3):(+4.017,+4.447), (12,5):(+5.089,+2.802),
    (12,8):(+1.987,+0.610), (12,16):(+1.111,+0.455),
    (16,2):(+0.780,+0.923), (16,3):(+1.448,+0.608), (16,5):(+0.444,+0.213),
    (16,8):(+1.420,+0.945), (16,16):(+0.955,+0.652),
}
# 我给你的 tab:spread seed 2 列
MY_S2 = {(3,2):0.625,(3,3):0.930,(3,5):0.270,(3,8):0.977,(3,16):0.349,
         (5,2):0.967,(5,3):0.964,(5,5):0.880,(5,8):0.931,(5,16):0.784,
         (8,2):0.763,(8,3):0.985,(8,5):0.947,(8,8):0.934,(8,16):0.432,
         (12,2):0.953,(12,3):0.998,(12,5):1.000,(12,8):0.992,(12,16):0.969,
         (16,2):0.972,(16,3):0.985,(16,5):0.870,(16,8):0.992,(16,16):0.969}

def load(paths):
    out = {}
    for p in paths:
        for line in open(p):
            line = line.strip()
            objs = (json.loads(line[4:]) if line.startswith("raw:")
                    else [json.loads(line)] if line.startswith("{") else [])
            for o in objs:
                if o.get("state") == "retr" and o.get("total_steps") == 16000:
                    out[(o["r_old"], o["dd"], o.get("seed", 0))] = o
    return out

g = load(sys.argv[1:])
bad = []
def chk(name, got, want, tol=0.0015):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not ok:
        bad.append(f"{name}: got {got}, expected {want}")
    return ok

print("=== A. tab:spread 的 s2 列与 tab:within 的 s2 两列 ===")
print(" R  dD      s2   my_s2    medHi    medLo   my_Hi   my_Lo")
for r in R_ORD:
    for d in D_ORD:
        o = g.get((r, d, 2))
        if not o:
            print(f"{r:>2}{d:>4}   MISSING"); bad.append(f"missing s2 ({r},{d})")
            continue
        f, hi, lo = o["frac_positive"], o["d_med_hiK"], o["d_med_loK"]
        mf, (mh, ml) = MY_S2[(r, d)], MY_WITHIN[(r, d)]
        chk(f"s2 frac ({r},{d})", round(f, 3), mf)
        chk(f"medHi ({r},{d})", round(hi, 3), mh, 0.0015)
        chk(f"medLo ({r},{d})", round(lo, 3), ml, 0.0015)
        print(f"{r:>2}{d:>4}{f:>8.3f}{mf:>8.3f}{hi:>9.3f}{lo:>9.3f}"
              f"{mh:>8.3f}{ml:>8.3f}")

print("\n=== B. 八格三 seed 不同向（第 9' 项的核心论据）===")
wide, disag, rng = [], [], {}
for r in R_ORD:
    for d in D_ORD:
        vs = [PAPER[(r, d)][0], PAPER[(r, d)][1], g[(r, d, 2)]["frac_positive"]]
        rng[(r, d)] = max(vs) - min(vs)
        if rng[(r, d)] > 0.3:
            wide.append((r, d))
        if any(v < 0.5 for v in vs) and any(v > 0.5 for v in vs):
            disag.append((r, d, min(vs), max(vs)))
print(f"range > 0.3 in {len(wide)} cells: {wide}")
print(f"max range {max(rng.values()):.3f} at {max(rng, key=rng.get)}")
print(f"seeds straddle 0.5 in {len(disag)} cells:")
for r, d, mn, mx in disag:
    print(f"   ({r},{d})  min {mn:.3f}  max {mx:.3f}")
# 双 seed 版本，用于"5 -> 8"的对比
d2 = sum(1 for r in R_ORD for d in D_ORD
         if min(PAPER[(r,d)]) < 0.5 < max(PAPER[(r,d)]))
print(f"(two seeds only: {d2} cells straddle 0.5)")
# 距 0.5 不足 2 个标准误的格，判据敏感
se = 0.025
fragile = [(r, d) for r in R_ORD for d in D_ORD
           for vs in [[*PAPER[(r,d)], g[(r,d,2)]["frac_positive"]]]
           if any(abs(v - 0.5) < 2 * se for v in vs)]
print(f"cells with a seed within 2 SE of 0.5 (verdict is fragile): {fragile}")

print("\n=== C. 行均值 ===")
pooled = []
for r in R_ORD:
    per = [np.mean([PAPER[(r,d)][s] if s < 2 else g[(r,d,2)]["frac_positive"]
                    for d in D_ORD]) for s in (0, 1, 2)]
    pm = np.mean([PAPER[(r,d)][s] if s < 2 else g[(r,d,2)]["frac_positive"]
                  for d in D_ORD for s in (0, 1, 2)])
    pooled.append(pm)
    print(f"R{r:<3} per-seed {per[0]:.3f} {per[1]:.3f} {per[2]:.3f}"
          f"   pooled {pm:.3f}")
print(f"pooled monotone through R12: "
      f"{all(pooled[i] < pooled[i+1] for i in range(3))}")
surv = [f"{R_ORD[i]}->{R_ORD[i+1]}" for i in range(4)
        if all(np.mean([PAPER[(R_ORD[i],d)][s] if s < 2
                        else g[(R_ORD[i],d,2)]["frac_positive"] for d in D_ORD])
               < np.mean([PAPER[(R_ORD[i+1],d)][s] if s < 2
                          else g[(R_ORD[i+1],d,2)]["frac_positive"]
                          for d in D_ORD]) for s in (0, 1, 2))]
print(f"steps holding in every seed: {surv}")

print("\n=== D. seed 2 组内梯度 ===")
hold, inv = 0, []
for r in R_ORD:
    for d in D_ORD:
        o = g[(r, d, 2)]
        m, hi, lo = o["d_median"], o["d_med_hiK"], o["d_med_loK"]
        ok = (hi > lo) if m > 0 else (hi < lo)
        hold += ok
        if not ok:
            inv.append((r, d, m, hi, lo, abs(hi - lo) / abs(m) * 100))
print(f"holds {hold}/25, inverts {len(inv)}")
for r, d, m, hi, lo, pc in inv:
    print(f"   ({r},{d}) med {m:+.3f}  hi {hi:+.3f}  lo {lo:+.3f}  gap {pc:.0f}%")
print(f"=> total across three seeds: {42 + hold}/75, "
      f"{8 + len(inv) - 6} inversions carrying an effect")

print("\n=== E. mass / 负中位数 / acc / gate ===")
low = [(r, d) for r in R_ORD for d in D_ORD if g[(r,d,2)]["mass"] < 0.85]
print(f"seed 2 mass < 0.85: {low}")
for r, d in low:
    o = g[(r, d, 2)]
    print(f"   ({r},{d}) mass {o['mass']:.3f}  ungated {o['d_median']:+.3f}"
          f"  gated {o['d_median_valid']:+.3f}"
          f"  shift {abs(o['d_median_valid']-o['d_median'])/abs(o['d_median'])*100:.0f}%")
print(f"=> total mass<0.85 across three seeds: {8 + len(low)}")
neg = [(r, d, g[(r,d,2)]["d_median"]) for r in R_ORD for d in D_ORD
       if r >= 5 and g[(r,d,2)]["d_median"] < 0]
print(f"seed 2 negative median at R>=5: {neg}")
acc = [(r, d, g[(r,d,2)]["acc"]) for r in R_ORD for d in D_ORD
       if g[(r,d,2)]["acc"] < 1.0]
print(f"seed 2 acc < 1.0: {acc}; all >= 0.999: "
      f"{all(a >= 0.999 for _, _, a in acc)}")
for r in R_ORD:
    print(f"R{r:<3} gate row mean {np.mean([g[(r,d,2)]['escape_step'] for d in D_ORD]):.0f}")

print("\n=== 断言核对 ===")
chk("13 cells > 0.3", len(wide), 13)
chk("8 cells straddle", len(disag), 8)
chk("max range", round(max(rng.values()), 3), 0.879)
chk("pooled means", [round(p, 3) for p in pooled],
    [0.479, 0.768, 0.858, 0.962, 0.879])
chk("seed2 holds", hold, 23)
chk("seed2 mass<0.85 count", len(low), 2)
print("ALL CLEAR" if not bad else "MISMATCHES:\n  " + "\n  ".join(bad))