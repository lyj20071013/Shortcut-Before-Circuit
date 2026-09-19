"""Summarize saved geometry records."""
import json, math, sys

for path in sys.argv[1:]:
    print(f"\n########## {path}")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs.sort(key=lambda r: r["step"])
    for r in recs:
        c, ro, gn = r["cos"], r["readout"], r["gnorm"]
        n = r["n_params"]
        chance = 1.0 / math.sqrt(n)
        print(f"\n=== step {r['step']}  L={r['loss']:.6f}  |th|={r['theta_norm']:.2f}")
        print(f"  readout mean={ro['mean']:+.4f} med={ro['median']:+.4f} "
              f"frac+={ro['frac_pos']:.3f} mass={ro['mass_mean']:.4f} "
              f"n={ro['n']}/{ro['n_all']} yield={r['pair_yield']:.2f}")
        print(f"  |gL|={gn['L']:.6e} |gD|={gn['D']:.6e} |gS|={gn['S']:.6e} "
              f"ratio D/S={gn['D']/gn['S']:.4f}")
        print(f"  cos L_D={c['L_D']:+.6f} L_S={c['L_S']:+.6f} L_R={c['L_R']:+.6f} "
              f"chance={chance:.3e}")
        print(f"  cos half={c['half_half']:+.4f} D_S={c['D_S']:+.6f} "
              f"perp={c['perp_frac']:.8f} corrected={c.get('L_D_corrected', float('nan')):+.6f}")
        print(f"  cos core L_D={c['L_D_core']:+.6f} L_S={c['L_S_core']:+.6f}")
        ulp = abs(r["loss"]) * 1.1920929e-7
        print(f"  ULP(L0)={ulp:.3e}   <- dL 小于此值即精度地板")
        for name in ("delta_perp", "delta_raw", "loss_dir", "random"):
            rows = r["fd"].get(name) or []
            if not rows:
                continue
            print(f"  -- {name}")
            for x in rows:
                e = x["eps"]
                slope = x["dD_plus"] / e if e else float("nan")
                cent = (x["dD_plus"] - x["dD_minus"]) / (2 * e) if e else float("nan")
                pred = gn["D"] * (c["perp_frac"] if name == "delta_perp" else 1.0)
                floor = "FLOOR" if abs(x["dL_plus"]) <= 2 * ulp else "     "
                print(f"     eps={e:9.5f} dL+={x['dL_plus']:+.3e} dL-={x['dL_minus']:+.3e} "
                      f"{floor} dD+={x['dD_plus']:+.5f} dD-={x['dD_minus']:+.5f}")
                print(f"       slope={slope:+.5f} central={cent:+.5f} "
                      f"reported={x['dD_central']:+.5f} |gD|x perp={pred:+.5f} "
                      f"curvL={x['curv_L']:+.4e}")