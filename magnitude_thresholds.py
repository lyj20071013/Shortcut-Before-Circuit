"""Aggregate displacement thresholds from saved per-document probes."""
import argparse, csv, json, math, re
from pathlib import Path

THRESHOLDS = (0.0, 0.01, 0.1, 0.5, 1.0)
R_VALUES, D_VALUES, SEEDS = (3, 5, 8, 12, 16), (2, 3, 5, 8, 16), (0, 1, 2)
TAG_RE = re.compile(r"^R(3|5|8|12|16)_D(2|3|5|8|16)_s([012])_grid$")


def percentile(xs, p):
    s = sorted(xs)
    if not s: return float("nan")
    if len(s) == 1: return s[0]
    k = p * (len(s) - 1); lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def rates(ds, tau):
    n = len(ds); kp = sum(x > tau for x in ds); km = sum(x < -tau for x in ds)
    kz = n - kp - km
    return dict(n=n, k_plus=kp, k_zero=kz, k_minus=km,
                plus=kp/n, zero=kz/n, minus=km/n, signed=(kp-km)/n)


def load_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line); tag = row.get("tag", "")
            if TAG_RE.fullmatch(tag): out[tag] = row
    return out


def load_runs(perdoc, summary):
    P, S = load_jsonl(perdoc), load_jsonl(summary)
    expected = {f"R{r}_D{d}_s{s}_grid" for r in R_VALUES for d in D_VALUES for s in SEEDS}
    if set(P) != expected or set(S) != expected:
        raise ValueError(f"grid mismatch: perdoc missing={sorted(expected-set(P))}; summary missing={sorted(expected-set(S))}")
    runs = {}
    for tag in sorted(expected):
        p, q = P[tag], S[tag]; recs = p.get("records")
        if not isinstance(recs, list): raise ValueError(f"{tag}: full-precision records missing")
        ds = [float(x["delta"]) for x in recs]; ms = [float(x["mass"]) for x in recs]
        if len(p.get("d_all", [])) != len(ds): raise ValueError(f"{tag}: d_all length mismatch")
        if len(ds) != p["n"] or len(ds) != q["n"]: raise ValueError(f"{tag}: n mismatch")
        if sum(m >= .5 for m in ms) != p["n_valid"] or p["n_valid"] != q["n_valid"]:
            raise ValueError(f"{tag}: n_valid mismatch")
        if abs(sum(x > 0 for x in ds)/len(ds) - q["frac_positive"]) > 1e-12:
            raise ValueError(f"{tag}: tau=0 does not reproduce frac_positive")
        for a, b in zip(p["d_all"], ds):
            if not math.isclose(float(a), b, abs_tol=5.1e-5): raise ValueError(f"{tag}: d_all mismatch")
        r, d, s = map(int, TAG_RE.fullmatch(tag).groups())
        runs[tag] = dict(tag=tag, r=r, d=d, seed=s, delta=ds, mass=ms)
    if sum(len(x["delta"]) for x in runs.values()) != 28563: raise ValueError("unexpected unrestricted total")
    if sum(sum(m >= .5 for m in x["mass"]) for x in runs.values()) != 26916: raise ValueError("unexpected mass-restricted total")
    return runs


def analyze(runs):
    per_run, cells, pooled = [], [], []
    for subset in ("all", "mass>=0.5"):
        for tau in THRESHOLDS:
            pool = []
            for run in runs.values():
                ds = [x for x, m in zip(run["delta"], run["mass"])
                      if subset == "all" or m >= .5]
                z = rates(ds, tau); z.update(subset=subset, tau=tau,
                    tag=run["tag"], r=run["r"], d=run["d"], seed=run["seed"])
                per_run.append(z); pool.extend(ds)
            z = rates(pool, tau); z.update(subset=subset, tau=tau); pooled.append(z)
            for r in R_VALUES:
                for d in D_VALUES:
                    rr = [x for x in per_run if x["subset"] == subset and x["tau"] == tau
                          and x["r"] == r and x["d"] == d]
                    c = dict(subset=subset, tau=tau, r=r, d=d)
                    for key in ("plus", "zero", "minus", "signed"):
                        vals = [x[key] for x in rr]; c[key+"_range"] = max(vals)-min(vals)
                    c["crosses_direction"] = min(x["signed"] for x in rr) < 0 < max(x["signed"] for x in rr)
                    cells.append(c)
    return per_run, cells, pooled


def selected_rows(runs, per_run):
    out = []
    for subset in ("all", "mass>=0.5"):
        for r, d in ((3, 8), (3, 5)):
            for s in SEEDS:
                run = runs[f"R{r}_D{d}_s{s}_grid"]
                ds = [x for x, m in zip(run["delta"], run["mass"])
                      if subset == "all" or m >= .5]
                q = {f"q{int(p*100):02d}": percentile(ds, p)
                     for p in (.05, .25, .5, .75, .95)}
                for tau in THRESHOLDS:
                    z = next(x for x in per_run if x["subset"] == subset
                             and x["tau"] == tau and x["r"] == r
                             and x["d"] == d and x["seed"] == s)
                    out.append({**{k:z[k] for k in
                        ("subset","r","d","seed","tau","n","plus","zero","minus","signed")}, **q})
    return out


def write_outputs(prefix, runs, per_run, cells, pooled):
    prefix = Path(prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subset","tau","tag","r","d","seed","n","k_plus","k_zero","k_minus","plus","zero","minus","signed"]
    with open(str(prefix)+".csv", "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k:x[k] for k in fields} for x in per_run)
    sel = selected_rows(runs, per_run)
    payload = dict(thresholds=THRESHOLDS, definitions={"plus":"P(delta>tau)","zero":"P(|delta|<=tau)","minus":"P(delta<-tau)","signed":"plus-minus"}, pooled=pooled, cells=cells, selected=sel)
    with open(str(prefix)+".json", "w", encoding="utf-8") as f: json.dump(payload,f,indent=2,allow_nan=False)
    lines=["Magnitude-threshold sensitivity (training run is the replication unit)"]
    for subset in ("all","mass>=0.5"):
        lines += ["", subset, "tau  pooled +/0/-   max range S+ (cell)  max range signed (cell)  cells S+ range>.3  direction-crossing cells"]
        for tau in THRESHOLDS:
            p=next(x for x in pooled if x["subset"]==subset and x["tau"]==tau)
            cc=[x for x in cells if x["subset"]==subset and x["tau"]==tau]
            a=max(cc,key=lambda x:x["plus_range"]); b=max(cc,key=lambda x:x["signed_range"])
            lines.append(f"{tau:>4g}  {p['plus']:.3f}/{p['zero']:.3f}/{p['minus']:.3f}   {a['plus_range']:.3f} (R{a['r']}D{a['d']})       {b['signed_range']:.3f} (R{b['r']}D{b['d']})             {sum(x['plus_range']>.3 for x in cc):2d}                  {sum(x['crosses_direction'] for x in cc):2d}")
    lines += ["", "Selected-cell quantiles:",
              "population cell seed n q05 q25 q50 q75 q95"]
    for x in (z for z in sel if z["tau"] == 0.0):
        lines.append(f"{x['subset']:<9} R{x['r']}D{x['d']} s{x['seed']} {x['n']} "
                     f"{x['q05']:+.3f} {x['q25']:+.3f} {x['q50']:+.3f} "
                     f"{x['q75']:+.3f} {x['q95']:+.3f}")
    lines += ["", "Selected cells:",
              "population cell seed tau n S+ S0 S- signed"]
    for x in sel:
        lines.append(f"{x['subset']:<9} R{x['r']}D{x['d']} s{x['seed']} "
                     f"{x['tau']:g} {x['n']} {x['plus']:.3f} {x['zero']:.3f} "
                     f"{x['minus']:.3f} {x['signed']:+.3f}")
    text="\n".join(lines)+"\n"; Path(str(prefix)+".txt").write_text(text,encoding="utf-8"); print(text)

    row_end = r"\\"
    tex = [r"\begin{table}[h]", r"\centering", r"\small",
           r"\caption{Magnitude-threshold sensitivity. $S^+=\Pr(\Delta>\tau)$, "
           r"$S^0=\Pr(|\Delta|\leq\tau)$ and $S^-=\Pr(\Delta<-\tau)$. "
           r"Pooled document shares are descriptive; cross-seed ranges use run-level summaries.}",
           r"\label{tab:magnitude-threshold}",
           r"\begin{tabular}{llrrrrr}", r"\toprule",
           (r"population & $\tau$ & pooled $S^+$ & pooled $S^0$ & pooled $S^-$ & "
            r"max cell range in $S^+$ & cells $>0.3$ " + row_end), r"\midrule"]
    for subset in ("all", "mass>=0.5"):
        for tau in THRESHOLDS:
            p = next(x for x in pooled if x["subset"] == subset and x["tau"] == tau)
            cc = [x for x in cells if x["subset"] == subset and x["tau"] == tau]
            a = max(cc, key=lambda x: x["plus_range"])
            name = "all" if subset == "all" else r"mass $\geq0.5$"
            tex.append(f"{name} & {tau:g} & {p['plus']:.3f} & {p['zero']:.3f} & "
                       f"{p['minus']:.3f} & {a['plus_range']:.3f} "
                       f"(R{a['r']}, D{a['d']}) & "
                       f"{sum(x['plus_range'] > .3 for x in cc)} " + row_end)
        tex.append(r"\addlinespace")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(str(prefix)+".tex").write_text("\n".join(tex)+"\n", encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--perdoc",default="runs_g2/go_nogo_argmax_v1.txt.perdoc.jsonl"); ap.add_argument("--summary",default="runs_g2/go_nogo_argmax_v1.txt.jsonl"); ap.add_argument("--out",default="runs_g2/magnitude_thresholds_v1")
    a=ap.parse_args(); runs=load_runs(a.perdoc,a.summary); write_outputs(a.out,runs,*analyze(runs))

if __name__ == "__main__": main()
