"""Build a run ledger and summarize the saved terminal/trajectory records."""
import argparse
import glob
import json
import math
import os
import re
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

# Manuscript reference values keyed by claim; retain their provenance.
# The --check option reports discrepancies as MISMATCH.
PAPER = {
    "cells_range_gt_0.3": 13,
    "cells_straddle_0.5": 8,
    "max_range": 0.879,
    "pre_lt_0.20": 32,
    "pre_lt_0.05": 22,
    "pre_mass_recorded": 28,
    "pre_mass_ge_0.5": 17,
    "pre_reverse_to_rarity": 27,
    "within_gradient": 65,
    "nine_seed_sigma": 0.268,
    "nbrec30_ten_sigma": 0.127,
}

TRAIN_RE = re.compile(r"R(\d+)_D(\d+)_s(\d+)(?:_(.+))?\.jsonl$")

# Canonical terminal records are identified by (runs_* directory, tag).
# Bare tags are intentionally reused by architecture and rerun arms, so tag
# alone is not a run identity.
BASELINE_D8_IDS = tuple(
    ("runs_g2_seeds", tag) for tag in (
        "R3_D8_s0_grid", "R3_D8_s1_grid", "R3_D8_s2_grid",
        "R3_D8_s3", "R3_D8_s4", "R3_D8_s5", "R3_D8_s6",
        "R3_D8_s7", "R3_D8_s9"))
BASELINE3_D5_IDS = tuple(
    ("runs_g2", f"R3_D5_s{s}_grid") for s in range(3))
RECENCY01_D8_IDS = tuple(
    ("runs_nb", f"R3_D8_s{s}_nbrec01")
    for s in (0, 1, 2, 3, 5, 6, 7, 8, 9))
RECENCY01_D5_IDS = tuple(
    ("runs_nb", f"R3_D5_s{s}_nbrec01") for s in range(10))
RECENCY30_D8_IDS = tuple(
    ("runs_nb", f"R3_D8_s{s}_nbrec30") for s in range(10))
# Seeds 3,4,5,6,7,9 did not participate in selecting the widest grid cell.
# Report the same six seeds in baseline and 30% arms as a selection-sensitivity
# summary; this is descriptive and is not substituted for an independent test.
NEW_D8_SEEDS = (3, 4, 5, 6, 7, 9)
BASELINE_D8_NEW_IDS = tuple(
    ("runs_g2_seeds", f"R3_D8_s{s}") for s in NEW_D8_SEEDS)
RECENCY30_D8_MATCHED_NEW_IDS = tuple(
    ("runs_nb", f"R3_D8_s{s}_nbrec30") for s in NEW_D8_SEEDS)
# At 1%, new baseline seed 8 fails its state gate and relabeled seed 4 fails;
# the common state-passing new-seed subset is therefore only five pairs.
NEW_D8_SHARED_01_SEEDS = (3, 5, 6, 7, 9)
BASELINE_D8_MATCHED_01_IDS = tuple(
    ("runs_g2_seeds", f"R3_D8_s{s}") for s in NEW_D8_SHARED_01_SEEDS)
RECENCY01_D8_MATCHED_NEW_IDS = tuple(
    ("runs_nb", f"R3_D8_s{s}_nbrec01") for s in NEW_D8_SHARED_01_SEEDS)


def sd1(x) -> float:
    """ddof=1。全文所有 sigma 用这个，不用 range/d_n。"""
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def read_train(path: str) -> dict:
    """训练 jsonl -> loss 序列、探针序列、门判逃逸步。

    键名与 paper_numbers.read_run 一致：probe 记录取
    causal.break_rarity.{frac_expected, d_margin, mass_mean}。
    """
    loss, probes, esc = [], [], None
    with open(path, encoding="utf-8") as f:
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
                    probes.append({"step": o["step"],
                                   "frac": c["frac_expected"],
                                   "margin": c.get("d_margin"),
                                   "mass": c.get("mass_mean")})
    loss.sort()
    probes.sort(key=lambda p: p["step"])
    return {"loss": loss, "probes": probes, "esc": esc}


def load_gonogo(paths: List[str]) -> List[dict]:
    """go_nogo reports -> rows, deduplicated only within one runs_* arm.

    Directory is part of the identity: several arms intentionally reuse bare tags
    such as ``R3_D8_s0``.  Per-document caches are skipped because they carry
    vectors rather than the terminal run summary.
    """
    latest = {}
    # ``.txt`` contains a full raw snapshot; ``.jsonl`` is the append-only cache.
    # Read snapshots first and caches second so a force-recomputed cache row wins
    # deterministically, independent of copied-file mtimes.
    def report_order(path: str) -> Tuple[int, str]:
        base = os.path.basename(path)
        return (1 if base.endswith(".jsonl") else 0, path)

    for p in sorted(paths, key=report_order):
        base = os.path.basename(p)
        if "perdoc" in base:
            continue
        directory = os.path.basename(os.path.dirname(p))
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                objs = []
                if line.startswith("raw:"):
                    try:
                        objs = json.loads(line[4:])
                    except json.JSONDecodeError:
                        continue
                elif line.startswith("{"):
                    try:
                        objs = [json.loads(line)]
                    except json.JSONDecodeError:
                        continue
                for o in objs:
                    # A force-recomputed cache row must replace the older snapshot
                    # even when the report's legacy ``seed`` field is wrong.  The
                    # run seed is recovered from the tag later in ``build``.
                    key = (directory, o.get("tag"),
                           o.get("total_steps"), o.get("step"))
                    o["_dir"] = directory
                    o["_src"] = os.path.join(directory, base)
                    latest[key] = o
    return list(latest.values())


def build(root: str) -> Tuple[List[dict], List[str]]:
    """扫描 root 下所有 runs_* 目录，逐 run 一行。返回 (行, 缺失说明)。"""
    gaps = []
    train = {}
    for d in sorted(glob.glob(os.path.join(root, "runs_*"))):
        directory = os.path.basename(d)
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            base = os.path.basename(p)
            if base.startswith("go_nogo") or base.startswith("flat"):
                continue
            m = TRAIN_RE.search(base)
            if not m:
                continue
            r, dd, s, tag = int(m[1]), int(m[2]), int(m[3]), m[4] or ""
            train[(directory, r, dd, s, tag)] = read_train(p)

    gg_paths = sorted(glob.glob(os.path.join(root, "runs_*", "*go_nogo*.jsonl")))
    gg_paths += sorted(glob.glob(os.path.join(root, "runs_*", "*go_nogo*.txt")))
    if not gg_paths:
        gaps.append("no go_nogo report found under runs_*/")
    gg = load_gonogo(gg_paths)

    rows = []
    matched_train = set()
    for g in gg:
        tag = g.get("tag") or ""
        r, dd = g.get("r_old"), g.get("dd")
        reported_seed = g.get("seed", 0)
        suffix = ""
        m = TRAIN_RE.search(tag + ".jsonl") if tag else None
        run_seed = int(m[3]) if m else reported_seed
        if m:
            suffix = m[4] or ""
        directory = g.get("_dir") or ""
        candidates = (
            (directory, r, dd, run_seed, suffix),
            (directory, r, dd, run_seed, ""),
        )
        train_key = next((key for key in candidates if key in train), None)
        t = train.get(train_key) if train_key is not None else None
        row = dict(g)
        row["seed"] = run_seed
        row["_reported_seed"] = reported_seed
        row["_has_train"] = t is not None
        if t:
            matched_train.add(train_key)
            pre = [p for p in t["probes"]
                   if t["esc"] is not None and p["step"] < t["esc"]
                   and p["step"] >= 200]
            row["_esc_train"] = t["esc"]
            row["_n_probe"] = len(t["probes"])
            row["_pre_step"] = pre[-1]["step"] if pre else None
            row["_pre_frac"] = pre[-1]["frac"] if pre else None
            row["_pre_mass"] = pre[-1]["mass"] if pre else None
            post = [p["frac"] for p in t["probes"]
                    if t["esc"] is not None and p["step"] >= t["esc"]]
            row["_within_sd"] = sd1(post) if len(post) >= 5 else None
        rows.append(row)
    missing = [key for key in train if key not in matched_train]
    for directory, r, dd, seed, suffix in sorted(missing):
        gaps.append(
            f"training jsonl with no go_nogo row: {directory}/"
            f"R{r}_D{dd}_s{seed}_{suffix}")
    return rows, gaps


def main_grid(rows: List[dict]) -> List[dict]:
    """Published 75-run grid only; similarly named rerun arms are excluded."""
    return [r for r in rows
            if r.get("_dir") == "runs_g2"
            and r.get("total_steps") == 16000 and r.get("state") == "retr"
            and not r.get("p_break") and r.get("r_old") in (3, 5, 8, 12, 16)
            and (r.get("tag") or "").endswith("_grid")]


def recompute(rows: List[dict]) -> Dict[str, object]:
    """所有论文计数，一处派生。"""
    out = {}
    grid = main_grid(rows)
    out["n_grid"] = len(grid)

    cells = {}
    for r in grid:
        cells.setdefault((r["r_old"], r["dd"]), {})[r.get("seed", 0)] = r
    rng = {c: max(v["frac_positive"] for v in d.values())
              - min(v["frac_positive"] for v in d.values())
           for c, d in cells.items() if len(d) >= 3}
    out["cells_range_gt_0.3"] = sum(v > 0.3 for v in rng.values())
    out["cells_straddle_0.5"] = sum(
        1 for c, d in cells.items() if len(d) >= 3
        and min(v["frac_positive"] for v in d.values()) < 0.5
        < max(v["frac_positive"] for v in d.values()))
    out["max_range"] = round(max(rng.values()), 3) if rng else None

    # 逃逸前：分母阶梯逐层收窄，每层都报，'of 75' 才可核
    have = [r for r in grid if r.get("_has_train")]
    with_pre = [r for r in have if r.get("_pre_frac") is not None]
    out["ladder"] = {"loaded": len(grid), "with_train": len(have),
                     "with_pre_probe": len(with_pre),
                     "pre_step_ge_200": sum(
                         1 for r in with_pre if r["_pre_step"] >= 200)}
    lo20 = [r for r in with_pre if r["_pre_frac"] < 0.20]
    out["pre_lt_0.20"] = len(lo20)
    out["pre_lt_0.05"] = sum(1 for r in with_pre if r["_pre_frac"] < 0.05)
    with_mass = [r for r in lo20 if r.get("_pre_mass") is not None]
    out["pre_mass_recorded"] = len(with_mass)
    out["pre_mass_ge_0.5"] = sum(1 for r in with_mass if r["_pre_mass"] >= 0.5)
    # 27：逃逸前 <0.20 且终点 >0.5。摘要的 "reverses" 指这个事件。
    out["pre_reverse_to_rarity"] = sum(
        1 for r in lo20 if r["frac_positive"] > 0.5)
    out["pre_no_reverse_cells"] = sorted(
        f"R{r['r_old']}D{r['dd']}s{r.get('seed', 0)}"
        for r in lo20 if r["frac_positive"] <= 0.5)
    by_row = {}
    for r in lo20:
        by_row.setdefault(r["r_old"], {}).setdefault(r.get("seed", 0), 0)
        by_row[r["r_old"]][r.get("seed", 0)] += 1
    out["pre_lt_0.20_by_row_seed"] = by_row

    grad = [r for r in grid if r.get("d_med_hiK") is not None
            and r.get("d_med_loK") is not None]
    # 梯度顺着该格自己的效应方向：med>0 时应 hi>lo，med<0 时应 hi<lo。
    out["within_gradient"] = sum(
        1 for r in grad
        if (r["d_med_hiK"] - r["d_med_loK"]) * math.copysign(
            1.0, gated_med(r)) > 0)
    out["within_gradient_n"] = len(grad)

    # Confirmatory dispersion summaries use explicit (directory, tag) IDs, so
    # unrelated arms and duplicate bare tags cannot enter.
    b8 = [r["frac_positive"] for r in canonical_group(rows, BASELINE_D8_IDS)]
    r30 = [r["frac_positive"] for r in canonical_group(rows, RECENCY30_D8_IDS)]
    out["nine_seed_sigma"] = sd1(b8)
    out["nbrec30_ten_sigma"] = sd1(r30)
    return out


def gated_med(r: dict) -> float:
    return r.get("d_median_valid", r.get("d_median", float("nan")))


def positive_count(row: dict) -> int:
    """Recover and validate the integer numerator behind a sign fraction."""
    n = int(row["n"])
    frac = float(row["frac_positive"])
    k = int(round(frac * n))
    if not 0 <= k <= n or not math.isclose(frac, k / n, abs_tol=1e-12):
        raise ValueError(
            f"{row.get('tag')}: frac_positive={frac} is not an integer/n "
            f"fraction for n={n}")
    return k


def canonical_group(
        rows: List[dict], run_ids: Tuple[Tuple[str, str], ...]) -> List[dict]:
    """Return one terminal record per explicit (runs_* directory, tag) ID."""
    wanted = set(run_ids)
    found = {}
    for row in rows:
        run_id = (row.get("_dir") or "", row.get("tag") or "")
        if run_id not in wanted:
            continue
        if row.get("step") != 16000 or row.get("total_steps") != 16000:
            continue
        if row.get("state") != "retr" or (row.get("sched") or "cos") != "cos":
            continue
        positive_count(row)
        old = found.get(run_id)
        if old is not None and (old["n"], old["frac_positive"]) != (
                row["n"], row["frac_positive"]):
            raise ValueError(f"conflicting terminal records for {run_id}")
        found[run_id] = row
    missing = [run_id for run_id in run_ids if run_id not in found]
    if missing:
        raise ValueError(f"missing canonical terminal records: {missing}")
    return [found[run_id] for run_id in run_ids]


def summarize_group(rows: List[dict]) -> Dict[str, object]:
    fr = [float(r["frac_positive"]) for r in rows]
    return {"n": len(fr), "sd": round(sd1(fr), 4),
            "range": round(max(fr) - min(fr), 4),
            "mean_frac": round(float(np.mean(fr)), 4),
            "median_frac": round(float(np.median(fr)), 4),
            "counts": [f"{positive_count(r)}/{int(r['n'])}" for r in rows],
            "vals": [round(v, 3) for v in fr]}


def dose_curve(rows: List[dict]) -> Dict[str, object]:
    """Confirmatory dose summaries from explicit run IDs, never fuzzy metadata."""
    b8 = canonical_group(rows, BASELINE_D8_IDS)
    r018 = canonical_group(rows, RECENCY01_D8_IDS)
    r308 = canonical_group(rows, RECENCY30_D8_IDS)
    b8_new = canonical_group(rows, BASELINE_D8_NEW_IDS)
    r308_new = canonical_group(rows, RECENCY30_D8_MATCHED_NEW_IDS)
    b8_new01 = canonical_group(rows, BASELINE_D8_MATCHED_01_IDS)
    r018_new = canonical_group(rows, RECENCY01_D8_MATCHED_NEW_IDS)
    b5 = canonical_group(rows, BASELINE3_D5_IDS)
    r015 = canonical_group(rows, RECENCY01_D5_IDS)
    return {
        "R3_D8": {
            "p=0,baseline,n=3": summarize_group(b8[:3]),
            "p=0,baseline,n=9": summarize_group(b8),
            "p=0,new-seed-sensitivity,n=6": summarize_group(b8_new),
            "p=0,matched-new-for-.01,n=5": summarize_group(b8_new01),
            "p=.01,recency,n=3": summarize_group(r018[:3]),
            "p=.01,recency,n=9": summarize_group(r018),
            "p=.01,recency,matched-new,n=5": summarize_group(r018_new),
            "p=.30,recency,n=3": summarize_group(r308[:3]),
            "p=.30,recency,n=10": summarize_group(r308),
            "p=.30,recency,matched-new,n=6": summarize_group(r308_new)},
        "R3_D5": {
            "p=0,baseline,n=3": summarize_group(b5),
            "p=.01,recency,n=3": summarize_group(r015[:3]),
            "p=.01,recency,n=10": summarize_group(r015)}}


def dev(x: np.ndarray) -> np.ndarray:
    return np.abs(x - np.median(x))


def variance_ratio(x: np.ndarray, y: np.ndarray) -> float:
    den = y.var(ddof=1)
    return x.var(ddof=1) / den if den else float("inf")


def variance_test(aliased: List[float], relabeled: List[float]) -> Dict[str, float]:
    """Exact one-sided tests over all fixed-size unpaired regroupings."""
    a, b = np.asarray(aliased, float), np.asarray(relabeled, float)
    pooled = np.concatenate([a, b])
    na = len(a)
    obs_bf = dev(a).mean() - dev(b).mean()
    obs_vr = variance_ratio(a, b)
    ge_bf = ge_bf2 = ge_vr = total = 0
    all_idx = set(range(len(pooled)))
    for left in combinations(range(len(pooled)), na):
        li = np.fromiter(left, dtype=int)
        ri = np.fromiter(all_idx.difference(left), dtype=int)
        x, y = pooled[li], pooled[ri]
        stat = dev(x).mean() - dev(y).mean()
        ge_bf += stat >= obs_bf - 1e-15
        ge_bf2 += abs(stat) >= abs(obs_bf) - 1e-15
        ge_vr += variance_ratio(x, y) >= obs_vr - 1e-15
        total += 1
    return {"n_aliased": len(a), "n_relabeled": len(b),
            "n_partitions": total,
            "sd_aliased": round(sd1(a), 6),
            "sd_relabeled": round(sd1(b), 6),
            "F": round(obs_vr, 6), "df": f"({len(a)-1},{len(b)-1})",
            "bf_stat": round(float(obs_bf), 6),
            "bf_p_onesided": round(ge_bf / total, 6),
            "bf_p_twosided": round(ge_bf2 / total, 6),
            "varratio_stat": round(float(obs_vr), 6),
            "varratio_p_onesided": round(ge_vr / total, 6)}


def paired_variance_test(
        aliased: List[float], relabeled: List[float]) -> Dict[str, float]:
    """Exact paired-label-swap sensitivity analysis for shared seeds.

    Each of the 2^n assignments independently exchanges the two condition labels
    within a seed. Medians and absolute deviations are recomputed after every
    exchange. This describes sensitivity to treating seed as a pair; it does not
    turn the non-randomized training arms into a randomized experiment.
    """
    a, b = np.asarray(aliased, float), np.asarray(relabeled, float)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("paired test requires equal-length groups with n >= 2")
    obs_bf = dev(a).mean() - dev(b).mean()
    obs_vr = variance_ratio(a, b)
    ge_bf = ge_bf2 = ge_vr = 0
    total = 1 << len(a)
    for mask in range(total):
        swap = np.fromiter(
            ((mask >> i) & 1 for i in range(len(a))), dtype=bool)
        x = np.where(swap, b, a)
        y = np.where(swap, a, b)
        stat = dev(x).mean() - dev(y).mean()
        ge_bf += stat >= obs_bf - 1e-15
        ge_bf2 += abs(stat) >= abs(obs_bf) - 1e-15
        ge_vr += variance_ratio(x, y) >= obs_vr - 1e-15
    return {"n_pairs": len(a), "n_assignments": total,
            "sd_aliased": round(sd1(a), 6),
            "sd_relabeled": round(sd1(b), 6),
            "bf_stat": round(float(obs_bf), 6),
            "bf_p_onesided": round(ge_bf / total, 6),
            "bf_p_twosided": round(ge_bf2 / total, 6),
            "varratio_stat": round(float(obs_vr), 6),
            "varratio_p_onesided": round(ge_vr / total, 6)}



FIELDS = ["_dir", "tag", "r_old", "dd", "seed", "_reported_seed",
          "p_break", "truth_rule", "sched", "total_steps", "step", "state",
          "acc", "copy_acc",
          "escape_step", "n", "n_positive", "n_docs_used", "n_docs_excl",
          "frac_positive", "sign_p", "d_median", "d_median_valid",
          "d_med_hiK", "d_med_loK", "ctrl_median", "mass", "frac_mass_ok",
          "sw_keep", "p_rec", "p_rar", "_has_train", "_esc_train",
          "_n_probe", "_pre_step", "_pre_frac", "_pre_mass", "_within_sd",
          "_src"]


def write_tsv(rows: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(FIELDS) + "\n")
        for r in sorted(rows, key=lambda x: (
                x.get("_dir") or "", x.get("tag") or "",
                x.get("r_old") or 0, x.get("dd") or 0, x.get("seed") or 0)):
            out = dict(r)
            if out.get("n") is not None and out.get("frac_positive") is not None:
                out["n_positive"] = positive_count(out)
            f.write("\t".join("" if out.get(k) is None else str(out.get(k))
                              for k in FIELDS) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="含 runs_* 目录的根。默认在 '.'、'..'、脚本上一级里"
                         "找第一个有 runs_* 的，所以从 Paper5 或 Github 跑都对")
    ap.add_argument("--out", default="ledger.tsv")
    ap.add_argument("--check", action="store_true", help="只对照，不写 TSV")
    ap.add_argument(
        "--n-perm", type=int, default=None,
        help="deprecated compatibility option; confirmatory tests are exhaustive")
    a = ap.parse_args()

    if a.root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (".", "..", os.path.dirname(here), here):
            if glob.glob(os.path.join(cand, "runs_*")):
                a.root = cand
                break
        else:
            ap.error("找不到任何 runs_* 目录。用 --root 指到含 runs_* 的目录，"
                     "从 Paper5 跑是 --root .")
    print(f"# root = {os.path.abspath(a.root)}")
    print(f"# runs_* dirs = "
          f"{[os.path.basename(d) for d in sorted(glob.glob(os.path.join(a.root, 'runs_*')))]}")

    rows, gaps = build(a.root)
    print(f"# {len(rows)} go_nogo rows, "
          f"{sum(1 for r in rows if r.get('_has_train'))} with training jsonl")
    if not a.check:
        write_tsv(rows, a.out)
        print(f"# wrote {a.out}")

    print("\n=== 论文计数对照 ===")
    got = recompute(rows)
    for k, want in PAPER.items():
        if k not in got:
            continue
        have = got[k]
        if have is None:
            print(f"NO DATA  {k}: paper={want}  (ledger produced nothing --- "
                  f"check --root and the main_grid filter)")
            continue
        tol = 1e-9 if isinstance(want, int) else 0.001
        ok = abs(have - want) <= tol
        print(f"{'OK      ' if ok else 'MISMATCH'} {k}: "
              f"ledger={have}  paper={want}")

    print("\n分母阶梯（'of 75' 只有每层都等于 75 才成立）")
    for k, v in got["ladder"].items():
        print(f"  {k:20s} {v}")
    print(f"\n逃逸前 <0.20 的逐 (R_old, seed) 分布  {got['pre_lt_0.20_by_row_seed']}")
    print(f"未反转的 {len(got['pre_no_reverse_cells'])} 个: "
          f"{got['pre_no_reverse_cells']}")

    print("\n=== 剂量曲线（直算 sigma，匹配 n）===")
    for cell, cur in dose_curve(rows).items():
        print(f"\n{cell}")
        for key, v in cur.items():
            print(f"  {key:24s} n={v['n']:2d}  sd={v['sd']:.4f}  "
                  f"range={v['range']:.4f}  med={v['median_frac']:.3f}")
            print(f"  {'':24s} {v['vals']}")

    print("\n=== 精确方差检验（canonical terminal records）===")
    baseline_rows = canonical_group(rows, BASELINE_D8_IDS)
    al = [r["frac_positive"] for r in baseline_rows]
    baseline_by_seed = {r["seed"]: r["frac_positive"]
                        for r in baseline_rows}
    comparisons = (
        ("p_break=0.01", RECENCY01_D8_IDS),
        ("p_break=0.30", RECENCY30_D8_IDS),
    )
    for name, run_ids in comparisons:
        relabeled_rows = canonical_group(rows, run_ids)
        relabeled = [r["frac_positive"] for r in relabeled_rows]
        print(f"\n{name}: aliased n={len(al)}  relabeled n={len(relabeled)}")
        print("  unpaired exhaustive regrouping")
        for k, v in variance_test(al, relabeled).items():
            print(f"    {k:22s} {v}")

        relabeled_by_seed = {r["seed"]: r["frac_positive"]
                             for r in relabeled_rows}
        shared = sorted(set(baseline_by_seed) & set(relabeled_by_seed))
        paired_a = [baseline_by_seed[s] for s in shared]
        paired_b = [relabeled_by_seed[s] for s in shared]
        print(f"  paired-label-swap sensitivity, shared seeds={shared}")
        for k, v in paired_variance_test(paired_a, paired_b).items():
            print(f"    {k:22s} {v}")

    if gaps:
        print("\n=== 缺口 ===")
        for g in gaps:
            print(f"  {g}")


if __name__ == "__main__":
    main()
