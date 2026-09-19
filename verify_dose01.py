"""Reconstruct the exploratory 1% dose comparisons from saved records."""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys
from collections import Counter

# 论文当前写的值；所有 SD 均由未舍入 frac+ 以 n-1 分母计算。
PAPER = {
    "D8_n": 9, "D8_sd": 0.132, "D8_range": 0.311, "D8_med": 0.043,
    "D5_n": 10, "D5_sd": 0.165, "D5_range": 0.445, "D5_med": 0.021,
    "D5_sd_no_s9": 0.171,
    "D8_sd3": 0.149, "D5_sd3": 0.126,
    "aliased_n": 9, "aliased_sd": 0.268,
    "F_001": 4.10, "BF_001": 0.070, "VR_001": 0.043,
    "ratio_D8": 1.12, "ratio_D5": 1.31,
    "neither_runs": 5,
}

BASELINE_D8 = tuple(
    ("runs_g2_seeds", f"R3_D8_s{s}{'_grid' if s < 3 else ''}")
    for s in (0, 1, 2, 3, 4, 5, 6, 7, 9))

# Confirmatory baseline records are selected only by the explicit IDs above.
# No metadata blacklist is used: several unrelated arms intentionally share
# r_old/dd/schedule fields and even reuse bare tags.
TOL = 0.0015
NAN = float("nan")


def fin(xs):
    """只留有限值。d_median_valid 在无过门文档时是 NaN。"""
    return [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]


def num(x, w=".3f"):
    """缺字段或 NaN 时打印 '--' 而不是抛 TypeError。"""
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return "--"
    return format(x, w)


def chk(name, got, want, tol=TOL):
    if got is None:
        print(f"  NO DATA  {name}: paper={want}")
        return
    tol = tol if isinstance(want, float) else 0
    ok = abs(got - want) <= tol
    print(f"  {'OK      ' if ok else 'MISMATCH'} {name}: "
          f"computed={got:.4f}  paper={want}")


def load_rows(root):
    """所有 go_nogo 产物 -> {(目录, tag): row}，缓存覆盖旧快照。

    旧报表的新增种子有时把 ``seed`` 字段重复写成 0；运行身份和真实种子
    因而从 tag 恢复，原字段保存在 ``_reported_seed`` 供审计。
    """
    rows, n_txt, n_jsonl = {}, 0, 0

    def keep(o, directory):
        if not isinstance(o, dict) or "tag" not in o:
            return False
        o["_dir"] = directory
        o["_reported_seed"] = o.get("seed")
        try:
            seed_part = o["tag"].split("_")[2]
            if seed_part.startswith("s"):
                o["seed"] = int(seed_part[1:])
        except (IndexError, TypeError, ValueError):
            pass
        rows[(directory, o["tag"])] = o
        return True

    for p in sorted(glob.glob(os.path.join(root, "runs_*", "*go_nogo*.txt"))):
        d = os.path.basename(os.path.dirname(p))
        for line in open(p, encoding="utf-8", errors="replace"):
            if not line.startswith("raw:"):
                continue
            try:
                for o in json.loads(line[4:]):
                    n_txt += int(keep(o, d))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    for p in sorted(glob.glob(os.path.join(root, "runs_*", "*go_nogo*.jsonl"))):
        if "perdoc" in os.path.basename(p):
            continue
        d = os.path.basename(os.path.dirname(p))
        for line in open(p, encoding="utf-8", errors="replace"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_jsonl += int(keep(o, d))
    print(f"# {len(rows)} unique (dir, tag) "
          f"({n_txt} rows from txt, {n_jsonl} from jsonl caches)")
    return rows


def sel(rows, dd, p_break, sub="nbrec01", gated=True):
    out = [r for r in rows.values()
           if r.get("_dir") == "runs_nb"
           and r.get("dd") == dd
           and abs((r.get("p_break") or 0.0) - p_break) < 1e-9
           and sub in r.get("tag", "")
           and (r.get("state") == "retr" if gated else True)]
    return sorted(out, key=lambda r: r["tag"])


def c123(rows):
    """C1/C2/C3：两个 cell 在 0.01 的 sigma、range、中位 median、剔 s9。"""
    print("\n=== C1/C2 两个 cell 在 p_break=0.01 ===")
    out = {}
    for dd in (8, 5):
        allr, g = sel(rows, dd, 0.01, gated=False), sel(rows, dd, 0.01)
        if not allr:
            print(f"\nD{dd}: 无数据。检查 runs_nb 下是否有 nbrec01 的 go_nogo。")
            continue
        f = fin([r.get("frac_positive") for r in g])
        # d_median_valid 是逐篇 mass 门后的中位数，与 tab:nbdose 的
        # median Δ 列同口径；d_median 未门控，不能混用。
        m = fin([r.get("d_median_valid") for r in g])
        sd = st.stdev(f) if len(f) > 1 else None
        print(f"\nD{dd}  trained={len(allr)}  gated={len(g)}")
        print(f"  frac+ = {[round(x, 4) for x in sorted(f)]}")
        chk(f"D{dd}_n", float(len(g)), float(PAPER[f'D{dd}_n']))
        chk(f"D{dd}_sd", sd, PAPER[f"D{dd}_sd"])
        chk(f"D{dd}_range", (max(f) - min(f)) if f else None,
            PAPER[f"D{dd}_range"], tol=0.006)
        chk(f"D{dd}_med", st.median(m) if m else None, PAPER[f"D{dd}_med"],
            tol=0.006)
        for r in allr:
            if r.get("state") != "retr":
                print(f"  not gated: {r['tag']}  state={r.get('state')}  "
                      f"acc={num(r.get('acc'))}  copy={num(r.get('copy_acc'))}")
        # 三种子（前三个 seed）与大样本的比值 → C5
        f3 = fin([r.get("frac_positive") for r in g
                  if r.get("seed") in (0, 1, 2)])
        if len(f3) > 1:
            sd3 = st.stdev(f3)
            chk(f"D{dd}_sd3", sd3, PAPER[f"D{dd}_sd3"])
            if sd:
                chk(f"ratio_D{dd}", max(sd3, sd) / min(sd3, sd),
                    PAPER[f"ratio_D{dd}"], tol=0.02)
        # swKeep / 行为标签，供 B6/B7/B11 的措辞
        sk = fin([r.get("sw_keep_grid") for r in g])
        ska = fin([r.get("sw_keep") for r in g])
        pr = fin([r.get("p_rec") for r in g])
        pa = fin([r.get("p_rar") for r in g])
        if sk:
            worst = max(g, key=lambda r: r.get("sw_keep_grid") or -1)
            second = f", 2nd largest={sorted(sk)[-2]:.4f}" if len(sk) > 1 else ""
            print(f"  swKeep_grid max={max(sk):.4f} @ {worst['tag']}{second}")
        if ska:
            print(f"  swKeep_arm  max={max(ska):.4f}")
        if pr and pa:
            print(f"  p_rec {min(pr):.3f}-{max(pr):.3f}   "
                  f"p_rar {min(pa):.3f}-{max(pa):.3f}")
        nd = [(r["tag"], r.get("frac_positive"), r.get("sign_p"))
              for r in g if (r.get("sign_p") or 0) > 0.05]
        for t, fp, sp in nd:
            print(f"  non-decisive: {t}  frac+={num(fp)}  p={num(sp, '.2f')}")
        out[dd] = {
            int(r["seed"]): float(r["frac_positive"])
            for r in g if isinstance(r.get("frac_positive"), (int, float))
            and math.isfinite(r["frac_positive"])
        }
    print("\n=== C3 D5 剔掉 s9 ===")
    f5 = fin([r.get("frac_positive") for r in sel(rows, 5, 0.01)
              if not r.get("tag", "").endswith("s9_nbrec01")])
    chk("D5_sd_no_s9", st.stdev(f5) if len(f5) > 1 else None,
        PAPER["D5_sd_no_s9"])
    print(f"  n={len(f5)}")
    return out


def aliased9(rows):
    """C4 前半：从明确的九个 terminal run ID 读取 aliased 锚点。"""
    print("\n=== C4a aliased 锚点（canonical IDs）===")
    keep = {}
    for key in BASELINE_D8:
        r = rows.get(key)
        if r is None:
            print(f"  MISSING  {key[0]:<16} {key[1]}")
            continue
        if (r.get("state") != "retr" or r.get("step") != 16000
                or r.get("total_steps") != 16000
                or (r.get("sched") or "cos") != "cos"):
            print(f"  INVALID  {key[0]:<16} {key[1]}  "
                  f"state={r.get('state')} step={r.get('step')}")
            continue
        seed = int(r["seed"])
        keep[seed] = float(r["frac_positive"])
        print(f"  kept     {key[0]:<16} {key[1]:<22} "
              f"seed={seed}  frac+={num(r.get('frac_positive'), '.6f')}")
    f = list(keep.values())
    chk("aliased_n", float(len(f)), float(PAPER["aliased_n"]))
    chk("aliased_sd", st.stdev(f) if len(f) > 1 else None,
        PAPER["aliased_sd"])
    return keep


def c4(al_by_seed, rel_by_seed):
    """C4 后半：未配对主复算，并报告共享-seed配对敏感性。"""
    print("\n=== C4b 精确方差检验 0.01 ===")
    a = [al_by_seed[s] for s in sorted(al_by_seed)]
    b = [rel_by_seed[s] for s in sorted(rel_by_seed)]
    if len(a) < 2 or len(b) < 2:
        print("  样本不足，跳过。")
        return
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from run_ledger import paired_variance_test, variance_test

    print("  unpaired exhaustive regrouping")
    res = variance_test(a, b)
    for k, v in res.items():
        print(f"    {k:22s} {v}")
    chk("F_001", res["F"], PAPER["F_001"], tol=0.05)
    chk("BF_001", res["bf_p_onesided"], PAPER["BF_001"], tol=0.001)
    chk("VR_001", res["varratio_p_onesided"], PAPER["VR_001"], tol=0.001)

    shared = sorted(set(al_by_seed) & set(rel_by_seed))
    print(f"  paired-label-swap sensitivity, shared seeds={shared}")
    paired = paired_variance_test(
        [al_by_seed[s] for s in shared],
        [rel_by_seed[s] for s in shared])
    for k, v in paired.items():
        print(f"    {k:22s} {v}")
    print("  ^ 主复算穷举所有固定样本量分组并逐次重算中位数；配对结果仅为"
          "共享 seed 的敏感性分析，不把训练臂解释成随机配对实验。")


def c6(rows):
    """app:power 那句 'N runs in App.nonalias acquire neither mechanism'。

    arm 里所有 p_break>0 且非 retr 的 run。state 以 pos 开头 = 终止在位置
    天花板，none = 两者都没获得；论文那句把两类合成一个数，故一并打印，
    由你确认合并口径与原文一致。
    """
    print("\n=== C6 arm 里未过门的 run ===")
    bad = [r for r in rows.values()
           if (r.get("p_break") or 0) > 0 and r.get("state") != "retr"]
    for r in sorted(bad, key=lambda x: x["tag"]):
        print(f"  {r['tag']:<26} state={str(r.get('state')):>8}  "
              f"acc={num(r.get('acc'))}  copy={num(r.get('copy_acc'))}"
              f"  swKeep_grid={num(r.get('sw_keep_grid'), '.4f')}")
    kinds = Counter("positional" if str(r.get("state")).startswith("pos")
                    else str(r.get("state")) for r in bad)
    print(f"  by state: {dict(kinds)}")
    chk("neither_runs", float(len(bad)), float(PAPER["neither_runs"]))
    print("  ^ 这个数填 app:power 1797 行的 'and N runs in App.nonalias'。"
          "\n    注意 ΔD=5 的 NEITHER 种子原文写在复制小节里、可能未计入"
          "\n    旧的 'three'，所以差值可能是 2 而不是 1。")


def c7(root):
    """C7：两个 cell 各十个 run 的 meta 是否逐字段一致（除 seed）。

    ``train.eval_docs`` 只控制状态诊断的评估样本量：前三个种子为 4000，
    新增种子为 20000。它仍被显式打印，但不改变训练流或终端探针总体；
    除这一已知差异外，任何配置差异都判为异常。
    """
    print("\n=== C7 meta 逐字段比对 ===")
    d = os.path.join(root, "runs_nb")
    for dd in (8, 5):
        ms, miss = {}, []
        for s in range(10):
            tag = f"R3_D{dd}_s{s}_nbrec01"
            p = os.path.join(d, tag + ".jsonl")
            if not os.path.exists(p):
                miss.append(tag)
                continue
            with open(p, encoding="utf-8") as f:
                o = json.loads(f.readline())
            if o.get("kind") != "meta":
                miss.append(tag + " (首行不是 meta)")
                continue
            ms[s] = o
        if miss:
            print(f"  D{dd} 缺: {miss}")
        if len(ms) < 2:
            continue
        unexpected = 0
        for blk in ("spec", "corpus", "train"):
            keys = set()
            for o in ms.values():
                keys |= set((o.get(blk) or {}).keys())
            for k in sorted(keys):
                vals = {}
                for s, o in ms.items():
                    vals.setdefault(
                        json.dumps((o.get(blk) or {}).get(k), sort_keys=True),
                        []).append(s)
                if len(vals) <= 1 or k in ("seed", "data_seed",
                                           "name", "tag", "out"):
                    continue
                eval_only = (blk == "train" and k == "eval_docs"
                             and set(vals) == {"4000", "20000"})
                unexpected += int(not eval_only)
                note = " (acknowledged evaluation-only difference)" if eval_only else ""
                print(f"  D{dd} {blk}.{k} differs{note}:")
                for v, ss in vals.items():
                    print(f"      seeds {ss}: {v[:90]}")
        print(f"  D{dd}: {len(ms)} metas, {unexpected} unexpected field(s)"
              + ("  <-- 必须为 0" if unexpected else "  OK"))


def c8(root):
    """C8：核对 app:surface 的臂特异 value-pool 大小。

    原始 ADJ 池有 170 项；rendered-from-scratch 配置使用 150 个值，
    Qwen fine-tuned 臂使用经 tokenizer 过滤后仍为单 token 的 145 项。
    The two pool sizes belong to different arms; both mappings must be injective.
    """
    print("\n=== C8 形容词池实际大小 ===")
    import ast
    for name in ("nl_render.py", "nl_generator.py", "ft_pool.py",
                 "ft_tokcheck.py", "nl_corpus.py"):
        p = os.path.join(root, name)
        if not os.path.exists(p):
            continue
        src = open(p, encoding="utf-8", errors="replace").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                continue
            if len(node.elts) < 20:
                continue
            if all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                   for e in node.elts):
                print(f"  {name}:{node.lineno}  string literal of "
                      f"len {len(node.elts)}")
        for i, line in enumerate(src.splitlines(), 1):
            low = line.lower()
            if ("adj" in low or "n_values" in low) and (
                    "145" in line or "150" in line or "512" in line):
                print(f"  {name}:{i}  {line.strip()[:100]}")
    print("  ^ 两个数字属于不同实验臂：rendered-from-scratch 使用"
          "\n    n_values=150；Qwen fine-tuned 臂使用 tokenizer 过滤后"
          "\n    的 145 个单-token adjectives。这是两个不同实验臂的配置。")


def c9(rows):
    """C9：'roughly 153' 与 'a rate near two per cent' 的分母。

    先按 tag 后缀分组，由你决定哪些臂进分母；bit-identical 重跑（fixband
    d=4、flat 重执行）和逐位复跑不该计入。三个数必须同源。
    """
    print("\n=== C9 训练总数的分母 ===")
    print(f"  unique tags (上界): {len(rows)}")
    suf = Counter(r["tag"].split("_")[-1] for r in rows.values())
    for k, v in sorted(suf.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<14} {v}")
    split = Counter()
    for r in rows.values():
        st_ = str(r.get("state"))
        split["retr" if st_ == "retr" else
              "positional" if st_.startswith("pos") else st_] += 1
    print(f"  by state: {dict(split)}")
    print("  ^ 定下分母后，app:surface 的 'roughly 153'、app:power 的"
          "\n    'near two per cent' 和 'probability near 0.94' 三处一起改。"
          "\n    两个条件劈叉的 gate failure（copy>0.95 且 acc<0.99）是分子，"
          "\n    与这里的 positional/none 不是同一类，别混。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="含 runs_* 的目录。默认在 '.'、'..'、脚本上一级里找")
    ap.add_argument(
        "--n-perm", type=int, default=None,
        help="deprecated compatibility option; exact tests enumerate all partitions")
    a = ap.parse_args()
    if a.root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (".", "..", os.path.dirname(here), here):
            if glob.glob(os.path.join(cand, "runs_*")):
                a.root = cand
                break
        else:
            ap.error("找不到 runs_*。用 --root 指到含 runs_* 的目录。")
    print(f"# root = {os.path.abspath(a.root)}")

    rows = load_rows(a.root)
    rel = c123(rows)
    al = aliased9(rows)
    if al and rel.get(8):
        c4(al, rel[8])
    c6(rows)
    c7(a.root)
    c8(a.root)
    c9(rows)
    print("\nCompleted. Investigate MISMATCH entries against run identities, field definitions and source records.")


if __name__ == "__main__":
    main()
