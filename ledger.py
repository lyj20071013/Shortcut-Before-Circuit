"""实验台账。扫描输出目录，把每个 run 的口径、配置、读数汇总成一张表。

为什么需要它：数据散在 runs/ 与 runs_g2/ 以及若干 aux txt 里，而三套口径
（12k/200篇/均值、16k/200篇/均值、go_nogo/400篇/中位数）之间不可比较。
正文引用的每个数都必须能指回 (文件, step, 文档数, 汇总方式)，靠记忆拼会出错。

用法：
  python ledger.py                          扫默认目录
  python ledger.py --dirs runs runs_g2      指定目录
  python ledger.py --csv ledger.csv         另存 csv 便于排序

口径分类（class 列）：
  A  total_steps=12000，train.py 探针路径，200 篇，均值。第一轮，只作附录
     的捷径瞬态证据与预算稳健性对照，禁止与 B/C 混入同一张图。
  B  total_steps=16000，同上路径。第二轮主网格。frac+ 可直接用（与 C 是
     嵌套子样本，同代码同文档流），中位数/IQR/ctrl 缺失。
  C  go_nogo.py，400 篇，中位数 + IQR + 二项检验 + 逐篇 mass + ctrl。
     唯一完整口径。本脚本从 <txt>.jsonl 缓存读取。
  X  其他步数（4k/32k 等辅助臂），单格，只用于预算稳健性。
"""
import argparse
import glob
import json
import math
import os

NAN = float("nan")
COPY_FLOOR = 0.95


def sign_test_p(k, n):
    """精确二项检验，双侧。frac+ 是主 DV，必须带 p —— n≈180 时 0.41 与 0.59
    都不显著，肉眼看表会当成方向。整数除法避免 2**n 溢出。"""
    if not n:
        return NAN
    tail = min(k, n - k)
    num = sum(math.comb(n, i) for i in range(tail + 1))
    return min(1.0, 2.0 * num / (2 ** n))


def pos_ceil(lo, hi):
    """纯位置规则的解析上限。unmarked 时每条语句恰 4 token、答案恒在倒数第
    1+ΔD 条，故命中率 = 1/|supp(ΔD)|。"""
    return 1.0 / (hi - lo + 1) if hi >= lo else NAN


def classify_state(acc, copy_acc, lo, hi):
    if acc != acc or copy_acc != copy_acc:
        return "?"
    if copy_acc >= COPY_FLOOR and acc >= 0.99:
        return "retr"
    ceil = pos_ceil(lo, hi)
    if copy_acc < 0.5 and ceil == ceil and acc >= 0.5 * ceil:
        return f"pos{acc / ceil:.0%}"
    return "none"


def kou_jing(total_steps, source):
    """口径标签。同一格在不同 total_steps 下的读数不可比 —— 实测 R3/D2 的
    frac+ 在 12k/16k/32k 分别是 0.43/0.41/0.89。"""
    if source == "go_nogo":
        return "C"
    if total_steps == 12000:
        return "A"
    if total_steps == 16000:
        return "B"
    return "X"


def scan_jsonl(path):
    """返回 (meta, 末条 eval, 末条 probe, 逃逸步, eval 计数)。
    末行可能因训练进行中而截断，逐行容错。"""
    meta = ev = probe = None
    esc = NAN
    n_eval = 0
    with open(path) as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = o.get("kind")
            if k == "meta":
                meta = o
            elif k == "eval":
                ev = o
                n_eval += 1
                if esc != esc and o.get("copy_acc", 0.0) >= COPY_FLOOR:
                    esc = o["step"]
            elif k == "probe":
                probe = o
    return meta, ev, probe, esc, n_eval


def row_from_jsonl(path):
    meta, ev, probe, esc, n_eval = scan_jsonl(path)
    if meta is None:
        return None
    tag = os.path.basename(path)[:-6]
    d = os.path.dirname(path)
    corpus = meta.get("corpus", {})
    spec = meta.get("spec", {})
    train = meta.get("train", {})
    ev = ev or {}
    total = train.get("total_steps", NAN)
    lo = corpus.get("delta_d_lo", 0)
    hi = corpus.get("delta_d_hi", 0)

    c = ((probe or {}).get("causal") or {}).get("break_rarity") or {}
    n = c.get("n", 0)
    fx = c.get("frac_expected", NAN)
    p = sign_test_p(round(fx * n), n) if (n and fx == fx) else NAN

    acc = ev.get("acc", NAN)
    ca = ev.get("copy_acc", NAN)
    return dict(
        tag=tag, dir=d, source="train.py",
        cls=kou_jing(total, "train.py"),
        r_old=corpus.get("r_old_lo", NAN), dd_lo=lo, dd_hi=hi,
        seed=corpus.get("seed", NAN),
        total_steps=total, last_eval=ev.get("step", NAN),
        probe_step=(probe or {}).get("step", NAN), n_eval=n_eval,
        n_values=spec.get("n_values", NAN),
        n_entities=spec.get("n_entities", NAN),
        stmts=f"{corpus.get('n_stmts_lo','?')}-{corpus.get('n_stmts_hi','?')}",
        pos_ceil=pos_ceil(lo, hi),
        acc=acc, copy_acc=ca, tail0=ev.get("acc_tail0", NAN),
        escape=esc, state=classify_state(acc, ca, lo, hi),
        n_docs=n, yield_rate=c.get("yield_rate", NAN),
        summary="mean", d_mid=c.get("d_margin", NAN),
        frac_pos=fx, sign_p=p, mass=c.get("mass_mean", NAN),
        ctrl=NAN, iqr="",
        has_pt=os.path.exists(path[:-6] + ".pt"),
        done=(ev.get("step") == total if total == total else False))


def rows_from_gonogo(path):
    """go_nogo 的增量缓存。d_mid 用中位数：均值在同一格 8 倍预算上动 40%
    （+2.713→+3.786）而中位数动 2%，重尾下均值不可作主 DV。"""
    out = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            lo, hi = r.get("dd_lo", 0), r.get("dd_hi", 0)
            q25, q75 = r.get("d_q25", NAN), r.get("d_q75", NAN)
            out.append(dict(
                tag=r["tag"], dir=os.path.dirname(path), source="go_nogo",
                cls="C", r_old=r.get("r_old", NAN), dd_lo=lo, dd_hi=hi,
                seed=r.get("seed", NAN),
                total_steps=r.get("total_steps", NAN),
                last_eval=r.get("step", NAN), probe_step=r.get("step", NAN),
                n_eval=NAN, n_values=NAN, n_entities=NAN, stmts="",
                pos_ceil=pos_ceil(lo, hi),
                acc=r.get("acc", NAN), copy_acc=r.get("copy_acc", NAN),
                tail0=r.get("tail0", NAN), escape=r.get("escape_step", NAN),
                state=r.get("state", "?"), n_docs=r.get("n", 0),
                yield_rate=r.get("yield_rate", NAN),
                summary="median", d_mid=r.get("d_median", NAN),
                frac_pos=r.get("frac_positive", NAN),
                sign_p=r.get("sign_p", NAN), mass=r.get("mass", NAN),
                ctrl=r.get("ctrl_median", r.get("ctrl_margin", NAN)),
                iqr=f"[{q25:+.2f},{q75:+.2f}]" if q25 == q25 else "",
                has_pt=True, done=True))
    return out


def fmt(rows):
    hdr = (f"{'cls':>3} {'tag':<24} {'R':>3} {'band':>7} {'s':>2} "
           f"{'steps':>6} {'read':>6} {'state':>7} {'esc':>5} {'acc':>6} "
           f"{'copy':>6} {'n':>4} {'yld':>5} {'sum':>6} {'Δ':>8} "
           f"{'IQR':>16} {'frac+':>6} {'p':>9} {'mass':>6} {'ctrl':>7} {'dir'}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        band = f"[{r['dd_lo']},{r['dd_hi']}]"
        mark = "" if r["done"] else " ⚠未完"
        lines.append(
            f"{r['cls']:>3} {r['tag']:<24} {r['r_old']:>3} {band:>7} "
            f"{r['seed']:>2} {r['total_steps']:>6} {r['probe_step']:>6} "
            f"{r['state']:>7} {r['escape']:>5.0f} {r['acc']:>6.3f} "
            f"{r['copy_acc']:>6.3f} {r['n_docs']:>4} {r['yield_rate']:>5.2f} "
            f"{r['summary']:>6} {r['d_mid']:>+8.3f} {r['iqr']:>16} "
            f"{r['frac_pos']:>6.2f} {r['sign_p']:>9.1e} {r['mass']:>6.2f} "
            f"{r['ctrl']:>+7.3f} {r['dir']}{mark}")
    return lines


def phase_map(rows, cls):
    """相图。只取 state=retr 且 mass 达标；跨 seed 用中位数并报 seed 极差
    （预注册判据：格间差异须超过格内 seed 极差才算有意义）。"""
    sel = [r for r in rows if r["cls"] == cls and r["state"] == "retr"
           and (r["mass"] != r["mass"] or r["mass"] >= 0.5)]
    if not sel:
        return []
    cells = {}
    for r in sel:
        cells.setdefault((r["r_old"], r["dd_lo"], r["dd_hi"]), []).append(r)
    rs = sorted({k[0] for k in cells})
    bands = sorted({(k[1], k[2]) for k in cells})
    out = [""]
    for name, key in (("Δ", "d_mid"), ("frac+", "frac_pos")):
        sumry = sel[0]["summary"]
        out += [f"口径 {cls} 的 {name}（{sumry}，state=retr）  行=R_old 列=ΔD带",
                "      " + "".join(f"{f'[{a},{b}]':>10}" for a, b in bands)]
        for rr in rs:
            cs = []
            for a, b in bands:
                v = cells.get((rr, a, b))
                if v:
                    xs = sorted(x[key] for x in v)
                    m = xs[len(xs) // 2] if len(xs) % 2 else \
                        (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2
                    cs.append(f"{m:>+10.2f}" + ("" if len(v) > 1 else ""))
                else:
                    cs.append(f"{'—':>10}")
            out.append(f"R{rr:>4} " + "".join(cs))
        out.append("")
    out += ["seed 极差（Δ）  格间差异须超过它才算有意义",
            "      " + "".join(f"{f'[{a},{b}]':>10}" for a, b in bands)]
    for rr in rs:
        cs = []
        for a, b in bands:
            v = cells.get((rr, a, b))
            if v and len(v) > 1:
                xs = [x["d_mid"] for x in v]
                cs.append(f"{max(xs) - min(xs):>10.2f}")
            else:
                cs.append(f"{'n=1':>10}" if v else f"{'—':>10}")
        out.append(f"R{rr:>4} " + "".join(cs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["runs", "runs_g2"])
    ap.add_argument("--txt", default="ledger.txt")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    rows = []
    for d in a.dirs:
        if not os.path.isdir(d):
            print(f"跳过不存在的目录 {d}")
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            if p.endswith(".perdoc.jsonl"):
                continue
            if ".txt.jsonl" in p:                 # go_nogo 缓存
                rows += rows_from_gonogo(p)
                continue
            r = row_from_jsonl(p)
            if r:
                rows.append(r)

    rows.sort(key=lambda r: (r["cls"], r["r_old"], r["dd_lo"], r["seed"]))
    lines = fmt(rows)

    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["cls"], []).append(r)
    lines += ["", "口径统计"]
    for c in sorted(by_cls):
        v = by_cls[c]
        nd = sum(1 for x in v if x["done"])
        lines.append(f"  {c}: {len(v)} run（完成 {nd}），"
                     f"summary={v[0]['summary']}, "
                     f"n_docs={sorted({x['n_docs'] for x in v})}")

    for c in ("B", "C"):
        if c in by_cls:
            lines += phase_map(rows, c)

    lines += ["",
              "口径 A=12k / B=16k / C=go_nogo400篇。A 与 B/C 不可混入同一张图：",
              "同格 R3/D2 的 frac+ 在 12k/16k/32k 分别为 0.43/0.41/0.89。",
              "B 的 frac+ 与 C 是嵌套子样本（同代码、同 seed_offset=1 文档流），",
              "可直接引用；B 缺中位数/IQR/ctrl，正文的这三项须用 C。",
              "p 为符号的精确二项检验（双侧）。p>0.05 即方向不成立。",
              "⚠未完 = 末条 eval 的 step 未达 total_steps，读数是中途值。"]

    out = "\n".join(lines)
    print(out)
    with open(a.txt, "w") as f:
        f.write(out + "\n")
    print(f"\n已写入 {a.txt}")

    if a.csv:
        import csv
        keys = list(rows[0].keys()) if rows else []
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"已写入 {a.csv}")


if __name__ == "__main__":
    main()