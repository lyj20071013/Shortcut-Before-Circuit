
"""slot 稀疏度臂的配对读数。

校准把 slot 数（14.59 -> 8.95）和副本间距（6.87 -> 4.71）都匹配到了
R16/D2，但代价是实际冗余度从 5.75 掉到 4.02。§5.4 已证效应随实际冗余度
放大，所以整格对比分不开「slot 数变了」和「冗余度变了」。

解法：取两格 q_kept 分布的重叠带，band 内 R_old 与实际冗余度一致，
只有 slot 数不同。整格读数也报，但结论落在配对子集上。
"""
import argparse, json, math

DELTA_KEYS = ["delta", "d_logodds", "dlogodds", "logodds_delta",
              "delta_logodds", "readout"]
KEPT_KEYS = ["q_kept", "kept", "n_kept", "r_kept", "realized_r",
             "q_mult", "mult_old", "n_old"]
MASS_KEYS = ["mass", "pair_mass", "contrast_mass"]


def pick(rec, cands, override=None):
    if override:
        return override
    for k in cands:
        if k in rec:
            return k
    return None


def load(path, dk, kk, mk):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path}: 空文件")
    dk = pick(rows[0], DELTA_KEYS, dk)
    kk = pick(rows[0], KEPT_KEYS, kk)
    mk = pick(rows[0], MASS_KEYS, mk)
    if dk is None or kk is None:
        raise SystemExit(
            f"{path}: 认不出字段。可用键：{sorted(rows[0].keys())}\n"
            f"用 --delta-key / --kept-key 指定")
    return rows, dk, kk, mk


def binom_two_sided(k, n):
    """精确二项检验，p=0.5，两侧。"""
    if n == 0:
        return float("nan")
    pk = [math.comb(n, i) for i in range(n + 1)]
    obs = pk[k]
    tot = sum(pk)
    return sum(v for v in pk if v <= obs) / tot


def median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def summarize(rows, dk, kk, mk, label, gate, lo=None, hi=None):
    sel = rows
    if lo is not None:
        sel = [r for r in sel if lo <= r[kk] <= hi]
    if mk and gate is not None:
        sel = [r for r in sel if r.get(mk) is None or r[mk] >= gate]
    ds = [r[dk] for r in sel]
    if not ds:
        print(f"{label:<28} 无文档")
        return None
    n = len(ds)
    npos = sum(1 for v in ds if v > 0)
    frac = npos / n
    p = binom_two_sided(npos, n)
    kept = sum(r[kk] for r in sel) / n
    print(f"{label:<28} n={n:>4}  frac+={frac:.3f}  p={p:.2e}  "
          f"median={median(ds):+.3f}  q_kept={kept:.2f}")
    return dict(label=label, n=n, frac_pos=frac, p=p,
                median=median(ds), q_kept=kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main_grid", help="R8/D2 主网格的逐篇 jsonl")
    ap.add_argument("shortened", help="R8/D2 缩短格的逐篇 jsonl")
    ap.add_argument("--ref", help="R16/D2 主网格的逐篇 jsonl（可选参照）")
    ap.add_argument("--band", default="3,5",
                    help="配对的 q_kept 闭区间，默认 3,5")
    ap.add_argument("--gate", type=float, default=0.5,
                    help="mass 门，逐篇施加；设 -1 关闭")
    ap.add_argument("--delta-key")
    ap.add_argument("--kept-key")
    ap.add_argument("--mass-key")
    ap.add_argument("--out", default="pair")
    a = ap.parse_args()

    lo, hi = (float(x) for x in a.band.split(","))
    gate = None if a.gate < 0 else a.gate
    out = []

    paths = [("R8 main grid", a.main_grid),
             ("R8 shortened", a.shortened)]
    if a.ref:
        paths.append(("R16 main grid", a.ref))

    loaded = []
    for label, path in paths:
        rows, dk, kk, mk = load(path, a.delta_key, a.kept_key, a.mass_key)
        loaded.append((label, rows, dk, kk, mk))
        print(f"{label}: {len(rows)} docs, delta={dk} kept={kk} mass={mk}")
    print()

    print("整格（未配对）")
    for label, rows, dk, kk, mk in loaded:
        r = summarize(rows, dk, kk, mk, label, gate)
        if r:
            r["subset"] = "ungated"
            out.append(r)

    print(f"\n配对 q_kept in [{lo:g}, {hi:g}]")
    for label, rows, dk, kk, mk in loaded:
        r = summarize(rows, dk, kk, mk, label, gate, lo, hi)
        if r:
            r["subset"] = f"matched[{lo:g},{hi:g}]"
            out.append(r)

    # 配对子集里两个 R8 格只差 slot 数，差值才是这个臂要的量
    m = {r["label"]: r for r in out if r["subset"].startswith("matched")}
    if "R8 main grid" in m and "R8 shortened" in m:
        d = m["R8 shortened"]["frac_pos"] - m["R8 main grid"]["frac_pos"]
        kd = abs(m["R8 shortened"]["q_kept"] - m["R8 main grid"]["q_kept"])
        print(f"\n配对子集内 frac+ 差 = {d:+.3f}  "
              f"（残余 q_kept 失配 {kd:.2f}）")
        if kd > 0.5:
            print("残余失配偏大，考虑收窄 --band")
        print("接近 0 -> slot 数无贡献；显著为正 -> slot 数是主因之一")

    with open(f"{a.out}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {a.out}.jsonl")


if __name__ == "__main__":
    main()
