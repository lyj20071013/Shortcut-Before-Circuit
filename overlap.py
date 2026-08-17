
"""App H：位置规则解释平台态负读数的可检验预测。

平台态的 Δ 恒为负（R2 行实测 −0.47 / −1.30 / −0.94）。App H 给的候选解释是：
多重性反转把 R−1 份 v_old 改写成 v_new，若位置规则读取的偏移 4ΔD+6 落在
被改写的副本上，规则的输出就从 v_old 翻到 v_new，而 v* = v_old，故 Δ<0。

这个解释给出一个不需要新训练的预测：偏移与副本位置的重合率应当预测 |Δ|
的大小，且在从未占据位置规则的那格重合率与 Δ 都应接近零。
"""
import argparse, json
from collections import Counter

from config import CorpusCfg, LangSpec, dd_band
from generator import generate_corpus
from vocab import Vocab

# App H 的五格：R_old=2，五个带宽。末列是实测 Δ 与符号比例，见 ledger
CELLS = [(2, 2, -0.469, 0.05), (2, 3, -0.441, 0.01), (2, 5, -0.001, 0.22),
         (2, 8, -1.299, 0.02), (2, 16, +0.327, 0.72)]


def q_stmts(d):
    return [i for i, s in enumerate(d.stmts)
            if (s.ent, s.attr) == (d.q_ent, d.q_attr)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1500)
    ap.add_argument("--out", default="overlap")
    a = ap.parse_args()

    spec = LangSpec(n_values=512, n_entities=200)
    vocab = Vocab(spec)
    rows = []

    for r, dd, d_obs, frac_obs in CELLS:
        lo, hi = dd_band(dd)
        cfg = CorpusCfg(name=f"R{r}_D{dd}", seed=0, p_update=0.5,
                        max_updates=1, r_old_lo=r, r_old_hi=r,
                        use_marker=False, delta_d_lo=lo, delta_d_hi=hi,
                        p_hist_query=0.0, n_stmts_lo=45, n_stmts_hi=55)
        docs = list(generate_corpus(vocab, cfg, a.docs, seed_offset=7))
        # 位置规则调准到经验众数 ΔD，与训练时它能达到的最优偏移一致
        dmode = Counter(d.realized_delta for d in docs).most_common(1)[0][0]
        k_off = 4 * dmode + 6          # App C 公式 6：末端反查偏移

        n_hit = n_tot = 0
        for d in docs:
            qi = q_stmts(d)
            if len(qi) < 2:
                continue
            n_tot += 1
            m = len(d.stmts)
            n_tok = 4 * m + 4
            # 编辑改写 qi[1:]（保留最早一份），落在这些语句的值 token 上
            # 才会改变位置规则读到的内容
            rewritten = {4 * i + 2 for i in qi[1:]}
            read_pos = n_tok - k_off
            if read_pos in rewritten:
                n_hit += 1

        rate = n_hit / n_tot if n_tot else float("nan")
        rows.append(dict(r_old=r, dd=dd, dd_mode=dmode, offset=k_off,
                         n=n_tot, overlap=rate,
                         d_obs=d_obs, frac_obs=frac_obs))
        print(f"R{r} D{dd:>2}  ΔDmode={dmode:>2} offset={k_off:>3}  "
              f"overlap={rate:.3f}  Δobs={d_obs:+.3f}  frac+={frac_obs:.2f}")

    with open(f"{a.out}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # 预测：overlap 越高 |Δ| 越大，且 overlap≈0 的格 Δ≈0
    xs = [row["overlap"] for row in rows]
    ys = [abs(row["d_obs"]) for row in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    rho = num / (dx * dy) if dx and dy else float("nan")
    print(f"\noverlap vs |Δ|  Pearson r = {rho:+.3f}  (n={n})")
    print("五点相关系数不足以确立解释；判据是符号方向与「零重合格 Δ≈0」")
    print(f"wrote {a.out}.jsonl")

if __name__ == "__main__":
    main()