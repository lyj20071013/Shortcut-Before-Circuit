
"""slot 稀疏度臂的长度校准。

R_old 与每篇文档的 slot 数共线（R3 是 27.6，R16 是 8.7），而 §7 把读数
报成 (R_old, slot 数) 这个对的性质。打破共线的办法是缩短 R8 格的文档，
让它的 slot 数匹配 R16，再看读数往哪边走。

一个旋钮改两个协变量：slot 数按 n_stmts 单调，副本间距按
spread * n_stmts / q_kept。两者要求的长度不同，所以这里两个都测，
让长度的选择有据可依而不是猜。
"""
import argparse, json

from config import CorpusCfg, LangSpec, dd_band
from generator import generate_corpus
from vocab import Vocab


def q_stmts(d):
    return [i for i, s in enumerate(d.stmts)
            if (s.ent, s.attr) == (d.q_ent, d.q_attr)]


def measure(vocab, r_old, dd, lo_n, hi_n, docs, seed_offset=7):
    lo, hi = dd_band(dd)
    cfg = CorpusCfg(name=f"R{r_old}_D{dd}_n{lo_n}", seed=0,
                    p_update=0.5, max_updates=1,
                    r_old_lo=r_old, r_old_hi=r_old, use_marker=False,
                    delta_d_lo=lo, delta_d_hi=hi, p_hist_query=0.0,
                    n_stmts_lo=lo_n, n_stmts_hi=hi_n)
    ds = list(generate_corpus(vocab, cfg, docs, seed_offset=seed_offset))

    n_slots = n_kept = n_ant = n_ant_d = 0
    n_dom = n_tok = 0
    for d in ds:
        n_slots += len({(s.ent, s.attr) for s in d.stmts})
        qi = q_stmts(d)
        kept = len(qi) - 1                      # v_old 副本数，减去重绑定
        n_kept += kept
        if kept >= 2:
            n_dom += 1
            # 副本间距：相邻两份 v_old 的语句下标差
            gaps = [qi[j + 1] - qi[j] for j in range(len(qi) - 2)]
            if gaps:
                n_ant += sum(gaps)
                n_ant_d += len(gaps)
        n_tok += 4 * len(d.stmts) + 4

    n = len(ds)
    return dict(r_old=r_old, dd=dd, n_stmts_lo=lo_n, n_stmts_hi=hi_n,
                docs=n,
                slots=n_slots / n,
                q_kept=n_kept / n,
                antq=n_ant / n_ant_d if n_ant_d else float("nan"),
                domain=n_dom / n,
                max_tok=n_tok / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dd", type=int, default=2)
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--out", default="calib")
    a = ap.parse_args()

    spec = LangSpec(n_values=512, n_entities=200)
    vocab = Vocab(spec)
    rows = []

    # 参照：主网格配置下的 R8 与 R16
    for r in (8, 16):
        rows.append(measure(vocab, r, a.dd, 45, 55, a.docs))
    target = rows[-1]
    print(f"reference (n_stmts 45-55):")
    for row in rows:
        print(f"  R{row['r_old']:>2}  slots={row['slots']:.2f}  "
              f"q_kept={row['q_kept']:.2f}  antq={row['antq']:.2f}  "
              f"domain={row['domain']:.3f}  tok={row['max_tok']:.0f}")
    print(f"\ntarget: slots={target['slots']:.2f}  antq={target['antq']:.2f}\n")

    # 扫 R8 的长度，带宽固定为 ±5
    print("R8 sweep:")
    best_slots = best_antq = None
    for center in range(20, 56, 2):
        lo_n, hi_n = center - 5, center + 5
        # 结构下界：副本要铺得开，2*R_old <= spread*n_stmts
        if 2 * 8 > 0.80 * lo_n:
            print(f"  n={lo_n}-{hi_n}  skipped: copies do not fit "
                  f"(need n_stmts >= {int(2 * 8 / 0.80) + 1})")
            continue
        # 带宽下界：n_stmts >= R_old + dd_hi + 5
        if lo_n < 8 + dd_band(a.dd)[1] + 5:
            print(f"  n={lo_n}-{hi_n}  skipped: band does not fit")
            continue
        row = measure(vocab, 8, a.dd, lo_n, hi_n, a.docs)
        rows.append(row)
        ds = abs(row["slots"] - target["slots"])
        da = abs(row["antq"] - target["antq"])
        if best_slots is None or ds < best_slots[0]:
            best_slots = (ds, row)
        if best_antq is None or da < best_antq[0]:
            best_antq = (da, row)
        print(f"  n={lo_n:>2}-{hi_n:<2}  slots={row['slots']:>5.2f}  "
              f"q_kept={row['q_kept']:.2f}  antq={row['antq']:.2f}  "
              f"domain={row['domain']:.3f}  tok={row['max_tok']:.0f}")

    with open(f"{a.out}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    bs, ba = best_slots[1], best_antq[1]
    print(f"\nslot 数最接近: n_stmts {bs['n_stmts_lo']}-{bs['n_stmts_hi']}  "
          f"slots={bs['slots']:.2f} (目标 {target['slots']:.2f})  "
          f"antq={bs['antq']:.2f} (目标 {target['antq']:.2f})")
    print(f"间距最接近:   n_stmts {ba['n_stmts_lo']}-{ba['n_stmts_hi']}  "
          f"slots={ba['slots']:.2f}  antq={ba['antq']:.2f}")
    if bs["n_stmts_lo"] != ba["n_stmts_lo"]:
        print("两个协变量要求不同的长度，按 slot 数选，间距如实报告")
    print(f"\nwrote {a.out}.jsonl")


if __name__ == "__main__":
    main()
