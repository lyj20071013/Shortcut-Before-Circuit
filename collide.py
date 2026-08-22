
"""App I：七条候选规则与真值的逐格碰撞率。

正文 §3.1 声称 FREQUENCY 与真值在 0.968–1.000 的文档上一致，这个数现在
没有出处。碰撞率决定哪些规则能被观测归因分开：碰撞率 1.000 的规则对必须
用因果编辑，否则任何观测方法都只能给出「两者都对」。
"""
import argparse, json
from collections import Counter

from config import CorpusCfg, LangSpec, dd_band
from generator import generate_corpus
from vocab import Vocab

R_OLDS = [3, 5, 8, 12, 16]
DDS = [2, 3, 5, 8, 16]

def band_of(d, fixw=0):
    """fixw>0: 固定宽度带 [d, d+fixw]，切断 posCeil 与 ΔD 的共线。
    fixw=0: 主网格的 dd_band(d)，宽度随 d 增长。"""
    return (d, d + fixw) if fixw else dd_band(d)

def q_stmts(d):
    """查询 slot 的语句下标，按文档顺序。"""
    return [i for i, s in enumerate(d.stmts)
            if (s.ent, s.attr) == (d.q_ent, d.q_attr)]


def rules(d):
    """七条候选规则各自预测的值。None 表示该规则在此文档上无定义。"""
    qi = q_stmts(d)
    if not qi:
        return {}
    vals = [d.stmts[i].val for i in qi]
    cnt = Counter(vals)
    out = {}
    out["recency"] = vals[-1]
    # rarity：出现次数最少者，平局取更靠后的那个
    lo = min(cnt.values())
    out["rarity"] = next(v for v in reversed(vals) if cnt[v] == lo)
    # frequency：出现次数最多者，平局取更靠后
    hi = max(cnt.values())
    out["frequency"] = next(v for v in reversed(vals) if cnt[v] == hi)
    out["primacy"] = vals[0]
    # global last update：全文最后一条 update 语句的值，不限 slot
    ups = [s for s in d.stmts if getattr(s, "upd", getattr(s, "is_update", False))]
    out["global_last_update"] = ups[-1].val if ups else None
    # last value token：查询前最后一个值 token，与 slot 无关
    out["last_token"] = d.stmts[-1].val
    # positional：末端固定偏移 4ΔD+6 处的值，用该格的众数 ΔD 调准
    out["_pos_target"] = d.realized_delta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1500)
    ap.add_argument("--n-values", type=int, default=512)
    ap.add_argument("--n-entities", type=int, default=200)
    ap.add_argument("--out", default="collide")
    ap.add_argument("--fixband", type=int, default=0,
                    help="固定带宽 W：ΔD ~ U[d, d+W]。0 = 用 dd_band(d)")
    ap.add_argument("--rows", type=int, nargs="+", default=R_OLDS)
    ap.add_argument("--cols", type=int, nargs="+", default=DDS)
    a = ap.parse_args()

    spec = LangSpec(n_values=a.n_values, n_entities=a.n_entities)
    vocab = Vocab(spec)
    names = ["recency", "rarity", "frequency", "primacy",
             "global_last_update", "last_token", "positional"]
    rows = []

    for r in a.rows:
        for dd in a.cols:
            lo, hi = band_of(dd, a.fixband)
            cfg = CorpusCfg(name=f"R{r}_D{dd}", seed=0, p_update=0.5,
                            max_updates=1, r_old_lo=r, r_old_hi=r,
                            use_marker=False, delta_d_lo=lo, delta_d_hi=hi,
                            p_hist_query=0.0, n_stmts_lo=45, n_stmts_hi=55)
            docs = list(generate_corpus(vocab, cfg, a.docs, seed_offset=7))
            # 位置规则先在同一批文档上调准偏移，再计分：这与 App C 的
            # posCeil 是同一个量，此处用经验众数而非解析式，作为交叉验证
            dmode = Counter(d.realized_delta for d in docs).most_common(1)[0][0]
            agree = {k: 0 for k in names}
            dfn = {k: 0 for k in names}
            for d in docs:
                truth = d.stmts[q_stmts(d)[-1]].val
                rr = rules(d)
                if not rr:
                    continue
                for k in names:
                    if k == "positional":
                        dfn[k] += 1
                        if d.realized_delta == dmode:
                            agree[k] += 1
                        continue
                    v = rr.get(k)
                    if v is None:
                        continue
                    dfn[k] += 1
                    if v == truth:
                        agree[k] += 1
            row = dict(r_old=r, dd=dd, dd_lo=lo, dd_hi=hi, n=len(docs),
                       dd_mode=dmode,
                       supp=hi - lo + 1, posceil=1.0 / (hi - lo + 1))
            for k in names:
                row[k] = agree[k] / dfn[k] if dfn[k] else float("nan")
            rows.append(row)
            print(f"R{r:>2} D{dd:>2}  " + "  ".join(
                f"{k[:4]}={row[k]:.3f}" for k in names))

    with open(f"{a.out}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    hdr = ["recency", "rarity", "frequency", "primacy",
           "global_last_update", "last_token", "positional"]
    short = {"recency": "REC", "rarity": "RAR", "frequency": "FRQ",
             "primacy": "PRI", "global_last_update": "GLU",
             "last_token": "LTK", "positional": "POS"}
    with open(f"{a.out}.tex", "w") as f:
        f.write("\\begin{tabular}{rr" + "r" * len(hdr) + "}\n\\toprule\n")
        f.write("$R_{\\mathrm{old}}$ & $\\Delta D$ & "
                + " & ".join(f"\\textsc{{{short[k].lower()}}}" for k in hdr)
                + " \\\\\n\\midrule\n")
        for i, row in enumerate(rows):
            if i and rows[i - 1]["r_old"] != row["r_old"]:
                f.write("\\midrule\n")
            f.write(f"{row['r_old']} & {row['dd']} & "
                    + " & ".join(f"{row[k]:.3f}" for k in hdr) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nwrote {a.out}.tex, {a.out}.jsonl")

    # 全格恒 1.000 的规则必须用因果编辑；正文的可辨识性论证依赖这一点
    for k in names:
        vs = [row[k] for row in rows]
        lo_v, hi_v = min(vs), max(vs)
        flag = "  ← 全格恒等，观测归因不可分" if lo_v > 0.9995 else ""
        print(f"{k:>20}: {lo_v:.3f}–{hi_v:.3f}{flag}")

if __name__ == "__main__":
    main()