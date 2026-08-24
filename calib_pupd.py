# -*- coding: utf-8 -*-
"""calib_pupd.py — 用 p_update 匹配 slot 数，固定文档长度。

app:slotmatch 靠缩短文档把 R_old=8 的 slot 数压到 R_old=16 的水平，代价是
长度从 204 token 掉到 116，slot 数/分散度/长度仍然绑在一起。p_update 只
作用于填充 slot（generator._build_slot 的 force_update=False 分支），
故它移动 slot 数而不动长度。

本脚本扫 p_update，报 slot 数与 q_gap（= app:gen 的 ant_q，即"copy
dispersion"），供选取匹配点。不训练，几分钟出结果。

用法:
    python calib_pupd.py                      # 默认扫 R8/R16 的 ΔD=2
    python calib_pupd.py --r 8 --d 2 --docs 2000
"""
import argparse
from statistics import mean

from config import CorpusCfg, LangSpec, dd_band
from generator import generate_corpus
from vocab import Vocab


def measure(vocab, cfg, n_docs):
    """返回该配置的结构统计。字段名对齐 tab:slotmatch 与 tab:inv。"""
    slots, qgap, toks, kept, tail0, ntail, clamp = [], [], [], [], [], [], []
    for d in generate_corpus(vocab, cfg, n_docs, seed_offset=1):
        slots.append(d.n_slots)
        qgap.append(d.q_gap)
        toks.append(len(d.tokens))
        kept.append(d.q_kept)
        tail0.append(int(d.n_tail_updates == 0))
        ntail.append(d.n_tail_updates)
        clamp.append(int(d.q_clamped))
    return dict(
        slots=mean(slots), q_gap=mean(qgap), tokens=mean(toks),
        q_kept=mean(kept), tail0=mean(tail0), n_tail=mean(ntail),
        clamped=mean(clamp),
        rho=cfg.p_update / (1.0 + cfg.p_update * cfg.r_old_hi),
    )


def main():
    ap = argparse.ArgumentParser(description="p_update 对 slot 数的标定")
    ap.add_argument("--docs", type=int, default=2000, help="与 app:slotmatch 同")
    ap.add_argument("--r", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--d", type=int, default=2, help="与缩短臂同格便于对比")
    ap.add_argument("--p", type=float, nargs="+",
                    default=[0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 1.00])
    ap.add_argument("--stmts-lo", type=int, default=45)
    ap.add_argument("--stmts-hi", type=int, default=55)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    spec = LangSpec(n_entities=200, n_values=512)   # 与主网格一致
    vocab = Vocab(spec)
    dlo, dhi = dd_band(a.d)

    print(f"ΔD ~ U[{dlo},{dhi}]  n_stmts ~ U[{a.stmts_lo},{a.stmts_hi}]  "
          f"{a.docs} 篇/配置  seed={a.seed}")
    print(f"\n{'R_old':>6}{'p_upd':>7}{'slots':>8}{'q_gap':>8}{'q_kept':>8}"
          f"{'tokens':>8}{'tail0':>7}{'rho':>7}{'clamp':>7}")
    print("-" * 68)

    ref = {}
    for r in a.r:
        for p in a.p:
            cfg = CorpusCfg(
                name=f"R{r}_D{a.d}_p{p}", seed=a.seed,
                p_update=p, max_updates=1,
                r_old_lo=r, r_old_hi=r,
                use_marker=False, delta_d_lo=dlo, delta_d_hi=dhi,
                p_hist_query=0.0,
                n_stmts_lo=a.stmts_lo, n_stmts_hi=a.stmts_hi,
            )
            m = measure(vocab, cfg, a.docs)
            star = " <- 主网格" if abs(p - 0.5) < 1e-9 else ""
            if star:
                ref[r] = m
            print(f"{r:>6}{p:>7.2f}{m['slots']:>8.2f}{m['q_gap']:>8.2f}"
                  f"{m['q_kept']:>8.2f}{m['tokens']:>8.1f}{m['tail0']:>7.3f}"
                  f"{m['rho']:>7.3f}{m['clamped']:>7.3f}{star}")
        print()

    if len(ref) == 2 and 8 in ref and 16 in ref:
        print("=" * 68)
        print(f"匹配目标：")
        print(f"  R8  主网格  slots={ref[8]['slots']:.2f}  "
              f"q_gap={ref[8]['q_gap']:.2f}  tokens={ref[8]['tokens']:.1f}")
        print(f"  R16 主网格  slots={ref[16]['slots']:.2f}  "
              f"q_gap={ref[16]['q_gap']:.2f}  tokens={ref[16]['tokens']:.1f}")
        print(f"\n找上表里 R8 的 slots 最接近 {ref[16]['slots']:.2f} 的那个 p，")
        print(f"和 R16 的 slots 最接近 {ref[8]['slots']:.2f} 的那个 p。")
        print(f"tokens 两侧应当几乎不变 —— 这正是本臂相对缩短臂的全部意义。")
        print(f"q_gap 会跟着动，动了多少要在附录里报，不能假定它不动。")


if __name__ == "__main__":
    main()