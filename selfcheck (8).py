"""设计正确性检查。任一断言失败都意味着生成器有捷径，此时训练毫无意义。

检查分两类，不要混淆：
- 硬断言：违反即数据不可用。
- 协变量测量（8, 11）：捷径可用性随旋钮变化是设计的内在性质，
  只要求非完美，数值须在论文中作为格间协变量报告。
"""
import random
from collections import Counter

from config import (LangSpec, CorpusCfg, EXTREME_A, EXTREME_B,
                    R_OLD_GRID, DELTA_D_GRID, dd_band, hist_configs)
from generator import generate_corpus
from vocab import Vocab


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def _mean(vals):
    vals = [v for v in vals if v == v]      # 去 nan
    return sum(vals) / len(vals) if vals else float("nan")


def check(cfg: CorpusCfg, spec=LangSpec(), n=4000, verbose=True):
    vocab = Vocab(spec)
    docs = list(generate_corpus(vocab, cfg, n))

    # ---- 1. 绑定不可记忆 ----
    binds = Counter((d.q_ent, d.q_attr, d.answer) for d in docs)
    max_bind = max(binds.values())
    assert max_bind <= 3, f"绑定重复 {max_bind} 次，模型可能记住事实"

    # ---- 2. 答案在值池上近似均匀 ----
    ans = Counter(d.answer for d in docs)
    top_share = max(ans.values()) / n
    assert top_share < max(0.01, 20.0 / spec.n_values), f"答案分布偏斜 {top_share:.4f}"

    # ---- 3. recency 捷径：答案 == 最后一个值 token。应恒为0
    last_val_hit = ans_multi = 0
    for d in docs:
        vals = [t for t in d.tokens[:d.answer_pos] if vocab.is_val(t)]
        if vals and vals[-1] == d.answer:
            last_val_hit += 1
        if d.q_hist_k:      # hist_k≥1 时答案是老值，按设计重复 R_old 次
            continue
        if vals.count(d.answer) != 1:
            ans_multi += 1

    # ---- 4. 长度分布足够宽 ----
    lens = [len(d.tokens) for d in docs]
    mean_len = sum(lens) / n
    sd = (sum((x - mean_len) ** 2 for x in lens) / n) ** 0.5
    assert sd > 0.05 * mean_len, f"长度过于集中 sd={sd:.1f} mean={mean_len:.1f}"

    # ---- 5. 查询位置不固定 ----
    qpos = Counter(d.answer_pos for d in docs)
    assert max(qpos.values()) / n < 0.2, "查询位置过于集中"

    # ---- 6. ΔD 精确成立 ----
    deltas = [d.realized_delta for d in docs]
    assert min(deltas) >= max(1, cfg.delta_d_lo), f"ΔD 下界被破坏 {min(deltas)}"
    assert max(deltas) <= cfg.delta_d_hi, f"ΔD 上界被破坏 {max(deltas)}"

    # ---- 7. ΔD 与长度独立 ----
    corr = _pearson(deltas, [d.n_stmts for d in docs])
    assert abs(corr) < 0.15, f"ΔD 与语句数相关 r={corr:.3f}，两轴不可分离"

    # ---- 8. 协变量：末条更新即答案。tail_upd≈(1-ρ)^ΔD，
    #         ρ=p_update/(1+p_update·R_old)，随两轴变化是内在性质，不强制归零。
    #         强制尾部插入干扰更新在 ΔD=1 时反而制造"答案=倒数第二条"的完美捷径。
    tail_upd = sum(d.n_tail_updates == 0 for d in docs) / n
    assert tail_upd < 0.9, f"{tail_upd:.3f} 的样本末条更新即答案，接近完美捷径"

    # ---- 9. update 位置均匀性（只看 head，避开 tail 的 ΔD 结构）----
    dec_u = dec_n = all_u = all_n = 0
    for d in docs:
        p_final = d.n_stmts - 1 - d.realized_delta
        if p_final < 20:
            continue
        cut = int(p_final * 0.9)
        head = [p for p in d.upd_positions if p < p_final]
        all_u += len(head); all_n += p_final
        dec_u += sum(1 for p in head if p >= cut); dec_n += p_final - cut
    ratio = ((dec_u / dec_n) / (all_u / all_n)) if dec_n and all_u else float("nan")
    assert 0.75 < ratio < 1.3, f"head 末段 update 密度是全局的 {ratio:.2f} 倍"

    # ---- 10. 纯位置规则上限。ΔD 支撑集为单点时 100% 可解 ----
    span = cfg.delta_d_hi - cfg.delta_d_lo + 1
    pos_ceiling = 1.0 / span
    assert pos_ceiling < 1.0, "ΔD 恒定，纯位置规则 100% 可解"
    if pos_ceiling > 0.55:
        print(f"  ⚠ [{cfg.name}] 位置规则上限 {pos_ceiling:.2f}：ΔD 支撑集仅 "
              f"{span} 个值，该格规则归因可信度低")

    # ---- 11. q_slot 是否结构可辨识。两项都是绕过 query 的直接路径 ----
    # adj：update 的前驱是否同 slot。连续平铺的旧实现给出 adj_fill=1.00、
    # adj_q=0.00，"找前驱不同 slot 的 update"即 100% 判别式，须严格对齐。
    adj_q, adj_fill = _mean([d.adj_q for d in docs]), _mean([d.adj_fill for d in docs])
    assert abs(adj_q - adj_fill) < 0.15, \
        f"update 前驱同 slot 率 q={adj_q:.3f} vs 填充={adj_fill:.3f}，" \
        f"可据此定位答案而无需读 query"
    # gap：老值到 update 的距离，条件在后半段以与 q_final 位置可比
    gap_ratio = _mean([d.q_gap for d in docs]) / \
    (_mean([d.fill_gap_late for d in docs]) or float("nan"))
    gap_near = _mean([d.q_gap for d in docs]) / \
    (_mean([d.fill_gap_near for d in docs]) or float("nan"))
    if not 0.80 < gap_near < 1.25:
        print(f"  ⚠ [{cfg.name}] gapNear={gap_near:.2f} 越界：q_final 前驱距离与"
          f"填充 update 不同分布，可据此定位答案。清干净后升级为硬断言。")
    
    # ---- 12. 相图配置不得出现时间索引 token ----
    if cfg.p_hist_query == 0:
        for d in docs[:200]:
            assert all(not (vocab.TIME0 <= t < vocab.ENT0) for t in d.tokens), \
                "p_hist_query=0 的配置里出现了 @k token"

    upd_density = sum(len(d.upd_positions) for d in docs) / \
        sum(d.n_stmts for d in docs)
    
    fq = [_mean([d.fill_quint[b] for d in docs]) for b in range(5)]
    kq = [_mean([d.key_quint[b] for d in docs]) for b in range(5)]
    

    stats = dict(
        mean_len=mean_len, sd=sd, last_val_hit=frac,
        d_lo=min(deltas), d_hi=max(deltas), d_mean=_mean(deltas), corr=corr,
        mean_stmts=_mean([d.n_stmts for d in docs]),
        mean_slots=_mean([d.n_slots for d in docs]),
        max_bind=max_bind, tail_upd=tail_upd,
        mean_tail_u=_mean([d.n_tail_updates for d in docs]),
        upd_uniform=ratio, upd_density=upd_density,
        pos_ceiling=pos_ceiling, gap_ratio=gap_ratio,
        adj_q=adj_q, adj_fill=adj_fill,
gap_near=gap_near, mean_w_st=_mean([d.w_st for d in docs]),
fill_quint=fq, key_quint=kq,
q_clamped=sum(d.q_clamped for d in docs) / n,
r_real=_mean([d.q_kept for d in docs]),
spread=cfg.spread)

    if verbose:
        print(f"[{cfg.name}] n={n} len={mean_len:.0f}±{sd:.0f} "
              f"stmts={stats['mean_stmts']:.1f} slots={stats['mean_slots']:.1f} "
              f"ΔD={min(deltas)}..{max(deltas)}(μ{stats['d_mean']:.1f}) "
              f"r(ΔD,len)={corr:+.3f} lastVal={frac:.4f} "
              f"tailUpd={tail_upd:.3f} tailU={stats['mean_tail_u']:.2f} "
              f"updDens={upd_density:.3f} updUnif={ratio:.2f} "
              f"posCeil={pos_ceiling:.2f} gapRatio={gap_ratio:.2f} "
f"gapNear={gap_near:.2f} adj={adj_q:.2f}/{adj_fill:.2f} "
f"spread={cfg.spread:.2f} wSt={stats['mean_w_st']:.0f} "
f"bind={max_bind} "
f"clamp={stats['q_clamped']:.3f} "
f"Rreal={stats['r_real']:.1f} "
)
        print(f"    fillQuint={' '.join(f'{v:.3f}' for v in fq)}  "
      f"keyQuint={' '.join(f'{v:.3f}' for v in kq)}")
              
    return stats


def check_grid_corners(n=1500):
    """相图四角 + 中心。角落格子最容易静默失真。"""
    cells = [(R_OLD_GRID[0], DELTA_D_GRID[0]), (R_OLD_GRID[0], DELTA_D_GRID[-1]),
             (R_OLD_GRID[-1], DELTA_D_GRID[0]), (R_OLD_GRID[-1], DELTA_D_GRID[-1]),
             (R_OLD_GRID[2], DELTA_D_GRID[2])]
    out = []
    for r, d in cells:
        dlo, dhi = dd_band(d)
        cfg = CorpusCfg(name=f"R{r}_D{d}", seed=0, p_update=0.5, max_updates=1,
                        r_old_lo=r, r_old_hi=r, use_marker=False,
                        delta_d_lo=dlo, delta_d_hi=dhi, p_hist_query=0.0)
        out.append((r, d, check(cfg, n=n)))
    return out


if __name__ == "__main__":
    print("== 极端配置 ==")
    a = check(EXTREME_A)
    b = check(EXTREME_B)

    assert a["last_val_hit"] == b["last_val_hit"] == 0.0, "recency 捷径不匹配"

    # 软性可比：捷径可用性差异过大时，arm 差异不可完全归因到规则学习
    for k, thr in (("tail_upd", 0.25), ("gap_ratio", 0.5)):
        gap = abs(a[k] - b[k])
        if gap > thr:
            print(f"  ⚠ {k} 两 arm 差 {gap:.3f}（{a[k]:.3f} vs {b[k]:.3f}）："
                  f"捷径可用性不同，生死门的 arm 差异需谨慎解读")

    print("\n== 相图角落 ==")
    corners = check_grid_corners()

    # slots 随 R_old 变化不可消除：n_stmts ≈ n_slots × (1 + p_update·R_old)，
    # 固定 n_stmts 与 p_update 后 n_slots 必然随 R_old 下降。改为固定 n_slots
    # 会让长度随 R_old 变，改 p_update 会混淆旋钮 1——两个自由度满足不了
    # 三个约束。作为格间协变量报告，不要试图消除。
    print("\n格间协变量（须在论文中报告）:")
    print(f"  {'格':<12} {'slots':>6} {'updDens':>8} {'tailUpd':>8} "
         f"{'posCeil':>8} {'gapRatio':>9} {'gapNear':>8}")
    for r, d, s in corners:
        print(f"  R={r:<3} ΔD={d:<4} {s['mean_slots']:6.1f} "
          f"{s['upd_density']:8.3f} {s['tail_upd']:8.3f} "
          f"{s['pos_ceiling']:8.2f} {s['gap_ratio']:9.2f} "
          f"{s['gap_near']:8.2f}") 

    v = Vocab(LangSpec())
    d = next(generate_corpus(v, EXTREME_A, 1))
    print("\n样例（截断）:\n", v.render(d.tokens[:80]), "...")
    print("答案:", v.decode(d.answer), "历史:", d.val_history,
          "ΔD:", d.realized_delta, "语句数:", d.n_stmts,
          "尾部更新:", d.n_tail_updates, "q_gap:", d.q_gap)
          
    print("\n== 旋钮5 对照 ==")
    for cfg in hist_configs():
        if cfg.seed == 0:
            check(cfg, n=1000)
        
    if verbose:
        print(f"    fillQuint={' '.join(f'{v:.3f}' for v in fq)}  "
          f"keyQuint={' '.join(f'{v:.3f}' for v in kq)}")