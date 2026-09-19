"""Check token/label matching between the two supervision conditions."""
import argparse
from dataclasses import replace

from config import CorpusCfg, LangSpec, dd_band
from generator import generate_corpus
from vocab import Vocab


def slot_kind(group):
    """按多重性判读一个 slot：normal / reversed / tie / no_upd。

    group 是同 (ent,attr) 的语句，按文档顺序。生成器保证 slot 内同值连续且
    is_update 单调，故 group[-1].val 是末代值。
    tie 来自窗口越界：反转 slot 的末代值副本若被丢到只剩钉在 p_final 那份，
    counts 退化成 {1,1}，判读不出方向。R_old=1 时正常 slot 也恒为 tie。
    """
    vals = [s.val for s in group]
    if len(set(vals)) < 2:
        return "no_upd"
    n_fin = vals.count(vals[-1])
    n_old = len(vals) - n_fin
    if n_fin > n_old:
        return "reversed"
    if n_fin < n_old:
        return "normal"
    return "tie"


def check_pair(cfg_minus, spec, n):
    """N+ 与 N− 逐篇比对。cfg_minus 是 truth_rule='recency' 的配置。"""
    vocab = Vocab(spec)
    cfg_plus = replace(cfg_minus, truth_rule="rarity")
    a = list(generate_corpus(vocab, cfg_minus, n, seed_offset=1))
    b = list(generate_corpus(vocab, cfg_plus, n, seed_offset=1))
    assert len(a) == len(b) == n

    n_brk = n_diff = 0
    for i, (x, y) in enumerate(zip(a, b)):
        ctx = f"doc {i}"
        # 前缀是 tokens[:answer_pos]，不是 tokens[:-1] —— 后者在 hist 查询下
        # 会把 @k 也算进去。arm 禁 hist，但断言按定义写才不会在别处误用。
        assert x.tokens[:x.answer_pos] == y.tokens[:y.answer_pos], \
            f"{ctx}: 两条 arm 的探针输入不同，配对失败"
        assert x.answer_pos == y.answer_pos, f"{ctx}: answer_pos 不同"
        assert x.is_break == y.is_break, f"{ctx}: is_break 不同"
        assert (x.n_old_kept, x.n_new_kept) == (y.n_old_kept, y.n_new_kept), \
            f"{ctx}: 存活份数不同"
        assert x.realized_delta == y.realized_delta, f"{ctx}: ΔD 不同"
        assert x.n_stmts == y.n_stmts, f"{ctx}: 语句数不同"
        assert x.val_history == y.val_history, f"{ctx}: 值历史不同"
        n_brk += x.is_break
        if x.answer != y.answer:
            n_diff += 1
            assert x.is_break, f"{ctx}: 非反转文档的标签竟不同"
            # N− 取末代值、N+ 取老值。_ValueDraw 保证一篇内值 id 不重复，
            # 故两者必不相等 —— 标签确实被 truth_rule 改动了。
            assert x.answer_val_id == x.val_history[-1], f"{ctx}: N− 标签非末代值"
            assert y.answer_val_id == y.val_history[-2], f"{ctx}: N+ 标签非老值"
        else:
            assert not x.is_break, \
                f"{ctx}: 反转文档在两条 arm 上标签相同，truth_rule 未生效"
    assert n_diff == n_brk, f"标签不同的篇数 {n_diff} ≠ 反转篇数 {n_brk}"
    return dict(n=n, n_break=n_brk, frac_break=n_brk / n, n_label_diff=n_diff)


def check_inert(cfg, spec, n):
    """p_break=0 时旋钮 6 的其余参数必须完全惰性。

    短路求值（p_break > 0.0 and rng.random() < ...）让 rng.random() 不被调用，
    故随机数流与已发表主网格逐比特相同。这是那 75 个 run 仍可复现的必要条件，
    也是 N− 剂量曲线能免费复用它们作 p_break=0 锚点的依据。
    """
    vocab = Vocab(spec)
    zero = replace(cfg, p_break=0.0, truth_rule="recency")
    base = [d.tokens for d in generate_corpus(vocab, zero, n, seed_offset=1)]
    for k, rb in ((2, 7), (3, 11)):
        alt = replace(zero, k_old_break=k, r_new_break=rb)
        got = [d.tokens for d in generate_corpus(vocab, alt, n, seed_offset=1)]
        assert got == base, \
            f"p_break=0 下 k_old_break={k}/r_new_break={rb} 改变了产出，" \
            f"旋钮 6 未惰性，已发表 run 不再可复现"
    for d in generate_corpus(vocab, zero, n, seed_offset=1):
        assert not d.is_break, "p_break=0 竟产出反转文档"
        assert d.answer_val_id == d.val_history[-1], "p_break=0 下标签非末代值"
    return dict(n=n, ok=True)


def check_inv6(cfg, spec, n, tol):
    """不变量 6：被查询侧与填充侧的反转率必须匹配。

    匹配的量是「有 update 的 slot 中被反转的比例」。无 update 的填充 slot 只有
    一代值，_build_slot 的 last>0 守卫让 reps_final 不生效，反转对它是空操作，
    故不进分母。

    被查询侧有 ground truth（d.is_break），用它量 slot_kind 的判读误差；填充侧
    只能靠判读。tie 单独报：它来自窗口越界丢尽末代值副本，是反转失效而非
    判读失败，probe_selfcheck 的 degen 门管的是同一件事。
    """
    vocab = Vocab(spec)
    q_rev = q_norm = q_tie = 0
    f_rev = f_norm = f_tie = 0
    q_wrong = 0
    # tie 必须按 is_break 分层，否则会把主网格的既有性质报成 arm 的缺陷。
    # 非反转文档的 q slot 是 [老值×R, 末代值×1]，窗口越界把老值砍到只剩 1 份时
    # vals=[老值,末代值]、份数相等 -> tie。这就是 covar 的 1−domain（R3_D8 约
    # 17%），是已发表主网格本来就有的，与旋钮 6 无关，且这些篇的别名照常成立
    # （counts={1,1}，平票裁给更近者 = 末代值 = recency = rarity），标签有规则可依。
    # 真正要担心的只有反转文档的 tie：那才是"反转失效、N+ 的标签无规则可依"。
    q_tie_brk = q_tie_nb = n_brk = 0
    for d in generate_corpus(vocab, cfg, n, seed_offset=1):
        n_brk += d.is_break
        per = {}
        for s in d.stmts:
            per.setdefault((s.ent, s.attr), []).append(s)
        for key, grp in per.items():
            kind = slot_kind(grp)
            if kind == "no_upd":
                continue
            if key == (d.q_ent, d.q_attr):
                q_rev += kind == "reversed"
                q_norm += kind == "normal"
                q_tie += kind == "tie"
                if kind == "tie":
                    q_tie_brk += d.is_break
                    q_tie_nb += not d.is_break
                if kind != "tie" and (kind == "reversed") != d.is_break:
                    q_wrong += 1
            else:
                f_rev += kind == "reversed"
                f_norm += kind == "normal"
                f_tie += kind == "tie"
    q_den, f_den = q_rev + q_norm, f_rev + f_norm
    q_rate = q_rev / q_den if q_den else float("nan")
    f_rate = f_rev / f_den if f_den else float("nan")
    gap = abs(q_rate - f_rate)
    out = dict(q_rate=q_rate, f_rate=f_rate, gap=gap,
               q_tie=q_tie / (q_den + q_tie) if q_den + q_tie else 0.0,
               f_tie=f_tie / (f_den + f_tie) if f_den + f_tie else 0.0,
               q_detect_err=q_wrong / q_den if q_den else float("nan"),
               # 分层 tie。分母各取自己那一层的文档数，故两者可独立解读。
               # tie_nb 是主网格既有性质（= 1−domain，窗口越界把老值砍到 1 份），
               # 与旋钮 6 无关且别名照常成立；tie_brk 才是"反转失效"。
               tie_nb=q_tie_nb / (n - n_brk) if n - n_brk else 0.0,
               tie_brk=q_tie_brk / n_brk if n_brk else 0.0)
    assert gap <= tol, (
        f"反转率失配 {gap:.4f} > {tol}：被查询侧 {q_rate:.4f} vs 填充侧 "
        f"{f_rate:.4f}。「末代值被重复」成为定位被查询 slot 的判别式，"
        f"模型可绕过 query")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-break", type=float, default=0.10)
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--k-old-break", type=int, default=1)
    ap.add_argument("--r-new-break", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="不变量 6 的反转率失配容差")
    # 以下四项必须与 sweep.FIXED 一致，否则测的不是主网格实际用的语料。
    # 不 import sweep 是为了不把 torch 拖进这个纯 CPU 门。
    ap.add_argument("--n-values", type=int, default=512)
    ap.add_argument("--n-entities", type=int, default=200)
    ap.add_argument("--stmts-lo", type=int, default=45)
    ap.add_argument("--stmts-hi", type=int, default=55)
    a = ap.parse_args()

    spec = LangSpec(n_values=a.n_values, n_entities=a.n_entities)
    lo, hi = dd_band(a.d)
    cfg = CorpusCfg(
        name=f"nb_R{a.r}_D{a.d}_s{a.seed}", seed=a.seed,
        p_update=0.5, max_updates=1, r_old_lo=a.r, r_old_hi=a.r,
        use_marker=False, delta_d_lo=lo, delta_d_hi=hi, p_hist_query=0.0,
        n_stmts_lo=a.stmts_lo, n_stmts_hi=a.stmts_hi,
        p_break=a.p_break, k_old_break=a.k_old_break,
        r_new_break=a.r_new_break, truth_rule="recency")

    print(f"[{cfg.name}] p_break={a.p_break} R={a.r} ΔD={a.d}(band {lo}-{hi}) "
          f"n={a.n}")

    inert = check_inert(cfg, spec, min(a.n, 800))
    print(f"  惰性     p_break=0 下旋钮 6 无影响，n={inert['n']}  ✓")

    pr = check_pair(cfg, spec, a.n)
    print(f"  配对     反转 {pr['n_break']}/{pr['n']}={pr['frac_break']:.4f}  "
          f"标签不同 {pr['n_label_diff']} 篇  前缀逐比特相同  ✓")
    print(f"           名义 p_break={a.p_break}，实测偏差 "
          f"{abs(pr['frac_break'] - a.p_break):.4f}")

    iv = check_inv6(cfg, spec, a.n, a.tol)
    print(f"  不变量6  被查询侧反转率 {iv['q_rate']:.4f}  填充侧 "
          f"{iv['f_rate']:.4f}  差 {iv['gap']:.4f} ≤ {a.tol}  ✓")
    print(f"           tie 率  反转文档 {iv['tie_brk']:.4f}  "
          f"非反转 {iv['tie_nb']:.4f}  填充侧 {iv['f_tie']:.4f}")
    print(f"           判读误差（被查询侧有 ground truth）"
          f"{iv['q_detect_err']:.4f}")
    # 只对反转文档的 tie 报警。非反转文档的 tie 是主网格既有性质
    # （= 1−domain，窗口越界把老值砍到 1 份、份数相等），R3_D8 约 17%，
    # 与旋钮 6 无关，且那些篇的别名照常成立（counts={1,1}，平票裁给更近者
    # = 末代值，recency 与 rarity 仍同指），标签有规则可依。
    # 拿池化 tie 报警会在每个低 R_old 格上放假警报。
    if iv["tie_brk"] > 0.02:
        print(f"  ⚠ 反转文档 tie 率 {iv['tie_brk']:.3f} > 0.02：这些篇反转失效"
              f"（末代值副本被窗口越界丢尽），别名未破除，N+ 下标签无规则可依。"
              f"收窄 spread 或提高 n_stmts_lo。")
    print("\n全部通过。这条断言成立 ⇒ 两条 arm 的探针输入分布严格相同，"
          "\n读数差异只能来自目标函数。")


if __name__ == "__main__":
    main()
