"""Inspect disagreement-supervision training-stream construction."""
import argparse
from collections import Counter

from config import CorpusCfg, LangSpec, dd_band
from train import Stream, _val_slots
from vocab import Vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--p-break", type=float, default=0.0)
    ap.add_argument("--truth-rule", default="recency",
                    choices=["recency", "rarity"])
    ap.add_argument("--n-values", type=int, default=512)
    ap.add_argument("--n-entities", type=int, default=200)
    ap.add_argument("--stmts-lo", type=int, default=45)
    ap.add_argument("--stmts-hi", type=int, default=55)
    a = ap.parse_args()

    spec = LangSpec(n_values=a.n_values, n_entities=a.n_entities)
    vocab = Vocab(spec)
    lo, hi = dd_band(a.d)
    cfg = CorpusCfg(name=f"R{a.r}_D{a.d}_s{a.seed}", seed=a.seed,
                    p_update=0.5, max_updates=1, r_old_lo=a.r, r_old_hi=a.r,
                    use_marker=False, delta_d_lo=lo, delta_d_hi=hi,
                    p_hist_query=0.0,
                    n_stmts_lo=a.stmts_lo, n_stmts_hi=a.stmts_hi,
                    p_break=a.p_break, truth_rule=a.truth_rule)

    # 走 Stream 而不是 generate_corpus：训练用的是前者，worker 0 的 offset
    # 是 1000。Stream 产出 d.tokens（list[int]），不是 Doc —— 这本身就是要
    # 检查的东西之一，因为下游的 collate 只吃 tokens。
    st = Stream(cfg, spec, n_workers=4)
    it = iter(st)
    toks = [next(it) for _ in range(a.n)]
    print(f"[{cfg.name}] p_break={a.p_break} truth_rule={a.truth_rule} "
          f"n={a.n}（训练流 worker 0，seed_offset=1000）")
    print(f"  token 数 {min(len(t) for t in toks)}–{max(len(t) for t in toks)}"
          f"  ctx_len={spec.ctx_len}")
    main_check(toks, vocab, spec, cfg)


def main_check(toks, vocab, spec, cfg):
    """训练流只给 tokens，故一切都从 token 序列反推 —— 这正是模型看到的。"""
    n_val_tok = 0
    ans_in_prefix = Counter()
    ans_pos_off = Counter()
    bad_struct = 0
    for t in toks:
        # 结构：末尾应是 ... QUERY ent attr ARROW val
        if not (t[-5] == vocab.QUERY and t[-2] == vocab.ARROW
                and vocab.is_val(t[-1])):
            bad_struct += 1
            continue
        ans = t[-1]
        prefix = t[:-1]
        vals = [x for x in prefix if vocab.is_val(x)]
        n_val_tok += len(vals)
        ans_in_prefix[vals.count(ans)] += 1
        # 答案值在前缀里的位置，从末尾数第几个 value token
        rev = [i for i, x in enumerate(reversed(vals)) if x == ans]
        ans_pos_off[rev[0] if rev else -1] += 1

    n = len(toks)
    print(f"  结构完好 {n - bad_struct}/{n}"
          + ("  ⚠ 有畸形文档" if bad_struct else ""))
    print(f"  每篇 value token 数 {n_val_tok / n:.1f}")
    print(f"  答案值在前缀出现次数的分布 "
          f"{dict(sorted(ans_in_prefix.items()))}")
    print(f"  答案值距末尾第几个 value token（-1=不在前缀）：")
    for k, v in sorted(ans_pos_off.items())[:8]:
        print(f"      {k:>3}: {v:>5}  ({v / n:.3f})")

    # copy 目标：_val_slots 标 is_rep 的比例。copy_diag 用它算 copyAcc，
    # copyNLL 不动就说明这条回路没形成 —— 先确认目标本身存在。
    print(f"\n  ⚠ 下面这段需要 Doc 对象，训练流给不出，故用 generate_corpus "
          f"（同 cfg，seed_offset=1）：")
    from generator import generate_corpus
    docs = list(generate_corpus(vocab, cfg, 400, seed_offset=1))
    rep = tot = 0
    rep_ok = 0
    for d in docs:
        seen = {}
        for (vpos, is_rep), s in zip(_val_slots(d, cfg), d.stmts):
            tot += 1
            if is_rep:
                rep += 1
                # is_rep 为真时，该位置的 token 必须等于同 slot 上一次的值
                rep_ok += int(d.tokens[vpos] == vocab.val(s.val)
                              and seen.get((s.ent, s.attr)) == s.val)
            seen[(s.ent, s.attr)] = s.val
    print(f"  可复制的 value token 占比 {rep / tot:.3f}"
          f"（docstring 称约 0.45）")
    print(f"  其中 token 与同 slot 前值一致的 {rep_ok}/{rep}"
          + ("  ✓" if rep_ok == rep else "  ⚠ copy 目标本身是错的"))

    brk = [d for d in docs if d.is_break]
    if brk:
        print(f"\n  反转文档 {len(brk)}/{len(docs)}")
        w = Counter()
        for d in brk:
            w[(d.n_old_kept, d.n_new_kept)] += 1
        print(f"  (n_old_kept, n_new_kept) 分布 {dict(w)}")
        lab_ok = sum(1 for d in brk
                     if d.answer_val_id == (d.val_history[-2]
                                            if cfg.truth_rule == "rarity"
                                            else d.val_history[-1]))
        print(f"  标签正确 {lab_ok}/{len(brk)}"
              + ("  ✓" if lab_ok == len(brk) else "  ⚠"))

    print("\n判读：")
    print("  答案值在前缀出现次数 —— 非反转文档恒为 1；N− 的反转文档为")
    print("  n_new_kept（设计使然），N+ 为 n_old_kept。出现 0 即标签不可达。")
    print("  「答案值距末尾第几个」应有质量集中在 ΔD 附近（位置规则的来源，")
    print("  posCeil=1/带宽）。若该分布完全平坦，位置捷径不存在，而主网格")
    print("  正是先爬满它再逃逸的 —— 那就解释了 acc 卡在 chance。")


if __name__ == "__main__":
    main()
