"""Measure pretrained-model rule preferences before fine-tuning."""
import argparse
import dataclasses
import os
from typing import Dict, List, Optional, Sequence, Tuple

MIRROR = "https://hf-mirror.com"


def inverted_cfg(r: int, d: int, seed: int):
    """全反转的 cfg。字段名取自 runs_dseeds 的 meta 记录，那里 p_break=0.0、
    k_old_break=1、r_new_break=0、truth_rule="recency" 是默认值。

    k=1、r=R_old 是 app:nonalias 的选择（该附录 Construction 节）：这让反转
    slot 仍有 R_old+1 条语句，故窗口放置、ΔD、token 数与每 slot 期望语句数
    都不变 —— 反转是多重性的严格交换。

    truth_rule 对本脚本无关：我们只取 prompt（答案 token 被 prompt_of 剥掉），
    而 v_old / v_new 由 stmts 直接算，不读 doc.answer。
    """
    from nl_corpus import nl_corpus_cfg
    return dataclasses.replace(nl_corpus_cfg(r, d, seed), p_break=1.0,
                               k_old_break=1, r_new_break=r)


def rules_of(doc) -> Optional[Tuple[int, int]]:
    """(rarity 选的 val, recency 选的 val)。None 表示这篇不可用。

    不读任何 truth 字段，两个量都从 stmts 重算：
      rarity  = 被查询 slot 里计数最小的值（反转后应当恰好是那 1 份）
      recency = 被查询 slot 里位置最后的值
    然后要求两者不等 —— 若相等说明这篇没真正反转（例如越界丢弃把 3 份
    v_new 削到 1 份，counts 变成 1:1，此时两条规则又重合了）。这正是
    nl_gates 的 G5 在主网格上看到的 n_tie=334 的同一个机制。
    """
    q = [s for s in doc.stmts if (s.ent, s.attr) == (doc.q_ent, doc.q_attr)]
    if len(q) < 2:
        return None
    cnt: Dict[int, int] = {}
    for s in q:
        cnt[s.val] = cnt.get(s.val, 0) + 1
    if len(cnt) != 2:
        return None
    v_rar = min(cnt, key=lambda v: cnt[v])
    v_rec = q[-1].val
    if cnt[v_rar] != 1 or v_rar == v_rec:
        return None
    return v_rar, v_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--src", default="hf", choices=["hf", "ms"])
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--copy-docs", type=int, default=60)
    ap.add_argument("--skip-copy", action="store_true",
                    help="跳过 phase A（慢速 tokenizer 上 offsets 不可用）")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    a = ap.parse_args()
    if a.src == "hf" and not a.no_mirror:
        os.environ.setdefault("HF_ENDPOINT", MIRROR)

    import torch
    from ft_zeroshot import cand_ids, load, prompt_of
    from generator import generate_corpus
    from nl_corpus import ADJ_OF, adj_slots, nl_corpus_cfg, nl_spec, tokenize
    from nl_generator import ADJ
    from nl_render import render
    from vocab import Vocab

    spec = nl_spec()
    v = Vocab(spec)
    tok, model, dev = load(a.model, a.src)
    cands = cand_ids(tok, ADJ[:spec.n_values])
    cid = torch.tensor(sorted(cands.values()), device=dev)
    print(f"{a.model}  单 token adj {len(cands)}/{spec.n_values}  device {dev}")

    # phase A 用 return_offsets_mapping，它只在 fast tokenizer 上实现，慢速
    # 实现直接抛 NotImplementedError。在这里查而不是等到 phase A —— 那时权重
    # 已经下完并搬上卡了。
    if not getattr(tok, "is_fast", False):
        print("  这个 tokenizer 不是 fast 版，phase A 的 offsets 定位不可用。")
        print("  phase B 不需要 offsets，可以用 --skip-copy 只跑它。")
        if not a.skip_copy:
            return

    # ---------------- phase A：零样本 copy 诊断 ----------------
    # 量"重复出现的值 token 上的 argmax 准确率"，与 train.py:321-329 同口径，
    # 只是 argmax 在 133 个单 token adj 上 gather 而不是在 value 块上切片。
    print(f"\n--- phase A：零样本 copy 诊断（{a.copy_docs} 篇 base 文档）---")
    hit = tot = hit_n = tot_n = 0
    base_docs = list(generate_corpus(v, nl_corpus_cfg(a.r, a.d, 7),
                                     a.copy_docs))
    for i in range(0, len(base_docs), 2):          # batch 2：全位置 logits 很大
        rds = [render(dc, v, tokenize) for dc in base_docs[i:i + 2]]
        texts = [" ".join(rd["tokens"]) for rd in rds]
        # offsets 而不是前缀长度。先前我用 len(encode(' '.join(toks[:p])))
        # 反推位置，那在 BPE 下不成立：merge 按优先级而非从左到右应用，故
        # encode(前缀) 不保证是 encode(全串) 的前缀，位置可能整体错位。错一格
        # 就在一个无关位置上取 argmax，而 copy_acc 只是"看起来低" —— 本会话
        # 最坏的失败方式。
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(dev),
                        attention_mask=enc["attention_mask"].to(dev))
        # 立刻降到候选集，否则 batch×seq×151665 的 fp32 副本会很大
        lg = out.logits[:, :, cid].float().cpu()
        ids_cpu = enc["input_ids"]
        del out
        for j, rd in enumerate(rds):
            pos, rep = adj_slots(rd)
            toks = rd["tokens"]
            # 每个 split token 在原串里的字符起点
            starts, c = [], 0
            for w in toks:
                starts.append(c)
                c += len(w) + 1
            om = offs[j].tolist()
            for p, rp in zip(pos, rep):
                cs = starts[p]
                # 覆盖该词首字符的 token。带前导空格的 token 其 offset 可能
                # 从空格开始，故用 <= cs < end 而不是相等。
                t = next((k for k, (s, e) in enumerate(om)
                          if s <= cs < e and e > s), None)
                if t is None or t == 0:
                    continue
                exp = tok.encode(" " + toks[p], add_special_tokens=False)
                # 自检：定位到的 token 必须就是那个词本身。不等说明该词在
                # 这个上下文里被切成多个 token（多 token 的 adj），跳过而不是
                # 在半个词上读 argmax。
                if len(exp) != 1 or int(ids_cpu[j, t]) != exp[0]:
                    continue
                want = (cid == exp[0]).nonzero()
                if not len(want):
                    continue
                ok = int(int(lg[j, t - 1].argmax()) == int(want[0]))
                if rp:
                    hit += ok
                    tot += 1
                else:
                    hit_n += ok
                    tot_n += 1
    nan = float("nan")
    copy_acc = hit / tot if tot else nan
    print(f"  copy_acc  {copy_acc:.4f}  (n={tot})"
          f"   novel {hit_n / tot_n if tot_n else nan:.4f}  (n={tot_n})")
    if copy_acc == copy_acc and copy_acc >= 0.95:
        print("  >= 0.95: the pretrained model already passes this copy diagnostic. ")
        print("  Copy accuracy and task accuracy must be reported separately; ")
        print("  passing a diagnostic does not identify a unique circuit.")
    else:
        print("  < 0.95: the pretrained model does not pass this copy diagnostic.")

    # ---------------- phase B：规则级先验 ----------------
    print(f"\n--- phase B：规则级先验（反转文档，p_break=1.0）---")
    cfg = inverted_cfg(a.r, a.d, 7)
    n_rar = n_rec = n_dec = n = 0
    lp_gap: List[float] = []
    n_skip = 0
    docs = list(generate_corpus(v, cfg, a.docs * 2))

    buf: List[Tuple[str, int, int]] = []
    for dc in docs:
        rr = rules_of(dc)
        if rr is None:
            n_skip += 1
            continue
        w_rar, w_rec = ADJ_OF(rr[0]), ADJ_OF(rr[1])
        if w_rar not in cands or w_rec not in cands:
            n_skip += 1
            continue
        rd = render(dc, v, tokenize)
        buf.append((prompt_of(rd), cands[w_rar], cands[w_rec]))
        if len(buf) >= a.docs:
            break

    for i in range(0, len(buf), a.batch):
        chunk = buf[i:i + a.batch]
        enc = tok([c[0] for c in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False)
        lens = enc["attention_mask"].sum(1)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(dev),
                        attention_mask=enc["attention_mask"].to(dev))
        lg = out.logits[torch.arange(len(chunk), device=dev),
                        lens.to(dev) - 1].float()
        del out
        blk = cid[lg[:, cid].argmax(-1)]
        for j, (_, i_rar, i_rec) in enumerate(chunk):
            n += 1
            lp_gap.append(float(lg[j, i_rar] - lg[j, i_rec]))
            am = int(blk[j])
            if am == i_rar:
                n_rar += 1
                n_dec += 1
            elif am == i_rec:
                n_rec += 1
                n_dec += 1

    if not n:
        print("  没有可用的反转文档。检查 p_break/k_old_break/r_new_break 是否")
        print("  是 CorpusCfg 的真实字段名，以及 rules_of 的两条规则是否分离。")
        return

    # 判据一：argmax 落在两个候选之一上的子集（决定性子集）
    p_rar = n_rar / n_dec if n_dec else nan
    se = (p_rar * (1 - p_rar) / n_dec) ** 0.5 if n_dec else nan
    # 判据二：全部文档上 logit(rarity) - logit(recency) 的符号。不需要 argmax
    # 落在候选上，故 n 更大、功效更高，但它是 logit 差而非行为。
    frac_rar_lp = sum(1 for x in lp_gap if x > 0) / len(lp_gap)
    med_gap = sorted(lp_gap)[len(lp_gap) // 2]

    print(f"  n={n}  跳过 {n_skip}（规则未分离或值非单 token）")
    print(f"  决定性子集 n_dec={n_dec}（argmax 落在两候选之一）")
    print(f"    p(rarity) = {p_rar:.4f}  se {se:.4f}"
          f"   95% CI [{max(0, p_rar - 1.96 * se):.3f}, "
          f"{min(1, p_rar + 1.96 * se):.3f}]")
    print(f"  全集 logit 差（rarity - recency）")
    print(f"    frac > 0 = {frac_rar_lp:.4f}   中位 {med_gap:+.4f} nats")

    print("\n判读")
    if n_dec < 30:
        print(f"  决定性子集只有 {n_dec} 篇，功效不足。加 --docs 或先看")
        print("  logit 差那个判据（它用全部 n 篇，功效更高但量的是 logit")
        print("  而非行为）。")
    lo = max(0.0, p_rar - 1.96 * se)
    hi = min(1.0, p_rar + 1.96 * se)
    if hi < 0.1:
        print("\n  Strong recency preference (upper bound for p(rarity) < 0.1). ")
        print("  The pretrained baseline is a competing explanation for ")
        print("  fine-tuning readouts; these baseline measurements alone ")
        print("  do not predict cross-seed dispersion after fine-tuning.")
    elif lo > 0.9:
        print("\n  Strong rarity preference in the pretrained baseline. ")
        print("  Separate this baseline preference from the effects of fine-tuning.")
    elif lo <= 0.5 <= hi:
        print("\n  The interval includes 0.5; this does not establish a neutral prior ")
        print("  or predict the dispersion of fine-tuned runs.")
    else:
        print("\n  Moderate baseline preference: report it alongside fine-tuning readouts.")
        print("  它是 σ 的一个竞争解释，读者需要它才能判断微调后的分散有多少")
        print("  是先验的残留。")
    print(f"\n  功效：n_dec={n_dec} 下 se={se:.3f}，能区分 0.5 与 "
          f"{0.5 + 3 * se:.2f}，不能区分 0.5 与 {0.5 + 1.5 * se:.2f}。")


if __name__ == "__main__":
    main()
