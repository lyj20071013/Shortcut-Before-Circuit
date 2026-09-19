"""Construct tokenizer-level probe data for the fine-tuning arm."""
import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MIRROR = "https://hf-mirror.com"


def canon(toks: Sequence[str]) -> str:
    """tokens -> 训练/编码用的底串。

    from-scratch 臂训练在 encode(rd["tokens"], vocab) 上（nl_corpus.py:148），
    answer_pos 索引 tokens，adj_slots 遍历 tokens —— 整条链的坐标系是 tokens。
    rd["text"] 没有任何已运行的代码验证过（prompt_of 走 tokens/answer_pos 那条
    路径，ft_ruleprior phase A 用的也是 join），而我在这个文件里把它当成了底串，
    于是 text 与 join 的差异让全部 1600 篇被跳过。

    用 join 而不是 text 还让两臂的训练输入一致：那边喂 tokens 的编码，这边喂
    同一个 token 序列的 BPE 编码。
    """
    return " ".join(toks)


def split_to_bpe(tok, toks: Sequence[str],
                 want: Sequence[int]) -> Dict[int, int]:
    """{split token 下标: 该词最后一个 BPE token 的下标}。

    只对 want 里的下标建映射。用 offset_mapping 而不是前缀长度：后者在 BPE 下
    不成立（见模块 docstring）。

    底串由 canon(toks) 生成，不接受外部传入 —— 先前它收一个 text 参数并自检
    text == ' '.join(toks)，而调用方传的是 rd["text"]，两者不等，于是全部文档
    在这里被跳过。tokens 是权威坐标系（answer_pos 索引它，adj_slots 遍历它），
    所以底串必须由它生成，没有第二个真相。
    """
    text = canon(toks)
    starts, c = [], 0
    for w in toks:
        starts.append(c)
        c += len(w) + 1

    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    om = enc["offset_mapping"]
    out: Dict[int, int] = {}
    for p in want:
        cs, ce = starts[p], starts[p] + len(toks[p])
        # 覆盖该词的最后一个 BPE token：它的 offset 与 [cs, ce) 相交且 end 最大
        last = None
        for k, (s, e) in enumerate(om):
            if e > s and s < ce and e > cs:
                last = k
        if last is not None:
            out[p] = last
    return out


def build(tok, r: int, d: int, n: int, seed: int, spec, cands: Dict[str, int]
          ) -> dict:
    """探针集。字段与 nl_corpus.build_probe 对应，但位置在 BPE 空间。"""
    import random as _random

    from generator import generate_corpus
    from nl_corpus import ADJ_OF, adj_slots, nl_corpus_cfg, tokenize
    from nl_render import render, render_edit_pair
    from probe import apply_edit, fit_position_offset
    from vocab import Vocab

    v = Vocab(spec)
    cfg = nl_corpus_cfg(r, d, seed)
    warm = list(generate_corpus(v, cfg, 600, seed_offset=1))
    offset = fit_position_offset(warm)
    rng = _random.Random(seed)

    base, edit, apos, eapos, ans, vst = [], [], [], [], [], []
    cpos, crep = [], []
    n_len = mx = 0
    # 分项计数。先前这四个原因合并成一个 n_skip，于是"只凑到 0/200（跳过
    # 1600）"这个报错说不出是哪一个 —— 而四个原因的修法完全不同。
    sk = dict(edit_none=0, ans_notin=0, vst_notin=0, bpe=0, truth=0)
    shown = set()

    for doc in generate_corpus(v, cfg, n * 8, seed_offset=1):
        ed = apply_edit(doc, "break_rarity", v, cfg, rng, offset)
        if ed is None:
            sk["edit_none"] += 1
            continue
        rb, re_ = render_edit_pair(doc, ed, v, tokenize)

        # 首篇结构探测：一次性打印全部相关字段。猜这些字段的语义是本会话
        # 最大的时间损失来源，所以这里直接看。
        if "probe" not in shown:
            shown.add("probe")
            tx, tk = rb.get("text"), rb.get("tokens")
            print(f"  [探测] rb 的键 {sorted(rb)}")
            print(f"  [探测] answer {rb.get('answer')!r}"
                  f"  answer_pos {rb.get('answer_pos')}")
            print(f"  [探测] v_star(raw) {ed.v_star}"
                  f"  ADJ_OF -> {ADJ_OF(ed.v_star)!r}")
            print(f"  [探测] n_values {spec.n_values}  cands {len(cands)}")
            if tk is not None:
                print(f"  [探测] tokens[:6] {tk[:6]}")
                print(f"  [探测] tokens[ap] {tk[rb['answer_pos']]!r}")
            if tx is not None:
                j = " ".join(tk) if tk else ""
                print(f"  [探测] text == ' '.join(tokens)? {tx == j}")
                print(f"  [探测] text[:70] {tx[:70]!r}")
                if tx != j:
                    print(f"  [探测] join[:70] {j[:70]!r}")

        # 答案与 v_star 必须都是单 token adj，否则单个位置的 argmax 取不到它们
        w_ans, w_vst = rb["answer"], ADJ_OF(ed.v_star)
        if w_ans not in cands:
            sk["ans_notin"] += 1
            if "ans" not in shown:
                shown.add("ans")
                print(f"  [诊断] answer {w_ans!r} 不在 cands。"
                      f"cands 前 5 {list(cands)[:5]}")
            continue
        if w_vst not in cands:
            sk["vst_notin"] += 1
            if "vst" not in shown:
                shown.add("vst")
                print(f"  [诊断] v_star 的 adj {w_vst!r} 不在 cands"
                      f"（v_star raw = {ed.v_star}，n_values = "
                      f"{spec.n_values}）。v_star 可能不在 [0, n_values) 里。")
            continue

        ok = True
        rec = {}
        for tag, rd in (("b", rb), ("e", re_)):
            # canon(tokens) 而不是 rd["text"]。实测两者不等（前 70 字符相同、
            # 末尾不同），而 tokens 是权威坐标系：answer_pos 索引它、adj_slots
            # 遍历它、from-scratch 臂训练在它的编码上。
            text = canon(rd["tokens"])
            ap = rd["answer_pos"]

            # answer_pos 走 split_to_bpe 的 offsets，不用前缀长度。
            #
            # split_to_bpe 返回该词**最后一个** BPE token 的下标 k。答案是单
            # token，故 k 就是答案自己的位置；读 logits[k-1] 预测第 k 个 token，
            # 与 train.py:313 的 ps.append(vpos - 1) 同一约定。
            ids = tok.encode(text, add_special_tokens=False)
            exp = tok.encode(" " + rd["tokens"][ap], add_special_tokens=False)
            m = split_to_bpe(tok, rd["tokens"], [ap])
            k = m.get(ap)
            if len(exp) != 1 or k is None or k == 0 or ids[k] != exp[0]:
                ok = False
                if "bpe" not in shown:
                    shown.add("bpe")
                    why = ("答案不是单 token" if len(exp) != 1 else
                           "offsets 没覆盖到答案的字符区间" if k is None else
                           "答案落在第 0 个 token（没有前文可读）" if k == 0 else
                           "ids[k] 不是答案")
                    print(f"  [诊断] BPE 定位失败（{tag} 侧）：{why}")
                    print(f"         answer {rd['tokens'][ap]!r}  ap {ap}"
                          f"  exp {exp}  k {k}  len(ids) {len(ids)}")
                    if k is not None and 0 <= k < len(ids):
                        print(f"         ids[k] = {tok.decode([ids[k]])!r}")
                break
            rec[tag] = (ids, k, exp[0], text, rd)
        if not ok:
            sk["bpe"] += 1
            continue

        ib, lb, aid, tb, rdb = rec["b"]
        ie, le, aid2, _te, _ = rec["e"]
        if aid != aid2:                       # 真值两侧必须相同
            sk["truth"] += 1
            continue

        # G1 逐篇。ft_pool 在 gate_pairs 上验过 n_bad=0/340，但探针集是另一批
        # 文档（seed=777000、seed_offset=1），而 G1 是 Δ 的承重前提：token 数
        # 不等意味着 base 与 edit 的长度差本身携带信息，而长度是模型看得见的
        # 量。抛而不是跳过 —— 过滤之后这里不该再有一例，出现了说明长度差另有
        # 来源（模板？attr？所有格？），那要先找出来而不是静默丢掉样本。
        if len(ib) != len(ie):
            raise ValueError(
                f"G1 破：base {len(ib)} token 而 edit {len(ie)}。ft_pool 报过"
                f"过滤后 n_bad=0/340，所以这里出现说明长度差还有别的来源。"
                f"答案 token {tok.decode([aid])!r}")
        # 答案位置也该相同：编辑只换被查询 slot 的值，不动位置或模板
        if lb != le:
            raise ValueError(
                f"answer_pos 两侧不等（{lb} vs {le}）而 token 数相等。编辑改"
                f"动了答案之前的 token 数，Δ 的两项不在同构位置上取。")

        if len(ib) > spec.ctx_len:
            n_len += 1
            continue
        mx = max(mx, len(ib))

        # copy 诊断位置。split 下标 -> BPE 下标，只保留能映射且是单 token 的
        sp, sr = adj_slots(rdb)
        m = split_to_bpe(tok, rdb["tokens"], sp)
        bp, br = [], []
        for p, rp in zip(sp, sr):
            k = m.get(p)
            if k is None or k == 0:
                continue
            e2 = tok.encode(" " + rdb["tokens"][p], add_special_tokens=False)
            if len(e2) != 1 or ib[k] != e2[0]:
                continue
            bp.append(k)
            br.append(rp)

        base.append(ib)
        edit.append(ie)
        apos.append(lb)
        eapos.append(le)
        ans.append(aid)
        vst.append(cands[w_vst])
        cpos.append(bp)
        crep.append(br)
        if len(base) >= n:
            break

    tot_skip = sum(sk.values())
    if len(base) < n:
        top = max(sk, key=lambda k: sk[k])
        hint = {
            "edit_none": "break_rarity 的域为空 —— apply_edit 对这个 cfg 返回"
                         "不了编辑对。查 probe.py:21-24 的三条判定条件。",
            "ans_notin": "答案 adj 不在 cands。cands 由 ADJ[:n_values] 建，"
                         "所以这说明 render 的 answer 不是一个 adj，或 install "
                         "改了池子而 cands 用了另一份。",
            "vst_notin": "v_star 的 adj 不在 cands。v_star 可能不是 [0, "
                         "n_values) 里的 raw 值索引 —— 若它带 VAL0 偏移，"
                         "ADJ_OF 会取到错的词或越界。",
            "bpe": "BPE 位置定位失败。用前缀长度反推不成立（encode(前缀) 不是 "
                   "encode(全串) 的前缀），要改用 offset_mapping —— "
                   "split_to_bpe 已经是那个做法，answer_pos 也该走它。",
            "truth": "真值两侧不同。编辑改了答案 token，那不是 break_rarity "
                     "的语义。",
        }[top]
        raise ValueError(
            f"只凑到 {len(base)}/{n} 篇。分项：{sk}，超长 {n_len}。\n"
            f"  主因是 {top}（{sk[top]} 篇）：{hint}")
    return dict(base=base, edit=edit, answer_pos=apos, edit_answer_pos=eapos,
                answer=ans, v_star=vst, copy_pos=cpos, copy_rep=crep,
                n_skip=tot_skip, skip_by=sk, n_len=n_len, max_tok=mx)


def pad2d(seqs: Sequence[Sequence[int]], width: int, fill: int) -> np.ndarray:
    a = np.full((len(seqs), width), fill, dtype=np.int32)
    for i, s in enumerate(seqs):
        a[i, :len(s)] = s
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--src", default="hf", choices=["hf", "ms"])
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--probe-docs", type=int, default=200)
    ap.add_argument("--ctx-len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=777000)
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--out", default="ft_data")
    a = ap.parse_args()
    if a.src == "hf" and not a.no_mirror:
        os.environ.setdefault("HF_ENDPOINT", MIRROR)

    from ft_pool import ft_spec, install, verify_install
    from ft_tokcheck import get_tokenizer
    from nl_generator import values_needed

    tok, via = get_tokenizer(a.model, a.src)
    if not getattr(tok, "is_fast", False):
        raise SystemExit("需要 fast tokenizer（offset_mapping）。")
    ver = install(tok, need=values_needed(55, 3), model_name=a.model)
    verify_install()
    spec = ft_spec(a.ctx_len)

    import nl_generator
    cands = {}
    for w in nl_generator.ADJ[:spec.n_values]:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{w!r} 不是单 token —— ft_pool 的过滤没生效")
        cands[w] = ids[0]
    print(f"来源 {via}  版本 {ver}  候选 {len(cands)}  ctx_len {spec.ctx_len}")

    # 探针集 seed 与训练流不交：训练用 1000*seed + w（seed<=9 时 <= 9003），
    # 探针用 777000，与 nl_corpus 的同一个约定。
    pr = build(tok, a.r, a.d, a.probe_docs, a.seed, spec, cands)
    os.makedirs(a.out, exist_ok=True)

    pad = tok.pad_token_id
    if pad is None:
        pad = tok.eos_token_id
    w = pr["max_tok"]
    cw = max((len(x) for x in pr["copy_pos"]), default=1)
    p = os.path.join(a.out, f"probe_R{a.r}_D{a.d}.npz")
    np.savez_compressed(
        p, version=ver, model=a.model, pad_id=pad,
        base=pad2d(pr["base"], w, pad), edit=pad2d(pr["edit"], w, pad),
        base_len=np.array([len(x) for x in pr["base"]], dtype=np.int32),
        edit_len=np.array([len(x) for x in pr["edit"]], dtype=np.int32),
        answer_pos=np.array(pr["answer_pos"], dtype=np.int32),
        edit_answer_pos=np.array(pr["edit_answer_pos"], dtype=np.int32),
        answer=np.array(pr["answer"], dtype=np.int32),
        v_star=np.array(pr["v_star"], dtype=np.int32),
        copy_pos=pad2d(pr["copy_pos"], cw, -1),
        copy_rep=pad2d(pr["copy_rep"], cw, -1),
        adj_ids=np.array([cands[x] for x in nl_generator.ADJ[:spec.n_values]],
                         dtype=np.int32))

    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(dict(version=ver, model=a.model, spec=spec.__dict__,
                       r=a.r, d=a.d, probe_seed=a.seed, pad_id=pad,
                       n_values=spec.n_values,
                       adj=nl_generator.ADJ[:spec.n_values],
                       noun=nl_generator.NOUN), f, indent=1,
                  ensure_ascii=False)

    n_cp = int((pad2d(pr["copy_pos"], cw, -1) >= 0).sum())
    print(f"\n探针 {len(pr['base'])} 篇  跳过 {pr['n_skip']}  超长 {pr['n_len']}")
    print(f"  max_tok {pr['max_tok']}  copy 位置 {n_cp}"
          f"  (每篇 {n_cp / len(pr['base']):.1f})")
    print(f"  -> {p}")
    print(f"\n训练集是流式的，不落盘。版本 {ver} 必须与 ft_train 记录的一致。")


if __name__ == "__main__":
    main()
