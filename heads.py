"""Patch residual streams and ablate heads in disagreement-supervision arms."""
import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch

from config import CorpusCfg, LangSpec
from flatdir import load_ckpt
from generator import Doc, generate_corpus
from model import LM
from probe import r_last_value, r_rarity, _q_pos, _q_stmts
from train import collate
from vocab import Vocab

NAN = float("nan")

def val_tok_pos(i: int) -> int:
    """语句 i 的 value token 在序列里的下标。

    语句恒占 4 个 token [ent][attr][val][SEP]（论文 §2.1），故 value 在
    4i+2。build() 会对每一篇断言这个位置的 token 等于 vocab.val(该语句的
    值)，所以布局若变了这里立刻失败，不会静默给出错位的注意力。"""
    return 4 * i + 2


class Row:
    """一篇文档的分析输入。

    p_old   被取代副本的 value token 位置。反转文档上它只有一个元素，
            即 rarity 的目标；非反转文档上是 R_old 个副本。
    p_last  q slot 最后一条语句的 value token 位置，即 recency 的目标。
    """
    __slots__ = ("toks", "apos", "p_old", "p_rare", "p_last",
                 "tok_rare", "tok_last", "is_break")

    def __init__(self, toks, apos, p_old, p_rare, p_last,
                 tok_rare, tok_last, is_break):
        self.toks, self.apos = toks, apos
        self.p_old, self.p_rare, self.p_last = p_old, p_rare, p_last
        self.tok_rare, self.tok_last = tok_rare, tok_last
        self.is_break = is_break


def build(vocab: Vocab, corpus: CorpusCfg, n_docs: int,
          seed_offset: int) -> Tuple[List[Row], List[Row]]:
    """返回 (反转文档, 非反转文档)。

    反转的判定是 r_rarity(d) != r_last_value(d)，即两条规则真的分歧 ——
    这正是"反转"的定义，不依赖生成器字段。
    """
    brk, nb = [], []
    for d in generate_corpus(vocab, corpus, n_docs, seed_offset=seed_offset):
        if d.q_hist_k:                      # 历史索引查询，读数不适用
            continue
        v_last, v_rare = r_last_value(d), r_rarity(d)
        if v_last is None or v_rare is None:
            continue
        qp = _q_pos(d)
        if len(qp) < 2:
            continue
        i_last = qp[-1]
        p_last = val_tok_pos(i_last)
        # 断言 token 布局：value 必须落在 4i+2
        assert d.tokens[p_last] == vocab.val(d.stmts[i_last].val), \
            f"token 布局不符：位置 {p_last} 不是 value token"
        if p_last >= d.answer_pos:           # 该 slot 的末条在答案位之后
            continue
        p_old = [val_tok_pos(i) for i in qp[:-1]
                 if val_tok_pos(i) < d.answer_pos]
        if not p_old:
            continue
        # p_rare 只含 rarity 的目标位置。反转文档上 p_old 里混着 R-1 份
        # v_new，那些不是任何规则的目标，会把「谁更注意罕见值」这个问题
        # 变成「谁更扫 q slot」。用 token id 直接筛出来。
        t_rare = vocab.val(v_rare)
        is_brk = v_rare != v_last
        # 非反转文档上 rarity 的目标就是末条（v_rare == v_last），它在
        # p_last 里而不在 p_old 里。上一版从 p_old 筛 t_rare，于是把全部
        # 非反转文档筛空了 —— 对照组消失，判据 A 只剩半条腿。
        if is_brk:
            p_rare = [p for p in p_old if d.tokens[p] == t_rare]
            if not p_rare:
                continue
        else:
            p_rare = [p_last]
        r = Row(d.tokens[:d.answer_pos], d.answer_pos,
                p_old, p_rare, p_last,
                t_rare, vocab.val(v_last),
                is_brk)
        (brk if r.is_break else nb).append(r)
    return brk, nb


def batches(rows: List[Row], bs: int, pad: int):
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ids, _ = collate([r.toks for r in chunk], pad)
        yield chunk, ids

@torch.no_grad()
def attention(model: LM, rows: List[Row], bs: int, pad: int,
              device) -> Dict[str, torch.Tensor]:
    """答案位上，逐 (layer, head) 的注意力质量去向。

    读的是最后一个前缀位置 apos-1 的那一行注意力。右 padding 在 causal
    注意力下不泄漏：该位置只能看到 <= 自己的下标。
    """
    L, H = model.cfg.n_layer, model.cfg.n_head
    a_old = torch.zeros(L, H, dtype=torch.float64)
    a_rare = torch.zeros(L, H, dtype=torch.float64)
    a_last = torch.zeros(L, H, dtype=torch.float64)
    n = 0
    for chunk, ids in batches(rows, bs, pad):
        cache = {"want_pattern": True}
        model(ids.to(device), cache=cache)
        for lay in range(L):
            att = cache[f"pattern.{lay}"]        # (B, H, T, T)
            for b, r in enumerate(chunk):
                row = att[b, :, r.apos - 1, :]   # (H, T)
                a_last[lay] += row[:, r.p_last].double().cpu()
                a_old[lay] += row[:, r.p_old].sum(-1).double().cpu()
                a_rare[lay] += row[:, r.p_rare].sum(-1).double().cpu()
        n += len(chunk)
        del cache
    return {"a_old": a_old / max(n, 1), "a_rare": a_rare / max(n, 1),
            "a_last": a_last / max(n, 1), "n": n}

def ablate_hook(head: int, d_head: int):
    """把某个 head 在 attn.out 输入侧的通道段置零。

    out 的输入是 concat 后的 (B, T, H*D)，所以 head h 占
    [h*D, (h+1)*D)。置零等价于移除该 head 的贡献，其余 head 不受影响。
    """
    lo, hi = head * d_head, (head + 1) * d_head

    def hook(mod, args):
        (x,) = args
        y = x.clone()
        y[..., lo:hi] = 0.0
        return (y,)
    return hook

def ablate_heads_hook(heads: List[int], d_head: int):
    """同时置零多个 head。判定 3.2 与 6.7 是冗余还是互补：
    掉幅叠加到接近 0 => 互补（各管一部分）；
    掉幅不叠加（约等于单个的） => 冗余（同一条路径的两个环节）。"""
    segs = [(h * d_head, (h + 1) * d_head) for h in heads]

    def hook(mod, args):
        (x,) = args
        y = x.clone()
        for lo, hi in segs:
            y[..., lo:hi] = 0.0
        return (y,)
    return hook

@torch.no_grad()
def behaviour(model: LM, brk: List[Row], nb: List[Row], bs: int, pad: int,
              device) -> Tuple[float, float]:
    """(反转文档上 rare 胜过 last 的比例, 非反转文档上的 argmax 准确率)。

    第一个量在两条 arm 上是精确互镜的（rarity 模型 ~1，recency 模型 ~0），
    所以它就是行为标签，不需要 truth_rule 字段。
    """
    win = tot = 0
    for chunk, ids in batches(brk, bs, pad):
        lg, _ = model(ids.to(device))
        for b, r in enumerate(chunk):
            z = lg[b, r.apos - 1]
            win += int(z[r.tok_rare] > z[r.tok_last])
            tot += 1
    frac = win / tot if tot else NAN

    hit = tot2 = 0
    for chunk, ids in batches(nb, bs, pad):
        lg, _ = model(ids.to(device))
        for b, r in enumerate(chunk):
            hit += int(int(lg[b, r.apos - 1].argmax()) == r.tok_last)
            tot2 += 1
    return frac, (hit / tot2 if tot2 else NAN)

def patch_hook(pos_pairs: List[Tuple[int, int]], src: torch.Tensor):
    """把 resid_pre 在指定位置换成 src 里对应行的激活（resample ablation）。

    pos_pairs 是 [(batch_row, position)]，src 形状 (len(pos_pairs), d_model)。
    比 head ablation 更精确：它定位的是 token 位置而非 head。"""
    def hook(mod, args, kwargs):
        x = args[0].clone()
        for i, (b, p) in enumerate(pos_pairs):
            x[b, p] = src[i].to(x.dtype)
        return (x,) + args[1:], kwargs
    return hook


@torch.no_grad()
def patching(model: LM, brk: List[Row], nb: List[Row], layer: int,
             bs: int, pad: int, device) -> float:
    """在 rarity 模型上，把反转文档罕见值位置的 resid_pre 换成非反转文档
    同角色位置的激活，看 frac(rare>last) 掉多少。

    源取非反转文档的被取代副本位置：那里的值在该文档里**不是**罕见的
    （它有 R 份），所以换过去等于抹掉"这个值只出现一次"这个信息。"""
    src_cache: List[torch.Tensor] = []
    for chunk, ids in batches(nb, bs, pad):
        c = {}
        model(ids.to(device), cache=c)
        rp = c[f"resid_pre.{layer}"]
        for b, r in enumerate(chunk):
            src_cache.append(rp[b, r.p_old[0]].detach().cpu())
        del c
        if len(src_cache) >= len(brk):
            break
    if len(src_cache) < len(brk):
        src_cache = (src_cache * (len(brk) // len(src_cache) + 1))[:len(brk)]

    win = tot = 0
    off = 0
    for chunk, ids in batches(brk, bs, pad):
        pairs = [(b, r.p_rare[0]) for b, r in enumerate(chunk)]
        src = torch.stack(src_cache[off:off + len(chunk)]).to(device)
        off += len(chunk)
        hk = model.blocks[layer].register_forward_pre_hook(
            patch_hook(pairs, src), with_kwargs=True)
        try:
            lg, _ = model(ids.to(device))
        finally:
            hk.remove()
        for b, r in enumerate(chunk):
            z = lg[b, r.apos - 1]
            win += int(z[r.tok_rare] > z[r.tok_last])
            tot += 1
    return win / tot if tot else NAN

def run_one(path: str, docs: int, offset: int, bs: int, device):
    model, spec, corpus, step = load_ckpt(path, device)
    vocab = Vocab(spec)
    brk, nb = build(vocab, corpus, docs, offset)
    if not brk:
        raise SystemExit(
            f"{path}: 没有反转文档。该 checkpoint 的 corpus 里 p_break="
            f"{getattr(corpus, 'p_break', None)}，臂的 ckpt 应当非零。")
    att_b = attention(model, brk, bs, vocab.PAD, device)
    att_n = attention(model, nb, bs, vocab.PAD, device)
    frac, acc = behaviour(model, brk, nb, bs, vocab.PAD, device)
    return dict(path=path, step=step, model=model, vocab=vocab, corpus=corpus,
                brk=brk, nb=nb, att_b=att_b, att_n=att_n,
                frac=frac, acc=acc)


def main():
    ap = argparse.ArgumentParser(
        description="规则的内部定域：注意力去向与逐 head ablation（非别名臂）")
    ap.add_argument("--rar", default="runs_nb/R3_D8_s0_nbrar30.pt",
                    help="truth_rule=rarity 的 ckpt（已知实现 rarity）")
    ap.add_argument("--rec", default="runs_nb/R3_D8_s0_nbrec30.pt",
                    help="truth_rule=recency 的 ckpt（已知实现 recency）")
    ap.add_argument("--docs", type=int, default=400,
                    help="生成文档数；反转的约占 p_break")
    ap.add_argument("--offset", type=int, default=1,
                    help="与 go_nogo 读数同一批文档")
    ap.add_argument("--batch", type=int, default=16,
                    help="注意力矩阵是 (B,H,T,T)，T~220，别开太大")
    ap.add_argument("--ablate-batch", type=int, default=64)
    ap.add_argument("--out", default="runs_nb/heads.jsonl")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--pair", default="",
                    help="同时 ablate 两个 head，格式 3.2,6.7")
    ap.add_argument("--patch-layers", default="",
                    help="逐层做 activation patching，如 0,1,2,3,4,5,6,7")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False

    R = run_one(a.rar, a.docs, a.offset, a.batch, device)
    C = run_one(a.rec, a.docs, a.offset, a.batch, device)

    L, H = R["model"].cfg.n_layer, R["model"].cfg.n_head
    print(f"反转文档 {len(R['brk'])} 篇，非反转 {len(R['nb'])} 篇 "
          f"（{a.docs} 篇生成，p_break={getattr(R['corpus'], 'p_break', '?')}）")
    print(f"\n行为标签（frac(rare>last) 应精确互镜）")
    print(f"  rarity 模型  frac {R['frac']:.3f}   非反转 acc {R['acc']:.4f}")
    print(f"  recency 模型 frac {C['frac']:.3f}   非反转 acc {C['acc']:.4f}")

    # ---- 判据 A：路由 ----
    print(f"\n[A] 注意力去向。brk = 反转文档，nb = 非反转（对照，两个目标"
          f"在那里重合，故 Δ 应接近 0）")
    print(f"{'layer.head':>11} {'Δrare_brk':>10} {'Δrare_nb':>10} "
          f"{'Δlast_brk':>10} {'Δlast_nb':>10}  判定")
    hits_a = []
    for lay in range(L):
        for h in range(H):
            dr_b = float(R['att_b']['a_rare'][lay, h] - C['att_b']['a_rare'][lay, h])
                        # 非反转文档上两个目标重合，故 a_rare 在那里等于 a_last，做不了
            # 对照。改用 a_old：那里被取代副本有 R 份，不是任何规则的目标，
            # 所以两个模型在它上面的差异是泛泛的模型差异而非规则特异的。
            dr_n = float(R['att_n']['a_old'][lay, h] - C['att_n']['a_old'][lay, h])
            dl_b = float(R['att_b']['a_last'][lay, h] - C['att_b']['a_last'][lay, h])
            dl_n = float(R['att_n']['a_last'][lay, h] - C['att_n']['a_last'][lay, h])
            # 规则特异 = 反转文档上有差异，而对照上没有。分工可以分给不同
            # 的 head，所以 rarity 路由与 recency 路由分别判。
            tag = ""
            if dr_b >= 0.10 and abs(dr_n) <= 0.05:
                tag = "<= rarity 路由"
                hits_a.append((lay, h, "rar", dr_b, dr_n))
            elif dl_b <= -0.10 and abs(dl_n) <= 0.05:
                tag = "<= recency 路由"
                hits_a.append((lay, h, "rec", dl_b, dl_n))
            elif dr_b >= 0.10 or dl_b <= -0.10:
                tag = "!! 对照也大，非规则特异"
            if abs(dr_b) >= 0.05 or abs(dl_b) >= 0.05 or tag:
                print(f"{lay:>6}.{h:<4} {dr_b:>+10.3f} {dr_n:>+10.3f} "
                      f"{dl_b:>+10.3f} {dl_n:>+10.3f}  {tag}")
    print(f"  => 满足判据 A 的 head：{len(hits_a)}")
    if not hits_a:
        print("     没有 head 表现出规则特异的路由。注意力层面不定域。")

    # ---- 判据 B：ablation ----
    hits_b = []
    if not a.skip_ablation:
        print(f"\n[B] 逐 head ablation（在 rarity 模型上）")
        print(f"  基线 frac {R['frac']:.3f}  acc {R['acc']:.4f}")
        print(f"{'layer.head':>11} {'frac':>7} {'Δfrac':>8} {'acc':>8} "
              f"{'Δacc':>8}  判定")
        m, v = R["model"], R["vocab"]
        dh = m.cfg.d_head
        for lay in range(L):
            for h in range(H):
                hk = m.blocks[lay].attn.out.register_forward_pre_hook(
                    ablate_hook(h, dh))
                try:
                    f2, a2 = behaviour(m, R["brk"], R["nb"],
                                       a.ablate_batch, v.PAD, device)
                finally:
                    hk.remove()
                df, da = f2 - R["frac"], a2 - R["acc"]
                ok = df <= -0.5 and da >= -0.01
                if ok:
                    hits_b.append((lay, h, df, da))
                if df <= -0.1 or da <= -0.05 or ok:
                    print(f"{lay:>6}.{h:<4} {f2:>7.3f} {df:>+8.3f} "
                          f"{a2:>8.4f} {da:>+8.4f}  {'<= 承载规则' if ok else ''}")
        print(f"  => 满足判据 B 的 head：{len(hits_b)}")
        if not hits_b:
            print("     没有单个 head 同时满足两个条件。规则不定域在单个 head 上，")
            print("     这与 chen2026rome 的近乎不交电路一致，且它本身是一个结果。")

    if a.pair:
        m, v, dh = R["model"], R["vocab"], R["model"].cfg.d_head
        picks = []
        for tok in a.pair.split(","):
            l_, h_ = tok.strip().split(".")
            picks.append((int(l_), int(h_)))
        print(f"\n[C] 联合 ablation {picks}")
        # 先各自单独，再联合，同一批文档同一口径
        singles = []
        for l_, h_ in picks:
            hk = m.blocks[l_].attn.out.register_forward_pre_hook(
                ablate_hook(h_, dh))
            try:
                f2, a2 = behaviour(m, R["brk"], R["nb"], a.ablate_batch,
                                   v.PAD, device)
            finally:
                hk.remove()
            singles.append(f2)
            print(f"  单独 {l_}.{h_}     frac {f2:.3f}  Δ {f2 - R['frac']:+.3f}  "
                  f"acc {a2:.4f}")
        by_layer: Dict[int, List[int]] = {}
        for l_, h_ in picks:
            by_layer.setdefault(l_, []).append(h_)
        hks = [m.blocks[l_].attn.out.register_forward_pre_hook(
                   ablate_heads_hook(hs, dh))
               for l_, hs in by_layer.items()]
        try:
            fj, aj = behaviour(m, R["brk"], R["nb"], a.ablate_batch,
                               v.PAD, device)
        finally:
            for hk in hks:
                hk.remove()
        print(f"  联合          frac {fj:.3f}  Δ {fj - R['frac']:+.3f}  "
              f"acc {aj:.4f}")
        add = sum(R['frac'] - s for s in singles)
        print(f"  单独掉幅之和 {add:.3f}   联合掉幅 {R['frac'] - fj:.3f}")
        print(f"  => {'互补（掉幅叠加）' if R['frac'] - fj > 0.8 * add else '冗余（掉幅不叠加）'}")

    if a.patch_layers:
        print(f"\n[D] Activation patching（rarity 模型，罕见值位置的 resid_pre）")
        print(f"  基线 frac {R['frac']:.3f}")
        for ls in a.patch_layers.split(","):
            lay = int(ls.strip())
            f2 = patching(R["model"], R["brk"], R["nb"], lay,
                          a.ablate_batch, R["vocab"].PAD, device)
            print(f"  layer {lay:>2}  frac {f2:.3f}  Δ {f2 - R['frac']:+.3f}")

    with open(a.out, "a") as f:
        f.write(json.dumps(dict(
            rar=a.rar, rec=a.rec, step_rar=R["step"], step_rec=C["step"],
            n_brk=len(R["brk"]), n_nb=len(R["nb"]),
            frac_rar=R["frac"], frac_rec=C["frac"],
            acc_rar=R["acc"], acc_rec=C["acc"],
            a_rare_brk_rar=R["att_b"]["a_rare"].tolist(),
            a_rare_brk_rec=C["att_b"]["a_rare"].tolist(),
            a_rare_nb_rar=R["att_n"]["a_rare"].tolist(),
            a_rare_nb_rec=C["att_n"]["a_rare"].tolist(),
            a_last_brk_rar=R["att_b"]["a_last"].tolist(),
            a_last_brk_rec=C["att_b"]["a_last"].tolist(),
            a_last_nb_rar=R["att_n"]["a_last"].tolist(),
            a_last_nb_rec=C["att_n"]["a_last"].tolist(),
            a_old_brk_rar=R["att_b"]["a_old"].tolist(),
            a_old_nb_rar=R["att_n"]["a_old"].tolist(),
            hits_a=hits_a, hits_b=hits_b)) + "\n")
    print(f"\n写入 {a.out}")


if __name__ == "__main__":
    main()
