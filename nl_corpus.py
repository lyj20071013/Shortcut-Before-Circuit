"""Build rendered-language vocabulary and probe arrays; training data stream online."""
import argparse
import dataclasses
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from config import CorpusCfg, LangSpec, dd_band
from generator import generate_corpus
from nl_render import check_pools, render, render_edit_pair
from vocab import Vocab

PAD = 0
CORPUS_VERSION = "nl-render-1"    # 语料版本。改生成或渲染就要 +1，见文件末尾


def tokenize(s: str) -> List[str]:
    """与 nl_gates.whitespace_tok 同一函数。改这里必须重跑所有门。"""
    return s.split()


def nl_spec(ctx_len: int = 600) -> LangSpec:
    """NL 臂的 LangSpec。

    n_values=150 而非主网格的 512。两条下界夹出来的：
      上界 n_values <= len(ADJ)=170，adj 必须对 val_id 单射
      下界 generator._ValueDraw 在 n_stmts_hi=55、R_old=3 上耗尽 100 个值，
           它自己的提示是"约需 120"
    150 同时满足，且留了余量。读数位置的随机基线因此是 1/150=0.0067。

    The vocabulary size differs from the main grid and changes the chance
    baseline; further effects are not isolated by this arm. Keeping the
    statement-count range avoids an additional document-length change.

    ctx_len=600 与主网格相同。NL 表层最坏 55×8+4 = 444 token（8 是 T1/T2/T3
    的长度，T0 是 6），留 156 的余量给 bump_freq 增加的语句。

    保持 600 而不是放宽到 900 是刻意的：ctx_len 进 ModelCfg，直接决定模型的
    位置容量与 attention 成本。两臂用不同的 ctx_len 会让"NL 臂的 σ"与主网格
    的不可比 —— 那是另一个旋钮，而 B1 只想换表层这一个。

    validate_cfg 的 worst_tok 检查按主网格的 4 token/语句 算，对 NL 表层
    不适用（它算出 225，实际 444）。真实上限由 build_probe 逐篇实测并在超出
    时抛异常。
    """
    return LangSpec(n_values=150, n_entities=40, n_attrs=8, ctx_len=ctx_len)


def nl_corpus_cfg(r: int, d: int, seed: int, n_lo: int = 45,
                  n_hi: int = 55) -> CorpusCfg:
    """与 sweep.py 的主网格 cfg 同构，只有 n_values/n_entities 由 nl_spec 定。
    p_update=0.5、max_updates=1、use_marker=False 全部照抄。"""
    dlo, dhi = dd_band(d)
    return CorpusCfg(name=f"nlR{r}_D{d}_s{seed}", seed=seed, p_update=0.5,
                     max_updates=1, r_old_lo=r, r_old_hi=r, use_marker=False,
                     delta_d_lo=dlo, delta_d_hi=dhi, p_hist_query=0.0,
                     n_stmts_lo=n_lo, n_stmts_hi=n_hi)


def build_vocab(cells: Sequence[Tuple[int, int]], spec: LangSpec,
                n_probe: int = 400, seeds: Sequence[int] = (7, 8)
                ) -> Dict[str, int]:
    """扫描语料建词表。id 0 留给 PAD。

    闭合性：seeds × cells 各采 n_probe 篇取并集。之后 encode 遇到未见 token
    抛异常而不是静默 UNK —— 静默 UNK 会让某些值在读数位置不可分，那正是
    Δ 恒为 0 的一种伪像。
    """
    vocab: Dict[str, int] = {"<pad>": PAD}
    v = Vocab(spec)
    for s in seeds:
        for r, d in cells:
            cfg = nl_corpus_cfg(r, d, s)
            for doc in generate_corpus(v, cfg, n_probe):
                for t in render(doc, v, tokenize)["tokens"]:
                    if t not in vocab:
                        vocab[t] = len(vocab)
    return vocab


def encode(toks: Sequence[str], vocab: Dict[str, int]) -> List[int]:
    out = []
    for t in toks:
        if t not in vocab:
            raise KeyError(
                f"token {t!r} 不在词表内。扩大 build_vocab 的 n_probe 或 "
                f"seeds，不要加 UNK —— 静默 UNK 会让值在读数位置不可分。")
        out.append(vocab[t])
    return out


class NLStream(IterableDataset):
    """流式训练集。每篇现生成，无重复文档（generator.py 第 1 条不变量）。

    worker 各自 seed：num_workers>1 时若共用 seed，每个 worker 产出同一串
    文档，等效 batch 里有 num_workers 份重复。

    偏移量是 1000*seed + w，与主网格的 Stream 相同（train.py:241 的
    docstring 与 :255 的代码都是这个式子）。先前这里写的是 seed + 1000 + w，
    那个式子在跨 seed 比较时会串流：seed=0 的 worker 3 与 seed=3 的 worker 0
    都得到 1003，于是十个 run 之间共享文档流，跨 seed σ 被压低。而 σ 正是
    这个臂唯一要产出的量。

    1000 倍数下不串：worker seed 落在 [1000s, 1000s+3]，十个 seed 的区间
    两两不交；轮次递进 +7919 也不与任何 (s,w) 组合重合（7919 与 1000 互素，
    且 |1000Δs + Δw| <= 9003 < 2*7919 时只有 Δk∈{-1,0,1} 可能，逐一代入
    无解）。
    """

    def __init__(self, r: int, d: int, seed: int, vocab: Dict[str, int],
                 spec: LangSpec, n_lo: int = 45, n_hi: int = 55):
        self.r, self.d, self.seed = r, d, seed
        self.vocab, self.spec = vocab, spec
        self.n_lo, self.n_hi = n_lo, n_hi

    def __iter__(self):
        wi = get_worker_info()
        w = wi.id if wi else 0
        v = Vocab(self.spec)
        cfg = nl_corpus_cfg(self.r, self.d, 1000 * self.seed + w,
                            self.n_lo, self.n_hi)
        while True:
            # generate_corpus 是有限生成器，耗尽后重开；seed 每轮递进，
            # 故不会重复上一轮的文档。
            for doc in generate_corpus(v, cfg, 4096):
                rd = render(doc, v, tokenize)
                yield encode(rd["tokens"], self.vocab), rd["answer_pos"]
            # dataclasses.replace 而非 CorpusCfg(**cfg.__dict__)：后者在
            # 字段有 default_factory 或 __post_init__ 时行为不同，且绕过了
            # dataclass 的字段校验。轮次递进 +7919 不与任何 (seed, worker)
            # 组合相撞，推理见 __init__ 的 docstring。
            cfg = dataclasses.replace(cfg, seed=cfg.seed + 7919)


def gate_docs(r: int, d: int, n: int, seed: int = 7,
              spec: Optional[LangSpec] = None) -> List[dict]:
    """门与碰撞表用的文档。格式与已废弃的 nl_generator.make_docs 相同。

    nl_gates 与 nl_collide 必须量**训练实际使用的**语料。它们先前 import
    nl_generator.make_docs，而那个平行生成器已被渲染层取代 —— 那些函数还在
    文件里，所以 import 不会报错，两个脚本会静默地量一个不再被训练的语料。
    posCeil 由此进 sweep.classify 决定哪些 run 进网格，量错了整条链都错。
    这个适配器是唯一入口。
    """
    spec = spec or nl_spec()
    v = Vocab(spec)
    cfg = nl_corpus_cfg(r, d, seed)
    return [render(doc, v, tokenize) for doc in generate_corpus(v, cfg, n)]


def gate_pairs(r: int, d: int, n: int, seed: int = 7,
               spec: Optional[LangSpec] = None) -> List[Tuple[str, str]]:
    """G1 用的编辑对。走 probe.apply_edit，与 build_probe 同一条路径。

    返回的对数会少于 n：apply_edit 对域外文档返回 None（probe.py:583）。
    G1 只需要"每一对两侧 token 数相等"，样本量少一些不影响判据。
    """
    import random as _r

    from probe import apply_edit, fit_position_offset

    spec = spec or nl_spec()
    v = Vocab(spec)
    cfg = nl_corpus_cfg(r, d, seed)
    docs = list(generate_corpus(v, cfg, n))
    offset = fit_position_offset(docs)
    rng = _r.Random(seed)
    out = []
    for doc in docs:
        ed = apply_edit(doc, "break_rarity", v, cfg, rng, offset)
        if ed is None:
            continue
        rb, re = render_edit_pair(doc, ed, v, tokenize)
        out.append((rb["text"], re["text"]))
    return out


def adj_slots(rd: dict) -> Tuple[List[int], List[int]]:
    """(adj token 的位置, 该位置是否为重复出现)。NL 臂的 _val_slots 等价物。

    train.py:312 用 _val_slots(d, cfg) 拿这两个量，copy_acc 是"重复出现的值
    token 上的 argmax 准确率"（train.py:321-329，argmax 限制在 value 块内）。
    NL 表层下 adj 位置不可由 ΔD 算出 —— 句长可变 —— 所以必须逐篇记录。

    比对的是**字符串**：render 返回的 tokens 是字符串列表，编码到 id 发生在
    调用方（encode）。先前这个函数的参数叫 adj_ids 且 docstring 说是 vocab
    id，那是错的 —— 拿 id 的集合去比字符串会一个都匹配不上，pos 恒为空，
    copy_acc 恒为 nan，而 sweep.classify 对 nan 返回 "?"，整格静默消失。

    "重复"的判据与主网格一致：该 adj 在本篇更早的位置出现过。答案位置本身
    排除在外，它是被预测的目标而不是诊断项。
    """
    from nl_generator import ADJ

    pool = set(ADJ)
    ap = rd["answer_pos"]
    seen, pos, rep = set(), [], []
    for i, t in enumerate(rd["tokens"]):
        if i == ap or t not in pool:
            continue
        pos.append(i)
        rep.append(int(t in seen))
        seen.add(t)
    return pos, rep


def build_probe(r: int, d: int, n: int, seed: int, vocab: Dict[str, int],
                spec: LangSpec) -> dict:
    """离线探针集 + break_rarity 的编辑对。

    编辑用 probe.apply_edit 而不是自己反转多重性：那个函数统一强制三条判定
    条件（目标规则在两侧预测不同、v* 的取法、v* 两侧都不等于真值，见
    probe.py:13-16），自己实现会得到一个与主网格不同口径的 Δ。
    """
    import random as _random

    from probe import apply_edit, fit_position_offset

    v = Vocab(spec)
    cfg = nl_corpus_cfg(r, d, seed)

    # offset 是 position 规则的拟合偏移，apply_edit 的第六个参数。它由一批
    # base 文档拟合，主网格在 probe_selfcheck / causal 里都先算它。
    warm = list(generate_corpus(v, cfg, 600, seed_offset=1))
    offset = fit_position_offset(warm)
    rng = _random.Random(seed)

    ids, apos, ans, vst, bl = [], [], [], [], []
    eids, eapos = [], []
    cpos, crep = [], []
    n_skip, mx = 0, 0
    for doc in generate_corpus(v, cfg, n * 8, seed_offset=1):
        # apply_edit 内部调 emit，故 edoc.tokens 是主网格的 4-token 形式。
        # 我们只要它的 stmts 与 v_star —— token 由 nl_render 重新生成。
        # 合法性检查（probe.py:597-603：rb!=re、真值不变、v* 不等于真值）
        # 全部基于 stmts，与表层无关，所以那些保证照样继承。
        edoc = apply_edit(doc, "break_rarity", v, cfg, rng, offset)
        if edoc is None:
            n_skip += 1
            continue
        rb, re = render_edit_pair(doc, edoc, v, tokenize)
        bi, ei = encode(rb["tokens"], vocab), encode(re["tokens"], vocab)
        mx = max(mx, len(bi), len(ei))
        if len(bi) > spec.ctx_len or len(ei) > spec.ctx_len:
            raise ValueError(
                f"R{r}_D{d} 探针文档 {max(len(bi), len(ei))} token > ctx_len "
                f"{spec.ctx_len}。增大 nl_spec 的 ctx_len。")
        ids.append(bi)
        eids.append(ei)
        apos.append(rb["answer_pos"])
        eapos.append(re["answer_pos"])
        ans.append(vocab[rb["answer"]])
        # v_star 是 **raw 值索引**，不减 VAL0。依据：probe.py:704-718 的
        # causal() 把 r_last_value(d) 与 ed.v_star 一起传给 _margin，而
        # r_last_value 读的是 d.stmts[i].val（raw，见 probe.py:513）。
        # 先前这里减了 VAL0，会得到负数并让 ADJ_OF 越界或取错 adj。
        vst.append(vocab[ADJ_OF(edoc.v_star)])
        bl.append(len(bi))
        # copy_acc 的诊断位置。base 侧即可 —— 它量的是"回路是否建成"，
        # 与编辑无关（train.py:328-329 在 eval 集上算，不在探针对上算）。
        sp, sr = adj_slots(rb)
        cpos.append(sp)
        crep.append(sr)
        if len(ids) >= n:
            break
    if len(ids) < n:
        raise ValueError(
            f"R{r}_D{d} 只凑到 {len(ids)}/{n} 篇（跳过 {n_skip}）。"
            f"break_rarity 的域为空或太小，见 probe.py:21-24。")
    return dict(base=ids, edit=eids, answer_pos=apos, edit_answer_pos=eapos,
                answer=ans, v_star=vst, base_len=bl, n_skip=n_skip, max_tok=mx,
                copy_pos=cpos, copy_rep=crep)


def ADJ_OF(val_id: int) -> str:
    """raw value id -> 它的 adj token。读数在 adj 位置取，而 adj 对 val_id
    单射（nl_render.check_pools 强制 n_values <= len(ADJ)），所以这是双射的
    逆向 —— "读数取 adj" 等价于 "读数取值"。"""
    from nl_generator import ADJ
    return ADJ[val_id]


def pad_to_array(seqs: Sequence[Sequence[int]], ctx_len: int,
                 vocab_size: int) -> np.ndarray:
    if vocab_size > 32767:
        raise ValueError(f"词表 {vocab_size} 超出 int16 范围")
    a = np.full((len(seqs), ctx_len), PAD, dtype=np.int16)
    for i, s in enumerate(seqs):
        a[i, :len(s)] = s
    return a


def pad_ragged(seqs: Sequence[Sequence[int]], fill: int) -> np.ndarray:
    """变长序列右填充到最长者。fill 必须是非法值域外的哨兵。

    copy_pos 用 -1 而不是 0：0 是合法的 token 位置索引，用它做哨兵会让
    每篇文档的第 0 个 token 被当成一个 adj 诊断位，copy_acc 掺进一批
    位置 0 上的预测。
    """
    w = max((len(s) for s in seqs), default=0)
    a = np.full((len(seqs), w), fill, dtype=np.int32)
    for i, s in enumerate(seqs):
        a[i, :len(s)] = s
    return a


def _ADJ_LIST() -> List[str]:
    """ADJ 的顺序快照。adj_ids 存进 npz 是为了让 nl_train 不必从 vocab 反推，
    也避免"生成时的 ADJ 顺序"与"训练时的 ADJ 顺序"不一致 —— 后者会让
    argmax 的候选集错位，读数静默失真。"""
    from nl_generator import ADJ
    return list(ADJ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[3])
    ap.add_argument("--cols", type=int, nargs="+", default=[8])
    ap.add_argument("--probe-docs", type=int, default=200)
    # 600 与主网格相同。CLI 默认必须跟着 nl_spec 的默认走，否则改了 nl_spec
    # 而这里没改，命令行会静默覆盖回旧值。
    ap.add_argument("--ctx-len", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="nl_data")
    ap.add_argument("--vocab-only", action="store_true")
    a = ap.parse_args()

    spec = nl_spec(a.ctx_len)
    check_pools(spec)
    cells = [(r, d) for r in a.rows for d in a.cols]

    print(f"建词表：{len(cells)} 格 × 2 seed × 400 篇 ...")
    vocab = build_vocab(cells, spec)
    print(f"词表大小 {len(vocab)}（含 PAD）")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "vocab.json"), "w") as f:
        json.dump(dict(version=CORPUS_VERSION, spec=spec.__dict__,
                       vocab=vocab), f, indent=1, ensure_ascii=False)
    print(f"已写入 {a.out}/vocab.json（version={CORPUS_VERSION}）")
    if a.vocab_only:
        return

    # 探针集与 --seed 无关：一份，全部训练 seed 共用。
    #
    # 两个理由。第一是正确性：训练流的 worker seed 是 1000*seed + w，故
    # seed=0 的 worker 覆盖 {0,1,2,3}，而探针集若也用 corpus seed 0 就与
    # worker 0 的前若干篇逐字节相同 —— 探针集不再是留出集。这是我把 worker
    # 偏移从 seed+1000+w 改成 1000*seed+w 时引入的，旧式恰好躲开了。
    #
    # 第二是口径：App.~cross 量过，固定模型换整批探针文档只移动 sign
    # fraction 至多 0.033，而跨 seed 项约 0.9；且"两个 run 共用一批文档"能
    # 复现该格的 seed 极差。共用一份因此不损失什么，反而把 batch 项从跨
    # seed 极差里彻底去掉 —— 十个 run 的差异全部来自模型。
    #
    # PROBE_SEED 取 777000：与任何 1000*seed + w（seed<=9 时 <= 9003）以及
    # 轮次递进 +7919 的可达集合不交。
    PROBE_SEED = 777000
    for r, d in cells:
        pr = build_probe(r, d, a.probe_docs, PROBE_SEED, vocab, spec)
        p = os.path.join(a.out, f"probe_R{r}_D{d}.npz")
        np.savez_compressed(
            p, version=CORPUS_VERSION,
            base=pad_to_array(pr["base"], a.ctx_len, len(vocab)),
            edit=pad_to_array(pr["edit"], a.ctx_len, len(vocab)),
            answer_pos=np.array(pr["answer_pos"], dtype=np.int32),
            edit_answer_pos=np.array(pr["edit_answer_pos"], dtype=np.int32),
            answer=np.array(pr["answer"], dtype=np.int32),
            v_star=np.array(pr["v_star"], dtype=np.int32),
            base_len=np.array(pr["base_len"], dtype=np.int32),
            # copy_acc 的诊断位置。变长，右填充 -1 —— 0 是合法位置索引，
            # 用它做哨兵会让第 0 个 token 被当成 adj 位置。
            copy_pos=pad_ragged(pr["copy_pos"], -1),
            copy_rep=pad_ragged(pr["copy_rep"], -1),
            # argmax 限制在这 100 个 id 上（train.py:318 的 lo:lo+n_val 在
            # NL 词表里不是连续区间，必须 gather）。存下来让 nl_train 不必
            # 重新从 vocab 反推，也避免两处 ADJ 顺序不一致。
            # 只取前 n_values 个。ADJ 有约 170 个（下界由 _ValueDraw 的耗尽
            # 定，见 values_needed），但语料里只出现 val_id < n_values 的那些
            # —— 第 150 个之后的 adj 从未被渲染，故不在词表里，vocab[w] 会
            # KeyError（实测 'brisk'）。argmax 的候选集必须恰是语料的值域。
            adj_ids=np.array([vocab[w] for w in _ADJ_LIST()[:spec.n_values]],
                             dtype=np.int32))
        print(f"R{r:>2}_D{d:<2}  probe {len(pr['base'])}  "
              f"skip {pr['n_skip']}  maxTok {pr['max_tok']}  -> {p}")

    print(f"\n训练集是流式的（NLStream），不落盘。version={CORPUS_VERSION}")
    print("改生成或渲染必须 +1 —— 每个 run 的 jsonl 记这个串，才能事后知道")
    print("它训练在哪一版语料上。主网格靠 corpus.name+seed，NL 臂靠这个。")


if __name__ == "__main__":
    main()
