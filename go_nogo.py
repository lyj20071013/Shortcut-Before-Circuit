"""Evaluate terminal checkpoints and save aggregate and per-document probes."""
import argparse
import fnmatch
import json
import math
import os
import random
import re
import time
from typing import List, Optional, Sequence, Tuple

import torch

from argmax_transitions import (COUNT_FIELD, ID_NAMESPACE,
                                ID_NAMESPACE_FIELD, MASS_COUNT_FIELD, SCHEMA,
                                aggregate, classify_argmax, counts_from_records,
                                format_latex, format_text, json_ready,
                                transition_key, validate_perdoc, validate_row)
from config import CorpusCfg, LangSpec
from generator import Doc, emit, generate_corpus
from model import LM, ModelCfg
from probe import (EditedDoc, _answer_val, _edit_break_rarity_ctrl, _margin,
                   _order_ok, _p_final, _q_pos, apply_edit,
                   fit_position_offset, r_last_value, swap_query)
from vocab import Vocab

MASS_FLOOR = 0.50
COPY_FLOOR = 0.95
ACC_FLOOR = 0.99
NAN = float("nan")


# ---------------- 统计量 ----------------

def percentile(srt: List[float], p: float) -> float:
    """线性插值分位数。原实现 srt[int(p*n)] 对 n=186 取 srt[93]，不是中位数
    （应为 srt[92] 与 srt[93] 的均值）；分位数是主 DV，这个偏差不能留。"""
    if not srt:
        return NAN
    if len(srt) == 1:
        return srt[0]
    k = p * (len(srt) - 1)
    lo = int(k)
    hi = min(lo + 1, len(srt) - 1)
    return srt[lo] + (k - lo) * (srt[hi] - srt[lo])


def trimmed_mean(xs: List[float], frac: float = 0.05) -> float:
    """两端各截 frac。重尾下比均值稳，比中位数多用信息；附录一并报告。"""
    if not xs:
        return NAN
    srt = sorted(xs)
    k = int(len(srt) * frac)
    core = srt[k:len(srt) - k] if len(srt) - 2 * k > 0 else srt
    return sum(core) / len(core)


def sign_test_p(k: int, n: int) -> float:
    """精确二项检验，H0: P(Δ>0)=0.5，双侧。frac+ 是主 DV，必须带 p 值 ——
    n=186 时 frac+=0.55 与 0.45 都不显著，肉眼看表容易当成方向。
    用整数除法避免 2.0**n 在大 n 上溢出。"""
    if n == 0:
        return NAN
    tail = min(k, n - k)
    num = sum(math.comb(n, i) for i in range(tail + 1))
    return min(1.0, 2.0 * num / (2 ** n))


# ---------------- 模型侧 ----------------

class Predictor:
    """按前缀缓存 log-softmax。与 train.ModelPredictor 同接口。"""

    def __init__(self, model: LM, vocab: Vocab, spec: LangSpec, device):
        self.m, self.vocab, self.dev = model, vocab, device
        self.lo = vocab.val(0)
        self.n_val = spec.n_values
        self._c = {}

    @torch.no_grad()
    def _lp(self, view) -> torch.Tensor:
        key = tuple(view.tokens[:view.answer_pos])
        got = self._c.get(key)
        if got is None:
            ids = torch.tensor([key], device=self.dev)
            logits, _ = self.m(ids)
            got = torch.log_softmax(
                logits[0, -1, self.lo:self.lo + self.n_val].float(), -1).cpu()
            if len(self._c) > 8192:
                self._c.clear()
            self._c[key] = got
        return got

    def predict(self, view) -> int:
        return int(self._lp(view).argmax())

    def logp(self, view, cands: Sequence[int]) -> List[float]:
        lp = self._lp(view)
        return [float(lp[v]) for v in cands]

    def mass(self, view, cands: Sequence[int]) -> float:
        lp = self._lp(view)
        return float(sum(lp[v].exp() for v in cands))


def load(tag: str, out_dir: str, device):
    """spec/corpus 从 jsonl 的 meta 取（checkpoint 里没存），权重从 .pt 取。"""
    jl = os.path.join(out_dir, f"{tag}.jsonl")
    pt = os.path.join(out_dir, f"{tag}.pt")
    with open(jl) as f:
        meta = json.loads(f.readline())
    assert meta["kind"] == "meta", f"{jl} 首行不是 meta"
    ck = torch.load(pt, map_location=device)
    spec = LangSpec(**meta["spec"])
    corpus = CorpusCfg(**meta["corpus"])
    model = LM(ModelCfg(**ck["model_cfg"])).to(device)
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    model.eval()
    conv = dict(ck.get("eval", {}))
    esc = NAN
    if "copy_acc" not in conv or esc != esc:
        # checkpoint 的 eval 只存 acc/acc_tail0，而 classify 需要 copy_acc
        # 区分 retrieval 与 position 态；逃逸步也只能从 jsonl 的 eval 序列拿。
        with open(jl) as f:
            for line in f:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("kind") != "eval":
                    continue
                if esc != esc and o.get("copy_acc", 0.0) >= COPY_FLOOR:
                    esc = o["step"]
                conv = o
    return model, spec, corpus, meta, conv, esc


def ctrl_view(d: Doc, vocab: Vocab, cfg: CorpusCfg, rng) -> Optional[EditedDoc]:
    """非 q slot 的多重性反转。q slot 未动故目标规则预测不变，apply_edit 的
    守卫会拒绝 —— 这里绕过它自行构造。"""
    out = _edit_break_rarity_ctrl(d, cfg, vocab.spec, rng)
    if out is None:
        return None
    stmts, delta = out
    if not _order_ok(stmts):
        return None
    av = _answer_val(d)
    toks, apos = emit(stmts, d.q_ent, d.q_attr, d.q_hist_k, av, vocab, cfg)
    if len(toks) != len(d.tokens) or apos != d.answer_pos:
        return None
    return EditedDoc("ctrl", toks, apos, vocab.val(av), stmts, d.q_ent,
                     d.q_attr, d.q_hist_k, delta, len(stmts), d, "rarity",
                     +1, -1)


def axis_of(tag: str, pat: str, fallback: int) -> int:
    """从 tag 解析轴值。corpus.delta_d_lo 是 dd_band 下界（轴值 5→2、16→8），
    直接打印会让 ΔD=2 与 ΔD=3 都显示成 2，75 行的表无法肉眼比对。"""
    m = re.search(pat, tag)
    return int(m.group(1)) if m else fallback


def classify(acc, copy_acc, corpus) -> str:
    """retr / posNN% / none。Δ 只在 retr 态有意义：position 态没有检索回路
    （copy_acc≈0），其 break_rarity 响应是位置规则的副产物，方向恒为负，
    混进相图会在低 R_old 角伪造出 frequency 型区域。
    posCeil = 1/(dd_hi-dd_lo+1) 是纯位置规则的解析上限（unmarked 时每条语句
    恰 4 token、答案恒在倒数第 1+ΔD 条）。实测吻合到 98%。"""
    if acc != acc or copy_acc != copy_acc:
        return "?"
    ceil = 1.0 / (corpus.delta_d_hi - corpus.delta_d_lo + 1)
    if copy_acc >= COPY_FLOOR and acc >= ACC_FLOOR:
        return "retr"
    if copy_acc < 0.5 and acc >= 0.5 * ceil:
        return f"pos{acc / ceil:.0%}"
    return "none"


# ---------------- 单格 ----------------

def run_one(tag: str, out_dir: str, n_docs: int, device) -> Tuple[dict, dict]:
    model, spec, corpus, meta, conv, esc = load(tag, out_dir, device)
    vocab = Vocab(spec)
    pred = Predictor(model, vocab, spec, device)
    all_docs = list(generate_corpus(vocab, corpus, n_docs, seed_offset=1))
    # 旋钮 6：读数只在非反转文档上做。三个理由，都不能省：
    # 1. break_rarity 在反转文档上没有域（probe 的守卫直接返回 None），留着
    #    它们只会把 yield_rate 的分母稀释成 1−p_break，看起来像域缩小了。
    # 2. 反转文档的 q slot 已是 [老值×k, 末代值×rb]，再反转一次没有定义。
    # 3. 探针输入分布要与主网格同型，arm 与主网格的极差才可比。
    # 代价是 arm 的有效样本比名义 --docs 少 p_break 比例，故报告剔除数。
    docs = [d for d in all_docs if not getattr(d, "is_break", False)]
    n_excl = len(all_docs) - len(docs)
    offset = fit_position_offset(docs)
    rng = random.Random(0)

    # Each record retains enough information to recompute the 3x3 transition
    # table without a checkpoint. ``predict`` reuses tensors cached by the two
    # margin calls, so recording argmax values adds no model forward passes.
    recs = []
    for doc_index, d in enumerate(docs):
        ed = apply_edit(d, "break_rarity", vocab, corpus, rng, offset)
        if ed is None:
            continue
        new_value = _answer_val(d)   # preserved current/ground-truth value
        if new_value != r_last_value(d):
            raise ValueError(f"{tag}: edit-domain truth is not the current value")
        old_value = ed.v_star         # superseded contrast value
        delta = (_margin(pred, ed, old_value, new_value)
                 - _margin(pred, d, old_value, new_value))
        mass = pred.mass(ed, [old_value, new_value])
        base_argmax, edit_argmax = pred.predict(d), pred.predict(ed)
        base_class = classify_argmax(base_argmax, new_value, old_value)
        edit_class = classify_argmax(edit_argmax, new_value, old_value)
        trans = transition_key(base_argmax, edit_argmax, new_value, old_value)
        recs.append(dict(doc_index=doc_index, delta=delta, mass=mass,
                         q_kept=d.q_kept, transition=trans,
                         base_class=base_class, edit_class=edit_class,
                         base_argmax=base_argmax, edit_argmax=edit_argmax,
                         old_value=old_value, new_value=new_value))

    # §8 第四条的直接测量，在主网格上也良定义。
    #
    # 那条限制说：把 R_old 个副本铺在固定窗口里，会让被查询 slot 的副本比填充
    # slot 更分散（App:gen 第五条，超出填充均值 2–22%），故模型原则上能不读
    # query 就定位它。现在的措辞是 "we cannot rule out a contribution" ——
    # 而 copy 诊断达到 1.000 只说明「存在真的 slot 匹配」，不排除「另有一条
    # 结构旁路」。arm 上有直接反驳（swKeep ≤ 0.007），但那只在反转文档上测过，
    # 转移到主网格靠论证不靠测量。
    #
    # 这个测量把 query 换成同篇另一个带 update 的 slot：语句序列一字不动、
    # token 数与 answer_pos 不变，故不是分布外输入，只是问了同一篇文档的另一个
    # 问题。生成器让值在文档内不重复抽取，故换 slot 必然换答案 —— 读 query 的
    # 模型必须改预测，靠结构定位的不会。它不依赖 is_break，反转文档只是结构
    # 差异更大的地方，不是测量成立的前提。
    sw_n = sw_keep = 0
    for d in docs:
        sv = swap_query(d, vocab, corpus)
        if sv is None:
            continue
        sw_n += 1
        sw_keep += int(pred.predict(sv) == pred.predict(d))

    cs = []
    rng2 = random.Random(1)
    for d in docs:
        cv = ctrl_view(d, vocab, corpus, rng2)
        if cv is None:
            continue
        truth = r_last_value(d)
        cand = [i for i in _q_pos(d) if i != _p_final(d)]
        if not cand:
            continue
        v_old = d.stmts[cand[-1]].val
        if v_old == truth:
            continue
        cs.append(_margin(pred, cv, v_old, truth)
                  - _margin(pred, d, v_old, truth))

    ds = [rec["delta"] for rec in recs]
    ms = [rec["mass"] for rec in recs]
    valid = [rec["delta"] for rec in recs if rec["mass"] >= MASS_FLOOR]
    n, nv = len(recs), len(valid)

    # Recompute both tables from raw ids; this also checks category labels,
    # duplicate document indices, and finite probability masses.
    transition_counts, transition_counts_mass = counts_from_records(
        recs, MASS_FLOOR)

    # Within-cell stratification by realized redundancy.
    ks = sorted(rec["q_kept"] for rec in recs)
    k_med = ks[len(ks) // 2] if ks else 0
    hi_k = sorted(rec["delta"] for rec in recs if rec["q_kept"] >= k_med)
    lo_k = sorted(rec["delta"] for rec in recs if rec["q_kept"] < k_med)
    if not lo_k or not hi_k:
        k_med = ks[len(ks) // 4] if ks else 0
        hi_k = sorted(rec["delta"] for rec in recs if rec["q_kept"] > k_med)
        lo_k = sorted(rec["delta"] for rec in recs if rec["q_kept"] <= k_med)

    srt, srt_v = sorted(ds), sorted(valid)
    k_pos = sum(1 for x in ds if x > 0)
    # 门后符号比例。主 DV 此前在全部有域文档上算，含 mass<0.5 的那些 —— 而
    # App:oracle 判定它们读数无效。中位数已经门了（d_median_valid），符号比例
    # 没有，两者口径不一致。mOK 最低的格是 R12_D5（0.66），那里三分之一文档
    # 个体无效却仍在投票。同时报门后版本，让 §sec:degrade 的
    # "unaffected by the gate by construction" 有一个可对照的数。
    k_pos_valid = sum(1 for x in valid if x > 0)

    # 旋钮 6：retrieval 门必须用非反转子集的准确率。arm 的 eval 集混有
    # p_break 比例的反转文档，混合 acc 会被压到 1−p_break 以下（p_break=0.10
    # 时上限 0.90），classify 判成 none，整个 run 被静默排除 —— 尽管它在非
    # 反转文档上的读数完全有效。acc_nb 由 train.evaluate 分层给出；旧数据
    # 没有该字段，回退到混合 acc（那些 run 本来就没有反转文档，两者相等）。
    acc_nb = conv.get("acc_nb", NAN)
    acc_mix = conv.get("acc", NAN)
    acc = acc_nb if acc_nb == acc_nb else acc_mix
    ca = conv.get("copy_acc", NAN)
    row = dict(
        tag=tag, seed=axis_of(tag, r"_s(\d+)_", 0),
        argmax_transition_schema=SCHEMA,
        argmax_id_namespace=ID_NAMESPACE,
        argmax_transition_mass_floor=MASS_FLOOR,
        argmax_transition_counts=transition_counts,
        argmax_transition_counts_mass=transition_counts_mass,
        n_docs_requested=n_docs,
        acc_mix=acc_mix, acc_brk=conv.get("acc_brk", NAN),
        p_rec=conv.get("p_rec", NAN), p_rar=conv.get("p_rar", NAN),
        rar_disc=conv.get("rar_disc", NAN),
        n_break_doc=conv.get("n_brk", 0),
        # 旁路检查（train.break_diag 的 _swap_query）。sw_keep 是「query 换成
        # 同篇另一个 slot 后预测完全不变」的比例。_ValueDraw 保证一篇内值 id
        # 不重复，故读 query 的模型换 slot 后必答另一个值 → sw_keep≈0。
        # sw_keep 高即模型在反转文档上没读 query，走的是结构旁路。
        sw_n=conv.get("sw_n", 0), sw_keep=conv.get("sw_keep", NAN),
        sw_rec=conv.get("sw_rec", NAN), sw_rar=conv.get("sw_rar", NAN),
        r_old=corpus.r_old_lo, dd=axis_of(tag, r"_D(\d+)_", corpus.delta_d_lo),
        dd_lo=corpus.delta_d_lo, dd_hi=corpus.delta_d_hi,
        # 旋钮 6 的两个轴。从 jsonl 的 corpus meta 取（不是从 tag 解析）：
        # tag 里的档位串是人写的，可能与实际配置不符。nb_dose.py 按这两个字段
        # 分组，已发表的主网格 run 反序列化后 p_break=0.0，正好是剂量曲线的
        # 0 点锚点，无需特殊处理。
        p_break=getattr(corpus, "p_break", 0.0),
        truth_rule=getattr(corpus, "truth_rule", "recency"),
        n=n, yield_rate=n / len(docs) if docs else NAN,
        n_docs_used=len(docs), n_docs_excl=n_excl,
        # 换 query 后预测不变的比例。读 query 的模型必须改预测，故这个数越小
        # 越好。字段名带 _grid 后缀以区别于 break_diag 写进 jsonl 的 sw_keep
        # （后者只在反转文档上算，且只有 arm 的 checkpoint 有）。
        sw_n_grid=sw_n, sw_keep_grid=(sw_keep / sw_n) if sw_n else NAN,
        d_margin=(sum(ds) / n) if n else NAN,
        d_median=percentile(srt, 0.50),
        d_q25=percentile(srt, 0.25), d_q75=percentile(srt, 0.75),
        d_trim05=trimmed_mean(ds, 0.05),
        frac_positive=(k_pos / n) if n else NAN,
        sign_p=sign_test_p(k_pos, n),
        frac_positive_valid=(k_pos_valid / nv) if nv else NAN,
        sign_p_valid=sign_test_p(k_pos_valid, nv),
        mass=(sum(ms) / n) if n else NAN,
        mass_min=min(ms) if ms else NAN,
        frac_mass_ok=(nv / n) if n else NAN,
        d_median_valid=percentile(srt_v, 0.50),
        n_valid=nv,
        ctrl_n=len(cs),
        ctrl_margin=(sum(cs) / len(cs)) if cs else NAN,
        ctrl_median=percentile(sorted(cs), 0.50),
        acc=acc, tail0=conv.get("acc_tail0", NAN), copy_acc=ca,
        escape_step=esc, step=conv.get("step", NAN),
        total_steps=meta.get("train", {}).get("total_steps", NAN),
        # schedule 必须进行，否则 nb_dose 会把同 p_break 不同 schedule 的 run
        # 静默池化。实测踩过：p_break=0.03 那一档有两个 cos run 未逃逸、改
        # const 后逃逸，于是该档变成 const/cos/const 混合，而 §6.1 量过
        # schedule 效应的极差是 0.55 —— 比 arm 里任何一档的极差都大，混进去
        # 算出的极差没有意义。
        sched=meta.get("train", {}).get("sched", "cos"),
        state=classify(acc, ca, corpus),
        k_split=k_med,
        d_med_hiK=percentile(hi_k, 0.50), n_hiK=len(hi_k),
        d_med_loK=percentile(lo_k, 0.50), n_loK=len(lo_k),
    )
    perdoc = dict(
        tag=tag,
        argmax_transition_schema=SCHEMA,
        argmax_transition_mass_floor=MASS_FLOOR,
        n_docs_requested=n_docs,
        n=n,
        n_valid=nv,
        # Legacy arrays remain for figs.py and existing downstream notebooks.
        d_all=[round(rec["delta"], 4) for rec in recs],
        mass_all=[round(rec["mass"], 6) for rec in recs],
        # Raw ids make the transition table independently reproducible without
        # loading a checkpoint. Full-precision mass preserves the gate exactly.
        records=recs,
    )
    row[ID_NAMESPACE_FIELD] = ID_NAMESPACE
    perdoc[ID_NAMESPACE_FIELD] = ID_NAMESPACE
    validate_row(row)
    validate_perdoc(row, perdoc, MASS_FLOOR)
    return row, perdoc


# ---------------- 批量 ----------------

def discover(out_dir: str, pattern: str) -> List[str]:
    tags = [f[:-3] for f in os.listdir(out_dir) if f.endswith(".pt")]
    tags = [t for t in tags if fnmatch.fnmatch(t, pattern)]
    return sorted(tags)


def report(rows: List[dict]) -> str:
    rows = sorted(rows, key=lambda r: (r["r_old"], r["dd"], r["seed"]))
    lines = [
        f"{'tag':<22} {'R':>3} {'ΔD':>3} {'band':>7} {'state':>7} {'esc':>6} "
        f"{'acc':>6} {'copy':>6} {'yld':>5} {'n':>4} {'med':>8} "
        f"{'IQR':>16} {'frac+':>6} {'p':>8} {'mean':>8} {'trim':>8} "
        f"{'mass':>6} {'mOK':>5} {'ctrlMed':>8} {'ctrlAvg':>8} {'kSpl':>5} "
f"{'medHi':>7} {'medLo':>7}"]
    for r in rows:
        band = f"[{r['dd_lo']},{r['dd_hi']}]" 
        iqr = f"[{r['d_q25']:+.2f},{r['d_q75']:+.2f}]"
        lines.append(
            f"{r['tag']:<22} {r['r_old']:>3} {r['dd']:>3} {band:>7} "
            f"{r['state']:>7} {r['escape_step']:>6.0f} {r['acc']:>6.3f} "
            f"{r['copy_acc']:>6.3f} {r['yield_rate']:>5.2f} {r['n']:>4} "
            f"{r['d_median']:>+8.3f} {iqr:>16} {r['frac_positive']:>6.2f} "
            f"{r['sign_p']:>8.1e} {r['d_margin']:>+8.3f} "
            f"{r['d_trim05']:>+8.3f} {r['mass']:>6.2f} "
            f"{r['frac_mass_ok']:>5.2f} {r['ctrl_median']:>+8.3f} "
            f"{r['ctrl_margin']:>+8.3f} "
            f"{r.get('k_split', 0):>5.1f} "
            f"{r.get('d_med_hiK', NAN):>+7.3f} "
            f"{r.get('d_med_loK', NAN):>+7.3f}")

    # Legacy terminal console summary: aggregate per-run medians and
    # report cross-seed ranges among runs passing the filters below.
    # The final paper figures use the sign-fraction analysis in figs2.py.
    cells = {}
    for r in rows:
        if r["state"] == "retr" and r["mass"] >= MASS_FLOOR:
            cells.setdefault((r["r_old"], r["dd"]), []).append(r)
    if cells:
        rs = sorted({k[0] for k in cells})
        dds = sorted({k[1] for k in cells})
        for name, key in (("中位数 Δ", "d_median"), ("frac+", "frac_positive")):
            lines += ["", f"{name}（state=retr 且 mass≥{MASS_FLOOR}）"
                          f"  行=R_old 列=ΔD",
                      "      " + "".join(f"{d:>9}" for d in dds)]
            for rr in rs:
                cs = []
                for dd in dds:
                    v = cells.get((rr, dd))
                    if v:
                        xs = sorted(x[key] for x in v)
                        cs.append(f"{percentile(xs, 0.5):>+9.2f}")
                    else:
                        cs.append(f"{'—':>9}")
                lines.append(f"R{rr:>4} " + "".join(cs))
        lines += ["", "seed 极差（中位数 Δ）  格间差异须超过它才算有意义",
                  "      " + "".join(f"{d:>9}" for d in dds)]
        for rr in rs:
            cs = []
            for dd in dds:
                v = cells.get((rr, dd))
                if v and len(v) > 1:
                    xs = [x["d_median"] for x in v]
                    cs.append(f"{max(xs) - min(xs):>9.2f}")
                else:
                    cs.append(f"{'—':>9}" if not v else f"{'n=1':>9}")
            lines.append(f"R{rr:>4} " + "".join(cs))

    # 换 query 检查。对每个 run 都有定义（不限 arm），故单列一节。
    sw = [r for r in rows
          if r.get("sw_keep_grid", NAN) == r.get("sw_keep_grid", NAN)]
    if sw:
        xs = sorted(r["sw_keep_grid"] for r in sw)
        worst = max(sw, key=lambda r: r["sw_keep_grid"])
        lines += ["", "换 query 检查（§8 第四条的直接测量）",
                  f"  {len(sw)} run：swKeep 范围 {xs[0]:.4f}–{xs[-1]:.4f}，"
                  f"中位 {percentile(xs, 0.5):.4f}",
                  f"  最大 {worst['sw_keep_grid']:.4f} @ {worst['tag']}"
                  f"（n={worst.get('sw_n_grid', 0)}）",
                  "  把 query 换成同篇另一个带 update 的 slot，语句序列一字不动、",
                  "  token 数与 answer_pos 不变。值在文档内不重复抽取，故读 query 的",
                  "  模型必须改预测 → swKeep≈0。高则模型靠结构定位被查询 slot：",
                  "  被查询 slot 的副本比填充 slot 更分散（App:gen 第五条，超出",
                  "  填充均值 2–22%）。copy 诊断达 1.000 只说明存在真 slot 匹配，",
                  "  不排除另有一条结构旁路 —— 这一列直接排除它。"]

    # 旋钮 6 的独立小节。不加宽主表：它已 23 列，且 report() 也用于重跑已发表
    # 的 75 格，加宽会让 go_nogo.txt 与已发表的格式对不上。
    arm = [r for r in rows if r.get("p_rec", NAN) == r.get("p_rec", NAN)]
    if arm:
        lines += ["", "别名破除臂（旋钮 6）  行为判别子与因果读数并列",
                  f"{'tag':<22} {'accNb':>6} {'accBrk':>7} {'pRec':>6} "
                  f"{'pRar':>6} {'rarDsc':>7} {'swKeep':>7} {'swRar':>6} "
                  f"{'used':>5} {'excl':>5} {'med':>8} {'frac+':>6}"]
        for r in sorted(arm, key=lambda x: x["tag"]):
            lines.append(
                f"{r['tag']:<22} {r['acc']:>6.3f} {r['acc_brk']:>7.3f} "
                f"{r['p_rec']:>6.3f} {r['p_rar']:>6.3f} "
                f"{r['rar_disc']:>7.3f} {r.get('sw_keep', NAN):>7.3f} "
                f"{r.get('sw_rar', NAN):>6.3f} "
                f"{r.get('n_docs_used', 0):>5} "
                f"{r.get('n_docs_excl', 0):>5} {r['d_median']:>+8.3f} "
                f"{r['frac_positive']:>6.2f}")
        lines += ["",
                  "pRec/pRar：反转层上模型 argmax 落在末代值/老值的比例。这是与",
                  "因果探针独立的规则标签 —— 主网格没有这一层，行为无法识别规则。",
                  "N−（truth_rule=recency）应 pRec→1；N+（rarity）应 pRar→1。",
                  "两条 arm 报同一对数字，可直接比较。pRar 起不来即 arm 没训起来，",
                  "不是关于共延性的结果。",
                  "rarDsc：rarity 在观测归因上的 rate_disc，第三个独立仪器。",
                  "主网格上 rarity 的 n_disc 恒为 0（式 1），这里第一次有信号。",
                  "swKeep/swRar：旁路检查。把反转文档的 query 换成同篇另一个带",
                  "update 的 slot（语句序列一字不动，token 数与 answer_pos 不变），",
                  "swKeep 是预测完全不变的比例。一篇内值 id 不重复，故读 query 的",
                  "模型换 slot 后必答另一个值 → swKeep≈0。swKeep 高即模型在反转",
                  "文档上没读 query，而是靠结构认出被查询 slot：反转文档的 q slot",
                  "语句数恒为 R+1（重试循环强制 q_old 全存活），非反转文档只有约",
                  "0.6R+1，且 slot 内相邻间距也不同（covar 的 antRbk）。若走旁路，",
                  "目标函数对 rarity 的惩罚落在一条独立回路上，主回路仍欠定 ——",
                  "表现为 N− 极差不收缩，会被误读成「读数是噪声」。这是本臂唯一",
                  "无法从结构上消除、只能测量的混淆。",
                  "accNb 是非反转子集的准确率（retrieval 门用它，不用混合 acc：",
                  "混合值被 1−p_break 压住会让整个 run 被误判成未收敛）。",
                  "excl 是被剔除的反转文档数 —— 读数只在非反转文档上做，",
                  "使探针输入分布与主网格同型，arm 与主网格的极差才可比。",
                  "判据：因果读数（med/frac+）必须与行为标签（pRec/pRar）一致。",
                  "不一致 ⇒ 探针无效，全文结论作废。这是本臂最强的否证入口。"]

    # 门后符号比例，单列一节而不加宽主表（主表已 23 列，加宽会与已发表的
    # go_nogo.txt 格式对不上）。
    #
    # 为什么必须报这个。App:oracle 的 mass 门是逐篇的，判据是「mass<0.5 的
    # 文档上读数无效」。d_median_valid 服从它，frac_positive 不服从 —— 后者
    # 的分母是全部有域文档。于是 §sec:degrade 那句 "the sign fraction is
    # unaffected by the gate by construction" 字面为真但不是辩护：它之所以
    # 不受门影响，正是因为它没有门。mOK=0.66 的格上三分之一文档个体无效却
    # 仍在投票。两个版本并报，差值大的格就是主 DV 含无效读数的格。
    gsf = [r for r in rows
           if r.get("frac_positive_valid", NAN) == r.get("frac_positive_valid", NAN)]
    if gsf:
        lines += ["", "门后符号比例（逐篇 mass≥0.5 的子集上重算主 DV）",
                  f"{'tag':<22} {'n':>4} {'nValid':>7} {'mOK':>5} "
                  f"{'frac+':>6} {'frac+G':>7} {'Δ':>7} {'pG':>8}"]
        for r in sorted(gsf, key=lambda x: (x["r_old"], x["dd"], x["seed"])):
            lines.append(
                f"{r['tag']:<22} {r['n']:>4} {r['n_valid']:>7} "
                f"{r['frac_mass_ok']:>5.2f} {r['frac_positive']:>6.3f} "
                f"{r['frac_positive_valid']:>7.3f} "
                f"{r['frac_positive_valid'] - r['frac_positive']:>+7.3f} "
                f"{r['sign_p_valid']:>8.1e}")
        dv = sorted(abs(r["frac_positive_valid"] - r["frac_positive"])
                    for r in gsf)
        worst = max(gsf, key=lambda r: abs(r["frac_positive_valid"]
                                           - r["frac_positive"]))
        lines += [f"  |Δ| 最大 {dv[-1]:.3f} @ {worst['tag']}"
                  f"（mOK {worst['frac_mass_ok']:.2f}），"
                  f"中位 {percentile(dv, 0.5):.3f}",
                  "  两者差得大的格即 mOK 低处；主 DV 在那里含无效读数。",
                  "  §sec:degrade 与 App:stats 的措辞须与这一列一致。"]
        # 逐格极差也要门后版本：论文的头条数字 0.879 是未门的跨 seed 极差。
        cells_g = {}
        for r in gsf:
            if r["state"] == "retr":
                cells_g.setdefault((r["r_old"], r["dd"]), []).append(r)
        multi = {k: v for k, v in cells_g.items() if len(v) > 1}
        if multi:
            lines += ["", "  跨 seed 极差：未门 vs 门后（头条 0.879 是未门的）",
                      f"  {'cell':<10} {'nSeed':>5} {'rangeU':>7} {'rangeG':>7} "
                      f"{'Δ':>7}"]
            rows_g = []
            for (rr, dd), v in multi.items():
                u = [x["frac_positive"] for x in v]
                g = [x["frac_positive_valid"] for x in v]
                rows_g.append((max(u) - min(u), max(g) - min(g), rr, dd, len(v)))
            for ru, rg, rr, dd, ns in sorted(rows_g, reverse=True):
                lines.append(f"  R{rr:<3}D{dd:<5} {ns:>5} {ru:>7.3f} "
                             f"{rg:>7.3f} {rg - ru:>+7.3f}")
            lines.append(f"  未门 >0.3 的格 {sum(1 for t in rows_g if t[0] > 0.3)}，"
                         f"门后 >0.3 的格 {sum(1 for t in rows_g if t[1] > 0.3)}"
                         f"  <- tab:spread 的 13 与摘要的 13 都取未门")

    bad = [f"{r['tag']} state={r['state']}" for r in rows
           if r["state"] != "retr"]
    bad += [f"{r['tag']} mass={r['mass']:.2f}" for r in rows
            if r["mass"] == r["mass"] and r["mass"] < MASS_FLOOR]
    bad += [f"{r['tag']} frac+={r['frac_positive']:.2f} p={r['sign_p']:.2f}"
            for r in rows
            if r["sign_p"] == r["sign_p"] and r["sign_p"] > 0.05]
    # 阈值 0.05：arm 上实测 ≤0.007，主网格的结构线索（副本分散度）比 arm 弱
    # （arm 还多一条 q slot 语句数恒为 R+1），故预期同量级。超过 0.05 说明
    # 有相当比例的文档上模型没读 query，§8 第四条不能弱化。
    bad += [f"{r['tag']} swKeep={r['sw_keep_grid']:.3f} 换 query 后预测不变，"
            f"可能存在结构旁路"
            for r in rows
            if r.get("sw_keep_grid", NAN) == r.get("sw_keep_grid", NAN)
            and r["sw_keep_grid"] > 0.05]

    lines += [
        "",
        "med/IQR/frac+ 是主读数；mean 与 trim 供附录，均值在单 checkpoint 上",
        f"有 ±0.5 nats 的时间噪声（见文件头），格间差 <0.3 不可信。",
        "Δ>0 rarity 型 / Δ<0 frequency 型 / |Δ|≈0 纯 recency。",
        "state=retr 才可用：pos 态无检索回路、Δ 是位置规则副产物、方向恒为负。",
        "p 是符号的精确二项检验（双侧）；p>0.05 即方向不成立，不论均值多大。",
        "mOK 是逐篇 mass≥0.5 的比例；格均值达标但 mOK 低说明尾部文档已散掉。",
        "|ctrl| 应远小于 |med|。跨格有梯度才算方向成立，单格绝对值不构成结论。"
        "medHi/medLo 是格内按逐篇 Rreal 中位数分层的两半。两者同向且差异与",
        "跨格方向一致 → R_old 通过统计起作用；两者无差异 → 跨格效应来自",
        "edit domain 的文档选择偏置（R3/D16 的 domain 0.66 vs R16 的 1.00）。",]
    lines += ([""] + ["注意：" + b for b in bad]) if bad else ["", "无异常。"]
    return "\n".join(lines)



def _restore_nan(value):
    """Restore internal NaNs after strict JSON serialisation wrote them as null."""
    if value is None:
        return NAN
    if isinstance(value, list):
        return [_restore_nan(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_nan(item) for key, item in value.items()}
    return value


def read_tagged_jsonl(path: str) -> dict:
    """Read the latest complete object per tag, tolerating a truncated tail."""
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = _restore_nan(json.loads(line))
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("tag"):
                rows[row["tag"]] = row
    return rows


def write_tagged_jsonl(path: str, rows: dict) -> None:
    """Atomically write one strict-JSON row per tag."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for tag in sorted(rows):
            handle.write(json.dumps(json_ready(rows[tag]), ensure_ascii=False,
                                    allow_nan=False) + "\n")
    os.replace(tmp, path)


def cache_pair_valid(row, perdoc, n_docs: int) -> Tuple[bool, str]:
    if not row or not perdoc:
        return False, "missing summary or per-document row"
    if row.get("n_docs_requested") != n_docs:
        return False, "different --docs"
    if perdoc.get("n_docs_requested") != n_docs:
        return False, "different per-document --docs"
    if row.get("argmax_transition_mass_floor") != MASS_FLOOR:
        return False, "different mass floor"
    try:
        validate_perdoc(row, perdoc, MASS_FLOOR)
    except (KeyError, TypeError, ValueError) as exc:
        return False, str(exc)
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", help="run tag，如 R3_D5_s0_grid")
    ap.add_argument("--pattern", default=None,
                    help="从 --out 里按通配符发现 tag，如 'R*_grid'")
    ap.add_argument("--out", default="runs_g2")
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--txt", default=None, help="默认 <out>/go_nogo.txt")
    ap.add_argument("--force", action="store_true", help="重算已有结果")
    a = ap.parse_args()
    txt = a.txt or os.path.join(a.out, "go_nogo.txt")
    cache = txt + ".jsonl"
    perdoc = txt + ".perdoc.jsonl"

    tags = a.tags or (discover(a.out, a.pattern) if a.pattern else [])
    if not tags:
        ap.error("给出 tag 或 --pattern")

    if a.docs <= 0:
        ap.error("--docs must be positive")
    tags = list(dict.fromkeys(tags))

    # Keep all pre-existing tags, but only reuse selected rows when both files
    # carry the current schema and independently reproduce the same 3x3 tables.
    summaries = read_tagged_jsonl(cache)
    perdocs = read_tagged_jsonl(perdoc)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    rows, failures, n_computed = [], [], 0
    for i, tag in enumerate(tags, 1):
        valid, reason = cache_pair_valid(
            summaries.get(tag), perdocs.get(tag), a.docs)
        if valid and not a.force:
            rows.append(summaries[tag])
            print(f"[{i}/{len(tags)}] {tag} cache", flush=True)
            continue

        if summaries.get(tag) or perdocs.get(tag):
            print(f"[{i}/{len(tags)}] {tag} recompute ({'--force' if a.force else reason})",
                  flush=True)
        else:
            print(f"[{i}/{len(tags)}] {tag} compute", flush=True)
        try:
            row, perdoc_row = run_one(tag, a.out, a.docs, dev)
        except (FileNotFoundError, AssertionError, KeyError, ValueError) as exc:
            failures.append(f"{tag}: {type(exc).__name__}: {exc}")
            print(f"  failed: {failures[-1]}", flush=True)
            continue

        summaries[tag], perdocs[tag] = row, perdoc_row
        rows.append(row)
        n_computed += 1
        # Atomic snapshots preserve run-level restartability and remove duplicate
        # tags left by the former append-only cache format.
        write_tagged_jsonl(cache, summaries)
        write_tagged_jsonl(perdoc, perdocs)
        elapsed = time.time() - t0
        remaining = len(tags) - i
        eta = elapsed / n_computed * remaining / 60 if n_computed else NAN
        print(f"  done; remaining about {eta:.0f} min", flush=True)

    if failures:
        raise RuntimeError("terminal readout failed:\n  " + "\n  ".join(failures))
    if len(rows) != len(tags):
        raise RuntimeError(f"readout produced {len(rows)} rows for {len(tags)} tags")
    for row in rows:
        validate_perdoc(row, perdocs[row["tag"]], MASS_FLOOR)

    out = report(rows)
    print(out)
    with open(txt, "w", encoding="utf-8") as handle:
        handle.write(out + "\n")
        handle.write("\nraw: " + json.dumps(rows, ensure_ascii=False) + "\n")

    unrestricted = aggregate(rows)
    restricted = aggregate(rows, mass_restricted=True)
    argmax_prefix = txt + ".argmax"
    argmax_text = format_text(unrestricted, restricted)
    with open(argmax_prefix + ".txt", "w", encoding="utf-8") as handle:
        handle.write(argmax_text + "\n")
    with open(argmax_prefix + ".json", "w", encoding="utf-8") as handle:
        json.dump({"unrestricted": unrestricted,
                   "mass_restricted": restricted}, handle,
                  ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(argmax_prefix + ".tex", "w", encoding="utf-8") as handle:
        handle.write(format_latex(unrestricted, restricted) + "\n")

    print(f"\nwrote {txt}")
    print(f"wrote {cache} and {perdoc} (one row per tag)")
    print(f"wrote {argmax_prefix}.txt/.json/.tex")



if __name__ == "__main__":
    main()