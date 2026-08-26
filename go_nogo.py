"""rarity 对齐的终态读数。不重训，直接吃 checkpoint。

判据（四条一起看，缺一条都不能下结论）：

  1. 跨格是否有梯度。主读数。R_old 越大，「重复⇒非当前」在语料里被演示得
     越多，若 rarity 更便宜学，高 R_old 格应更大。单格绝对值不说明什么。
  2. 符号。break_rarity 让 rarity 与 frequency 反向移动，故 + 是 rarity 型、
     - 是 frequency 型。
  3. 概率质量是否仍在 {v_old, v_new}。编辑让答案值重复出现，是训练分布外的
     配置；质量散掉（mass<0.5）说明模型已不做任务，读数无意义。
  4. 非 q slot 对照的 |Δ| 应远小于主读数，排除「编辑本身让模型脱离任务」。

【为什么主 DV 是中位数与符号比例，不是均值】逐篇 Δ 重尾。R3_D2 在 32k 上
逃逸后 13 个 checkpoint 的均值是 1.79±0.49（范围 1.33–3.15，相对波动 27%），
而 frac+ 是 0.904±0.03（3.3%）。同一批 186 篇文档，所以这不是采样噪声，是
少数尾部文档把均值拉走 1.4 nats。要分辨的格间差异只有 2 nats 量级，用均值
画相图相邻格不可信。均值保留在附录。

R_old=1 列在此探针下无域（无副本可改写），按 N/A 报告 —— 结构性零信号对照。
"""
import argparse
import fnmatch
import json
import math
import os
import random
import re
import time
from typing import List, Optional, Sequence

import torch

from config import CorpusCfg, LangSpec
from generator import Doc, emit, generate_corpus
from model import LM, ModelCfg
from probe import (EditedDoc, _answer_val, _edit_break_rarity_ctrl, _margin,
                   _order_ok, _p_final, _q_pos, apply_edit,
                   fit_position_offset, r_last_value)
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

def run_one(tag: str, out_dir: str, n_docs: int, device) -> dict:
    model, spec, corpus, meta, conv, esc = load(tag, out_dir, device)
    vocab = Vocab(spec)
    pred = Predictor(model, vocab, spec, device)
    docs = list(generate_corpus(vocab, corpus, n_docs, seed_offset=1))
    offset = fit_position_offset(docs)
    rng = random.Random(0)

    recs = []                      # (Δ, mass, q_kept)
    for d in docs:
        ed = apply_edit(d, "break_rarity", vocab, corpus, rng, offset)
        if ed is None:
            continue
        truth = r_last_value(d)
        v_star = ed.v_star
        dm = (_margin(pred, ed, v_star, truth)
          - _margin(pred, d, v_star, truth))
        recs.append((dm, pred.mass(ed, [v_star, truth]), d.q_kept))

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

    ds = [x for x, _, _ in recs]
    ms = [m for _, m, _ in recs]
    # 逐篇 mass 门。格均值 mass=0.95 可以掩盖 10% 文档 mass=0.2，那些文档的
    # Δ 不是两条规则的竞争而是分布已经散掉，必须能单独看到。
    valid = [x for x, m, _ in recs if m >= MASS_FLOOR]
    
    # 格内按逐篇实际冗余度分层。跨格的 R_old 效应与 edit domain 共线
# （R3/D16 的 domain 0.66 vs R16 的 1.00），所以低 R_old 格的读数条件在
# 「副本存活较多」的子集上。若格内高/低 Rreal 两半也有同向差异，说明
# R_old 通过统计本身起作用；若无差异，说明跨格效应来自文档选择偏置。
    ks = sorted(k for _, _, k in recs)
    k_med = ks[len(ks) // 2] if ks else 0
# R_old 小的格上 k 只取两三个值，用 > 会把整格压进一侧（R3 实测 n_hiK=0）。
# 改用 >= 并在退化时降一级，保证两侧都非空。
    hi_k = sorted(x for x, _, k in recs if k >= k_med)
    lo_k = sorted(x for x, _, k in recs if k < k_med)
    if not lo_k or not hi_k:
        k_med = ks[len(ks) // 4] if ks else 0
        hi_k = sorted(x for x, _, k in recs if k > k_med)
        lo_k = sorted(x for x, _, k in recs if k <= k_med)

    n, nv = len(ds), len(valid)
    srt, srt_v = sorted(ds), sorted(valid)
    k_pos = sum(1 for x in ds if x > 0)

    acc, ca = conv.get("acc", NAN), conv.get("copy_acc", NAN)
    return dict(
        tag=tag, seed=axis_of(tag, r"_s(\d+)_", 0),
        r_old=corpus.r_old_lo, dd=axis_of(tag, r"_D(\d+)_", corpus.delta_d_lo),
        dd_lo=corpus.delta_d_lo, dd_hi=corpus.delta_d_hi,
        n=n, yield_rate=n / len(docs) if docs else NAN,
        d_margin=(sum(ds) / n) if n else NAN,
        d_median=percentile(srt, 0.50),
        d_q25=percentile(srt, 0.25), d_q75=percentile(srt, 0.75),
        d_trim05=trimmed_mean(ds, 0.05),
        frac_positive=(k_pos / n) if n else NAN,
        sign_p=sign_test_p(k_pos, n),
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
        state=classify(acc, ca, corpus),
        d_all=[round(x, 4) for x in ds],
        k_split=k_med,
d_med_hiK=percentile(hi_k, 0.50), n_hiK=len(hi_k),
d_med_loK=percentile(lo_k, 0.50), n_loK=len(lo_k),
        )


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

    # 相图用中位数聚合。均值在同一 checkpoint 上有 ±0.5 nats 的时间噪声，
    # 见文件头；跨 seed 也用中位数，并报告 seed 极差 —— 预注册的判据是
    # 「格间差异须超过格内 seed 极差才算有意义」。
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

    bad = [f"{r['tag']} state={r['state']}" for r in rows
           if r["state"] != "retr"]
    bad += [f"{r['tag']} mass={r['mass']:.2f}" for r in rows
            if r["mass"] == r["mass"] and r["mass"] < MASS_FLOOR]
    bad += [f"{r['tag']} frac+={r['frac_positive']:.2f} p={r['sign_p']:.2f}"
            for r in rows
            if r["sign_p"] == r["sign_p"] and r["sign_p"] > 0.05]

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

    # 增量缓存：75 格约 1.5 小时，中途崩了不该从头再来
    done = {}
    if os.path.exists(cache) and not a.force:
        with open(cache) as f:
            for line in f:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[o["tag"]] = o

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    rows, pd_out = [], []
    for i, t in enumerate(tags, 1):
        if t in done:
            rows.append(done[t])
            print(f"[{i}/{len(tags)}] {t} 缓存", flush=True)
            continue
        eta = (time.time() - t0) / max(1, len(rows) - len(done)) * (
            len(tags) - i + 1) / 60 if len(rows) > len(done) else NAN
        print(f"[{i}/{len(tags)}] {t}  剩余约 {eta:.0f}min", flush=True)
        try:
            r = run_one(t, a.out, a.docs, dev)
        except (FileNotFoundError, AssertionError, KeyError) as e:
            print(f"  跳过：{type(e).__name__}: {e}", flush=True)
            continue
        pd_out.append(dict(tag=t, d_all=r.pop("d_all")))
        rows.append(r)
        with open(cache, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for r in rows:
        r.pop("d_all", None)
    if pd_out:
        with open(perdoc, "a") as f:
            for o in pd_out:
                f.write(json.dumps(o) + "\n")

    out = report(rows)
    print(out)
    with open(txt, "w") as f:
        f.write(out + "\n")
        f.write("\nraw: " + json.dumps(rows, ensure_ascii=False) + "\n")
    print(f"\n已写入 {txt}")
    print(f"逐篇 Δ 在 {perdoc}（画分布图用，先画直方图再定摘要措辞）")


if __name__ == "__main__":
    main()