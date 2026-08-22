"""单格训练 + 在线规则归因。每个相图格子一个 run。

跨格严格对齐的量：total_steps、batch_docs、lr schedule、模型规模、seed 策略。
数据统计是唯一自变量，训练量不得随格变化。文档长度已由 selfcheck 确认
跨格一致（stmts≈110、len≈445），故 token 预算天然可比，日志仍记实际值备查。

不做 document packing：把多篇文档拼进一个 ctx 会让"倒数第 k 条语句"跨越
文档边界，position 规则的语义被污染，ΔD 也不再是文档内量。代价是 padding
浪费约 55% 的 ctx（len≈445 vs ctx_len=1024），换来自变量干净。若要省算力，
应降 ctx_len 到 512（validate_cfg 会据此收紧 n_stmts_hi），不要 packing。

流式数据：训练集用无限流（每篇文档全新采样，worker 间 seed 不相交），
评估集与探针集固定在 seed_offset=1，与训练流
不相交。这是"预训练数据统计 -> 规则"这一因果链的干净实现。

在线探针：只存 final checkpoint，规则轨迹在训练中直接算完写 jsonl。
90 run × 4 个 log-spaced 探针点，若改为存 ckpt 离线跑，磁盘要 20GB+
且要重载 90 次。轨迹本身是结果的一部分（规则可能在训练中切换，
参 Singh et al. 的 ICL 瞬态性）。
"""
import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from config import CorpusCfg, LangSpec, dd_band
from generator import Doc, generate_corpus
from model import LM, ModelCfg
from probe import (EDITS, MAIN_GRID_EXCLUDE, RULE_NAMES, TRUTH, attribute,
                   causal, fit_position_offset, identifiability, rule_groups)
from vocab import Vocab

PROBE_N = 200          # 在线探针文档数。final ckpt 另跑离线全量
YIELD_MIN = 0.05


@dataclass
class TrainCfg:
    total_steps: int = 15000
    batch_docs: int = 192
    lr: float = 1e-3
    lr_min_frac: float = 0.1
    warmup: int = 500
    wd: float = 0.1        # 无限流语料无过拟合对象；且 bench 显示权重范数
                           # 增长本身是 attn sharpening 的机制之一，衰减有害
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_every: int = 1500
    eval_docs: int = 8000      # R12/ΔD=2 的 n_tail_updates==0 子集仅 12%，
                               # 8000 篇才有 ~960 个样本够点估计
    probe_points: int = 7      # log-spaced，含最后一步
    num_workers: int = 8
    seed: int = 0
    out_dir: str = "runs"
    compile: bool = False      # 90 run 各自编译不划算；同进程连跑多格时可开
    sched: str = "cos"         # cos | const。诊断 run 用 const：衰减会在相变
                           # 刚起时掐掉 lr（bench e1 实测）

def infer_vocab_size(vocab: Vocab, spec: LangSpec) -> int:
    """不依赖 vocab 是否暴露 size：取所有已知最大 token id + 1。"""
    m = max(vocab.val(spec.n_values - 1), vocab.ent(spec.n_entities - 1),
            vocab.attr(spec.n_attrs - 1), vocab.time(spec.n_time_idx - 1),
            vocab.PAD, vocab.SEP, vocab.UPD, vocab.QUERY, vocab.ARROW)
    return m + 1


def val_token_range(vocab: Vocab, spec: LangSpec) -> range:
    """value token 必须是连续区间：受限 argmax 与探针的候选集都依赖这一点。"""
    lo, hi = vocab.val(0), vocab.val(spec.n_values - 1)
    assert hi - lo == spec.n_values - 1, "value token 不连续，受限 argmax 失效"
    for v in (0, spec.n_values // 2, spec.n_values - 1):
        assert vocab.val(v) == lo + v
    return range(lo, hi + 1)


# ---------------- 数据 ----------------

class Stream(IterableDataset):
    """无限流。worker w 用 seed_offset = 1000 + w + 每轮递增，与评估集(1)不相交。

    注意：generate_corpus 对同一 seed_offset 产出相同文档，故外层循环必须
    推进 offset，否则 4096 篇会被反复重放 —— batch 48 时每 341 步一个 epoch，
    25.7M 参数在 30+ epoch 后会背下这 4096 篇绑定，表现为训练 loss 破解析
    地板而评估 loss 反向上升（floor_wd 实测：train 0.68 / eval 4.87）。"""

    CHUNK = 4096

    def __init__(self, cfg: CorpusCfg, spec: LangSpec, n_workers: int = 1):
        self.cfg, self.spec = cfg, spec
        self.stride = max(1, n_workers)

    def __iter__(self):
        wi = get_worker_info()
        w = wi.id if wi else 0
        vocab = Vocab(self.spec)
        off = 1000 + w
        while True:
            for d in generate_corpus(vocab, self.cfg, self.CHUNK, seed_offset=off):
                yield d.tokens
            off += self.stride          # 各 worker 的 offset 序列不重叠


def collate(batch: List[List[int]], pad: int):
    """动态 padding 到 batch 内最长。labels 在 padding 位置为 -100。"""
    L = max(len(t) for t in batch)
    ids = torch.full((len(batch), L), pad, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, t in enumerate(batch):
        ids[i, :len(t)] = torch.tensor(t)
        lab[i, :len(t)] = torch.tensor(t)
    return ids, lab


# ---------------- 评估 ----------------

@torch.no_grad()
def evaluate(model: LM, docs: Sequence[Doc], vocab: Vocab, spec: LangSpec,
             device, bs: int = 64) -> dict:
    """答案位单独度量。全 token loss 的动态范围被填充 token 占满：
    unmarked 语料的解析地板是 (ln n_ent + ln n_attr + ln n_val)/4 ≈ 4.32，
    而答案 token 只占 1/445 个位置、约 0.4% 的 loss，从随机到全对总 loss
    只变 0.017。ans_nll（对比 ln n_values = 7.60）与 ans_rank 才是训练信号。
    按 n_tail_updates 分层：==0 的子集里"最后一条 update 就是答案"这条捷径
    失效，是规则归因唯一干净的子集（R12/ΔD=2 该子集仅占 12%）。"""
    model.eval()
    vr = val_token_range(vocab, spec)
    lo, n_val = vr.start, len(vr)
    hit = [[0, 0], [0, 0]]          # [tail0, tailpos] × [correct, n]
    nll = [0.0, 0.0]
    rank = [0, 0]
    top10 = [0, 0]
    loss_sum = loss_n = 0.0
    for i in range(0, len(docs), bs):
        chunk = docs[i:i + bs]
        ids, lab = collate([d.tokens for d in chunk], vocab.PAD)
        ids, lab = ids.to(device), lab.to(device)
        logits, loss = model(ids, lab)
        loss_sum += float(loss) * len(chunk)
        loss_n += len(chunk)
        for j, d in enumerate(chunk):
            row = logits[j, d.answer_pos - 1, lo:lo + n_val].float()
            tv = d.answer - lo
            lp = torch.log_softmax(row, -1)
            r = int((row > row[tv]).sum())
            s = 0 if d.n_tail_updates == 0 else 1
            hit[s][0] += int(r == 0)
            hit[s][1] += 1
            nll[s] += -float(lp[tv])
            rank[s] += r
            top10[s] += int(r < 10)
    model.train()
    nan = float("nan")

    def agg(s):
        n = hit[s][1]
        return (dict(acc=hit[s][0] / n, ans_nll=nll[s] / n,
                     ans_rank=rank[s] / n, ans_top10=top10[s] / n, n=n)
                if n else dict(acc=nan, ans_nll=nan, ans_rank=nan,
                               ans_top10=nan, n=0))

    a0, a1 = agg(0), agg(1)
    n0, n1 = a0["n"], a1["n"]
    tot = {k: ((a0[k] * n0 + a1[k] * n1) / (n0 + n1)) if n0 + n1 else nan
           for k in ("acc", "ans_nll", "ans_rank", "ans_top10")}
    return dict(loss=loss_sum / loss_n, **tot,
                acc_tail0=a0["acc"], ans_nll_tail0=a0["ans_nll"],
                ans_rank_tail0=a0["ans_rank"], n_tail0=n0,
                acc_tailpos=a1["acc"], ans_nll_tailpos=a1["ans_nll"],
                n_tailpos=n1,
                chance_acc=1.0 / n_val, chance_nll=math.log(n_val))
#  --------- 探针适配器 ----------------

class ModelPredictor:
    """probe.py 的 predictor 接口（predict / logp）。按前缀缓存 log-softmax，
    同一 view 在 attribute 与 causal 里各调一次，缓存省一半前向。"""

    def __init__(self, model: LM, vocab: Vocab, spec: LangSpec, device,
                 cache_size: int = 4096):
        self.m, self.vocab, self.spec, self.dev = model, vocab, spec, device
        vr = val_token_range(vocab, spec)
        self.lo, self.n_val = vr.start, len(vr)
        self._cache: Dict[tuple, torch.Tensor] = {}
        self._cap = cache_size

    @torch.no_grad()
    def _lp(self, view) -> torch.Tensor:
        key = tuple(view.tokens[:view.answer_pos])
        got = self._cache.get(key)
        if got is None:
            ids = torch.tensor([key], device=self.dev)
            logits, _ = self.m(ids)
            got = torch.log_softmax(
                logits[0, -1, self.lo:self.lo + self.n_val].float(), -1).cpu()
            if len(self._cache) >= self._cap:
                self._cache.clear()
            self._cache[key] = got
        return got

    def predict(self, view) -> int:
        return int(self._lp(view).argmax())          # raw value id

    def logp(self, view, cands: Sequence[int]) -> List[float]:
        lp = self._lp(view)
        return [float(lp[v]) for v in cands]


@torch.no_grad()
def run_probe(model: LM, docs: Sequence[Doc], vocab: Vocab, spec: LangSpec,
              corpus: CorpusCfg, device) -> dict:
    model.eval()
    pred = ModelPredictor(model, vocab, spec, device)
    offset = fit_position_offset(docs)
    rows = attribute(docs, pred, offset, vocab)
    out = dict(offset=offset,
           agree={k: rows[k]["rate_disc"] for k in RULE_NAMES},
           truth_on_disc={k: rows[k]["rate_truth_disc"] for k in RULE_NAMES},
           n_disc={k: rows[k]["n_disc"] for k in RULE_NAMES},
           agree_all={k: rows[k]["rate"] for k in RULE_NAMES},
           causal={})
    KEYS = ("target", "sign", "n", "yield_rate", "d_margin", "frac_expected",
        "dd_cond", "dd_all", "mass_mean", "d_all")
    for kind in EDITS:
        r = causal(docs, pred, kind, offset, vocab, corpus)
        if r["yield_rate"] >= YIELD_MIN:
            out["causal"][kind] = {k: r[k] for k in KEYS if k in r}
    model.train()
    return out


def dominant_rule(probe: dict, groups: Optional[Dict[str, str]] = None) -> str:
    """相图着色。判据是同一子集上的配对差 rate_disc - rate_truth_disc：
    模型在"规则 k 与真值分歧"的样本上更常跟 k 走，才算被 k 驱动。
    阈值 0.05 是保守占位，正文须报告着色对阈值的敏感性。
    返回值是等价类标签（见 probe.rule_groups）：不可辨识的规则对必须
    合并输出，否则严格 > 会按 RULE_NAMES 顺序任意挑一个。"""
    a, t = probe["agree"], probe.get("truth_on_disc", {})
    best, margin = TRUTH, 0.05
    for k in RULE_NAMES:
        if k == TRUTH or k in MAIN_GRID_EXCLUDE:
            continue
        v, tv = a.get(k, float("nan")), t.get(k, float("nan"))
        if v == v and tv == tv and v - tv > margin:
            best, margin = k, v - tv
    return (groups or {}).get(best, best)


def _val_slots(d, cfg: CorpusCfg):
    """[(val token 位置, 该值是否为同 slot 上一次的重复)]。位置算法与 emit 一致。"""
    out, off, seen = [], 0, {}
    for s in d.stmts:
        if cfg.use_marker and s.is_update:
            off += 1
        key = (s.ent, s.attr)
        out.append((off + 2, seen.get(key) == s.val))
        seen[key] = s.val
        off += 4
    return out


@torch.no_grad()
def copy_diag(model: LM, docs: Sequence[Doc], vocab: Vocab, spec: LangSpec,
              cfg: CorpusCfg, device, bs: int = 16, max_docs: int = 256) -> dict:
    """上下文复制诊断。语料里每篇约 50 个 val token 是"同 slot 上一次的值"的
    重复，预测它们需要 (e,a) 匹配 + 复制，正是答案位要用的同一条回路，
    但样本数多 50 倍、占全 token loss 约 45%（答案位 0.4%）。

    copy_nll 不动 ⇒ 回路没形成，与答案位的信号比例无关，加权也救不了。
    novel_nll 是对照：新值不可预测，应恒等于 ln n_values。"""
    model.eval()
    lo, n_val = vocab.VAL0, spec.n_values
    nll = [0.0, 0.0]
    hit = [0, 0]
    cnt = [0, 0]
    docs = list(docs[:max_docs])
    for i in range(0, len(docs), bs):
        chunk = docs[i:i + bs]
        ids, _ = collate([d.tokens for d in chunk], vocab.PAD)
        logits, _ = model(ids.to(device))
        js, ps, tv, ks = [], [], [], []
        for j, d in enumerate(chunk):
            for vpos, is_rep in _val_slots(d, cfg):
                js.append(j); ps.append(vpos - 1)
                tv.append(d.tokens[vpos] - lo); ks.append(int(is_rep))
        js = torch.tensor(js, device=device)
        ps = torch.tensor(ps, device=device)
        tvt = torch.tensor(tv, device=device)
        sel = logits[js, ps, lo:lo + n_val].float()
        lp = torch.log_softmax(sel, -1)
        n_ = -lp[torch.arange(len(tv), device=device), tvt]
        ok = (sel.argmax(-1) == tvt)
        kt = torch.tensor(ks, device=device, dtype=torch.bool)
        for k, m in ((1, kt), (0, ~kt)):
            nll[k] += float(n_[m].sum()); hit[k] += int(ok[m].sum())
            cnt[k] += int(m.sum())
    model.train()
    nan = float("nan")
    return dict(copy_nll=nll[1] / cnt[1] if cnt[1] else nan,
            copy_acc=hit[1] / cnt[1] if cnt[1] else nan, n_copy=cnt[1],
            novel_nll=nll[0] / cnt[0] if cnt[0] else nan,
            novel_acc=hit[0] / cnt[0] if cnt[0] else nan, n_novel=cnt[0],
            copy_chance_nll=math.log(n_val))

# ---------------- 训练 ----------------

def lr_at(step: int, c: TrainCfg) -> float:
    if step < c.warmup:
        return c.lr * (step + 1) / c.warmup
    if c.sched == "const":
        return c.lr
    t = (step - c.warmup) / max(1, c.total_steps - c.warmup)
    return c.lr * (c.lr_min_frac + (1 - c.lr_min_frac) *
                   0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def train(corpus: CorpusCfg, tc: TrainCfg, mc_kw: Optional[dict] = None,
          spec: Optional[LangSpec] = None, tag_suffix: str = "",
          ckpt_steps: Optional[Sequence[int]] = None) -> dict:
    spec = spec or LangSpec()
    vocab = Vocab(spec)
    torch.manual_seed(tc.seed)
    random.seed(tc.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
        if device == "cuda" else torch.float32

    mc = ModelCfg(vocab_size=infer_vocab_size(vocab, spec), ctx_len=spec.ctx_len,
                  **(mc_kw or {}))
    model = LM(mc).to(device)
    if tc.compile:
        model = torch.compile(model)
  
    opt = torch.optim.AdamW(
    (model._orig_mod if tc.compile else model).param_groups(tc.wd),
    lr=tc.lr, betas=tc.betas, eps=1e-8, fused=(device == "cuda"))

    scaler = torch.amp.GradScaler(enabled=(amp == torch.float16))

    
    loader = DataLoader(Stream(corpus, spec, n_workers=max(1, tc.num_workers)), batch_size=tc.batch_docs,
                        num_workers=tc.num_workers,
                        collate_fn=lambda b: collate(b, vocab.PAD),
                        pin_memory=(device == "cuda"),
                        persistent_workers=(tc.num_workers > 0),
                        prefetch_factor=4 if tc.num_workers else None)


    eval_docs = list(generate_corpus(vocab, corpus, tc.eval_docs, seed_offset=1))
    probe_docs = eval_docs[:PROBE_N]
    ident = identifiability(probe_docs, fit_position_offset(probe_docs))
    groups = rule_groups(ident)
    
    os.makedirs(tc.out_dir, exist_ok=True)
    tag = corpus.name + tag_suffix
    logf = open(os.path.join(tc.out_dir, f"{tag}.jsonl"), "w")

    def emit(rec):
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logf.flush()

    ck_set = set(ckpt_steps or ())
    ck_dir = os.path.join(tc.out_dir, "ckpt")
    if ck_set:
        os.makedirs(ck_dir, exist_ok=True)

    def save_ckpt(step: int):
        """中途 checkpoint，只为 flatdir.py 的梯度测量而存。

        两个刻意的差别，都不能改：
        fp32 而非 bf16 —— 终态 .pt 存 bf16 省盘，但 cos(g_L, g_Δ) 的量级在
        1e-3 附近，bf16 的 8 位尾数会把 θ 舍到与该量级同阶，测出来的角度是
        舍入噪声。梯度算术本身在 fp32，起点也必须是 fp32。
        不存 optimizer state —— Adam 的二阶矩会让盘占翻三倍，而 flatdir 测
        的是 ∇L 与 ∇Δ 的几何关系，不重启训练。代价是这些 ckpt 不能续跑，
        flatdir.py 的 docstring 里写清了这一条对结论的限制。

        存盘不取随机数、不动 dataloader 迭代器，故加 --ckpt-steps 之后
        训练轨迹与不加时逐比特相同（App A 的可复现性论证不受影响）。
        """
        mdl = model._orig_mod if tc.compile else model
        torch.save(dict(model={k: v.float().cpu()
                               for k, v in mdl.state_dict().items()},
                        model_cfg=asdict(mc), corpus=asdict(corpus),
                        train=asdict(tc), spec=asdict(spec), step=step),
                   os.path.join(ck_dir, f"{tag}_s{step}.pt"))

    n_par = (model._orig_mod if tc.compile else model).n_params()
    emit(dict(kind="meta", corpus=asdict(corpus), train=asdict(tc),
              model=asdict(mc), spec=asdict(spec), n_params=n_par,
              n_params_nonembed=(model._orig_mod if tc.compile else model)
              .n_params(embed=False),
              device=device, amp=str(amp),
              rule_groups=groups,
              ckpt_steps=sorted(ck_set),
              ident_top={k: v for k, v in
                         sorted(ident.items(), key=lambda kv: -kv[1])[:5]})
                         )
    print(f"[{tag}] params={n_par/1e6:.1f}M device={device} amp={amp}")

    # 探针点：几何 spacing（16000 步下是 1/5/25/127/635/3187/16000）覆盖早期相变，
# 但 3187 之后直接跳到终点，中间 13000 步空白，而「逃逸后 Δ 是否稳定」是相图
# 的主要混淆（R16/D2 实测 3187 的 +2.83 到 16000 的 +2.24，降 21%，中间无点
# 无法判断是单调下降还是早已走平）。故并入每 2000 一个的线性点。
    n_log = max(2, tc.probe_points)
    pts = {max(1, int(round(tc.total_steps ** (i / (n_log - 1)))))
       for i in range(n_log)}
    pts |= set(range(2000, tc.total_steps + 1, 2000))
    pts.add(tc.total_steps)
    pts = sorted(p for p in pts if 1 <= p <= tc.total_steps)
   
    it = iter(loader)
    tok_seen = 0
    t0 = time.time()
    model.train()
    for step in range(1, tc.total_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, tc)
        ids, lab = next(it)
        ids, lab = ids.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        tok_seen += int((lab != -100).sum())
        with torch.autocast(device_type=device, dtype=amp, enabled=(device == "cuda")):
            _, loss = model(ids, lab)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % 100 == 0:
            mdl = model._orig_mod if tc.compile else model
            gn = float(sum(p.norm() ** 2 for p in mdl.parameters()) ** 0.5)
            hn = float(mdl.head.weight.norm())
            emit(dict(kind="train", step=step, loss=float(loss),
              lr=lr_at(step - 1, tc), tokens=tok_seen,
              param_norm=gn, head_norm=hn,
              sec_per_step=(time.time() - t0) / step))

        if step % tc.eval_every == 0 or step == tc.total_steps:
            ev = evaluate(model, eval_docs, vocab, spec, device)
            cd = copy_diag(model, eval_docs, vocab, spec, corpus, device)
            emit(dict(kind="eval", step=step, tokens=tok_seen, **ev, **cd))
            print(f"  step {step:>6} loss {ev['loss']:.4f} "
          f"copyNLL {cd['copy_nll']:.3f} copyAcc {cd['copy_acc']:.3f} "
          f"novelNLL {cd['novel_nll']:.2f} "
          f"ansNLL {ev['ans_nll']:.3f} rank {ev['ans_rank']:.0f} "
          f"acc {ev['acc']:.3f} tail0 {ev['acc_tail0']:.3f}")
        
        if step in pts:
            pb = run_probe(model, probe_docs, vocab, spec, corpus, device)
            dom = dominant_rule(pb, groups)
            emit(dict(kind="probe", step=step, dominant=dom, **pb))
            print(f"  step {step:>6} probe dominant={dom} "
          + " ".join(f"{k}={pb['agree'][k]:.2f}" for k in RULE_NAMES))

        if step in ck_set:
            save_ckpt(step)
            emit(dict(kind="ckpt", step=step,
                      path=os.path.join("ckpt", f"{tag}_s{step}.pt")))
            print(f"  step {step:>6} ckpt -> ckpt/{tag}_s{step}.pt")
    
    final_eval = evaluate(model, eval_docs, vocab, spec, device)
    final_copy = copy_diag(model, eval_docs, vocab, spec, corpus, device)
# 收敛门：copy_acc 量的是 (e,a) 匹配 + 复制这条回路，与答案位同一机制但
# 样本多 50 倍。未过门的格子其 dominant_rule 无意义（模型还没学会检索，
# 谈不上"用哪条规则解决冲突"），相图必须把它们标为未收敛而非着色。  
    converged = bool(final_copy["copy_acc"] > 0.9)
    sd = {k: v.to(torch.bfloat16) for k, v in
      (model._orig_mod if tc.compile else model).state_dict().items()}
    torch.save(dict(model=sd, model_cfg=asdict(mc), corpus=asdict(corpus),
                train=asdict(tc), step=tc.total_steps, eval=final_eval,
                converged=converged),
           os.path.join(tc.out_dir, f"{tag}.pt"))
    emit(dict(kind="done", step=tc.total_steps, tokens=tok_seen,
          minutes=(time.time() - t0) / 60, converged=converged,
          **final_eval, **final_copy))
    print(f"[{tag}] 完成 copyAcc={final_copy['copy_acc']:.3f} "
      f"converged={converged} acc={final_eval['acc']:.3f} "
      f"tail0={final_eval['acc_tail0']:.3f}")
    logf.close()
    return final_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=TrainCfg.total_steps)
    ap.add_argument("--batch", type=int, default=TrainCfg.batch_docs)
    ap.add_argument("--lr", type=float, default=TrainCfg.lr)
    ap.add_argument("--workers", type=int, default=TrainCfg.num_workers)
    ap.add_argument("--out", type=str, default="runs")
    ap.add_argument("--eval-docs", type=int, default=TrainCfg.eval_docs)
    ap.add_argument("--eval-every", type=int, default=TrainCfg.eval_every)
    ap.add_argument("--compile", action="store_true")
    # 诊断用：语言规模与文档形状。都不是论文自变量（自变量只有 R_old 与 ΔD），
    # 缩小规模不损害主张，只需在正文声明规模并说明 n_bindings 仍远超模型容量。
    ap.add_argument("--tie", action="store_true", help="tie embedding 与 unembedding")
    ap.add_argument("--n-values", type=int, default=None)
    ap.add_argument("--n-entities", type=int, default=None)
    ap.add_argument("--ctx-len", type=int, default=None)
    ap.add_argument("--stmts-lo", type=int, default=None)
    ap.add_argument("--stmts-hi", type=int, default=None)
    ap.add_argument("--tag", type=str, default="", help="日志后缀，避免多 pilot 互相覆盖")
    ap.add_argument("--sched", default="cos", choices=["cos", "const"])
    ap.add_argument("--wd", type=float, default=TrainCfg.wd)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--n-head", type=int, default=None)
    ap.add_argument("--qk-gain", type=float, default=None,
                    help="QK-norm 增益初值，默认 2.0（App B 的 Eq.5）")
    # 固定带宽 arm：显式给 ΔD 支撑区间，绕过 dd_band 的相对带宽。
    # dd_band(d, rel=0.5) 的支撑随 d 变宽（3/3/7/9/17），posCeil 从 0.333
    # 掉到 0.059，与 ΔD 轴共线 —— 这是 §8 点名但未完成的那个控制。
    # 给 --dd-lo/--dd-hi 即可令支撑恒定（宽 9 -> posCeil 恒为 0.111）。
    # --d 仍是标签轴值，只决定 tag 里的 D 号，不再决定区间。
    ap.add_argument("--dd-lo", type=int, default=None,
                    help="ΔD 支撑下界，覆盖 dd_band；须与 --dd-hi 同时给")
    ap.add_argument("--dd-hi", type=int, default=None,
                    help="ΔD 支撑上界，覆盖 dd_band；须与 --dd-lo 同时给")
    # 中途 checkpoint：只为 flatdir.py 的梯度对齐测量而存，默认不存。
    # 存盘不消耗 RNG、不动 dataloader，故训练轨迹仍逐比特可复现。
    ap.add_argument("--ckpt-steps", type=str, default="",
                    help="逗号分隔的步号，在这些步存 fp32 checkpoint")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="每 N 步存一个 fp32 checkpoint（与 --ckpt-steps 取并集）")
    a = ap.parse_args()          # <- 必须在所有 add_argument 之后

    over = {k: v for k, v in
            dict(n_values=a.n_values, n_entities=a.n_entities,
                 ctx_len=a.ctx_len).items() if v is not None}
    spec = replace(LangSpec(), **over) if over else LangSpec()

    if (a.dd_lo is None) != (a.dd_hi is None):
        ap.error("--dd-lo 与 --dd-hi 必须同时给：只给一半会静默退回 dd_band")
    if a.dd_lo is not None:
        dlo, dhi = a.dd_lo, a.dd_hi
        print(f"[band] ΔD ~ U[{dlo},{dhi}] 支撑宽 {dhi - dlo + 1} "
              f"posCeil={1.0 / (dhi - dlo + 1):.3f}（显式，非 dd_band）")
    else:
        dlo, dhi = dd_band(a.d)
    ckw = {}
    if a.stmts_lo is not None:
        ckw["n_stmts_lo"] = a.stmts_lo
    if a.stmts_hi is not None:
        ckw["n_stmts_hi"] = a.stmts_hi
    
    mkw = dict(tie_embed=a.tie)
    if a.d_model is not None:
        mkw.update(d_model=a.d_model, d_mlp=a.d_model * 8 // 3)
    if a.n_layer is not None:
        mkw["n_layer"] = a.n_layer
    if a.n_head is not None:
        mkw["n_head"] = a.n_head
    if a.qk_gain is not None:
        mkw["qk_norm_gain"] = a.qk_gain
    
    corpus = CorpusCfg(name=f"R{a.r}_D{a.d}_s{a.seed}", seed=a.seed,
                       p_update=0.5, max_updates=1,
                       r_old_lo=a.r, r_old_hi=a.r, use_marker=False,
                       delta_d_lo=dlo, delta_d_hi=dhi, p_hist_query=0.0, **ckw)
    tc = TrainCfg(total_steps=a.steps, batch_docs=a.batch, lr=a.lr, seed=a.seed,
              sched=a.sched, wd=a.wd, num_workers=a.workers, out_dir=a.out,
              eval_docs=a.eval_docs, eval_every=a.eval_every,
              compile=a.compile)
    ck = {int(x) for x in a.ckpt_steps.split(",") if x.strip()}
    if a.ckpt_every > 0:
        ck |= set(range(a.ckpt_every, a.steps + 1, a.ckpt_every))
    train(corpus, tc, mc_kw=mkw, spec=spec,
        tag_suffix=("_" + a.tag if a.tag else ""),
        ckpt_steps=sorted(s for s in ck if 1 <= s <= a.steps))
if __name__ == "__main__":
    main()