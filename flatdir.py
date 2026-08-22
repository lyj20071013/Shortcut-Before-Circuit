"""∇θΔ 与 ∇θL 的几何关系。§5.6 那句"the direct test is not in this paper"的补丁。

测的是什么
  g_L  = ∇θ L，L 是训练分布上的全 token CE（与 train.py 完全同一个 loss）
  g_Δ  = ∇θ E[Δ]，Δ 是 break_rarity 的配对读数（+ 为 rarity 型）
  另测 g_S = ∇θ E[σ(Δ/τ)]，符号比例的光滑替身。正文主 DV 是符号比例而不是
       均值，符号比例不可微，σ(Δ/τ) 是最接近的可微代理，τ 是唯一自由量。

四层证据，缺一层都不足以说"存在平坦方向"
  1 一阶角度   cos(g_L, g_Δ)。单独看没有意义，必须与下面两个尺度并列。
  2 噪声地板   把 loss 批分两半，cos(g_L^A, g_L^B) 给出 g_L 里有多少是信号；
               再给一个随机方向的 cos 作为纯偶然水平（26M 维下约 2e-4）。
               只有当 cos(g_L, g_Δ) 落在偶然水平附近、而 cos(g_L^A, g_L^B)
               显著大于它时，"正交"才是关于真梯度的陈述而不是关于噪声的。
  3 有限差分   沿 û 走 ±ε，同一批文档上读 L 与 Δ 的实际变化。头条数字是
               "每抬高一个单位训练 loss 能换到多少 nats 的 Δ"，并与沿 û_L
               走同样步长的对照相比。一阶角度可以被二阶效应推翻，这一层不会。
  4 曲率       对称二阶差分，沿 û_Δ⊥ / û_L / 随机方向各一次，同批同点。
               平坦是关于二阶的陈述；只报一阶角度会被审稿人一句话打回。

û_Δ⊥ 是 g_Δ 剔除 g_L 分量后的单位向量，是"交换两条规则但不动损失"这个对象
最干净的实现。头条结论应该用它，g_Δ 原方向作为参照一起报。

这个脚本不能建立什么（正文里必须照抄）
  ckpt 不含 optimizer state（省三倍磁盘），所以这里测的是几何，不能从
    checkpoint 续训，也不能说"优化器确实没往这个方向走"，只能说"目标函数在
    这个方向上没有可用的一阶或二阶信号"。
  g_L 是有限批估计。噪声地板量化了这一点，但 ε→0 的极限不可测。
  Δ 的梯度取自未按 mass 门筛过的全域文档：逐文档门是不连续的，放进被求导的
    目标里会引入伪梯度。mass 均值一并输出，低于 0.5 时该 checkpoint 的读数
    本身无效，整行应弃用（--mass-gate 可开硬筛做稳健性对照）。

精度
  全程 fp32 且显式关掉 TF32。关心的 ΔL 在 1e-5 量级，TF32 的 10 位尾数会把它
  变成舍入噪声。FD 的每一次求值都复用同一批文档与同一组编辑对，否则采样噪声
  比信号大三个数量级。
"""
import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from config import CorpusCfg, LangSpec
from generator import Doc, generate_corpus
from model import LM, ModelCfg
from probe import apply_edit, fit_position_offset, r_last_value
from train import collate, val_token_range
from vocab import Vocab

NAN = float("nan")


# ---------------- 精度 ----------------

def strict_fp32() -> None:
    """TF32 必须关。ΔL 的量级是 1e-5，TF32 尾数 10 位在 L≈1 附近的分辨率
    约 1e-3，开着测出来的是零。"""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


# ---------------- 参数向量化 ----------------

class Flat:
    """参数与梯度的扁平视图。self.params 的顺序即向量分块顺序，
    全程用同一个实例，避免两次 flatten 的分块顺序不一致。"""

    def __init__(self, model: LM):
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.n = sum(p.numel() for p in self.params)
        # 非 embedding 掩码：emb/head 的梯度是稀疏的、被 token 频率支配，
        # 全参数余弦会被它们的量级淹没，两个版本都要报。
        emb_ids = {id(model.emb.weight)}
        if model.wpe is not None:
            emb_ids.add(id(model.wpe.weight))
        emb_ids.add(id(model.head.weight))
        mask = torch.zeros(self.n, dtype=torch.bool)
        i = 0
        for p in self.params:
            k = p.numel()
            if id(p) not in emb_ids:
                mask[i:i + k] = True
            i += k
        self.core = mask                      # True = 非 embedding

    def get(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    def set_(self, vec: torch.Tensor) -> None:
        i = 0
        with torch.no_grad():
            for p in self.params:
                k = p.numel()
                p.copy_(vec[i:i + k].view_as(p))
                i += k

    def grad(self) -> torch.Tensor:
        out = []
        for p in self.params:
            out.append(torch.zeros_like(p).reshape(-1) if p.grad is None
                       else p.grad.detach().reshape(-1))
        return torch.cat(out)

    def zero(self) -> None:
        for p in self.params:
            p.grad = None


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return NAN
    return float((a @ b) / (na * nb))


def unit(v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    return v / n if n > 0 else v


def reject(v: torch.Tensor, along: torch.Tensor) -> torch.Tensor:
    """v 中剔除 along 分量。û_Δ⊥ 由此得到。"""
    u = unit(along)
    return v - (v @ u) * u


# ---------------- 载入 ----------------

def load_ckpt(path: str, device) -> Tuple[LM, LangSpec, CorpusCfg, int]:
    """中途 ckpt 自带 spec/corpus；终态 .pt 没有 spec，退回同目录 jsonl 的 meta
    首行（与 go_nogo.load 同一约定）。"""
    ck = torch.load(path, map_location=device)
    if "spec" in ck and "corpus" in ck:
        spec, corpus = LangSpec(**ck["spec"]), CorpusCfg(**ck["corpus"])
    else:
        base = os.path.basename(path)
        tag = base[:-3]
        d = os.path.dirname(path)
        cand = [os.path.join(d, f"{tag}.jsonl"),
                os.path.join(os.path.dirname(d), f"{tag}.jsonl")]
        jl = next((p for p in cand if os.path.exists(p)), None)
        if jl is None:
            raise FileNotFoundError(
                f"{path} 里没有 spec，且找不到配套 jsonl：{cand}。"
                f"终态 .pt 必须与它的 jsonl 同目录。")
        with open(jl) as f:
            meta = json.loads(f.readline())
        assert meta["kind"] == "meta", f"{jl} 首行不是 meta"
        spec, corpus = LangSpec(**meta["spec"]), CorpusCfg(**meta["corpus"])
    model = LM(ModelCfg(**ck["model_cfg"])).to(device)
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    model.eval()                  # 本模型无 dropout，eval 只是去掉歧义
    return model, spec, corpus, int(ck.get("step", -1))


# ---------------- 两个目标的输入 ----------------

def loss_batches(vocab: Vocab, corpus: CorpusCfg, n_docs: int, bs: int,
                 seed_offset: int, device) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """训练分布上的固定批。seed_offset 默认远离评估集(1)与训练流(1000+w)，
    所以这些文档既不在评估集里，也不是模型见过的那一份。
    返回 (ids, lab, n_label_tokens)，token 数用于把小批均值还原成大批均值 ——
    CE 是 token 平均而非文档平均，等权累加会给短文档过高权重。"""
    docs = list(generate_corpus(vocab, corpus, n_docs, seed_offset=seed_offset))
    out = []
    for i in range(0, len(docs), bs):
        ids, lab = collate([d.tokens for d in docs[i:i + bs]], vocab.PAD)
        ntok = int((lab[:, 1:] != -100).sum())     # 与 forward 的移位对齐
        out.append((ids.to(device), lab.to(device), ntok))
    return out


class Pairs:
    """break_rarity 的配对输入，一次构好反复使用。

    每对是 (base 前缀, edit 前缀, v* 的 token id, truth 的 token id)。前缀取
    tokens[:answer_pos]，读 logits 的最后一个位置。右 padding 在 causal 注意力
    下不泄漏：位置 answer_pos-1 只能看到 ≤ 自己的位置。
    """

    def __init__(self, model_vocab: Vocab, corpus: CorpusCfg, n_docs: int,
                 bs: int, seed_offset: int, device, edit_seed: int = 0):
        self.vocab, self.dev, self.bs = model_vocab, device, bs
        docs = list(generate_corpus(model_vocab, corpus, n_docs,
                                    seed_offset=seed_offset))
        offset = fit_position_offset(docs)
        rng = random.Random(edit_seed)          # 与 go_nogo.run_one 同一约定
        rows = []
        for d in docs:
            ed = apply_edit(d, "break_rarity", model_vocab, corpus, rng, offset)
            if ed is None:                      # 不在编辑域内
                continue
            truth = r_last_value(d)
            rows.append((d.tokens[:d.answer_pos],
                         ed.tokens[:ed.answer_pos],
                         model_vocab.val(ed.v_star),
                         model_vocab.val(truth)))
        self.n = len(rows)
        self.yield_rate = self.n / len(docs) if docs else NAN
        self.batches = [self._pack(rows[i:i + bs]) for i in range(0, self.n, bs)]

    def _pack(self, rows):
        toks = [r[0] for r in rows] + [r[1] for r in rows]
        L = max(len(t) for t in toks)
        ids = torch.full((len(toks), L), self.vocab.PAD, dtype=torch.long)
        pos = torch.empty(len(toks), dtype=torch.long)
        for i, t in enumerate(toks):
            ids[i, :len(t)] = torch.tensor(t)
            pos[i] = len(t) - 1                 # 读答案分布的位置
        m = len(rows)
        return dict(ids=ids.to(self.dev), pos=pos.to(self.dev), m=m,
                    vstar=torch.tensor([r[2] for r in rows]).to(self.dev),
                    truth=torch.tensor([r[3] for r in rows]).to(self.dev))


def pair_delta(model: LM, b: dict, val_lo: int, n_val: int,
               want_mass: bool = False):
    """一个小批的逐文档 Δ（可求导）。

    Δ = [logit(v*) - logit(truth)]_edit - [同]_base。两项各自的 log-softmax
    归一化在相减时抵消，所以不必过 softmax —— 这既省一次 logsumexp，也避免
    在 512 类上做 log-softmax 引入的额外舍入。mass 需要真的归一化，只在
    诊断时算，且 detach。
    """
    logits, _ = model(b["ids"])
    row = logits[torch.arange(logits.shape[0], device=logits.device), b["pos"]]
    m = b["m"]
    vs, tr = b["vstar"], b["truth"]
    base = row[:m].gather(1, vs[:, None]).squeeze(1) - \
        row[:m].gather(1, tr[:, None]).squeeze(1)
    edit = row[m:].gather(1, vs[:, None]).squeeze(1) - \
        row[m:].gather(1, tr[:, None]).squeeze(1)
    delta = edit - base
    mass = None
    if want_mass:
        with torch.no_grad():
            p = torch.log_softmax(row[m:, val_lo:val_lo + n_val].float(), -1).exp()
            mass = (p.gather(1, (vs - val_lo)[:, None]).squeeze(1) +
                    p.gather(1, (tr - val_lo)[:, None]).squeeze(1))
    return delta, mass


# ---------------- 求值与求梯度 ----------------

@torch.no_grad()
def eval_loss(model: LM, batches) -> float:
    tot = sum(n for _, _, n in batches)
    s = 0.0
    for ids, lab, n in batches:
        _, loss = model(ids, lab)
        s += float(loss) * n / tot
    return s


@torch.no_grad()
def eval_delta(model: LM, pairs: Pairs, val_lo: int, n_val: int,
               tau: float, mass_floor: Optional[float] = None) -> dict:
    ds, ms = [], []
    for b in pairs.batches:
        d, mass = pair_delta(model, b, val_lo, n_val, want_mass=True)
        ds.append(d.float())
        ms.append(mass.float())
    d = torch.cat(ds)
    mass = torch.cat(ms)
    keep = mass >= mass_floor if mass_floor is not None else torch.ones_like(mass, dtype=torch.bool)
    dk = d[keep]
    return dict(mean=float(dk.mean()) if len(dk) else NAN,
                median=float(dk.median()) if len(dk) else NAN,
                frac_pos=float((dk > 0).float().mean()) if len(dk) else NAN,
                surrogate=float(torch.sigmoid(dk / tau).mean()) if len(dk) else NAN,
                mass_mean=float(mass.mean()), n=int(keep.sum()), n_all=len(d))


def grad_loss(model: LM, flat: Flat, batches) -> torch.Tensor:
    """全批 token 平均 loss 的梯度。小批按 token 数加权累加，等于大批一次求导。"""
    tot = sum(n for _, _, n in batches)
    flat.zero()
    for ids, lab, n in batches:
        _, loss = model(ids, lab)
        (loss * (n / tot)).backward()
    g = flat.grad()
    flat.zero()
    return g


def grad_delta(model: LM, flat: Flat, pairs: Pairs, val_lo: int, n_val: int,
               objective: str, tau: float) -> torch.Tensor:
    """objective='mean' 取 E[Δ]；'sign' 取 E[σ(Δ/τ)]，符号比例的可微替身。
    两个都要报：正文主 DV 是符号比例，但均值是更简单、更少自由量的对象。"""
    flat.zero()
    for b in pairs.batches:
        d, _ = pair_delta(model, b, val_lo, n_val)
        y = d if objective == "mean" else torch.sigmoid(d / tau)
        (y.sum() / pairs.n).backward()
    g = flat.grad()
    flat.zero()
    return g


# ---------------- 有限差分 ----------------

def fd_probe(model: LM, flat: Flat, theta: torch.Tensor, u: torch.Tensor,
             eps_list: Sequence[float], batches, pairs: Pairs,
             val_lo: int, n_val: int, tau: float, L0: float, D0: float) -> List[dict]:
    """沿单位方向 u 走 ±ε，同一批文档上读 L 与 Δ。

    中心差分给一阶（应等于 ∇·u，是对余弦的独立校验），二阶差分给曲率。
    每次求值都用同一个 batches / pairs，这是能看见 1e-5 量级 ΔL 的唯一办法。
    """
    rows = []
    for eps in eps_list:
        out = {}
        for sgn in (+1.0, -1.0):
            flat.set_(theta + sgn * eps * u)
            out[sgn] = (eval_loss(model, batches),
                        eval_delta(model, pairs, val_lo, n_val, tau)["mean"])
        flat.set_(theta)
        Lp, Dp = out[+1.0]
        Lm, Dm = out[-1.0]
        rows.append(dict(
            eps=eps,
            dL_plus=Lp - L0, dL_minus=Lm - L0,
            dD_plus=Dp - D0, dD_minus=Dm - D0,
            dL_central=(Lp - Lm) / (2 * eps),          # ≈ ∇L·u
            dD_central=(Dp - Dm) / (2 * eps),          # ≈ ∇Δ·u
            curv_L=(Lp - 2 * L0 + Lm) / (eps ** 2),    # ≈ uᵀHu
            curv_D=(Dp - 2 * D0 + Dm) / (eps ** 2),
            # 头条比值：走这一步换到多少 nats 的 Δ，每单位 loss 抬升。
            # 分母用两侧里较大的抬升，避免拿到恰好为负的一侧。
            nats_per_loss=(abs(Dp - D0) / max(Lp - L0, 1e-12)
                           if Lp - L0 > 0 else NAN),
        ))
    return rows


# ---------------- 单个 checkpoint ----------------

def measure(path: str, a, device) -> dict:
    model, spec, corpus, step = load_ckpt(path, device)
    vocab = Vocab(spec)
    vr = val_token_range(vocab, spec)
    val_lo, n_val = vr.start, len(vr)
    flat = Flat(model)

    batches = loss_batches(vocab, corpus, a.loss_docs, a.batch,
                           a.loss_offset, device)
    half = max(1, len(batches) // 2)
    pairs = Pairs(vocab, corpus, a.probe_docs, a.batch, a.probe_offset,
                  device, edit_seed=a.edit_seed)
    if pairs.n == 0:
        raise RuntimeError(f"{path}: break_rarity 在此配置下无域（R_old 太小？）")

    theta = flat.get()
    L0 = eval_loss(model, batches)
    d0 = eval_delta(model, pairs, val_lo, n_val, a.tau,
                    a.mass_gate if a.mass_gate > 0 else None)

    gL = grad_loss(model, flat, batches)
    gLa = grad_loss(model, flat, batches[:half])       # 噪声地板用的两半
    gLb = grad_loss(model, flat, batches[half:])
    gD = grad_delta(model, flat, pairs, val_lo, n_val, "mean", a.tau)
    gS = grad_delta(model, flat, pairs, val_lo, n_val, "sign", a.tau)

    torch.manual_seed(a.rand_seed)
    gR = torch.randn_like(gL)

    # g_Δ 剔除 g_L 分量：这才是"交换规则但不动损失"的方向
    gD_perp = reject(gD, gL)

    c = dict(
        L_D=cos(gL, gD), L_S=cos(gL, gS), L_R=cos(gL, gR),
        half_half=cos(gLa, gLb),                       # g_L 的信噪比
        L_D_core=cos(gL[flat.core], gD[flat.core]),
        L_S_core=cos(gL[flat.core], gS[flat.core]),
        D_S=cos(gD, gS),                               # 两个目标是否同向
        perp_frac=float(gD_perp.norm() / gD.norm()),    # g_Δ 有多少不在 g_L 上
    )
    # 去衰减：与噪声估计量的余弦被 sqrt(信噪比) 压低。
    # r = cos(g^A, g^B) 是同一真梯度两个独立估计的相关，corrected = raw/sqrt(r)。
    r = c["half_half"]
    c["L_D_corrected"] = c["L_D"] / math.sqrt(r) if r > 0 else NAN

    eps = [f * float(theta.norm()) for f in a.eps_frac]
    fd = dict(
        delta_perp=fd_probe(model, flat, theta, unit(gD_perp), eps, batches,
                            pairs, val_lo, n_val, a.tau, L0, d0["mean"]),
        delta_raw=fd_probe(model, flat, theta, unit(gD), eps, batches,
                           pairs, val_lo, n_val, a.tau, L0, d0["mean"]),
        loss_dir=fd_probe(model, flat, theta, unit(gL), eps, batches,
                          pairs, val_lo, n_val, a.tau, L0, d0["mean"]),
        random=fd_probe(model, flat, theta, unit(gR), eps, batches,
                        pairs, val_lo, n_val, a.tau, L0, d0["mean"]),
    )
    flat.set_(theta)                       # 还原，同进程连测多个 ckpt 时必须

    return dict(
        path=path, step=step, tag=os.path.basename(path)[:-3],
        r_old=corpus.r_old_hi, dd=[corpus.delta_d_lo, corpus.delta_d_hi],
        seed=corpus.seed, n_params=flat.n,
        loss=L0, readout=d0, pair_n=pairs.n, pair_yield=pairs.yield_rate,
        gnorm=dict(L=float(gL.norm()), D=float(gD.norm()), S=float(gS.norm())),
        theta_norm=float(theta.norm()),
        cos=c, eps=eps, fd=fd,
        chance_cos=1.0 / math.sqrt(flat.n),   # 26M 维下约 1.9e-4
    )


# ---------------- 报告 ----------------

def report(r: dict) -> None:
    c, ch = r["cos"], r["chance_cos"]
    print(f"\n=== {r['tag']}  step {r['step']}  "
          f"R_old={r['r_old']} ΔD~U{tuple(r['dd'])} seed={r['seed']}")
    print(f"  L={r['loss']:.6f}  Δ mean={r['readout']['mean']:+.3f} "
          f"median={r['readout']['median']:+.3f} "
          f"frac+={r['readout']['frac_pos']:.3f} "
          f"mass={r['readout']['mass_mean']:.3f}  "
          f"pairs={r['pair_n']} (yield {r['pair_yield']:.2f})")
    print(f"  |g_L|={r['gnorm']['L']:.4e}  |g_Δ|={r['gnorm']['D']:.4e}  "
          f"|θ|={r['theta_norm']:.1f}")
    print(f"  cos(g_L,g_Δ)      {c['L_D']:+.5f}   （非 emb {c['L_D_core']:+.5f}）")
    print(f"  cos(g_L,g_S)      {c['L_S']:+.5f}   （非 emb {c['L_S_core']:+.5f}）")
    print(f"  cos(g_L,随机)     {c['L_R']:+.5f}   偶然水平 ±{ch:.2e}")
    print(f"  cos(g_L^A,g_L^B)  {c['half_half']:+.5f}   ← g_L 的信噪比")
    print(f"  去衰减后 cos(g_L,g_Δ) {c['L_D_corrected']:+.5f}")
    print(f"  cos(g_Δ,g_S) {c['D_S']:+.4f}   g_Δ 落在 g_L 之外的比例 "
          f"{c['perp_frac']:.6f}")
    for name in ("delta_perp", "loss_dir", "random"):
        print(f"  -- 沿 {name}")
        print(f"     {'eps':>10} {'ΔL(+)':>12} {'ΔΔ(+)':>10} "
              f"{'∇L·u':>12} {'∇Δ·u':>10} {'uᵀH_Lu':>12} {'nats/loss':>11}")
        for x in r["fd"][name]:
            print(f"     {x['eps']:10.4f} {x['dL_plus']:+12.3e} "
                  f"{x['dD_plus']:+10.4f} {x['dL_central']:+12.3e} "
                  f"{x['dD_central']:+10.4f} {x['curv_L']:+12.3e} "
                  f"{x['nats_per_loss']:11.1f}")


def main():
    ap = argparse.ArgumentParser(
        description="∇θΔ 与 ∇θL 的角度、有限差分与曲率（论文 §5.6 的直接检验）")
    ap.add_argument("ckpt", nargs="+",
                    help="fp32 checkpoint 路径，可多个。用 train.py --ckpt-every 生成")
    ap.add_argument("--out", default="flatdir.jsonl", help="逐 ckpt 追加写入")
    ap.add_argument("--batch", type=int, default=32, help="前向小批（显存换精度无关）")
    ap.add_argument("--loss-docs", type=int, default=512,
                    help="估 g_L 的文档数。越大噪声地板越低，代价线性")
    ap.add_argument("--probe-docs", type=int, default=400,
                    help="估 g_Δ 的文档数，与终态读数同量级；实际配对数看 yield")
    ap.add_argument("--loss-offset", type=int, default=7000,
                    help="训练分布的新鲜样本。避开评估集(1)与训练流(1000+w)")
    ap.add_argument("--probe-offset", type=int, default=1,
                    help="默认与 go_nogo 的读数用同一批文档，便于对齐")
    ap.add_argument("--edit-seed", type=int, default=0, help="与 go_nogo 一致")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="σ(Δ/τ) 的温度，符号比例代理的唯一自由量")
    ap.add_argument("--mass-gate", type=float, default=0.0,
                    help=">0 时读数按 mass 硬筛（仅诊断；梯度目标始终不筛）")
    ap.add_argument("--eps-frac", type=float, nargs="+",
                    default=[1e-4, 3e-4, 1e-3, 3e-3],
                    help="FD 步长，按 |θ| 的比例给。四个量级用于查线性区")
    ap.add_argument("--rand-seed", type=int, default=1234)
    a = ap.parse_args()

    strict_fp32()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  fp32 严格模式（TF32 已关）")

    with open(a.out, "a") as f:
        for p in a.ckpt:
            t0 = time.time()
            r = measure(p, a, device)
            r["seconds"] = time.time() - t0
            report(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()