# -*- coding: utf-8 -*-
"""flatctrl.py — 平坦方向的正对照。

flatdir.py 测 break_rarity 读数的梯度与 g_L 的几何，结论是"平"。缺的是
正对照：同一流程、同一 checkpoint、另一条读数，且那条读数交易的规则对是
目标函数区分的。若它也平，"平"是流程的性质；若它不平，"平"追踪的就是
"目标函数是否区分这一对"。

本脚本把 flatdir 的测量套在 probe.EDITS 的每条编辑上，逐 checkpoint 输出
(目标规则, 碰撞率, mass, 几何量) 的对照表。

判读
  break_rarity  目标 rarity，碰撞率 1.000（Eq.1）。预期 perp_frac≈1、
                cos≈偶然水平、nats 比值 1e3–1e5。
  drop_freq     目标 frequency，域与 break_rarity 相同。定向方向朝 frequency
                型 = 在 base（训练）文档上更偏好 v_old = 抬高训练损失，
                故预期 perp_frac 明显 <1、nats 比值 ~1。
  其余          primacy / position / last_update_global，碰撞率各异。
                last_update_global 的碰撞率随 ΔD 变，可做同规则内剂量反应。

不能建立什么
  break_rarity 的编辑后文档在训练分布外（答案值重复，probe_selfcheck 对它
  单独放宽断言），对照编辑的编辑后文档在分布内。本对照排除的是"任何结构化
  方向在 26M 维里都显得平"。它不单独排除"平来自 OOD 而非 alias" —— 但
  编辑必须 OOD 才能分开共延规则，这正是 Eq.1 的内容，OOD 性是 alias 的
  后果而非独立因素。附录须照抄这一句。
"""
import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import torch

from config import CorpusCfg, LangSpec
from generator import generate_corpus
from probe import (EDITS, apply_edit, fit_position_offset, identifiability,
                   r_last_value)
from train import val_token_range
from vocab import Vocab

# 全部机器复用 flatdir，保证与论文已有数字同一实现
from flatdir import (NAN, Flat, cos, eval_loss, grad_loss, load_ckpt,
                     loss_batches, pair_delta, reject, strict_fp32, unit)

# 默认跑哪些编辑。bump_freq 的域是 R_real==1，与其余编辑互补但在高 R_old
# 列基本为空；late_update 的域是尾部无 update，随 ΔD 上升趋零。两者都留在
# 默认列表里，域太小时自动跳过并记录原因。
DEFAULT_EDITS = ["break_rarity", "drop_freq", "relabel_old", "shift_delta",
                 "clear_late_update", "late_update", "bump_freq"]


# ---------------- 配对输入（flatdir.Pairs + kind） ----------------

class PairsK:
    """probe.EDITS[kind] 的配对输入。

    _pack 的行布局必须与 flatdir.Pairs 一致（前 m 行 base、后 m 行 edit），
    pair_delta 按这个布局切片。sign 取自 EDITS，用于把读数定向到"目标规则型
    为正"，使不同编辑的几何量可比。
    """

    def __init__(self, kind: str, vocab: Vocab, corpus: CorpusCfg, n_docs: int,
                 bs: int, seed_offset: int, device, edit_seed: int = 0):
        self.kind, self.vocab, self.dev, self.bs = kind, vocab, device, bs
        self.target, self.sign = EDITS[kind][1], EDITS[kind][2]
        docs = list(generate_corpus(vocab, corpus, n_docs,
                                    seed_offset=seed_offset))
        offset = fit_position_offset(docs)
        rng = random.Random(edit_seed)          # 与 go_nogo / flatdir 同约定
        rows = []
        for d in docs:
            ed = apply_edit(d, kind, vocab, corpus, rng, offset)
            if ed is None:
                continue
            truth = r_last_value(d)
            rows.append((d.tokens[:d.answer_pos],
                         ed.tokens[:ed.answer_pos],
                         vocab.val(ed.v_star),
                         vocab.val(truth)))
        self.n = len(rows)
        self.n_docs = len(docs)
        self.yield_rate = self.n / len(docs) if docs else NAN
        self.batches = [self._pack(rows[i:i + bs])
                        for i in range(0, self.n, bs)]

    def _pack(self, rows):
        toks = [r[0] for r in rows] + [r[1] for r in rows]
        L = max(len(t) for t in toks)
        ids = torch.full((len(toks), L), self.vocab.PAD, dtype=torch.long)
        pos = torch.empty(len(toks), dtype=torch.long)
        for i, t in enumerate(toks):
            ids[i, :len(t)] = torch.tensor(t)
            pos[i] = len(t) - 1
        m = len(rows)
        return dict(ids=ids.to(self.dev), pos=pos.to(self.dev), m=m,
                    vstar=torch.tensor([r[2] for r in rows]).to(self.dev),
                    truth=torch.tensor([r[3] for r in rows]).to(self.dev))


# ---------------- 定向读数与梯度 ----------------

@torch.no_grad()
def eval_delta_k(model, pairs: PairsK, val_lo: int, n_val: int, tau: float,
                 mass_floor: Optional[float] = None) -> dict:
    """定向读数。mean_raw 供与 flatdir 对齐，其余量按 sign 定向。"""
    ds, ms = [], []
    for b in pairs.batches:
        d, mass = pair_delta(model, b, val_lo, n_val, want_mass=True)
        ds.append(d.float())
        ms.append(mass.float())
    d = torch.cat(ds)
    mass = torch.cat(ms)
    keep = (mass >= mass_floor if mass_floor is not None
            else torch.ones_like(mass, dtype=torch.bool))
    dk, s = d[keep], float(pairs.sign)
    if not len(dk):
        return dict(mean_raw=NAN, mean=NAN, median=NAN, frac_expected=NAN,
                    surrogate=NAN, mass_mean=float(mass.mean()), n=0,
                    n_all=len(d))
    return dict(mean_raw=float(dk.mean()),
                mean=float((s * dk).mean()),
                median=float((s * dk).median()),
                frac_expected=float((s * dk > 0).float().mean()),
                surrogate=float(torch.sigmoid(s * dk / tau).mean()),
                mass_mean=float(mass.mean()),
                n=int(keep.sum()), n_all=len(d))


def grad_delta_k(model, flat: Flat, pairs: PairsK, val_lo: int, n_val: int,
                 objective: str, tau: float) -> torch.Tensor:
    """定向后的 g_Δ：正方向恒指"更像目标规则"。

    不定向的话不同编辑的 cos 与 nats 比值符号不可比 —— 曲率与 perp_frac
    对方向取反不变，但一阶量会翻号。
    """
    s = float(pairs.sign)
    flat.zero()
    for b in pairs.batches:
        d, _ = pair_delta(model, b, val_lo, n_val)
        ds = s * d
        y = ds if objective == "mean" else torch.sigmoid(ds / tau)
        (y.sum() / pairs.n).backward()
    g = flat.grad()
    flat.zero()
    return g


def fd_probe_k(model, flat: Flat, theta, u, eps_list, batches, pairs: PairsK,
               val_lo: int, n_val: int, tau: float, L0: float,
               D0: float) -> List[dict]:
    """沿 u 走 ±ε，同一批文档上读 L 与该编辑的定向 Δ。"""
    rows = []
    for eps in eps_list:
        out = {}
        for sgn in (+1.0, -1.0):
            flat.set_(theta + sgn * eps * u)
            out[sgn] = (eval_loss(model, batches),
                        eval_delta_k(model, pairs, val_lo, n_val, tau)["mean"])
        flat.set_(theta)
        Lp, Dp = out[+1.0]
        Lm, Dm = out[-1.0]
        rows.append(dict(
            eps=eps,
            dL_plus=Lp - L0, dL_minus=Lm - L0,
            dD_plus=Dp - D0, dD_minus=Dm - D0,
            dL_central=(Lp - Lm) / (2 * eps),
            dD_central=(Dp - Dm) / (2 * eps),
            curv_L=(Lp - 2 * L0 + Lm) / (eps ** 2),
            curv_D=(Dp - 2 * D0 + Dm) / (eps ** 2),
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
    theta = flat.get()
    L0 = eval_loss(model, batches)

    gL = grad_loss(model, flat, batches)
    gLa = grad_loss(model, flat, batches[:half])
    gLb = grad_loss(model, flat, batches[half:])
    half_half = cos(gLa, gLb)
    torch.manual_seed(a.rand_seed)
    gR = torch.randn_like(gL)
    eps = [f * float(theta.norm()) for f in a.eps_frac]

    # 碰撞率：与探针同一批文档，供对照表的横轴用
    docs = list(generate_corpus(vocab, corpus, a.collide_docs,
                                seed_offset=a.probe_offset))
    ident = identifiability(docs, fit_position_offset(docs))

    # 随机方向的曲率只依赖 gR，所有编辑共用一次
    rand_fd = None

    rows: Dict[str, dict] = {}
    for kind in a.edits:
        pairs = PairsK(kind, vocab, corpus, a.probe_docs, a.batch,
                       a.probe_offset, device, a.edit_seed)
        base = dict(kind=kind, target=pairs.target, sign=pairs.sign,
                    n=pairs.n, yield_rate=pairs.yield_rate,
                    collide=ident.get(f"last_value|{pairs.target}", NAN))
        if pairs.n < a.min_pairs:
            rows[kind] = dict(base, skipped=f"域太小 n={pairs.n}")
            continue

        d0 = eval_delta_k(model, pairs, val_lo, n_val, a.tau,
                          a.mass_gate if a.mass_gate > 0 else None)
        gD = grad_delta_k(model, flat, pairs, val_lo, n_val, "mean", a.tau)
        gS = grad_delta_k(model, flat, pairs, val_lo, n_val, "sign", a.tau)
        gDp = reject(gD, gL)

        c = dict(
            L_D=cos(gL, gD), L_S=cos(gL, gS),
            L_D_core=cos(gL[flat.core], gD[flat.core]),
            D_S=cos(gD, gS),
            perp_frac=float(gDp.norm() / gD.norm()) if gD.norm() > 0 else NAN,
        )
        # 主读数：正交化削掉的比例。1 - perp_frac。无有限差分，不受步长影响。
        c["removed_frac"] = 1.0 - c["perp_frac"] if c["perp_frac"] == c["perp_frac"] else NAN
        c["L_D_corrected"] = (c["L_D"] / math.sqrt(half_half)
                              if half_half > 0 else NAN)

        fd = dict(
            delta_perp=fd_probe_k(model, flat, theta, unit(gDp), eps, batches,
                                  pairs, val_lo, n_val, a.tau, L0, d0["mean"]),
            loss_dir=fd_probe_k(model, flat, theta, unit(gL), eps, batches,
                                pairs, val_lo, n_val, a.tau, L0, d0["mean"]),
        )
        if a.with_raw:
            fd["delta_raw"] = fd_probe_k(model, flat, theta, unit(gD), eps,
                                         batches, pairs, val_lo, n_val, a.tau,
                                         L0, d0["mean"])
        if rand_fd is None:
            rand_fd = fd_probe_k(model, flat, theta, unit(gR), eps, batches,
                                 pairs, val_lo, n_val, a.tau, L0, d0["mean"])
        flat.set_(theta)

        rows[kind] = dict(base, readout=d0, cos=c, fd=fd,
                          gnorm=dict(D=float(gD.norm()), S=float(gS.norm())))

    flat.set_(theta)
    return dict(path=path, step=step, tag=os.path.basename(path)[:-3],
                r_old=corpus.r_old_hi, dd=[corpus.delta_d_lo, corpus.delta_d_hi],
                seed=corpus.seed, n_params=flat.n, loss=L0,
                gnorm_L=float(gL.norm()), theta_norm=float(theta.norm()),
                half_half=half_half, chance_cos=1.0 / math.sqrt(flat.n),
                eps=eps, rand_fd=rand_fd, edits=rows)


# ---------------- 报告 ----------------

def _ratio(fd_rows, key_num, key_den, i=0):
    """同一 ε 下两条方向的比值。i 索引 eps_list。"""
    try:
        a, b = fd_rows[key_num][i], fd_rows[key_den][i]
    except (KeyError, IndexError):
        return NAN
    return a / b if b not in (0, None) and b == b else NAN


def report(r: dict) -> None:
    print(f"\n=== {r['tag']}  step {r['step']}  R_old={r['r_old']} "
          f"ΔD~U{tuple(r['dd'])} seed={r['seed']}")
    print(f"  L={r['loss']:.6f}  |g_L|={r['gnorm_L']:.4e}  "
          f"|θ|={r['theta_norm']:.1f}  cos(g_L^A,g_L^B)={r['half_half']:+.4f}  "
          f"偶然水平 ±{r['chance_cos']:.2e}")
    print(f"  ε={r['eps'][0]:.4f}"
          + (f" (+{len(r['eps'])-1} 个)" if len(r['eps']) > 1 else ""))

    hdr = (f"  {'编辑':<18}{'目标规则':<20}{'碰撞':>7}{'n':>5}{'mass':>6}"
           f"{'frac':>6}{'|g_Δ|':>10}{'cos':>10}{'剔除':>8}"
           f"{'nats⊥/L':>11}{'curv⊥/L':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for kind, e in r["edits"].items():
        if "skipped" in e:
            print(f"  {kind:<18}{e['target']:<20}{e['collide']:>7.3f}"
                  f"{e['n']:>5}    —— {e['skipped']}")
            continue
        fd = e["fd"]
        nats = _ratio({k: [x["nats_per_loss"] for x in v] for k, v in fd.items()},
                      "delta_perp", "loss_dir")
        curv = _ratio({k: [x["curv_L"] for x in v] for k, v in fd.items()},
                      "delta_perp", "loss_dir")
        print(f"  {kind:<18}{e['target']:<20}{e['collide']:>7.3f}"
              f"{e['n']:>5}{e['readout']['mass_mean']:>6.2f}"
              f"{e['readout']['frac_expected']:>6.2f}"
              f"{e['gnorm']['D']:>10.2e}{e['cos']['L_D']:>+10.5f}"
              f"{e['cos']['removed_frac']:>8.4f}"
              f"{nats:>11.2e}{curv:>10.3f}")

    # 碰撞率 vs 平坦度的秩相关。编辑数少，只作方向性参考。
    pts = [(e["collide"], e["cos"]["removed_frac"])
           for e in r["edits"].values()
           if "skipped" not in e and e["collide"] == e["collide"]]
    if len(pts) >= 4:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        def rk(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            out = [0.0] * len(v)
            i = 0
            while i < len(v):
                j = i
                while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                    j += 1
                for k in range(i, j + 1):
                    out[o[k]] = (i + j) / 2.0 + 1.0
                i = j + 1
            return out

        rx, ry = rk(xs), rk(ys)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
        den = (sum((p - mx) ** 2 for p in rx) ** 0.5
               * sum((q - my) ** 2 for q in ry) ** 0.5)
        print(f"\n  碰撞率 vs 剔除比例: Spearman rho = "
              f"{num/den if den else NAN:+.3f}  (n={len(pts)})")
        print("  预期为负：碰撞率越高（越不可分）=> g_Δ 落在 g_L 上的越少")


def main():
    ap = argparse.ArgumentParser(
        description="平坦方向的正对照：逐编辑测 g_Δ 与 g_L 的几何")
    ap.add_argument("ckpt", nargs="+", help="fp32 checkpoint，用 flatdir 那批")
    ap.add_argument("--out", default="flatctrl.jsonl")
    ap.add_argument("--edits", nargs="+", default=DEFAULT_EDITS,
                    help=f"默认 {DEFAULT_EDITS}")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--loss-docs", type=int, default=512, help="与 flatdir 同")
    ap.add_argument("--probe-docs", type=int, default=400, help="与 flatdir 同")
    ap.add_argument("--collide-docs", type=int, default=1500,
                    help="算碰撞率的文档数，与 tab:collide 同量级")
    ap.add_argument("--loss-offset", type=int, default=7000)
    ap.add_argument("--probe-offset", type=int, default=1)
    ap.add_argument("--edit-seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--mass-gate", type=float, default=0.0)
    ap.add_argument("--min-pairs", type=int, default=40,
                    help="配对少于此数不出几何量（域太小）")
    ap.add_argument("--eps-frac", type=float, nargs="+", default=[3e-4],
                    help="默认只跑 app:flat 认定可用的那一个步长")
    ap.add_argument("--with-raw", action="store_true",
                    help="额外沿未剔除的 g_Δ 走一遍")
    ap.add_argument("--rand-seed", type=int, default=1234)
    a = ap.parse_args()

    strict_fp32()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  fp32 严格模式（TF32 已关）")
    print(f"编辑: {' '.join(a.edits)}")

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