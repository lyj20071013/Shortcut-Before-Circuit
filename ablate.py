"""Single-head ablations in the aliased main grid."""
import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import go_nogo as gn
from generator import generate_corpus
from interp import build, load_pair, make_pairs, readout
from model import LM
from train import copy_diag, evaluate
from vocab import Vocab

NAN = float("nan")
D_FRAC = 0.20      # [A] 的读出移动门限
ACC_FLOOR = 0.99   # [A] 的任务保持门限
COPY_FLOOR = 0.95

class HeadZero:
    """在 blk.attn.out 的输入上把 head h 的通道段置零。

    forward pre-hook 收到的是传给 out 的那一个位置参数，形状 (B,T,C)，
    其中 C = n_head * d_head 且 head h 占 [h*d_head, (h+1)*d_head)。
    返回一个新 tuple 即替换该参数；不写回原张量，避免影响 autograd 图和
    同一 batch 内其他 hook。
    """

    def __init__(self, model: LM, layer: int, head: int):
        self.d = model.cfg.d_head
        self.h = head
        self.mod = model.blocks[layer].attn.out
        self.handle = None

    def __enter__(self):
        lo, hi = self.h * self.d, (self.h + 1) * self.d

        def pre(_m, args):
            x = args[0].clone()
            x[..., lo:hi] = 0.0
            return (x,) + tuple(args[1:])

        self.handle = self.mod.register_forward_pre_hook(pre)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        return False


@torch.no_grad()
def sanity(model: LM, cfg, device) -> None:
    """hook 真的只动一个 head 吗。

    两条断言。全部 head 依次 ablate 后 out 的输入应当逐通道被覆盖一次，
    所以对同一输入，n_head 次单 head ablation 的差异之和等于把整个输入置零
    的差异 —— 这里用更弱但足够的版本：ablate head h 只改变 [h*d, (h+1)*d)
    这一段，其余不变。零成本，防的是 d_head 或 reshape 顺序变了而 hook 没跟上。
    """
    B, T, C = 2, 8, cfg.d_model
    x = torch.randn(B, T, C, device=device)
    ref = model.blocks[0].attn.out(x).clone()
    d = cfg.d_head
    with HeadZero(model, 0, 1):
        got = model.blocks[0].attn.out(x)
    # 用 head 1 被置零的输入直接算一遍，应当逐元素相等
    x2 = x.clone()
    x2[..., d:2 * d] = 0.0
    exp = model.blocks[0].attn.out(x2)
    assert torch.allclose(got, exp, atol=0, rtol=0), \
        "hook 置零的通道段与 head 1 不符：d_head 或 reshape 顺序变了"
    assert not torch.allclose(got, ref), "hook 没有生效"


@torch.no_grad()
def one_head(model, layer, head, pairs, ev_docs, cp_docs, vocab, spec,
             corpus, device, base) -> dict:
    """单个 head 的 ablation 读数 + 任务保持检查。"""
    with HeadZero(model, layer, head):
        r = readout(model, pairs, vocab, spec, device)
        ev = evaluate(model, ev_docs, vocab, spec, device)
        cd = copy_diag(model, cp_docs, vocab, spec, corpus, device)
        model.eval()
    d_frac = r["frac_positive"] - base["frac_positive"]
    keeps = (ev["acc"] >= ACC_FLOOR) and (cd["copy_acc"] >= COPY_FLOOR)
    return dict(layer=layer, head=head,
                frac=r["frac_positive"], d_frac=d_frac,
                median=r["d_median"], mass=r["mass"],
                acc=ev["acc"], copy=cd["copy_acc"], keeps=keeps,
                carries=bool(abs(d_frac) >= D_FRAC and keeps))


def scan(state, cfg, spec, corpus, pairs, ev_docs, cp_docs, vocab,
         device, tag: str) -> dict:
    m = build(state, cfg, device)
    sanity(m, cfg, device)
    base = readout(m, pairs, vocab, spec, device)
    ev = evaluate(m, ev_docs, vocab, spec, device)
    cd = copy_diag(m, cp_docs, vocab, spec, corpus, device)
    m.eval()
    print(f"[{tag}] 未 ablate: frac+={base['frac_positive']:.3f} "
          f"med={base['d_median']:+.3f} acc={ev['acc']:.4f} "
          f"copy={cd['copy_acc']:.3f}", flush=True)

    rows = []
    for L in range(cfg.n_layer):
        for h in range(cfg.n_head):
            rows.append(one_head(m, L, h, pairs, ev_docs, cp_docs, vocab,
                                 spec, corpus, device, base))
        got = [r for r in rows if r["layer"] == L and r["carries"]]
        mx = max((abs(r["d_frac"]) for r in rows if r["layer"] == L),
                 default=0.0)
        print(f"  L{L}: max|dfrac|={mx:.3f}  carriers={len(got)}", flush=True)
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(tag=tag, base=base["frac_positive"],
                base_median=base["d_median"], rows=rows)


def summarize(A: dict, B: dict, n_layer: int, n_head: int) -> str:
    out = ["", "=" * 70]

    def carriers(S):
        return sorted((r for r in S["rows"] if r["carries"]),
                      key=lambda r: -abs(r["d_frac"]))

    cA, cB = carriers(A), carriers(B)
    for S, c in ((A, cA), (B, cB)):
        out.append(f"{S['tag']}: base frac+={S['base']:.3f}, "
                   f"{len(c)} 个承载 head"
                   + (f"，最强 L{c[0]['layer']}H{c[0]['head']} "
                      f"dfrac={c[0]['d_frac']:+.3f}" if c else ""))
        # 层级剖面：每层最大 |dfrac|，与置换无关
        prof = []
        for L in range(n_layer):
            mx = max((abs(r["d_frac"]) for r in S["rows"] if r["layer"] == L),
                     default=0.0)
            prof.append(f"{mx:.2f}")
        out.append(f"  每层 max|dfrac|: " + " ".join(prof))

    out += ["", "预注册判据"]
    a_ok = bool(cA) and bool(cB)
    out.append(f"  [A] 定域（两个 run 都有 |dfrac|>={D_FRAC} 且保任务的 head）"
               f": {'成立' if a_ok else '不成立'}")
    if a_ok:
        lA, lB = cA[0]["layer"], cB[0]["layer"]
        b_ok = (lA == lB)
        c_ok = abs(len(cA) - len(cB)) <= 2
        out.append(f"  [B] 层一致（最强承载 head 同层）: "
                   f"L{lA} vs L{lB} -> {'成立' if b_ok else '不成立'}")
        out.append(f"  [C] 稀疏一致（承载 head 个数差 <=2）: "
                   f"{len(cA)} vs {len(cB)} -> {'成立' if c_ok else '不成立'}")
        # 索引重叠只作附带信息，不进判据：两个 run 在不同置换胞腔里
        iA = {(r["layer"], r["head"]) for r in cA}
        iB = {(r["layer"], r["head"]) for r in cB}
        out.append(f"  附带：索引交集 {len(iA & iB)} / 并集 {len(iA | iB)}"
                   f"（不可解释，两 run 置换不同）")
        out += ["", "读法："]
        if b_ok:
            out += ["  Both runs have qualifying heads in the same layer; "
                    "head-index correspondence is not established across runs.",
                    "  This localization result does not identify a unique circuit."]
        else:
            out += ["  The strongest qualifying heads occur in different layers. ",
                    "  Report localization separately for each run; "
                    "cross-run correspondence requires additional evidence."]
    else:
        out += ["", "读法：",
                "  No head meets the chosen localization criterion. ",
                "  This does not establish that the readout is intrinsically diffuse.",
                f"  报告每层 max|dfrac| 的剖面作为弥散程度的证据，并检查",
                f"  是否只是门限 {D_FRAC} 偏严（下调到 0.1 再看一次，"
                f"但要声明那是事后阈值）。"]
    out.append("=" * 70)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_a", help="如 R3_D8_s0_grid")
    ap.add_argument("tag_b", help="如 R3_D8_s2_grid")
    ap.add_argument("--out", default="runs_g2")
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--eval-docs", type=int, default=512,
                    help="ablation 要跑 n_layer*n_head 次，比 interp 小一档")
    ap.add_argument("--copy-docs", type=int, default=128)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False

    sa, sb, cfg, spec, corpus, cor_b, info = load_pair(
        a.tag_a, a.tag_b, a.out, device)
    print(f"A {info['tag_a']}  B {info['tag_b']}  "
          f"{cfg.n_layer} 层 x {cfg.n_head} head = "
          f"{cfg.n_layer * cfg.n_head} 次 ablation x 2 run")

    vocab = Vocab(spec)
    # 与 interp.py 同一条读数路径：A 的 corpus 生成一批，两个 run 共用。
    pairs, docs = make_pairs(vocab, corpus, a.docs)
    ev_docs = list(generate_corpus(vocab, corpus, a.eval_docs, seed_offset=1))
    cp_docs = ev_docs[:a.copy_docs]
    print(f"编辑对 {len(pairs)}，eval {len(ev_docs)}，copy {len(cp_docs)}\n")

    A = scan(sa, cfg, spec, corpus, pairs, ev_docs, cp_docs, vocab, device,
             a.tag_a)
    B = scan(sb, cfg, spec, corpus, pairs, ev_docs, cp_docs, vocab, device,
             a.tag_b)
    print(summarize(A, B, cfg.n_layer, cfg.n_head))

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(info=info, n_pairs=len(pairs),
                           thresholds=dict(d_frac=D_FRAC, acc=ACC_FLOOR,
                                           copy=COPY_FLOOR),
                           A=A, B=B), f, indent=1)
        print(f"\n已写入 {a.json}")


if __name__ == "__main__":
    main()
