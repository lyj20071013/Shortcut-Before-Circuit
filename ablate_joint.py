"""Joint-head ablations and accuracy/copy diagnostics."""
import argparse
import json
from contextlib import ExitStack
from itertools import combinations
from typing import List, Sequence, Tuple

import torch

from ablate import ACC_FLOOR, COPY_FLOOR, D_FRAC, HeadZero, sanity
from generator import generate_corpus
from interp import build, load_pair, make_pairs, readout
from model import LM
from train import copy_diag, evaluate
from vocab import Vocab

def parse_heads(specs: Sequence[str]) -> List[Tuple[int, int]]:
    out = []
    for s in specs:
        a, b = s.split(",")
        out.append((int(a), int(b)))
    return out


@torch.no_grad()
def measure(model: LM, heads: Sequence[Tuple[int, int]], pairs, ev_docs,
            cp_docs, vocab, spec, corpus, device) -> dict:
    """同时 ablate heads 里的每一个，读一次。

    多个 pre-hook 可以共存：PyTorch 按注册顺序调用，每个收到前一个返回的
    args，所以链式生效。这里的 head 落在不同层，彼此不冲突。ExitStack
    保证任一异常都把全部 hook 摘掉。
    """
    with ExitStack() as st:
        for (L, h) in heads:
            st.enter_context(HeadZero(model, L, h))
        r = readout(model, pairs, vocab, spec, device)
        ev = evaluate(model, ev_docs, vocab, spec, device)
        cd = copy_diag(model, cp_docs, vocab, spec, corpus, device)
        model.eval()
    return dict(heads=[list(x) for x in heads],
                frac=r["frac_positive"], median=r["d_median"],
                mass=r["mass"], acc=ev["acc"], copy=cd["copy_acc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_a")
    ap.add_argument("tag_b")
    ap.add_argument("--out", default="runs_g2")
    ap.add_argument("--which", choices=["a", "b"], default="b",
                    help="在哪个 run 上做联合 ablation")
    ap.add_argument("--heads", nargs="+", required=True,
                    help='形如 1,0 2,5 —— layer,head')
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--eval-docs", type=int, default=512)
    ap.add_argument("--copy-docs", type=int, default=128)
    ap.add_argument("--acc-floor", type=float, default=ACC_FLOOR,
                    help="Changing this after inspecting results is a post-hoc threshold choice.")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False

    sa, sb, cfg, spec, corpus, cor_b, info = load_pair(
        a.tag_a, a.tag_b, a.out, device)
    state = sa if a.which == "a" else sb
    tag = a.tag_a if a.which == "a" else a.tag_b
    heads = parse_heads(a.heads)
    for (L, h) in heads:
        if not (0 <= L < cfg.n_layer and 0 <= h < cfg.n_head):
            raise SystemExit(f"L{L}H{h} 越界：{cfg.n_layer} 层 x {cfg.n_head} head")

    vocab = Vocab(spec)
    # 与 ablate.py 同一条路径：A 的 corpus 生成编辑对，两个 run 共用，
    # 所以这里的 base 与 ablate.py 的 base 可直接对读（与已发表值差一个
    # docMove，见 App C.2）。
    pairs, _ = make_pairs(vocab, corpus, a.docs)
    ev_docs = list(generate_corpus(vocab, corpus, a.eval_docs, seed_offset=1))
    cp_docs = ev_docs[:a.copy_docs]

    m = build(state, cfg, device)
    sanity(m, cfg, device)
    print(f"{tag}，编辑对 {len(pairs)}，"
          f"head {' '.join(f'L{L}H{h}' for L, h in heads)}\n")

    base = measure(m, [], pairs, ev_docs, cp_docs, vocab, spec, corpus, device)
    print(f"{'ablated':<22} {'frac+':>7} {'dfrac':>7} {'|d|':>6} "
          f"{'acc':>7} {'copy':>6} {'gate':>5}")
    print(f"{'(none)':<22} {base['frac']:>7.3f} {'':>7} {'':>6} "
          f"{base['acc']:>7.4f} {base['copy']:>6.3f}")

    rows = []
    singles = []
    for (L, h) in heads:
        r = measure(m, [(L, h)], pairs, ev_docs, cp_docs, vocab, spec,
                    corpus, device)
        r["d_frac"] = r["frac"] - base["frac"]
        r["keeps"] = (r["acc"] >= a.acc_floor) and (r["copy"] >= COPY_FLOOR)
        singles.append(r)
        rows.append(r)
        print(f"{f'L{L}H{h}':<22} {r['frac']:>7.3f} {r['d_frac']:>+7.3f} "
              f"{abs(r['d_frac']):>6.3f} {r['acc']:>7.4f} {r['copy']:>6.3f} "
              f"{'ok' if r['keeps'] else 'FAIL':>5}")

    joint = None
    if len(heads) > 1:
        joint = measure(m, heads, pairs, ev_docs, cp_docs, vocab, spec,
                        corpus, device)
        joint["d_frac"] = joint["frac"] - base["frac"]
        joint["keeps"] = ((joint["acc"] >= a.acc_floor)
                          and (joint["copy"] >= COPY_FLOOR))
        rows.append(joint)
        lbl = "+".join(f"L{L}H{h}" for L, h in heads)
        print(f"{lbl:<22} {joint['frac']:>7.3f} {joint['d_frac']:>+7.3f} "
              f"{abs(joint['d_frac']):>6.3f} {joint['acc']:>7.4f} "
              f"{joint['copy']:>6.3f} {'ok' if joint['keeps'] else 'FAIL':>5}")

        # 所有 dfrac 都指向 0.5，所以比较 |dfrac| 的可加性而非带符号的和
        s = sum(abs(r["d_frac"]) for r in singles)
        j = abs(joint["d_frac"])
        print(f"\n|joint| = {j:.3f}   sum|singles| = {s:.3f}   "
              f"比值 = {j / s if s else float('nan'):.3f}")
        # 尺度参照：该 cell 的 within-run sd 上界（App F，8000--16000 窗口）
        band = 0.098
        if s - j > band:
            print(f"  次可加，差 {s - j:.3f} 超过 within-run band {band} "
                  f"-> 冗余：同一条路径，互为备份。")
            print(f"  与 App I.1 在非等价臂上的结论同向（那里 0.986 对 1.561）。")
        elif j - s > band:
            print(f"  超可加，差 {j - s:.3f} 超过 band {band} -> 协同：")
            print(f"  单独移除任一个都被另一个补偿。")
        else:
            print(f"  |joint| 与 sum|singles| 的差在 band {band} 内 "
                  f"-> 可加，两条独立路径；或三个读数不足以分辨。")

        # frac+ 有界于 [0,1]，接近端点时和会被截断，需要提醒
        if base["frac"] > 0.9 or base["frac"] < 0.1:
            room = base["frac"] - 0.5 if base["frac"] > 0.5 else 0.5 - base["frac"]
            if s > room:
                print(f"  注意：base 距 0.5 只有 {room:.3f}，而 sum|singles| "
                      f"是 {s:.3f}。ablation 把读出推向 0.5，所以次可加性"
                      f"部分是这个上界造成的，不能全部读成冗余。")

    if a.acc_floor != ACC_FLOOR:
        print(f"\n注意：acc 门限已从预注册的 {ACC_FLOOR} 放宽到 {a.acc_floor}。"
              f"Report this as a post-hoc threshold choice.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(info=info, tag=tag, which=a.which,
                           acc_floor=a.acc_floor, copy_floor=COPY_FLOOR,
                           d_frac_floor=D_FRAC, base=base, rows=rows,
                           joint=joint), f, indent=1)
        print(f"\n已写入 {a.json}")


if __name__ == "__main__":
    main()
