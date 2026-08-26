r"""深度臂的 loss 导数峰与格内 sd。只吃训练 jsonl，不需要 .pt。

paper_numbers.py 按 R{r}_D{dd}_s{s}_grid.jsonl 匹配，认不出 _L4/_L12，
所以单独一份。deriv / escape_peak / read_run 与 paper_numbers.py 逐行相同，
否则两边的峰位不可比。

用法:
  python depth_peaks.py --dir ../runs_depth
  python depth_peaks.py --dir ../runs_depth --tail-frac 0.8
"""
import argparse
import glob
import json
import math
import os

import numpy as np

COPY_FLOOR = 0.95


def read_run(path):
    """训练 jsonl -> (loss 序列, 探针序列, 门判逃逸步)。"""
    loss, probes, esc = [], [], None
    with open(path) as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = o.get("kind")
            if k == "train" and "loss" in o:
                loss.append((o["step"], o["loss"]))
            elif k == "eval":
                if esc is None and o.get("copy_acc", 0) >= COPY_FLOOR:
                    esc = o["step"]
            elif k == "probe":
                c = (o.get("causal") or {}).get("break_rarity")
                if c and c.get("frac_expected") is not None:
                    probes.append((o["step"], c["frac_expected"]))
    loss.sort()
    probes.sort()
    return loss, probes, esc


def deriv(loss):
    """d loss / d log step，中心差分。"""
    out = []
    for i in range(1, len(loss) - 1):
        s0, l0 = loss[i - 1]
        s1, _ = loss[i]
        s2, l2 = loss[i + 1]
        if s0 <= 0 or s2 <= 0:
            continue
        out.append((s1, (l0 - l2) / (math.log(s2) - math.log(s0))))
    return out


def escape_peak(loss, tail_frac=0.8):
    """内部极大值，排除余弦末段衰减。"""
    d = deriv(loss)
    if not d:
        return None
    cut = loss[-1][0] * tail_frac
    head = [t for t in d if t[0] <= cut]
    tail = [t for t in d if t[0] > cut]
    if not head:
        return None
    pk = max(head, key=lambda t: t[1])
    tl = max(tail, key=lambda t: t[1]) if tail else (None, float("nan"))
    return {"peak_step": pk[0], "peak_h": pk[1],
            "tail_step": tl[0], "tail_h": tl[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../runs_depth")
    ap.add_argument("--tail-frac", type=float, default=0.8)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.dir, "*.jsonl")))
    if not paths:
        ap.error(f"{a.dir} 里没有 jsonl")

    rows = []
    for p in paths:
        tag = os.path.basename(p)[:-6]
        loss, probes, esc = read_run(p)
        pk = escape_peak(loss, a.tail_frac)
        post = [f for s, f in probes if esc and s > esc]
        rows.append(dict(
            tag=tag, esc=esc, n_loss=len(loss), n_probe=len(probes),
            peak=pk["peak_step"] if pk else None,
            h=pk["peak_h"] if pk else float("nan"),
            tail_h=pk["tail_h"] if pk else float("nan"),
            tail_step=pk["tail_step"] if pk else None,
            n_post=len(post),
            sd=float(np.std(post)) if len(post) >= 5 else float("nan"),
            lo=min(post) if post else float("nan"),
            hi=max(post) if post else float("nan")))

    print(f"# {len(rows)} runs from {a.dir}  (tail_frac={a.tail_frac})\n")
    print(f"{'tag':<16}{'gate':>6}{'peak':>7}{'h':>7}{'tailh':>7}"
          f"{'flag':>6}{'nPost':>6}{'sd':>7}{'min':>6}{'max':>6}")
    for r in sorted(rows, key=lambda x: x["tag"]):
        # tail_h > peak_h：余弦末段衰减盖过逃逸峰，该峰不可用（tab:depthpeak 的 †）
        flag = "DAG" if r["tail_h"] > r["h"] else ""
        print(f"{r['tag']:<16}{(r['esc'] or 0):>6}{(r['peak'] or 0):>7}"
              f"{r['h']:>7.2f}{r['tail_h']:>7.2f}{flag:>6}"
              f"{r['n_post']:>6}{r['sd']:>7.3f}{r['lo']:>6.2f}{r['hi']:>6.2f}")

    print("\n--- tab:depthpeak 用（† = DAG 行）---")
    for L in (4, 8, 12):
        for R in (3, 8, 16):
            got = {}
            for r in rows:
                if f"R{R}_" in r["tag"] and f"_L{L}" in r["tag"]:
                    s = 0 if "_s0_" in r["tag"] else 1
                    got[s] = r
            if got:
                cs = []
                for s in (0, 1):
                    v = got.get(s)
                    cs.append(f"{v['peak']}{'$^dagger$' if v['tail_h'] > v['h'] else ''}"
                              if v else "---")
                print(f"{L:>3} & {R:>3} & {cs[0]:>16} & {cs[1]:>16} \\\\")

    sds = [r["sd"] for r in rows if r["sd"] == r["sd"]]
    if sds:
        print(f"\n--- App:drift 的深度臂区间 ---")
        print(f"runs with >= 5 post-formation probes : {len(sds)} of {len(rows)}")
        print(f"within-run sd range                 : "
              f"{min(sds):.3f} to {max(sds):.3f}")
        print(f"median                              : {np.median(sds):.3f}")
        wk = max(rows, key=lambda r: r["sd"] if r["sd"] == r["sd"] else -1)
        print(f"largest at                          : {wk['tag']}")
        print(f"  ^ 论文写 '0.007--0.332 over its twelve runs, largest at four layers'")


if __name__ == "__main__":
    main()
