"""Re-evaluate rendered-language checkpoints with mass-restricted summaries."""
import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch

# 主网格 R3_D8 在同一在线探针、同调度、同预算、step 16000 上的十个 seed
# （constlr10.json 的 paired.rows）。前三个是与 NL 臂可配对比较的子集。
GRID10 = [0.094, 0.448, 0.988, 0.750, 0.805, 0.524, 0.713, 0.317, 0.390, 0.768]
D3 = 1.693          # n=3 的期望极差系数
D10 = 3.078


def regate(pt: str, probe_path: str, dev: str) -> dict:
    """载入末态权重，在探针集上重算门控与未门控读数。"""
    from model import LM, ModelCfg
    from nl_train import Probe

    ck = torch.load(pt, map_location="cpu", weights_only=False)
    mc = ModelCfg(**ck["cfg"])
    m = LM(mc).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()

    pr = Probe(probe_path, dev)
    if pr.version != str(ck.get("version", "?")):
        raise SystemExit(
            f"版本不符：探针集 {pr.version}，checkpoint {ck.get('version')}。"
            f"adj_ids 顺序可能不同，读数会错位。")

    rd = pr.read(m)                      # 未门控（nl_train 的原样输出）
    cd = pr.copy_diag(m)

    # 门控。read 返回了 d_all，但没返回 per-document mass —— 所以这里重算
    # mass。这正是 mass_all 该被记进 jsonl 的理由（已在 ft_train 里加上）。
    d = np.array(rd["d_all"])
    mass = _mass(m, pr)
    ok = mass >= 0.5
    nan = float("nan")
    return dict(
        n=len(d), n_gated=int(ok.sum()),
        frac_gated=float(np.mean(d[ok] > 0)) if ok.any() else nan,
        d_median_gated=float(np.median(d[ok])) if ok.any() else nan,
        frac_ungated=float(np.mean(d > 0)),
        d_median_ungated=float(np.median(d)),
        mass_mean=float(np.mean(mass)), mass_ok=float(np.mean(ok)),
        acc=rd["acc_base"], copy=cd["copy_acc"], n_copy=cd["n_copy"])


@torch.no_grad()
def _mass(model, pr) -> np.ndarray:
    """per-document 的候选对质量，在**全词表**上算。

    全词表而非 adj 集内：probe._mass 的判据是"候选对是否占住概率质量"，若在
    adj 集内归一化，模型把 99% 质量放在非 adj token 上也看不出来，门就是空的。
    """
    lg = pr._lp(model, pr.edit, pr.eapos)      # 全词表 logits 行
    p = torch.softmax(lg, -1)
    r = torch.arange(len(pr.ans))
    return (p[r, pr.vst] + p[r, pr.ans]).numpy()


def sd(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mu = sum(xs) / n
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def binom_p(k: int, n: int) -> float:
    """双尾精确二项检验对 0.5。App.~fixband 用它判断端点是否 decisive。"""
    from math import comb
    if n == 0:
        return float("nan")
    obs = abs(k - n / 2)
    tot = sum(comb(n, i) for i in range(n + 1)
              if abs(i - n / 2) >= obs)
    return min(1.0, tot / (2 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs_nl")
    ap.add_argument("--data", default="nl_data")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    probe = os.path.join(a.data, f"probe_R{a.r}_D{a.d}.npz")
    pts = sorted(glob.glob(os.path.join(a.dir, f"R{a.r}_D{a.d}_s*_nl.pt")))
    if not pts:
        raise SystemExit(f"{a.dir} 下没有 .pt")

    rows = []
    for p in pts:
        m = re.search(r"_s(\d+)_", os.path.basename(p))
        seed = int(m.group(1)) if m else -1
        print(f"重算 {os.path.basename(p)} ...", flush=True)
        rec = regate(p, probe, dev)
        rec["seed"] = seed
        rows.append(rec)

    print(f"\n{'seed':>4} {'acc':>7} {'copy':>7} {'mass':>6} {'mOK':>6} "
          f"{'nGate':>6} {'frac+ 门控':>11} {'(未门控)':>9} {'Δmed 门控':>10}")
    print("-" * 78)
    for r in rows:
        print(f"{r['seed']:>4} {r['acc']:>7.4f} {r['copy']:>7.4f} "
              f"{r['mass_mean']:>6.3f} {r['mass_ok']:>6.3f} "
              f"{r['n_gated']:>6} {r['frac_gated']:>11.4f} "
              f"{r['frac_ungated']:>9.4f} {r['d_median_gated']:>+10.3f}")

    # 三态门。论文的规则是 copy >= 0.95 且 acc >= 0.99，而它**机械地**应用：
    # App.~constlr 第 2884 行的 sigma=0.267 是 "over the nine gated seeds"，
    # 被排除的第十个是 App.~fixband 第 2384 行那个 acc=0.985 的 seed 8 ——
    # "missing the accuracy condition by four thousandths"。
    #
    # 所以 acc 0.975 的 run 要排除，即使 mass 正常、读数有定义。这不是我们的
    # 取舍，是与论文口径一致的必然。
    print("\n三态门（copy >= 0.95 且 acc >= 0.99）")
    gated = []
    for r in rows:
        ok_c = r["copy"] >= 0.95
        ok_a = r["acc"] >= 0.99
        lab = "retrieval" if (ok_c and ok_a) else "neither"
        why = ""
        if ok_c and not ok_a:
            # 200 篇上 acc 的二项 se，用来说明差距是真实的还是噪声
            se = (0.99 * 0.01 / r["n"]) ** 0.5
            z = (0.99 - r["acc"]) / se if se else float("nan")
            why = (f"  两条件分离：copy 过而 acc 差 {0.99 - r['acc']:.4f}"
                   f"（{z:.1f} se，n={r['n']}）")
        print(f"  seed {r['seed']}  copy {r['copy']:.4f} {'过' if ok_c else '否'}"
              f"   acc {r['acc']:.4f} {'过' if ok_a else '否'}"
              f"   mass {r['mass_mean']:.3f}   -> {lab}{why}")
        if ok_c and ok_a:
            gated.append(r)

    if len(rows) > len(gated):
        print(f"\n  {len(rows) - len(gated)} 个 run 落在 neither，按论文口径排除。")
        print("  The copy and task-accuracy conditions can diverge. ")
        print("  Report their values separately; rendered-language diagnostics ")
        print("  do not identify circuit acquisition.")

    fr = [r["frac_gated"] for r in gated if r["frac_gated"] == r["frac_gated"]]
    if len(fr) < 2:
        print(f"\n只有 {len(fr)} 个过门的 seed，还不能判读极差。")
        print("At least two eligible runs are needed to describe a cross-run range.")
        return

    rng = max(fr) - min(fr)
    print(f"\nNL 臂 {len(fr)} 个 seed：极差 {rng:.4f}   sigma-hat "
          f"{rng / D3 if len(fr) == 3 else sd(fr):.4f}")

    # App.~fixband 的口径：三 seed 极差 + 每个端点自身的二项检验。
    # 只对**过门**的 run 做 —— 落在 neither 的 run 不进网格，对它算 p 值会让
    # 读者以为它参与了那个极差。
    print("\n端点的二项检验（对 0.5，n = n_gated，仅过门的 run）")
    for r in gated:
        if r["frac_gated"] != r["frac_gated"]:
            continue
        k = int(round(r["frac_gated"] * r["n_gated"]))
        p = binom_p(k, r["n_gated"])
        dec = "decisive" if p < 0.05 else "不决定性"
        print(f"  seed {r['seed']}  {k}/{r['n_gated']}  p = {p:.2e}  {dec}")

    # 与主网格的比较走 **sigma**，不走极差。
    #
    # 极差随 n 增长（d_3=1.693、d_10=3.078），而 sigma 不随 n 变 —— 只是精度
    # 变。所以跨 n 比 sigma 是well posed 的，比极差不是。
    #
    # 而"取主网格前 len(fr) 个 seed 做同样本量对照"这条路有一个我先前没注意到
    # 的偏置：App.~constlr 第 1965-1967 行说那三个 seed 恰好夹住了全样本
    # （"the cosine extremes happened to already lie among the original three"），
    # 故 GRID10[:3] 的极差 0.894 等于全十个的极差。拿 NL 的三 seed 极差去对它，
    # 是在跟一个异常宽的子集比。
    g_sd = sd(GRID10)
    print(f"\n主网格同格十 seed：sigma {g_sd:.4f}"
          f"（极差 {max(GRID10) - min(GRID10):.4f}）")
    print(f"  注意 GRID10[:3] = {GRID10[:3]} 的极差 "
          f"{max(GRID10[:3]) - min(GRID10[:3]):.4f} 等于全样本极差 —— 那三个")
    print("  seed 夹住了全样本（App.~constlr 第 1965-1967 行），所以它不是一个")
    print("  中性的同样本量对照。")

    # sigma-hat 在 n=3 上的 chi^2 区间。断言"NL 的 sigma 小于主网格"需要这个
    # 区间不含 g_sd；n=3 下它极宽（[0.52, 6.29] 倍），所以多半含。
    nl_sd = rng / D3 if len(fr) == 3 else sd(fr)
    chi = {2: (7.378, 0.0506), 3: (9.348, 0.216), 9: (19.023, 2.700)}
    df = len(fr) - 1
    print(f"\nNL 臂 sigma-hat {nl_sd:.4f}   对主网格 {g_sd:.4f}"
          f"   比 {nl_sd / g_sd if g_sd else float('nan'):.2f}")
    if df in chi:
        hi, lo = chi[df]
        l, h = nl_sd * math.sqrt(df / hi), nl_sd * math.sqrt(df / lo)
        print(f"  95% CI [{l:.3f}, {h:.3f}]"
              f"   {'含' if l <= g_sd <= h else '不含'} 主网格的 {g_sd:.3f}")
        if l <= g_sd <= h:
            print("  The reference lies inside this interval, conditional on its assumptions. "
                  "This is not a two-sample equality test ")
            print("  and does not establish equal dispersion.")

    # ---- 水平与可用空间。这一段决定极差该跟什么比 ----
    #
    # App.~drift 第 1924-1932 行：sign fraction 的均值与极差反相关（Spearman
    # rho = -0.70 over 20 cells），因为读数限在 [0,1] 里 —— 全部 seed 都在 0.99
    # 以上时 "no range wider than 0.02 is available to them"，而那被明确判为
    # "a ceiling effect and not evidence of stability"。
    #
    # 所以窄极差在高水平上是**被界解释的**，不是"什么都说不了"。要检验它，
    # 必须比同水平的主网格格子，而不是比 R3_D8 的 0.894。
    lvl = sum(fr) / len(fr)
    head_up, head_dn = 1.0 - max(fr), min(fr)
    print(f"\n水平与可用空间")
    print(f"  NL 臂 frac+ 水平 {lvl:.4f}"
          f"   向上余量 {head_up:.4f}   向下余量 {head_dn:.4f}")
    print(f"  观测极差 {rng:.4f}")

    # 主网格里同水平格子的极差（App.~drift 第 1926-1928 行的两个例子）。
    HIGH_LEVEL_RANGES = [(12, 5, 0.012, 0.038), (12, 8, 0.012, 0.040)]
    if lvl >= 0.95:
        print(f"\n  水平 >= 0.95：主网格同水平格子的极差是")
        for r_, d_, late, early in HIGH_LEVEL_RANGES:
            print(f"    R{r_}_D{d_}  早窗 {early:.3f} -> 晚窗 {late:.3f}"
                  f"（三 seed 全在 0.99 以上）")
        print(f"  NL range {rng:.4f}: compare with the references above. A bounded statistic ")
        print("  can have ceiling compression; this comparison does not identify its cause.")
        print("\n  但要注意 NL 臂落在这个水平本身就是一个结果：主网格同格")
        print(f"  （R3_D8）三 seed 是 {GRID10[:3]}，水平 0.51。表层加协变量的")
        print("  改变把这一格从'跨 seed 分散最大'移到了'饱和'。")
        print("\n  可归因性：这个臂同时动了表层、n_values（512->150）、语句长度")
        print("  （4->7.5 token）与 posCeil（0.111->0.039 实测）。后三者都是")
        print("  §sec:covar/§sec:limits 认定的承重协变量，故'表层导致了这个")
        print("  移动'不可从这一个臂断言。而协变量在这个设计里无法匹配：")
        print("  n_values 要 512 需要 512 个单 token 英文形容词（现有 170），")
        print("  语句长度要 4 token 就不是英语。")
    elif rng > 0.5:
        print(f"\n  极差 {rng:.3f} > 0.5 且水平中等：分散在真英文表层上仍然")
        print("  present in the rendered-language arm. ")
        print("  A broad range is a descriptive observation; ")
        print("  its precision still depends on the number of independent runs.")
    elif rng < 0.3:
        print(f"\n  极差 {rng:.3f} < 0.3 而水平 {lvl:.3f} 不在天花板：这一档")
        print("  才是真的说不了 —— 有空间却没用到，而 n=3 下 sigma 的 chi^2")
        print("  上界是 6.29 倍。要补到十个 seed。")
    else:
        print(f"\n  极差 {rng:.3f} 落在 0.3-0.5：不确定。补到十个 seed。")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(
                rows=rows,
                gated_seeds=[r["seed"] for r in gated],
                excluded_seeds=[r["seed"] for r in rows if r not in gated],
                range_gated=rng, sd_nl=nl_sd, sd_grid10=g_sd,
                grid10=GRID10), f, indent=1)
        print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
