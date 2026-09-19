"""Evaluate candidate-rule collisions in the rendered-language arm."""
import argparse
import json
from collections import Counter
from typing import Dict, List, Optional, Sequence

from coext_audit import _predict
from config import dd_band
# 必须走适配器：nl_generator 的平行生成器已被渲染层取代，函数还在文件里，
# import 不会报错 —— 会静默地量一个不再被训练的语料。
from nl_corpus import gate_docs as make_docs

R_OLDS = [3, 5, 8, 12, 16]
DDS = [2, 3, 5, 8, 16]

# 主网格参照（tab:collide）。NL 臂的对应列应当落在这些区间内或附近。
MAIN = dict(rarity="1.000 全格", frequency="0.001-0.309", primacy="0.000",
            last_token="0.000 全格")


def q_values(d: dict) -> List[str]:
    """被查询 slot 的值序列，按文档顺序。对应 collide.py:34 的 vals。

    读数在 adj 位置取（选项 C），所以规则比较的是 adj 而不是完整值 ——
    v* 与 v_truth 在 adj 上必不同（sample_values 保证），故这是可分的。
    """
    return [s["value"].split()[0] for s in d["stmts"]
            if s["slot"] == d["q_slot"]]


def rates(docs: Sequence[dict]) -> dict:
    """四条有定义的规则 + LTK 的碰撞率。truth = recency 的预测，
    与 collide.py:96 同口径，故每一列都是「与 recency 同指的比例」。"""
    n = 0
    agree = Counter()
    for d in docs:
        vals = q_values(d)
        if len(vals) < 2:
            continue
        n += 1
        p = _predict(vals)
        t = p["recency"]
        agree["rarity"] += int(p["rarity"] == t)
        agree["frequency"] += int(p["frequency"] == t)
        agree["primacy"] += int(p["primacy"] == t)
        agree["tie"] += int(p["tie"])
        # LTK：查询前最后一条语句的值，与 slot 无关（collide.py:49）
        ltk = d["stmts"][-1]["value"].split()[0] if d["stmts"] else None
        agree["last_token"] += int(ltk == t)
    if not n:
        return dict(n=0)
    out = {k: agree[k] / n for k in
           ("rarity", "frequency", "primacy", "last_token", "tie")}
    out["n"] = n
    out["n_skipped"] = len(docs) - n
    return out


def emp_posceil(docs: Sequence[dict]) -> dict:
    """经验 posCeil，以及它相对随机基线的倍数。

    不设偏移上限：最优偏移约为 (1+ΔD) 条语句 × 每条 7.5 token，ΔD=16 时
    落在 68 附近。任何固定 cap（我先前写的 80、nl_gates 里的 60）都会在
    高 ΔD 上截断搜索并低估 posCeil。

    chance 是"知道答案是个形容词、但不知道是哪个"的命中率 = 1/n_values。

    注意分母是 spec.n_values 而不是 len(ADJ)。两者先前相等（都是 100），现在
    不等：ADJ 扩到约 170 以避免 _ValueDraw 耗尽，而 n_values 停在 150。
    实际出现在语料里的 adj 只有前 n_values 个（val_id < n_values），所以
    用 len(ADJ) 会把基线算低约 12%，让 posCeil/chance 虚高。
    位置规则只有在 posCeil > chance 时才是一条模型有动机去学的规则；
    低于 chance 就是比瞎猜还差，此时"位置捷径"这个相位不存在。主网格的
    chance 是 1/512 ≈ 0.002，posCeil 是它的 30-170 倍，所以那里这个问题
    从不出现；NL 臂的 chance 是 1/150 ≈ 0.0067，高 3.4 倍。
    """
    from nl_corpus import nl_spec

    n_val = nl_spec().n_values
    toks = [d["text"].split() for d in docs]
    ans = [d["answer"] for d in docs]
    lim = max((len(t) for t in toks), default=0)
    best_k, best = None, 0.0
    for k in range(1, lim + 1):
        hit = sum(1 for t, a in zip(toks, ans) if len(t) >= k and t[-k] == a)
        r = hit / len(docs) if docs else 0.0
        if r > best:
            best, best_k = r, k
    chance = 1.0 / n_val
    return dict(emp=best, offset=best_k, chance=chance, n_values=n_val,
                over_chance=(best / chance if chance else float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1500)
    ap.add_argument("--rows", type=int, nargs="+", default=R_OLDS)
    ap.add_argument("--cols", type=int, nargs="+", default=DDS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="nl_collide")
    a = ap.parse_args()

    rows = []
    hdr = ("R  dD   n  REC    RAR    FRQ    PRI    LTK    tie    "
           "posEmp  off  1/supp  /chance")
    print(hdr)
    print("-" * len(hdr))
    for r in a.rows:
        for d in a.cols:
            docs = make_docs(r, d, a.docs, seed=a.seed)
            rt = rates(docs)
            pc = emp_posceil(docs)
            lo, hi = dd_band(d)
            supp = hi - lo + 1
            row = dict(r_old=r, dd=d, supp=supp, analytic=1.0 / supp, **rt,
                       pos_emp=pc["emp"], pos_off=pc["offset"],
                       chance=pc["chance"], over_chance=pc["over_chance"])
            rows.append(row)
            print(f"{r:>2} {d:>3} {rt['n']:>4} 1.000  "
                  f"{rt['rarity']:.3f}  {rt['frequency']:.3f}  "
                  f"{rt['primacy']:.3f}  {rt['last_token']:.3f}  "
                  f"{rt['tie']:.3f}  {pc['emp']:.3f}  {pc['offset']:>3}  "
                  f"{1.0 / supp:.3f}  {pc['over_chance']:>5.1f}x")

    with open(f"{a.out}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print("\n主网格参照（tab:collide）")
    for k, v in MAIN.items():
        print(f"  {k:>12}  {v}")

    print("\n判读")
    rar = [x["rarity"] for x in rows]
    ltk = [x["last_token"] for x in rows]
    tie = [x["tie"] for x in rows]
    print(f"  RAR  {min(rar):.3f}-{max(rar):.3f}   "
          f"{'式 1 恒等成立' if min(rar) > 0.9995 else '式 1 破了 -> §sec:flat 在此臂失效'}")
    print(f"  LTK  {min(ltk):.3f}-{max(ltk):.3f}   "
          f"{'与主网格一致' if max(ltk) <= 0.005 else '读数与复制最后一个值 token 混淆'}")
    print(f"  tie  {min(tie):.3f}-{max(tie):.3f}   "
          f"{'非平局区' if max(tie) < 1e-9 else '有平局文档 -> rarity 退化，counts 不是 old=R new=1'}")

    # posCeil 是这个臂相对主网格的唯一实质收益，单独判读
    worse = [x for x in rows if x["pos_emp"] > x["analytic"] + 1e-9]
    gains = [(x["analytic"] - x["pos_emp"]) for x in rows]

    # 比"经验低于解析"更要紧的是"经验是否高于随机"。位置规则只有在
    # posCeil > chance 时才是模型有动机去学的规则；低于 chance 时那个相位
    # 不存在，而 sweep.classify 的 pos 判据（acc >= 0.5*posCeil）会把任何
    # 随机水平的模型都判成 pos，读出无意义的 posNNNN%。
    below = [x for x in rows if x["over_chance"] <= 1.0]
    marginal = [x for x in rows if 1.0 < x["over_chance"] <= 2.0]
    print(f"\n  posCeil vs 随机基线 1/|ADJ| = {rows[0]['chance']:.4f}")
    print(f"    低于或等于随机 {len(below)} 格 -> 这些格没有位置捷径相位")
    for x in below:
        print(f"      R{x['r_old']}_D{x['dd']}  {x['pos_emp']:.3f} "
              f"= {x['over_chance']:.1f}x chance")
    print(f"    1-2x 随机 {len(marginal)} 格 -> 捷径存在但优势很薄")
    for x in marginal:
        print(f"      R{x['r_old']}_D{x['dd']}  {x['pos_emp']:.3f} "
              f"= {x['over_chance']:.1f}x chance")
    if below or marginal:
        print("    主网格的 chance 是 1/512≈0.002、posCeil 是它的 30-170 倍，")
        print("    所以那里全格都有捷径。NL 臂的 chance 是 1/n_values，现在")
        print(f"    1/{rows[0]['n_values']}≈{rows[0]['chance']:.4f}，高 3.4 倍。")
        print("    要恢复只能提 n_values，但它受 len(ADJ) 的单射上界约束，")
        print("    而 ADJ 的下界由 _ValueDraw 的耗尽定（nl_generator.")
        print("    values_needed）。两边夹得很紧，提之前先算 values_needed。")

    print(f"\n  posCeil 经验 vs 解析：平均降低 {sum(gains) / len(gains):+.3f}")
    if worse:
        print(f"  {len(worse)} 格经验值高于 1/|supp| —— 句长分布让答案位置"
              f"比 ΔD 更可预测，会在低 ΔD 行伪造位置态：")
        for x in worse[:5]:
            print(f"    R{x['r_old']}_D{x['dd']}  "
                  f"emp {x['pos_emp']:.3f} > {x['analytic']:.3f}")
    else:
        print("  全格经验值 <= 解析值：变长句子确实削弱了位置捷径。")
    print("  Use the measured position-only ceiling for the rendered corpus; "
          "this choice affects run eligibility under the state classifier.")

    print("\nGLU 未报：collide.py:46 读 is_update 标志，nl_generator 不产出。"
          "\n若主网格的 GLU 列非平凡（不是全 0 或全 nan），NL 臂需要补这个"
          "\n字段才能并列七列；若主网格 GLU 也退化，两边都不报即可。")


if __name__ == "__main__":
    main()
