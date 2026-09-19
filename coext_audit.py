"""Sequence-rule collision utilities used by the rendered-language checks."""
import argparse
import json
import random
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Sequence

TIE_NOTE = "lo == hi：rarity/frequency 退化为 vals[-1]，与 recency 平凡同指"


def _predict(vals: Sequence[str]) -> dict:
    """四条规则的预测。复刻 collide.py:34-44，同一平局约定（取更靠后）。"""
    cnt = Counter(vals)
    lo, hi = min(cnt.values()), max(cnt.values())
    return dict(
        recency=vals[-1],
        rarity=next(v for v in reversed(vals) if cnt[v] == lo),
        frequency=next(v for v in reversed(vals) if cnt[v] == hi),
        primacy=vals[0],
        tie=(lo == hi),
        n_assign=len(vals),
        n_distinct=len(cnt),
    )


def shape_of(vals: Sequence[str]) -> str:
    """非平局子集内的形态。二分，因为全不同 -> lo==hi -> 已被平局过滤掉。

    grid:     末值恰出现 1 次且有别的值重复，即 [old×R, new×1] —— 与合成网格
              同形。此形态下 lo==1 落在末值上，rarity 反向扫到的就是末值，
              故 rarity==recency 必然成立。tab:collide 的 RAR=1.000 就是它。
    tail_rep: 末值自身重复。此时 rarity 可能选别的值（分歧），也可能因为
              末值恰是最小计数而同指。
    """
    cnt = Counter(vals)
    if cnt[vals[-1]] == 1 and max(cnt.values()) > 1:
        return "grid"
    return "tail_rep"


def classify(rec: dict) -> dict:
    """truth = recency（collide.py:96），故各列都是「与 recency 同指」。"""
    p = _predict(rec["vals"])
    t = p["recency"]
    slot = rec.get("slot")
    return dict(doc_id=rec.get("doc_id", ""), slot=slot,
                sub=(slot[0] if isinstance(slot, (list, tuple)) and slot
                     else ""),
                n_assign=p["n_assign"], n_distinct=p["n_distinct"],
                tie=p["tie"], agree=(p["rarity"] == t),
                shape=("tie" if p["tie"] else shape_of(rec["vals"])),
                frq=(p["frequency"] == t), pri=(p["primacy"] == t),
                vals=list(rec["vals"]))


def audit(records: Iterable[dict], min_assign: int = 2) -> dict:
    rows = [classify(r) for r in records if len(r["vals"]) >= min_assign]
    rep = [r for r in rows if not r["tie"]]
    tie = [r for r in rows if r["tie"]]

    def rate(xs, key="agree"):
        return (sum(x[key] for x in xs) / len(xs)) if xs else None

    def group(xs, key):
        g: Dict[str, dict] = {}
        for x in xs:
            b = g.setdefault(str(x[key]), dict(n=0, k=0))
            b["n"] += 1
            b["k"] += int(x["agree"])
        return {k: dict(n=v["n"], rate=v["k"] / v["n"])
                for k, v in sorted(g.items(), key=lambda kv: -kv[1]["n"])}

    by_n: Dict[int, dict] = {}
    for r in rep:
        b = by_n.setdefault(r["n_assign"], dict(n=0, k=0))
        b["n"] += 1
        b["k"] += int(r["agree"])

    # 自检：shape=="grid" 蕴含 agree。grid 的定义是末值恰 1 次且有值重复，
    # 此时 lo==1 只落在末值上，rarity 反向扫到的就是末值 == recency。
    # 非空即 shape_of 或 _predict 有一个是错的，不要忽略。
    bad_grid = [r["vals"] for r in rep if r["shape"] == "grid" and not r["agree"]]

    return dict(
        n_slots=len(rows), n_repeat=len(rep), n_tie=len(tie),
        coext_repeat=rate(rep),      # <- 主数，与 tab:collide 的 RAR 列同口径
        coext_tie=rate(tie),         # 应恒为 1.000，见 TIE_NOTE
        coext_pooled=rate(rows),     # 仅供报告，勿作主数
        frq_repeat=rate(rep, "frq"), pri_repeat=rate(rep, "pri"),
        by_shape=group(rep, "shape"),
        by_sub=group(rep, "sub"),
        by_n_assign={str(k): dict(n=v["n"], rate=v["k"] / v["n"])
                     for k, v in sorted(by_n.items())},
        n_bad_grid=len(bad_grid), bad_grid_examples=bad_grid[:5],
        disagree_examples=[r["vals"] for r in rep if not r["agree"]][:20],
        grid_examples=[r["vals"] for r in rep if r["shape"] == "grid"][:10],
    )


def read_jsonl(path: str) -> Iterator[dict]:
    """抽取产物。每行 {"doc_id":..., "slot":[e,a], "vals":[v1,v2,...]}，
    vals 按文档顺序，即 collide.py 里 q_stmts 过滤后的 [s.val for s in ...]。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", required=True, help="抽取产物 jsonl")
    ap.add_argument("--min-assign", type=int, default=2)
    ap.add_argument("--sample", type=int, default=0,
                    help="随机抽 N 条打印供人工核对抽取精度")
    ap.add_argument("--sub", nargs="+", default=None,
                    help="只保留 slot[0] 在此列表内的记录，如 .yaml .toml。"
                         "用于切片而不重抽")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    recs = list(read_jsonl(a.slots))
    if a.sub:
        keep = set(a.sub)
        n0 = len(recs)
        recs = [r for r in recs
                if isinstance(r.get("slot"), (list, tuple)) and r["slot"]
                and r["slot"][0] in keep]
        print(f"--sub 过滤：{n0} -> {len(recs)} 条（保留 {sorted(keep)}）\n")
    res = audit(recs, a.min_assign)

    print(f"slot 总数 {res['n_slots']}   非平局 {res['n_repeat']}"
          f"（主数分母）   平局 {res['n_tie']}")
    print()
    for k in ("coext_repeat", "coext_tie", "coext_pooled"):
        v = res[k]
        print(f"{k:>14}  " + ("-" if v is None else f"{v:.4f}"))
    print(f"\n非平局子集  FRQ={res['frq_repeat']}  PRI={res['pri_repeat']}")
    print(f"\ntab:collide 的合成端：RAR 全格 1.000，FRQ 0.001-0.309，PRI 0.000")

    print("\n按形态分解（非平局子集）—— grid 是与合成网格同形的那些")
    for s, b in res["by_shape"].items():
        print(f"  {s:>9}  n={b['n']:>6}  同指={b['rate']:.4f}")
    print("  grid: 末值恰 1 次且有值重复 = [old×R, new×1]，网格恒为此形")
    print("  tail_rep: 末值自身重复，网格里不存在")

    print("\n按后缀分解（非平局子集）")
    for s, b in list(res["by_sub"].items())[:15]:
        print(f"  {s:>12}  n={b['n']:>6}  同指={b['rate']:.4f}")

    print("\n按赋值次数分解（非平局子集）")
    for n, b in res["by_n_assign"].items():
        print(f"  n_assign={n:>3}  n={b['n']:>6}  同指={b['rate']:.4f}")

    if res["n_bad_grid"]:
        print(f"\n严重：{res['n_bad_grid']} 条 shape=grid 但不同指。"
              f"grid 的定义蕴含同指，非零说明 shape_of 或 _predict 有错，"
              f"整个形态分解不可信。样例：")
        for v in res["bad_grid_examples"]:
            print("  " + " -> ".join(map(str, v)))
    else:
        print("\n自检通过：所有 shape=grid 的记录都同指（定义蕴含此结果）")

    if res["grid_examples"]:
        print("\ngrid 形态样例（应当形如 A -> A -> B），最多 10 条：")
        for v in res["grid_examples"]:
            print("  " + " -> ".join(map(str, v)))

    if res["disagree_examples"]:
        print("\n分歧样例（rarity != recency），最多 20 条：")
        for v in res["disagree_examples"]:
            print("  " + " -> ".join(map(str, v)))

    if res["coext_tie"] is not None and abs(res["coext_tie"] - 1.0) > 1e-9:
        print(f"\n警告：平局子集共延率 != 1.000。{TIE_NOTE} 这个假设已失效，"
              f"说明 _predict 与 collide.py:34-44 不再一致。")

    if a.sample:
        print(f"\n--- 随机 {a.sample} 条，人工核对抽取精度 ---")
        for r in random.Random(0).sample(recs, min(a.sample, len(recs))):
            print(f"  {r.get('slot')}  {r['vals']}  [{r.get('doc_id','')}]")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print(f"\n已写入 {a.json}")


if __name__ == "__main__":
    main()
