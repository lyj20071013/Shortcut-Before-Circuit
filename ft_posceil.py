"""Estimate the tokenizer-level fixed-position diagnostic."""
import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

MIRROR = "https://hf-mirror.com"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--src", default="hf", choices=["hf", "ms"])
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--docs", type=int, default=1500)
    ap.add_argument("--rows", type=int, nargs="+", default=[3])
    ap.add_argument("--cols", type=int, nargs="+", default=[8])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.src == "hf" and not a.no_mirror:
        os.environ.setdefault("HF_ENDPOINT", MIRROR)

    from ft_data import canon
    from ft_pool import ft_spec, install, verify_install
    from ft_tokcheck import get_tokenizer
    from generator import generate_corpus
    # 不导 dd_band：它在 nl_corpus 里可能是函数内 import，那样它不是模块级
    # 名字，from nl_corpus import dd_band 会 ImportError。supp 直接从 cfg 的
    # delta_d_lo/hi 算，那两个字段一定在（runs_dseeds 的 meta 记录里有）。
    from nl_corpus import nl_corpus_cfg, tokenize
    from nl_generator import values_needed
    from nl_render import render
    from vocab import Vocab

    tok, via = get_tokenizer(a.model, a.src)
    if not getattr(tok, "is_fast", False):
        raise SystemExit("需要 fast tokenizer")
    ver = install(tok, need=values_needed(55, 3), model_name=a.model)
    verify_install()
    spec = ft_spec()
    v = Vocab(spec)
    print(f"{a.model}  来源 {via}  版本 {ver}  n_values {spec.n_values}\n")

    rows = []
    for r in a.rows:
        for d in a.cols:
            cfg = nl_corpus_cfg(r, d, 7)
            supp = cfg.delta_d_hi - cfg.delta_d_lo + 1

            # 搜索底串是 **prompt**（不含答案），不是全文。
            #
            # 第一版用 canon(rd["tokens"])，那含答案，于是"从末尾数第 1 个
            # token"就是答案本身 —— 实测 emp=1.0 @ 偏移 1，一个平凡的 1.0。
            #
            # 位置规则的定义是"忽略查询、输出 prompt 末尾固定偏移处的 token"，
            # 所以搜索空间是 prompt。nl_collide 用 d["text"] 而得到 0.039 @
            # 偏移 70，那说明 rd["text"] 是 prompt 而 rd["tokens"] 含答案 ——
            # 两个底串的这个差别此前没有被任何测量确认过，现在被确认了。
            #
            # ft_data 用 canon(tokens) 仍然是对的：训练需要答案在序列里（LM
            # loss 要预测它），探针读 logits[k-1] 也需要它。只有偏移搜索不需要。
            seqs: List[List[int]] = []
            ans: List[int] = []
            n_multi = 0
            for doc in generate_corpus(v, cfg, a.docs):
                rd = render(doc, v, tokenize)
                ap = rd["answer_pos"]
                w = rd["tokens"][ap]
                e = tok.encode(" " + w, add_special_tokens=False)
                if len(e) != 1:
                    n_multi += 1
                    continue                  # 过滤后不该发生
                prompt = canon(rd["tokens"][:ap])
                seqs.append(tok.encode(prompt, add_special_tokens=False))
                ans.append(e[0])
            if n_multi:
                print(f"  警告：{n_multi} 篇的答案不是单 token —— ft_pool 的"
                      f"过滤应当已排除，检查 install() 是否生效")
            if not seqs:
                print(f"R{r}_D{d}：无可用文档")
                continue

            # 从末尾数第 k 个 token 等于答案的比例，k 无上限。
            #
            # 不设 cap：最优偏移约 (1+ΔD) 条语句 × 每条约 9 个 BPE token，
            # ΔD=12 时落在 120 附近。任何固定 cap 都会在高 ΔD 上截断搜索并
            # 低估 posCeil —— nl_collide 的 docstring 记了同一个坑。
            lim = max(len(s) for s in seqs)
            best_k, best = None, 0.0
            curve = []
            for k in range(1, lim + 1):
                hit = sum(1 for s, t in zip(seqs, ans)
                          if len(s) >= k and s[-k] == t)
                p = hit / len(seqs)
                curve.append(p)
                if p > best:
                    best, best_k = p, k

            chance = 1.0 / spec.n_values
            L = [len(s) for s in seqs]
            # top5 带上命中率。先前只存下标，于是输出里看不出次优偏移的高度 ——
            # 而"有没有第二个峰"决定 posCeil 是一个尖峰还是一片平台。
            # tok_mean/max 现在是 **prompt** 的长度（不含答案），与偏移同一坐标系。
            rec = dict(r=r, d=d, n=len(seqs), supp=supp,
                       analytic=1.0 / supp, emp=best, offset=best_k,
                       chance=chance, over_chance=best / chance,
                       prompt_tok_mean=sum(L) / len(L), prompt_tok_max=max(L),
                       top5=[(i + 1, round(curve[i], 4))
                             for i in sorted(range(len(curve)),
                                             key=lambda i: -curve[i])[:5]])
            rows.append(rec)
            print(f"R{r}_D{d}  n={rec['n']}")
            print(f"  经验 posCeil {best:.4f} 在偏移 {best_k}"
                  f"（从末尾数）")
            print(f"  解析 1/|supp| = {rec['analytic']:.4f}"
                  f"   随机 1/n_values = {chance:.4f}"
                  f"   posCeil/随机 = {rec['over_chance']:.1f}x")
            print(f"  prompt BPE token 数  均值 {rec['prompt_tok_mean']:.0f}"
                  f"  最大 {rec['prompt_tok_max']}")
            print(f"  前 5 个偏移及命中率  {rec['top5']}")

    print("\n" + "=" * 60)
    print("与另外两个口径对照")
    print("  主网格（4-token 定长）      解析 = 经验 = 1/|supp| = 0.111")
    print("  NL 臂（空白分词，词表 313） 经验 0.039，偏移 70")
    for rec in rows:
        print(f"  微调臂（{a.model} 的 BPE） 经验 {rec['emp']:.4f}，"
              f"偏移 {rec['offset']}")
    print("\n判读")
    print("  经验值低于解析的 1/|supp| 是预期的：变长语句让同一个 ΔD 对应多个")
    print("  token 偏移，故没有单一偏移能吃到整个 ΔD 分布的众数。这削弱了位置")
    print("  捷径，是这两个表层臂与主网格的一处实质差异。")
    for rec in rows:
        if rec["emp"] <= rec["chance"]:
            print(f"  R{rec['r']}_D{rec['d']}：posCeil <= 随机基线 —— 位置捷径")
            print("  在这一格不存在（比瞎猜还差），故'位置'相位无从谈起。")
        elif rec["over_chance"] < 2:
            print(f"  R{rec['r']}_D{rec['d']}：posCeil 只有随机的 "
                  f"{rec['over_chance']:.1f} 倍，捷径极薄。")
    print("\n  三个微调 run 的 acc 都是 1.000，远高于上面任何一个 posCeil，")
    print("  所以三态判定不受这个数影响。它的用途是 app:surface 的引用。")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(model=a.model, version=ver,
                           n_values=spec.n_values, rows=rows), f, indent=1)
        print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
