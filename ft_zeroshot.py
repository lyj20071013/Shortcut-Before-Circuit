"""Measure pretrained-model task and copy diagnostics."""
import argparse
import os
from typing import Dict, List, Optional, Sequence

MIRROR = "https://hf-mirror.com"
MS_ID = {
    "Qwen/Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B": "Qwen/Qwen2.5-3B",
}


def load(name: str, src: str):
    """(tokenizer, model)。ModelScope 先落盘再从目录加载。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = name
    if src == "ms" and name in MS_ID:
        from modelscope import snapshot_download
        path = snapshot_download(MS_ID[name])
    tok = AutoTokenizer.from_pretrained(path)

    # base 模型多数没有 pad_token（它们不做批推理），而 tok(..., padding=True)
    # 在没有 pad_token 时直接抛 ValueError。用 eos 代替：pad 位置被
    # attention_mask 屏蔽，且我们只读真实末位的 logits，故填什么都不影响读数。
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # padding_side 必须显式设成 right。下游按 attention_mask.sum(1) - 1 取
    # 末位 logits，那个下标只在右填充下正确 —— 左填充时真实末位恒在 -1，
    # 而 sum-1 会指到序列中间，读到一个无关位置的分布。Qwen 的 base 与
    # instruct 在这个字段上不同，不能依赖默认值。
    tok.padding_side = "right"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) \
        else torch.float32
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt).to(dev)
    m.eval()
    return tok, m, dev


def prompt_of(doc: dict) -> str:
    """文档去掉答案 token 之后的前缀。

    render 的 text 含答案（训练时 LM loss 要它），零样本要的是它之前的部分。

    两条路径，因为我没有验证过 gate_docs 返回的字段集：nl_collide 用的是
    d["text"].split() 而不是 d["tokens"]，这暗示 tokens 可能不在里面。
      有 tokens/answer_pos -> 用 tokens[:answer_pos]，answer_pos 是 token
                              下标而 tokenize 是 str.split，故 ' '.join 是逆
      否则                 -> 从 text 去掉末词。答案是查询后的最后一个词
                              （'... colour is vast'），故 rsplit 一次即可
    两条路径都断言被丢掉的词等于 doc["answer"] —— 若前缀切错位置，读数会在
    一个无关分布上取 argmax 而 acc 看起来只是"低"，那是最坏的失败方式。
    """
    if "tokens" in doc and "answer_pos" in doc:
        toks = doc["tokens"]
        ap = doc["answer_pos"]
        dropped = toks[ap]
        pre = " ".join(toks[:ap])
    else:
        pre, _, dropped = doc["text"].rpartition(" ")
    if dropped != doc["answer"]:
        raise ValueError(
            f"前缀切在了错误位置：丢掉的是 {dropped!r} 而 answer 是 "
            f"{doc['answer']!r}。检查 render 的 text 是否以答案 token 结尾，"
            f"或 gate_docs 是否返回 tokens/answer_pos。")
    return pre


def cand_ids(tok, adjs: Sequence[str]) -> Dict[str, int]:
    """单 token 的 adj -> 它的 id。带前导空格，因为读数位置在句中。

    多 token 的 adj 在这里排除：单个位置的 argmax 无法选出它们，把它们算进
    候选集会让分母虚高而分子不变，acc 被系统性压低。ft_tokcheck 报的存活数
    就是这个集合的大小。
    """
    out = {}
    for w in adjs:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            out[w] = ids[0]
    return out


def prior_prompt(doc: dict, names: Sequence[str]) -> Optional[str]:
    """把查询问向一个本文档里不出现的实体。返回 None 表示找不到替身。

    这是这个脚本的地板控制，没有它零样本的数不可解释。理由：chance = 1/n_cands
    假设模型在候选集上均匀猜，但它不会 —— 读到 "Gus's favorite colour is"
    它会预测一个合理的颜色词，而池子里恰好有 pale / ivory / olive / teal。
    于是它不做任何检索也能高于 1/145，而 acc/chance 这个比值把先验算成了检索。

    top1_counts 那个诊断只抓得住"反复预测同一个 adj"的极端情形，抓不到"在
    十几个颜色词上均匀猜"，而后者给出的 acc 约 0.08 看起来像"部分会做"。

    替身实体在本文档里没有任何语句，所以没有可检索的信息，此时的 acc 就是
    纯先验。查询恒为 4 个 split token（"{Name}'s {attr1} {attr2} is"），故
    换掉倒数第 4 个即可，文档主体一字不动。
    """
    pre = prompt_of(doc)
    parts = pre.rsplit(" ", 4)
    if len(parts) < 5:
        return None
    body, poss, a1, a2, isw = parts
    for nm in names:
        # 名字的裸形与所有格形都不能出现在正文里，否则替身仍有语句可检索
        if nm not in body and f"{nm}'s" not in body:
            return " ".join([body, f"{nm}'s", a1, a2, isw])
    return None


def few_shot_prefix(docs: Sequence[dict], k: int) -> str:
    """k 个完整示例。零样本失败可能只是不懂格式而不是做不了任务 —— 这两者
    对微调臂的含义完全不同：若 8-shot 就能做，那模型本来会，微调不是在
    '形成'任何东西。"""
    if k <= 0:
        return ""
    return "\n\n".join(d["text"] for d in docs[:k]) + "\n\n"


def run_eval(model, tok, dev, docs, cands, id2adj, cid, prefix, batch,
             build, label) -> dict:
    """在 answer 位置取 argmax。build(doc) 决定 prompt 怎么造，返回 None 跳过。

    真 prompt 与先验地板共用这个函数是刻意的：若两条路径在批处理、末位取法
    或候选集上有任何差异，两个 acc 之差就不再只反映"文档里有没有可检索的
    信息"，而地板控制的全部意义就在那个差。
    """
    import torch

    n = n_blk = n_full = n_skip = 0
    top1: Dict[str, int] = {}
    ranks: List[int] = []

    for i in range(0, len(docs), batch):
        texts, golds = [], []
        for d in docs[i:i + batch]:
            if d["answer"] not in cands:
                n_skip += 1
                continue
            p = build(d)
            if p is None:
                n_skip += 1
                continue
            texts.append(prefix + p)
            golds.append(cands[d["answer"]])
        if not texts:
            continue

        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False)
        lens = enc["attention_mask"].sum(1)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(dev),
                        attention_mask=enc["attention_mask"].to(dev))
        lg = out.logits[torch.arange(len(texts), device=dev),
                        lens.to(dev) - 1].float()
        del out

        full = lg.argmax(-1)
        blk = cid[lg[:, cid].argmax(-1)]
        for j, g in enumerate(golds):
            n += 1
            n_full += int(full[j].item() == g)
            n_blk += int(blk[j].item() == g)
            w = id2adj.get(int(blk[j].item()), "?")
            top1[w] = top1.get(w, 0) + 1
            order = lg[j, cid].argsort(descending=True)
            pos = (cid[order] == g).nonzero()
            ranks.append(int(pos[0].item()) + 1 if len(pos) else len(cid))

    if not n:
        return dict(label=label, n=0, n_skip=n_skip)
    return dict(label=label, n=n, n_skip=n_skip,
                acc_blk=n_blk / n, acc_full=n_full / n,
                med_rank=sorted(ranks)[len(ranks) // 2],
                top=sorted(top1.items(), key=lambda x: -x[1])[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--src", default="hf", choices=["hf", "ms"])
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    a = ap.parse_args()

    if a.src == "hf" and not a.no_mirror:
        os.environ.setdefault("HF_ENDPOINT", MIRROR)

    import torch
    from nl_corpus import gate_docs, nl_spec
    from nl_generator import ADJ

    spec = nl_spec()
    # 多取 shots 篇做示例，示例文档不进评测集
    pool = gate_docs(a.r, a.d, a.docs + max(a.shots, 0) + 8, seed=7, spec=spec)
    shots, docs = pool[:a.shots], pool[a.shots:a.shots + a.docs]

    tok, model, dev = load(a.model, a.src)
    print(f"{a.model}  vocab {len(tok):,}  device {dev}  "
          f"dtype {next(model.parameters()).dtype}")

    # shots 把序列拉长约 (1+shots) 倍，而 logits 显存 ∝ batch × seq × vocab。
    # 按 batch × (1+shots) <= 8 保守降，免得在 --shots 8 上 OOM 而看起来像
    # 脚本坏了。
    if a.shots > 0:
        b2 = max(1, 8 // (1 + a.shots))
        if b2 < a.batch:
            print(f"  --shots {a.shots}：batch {a.batch} -> {b2}"
                  f"（logits 显存 ∝ batch × seq × {len(tok):,}）")
            a.batch = b2

    cands = cand_ids(tok, ADJ[:spec.n_values])
    print(f"单 token adj 候选 {len(cands)}/{spec.n_values}"
          f"  随机基线 1/{len(cands)} = {1 / len(cands):.4f}")
    if len(cands) < 0.5 * spec.n_values:
        print("  警告：过半的值不是单 token。acc 的分母是这个子集，故与")
        print("  from-scratch 臂的 acc 不同口径 —— 先跑 ft_tokcheck。")

    id2adj = {v: k for k, v in cands.items()}
    cid = torch.tensor(sorted(cands.values()), device=dev)
    pre = few_shot_prefix(shots, a.shots)

    from nl_generator import NAMES
    real = run_eval(model, tok, dev, docs, cands, id2adj, cid, pre, a.batch,
                    prompt_of, "真 prompt")
    floor = run_eval(model, tok, dev, docs, cands, id2adj, cid, pre, a.batch,
                     lambda d: prior_prompt(d, NAMES), "先验地板")

    if not real.get("n"):
        print("没有可评测文档 —— 全部答案都不是单 token。")
        return

    chance = 1 / len(cands)
    for rec in (real, floor):
        if not rec.get("n"):
            print(f"\n{rec['label']}：无可评测文档（跳过 {rec['n_skip']}）")
            continue
        print(f"\n{rec['label']}  n={rec['n']}  跳过 {rec['n_skip']}"
              f"  shots={a.shots}")
        print(f"  候选集内 acc  {rec['acc_blk']:.4f}"
              f"   （均匀随机 {chance:.4f}，{rec['acc_blk'] / chance:.1f}x）")
        print(f"  全词表 acc    {rec['acc_full']:.4f}")
        print(f"  gold 中位排名 {rec['med_rank']}/{len(cands)}")
        print(f"  预测最多的五个 adj  {rec['top']}")
        if rec["top"] and rec["top"][0][1] > 0.5 * rec["n"]:
            print(f"  注：{rec['top'][0][1]}/{rec['n']} 次预测同一个 adj。")

    print("\n" + "=" * 60)
    print("判读")
    print("  posCeil 在 str.split() 下实测 0.039（nl_collide，R3_D8）。BPE 下")
    print("  句长变了（均值 380 -> 463），要用目标 tokenizer 重测才能作为")
    print("  位置捷径的门。")

    if not floor.get("n"):
        print("\n  地板控制没跑成（找不到不出现在正文里的替身名字）。没有它")
        print("  acc 不可解释：均匀随机不是这个任务的地板，模型不检索也能")
        print("  靠颜色词先验高于 1/145。")
        return

    ab, af = real["acc_blk"], floor["acc_blk"]
    gap = ab - af
    # 两个 acc 的差的标准误。两组同 n 且独立，故 se = sqrt(se1^2 + se2^2)。
    n_ = real["n"]
    se = (ab * (1 - ab) / n_ + af * (1 - af) / max(floor["n"], 1)) ** 0.5
    print(f"\n  检索增益 = {ab:.4f} - {af:.4f} = {gap:+.4f}"
          f"   (se {se:.4f}, {gap / se if se else 0:.1f}σ)")

    if ab >= 0.99:
        print("\n  真 acc >= 0.99：模型来的时候就会。微调臂测不到回路形成，")
        print("  '形成后固定距离读数'失去锚点。这个臂要改设计 —— 例如改成")
        print("  零样本读数在 prompt 变体上的分散。")
    elif gap < 2 * se:
        print("\n  检索增益不显著：模型没有从文档里取信息，acc 全部来自先验。")
        print("  这是微调臂最干净的起点 —— 形成过程从零开始，与 from-scratch")
        print("  臂可比。注意此时 acc/chance 那个倍数是先验而非能力，报告时")
        print("  要用地板而不是均匀随机作基线。")
    elif ab < 0.5:
        print("\n  有检索但远未解决：模型部分会做。形成过程可测，但起点不是")
        print("  零，要在附录报告零样本基线与地板两个数。")
    else:
        print("\n  0.5 <= acc < 0.99 且检索显著：模型大体会做。微调只是把它")
        print("  推到饱和，'形成'这个概念在这个臂上与 from-scratch 臂不同义，")
        print("  三态分类器的 retrieval 门要重新定义。")

    print("\n下一步：--shots 8 再跑一次。零样本失败而 8-shot 成功意味着模型")
    print("本来会做、只是不懂格式 —— 那时微调不是在'形成'回路，读数门控的")
    print("语义要重新定义。两次都接近地板才说明是能力问题。")


if __name__ == "__main__":
    main()
