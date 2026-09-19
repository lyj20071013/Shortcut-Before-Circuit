"""Check token-pool compatibility for pretrained tokenizers."""
import argparse
import os
import sys
from typing import Dict, List, Sequence

# HF_ENDPOINT 必须在 import transformers / huggingface_hub **之前**设好：
# 那两个包在 import 时就读它并缓存进模块级常量，之后改 os.environ 无效。
# 本文件把 transformers 的 import 放在 main() 里，就是为了让这里先跑。
MIRROR = "https://hf-mirror.com"

DEFAULT = [
    "Qwen/Qwen2.5-0.5B",        # 架构与 from-scratch 模型同构，词表 152k
    "EleutherAI/pythia-410m",   # 第二族；the Pile 公开，可关联先验来源
    "HuggingFaceTB/SmolLM2-360M",   # 小、现代架构、FineWeb-Edu 公开
    "Qwen/Qwen2.5-1.5B",        # 只在 0.5B 出结果后才需要
]

# ModelScope 的 id 与 HF 不总是一致：Qwen 系是同形的，非阿里的模型多数不在
# ModelScope 上。表里没有的走 hf-mirror。
MS_ID = {
    "Qwen/Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B": "Qwen/Qwen2.5-3B",
}


def get_tokenizer(name: str, src: str):
    """取 tokenizer。返回 (tok, 实际来源) 或抛异常。

    ModelScope 先 snapshot_download 到本地再从目录加载 —— 它的 SDK 不与
    transformers 的 from_pretrained 直接互通。只下 tokenizer 相关文件，
    不下权重（allow_patterns），否则这个 CPU 检查会拖下几个 GB。
    """
    from transformers import AutoTokenizer

    if src == "ms":
        if name not in MS_ID:
            return AutoTokenizer.from_pretrained(name), "hf-mirror（ms 无此 id）"
        from modelscope import snapshot_download
        d = snapshot_download(
            MS_ID[name],
            allow_patterns=["*.json", "*.txt", "*.model", "merges.txt",
                            "tokenizer*"])
        return AutoTokenizer.from_pretrained(d), "modelscope"
    return AutoTokenizer.from_pretrained(name), os.environ.get(
        "HF_ENDPOINT", "huggingface.co")


def single(tok, w: str, lead: bool = True) -> bool:
    """w 在该 tokenizer 下是否恰一个 token。

    lead=True 时前置空格：句中的词几乎总以 ' pale' 的形式出现，而 BPE 对
    ' pale' 和 'pale' 给不同的 id。读数在 adj 位置取，那个位置是句中，所以
    带空格的形式才是实际要用的。
    """
    s = (" " + w) if lead else w
    return len(tok.encode(s, add_special_tokens=False)) == 1


def survey(tok, pool: Sequence[str], name: str) -> List[str]:
    ok = [w for w in pool if single(tok, w)]
    print(f"    {name:<12} {len(ok):>4}/{len(pool):<4} 单 token")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=[])
    ap.add_argument("--src", default="hf", choices=["hf", "ms"],
                    help="hf=HuggingFace（默认走 hf-mirror），ms=ModelScope")
    ap.add_argument("--no-mirror", action="store_true",
                    help="直连 huggingface.co，境外环境用")
    a = ap.parse_args()
    names = a.models or DEFAULT

    # 在 import transformers 之前设。放在 main() 开头是刻意的 —— 见文件头。
    if a.src == "hf" and not a.no_mirror:
        os.environ.setdefault("HF_ENDPOINT", MIRROR)
    print(f"来源 {a.src}"
          + (f"  HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}"
             if a.src == "hf" else "") + "\n")

    from nl_generator import ADJ, ATTR, NAMES, NOUN, values_needed
    need = values_needed(55, 3)
    print(f"值域下界 values_needed(55, 3) = {need}")
    print(f"当前池子 ADJ={len(ADJ)} NOUN={len(NOUN)} NAMES={len(NAMES)}"
          f" ATTR={len(ATTR)}\n")

    # 只是存在性检查：真正的 import 在 get_tokenizer 里，那里才是 HF_ENDPOINT
    # 已生效之后。
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("需要 transformers：pip install transformers")
        sys.exit(2)
    if a.src == "ms":
        try:
            import modelscope  # noqa: F401
        except ImportError:
            print("需要 modelscope：pip install modelscope")
            sys.exit(2)

    # 渲染一批文档，用于 G1/G4 的端到端重测。用主网格的放置层，
    # 与 nl_gates 量的是同一个语料。
    from nl_corpus import gate_docs, gate_pairs, nl_spec
    spec = nl_spec()
    docs = gate_docs(3, 8, 400, seed=7, spec=spec)
    pairs = gate_pairs(3, 8, 400, seed=7, spec=spec)
    print(f"渲染 {len(docs)} 篇 + {len(pairs)} 个编辑对用于 G1/G4 重测\n")

    verdict = {}
    for nm in names:
        print(f"=== {nm} ===")
        try:
            tok, via = get_tokenizer(nm, a.src)
            print(f"    来源 {via}  vocab {len(tok):,}")
        except Exception as e:
            print(f"    取 tokenizer 失败：{type(e).__name__}: {e}\n")
            verdict[nm] = "tokenizer 不可用"
            continue

        adj_ok = survey(tok, ADJ, "ADJ")
        noun_ok = survey(tok, NOUN, "NOUN")
        survey(tok, NAMES, "NAMES")

        # ---- 池子是否够大 ----
        # 只有 ADJ 和 NOUN 需要单 token：值是 '{adj} {noun}'，G1 要求两个值
        # token 数相同。NAMES/ATTR/功能词可以多 token —— 它们在 base 和 edit
        # 里逐字相同，不影响长度差。
        # 两个条件分开报。先前把它们 and 成一个布尔，于是 NOUN 少一个存活
        # 就打印"池子够大 否"，而括号里同时显示 ADJ 145 >= 130 成立 ——
        # 自相矛盾的一行，且掩盖了"ADJ 其实够用、只有一个 noun 要处理"。
        adj_ok_n, noun_bad = len(adj_ok), [w for w in NOUN if w not in noun_ok]
        print(f"    ADJ 够大        {'是' if adj_ok_n >= need else '否'}"
              f"  {adj_ok_n} >= {need}"
              + ("" if adj_ok_n >= need else f"  缺 {need - adj_ok_n} 个"))
        print(f"    NOUN 全单 token {'是' if not noun_bad else '否'}"
              + (f"  多 token 的：{noun_bad}" if noun_bad else ""))
        if noun_bad:
            print(f"       -> 从池子里删掉它们即可（NOUN[val_id % "
                  f"{len(NOUN) - len(noun_bad)}] 与 % {len(NOUN)} 一样合法，"
                  f"池子大小不必是特定数）")
        # 过滤后可用的 n_values 上界
        print(f"    过滤后 n_values 可取 <= {adj_ok_n}"
              f"（当前 {spec.n_values}，下界 {need}）")
        pool_ok = adj_ok_n >= need and not noun_bad

        # ---- G1：base/edit token 数逐对守恒 ----
        # 用当前池子渲染的真实文档。若 ADJ/NOUN 有多 token 词，这里会破。
        worst, n_bad = 0, 0
        for tb, te in pairs:
            # 不要用 a / b 做局部名：a 是 argparse 的 namespace，被覆盖后
            # 下一轮的 get_tokenizer(nm, a.src) 会抛 AttributeError，而它
            # 被上面的 except Exception 吞掉并误报成 "tokenizer 不可用"。
            na = len(tok.encode(tb, add_special_tokens=False))
            nb = len(tok.encode(te, add_special_tokens=False))
            if na != nb:
                n_bad += 1
                worst = max(worst, abs(na - nb))
        print(f"    G1 守恒         {'过' if n_bad == 0 else '破'}"
              f"  n_bad={n_bad}/{len(pairs)} worst_diff={worst}")

        # ---- G4：值的 token 长度直方图 ----
        # str.split() 下是 {2: 150}。BPE 下若不是单一值，说明值长度可判别，
        # 而长度是模型能看见的量 —— 那是一条不经 slot 匹配的捷径。
        hist: Dict[int, int] = {}
        for w in ADJ[:spec.n_values]:
            n = len(tok.encode(" " + w, add_special_tokens=False))
            hist[n] = hist.get(n, 0) + 1
        print(f"    G4 值长度直方   {dict(sorted(hist.items()))}")

        # ---- 文档长度：影响 ctx 与显存 ----
        lens = [len(tok.encode(d["text"], add_special_tokens=False))
                for d in docs[:100]]
        print(f"    文档 token 数   均值 {sum(lens) / len(lens):.0f}"
              f"  最大 {max(lens)}"
              f"  （str.split() 下均值 380 最大 433）")

        # 三态而不是两态。"可修" 与 "不可用" 的区别是：前者只要过滤池子，
        # 后者要找新词 —— 成本差一个数量级，而先前的措辞把两者都叫"不可用"。
        if pool_ok and n_bad == 0:
            verdict[nm] = "可用（池子不必改）"
        elif adj_ok_n >= need:
            verdict[nm] = f"可修（过滤到 {adj_ok_n} 个单 token adj）"
        else:
            verdict[nm] = f"不可用（ADJ 只剩 {adj_ok_n} < {need}，要找新词）"
        print()

    print("=" * 60)
    for nm, v in verdict.items():
        print(f"  {nm:<32} {v}")
    print()
    print("判读")
    print("  可用  -> 池子不必改")
    print("  可修  -> 过滤 ADJ/NOUN 成单 token、n_values 降到存活数，重跑")
    print("          nl_gates 与 nl_collide。G1 破的机制就是多 token 值在")
    print("          编辑前后多重性变了、总长跟着变，过滤后自动消失。")
    print("  不可用 -> 该 tokenizer 下单 token 形容词不足 values_needed，")
    print("          要先扩 ADJ 再谈别的。")
    print()
    print("注意：过滤后的池子与 from-scratch NL 臂用的不同，两臂的语料因此")
    print("不逐字节相同。n_values 的**大小**可以对齐，池子的**成员**不能。")
    print("这是要报告的两臂差异，但它是背景常量而非承重协变量。")


if __name__ == "__main__":
    main()
