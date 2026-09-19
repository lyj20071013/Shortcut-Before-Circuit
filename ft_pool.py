"""Build single-token value pools for the selected tokenizer."""
import argparse
import os
from typing import List, Optional, Sequence, Tuple

MIRROR = "https://hf-mirror.com"

# install() 成功后被设上，供语料工件盖版本戳用。None 表示还没装。
ACTIVE_VERSION: Optional[str] = None

# 被过滤掉的词。verify_install 要它 —— 见那个函数里为什么"检查留下的词"
# 是一个永远为真的断言。
REMOVED: set = set()


def single_token(tok, w: str) -> bool:
    """带前导空格恰一个 token。读数位置在句中，故 ' pale' 才是实际形式。"""
    return len(tok.encode(" " + w, add_special_tokens=False)) == 1


def filtered(tok, need: int = 130) -> Tuple[List[str], List[str]]:
    """(adj 子集, noun 子集)，保序。保序是必要的：val_id -> adj 的映射由下标
    定，乱序会让同一个 val_id 在两次运行里指不同的词。"""
    from nl_generator import ADJ, NOUN

    a = [w for w in ADJ if single_token(tok, w)]
    n = [w for w in NOUN if single_token(tok, w)]
    if len(a) < need:
        raise ValueError(
            f"该 tokenizer 下只有 {len(a)} 个单 token adj < values_needed="
            f"{need}. Expand the adjective pool to preserve the document-length configuration. "
            f"Reducing n_stmts changes an additional experimental covariate.")
    if not n:
        raise ValueError("没有单 token 的 noun。")
    return a, n


def install(tok, n_values: Optional[int] = None, need: int = 130,
            model_name: Optional[str] = None) -> str:
    """把过滤后的池子装进 nl_generator，返回版本串。

    这是一次性的模块级替换，刻意做得吵：它改变 val_id -> adj 的映射，故任何
    在它之前生成的工件都与之后的不兼容。返回的版本串必须写进语料工件，
    ft_preflight 会核对。

    只能在任何语料生成之前调用一次。重复调用抛异常 —— 第二次调用会在已经
    过滤过的池子上再过滤，得到的下标映射与第一次不同，而已生成的探针集不会
    跟着变。
    """
    global ACTIVE_VERSION
    if ACTIVE_VERSION is not None:
        raise RuntimeError(
            f"install() 已经调用过（{ACTIVE_VERSION}）。它改变 val_id -> adj "
            f"的映射，二次调用会让先后生成的工件不兼容。")

    import nl_generator
    a, n = filtered(tok, need)
    # 默认用**全部**存活的 adj，不截到 need。
    #
    # 截到 130 会正好压在 values_needed(55,3)=130 这个下界上，余量为零 ——
    # 而那个下界的系数 2.0 是拿两个锚点标定的经验值（见 nl_generator.
    # values_needed 的 docstring），不是推导的。标定稍微乐观就会让
    # _ValueDraw 在生成中途耗尽，而那发生在语料建到一半的时候。
    #
    # Qwen2.5-0.5B 下存活 145，取全部给 15 的余量，且更接近 from-scratch 臂
    # 的 150 —— n_values 影响随机基线与 mass 的绝对水平，两臂靠近一点便于
    # 并列报告。
    n_val = n_values or len(a)
    if n_val > len(a):
        raise ValueError(f"n_values={n_val} > 单 token adj 数 {len(a)}")
    if n_val < need:
        raise ValueError(
            f"n_values={n_val} < values_needed={need}，_ValueDraw 会耗尽")

    global REMOVED
    old_a, old_n = len(nl_generator.ADJ), len(nl_generator.NOUN)
    # 补集要在替换之前算。verify_install 用它做非平凡的检查。
    REMOVED = (set(nl_generator.ADJ) - set(a)) | (set(nl_generator.NOUN) - set(n))

    # **原地**改，不重新绑定。
    #
    # 先前写的是 nl_generator.ADJ = a，那只换了 nl_generator 模块里的名字。
    # 任何在模块顶层做过 `from nl_generator import ADJ` 的模块持有旧列表对象的
    # 引用，重新绑定对它无效 —— nl_render 就是这样，于是 ft_posceil 报出
    # "被过滤掉的词仍出现在渲染结果里：['drab','dyed','feeble','flint',...]"。
    #
    # ft_data 与 ft_train 恰好躲开了，因为它们的 nl_render import 写在函数体内、
    # 在 install() 之后才执行。那让这个 bug 变成顺序依赖的陷阱：谁在
    # install() 之前 import 了 nl_render 就中招。
    #
    # 切片赋值改的是列表对象本身，所有引用（包括旧的）都看到新内容。
    for nm, obj in (("ADJ", nl_generator.ADJ), ("NOUN", nl_generator.NOUN)):
        if not isinstance(obj, list):
            raise TypeError(
                f"nl_generator.{nm} 是 {type(obj).__name__} 而不是 list，"
                f"无法原地改。若改成 tuple，就必须回到重新绑定，而那要求所有"
                f"消费者都在 install() 之后才 import —— 见上面为什么那个约定"
                f"不可靠。")
    nl_generator.ADJ[:] = a
    nl_generator.NOUN[:] = n

    # ModelScope 的 name_or_path 是本地快照目录，尾段是分支名（'master'），
    # 不是模型名 —— 实测得到 ft-master-...，而版本串必须标识模型，因为
    # Qwen 与 Pythia 的过滤池不同且两者要并列报告。
    # model_name 优先。下面那个启发式靠一张分支名黑名单，而黑名单会漏 ——
    # 调用方知道真名，传进来比猜可靠。
    if model_name:
        name = model_name.rstrip("/").split("/")[-1].split("@")[0]
    else:
        raw = str(getattr(tok, "name_or_path", "?")).replace("\\", "/").rstrip("/")
        segs = [s for s in raw.split("/") if s]
        name = "?"
        for s in reversed(segs):
            s = s.split("@")[0]                 # 'Qwen2.5-0.5B@master'
            if s and s not in ("master", "main", "snapshots", "models"):
                name = s
                break
    ACTIVE_VERSION = f"ft-{name}-a{len(a)}n{len(n)}v{n_val}"
    print(f"[ft_pool] ADJ {old_a} -> {len(a)}   NOUN {old_n} -> {len(n)}"
          f"   n_values {n_val}")
    print(f"[ft_pool] 版本 {ACTIVE_VERSION}")
    print(f"[ft_pool] 注意：val_id -> adj 的映射已改变。此前生成的探针集"
          f"（nl_data/）与此后的不兼容。")
    return ACTIVE_VERSION


def ft_spec(ctx_len: int = 768, n_values: Optional[int] = None):
    """微调臂的 LangSpec。必须在 install() 之后调用。

    ctx_len 768 而非 600：BPE 下文档从均值 380/最大 433 涨到 463/519
    （ft_tokcheck 实测），600 的余量只剩 13.5%。过滤后值全是单 token，长度会
    回落一些，但留余量比事后发现截断便宜。Qwen 支持 32k，这个数只约束我们
    自己的探针集校验。

    LangSpec 的替换走 dataclasses.replace，NamedTuple 走 _replace。我没有读过
    vocab.LangSpec 的定义，但 nl_corpus 用 spec.__dict__ 做 json dump，而
    NamedTuple 在 py3.8+ 没有可用的 __dict__ —— 所以它几乎确定是 dataclass。
    仍然保留第二条路径，但用显式分支而不是 hasattr 三元表达式（后者会调三次
    nl_spec 并把两条路径的逻辑分散在两个函数里）。
    """
    import dataclasses

    import nl_generator
    from nl_corpus import nl_spec

    if ACTIVE_VERSION is None:
        raise RuntimeError("先调 install()，否则拿到的是未过滤的池子。")

    s = nl_spec(ctx_len)
    # 与 install() 同一个默认：全部存活的 adj，不截到 need。两处必须一致，
    # 否则 install 装了 145 个而 spec 说 130，val_id 145 会渲染成越界索引。
    nv = n_values or len(nl_generator.ADJ)
    if dataclasses.is_dataclass(s):
        return dataclasses.replace(s, n_values=nv)
    if hasattr(s, "_replace"):
        return s._replace(n_values=nv)
    raise TypeError(f"不知道怎么替换 {type(s).__name__} 的字段")


def verify_install(r: int = 3, d: int = 8, n: int = 40) -> None:
    """确认渲染路径真的看到了过滤后的池子。

    install() 替换的是 nl_generator 的模块级名字。任何在**模块顶层**做
    `from nl_generator import ADJ` 的地方持有旧列表的引用，替换对它不生效 ——
    于是语料会一部分用过滤池、一部分用原池渲染，而这不会报错：G1 仍会破，
    但看起来像"过滤没修好 G1"而不是"过滤没生效"。

    检查的是**补集**，不是留下的词。先前我写的是"每个 adj 位置的词都在过滤后
    的池子里"，那是一个永远为真的断言：adj_slots 的实现是 pool = set(ADJ) 然后
    按 t in pool 筛位置，所以它按过滤后的池子**选**位置 —— 旧池子渲染出的
    'flint' 根本不会被报成一个 adj 位置，检查空过。

    正确的判据：REMOVED 里的词一个都不许出现在渲染出的 token 流里。那些词
    只可能来自未被替换的旧引用。
    """
    from generator import generate_corpus
    from nl_corpus import nl_corpus_cfg, tokenize
    from nl_render import render
    from vocab import Vocab

    if ACTIVE_VERSION is None:
        raise RuntimeError("install() 还没调用。")
    if not REMOVED:
        print("[ft_pool] verify_install：没有词被过滤掉，无可检查")
        return
    spec = ft_spec()
    v = Vocab(spec)
    seen = set()
    for doc in generate_corpus(v, nl_corpus_cfg(r, d, 7), n):
        rd = render(doc, v, tokenize)
        # 带句点的形式也要查：NOUN 在模板末尾是 'brick.' 而非 'brick'
        for t in rd["tokens"]:
            w = t[:-1] if t.endswith(".") else t
            if w in REMOVED:
                seen.add(w)
    if seen:
        raise RuntimeError(
            f"被过滤掉的词仍出现在渲染结果里：{sorted(seen)[:5]}"
            f"（共 {len(seen)} 个）。说明某处在模块顶层 import 了 ADJ/NOUN，"
            f"install() 对那个引用无效 —— 把那个 import 移进函数体。")
    print(f"[ft_pool] verify_install 过：{n} 篇里没有出现被过滤的 "
          f"{len(REMOVED)} 个词")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--src", default="hf", choices=["hf", "ms"])
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--pairs", type=int, default=400,
                    help="过滤后重测 G1 用的编辑对数")
    a = ap.parse_args()
    if a.src == "hf" and not a.no_mirror:
        os.environ.setdefault("HF_ENDPOINT", MIRROR)

    from ft_tokcheck import get_tokenizer
    from nl_generator import values_needed
    need = values_needed(55, 3)
    tok, via = get_tokenizer(a.model, a.src)
    print(f"{a.model}  来源 {via}  vocab {len(tok):,}  need {need}\n")

    ver = install(tok, need=need, model_name=a.model)

    # 先确认替换真的生效，再谈 G1。顺序很重要：若 install 没生效而我们直接
    # 测 G1，看到的 "仍破" 会被误读成"过滤修不了 G1"，而真因是"过滤没生效"。
    verify_install()

    # 过滤后重测 G1。这是这个脚本存在的理由 —— 若过滤没修好 G1，说明长度差
    # 还有别的来源（模板？attr？名字所有格？），那要先找出来。
    from nl_corpus import gate_pairs
    spec = ft_spec()
    print(f"\n重测 G1（n_values={spec.n_values}, ctx_len={spec.ctx_len}）")
    pairs = gate_pairs(3, 8, a.pairs, seed=7, spec=spec)
    worst = n_bad = 0
    for tb, te in pairs:
        na = len(tok.encode(tb, add_special_tokens=False))
        nb = len(tok.encode(te, add_special_tokens=False))
        if na != nb:
            n_bad += 1
            worst = max(worst, abs(na - nb))
    print(f"  G1  n_bad={n_bad}/{len(pairs)}  worst_diff={worst}"
          f"   {'过' if n_bad == 0 else '仍破'}")

    from nl_corpus import gate_docs
    docs = gate_docs(3, 8, 200, seed=7, spec=spec)
    lens = [len(tok.encode(d["text"], add_special_tokens=False)) for d in docs]
    print(f"  文档 token 数  均值 {sum(lens) / len(lens):.0f}  最大 {max(lens)}"
          f"   ctx_len {spec.ctx_len}"
          f"   {'够' if max(lens) <= spec.ctx_len else '不够'}")

    hist = {}
    import nl_generator
    for w in nl_generator.ADJ[:spec.n_values]:
        k = len(tok.encode(" " + w, add_special_tokens=False))
        hist[k] = hist.get(k, 0) + 1
    print(f"  G4 值长度直方  {dict(sorted(hist.items()))}"
          f"   {'过' if list(hist) == [1] else '仍有多 token 值'}")

    print(f"\n版本串 {ver}")
    if n_bad == 0 and list(hist) == [1] and max(lens) <= spec.ctx_len:
        print("三项全过。微调臂的语料层可以建了。")
        print("下一步：用这个 spec 建 ft_data/ 的词表与探针集，然后 ft_preflight。")
    else:
        print("有项未过，不要建语料。")


if __name__ == "__main__":
    main()
