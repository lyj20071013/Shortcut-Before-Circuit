"""Render synthetic assignment documents as English statements."""
from typing import Dict, List, Optional, Sequence, Tuple

from generator import Doc, Stmt
from nl_generator import ADJ, ATTR, NAMES, NOUN, POSS, TMPL, stmt_with
from vocab import Vocab

QUERY_TMPL = "{subj}'s {attr} is"


def check_pools(spec, n_stmts_hi: int = 55, r_old_lo: int = 3) -> None:
    """spec 与词表池的相容性。生成前炸掉，不要在数据里留静默失真。

    两条独立的下界，都作用在 n_values 上：
      单射   n_values <= len(ADJ)，否则两个值渲染成同一个 adj
      不耗尽 n_values >= values_needed(...)，否则 _ValueDraw 在生成中途炸

    第二条先前不在这里，于是 n_values=100 通过了 check_pools 却在
    generator._ValueDraw.take 里炸，调用方看不出该扩多少。r_old_lo 而非 hi：
    消耗量随 R_old **减小**而增大（slot 数变多），最坏情形在轴的低端。
    """
    from nl_generator import values_needed

    if spec.n_values > len(ADJ):
        raise ValueError(
            f"n_values={spec.n_values} > len(ADJ)={len(ADJ)}。adj 必须对 "
            f"val_id 单射，否则两个不同的值渲染成同一个 adj，而读数在 adj "
            f"位置取 —— 那两个值在读数位置不可分，Δ 恒为 0 且原因是映射碰撞。")
    need = values_needed(n_stmts_hi, r_old_lo)
    if spec.n_values < need:
        raise ValueError(
            f"n_values={spec.n_values} < {need}，在 n_stmts_hi={n_stmts_hi}、"
            f"R_old={r_old_lo} 下 _ValueDraw 会耗尽（不变量 3：文档内值不"
            f"重复）。扩 ADJ 到至少 {need} 个，或降 n_stmts_hi —— 但后者动的是"
            f"文档长度这一额外协变量，"
            f"会让 NL 臂在长度维度上与主网格不可比。")
    if spec.n_entities > len(NAMES):
        raise ValueError(
            f"n_entities={spec.n_entities} > len(NAMES)={len(NAMES)}")
    if spec.n_attrs > len(ATTR):
        raise ValueError(f"n_attrs={spec.n_attrs} > len(ATTR)={len(ATTR)}")


def _tmpl_idx(ent: int, attr: int) -> int:
    """模板索引。只用 (ent, attr)，**不用 val**。三个理由，缺一不可：

    不用 rng —— probe.py 的编辑改 Doc.stmts 后重新渲染，现抽会让同一篇文档
    编辑前后抽到不同模板。

    不用 val —— 四个模板的 token 数不同（T0 是 6，T1/T2/T3 是 8，均值 7.5，
    与 G4 实测的 q_mean_len=7.497 吻合）。而 break_rarity 这类编辑正是改
    val，于是某条语句从 T0 跳到 T1，token 数差 2，G1 按构造失败。

    不用 position —— shift_delta 搬移语句位置，模板随位置变则搬移改句长。

    (ent, attr) 是编辑从不触碰的部分：七条编辑改的是值、多重性、位置，
    没有一条改 slot 身份。所以绑定到它就是绑定到编辑的不动点。

    代价：同一 slot 的 R+1 条语句用同一模板。这不构成判别式（(ent,attr)
    均匀随机，故每个 slot 的模板均匀），但让句长在 slot 内不变化 ——
    posCeil 的变异性来自尾部那 ΔD 条填充语句的不同 slot，仍然保留。
    """
    return (ent * 31 + attr * 7) % len(TMPL)


def render_stmt(st: Stmt, vocab: Vocab) -> str:
    """Stmt.ent/attr/val 是 **raw 索引**，不是 token id。

    依据：generator.py:369-370 只对 answer 做 vocab.val() 转换
    （`answer=vocab.val(answer_val)`），Stmt 直接进 Doc.stmts；
    probe.py:522 构造 `Stmt(e, a, v_last, True)`，其中 e,a 来自与
    (d.q_ent, d.q_attr) 的比较、v_last 来自 d.stmts[i].val —— 同一个空间。
    emit() 在 token 化时才调 vocab.ent()/attr()/val()。

    先前这里写了 `st.ent - vocab.ENT0`，于是 attr 变成 0..7-49 = 负数，
    ATTR[-49] 越界 IndexError。check_pools 放行是因为它查的是 spec 的
    n_attrs=8 <= len(ATTR)=8，那一条本身没错。
    """
    e, a, v = st.ent, st.attr, st.val
    return stmt_with(_tmpl_idx(e, a), NAMES[e], ATTR[a],
                     (ADJ[v], NOUN[v % len(NOUN)]))


def render(doc: Doc, vocab: Vocab, tokenize) -> dict:
    """Doc -> {text, tokens, answer_pos, answer, ...}。

    tokens 是自然语言 token 的**字符串**列表；调用方用 nl_corpus.encode 映射
    到 id。answer_pos 指向答案 token，与 Doc.answer_pos 的约定相同。

    doc.q_hist_k 必须是 None：hist 查询下真值是老值，而 QUERY_TMPL 没有时间
    索引的位置。主网格的 p_hist_query=0.0，所以这条在相图上不触发。
    """
    if doc.q_hist_k is not None:
        raise ValueError(
            "q_hist_k 非 None：NL 表层没有时间索引的写法。主网格 "
            "p_hist_query=0.0，此路径不该被触发。")
    texts = [render_stmt(st, vocab) for st in doc.stmts]
    # q_ent / q_attr 是 raw 索引（probe.py:508 拿它们与 (s.ent, s.attr) 直接
    # 比较）。doc.answer 相反，是 token id —— generator.py:370 写着
    # `answer=vocab.val(answer_val)`。两者语义不同，这是 Doc 里唯一的不对称。
    e, a = doc.q_ent, doc.q_attr
    q = QUERY_TMPL.format(subj=NAMES[e], attr=ATTR[a])
    ans = ADJ[vocab.val_index(doc.answer)]

    toks = tokenize(" ".join(texts) + " " + q) + [ans]
    return dict(text=" ".join(texts) + " " + q, tokens=toks,
                answer_pos=len(toks) - 1, answer=ans,
                # 下面是给门与读数用的结构字段，全部来自 Doc，不重算
                q_slot=(NAMES[e], ATTR[a]),
                stmts=[dict(slot=(NAMES[st.ent], ATTR[st.attr]),
                            value=f"{ADJ[st.val]} "
                                  f"{NOUN[st.val % len(NOUN)]}",
                            text=t)
                       for st, t in zip(doc.stmts, texts)],
                realized_delta=doc.realized_delta, n_stmts=doc.n_stmts,
                # 下面三个是 Doc 独有的结构协变量。EditedDoc 是"与 Doc 共享
                # 规则函数所需的最小字段集"（probe.py:87），只有 14 个字段，
                # 不含它们 —— 它带一个 base: Doc 引用代替。所以 render 对
                # EditedDoc 报 None，需要这些量时从 .base 取。
                q_kept=getattr(doc, "q_kept", None),
                n_slots=getattr(doc, "n_slots", None),
                q_gap=getattr(doc, "q_gap", None))


def render_edit_pair(base: Doc, edit: Doc, vocab: Vocab,
                     tokenize) -> Tuple[dict, dict]:
    """一对 (base, edit) 的渲染。两侧 token 数必须相等 —— 这是 G1，也是
    probe.py:26 的硬约束之一。

    _tmpl_idx 只用 (ent, attr)，而七条编辑没有一条改 slot 身份，所以逐位置
    的模板两侧相同；所有值都渲染成两 token。于是 token 数相等**应当**成立。

    仍然实测而不是推理：bump_freq / drop_freq 增删语句副本，语句数本身会变，
    那时两侧长度不等是编辑的性质而非渲染的 bug。probe.py:26 说编辑后语句数
    不变，但那是对主网格 token 化的断言，NL 表层下必须自己验一遍。
    """
    rb, re = render(base, vocab, tokenize), render(edit, vocab, tokenize)
    if len(rb["tokens"]) != len(re["tokens"]):
        raise ValueError(
            f"编辑对 token 数不等 {len(rb['tokens'])} vs {len(re['tokens'])}。"
            f"_tmpl_idx 依赖 val，而编辑改了 val，于是某条语句换了模板而四个"
            f"模板长度不同。修法：_tmpl_idx 只用 (ent, attr)，不用 val。")
    return rb, re
