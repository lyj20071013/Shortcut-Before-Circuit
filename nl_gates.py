"""Check rendered-language construction and edit invariants."""
import argparse
import sys
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Tokenize = Callable[[str], List[str]]


def whitespace_tok(s: str) -> List[str]:
    """占位分词器。真实臂应传入实际使用的那一个 —— 门的结论只对被检查的
    分词器有效，换分词器必须重跑。"""
    return s.split()


def g1_token_conservation(pairs: Sequence[Tuple[str, str]],
                          tok: Tokenize) -> dict:
    """G1：逐篇比对 base 与 edit 的 token 数。

    任何一篇不等即失败 —— 不设容差，因为答案位置错一个 token 就够让 Δ
    混进位置效应。

    同时检查 base != edit。只比 token 数的话，一个什么都没改的生成器会
    平凡通过这道门 —— 与 coext_audit 里平局子集恒读 1.000 同一类陷阱：
    一个恒等的比较必然"通过"，而通过的原因是没有比较发生。
    """
    bad, worst, identical = [], 0, 0
    for i, (b, e) in enumerate(pairs):
        nb, ne = len(tok(b)), len(tok(e))
        if nb != ne:
            bad.append((i, nb, ne))
            worst = max(worst, abs(nb - ne))
        if b == e:
            identical += 1
    return dict(name="G1 token 数守恒", ok=(not bad and identical == 0),
                n_bad=len(bad), n_identical=identical,
                n_total=len(pairs), worst_diff=worst, examples=bad[:5],
                note="n_identical>0 表示重数反转没有真的发生，"
                     "这道门是平凡通过的")


def g2_ltk_not_coextensive(docs: Sequence[dict]) -> dict:
    """G2：查询前最后一条语句是否为被查询 slot。

    docs 每项需含 'stmts'（语句列表，每项含 slot 标识与值）与 'q_slot'。
    与 collide.py:49 同口径：LTK 预测 = 最后一条语句的值。
    """
    n_coext = 0
    for d in docs:
        if not d["stmts"]:
            continue
        if d["stmts"][-1]["slot"] == d["q_slot"]:
            n_coext += 1
    rate = n_coext / len(docs) if docs else float("nan")
    return dict(name="G2 LTK 非共延", ok=(rate <= 0.005),
                ltk_coext_rate=rate, n_coext=n_coext, n_total=len(docs),
                note="主网格全 25 格 0.000；>0.005 即读数与'复制最后一个"
                     "值 token'开始混淆")


def g3_pos_ceiling(docs: Sequence[dict], tok: Tokenize,
                   supp: int) -> dict:
    """G3：经验 posCeil，并与解析值 1/|supp| 对照。

    对每个负向 token 偏移 k，算"取倒数第 k 个 token 作为答案"的命中率，
    取最大值。这是纯位置规则在这个语料上的实际上限，替代 sweep.py:96-99
    的 1/|supp|（那个式子依赖每条语句恰 4 token）。

    判据：经验值应当 <= 解析值。句长可变让固定偏移更难命中，所以变长语料
    的位置上限不应高于定长语料。若反而更高，说明句长分布有规律使答案位置
    比 ΔD 本身更可预测 —— 那是设计缺陷，会在低 ΔD 行伪造出位置态。

    每篇只分词一次。之前的写法对同一篇文档重复分词 60 次，在 2000 篇上是
    12 万次多余调用。
    """
    toks_all = [tok(d["text"]) for d in docs]
    answers = [d["answer"] for d in docs]
    # 不设固定上限。之前写的 min(60, ...) 把搜索截断在 60，而 ΔD=16 的最优
    # 偏移在 68 附近（约 1+ΔD 条语句 × 7.5 token），于是那些格的 posCeil
    # 被低估约 40 倍。上限只能由文档长度决定。
    max_off = max((len(t) for t in toks_all), default=0)
    best_k, best_hit = None, 0.0
    for k in range(1, max_off + 1):
        hit = sum(1 for t, ans in zip(toks_all, answers)
                  if len(t) >= k and t[-k] == ans)
        r = hit / len(docs) if docs else 0.0
        if r > best_hit:
            best_hit, best_k = r, k
    analytic = 1.0 / supp if supp else float("nan")
    return dict(name="G3 经验 posCeil", ok=(best_hit <= analytic + 1e-9),
                emp_posceil=best_hit, best_offset=best_k,
                analytic_posceil=analytic, supp=supp,
                note="经验值 > 解析值：句长分布让答案位置比 ΔD 更可预测，"
                     "会影响低 ΔD 行的状态分类。此时经验值应替换 "
                     "1/|supp|，它进三态分类器（sweep.py:109）")


def g4_length_discriminant(docs: Sequence[dict], tok: Tokenize) -> dict:
    """G4：句长能否判别答案所在语句、或判别反转 slot。

    两项：
      per_value_len  各值的 token 数是否一致。不一致 -> 值本身可由长度识别
      q_vs_filler    被查询 slot 的语句长度分布 vs filler slot 的
    """
    vlen: Dict[str, int] = {}
    bad_v = []
    for d in docs:
        for s in d["stmts"]:
            v = str(s["value"])
            n = len(tok(v))
            if v in vlen and vlen[v] != n:
                bad_v.append((v, vlen[v], n))
            vlen[v] = n
    lens = Counter(vlen.values())

    ql, fl = [], []
    for d in docs:
        for s in d["stmts"]:
            n = len(tok(s["text"])) if "text" in s else None
            if n is None:
                continue
            (ql if s["slot"] == d["q_slot"] else fl).append(n)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    # 缺 'text' 字段时 ql/fl 为空，gap 为 nan。之前的写法让 nan 通过
    # （`gap != gap` 为真即放行），于是"没测到"和"测到且合格"同样通过。
    # 缺字段必须失败：门的价值在于它会拒绝。
    measurable = bool(ql) and bool(fl)
    gap = abs(mean(ql) - mean(fl)) if measurable else float("nan")
    ok = (len(lens) == 1) and (not bad_v) and measurable and gap < 0.05
    return dict(name="G4 长度非判别式", ok=ok,
                value_len_hist=dict(lens), n_inconsistent=len(bad_v),
                inconsistent_examples=bad_v[:5],
                measurable=measurable, n_q_stmts=len(ql), n_fill_stmts=len(fl),
                q_mean_len=mean(ql), filler_mean_len=mean(fl), gap=gap,
                note="值 token 数必须唯一；被查询 slot 与 filler 的句长"
                     "均值差应 <0.05。measurable=False 表示 stmts 缺 'text' "
                     "字段，门无法执行 —— 这算失败而不是通过")


def g5_eq1_coextensive(docs: Sequence[dict]) -> dict:
    """G5：式 \\ref{eq:alias} 在这个语料上是否仍是恒等式。

    这是全篇的地基，也是 B1 相对于真实语料的唯一优势 —— 换表层不该动它。
    collide.py 的 RAR 列在主网格全 25 格读 1.000；这里必须同样读 1.000，
    否则 §sec:flat 的论证在这个臂上不成立。

    规则计算复用 coext_audit._predict，那份代码与 collide.py:34-44 逐行
    对照过，且平局约定相同（取更靠后）。不重新实现 —— 重新实现会得到一个
    与 tab:collide 不可比的数，而这道门的全部意义就是可比。
    """
    try:
        from coext_audit import _predict
    except ImportError:
        return dict(name="G5 式 1 恒等", ok=False,
                    note="coext_audit 不可导入，无法与 tab:collide 同口径")

    n_agree = n_tie = 0
    bad = []
    for i, d in enumerate(docs):
        vals = [str(s["value"]) for s in d["stmts"]
                if s["slot"] == d["q_slot"]]
        if len(vals) < 2:
            continue
        p = _predict(vals)
        if p["tie"]:
            n_tie += 1
        if p["rarity"] == p["recency"]:
            n_agree += 1
        else:
            bad.append((i, vals))
    n = sum(1 for d in docs
            if sum(1 for s in d["stmts"] if s["slot"] == d["q_slot"]) >= 2)
    rate = n_agree / n if n else float("nan")
    return dict(name="G5 式 1 恒等", ok=(n == len(docs) and rate >= 0.9995),
                rar_recency_rate=rate, n_checked=n, n_docs=len(docs),
                n_tie=n_tie, disagree_examples=bad[:5],
                note="必须 1.000 且 n_checked == n_docs。前者不成立则"
                     "§sec:flat 在此臂失效；后者不成立说明有文档的被查询 "
                     "slot 少于两个值，那些文档没有取代事件。n_tie 应为 0："
                     "网格恒在非平局区（旧值 R 份、末值 1 份）")


def g6_slot_multiplicity(docs: Sequence[dict]) -> dict:
    """G6：被查询 slot 能否由"语句数最多"识别出来。

    generator.py 的第 5 条不变量：q_slot 与填充 slot 的局部结构同分布，
    否则可绕过 query 定位答案。主网格靠 p_update=0.5 保证约一半 filler slot
    也有 update 事件、各带 R_old 份旧值副本，于是"多次出现的 slot"到处都是。

    论文已有量这件事的仪器：train.py 的 swap_query 换掉 query 后看预测是否
    不变，主网格实测 0.0025（附录 app:constlr 那段）。本门是它的生成侧
    先验版 —— 纯 CPU，不需要模型。

    两个数：
      uniq_argmax  被查询 slot 是语句数唯一最大者的比例。1.000 = 完美判别式
      n_slots      每篇的不同 slot 数，与主网格的约 11.5 对照
    """
    uniq, n_slots = 0, []
    for d in docs:
        cnt = Counter(s["slot"] for s in d["stmts"])
        n_slots.append(len(cnt))
        mx = max(cnt.values())
        top = [s for s, c in cnt.items() if c == mx]
        if len(top) == 1 and top[0] == d["q_slot"]:
            uniq += 1
    rate = uniq / len(docs) if docs else float("nan")
    mean_slots = sum(n_slots) / len(n_slots) if n_slots else float("nan")
    return dict(name="G6 slot 语句数非判别式", ok=(rate <= 0.05),
                uniq_argmax=rate, mean_n_slots=mean_slots,
                note="uniq_argmax 高 -> 模型可靠'找出现多次的 slot'定位答案"
                     "而无需读 query，swap_query 会读到接近 1.0（主网格 "
                     "0.0025）。修法：filler slot 也按 p_update 带 update "
                     "事件，各带 R_old 份旧值。mean_n_slots 应接近主网格的 "
                     "约 11.5（tab:covar），48 说明 filler 全是单次 slot。")


def g7_value_uniqueness(docs: Sequence[dict]) -> dict:
    """G7：generator.py 第 3 条 —— 文档内值不跨 slot 重复。

    同一 slot 内旧值重复 R 次是 R_old 的定义，不违反这条；这条禁的是两个
    不同 slot 用同一个值。违反时 rarity 的计数域被污染：模型数"这个值出现
    几次"会跨 slot 累加。

    G5 只看被查询 slot 的值序列，不查这条。上一轮加 used_val 时我说它修了
    第 3 条，但没有门验证它 —— 这个门补上。
    """
    bad, n_pairs = [], 0
    for i, d in enumerate(docs):
        by_slot = {}
        for s in d["stmts"]:
            by_slot.setdefault(s["slot"], set()).add(s["value"])
        slots = list(by_slot)
        for a in range(len(slots)):
            for b in range(a + 1, len(slots)):
                n_pairs += 1
                sh = by_slot[slots[a]] & by_slot[slots[b]]
                if sh:
                    bad.append((i, slots[a], slots[b], sorted(sh)[:2]))
    return dict(name="G7 值不跨 slot 重复", ok=not bad, n_bad=len(bad),
                n_slot_pairs=n_pairs, examples=bad[:3],
                note="两个 slot 共用一个值 -> rarity 的计数域跨 slot 污染")


def g8_update_position_uniform(docs: Sequence[dict]) -> dict:
    """G8：generator.py 第 4 条 —— update 位置均匀，不随 R_old 向末尾聚集。

    主网格用 Doc.fill_quint（填充语句 update 占比，按最终位置五分位）量这件
    事。这里同构：把每篇的 update 点（各 slot 的末次赋值位置，含 q_slot）
    按位置五分位归桶，看分布是否平坦。

    判据是最松的一档：最低桶占比 >= 0.5/5 = 0.10。完全均匀是 0.20 每桶。
    _take_group 要求"末次赋值点前面有至少 r 个可用位置"，所以最前面约 r 个
    位置永远不能是 update 点，第一桶会偏低，且偏低程度随 R_old 增长 ——
    这正是第 4 条禁止的。这个门量它有多严重。
    """
    q = [0.0] * 5
    tot = 0
    for d in docs:
        n = d["n_stmts"]
        last = {}
        for i, s in enumerate(d["stmts"]):
            last[s["slot"]] = i
        for slot, pos in last.items():
            cnt = sum(1 for s in d["stmts"] if s["slot"] == slot)
            if cnt < 2:          # 未被更新的 slot 没有 update 事件
                continue
            b = min(4, int(5 * pos / n))
            q[b] += 1
            tot += 1
    frac = [x / tot for x in q] if tot else [float("nan")] * 5
    lo = min(frac) if tot else float("nan")
    return dict(name="G8 update 位置均匀", ok=(lo >= 0.10),
                quintiles=[round(x, 4) for x in frac], n_updates=tot,
                min_bin=round(lo, 4) if tot else lo,
                note="完全均匀是每桶 0.20。第一桶偏低 = update 点被排除在"
                     "文档开头，偏低程度随 R_old 增长即违反第 4 条。主网格"
                     "用 _keyed 的画布坐标 + 越界丢弃解决，见 Doc.key_quint")


def g9_local_structure(docs: Sequence[dict]) -> dict:
    """G9：generator.py 第 6-7 条点名的两个量 —— 老值散布宽度、update 前驱
    是否同 slot。G6 查的是语句数，这两个它没查。

    q_gap    q_final 到最近同 slot 老值的距离（Doc.q_gap）
    fill_gap 填充 slot 的同一距离（Doc.fill_gap_late）
    adj_q    q_final 的前驱是否同 slot（Doc.adj_q）
    adj_fill 填充 update 的同一比率（Doc.adj_fill）

    两对都必须接近。差得远则"被查询 slot 的局部结构"本身是判别式，模型
    可绕过 query 定位答案 —— 与 G6 同一类漏洞，只是量的维度不同。
    """
    qg, fg, aq, af = [], [], [], []
    for d in docs:
        pos_by_slot = {}
        for i, s in enumerate(d["stmts"]):
            pos_by_slot.setdefault(s["slot"], []).append(i)
        for slot, ps in pos_by_slot.items():
            if len(ps) < 2:
                continue
            pf = ps[-1]
            gap = pf - ps[-2]
            adj = int(pf > 0 and d["stmts"][pf - 1]["slot"] == slot)
            if slot == d["q_slot"]:
                qg.append(gap)
                aq.append(adj)
            else:
                fg.append(gap)
                af.append(adj)

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    gq, gf = mean(qg), mean(fg)
    rq, rf = mean(aq), mean(af)
    d_gap = abs(gq - gf) if qg and fg else float("nan")
    d_adj = abs(rq - rf) if aq and af else float("nan")
    ok = bool(qg) and bool(fg) and d_gap < 1.5 and d_adj < 0.05
    return dict(name="G9 局部结构同分布", ok=ok,
                q_gap=round(gq, 3), fill_gap=round(gf, 3),
                d_gap=round(d_gap, 3) if d_gap == d_gap else d_gap,
                adj_q=round(rq, 4), adj_fill=round(rf, 4),
                d_adj=round(d_adj, 4) if d_adj == d_adj else d_adj,
                n_q=len(qg), n_fill=len(fg),
                note="q_gap 与 fill_gap 差 >1.5 条语句、或 adj 差 >0.05，"
                     "则被查询 slot 的局部结构是判别式（第 6-7 条）")


def run_gates_impl(docs, pairs, tok, supp):
    """九门。渲染层切换之后它们分成两类，判读方式不同。

    渲染门（G1 G3 G4）—— 量的是表层，只有本臂有，必须过：
      G1 base/edit token 数守恒。渲染是 Doc 的纯函数且 _tmpl_idx 不含 val，
         但 bump_freq 那类增删语句的编辑会改语句数，故仍需实测
      G3 经验 posCeil。1/|supp| 依赖定长语句，NL 表层必须测
      G4 句长非判别式

    继承门（G2 G5 G6 G7 G8 G9）—— 量的是放置，而放置来自主网格且已被
    selfcheck.py 验过。它们现在是**交叉验证**：过了说明渲染没有破坏继承的
    结构，不过说明渲染层有 bug（比如 slot 或值的映射不是单射）。成本为零，
    所以保留。

    G8 有一处测量偏差要记着：Doc.fill_quint 的定义是"**填充**语句 update
    占比按最终位置五分位"，而本实现把 q_slot 也算进去了。q_slot 的 p_final
    被 ΔD 钉死在 n_stmts-1-dd，按设计就不可能均匀，它贡献约 1/(1+n_upd_fill)
    的样本且全在末桶。所以 G8 现在偏严；若它在渲染层下失败，先剔掉 q_slot
    再判，不要先改生成器。
    """
    return [g1_token_conservation(pairs, tok),
            g2_ltk_not_coextensive(docs),
            g3_pos_ceiling(docs, tok, supp),
            g4_length_discriminant(docs, tok),
            g5_eq1_coextensive(docs),
            g6_slot_multiplicity(docs),
            g7_value_uniqueness(docs),
            g8_update_position_uniform(docs),
            g9_local_structure(docs)]


# G2 也在这里，虽然它量的是放置而不是表层。理由是它的判据有外部依据而不是
# 我拍的：tab:collide 的 LTK 列在主网格全 25 格读 0.000，所以 <=0.005 是从
# 论文来的绝对阈值，不需要对照。
#
# 更实际的原因：先前这个集合只有 G1/G3/G4，而 COMPARATIVE 只有 G5/G6/G8/G9，
# G2 两边都不在 —— run_gates_impl 照样计算它，然后被 main 静默丢弃。一个算了
# 却没人看的门比没有门更糟，因为它让人以为查过了。
RENDER_GATES = {"G1 token 数守恒", "G2 LTK 非共延", "G3 经验 posCeil",
                "G4 长度非判别式"}

# 对照门：在主网格 Doc 上算同一个量，判据是**差值**而不是绝对阈值。
#
# 为什么必须这样。上一轮 G5/G6/G8/G9 全部"失败"，而四个里三个是判据错：
#   G5 n_tie=334 被判失败，但它是正确的 —— q_kept 因越界丢弃可低于名义 R，
#      丢到 1 份时 counts 变成 old=1 new=1，平局，rarity 的平局规则取更靠后
#      = new = recency。这正是主网格 FRQ 列非零的机制，而 NL 臂的 FRQ 现在
#      读 0.077-0.166，落在主网格 tab:collide 的 0.001-0.309 区间内。
#   G6 uniq_argmax=0.058 对阈值 0.05。那个阈值是拍的：主网格的 swap_query
#      是 0.0025，但它量"换 query 后模型预测是否不变"（需要模型），与本门量
#      的"语句数 argmax 是否唯一指向 q_slot"（纯生成侧）不是同一个量。
#   G9 d_gap=2.379 对阈值 1.5。而 Doc.q_gap 与 Doc.fill_gap_late 在论文里是
#      并列报告的协变量（tab:covar）—— 若它们相等就不必分两列报。
# 只有 G8 可能是真问题，但本实现把 q_slot 也算进五分位，而 Doc.fill_quint
# 的定义只算填充语句。
#
# 渲染层之后这些量全部继承自主网格，所以唯一有意义的判读是"渲染是否改变了
# 它们"，而不是"它们是否满足我想象的标准"。
COMPARATIVE = {"G5 式 1 恒等": 0.02, "G6 slot 语句数非判别式": 0.02,
               "G8 update 位置均匀": 0.05, "G9 局部结构同分布": 0.10}

# 每个对照门取哪个标量做比较
CMP_KEY = {"G5 式 1 恒等": "rar_recency_rate",
           "G6 slot 语句数非判别式": "uniq_argmax",
           "G8 update 位置均匀": "min_bin",
           "G9 局部结构同分布": "d_gap"}


def project_doc(doc, vocab) -> dict:
    """主网格 Doc -> 与 nl_render.render 同形的字典（无 text 字段）。

    对照门在两侧算同一个函数，所以两侧的输入必须同形。值用 raw id 的字符串
    形式：门只做相等比较与计数，而 adj 对 val_id 单射，故 str(val) 与
    "pale blue" 这两种表示给出相同的相等关系。
    """
    return dict(q_slot=(doc.q_ent, doc.q_attr),
                stmts=[dict(slot=(s.ent, s.attr), value=str(s.val))
                       for s in doc.stmts],
                n_stmts=doc.n_stmts, answer=str(vocab.val_index(doc.answer)))


def run_comparative(nl_docs: Sequence[dict], grid_docs: Sequence[dict]
                    ) -> List[dict]:
    """四个对照门。返回带 nl / grid / diff / tol 的记录。"""
    fns = {"G5 式 1 恒等": g5_eq1_coextensive,
           "G6 slot 语句数非判别式": g6_slot_multiplicity,
           "G8 update 位置均匀": g8_update_position_uniform,
           "G9 局部结构同分布": g9_local_structure}
    out = []
    for name, fn in fns.items():
        a, b = fn(nl_docs), fn(grid_docs)
        k = CMP_KEY[name]
        va, vb = a.get(k), b.get(k)
        tol = COMPARATIVE[name]
        bad = (va is None or vb is None or va != va or vb != vb)
        diff = abs(va - vb) if not bad else float("nan")
        out.append(dict(name=f"{name}（对照）", ok=(not bad and diff <= tol),
                        key=k, nl=va, grid=vb, diff=diff, tol=tol,
                        nl_full=a, grid_full=b,
                        note="判据是 NL 与主网格的差值。超出容差说明渲染改变"
                             "了这个结构量；两侧都偏离我原先拍的绝对阈值是"
                             "正常的 —— 那些阈值没有依据。"))
    return out


def run_gates(docs: Sequence[dict], pairs: Sequence[Tuple[str, str]],
              tok: Tokenize, supp: int) -> Tuple[bool, List[dict]]:
    res = run_gates_impl(docs, pairs, tok, supp)
    return all(r["ok"] for r in res), res


def report(res: Sequence[dict], r: int, d: int) -> None:
    print(f"\n=== cell R{r}_D{d} ===")
    for x in res:
        flag = "ok  " if x["ok"] else "FAIL"
        print(f"[{flag}] {x['name']}")
        for k, v in x.items():
            if k in ("name", "ok", "note", "examples"):
                continue
            print(f"         {k} = {v}")
        if not x["ok"] and x.get("examples"):
            print(f"         样例 {x['examples']}")
        if not x["ok"] and x.get("note"):
            print(f"         -> {x['note']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--rows", type=int, nargs="+", default=[3])
    ap.add_argument("--cols", type=int, nargs="+", default=[8])
    a = ap.parse_args()

    # 必须走 nl_corpus 的适配器，不能 import nl_generator.make_docs：那个
    # 平行生成器已被渲染层取代，但函数还在文件里，import 不会报错 —— 门会
    # 静默地量一个不再被训练的语料。G3 的 posCeil 由此进 sweep.classify
    # 决定哪些 run 进网格，量错了整条链都错。
    try:
        from nl_corpus import gate_docs as make_docs
        from nl_corpus import gate_pairs as make_edit_pairs
    except ImportError:
        print(__doc__)
        print("=" * 66)
        print("nl_generator 还不存在。本文件是门，不是生成器。")
        print("生成器需提供两个函数：")
        print("  make_docs(r, d, n, seed) -> [dict]，每项含")
        print("      text     完整文档字符串（含查询）")
        print("      answer   真值 token（单 token）")
        print("      q_slot   被查询 slot 的标识")
        print("      stmts    [{slot, value, text}]，按文档顺序")
        print("  make_edit_pairs(r, d, n, seed) -> [(base_text, edit_text)]")
        print("      重数反转前后的同一篇文档，token 数必须相同（G1 检查）")
        print("")
        print("先写生成器，再跑本门。五门决定生成器的设计约束：")
        print("  G1 -> 值必须单 token 或等 token 数，且 base != edit")
        print("  G2 -> 查询前最后一条语句不得是被查询 slot")
        print("  G3 -> posCeil 要测，且经验值不得高于 1/|supp|")
        print("  G4 -> 值 token 数唯一，且查询 slot 与 filler 句长同分布")
        print("  G5 -> 式 1 仍须逐篇成立（RAR=1.000，n_tie=0）")
        sys.exit(2)

    # supp 必须来自生成器实际使用的那个带，否则 G3 的解析对照是假的。
    # dd_band 是主网格的来源（config.py:140），NL 臂沿用它才可比。
    from config import dd_band

    # 主网格文档，用于对照门。同 cfg 同 seed，只是不经渲染。
    from generator import generate_corpus
    from nl_corpus import nl_corpus_cfg, nl_spec
    from vocab import Vocab

    spec = nl_spec()
    gv = Vocab(spec)

    tok = whitespace_tok
    render_ok, cmp_ok = True, True
    for r in a.rows:
        for d in a.cols:
            lo, hi = dd_band(d)
            supp = hi - lo + 1
            docs = make_docs(r, d, a.docs, seed=7)
            pairs = make_edit_pairs(r, d, a.docs, seed=7)
            res = run_gates_impl(docs, pairs, tok, supp)

            # 渲染门：绝对判据，只有本臂有这些量，必须过
            rend = [x for x in res if x["name"] in RENDER_GATES]
            report(rend, r, d)
            render_ok = render_ok and all(x["ok"] for x in rend)

            # 对照门：与主网格比差值
            grid = [project_doc(x, gv) for x in
                    generate_corpus(gv, nl_corpus_cfg(r, d, 7), a.docs)]
            cmp_res = run_comparative(docs, grid)
            print(f"\n--- 对照门 R{r}_D{d}（NL vs 主网格）---")
            for x in cmp_res:
                flag = "ok  " if x["ok"] else "FAIL"
                nl, gd = x["nl"], x["grid"]
                print(f"[{flag}] {x['name']}  {x['key']}")
                print(f"         NL {nl:.4f}   主网格 {gd:.4f}   "
                      f"差 {x['diff']:.4f}   容差 {x['tol']:.2f}")
            cmp_ok = cmp_ok and all(x["ok"] for x in cmp_res)

            # G7 单独报：它查的是不变量 3，主网格靠 _ValueDraw 保证，
            # 渲染不可能破坏它 —— 报出来只是确认 project/render 没错位。
            g7 = [x for x in res if x["name"].startswith("G7")]
            if g7 and not g7[0]["ok"]:
                print(f"[FAIL] {g7[0]['name']}  n_bad={g7[0]['n_bad']}")
                cmp_ok = False

    print("\n" + "=" * 66)
    if render_ok and cmp_ok:
        print("渲染门与对照门全过。渲染没有改变继承的结构量。")
        print("")
        print("下一步（都还没做）：")
        print("  1. nl_collide 重跑，取经验 posCeil 逐格记下")
        print("  2. sweep.py --pos-emp 指向新的 nl_collide.jsonl")
        print("  3. nl_train.py")
    else:
        if not render_ok:
            print("渲染门失败 —— 表层有问题，失真会静默进入数据。")
        if not cmp_ok:
            print("对照门失败 —— 渲染改变了某个继承的结构量。先查 render/")
            print("project 的字段语义（Stmt 是 raw 索引，Doc.answer 是 token")
            print("id，EditedDoc 无 q_kept/n_slots/q_gap），再怀疑生成器。")
    sys.exit(0 if (render_ok and cmp_ok) else 1)


if __name__ == "__main__":
    main()
