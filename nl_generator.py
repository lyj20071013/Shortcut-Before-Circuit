"""Vocabulary pools and supporting rendered-language definitions."""
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from config import dd_band

# 多个模板 + 变长属性短语，两者共同让句长可变。这是本臂相对主网格的
# 唯一实质收益：定长语句下答案的 token 偏移是 c*ΔD + const，固定偏移的
# 命中率恰为 1/|supp|，与主网格分毫不差，posCeil 一点没降低。变长之后
# 偏移是 ΔD 条语句长度之和，固定偏移命中率下降 -> 位置捷径变弱 ->
# 更多格子进 retrieval 态。nl_gates 的 G3 量这件事。
#
# 查询 slot 与 filler 从同一个模板池抽（G4 要求句长同分布），所以模板
# 不能携带任何"这是被查询 slot"的信号。
TMPL = [
    "{subj}'s {attr} is {adj} {noun}.",
    "{subj} has {adj} {noun} as {subj_p} {attr}.",
    "The {attr} for {subj} is {adj} {noun}.",
    "{subj} lists {adj} {noun} under {subj_p} {attr}.",
]
QUERY = "{subj}'s {attr} is"

# 值空间 = ADJ × NOUN，但**读数只看 adj**（选项 C 取第一个 token），所以
# 决定随机基线的是 len(ADJ) 而不是 len(ADJ)*len(NOUN)。noun 只贡献自然度
# 与 token 数，对规则计算和读数都不携带信息 —— q_values 取 split()[0]。
#
# A larger adjective pool lowers the chance baseline. Empirical
# position-only accuracy must still be measured for the rendered corpus;
# no learning-stage or mechanism conclusion follows from pool size alone.
# 差距但每一格都有可爬的天花板。实际倍数由 nl_collide 重跑后确认 —— 上面
# 那些 posCeil 值来自 24 个 adj 的那一版，扩池不该改变它们，但要验。
#
# 全部选常见短形容词：从头训练用自定义 vocab 时它们必然单 token，但若之后
# 复用这份语料做预训练模型的微调臂，BPE 下的单 token 性要重新验 —— 常见短词
# 的命中率高得多。
ADJ = [
    # 明度
    "pale", "bright", "dark", "light", "deep", "dull", "faint", "vivid",
    "dim", "glowing", "shining", "radiant",
    # 饱和度与强度
    "rich", "thin", "muted", "intense", "mild", "strong", "weak", "bold",
    "subtle", "heavy",
    # 温度
    "warm", "cool", "cold", "hot", "icy", "fiery", "frosty", "sunny",
    # 质感
    "smooth", "rough", "soft", "harsh", "silky", "coarse", "glossy", "matte",
    "sleek", "fuzzy",
    # 清澈度
    "clean", "dirty", "muddy", "clear", "cloudy", "pure", "dusty", "grimy",
    "hazy", "crisp",
    # 新旧
    "new", "old", "worn", "fresh", "faded", "aged", "modern", "ragged",
    "pristine", "shabby",
    # 形状与边缘
    "sharp", "blunt", "jagged", "rounded", "angular", "curved",
    # 朴素与繁复
    "plain", "fancy", "simple", "ornate", "gaudy", "stark",
    # 观感
    "lovely", "ugly", "pretty", "drab", "cheerful", "gloomy", "serene",
    "restless",
    # 尺度
    "broad", "narrow", "wide", "slim", "dense", "sparse",
    # 其他
    "rustic", "polished", "painted", "dyed", "stained", "tinted", "shaded",
    "striped", "quiet", "loud", "calm", "wild", "gentle", "fierce",
    # ---- 扩到 ~170：generator 的 _ValueDraw 在 n_stmts=55 上耗尽 100 个值 ----
    # 报错原文："每篇文档消耗约 1.5×slot 数个值 …… n_stmts=55 时约需 120"。
    # 实际更高，因为 _keyed 先超量生成填充 slot 再按 key 裁剪（generator.py:
    # 241 的 need = budget0 + len(q_old) + 8），裁掉的那些也消耗了值。
    #
    # Expanding the pool preserves the statement-count range and avoids
    # introducing an additional length change. The vocabulary and surface
    # still differ from the main grid; their effects are not isolated.
    # 尺度
    "tiny", "huge", "vast", "small", "large", "giant", "minor", "major",
    "bulky", "lean", "stout", "spare", "ample", "scant",
    # 做工
    "fine", "crude", "neat", "messy", "tidy", "lavish", "modest", "humble",
    "grand", "noble", "common", "rare", "odd", "quaint",
    # 硬度与弹性
    "stiff", "limp", "firm", "loose", "tight", "slack", "brittle", "tough",
    "supple", "sturdy", "frail", "hardy", "feeble", "robust",
    # 湿度
    "damp", "dry", "wet", "moist", "arid", "humid",
    # 气候与光
    "bleak", "balmy", "brisk", "mellow", "dusky", "murky", "shady",
    # 味与气
    "tart", "sweet", "bitter", "sour", "salty", "bland", "spicy", "smoky",
    "earthy", "fruity", "woody", "floral", "minty", "musky", "grassy",
]
NOUN = ["blue", "green", "amber", "grey", "rose", "teal", "olive", "rust",
        "ivory", "slate", "coral", "sand", "plum", "moss", "clay", "ash",
        "wheat", "steel", "brick", "fern", "linen", "cedar", "flint", "mist"]

ATTR = ["favorite colour", "chosen tile", "assigned badge", "picked marker",
        "listed shade", "marked label", "sample swatch", "record tag"]

NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Henry",
         "Irene", "Jack", "Karen", "Leo", "Mona", "Nate", "Olive", "Peter",
         "Quinn", "Rita", "Sam", "Tina", "Umar", "Vera", "Wade", "Xena",
         "Yuri", "Zoe", "Adam", "Beth", "Cyrus", "Dora", "Elias", "Fay",
         "Gus", "Hana", "Ivan", "June", "Kurt", "Lena", "Miles", "Nora"]

Value = Tuple[str, str]


def _validate_pools() -> None:
    """import 时自检。config.validate_cfg 的同类：宁可在生成前炸掉，也不要
    在数据里留静默失真。四条：

    1. ADJ 无重复。重复会让"抽两个不同的 adj"变成可能抽到同一个字符串，
       于是 v* 与 v_truth 在读数位置不可分，Δ 恒为 0 而原因是池子有重复。
    2. NOUN 无重复。不影响读数，但会让值空间小于宣称值。
    3. ADJ 足够大。sample_values 抽 2 个，_fillers 从剩下的抽 —— 池子太小时
       ban_adj 之后可选项过少，filler 的 adj 分布会显著偏离均匀。
    4. 全部 ASCII 小写单词、无空格。含空格的"形容词"会破坏 G1 的 token 数
       守恒（值不再是两 token），而 G1 是最硬的一条门。
    """
    for name, pool in (("ADJ", ADJ), ("NOUN", NOUN)):
        dup = [w for w, c in Counter(pool).items() if c > 1]
        if dup:
            raise ValueError(
                f"{name} 有重复项 {dup}。ADJ 重复会让 sample_values 可能"
                f"抽到同一字符串两次，v* 与 v_truth 在读数位置不可分。")
        bad = [w for w in pool if not w.isascii() or not w.islower()
               or " " in w]
        if bad:
            raise ValueError(
                f"{name} 含非法项 {bad}：须为 ASCII 小写单词、无空格。"
                f"含空格会让值超过两 token，G1 的 token 数守恒失效。")
    if len(ADJ) < 32:
        raise ValueError(
            f"len(ADJ)={len(ADJ)} < 32。读数的随机基线是 1/len(ADJ)，"
            f"池子小则基线高，而经验 posCeil 在高 ΔD 上只有 0.016-0.041，"
            f"基线一旦超过它，位置捷径相位就不存在（nl_collide 会报）。")


def values_needed(n_stmts_hi: int, r_old: int, p_update: float = 0.5,
                  spread: float = 0.85) -> int:
    """一篇文档最坏消耗多少个不同的值。

    generator._ValueDraw 保证文档内值不重复（不变量 3），耗尽即 RuntimeError。
    消耗量由 slot 数与每 slot 的代数决定：
      slot 数 ≈ n_stmts / ((1 + p_update·R_old) / (1 + spread))
      每 slot 消耗 ≈ 1 + p_update  （被更新的 slot 有两代值）
    再乘 _keyed 的超量系数：generator.py:241 先生成 need = budget0 +
    len(q_old) + 8 条再按 key 裁剪，裁掉的也已消耗值。经验上 1.6 够。

    这个函数存在的理由是 check_pools 要给出**具体数字**而不是"池子太小"。
    先前 n_values=100 撞上这条，报错来自生成器深处（_ValueDraw.take），
    调用方看不出该扩多少。
    系数 2.0 是**经验标定，不是推导**。已知的两个锚点：n_stmts=55、R_old=3
    时实测耗尽 100（故真实需求 >100），而生成器自己对 R_old=1 的提示是
    "约需 120"。先前我写 1.6，代入得 105 —— 那个值大于 100 但只多 5%，
    而门宁可偏严：一个在该拦的时候放行的门比没有门更糟。2.0 给 130。
    """
    per_slot = 1.0 + p_update
    n_slots = n_stmts_hi * (1.0 + spread) / (1.0 + p_update * r_old)
    return int(n_slots * per_slot * 2.0) + 8


_validate_pools()


POSS = {"Alice": "her", "Bob": "his", "Carol": "her", "Dave": "his",
        "Erin": "her", "Frank": "his", "Grace": "her", "Henry": "his",
        "Irene": "her", "Jack": "his", "Karen": "her", "Leo": "his",
        "Mona": "her", "Nate": "his", "Olive": "her", "Peter": "his",
        "Quinn": "their", "Rita": "her", "Sam": "their", "Tina": "her",
        "Umar": "his", "Vera": "her", "Wade": "his", "Xena": "her",
        "Yuri": "their", "Zoe": "her", "Adam": "his", "Beth": "her",
        "Cyrus": "his", "Dora": "her", "Elias": "his", "Fay": "her",
        "Gus": "his", "Hana": "her", "Ivan": "his", "June": "her",
        "Kurt": "his", "Lena": "her", "Miles": "his", "Nora": "her"}


def stmt_with(ti: int, subj: str, attr: str, v: Value) -> str:
    """用第 ti 个模板生成一条语句。

    模板索引由调用方逐位置先抽定，不在此处抽 —— base 与 edit 必须在同一
    位置用同一模板，否则模板长度差会让两侧 token 数不等（G1 失败）。
    模板的选择不携带 slot 信息，故 G4 的句长同分布不受影响。
    """
    return TMPL[ti].format(subj=subj, attr=attr, adj=v[0], noun=v[1],
                           subj_p=POSS.get(subj, "their"))


def sample_values(rng: random.Random) -> Tuple[Value, Value]:
    """(v_old, v_new)，adj 必不相同。

    读数在 adj 位置取，adj 相同则 v* 与 v_truth 在读数位置不可分 —— 那样
    Δ 恒为 0 而原因是设计缺陷，不是模型行为。
    """
    a1, a2 = rng.sample(ADJ, 2)
    return (a1, rng.choice(NOUN)), (a2, rng.choice(NOUN))


def _take_group(rng: random.Random, pool: List[int], r: int,
                W: int) -> Optional[Tuple[List[int], List[int]]]:
    """从 pool 里切出一个"被更新的填充 slot"的 R+1 个位置。

    末次赋值在 pf，R 份旧值在 [pf-W, pf) 内且都取自 pool。窗口宽度 W 与
    q_slot 相同 —— generator.py 第 5 条不变量要求"老值散布宽度"不得区分
    q_slot 与填充 slot，用不同的 W 本身就是判别式。

    返回 (排序后的位置, 剩余 pool)，或 None（pool 里没有可行窗口）。
    """
    cands = []
    for idx, pf in enumerate(pool):
        earlier = [p for p in pool[:idx] if p >= pf - W]
        if len(earlier) >= r:
            cands.append((pf, earlier))
    if not cands:
        return None
    pf, earlier = cands[rng.randrange(len(cands))]
    take = set(rng.sample(earlier, r)) | {pf}
    return sorted(take), [p for p in pool if p not in take]


def _fresh_slot(rng: random.Random, used: set) -> Tuple[str, str]:
    guard = 0
    while True:
        guard += 1
        if guard > 20000:
            raise RuntimeError(
                f"slot 池耗尽：已用 {len(used)}，池 {len(NAMES) * len(ATTR)}。"
                f"扩 NAMES 或减小 n_stmts_hi。")
        s = (rng.choice(NAMES), rng.choice(ATTR))
        if s not in used:
            used.add(s)
            return s


def _fresh_val(rng: random.Random, used: set,
               ok_adj: Sequence[str]) -> Value:
    """文档内值互异（generator.py 第 3 条）。同一 slot 内旧值重复 R 次是
    R_old 的定义，不违反这条 —— 这条约束的是**跨 slot** 不共用值。"""
    guard = 0
    while True:
        guard += 1
        if guard > 20000:
            raise RuntimeError(
                f"值池耗尽：已用 {len(used)}，池 {len(ok_adj) * len(NOUN)}。"
                f"扩 ADJ 或减小 n_stmts_hi。")
        v = (rng.choice(ok_adj), rng.choice(NOUN))
        if v not in used:
            used.add(v)
            return v


def _layout(rng: random.Random, n_stmts: int, r: int, dd: int, spread: float,
            p_update: float) -> Optional[Tuple[List[int], int, list]]:
    """位置布局。返回 (q_pos, p_final, filler_groups) 或 None。

    filler_groups 每项是 (positions, is_updated)。被更新的填充 slot 占 R+1
    个位置，未更新的占 1 个 —— 与主网格的 p_update 语义相同（config.py:26
    "非查询 slot 收到更新的概率"），且 max_updates=1。

    这是 G6 的修法。先前每个填充 slot 只占 1 个位置，于是"语句数最多的
    slot"唯一确定被查询 slot，模型可以不读 query 就答对，而 train.py 的
    swap_query 会读到接近 1.0（主网格 0.0025）。
    """
    W = int(spread * n_stmts)
    p_final = n_stmts - 1 - dd
    lo = max(0, p_final - W)
    if p_final - lo < r:
        return None
    q_pos = sorted(rng.sample(range(lo, p_final), r))
    taken = set(q_pos) | {p_final}
    pool = [i for i in range(n_stmts) if i not in taken]

    groups = []
    while pool:
        if len(pool) >= r + 1 and rng.random() < p_update:
            got = _take_group(rng, pool, r, W)
            if got is not None:
                pos, pool = got
                groups.append((pos, True))
                continue
        groups.append(([pool.pop(rng.randrange(len(pool)))], False))
    return q_pos, p_final, groups


def _positions_unused(rng: random.Random, n_stmts: int, r: int, dd: int,
                      spread: float) -> Optional[Tuple[List[int], int]]:
    """查询 slot 的 R 个旧值位置与末次赋值位置。None = 该抽样不可行。

    末次赋值在 n_stmts-1-dd，其后恰 dd 条 filler，故查询前最后一条语句
    必是 filler（G2）。R 个旧值散布在末次赋值之前宽 spread*n_stmts 的窗口
    内，而不是紧贴它 —— config.py:134 拒绝 spread*n_stmts_lo < 2*q_old
    的理由是那样 q_gap 恒为 1，构成完美判别式。
    """
    p_final = n_stmts - 1 - dd
    window = int(spread * n_stmts)
    lo = max(0, p_final - window)
    if p_final - lo < r:
        return None
    return sorted(rng.sample(range(lo, p_final), r)), p_final


def _fillers(rng: random.Random, n: int, q_slot: Tuple[str, str],
             used: set, ban_adj: Sequence[str]) -> List[Tuple[str, str, Value]]:
    """n 条 filler 语句的 (subj, attr, value)。

    slot 互不相同且异于 q_slot：filler slot 若重复赋值，那个 slot 也带取代
    事件，G5 的 n_checked 与 n_docs 就不再相等。

    ban_adj 排除 v_old 与 v_new 的 adj。读数在 adj 位置取，若某条 filler
    用了同一个 adj，那个 token 在答案位置上的概率就有一份与被查询 slot
    无关的来源，Δ 不再只反映 slot 追踪。主网格靠 n_values=512 让这种碰撞
    罕见；这里 adj 只有 24 个，52 条 filler 下碰撞率约 1-(23/24)^52 ≈ 0.89，
    所以必须显式排除而不能靠稀疏性。

    两条 arm 的 ban_adj 相同（v_old/v_new 在 _one 里先抽定），故拒绝次数
    相同，rng 消耗序列不错位。
    """
    ok_adj = [x for x in ADJ if x not in ban_adj]
    out = []
    seen_val = set()          # 文档内值唯一，见下
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > 100 * n + 1000:
            raise RuntimeError(
                f"filler 采样卡死：需要 {n} 个互异 slot 与互异值，slot 池 "
                f"{len(NAMES) * len(ATTR)}、值池 {len(ok_adj) * len(NOUN)}。"
                f"减小 n_stmts_hi 或扩 NAMES/ADJ。")
        s, a = rng.choice(NAMES), rng.choice(ATTR)
        if (s, a) == q_slot or (s, a) in used:
            continue
        # generator.py 的第 3 条不变量："文档内值 id 不重复，答案值只出现
        # 一次（消除 recency 捷径的偶然可达）"。主网格用 _ValueDraw 保证；
        # 这里先前只保证 slot 互异，值可以重复 —— 两条 filler 撞上同一个
        # (adj, noun) 时，那个值在文档内出现两次，rarity 的计数域被污染。
        # ban_adj 已挡住撞上 v*/v_truth 的情形，所以读数位置一直是干净的，
        # 但这条不变量本身是论文列出的五项之一，不能只在主网格成立。
        v = (rng.choice(ok_adj), rng.choice(NOUN))
        if v in seen_val:
            continue
        used.add((s, a))
        seen_val.add(v)
        out.append((s, a, v))
    return out


def _one(rng: random.Random, r: int, d: int, n_lo: int, n_hi: int,
         spread: float, inverted: bool, p_update: float = 0.5
         ) -> Optional[dict]:
    """一篇文档。inverted=False -> base，True -> edit。

    p_update 与 config.py:26 同义：非查询 slot 收到更新的概率。被更新的
    填充 slot 也带 R 份旧值 + 1 份末代值，故"语句数最多的 slot"不再唯一
    确定被查询 slot（G6）。默认 0.5 与主网格一致。
    """
    dlo, dhi = dd_band(d)
    dd = rng.randint(dlo, dhi)
    n_stmts = rng.randint(n_lo, n_hi)
    got = _layout(rng, n_stmts, r, dd, spread, p_update)
    if got is None:
        return None
    q_pos, p_final, groups = got

    subj, attr = rng.choice(NAMES), rng.choice(ATTR)
    v_old, v_new = sample_values(rng)
    q_slot = (subj, attr)

    # 编辑后 old 落在 R 个位置里的哪一个。必须随机，且必须在**两条 arm 上
    # 都抽**：
    #   随机 —— 固定放在最后一个位置的话，edit 里"只出现一次的值"恒在
    #   倒数第二个查询语句处，模型可以靠位置识别罕见值而无需计数，那是
    #   一个判别式（与 config.py:134 拒绝 q_gap 恒为 1 同一类问题）。
    #   两条都抽 —— 只在 inverted 分支抽会让 base 与 edit 的 rng 消耗序列
    #   错开一步，此后 tmpl_idx、filler 全部不同，token 数随之不等，G1 失败。
    k_old = rng.randrange(r)

    # 编辑只改 R+1 个查询 slot 位置上的值，位置本身不动 -> token 数不变。
    if inverted:
        # [new×..., old 在第 k_old 个, ..., new]，counts new=R old=1
        vals = [v_new] * r
        vals[k_old] = v_old
    else:
        vals = [v_old] * r
    if len(vals) != len(q_pos):
        # zip 会静默截断，于是 R 个位置里只有一部分被赋值，counts 变了而
        # G5 仍可能通过。
        return None
    assign = dict(zip(q_pos, vals))
    assign[p_final] = v_new

    # 填充 slot。被更新的那些拿 R 份旧值 + 1 份末代值，与 q_slot 同构。
    # ban_adj 排除 v_old/v_new 的 adj：读数在 adj 位置取，填充用了同一个
    # adj 会让那个 token 在答案位置上有一份与被查询 slot 无关的来源。
    ok_adj = [x for x in ADJ if x not in (v_old[0], v_new[0])]
    used_slot = {q_slot}
    used_val = {v_old, v_new}
    slot_of = {}
    for pos, upd in groups:
        s = _fresh_slot(rng, used_slot)
        if upd:
            fo = _fresh_val(rng, used_val, ok_adj)
            fn = _fresh_val(rng, used_val, ok_adj)
            # 末次赋值在这一组的最后一个位置，与 q_slot 的排法相同
            for j, p in enumerate(pos):
                assign[p] = fn if j == len(pos) - 1 else fo
                slot_of[p] = s
        else:
            fv = _fresh_val(rng, used_val, ok_adj)
            assign[pos[0]] = fv
            slot_of[pos[0]] = s
    for p in q_pos:
        slot_of[p] = q_slot
    slot_of[p_final] = q_slot

    if len(assign) != n_stmts:
        # 每个位置恰好被赋值一次。不等说明 _layout 的分组漏了或重了，
        # 而那会让 base/edit 的 token 数不等（G1）。
        return None

    # 模板逐位置先抽定，base 与 edit 用同一串 —— 编辑只换值，不换模板。
    # 若在循环里现抽，base 的 R 个位置和 edit 的 R 个位置会抽到不同模板，
    # 而模板长度不同（"{subj}'s {attr} is ..." 是 4 词前缀，"{subj} lists
    # ... under {subj_p} {attr}" 是 2+3），两侧 token 数就不等 -> G1 失败。
    # 这是 config.py:56-62 那个"不消耗随机数"技巧的同类要求：让 base 与
    # edit 的差异只落在被改的那一处。
    tmpl_idx = [rng.randrange(len(TMPL)) for _ in range(n_stmts)]

    # 每个位置都在 assign 与 slot_of 里（上面的 len(assign) != n_stmts 检查
    # 保证了这一点），所以渲染是一个直通循环，不再有 q_slot / filler 分支。
    stmts, texts = [], []
    for i in range(n_stmts):
        v = assign[i]
        s, a = slot_of[i]
        t = stmt_with(tmpl_idx[i], s, a, v)
        stmts.append(dict(slot=(s, a), value=f"{v[0]} {v[1]}", text=t))
        texts.append(t)

    text = " ".join(texts) + " " + QUERY.format(subj=subj, attr=attr)
    return dict(text=text, answer=v_new[0], q_slot=q_slot, stmts=stmts,
                v_star=v_old[0], realized_delta=dd, n_stmts=n_stmts)


_DEAD = (
    "nl_generator 的平行生成器已被渲染层取代。放置逻辑现在复用主网格：\n"
    "  generator.generate_corpus -> nl_render.render\n"
    "改用 nl_corpus.gate_docs / gate_pairs（门与碰撞表）或 nl_corpus.NLStream\n"
    "（训练）。\n"
    "\n"
    "为什么这些函数抛异常而不是删掉：它们的放置逻辑有两个已实测的缺陷 ——\n"
    "G8 的 update 位置五分位 [0.05,0.13,0.17,0.23,0.42]（末桶是首桶 7.8 倍，\n"
    "违反 generator.py 第 4 条），G9 的 q_gap 9.75 对 fill_gap 6.14（违反\n"
    "第 6-7 条）。两者同一根因：q_slot 先从干净区间挑位置、填充组从残余池子\n"
    "挑。留着代码作记录，但任何调用都是在量一个不再被训练的语料，而 posCeil\n"
    "由此进 sweep.classify 决定哪些 run 进网格 —— 静默走这条路会让整条链错。")


def make_docs(*a, **k):
    raise RuntimeError(_DEAD)


def make_edit_pairs(*a, **k):
    raise RuntimeError(_DEAD)
