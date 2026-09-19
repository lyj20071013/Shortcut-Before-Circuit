"""Candidate rules, answer-preserving edits, and intervention measurements."""
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from config import CorpusCfg, LangSpec, dd_band
from generator import Doc, Stmt, emit, generate_corpus
from vocab import Vocab

TRUTH = "last_value"

# max_updates=1 时 q slot 只有两代值，frequency 解析上等于 last_value
# （Rreal=1，平票裁给更近者）或 primacy（Rreal≥2）。碰撞率的缺口正是
# Rreal=1 的比例（R3 格实测 3.2%），且该比例随 ΔD 上升 —— 低 R_old 列
# 可能两侧都碰不到 COLLIDE，让 frequency 伪装成独立规则参与着色。
# 阈值合并治不了这个，主相图直接排除；agree 字段照常记录，附录要用。
# 三代值（max_updates≥2）下 frequency 才独立，另做小臂。
# MAIN_GRID_EXCLUDE = frozenset({"frequency"})

RULE_NAMES = [TRUTH, "primacy", "frequency", "rarity",
              "recency_global", "last_update_global", "position"]

# 观测归因里 rarity 与 last_value 在 max_updates=1 下是恒等式（当前值的 reps
# 恒为 1），n_disc 恒为 0、rate_disc 为 nan，dominant_rule 不会选中它。
# 主相图的 DV 改用 break_rarity 的因果读数，见模块 docstring。
MAIN_GRID_EXCLUDE = frozenset({"frequency", "rarity"})


def grid_exclude(cfg: CorpusCfg) -> frozenset:
    """该配置下应从观测归因中排除的规则。

    主网格（p_break=0）排除 rarity：式 1 让它与 last_value 逐篇同指，n_disc
    恒为 0、rate_disc 为 nan。旋钮 6 打开后反转文档上二者分歧，n_disc>0，
    rarity 第一次在观测路径上有信号 —— 这是与因果探针独立的第二个仪器，
    必须放回，否则 arm 白跑了一半。

    frequency 恒排除：max_updates=1 下它是 last_value（Rreal=1）或 primacy
    （Rreal≥2）的恒等式，与 p_break 无关。反转文档上 counts 互换，frequency
    指向末代值、仍与 last_value 重合，所以打开 p_break 也救不了它。
    """
    if getattr(cfg, "p_break", 0.0) > 0.0:
        return frozenset({"frequency"})
    return MAIN_GRID_EXCLUDE

YIELD_MIN = 0.05        # 域小于此值：该格该编辑不出因果读数（不是错误）
CONTAM_WARN = 0.02      # 非目标规则被连带移动的比例，超过则打印警告


# ---------------- 视图 ----------------

@dataclass
class EditedDoc:
    """与 Doc 共享规则函数所需的最小字段集。"""
    kind: str
    tokens: List[int]
    answer_pos: int
    answer: int
    stmts: List[Stmt]
    q_ent: int
    q_attr: int
    q_hist_k: Optional[int]
    realized_delta: int
    n_stmts: int
    base: Doc
    target_rule: str
    sign: int
    v_star: int             # 固定对照值（raw value id）


def _is_q(s: Stmt, d) -> bool:
    return s.ent == d.q_ent and s.attr == d.q_attr


def _q_pos(d) -> List[int]:
    return [i for i, s in enumerate(d.stmts) if _is_q(s, d)]


def _q_stmts(d) -> List[Stmt]:
    return [s for s in d.stmts if _is_q(s, d)]


def _p_final(d) -> int:
    return d.n_stmts - 1 - d.realized_delta


def _q_gap(d) -> int:
    """q_final 到最近同 slot 前驱的距离，与 generator 的定义一致。"""
    p = _p_final(d)
    prev = [i for i in _q_pos(d) if i < p]
    return p - max(prev) if prev else p


def _answer_val(d: Doc) -> int:
    """标签的 raw value id。

    反转文档（is_break，旋钮 6）在 truth_rule=rarity 下标签是老值，从
    val_history[-1] 反推会得到末代值，进而让 apply_edit 造出"真值被改"的
    编辑文档，probe_selfcheck 的 ed.answer == d.answer 立刻失败。故一律以
    生成器记下的 answer_val_id 为准；hist 分支保留作旧数据（无该字段）回退。
    """
    v = getattr(d, "answer_val_id", -1)
    if v >= 0:
        return v
    k = d.q_hist_k
    return d.val_history[-1] if not k else d.val_history[-1 - k]


# ---------------- A. 候选规则，全部返回 raw value id ----------------

def r_last_value(d, offset=None):
    q = _q_stmts(d)
    return q[-1].val if q else None


def r_primacy(d, offset=None):
    q = _q_stmts(d)
    return q[0].val if q else None


def r_frequency(d, offset=None):
    """q slot 内出现次数最多的值，平票裁给更近者。

    这个平票规则让 R_old=1 时 frequency 与 last_value 重合（而非与 primacy
    重合）：碰撞落在与真值之间，是保守的一侧，不会把跟踪型模型误判成频率型。
    """
    q = _q_stmts(d)
    if not q:
        return None
    cnt = Counter(s.val for s in q)
    best = max(cnt.values())
    for s in reversed(q):
        if cnt[s.val] == best:
            return s.val
def r_rarity(d, offset=None):
    """q slot 内出现次数最少的值，平票取更近者。

    训练分布上恒等于 last_value：_build_slot 让最后一代值的 reps 恒为 1，
    于是 counts={v_old:R, v_new:1}，least=1 只命中 v_new。这是设计好的
    对齐（alias），不是缺陷 —— 正因为逐篇不冲突，目标函数对 rarity 与
    recency 无差别，模型实现哪一条由"哪条更便宜学"决定，而便宜程度由
    R_old（"重复⇒非当前"被演示多少次）控制。identifiability 会报
    rarity|last_value≈1.000，正是预期。分开二者只能靠 break_rarity。
    """
    q = _q_stmts(d)
    if not q:
        return None
    cnt = Counter(s.val for s in q)
    least = min(cnt.values())
    for s in reversed(q):
        if cnt[s.val] == least:
            return s.val


def r_recency_global(d, offset=None):
    return d.stmts[-1].val


def r_last_update_global(d, offset=None):
    u = [s for s in d.stmts if s.is_update]
    return u[-1].val if u else None


def r_position(d, offset: int):
    """复制倒数第 offset+1 条语句的值，完全不读 query。offset 由语料拟合，
    故这是纯位置规则的上界，应与 selfcheck 的 posCeil 对得上。"""
    i = d.n_stmts - 1 - offset
    return d.stmts[i].val if 0 <= i < d.n_stmts else None


_RULES: Dict[str, Callable] = {
    TRUTH: r_last_value, "primacy": r_primacy, "frequency": r_frequency,
    "recency_global": r_recency_global,
    "last_update_global": r_last_update_global, "position": r_position, "rarity": r_rarity,}


def fit_position_offset(docs: Sequence[Doc]) -> int:
    return Counter(d.realized_delta for d in docs).most_common(1)[0][0]


def rule_predictions(d, offset: int) -> Dict[str, Optional[int]]:
    return {k: f(d, offset) for k, f in _RULES.items()}


# ---------------- 预测器 ----------------

class RulePredictor:
    """合成预测器：严格执行某一条规则。用于验证探针机器本身 —— 喂进
    "只会 frequency 的模型"，attribute 必须把它读成 frequency。"""

    def __init__(self, rule: str, offset: int, sharp: float = 8.0):
        self.rule, self.offset, self.sharp = rule, offset, sharp

    def predict(self, view) -> Optional[int]:
        return _RULES[self.rule](view, self.offset)

    def logp(self, view, cands: Sequence[int]) -> List[float]:
        hit = self.predict(view)
        return [0.0 if v == hit else -self.sharp for v in cands]


class LogitsPredictor:
    """真模型适配器。logits_fn(tokens) -> 该前缀下一位置的 log-softmax 向量。
    候选一律用 raw value id，内部转 token，故 A/B 两条路径与合成预测器同接口。"""

    def __init__(self, logits_fn: Callable[[List[int]], Sequence[float]],
                 vocab: Vocab):
        self.fn, self.vocab = logits_fn, vocab
        self._val_tok = [vocab.val(v) for v in range(vocab.spec.n_values)]

    def _lp(self, view):
        return self.fn(view.tokens[:view.answer_pos])

    def predict(self, view) -> int:
        lp = self._lp(view)
        return max(range(self.vocab.spec.n_values),
                   key=lambda v: lp[self._val_tok[v]])

    def logp(self, view, cands: Sequence[int]) -> List[float]:
        lp = self._lp(view)
        return [lp[self._val_tok[v]] for v in cands]


# ---------------- B. 最小编辑 ----------------

def _fresh_val(used: set, spec: LangSpec, rng: random.Random) -> int:
    while True:
        v = rng.randrange(spec.n_values)
        if v not in used:
            return v


def _fresh_slot(used: set, spec: LangSpec, rng: random.Random) -> Tuple[int, int]:
    while True:
        e, a = rng.randrange(spec.n_entities), rng.randrange(spec.n_attrs)
        if (e, a) not in used:
            used.add((e, a))
            return e, a


def _singleton(d) -> Counter:
    return Counter((s.ent, s.attr) for s in d.stmts)


def _is_filler(d, i: int, cnt: Counter) -> bool:
    s = d.stmts[i]
    return (not _is_q(s, d) and not s.is_update
            and cnt[(s.ent, s.attr)] == 1)

def _order_ok(stmts: Sequence[Stmt]) -> bool:
    """语句序列是否满足生成器的时序不变量：每个 slot 内 is_update 单调不减
    （老值不得出现在自己的 update 之后），且同一值的出现必须连续
    （不同世代不得交错）。

    编辑只要移动语句就可能破坏它，逐条编辑各写自己的位置条件迟早漏项，
    统一在 apply_edit 里校验，不合法即视作该文档不在域内。
    """
    per: Dict[Tuple[int, int], List[Stmt]] = {}
    for s in stmts:
        per.setdefault((s.ent, s.attr), []).append(s)
    for seq in per.values():
        f = [x.is_update for x in seq]
        if any(a and not b for a, b in zip(f, f[1:])):
            return False
        run, seen = None, set()
        for x in seq:
            if x.val != run:
                if x.val in seen:
                    return False
                run = x.val
                seen.add(x.val)
    return True


def _movable(d, i: int) -> bool:
    """可被搬移的填充语句：非 q slot、非 update。不要求单例 —— 时序合法性
    交给 _order_ok。R_old=12 时填充 slot 也吃 R_old，单例只占约 10%，
    卡单例会让 shift_delta 在整条高 R_old 列上没有域。"""
    s = d.stmts[i]
    return not _is_q(s, d) and not s.is_update

def _edit_bump_freq(d: Doc, cfg, spec, rng, k: int = 1):
    """q slot 的老值多出现 k 次：改写 k 条单例填充语句为老值的副本。
    位点限制在最早的 q_old 之前，故最近同 slot 前驱不变，q_gap 守恒。
    非 update 换非 update，逐条 token 数相等。域 = R_real 1（k=1 即翻 argmax）。"""
    qp = _q_pos(d)
    if len(qp) < 2:
        return None
    v_old = d.stmts[qp[-2]].val
    cnt = _singleton(d)
    cand = [i for i in range(qp[0]) if _is_filler(d, i, cnt)]
    if len(cand) < k:
        return None
    new = list(d.stmts)
    for i in rng.sample(cand, k):
        new[i] = Stmt(d.q_ent, d.q_attr, v_old, False)
    return new, d.realized_delta


def _edit_drop_freq(d: Doc, cfg, spec, rng):
    """bump_freq 的镜像：老值副本减到只剩 1 份，多出的改写成全新单例 slot。
    保留最靠近 p_final 的那份，故 q_gap 守恒。老值仍出现一次，冲突结构不消失。
    域 = R_real≥2。"""
    qp = _q_pos(d)
    if len(qp) < 3:
        return None
    v_old = d.stmts[qp[-2]].val
    olds = [i for i in qp[:-1] if d.stmts[i].val == v_old]
    if len(olds) < 2:
        return None
    used_v = {s.val for s in d.stmts}
    used_s = {(s.ent, s.attr) for s in d.stmts}
    new = list(d.stmts)
    for i in olds[:-1]:                     # 留最近的一份
        e, a = _fresh_slot(used_s, spec, rng)
        v = _fresh_val(used_v, spec, rng)
        used_v.add(v)
        new[i] = Stmt(e, a, v, False)
    return new, d.realized_delta

def _edit_shift_delta(d: Doc, cfg, spec, rng, lo: int = 2, hi: int = 8):
    """q_final 与更早的一条填充语句互换，ΔD 增大，固定偏移处换成填充值。

    同时把位置最大的那条 q_old 往前搬同样的距离 s，q_gap 精确守恒。守恒的
    前提是搬移后它仍是最大的 q_old，故要求 mp > 第二大的 q_old 位置。
    早期版本要求的是"mp > 值与它不同的最后一条 q_old"：R_old=1 时只有一条
    老值、条件恒真，R_old=12 时 12 条老值同值、条件形同虚设，
    最近前驱变成第二大那条，gap 从 10 掉到 8。
    只搬 q_final 也不行：ΔD 与 q_gap 在语料里基本独立，反向同步变化是
    off-manifold 组合，Δ 可能来自样本变怪而非位置规则。
    互换保证 token 多重集不变（marked 时 5↔4 也是互换，总量守恒）。"""
    p = _p_final(d)
    qp = _q_pos(d)
    olds = [i for i in qp if i != p]
    if not olds:
        return None
    m = olds[-1]
    second = olds[-2] if len(olds) >= 2 else -1
    cnt = _singleton(d)
    cand = []
    for j in range(m + 1, p):               # 目标须晚于所有 q_old
        s = p - j
        mp = m - s
        if not (lo <= s <= hi) or mp <= second or mp < 0:
            continue
        if mp in qp or not _movable(d, j) or not _movable(d, mp):
            continue
        cand.append((j, mp))
    if not cand:
        return None
    j, mp = rng.choice(cand)
    new = list(d.stmts)
    new[j], new[p] = new[p], new[j]
    new[mp], new[m] = new[m], new[mp]
    return new, d.realized_delta + (p - j)

def _edit_late_update(d: Doc, cfg, spec, rng):
    """在 p_final 之后放一条别的 slot 的 update，其后不再有 update。
    排除最末一条语句，否则 recency_global 与 last_update_global 同时翻。
    域 = 尾部本无（或仅有更早）update 的文档，即低 ΔD 侧。"""
    if cfg.use_marker:
        return None
    p = _p_final(d)
    cand = [i for i in range(p + 1, d.n_stmts - 1)
            if not _is_q(d.stmts[i], d)
            and not any(d.stmts[j].is_update for j in range(i + 1, d.n_stmts))]
    if not cand:
        return None
    seen = {(s.ent, s.attr) for i, s in enumerate(d.stmts)
            if i <= p and not _is_q(s, d)}
    i = rng.choice(cand)
    pool = [s for s in seen if s != (d.stmts[i].ent, d.stmts[i].attr)]
    if not pool:
        return None
    e, a = rng.choice(pool)
    v = _fresh_val({s.val for s in d.stmts}, spec, rng)
    new = list(d.stmts)
    new[i] = Stmt(e, a, v, True)
    return new, d.realized_delta


def _edit_clear_late_update(d: Doc, cfg, spec, rng):
    """late_update 的镜像：p_final 之后所有 update 的 slot 换成从未出现的新
    slot，值不动。它们不再是"同 slot 的再次赋值"，last_update_global 回落到
    q_final；末条语句的值未变，recency_global 不动。域 = 高 ΔD 侧。"""
    if cfg.use_marker:
        return None
    p = _p_final(d)
    tail = [i for i in range(p + 1, d.n_stmts) if d.stmts[i].is_update]
    if not tail:
        return None
    used_s = {(s.ent, s.attr) for s in d.stmts}
    new = list(d.stmts)
    for i in tail:
        e, a = _fresh_slot(used_s, spec, rng)
        new[i] = Stmt(e, a, d.stmts[i].val, False)
    return new, d.realized_delta


def _edit_relabel_old(d: Doc, cfg, spec, rng):
    """把 q slot 全部老值副本换成同一个全新值。结构逐位不变，只换 value id，
    故 primacy 的预测值必须跟着走 —— 这是"模型读结构还是记住了某个值"的对照。
    R_real≥2 时 frequency 与 primacy 同为老值，两者会一起移动，这正是
    identifiability 报告的碰撞；R_real=1 时 frequency 恒等于真值，不受影响。"""
    p = _p_final(d)
    qp = _q_pos(d)
    olds = [i for i in qp if i != p]
    if not olds:
        return None
    if d.n_stmts - 1 - _fit_offset_hint(d) in olds:
        pass        # 位置规则可能连带移动，交给 contamination 报告
    v = _fresh_val({s.val for s in d.stmts}, spec, rng)
    new = list(d.stmts)
    for i in olds:
        new[i] = Stmt(d.q_ent, d.q_attr, v, False)
    return new, d.realized_delta

def _edit_break_rarity(d: Doc, cfg, spec, rng):
    """反转 q slot 的多重性：老值留 1 份，其余副本改写成新值的副本。
    编辑后 q slot = [v_old ×1, v_new ×m]，counts 互换。

    这是训练分布里从不出现的配置，也是唯一能把 rarity 与 last_value 分开的
    编辑：recency → v_new（末条 q 语句未动）、primacy → v_old（两侧同为
    v_old，配对差分自动消掉）、frequency → v_new、rarity → v_old。
    故 Δ 的符号本身就区分机制：+ 是 rarity 型，- 是 frequency 型。

    保留的必须是最早那份：_order_ok 要求 slot 内同值连续且 is_update 单调，
    v_old 只能占前缀。这意味着 v_old 在两侧都位于 q slot 首位，位置类
    解释（含 OOD 下注意力退化到首位）在配对差分中同向，不产生 Δ。

    q_gap 守恒：最近同 slot 前驱的位置未动，只换了值。
    marked 下 is_update 会带 UPD token，改写会破坏长度不变，故返回 None。
    域 = R_real≥2。R_old=1 列无域，是结构性的零信号对照，按 N/A 报告。
    """
    if cfg.use_marker:
        return None
    if getattr(d, "is_break", False):
        # 反转文档的多重性已是 [老值×k, 末代值×rb]，再反转一次没有定义，且它
        # 只有 k=1 份老值故 olds 恒不足。显式返回 None 而不依赖下面的
        # len(olds)<2 静默退出 —— 后者会让「arm 上读数为空」看起来像
        # 「效应不存在」。读数一律只在非反转文档上做，见 go_nogo。
        return None
    p = _p_final(d)
    qp = _q_pos(d)
    if len(qp) < 3:
        return None
    # 按世代取老值，不按位置。qp[-2] 隐含假定末代值恰 1 份：这在主网格成立
    # （reps_final=1），但旋钮 6 下反转 slot 的 qp[-2] 会落在末代值的副本上，
    # 于是 v_old==v_new、函数静默返回 None。val_history 是世代记录，与 reps
    # 无关，两种文档上都正确。
    # p_break=0 时二者恒等，故此改动对已发表数据是严格 no-op：拿已发表的
    # checkpoint 重跑 go_nogo 必须逐位复现原表，这是免费的回归门。
    if len(d.val_history) < 2:
        return None
    v_old, v_new = d.val_history[-2], d.val_history[-1]
    if v_old == v_new:
        return None
    olds = [i for i in qp if i != p and d.stmts[i].val == v_old]
    if len(olds) < 2:                  # R_real==1：无副本可改写
        return None
    new = list(d.stmts)
    for i in olds[1:]:                 # 留 olds[0] 作唯一老值副本
        new[i] = Stmt(d.q_ent, d.q_attr, v_new, True)
    return new, d.realized_delta


def _edit_break_rarity_ctrl(d: Doc, cfg, spec, rng):
    """对照：在一个非 q slot 上做同样的多重性反转，q 候选的 log-odds 不应移动。
    排除"编辑本身让模型脱离任务"这一泛化解释。目标规则仍标 rarity，但由于
    q slot 未动，rb==re，apply_edit 的守卫会拒绝 —— 故它不走 causal()，
    由 go_nogo.py 直接调用并手工比较 margin（见该文件）。
    """
    cnt = _singleton(d)
    pool = [k for k, c in cnt.items()
            if c >= 3 and k != (d.q_ent, d.q_attr)]
    if not pool:
        return None
    e, a = rng.choice(pool)
    idx = [i for i, s in enumerate(d.stmts) if (s.ent, s.attr) == (e, a)]
    vals = [d.stmts[i].val for i in idx]
    if len(set(vals)) < 2:
        return None
    v_last = vals[-1]
    firsts = [i for i in idx if d.stmts[i].val != v_last]
    if len(firsts) < 2:
        return None
    new = list(d.stmts)
    for i in firsts[1:]:
        new[i] = Stmt(e, a, v_last, True)
    return new, d.realized_delta

class _View:
    """predictor 只读 tokens[:answer_pos]，故这两个字段就是全部接口。"""

    __slots__ = ("tokens", "answer_pos")

    def __init__(self, tokens, answer_pos):
        self.tokens, self.answer_pos = tokens, answer_pos


def swap_query(d, vocab: Vocab, cfg: CorpusCfg):
    """把 query 换成同篇另一个有 update 的 slot，语句序列一字不动。

    token 数与 answer_pos 都不变（query 恒是 ent+attr 两个 token），故这不是
    分布外输入，只是问了同一篇文档的另一个问题。返回 None 表示该篇没有第二个
    带 update 的 slot 可用。

    用途：直接测「模型是否真的读了 query」。生成器让值在文档内不重复抽取
    （_ValueDraw），故换 slot 必然换答案 —— 读 query 的模型必须改预测，靠结构
    定位被查询 slot 的模型不会。

    与 is_break 无关，故在主网格上同样良定义。反转文档只是结构差异**更大**的
    地方（q slot 语句数恒为 R+1 而非反转文档约 0.6R+1，加上 antRbk 的间距差
    异），不是这个测量成立的前提。主网格上待检的结构线索是 App:gen 第五条那
    个部分成立的不变量：被查询 slot 的副本比填充 slot 更分散。
    """
    per = {}
    for s in d.stmts:
        per.setdefault((s.ent, s.attr), []).append(s.val)
    cand = [k for k, vs in per.items()
            if k != (d.q_ent, d.q_attr) and len(set(vs)) >= 2]
    if not cand:
        return None
    e, a = cand[len(cand) // 2]          # 取中间一个，不消耗随机数
    toks, apos = emit(d.stmts, e, a, d.q_hist_k, _answer_val(d), vocab, cfg)
    return _View(toks, apos)


def _fit_offset_hint(d) -> int:
    return d.realized_delta


# kind -> (编辑函数, 目标规则, 期望 Δ 符号)
EDITS: Dict[str, Tuple[Callable, str, int]] = {
    "bump_freq": (_edit_bump_freq, "frequency", +1),
    "drop_freq": (_edit_drop_freq, "frequency", -1),
    "shift_delta": (_edit_shift_delta, "position", +1),
    "late_update": (_edit_late_update, "last_update_global", +1),
    "clear_late_update": (_edit_clear_late_update, "last_update_global", -1),
    "relabel_old": (_edit_relabel_old, "primacy", -1),
"break_rarity": (_edit_break_rarity, "rarity", +1),}

RULE_EDITS: Dict[str, List[str]] = {}
for _k, (_f, _t, _s) in EDITS.items():
    RULE_EDITS.setdefault(_t, []).append(_k)


def apply_edit(d: Doc, kind: str, vocab: Vocab, cfg: CorpusCfg,
               rng: random.Random, offset: int) -> Optional[EditedDoc]:
    """返回 None 表示该文档不在此编辑的域内。域筛选见模块 docstring。"""
    if d.q_hist_k:
        return None
    fn, tgt, sign = EDITS[kind]
    out = fn(d, cfg, vocab.spec, rng)
    if out is None:
        return None
    stmts, delta = out
    if not _order_ok(stmts):        # 编辑破坏了 slot 内时序，该文档不在域内
        return None
    av = _answer_val(d)
    toks, apos = emit(stmts, d.q_ent, d.q_attr, d.q_hist_k, av, vocab, cfg)
    ed = EditedDoc(kind, toks, apos, vocab.val(av), stmts, d.q_ent, d.q_attr,
                   d.q_hist_k, delta, len(stmts), d, tgt, sign, -1)
    rb, re = _RULES[tgt](d, offset), _RULES[tgt](ed, offset)
    tb, te = r_last_value(d), r_last_value(ed)
    if rb is None or re is None or tb != te or rb == re:
        return None
    v_star = re if sign > 0 else rb
    if v_star == tb or v_star == te:
        return None
    ed.v_star = v_star
    return ed

# ----- 归因 -----
COLLIDE = 0.95    # 碰撞率 ≥ 此值：两条规则在该格视为同一条

def rule_groups(ident: Dict[str, float], thr: float = COLLIDE) -> Dict[str, str]:
    """规则 -> 等价类标签。把碰撞率 ≥ thr 的规则并成一类（union-find）。

    max_updates=1 时 q slot 只有两代值（老值 R 份、新值 1 份），于是
    frequency 恒等于 last_value（R_old=1，平票裁给更近者）或 primacy
    （R_old≥2，老值占多数）—— 它在整条网格上都不是独立规则。不合并的话
    dominant_rule 的严格 > 会按 RULE_NAMES 顺序任意挑一个，相图上标着
    "primacy" 的格子实际含义是 "primacy=frequency"。
    区分二者需要三代值（max_updates≥2），放附录小臂。
    """
    parent = {k: k for k in RULE_NAMES}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key, v in ident.items():
        if v < thr:
            continue
        a, b = key.split("|")
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb), key=RULE_NAMES.index)
            parent[hi] = lo          # root 取 RULE_NAMES 序，标签跨 run 稳定
    members: Dict[str, List[str]] = {}
    for k in RULE_NAMES:
        members.setdefault(find(k), []).append(k)
    return {k: "=".join(sorted(members[find(k)], key=RULE_NAMES.index))
            for k in RULE_NAMES}


def identifiability(docs: Sequence[Doc], offset: int) -> Dict[str, float]:
    """规则两两同预测的比例。接近 1 的一对在该格不可分离，须并列报告。"""
    out: Dict[str, float] = {}
    preds = [rule_predictions(d, offset) for d in docs]
    for i, a in enumerate(RULE_NAMES):
        for b in RULE_NAMES[i + 1:]:
            hit = sum(1 for p in preds if p[a] is not None and p[a] == p[b])
            out[f"{a}|{b}"] = hit / len(docs)
    return out

def attribute(docs: Sequence[Doc], predictor, offset: int,
              vocab: Vocab) -> Dict[str, dict]:
    """观测一致率。主读数是同一子集上的配对对比：在 S_k={规则 k 与真值分歧}
    上，rate_disc = 模型跟 k 一致的比例，rate_truth_disc = 模型跟真值一致的
    比例，两者之差是证据强度。单看 rate_disc 会被"两条都不中"的情形误导。

    真值行的 disc 子集按定义为空（k==TRUTH 时条件恒假），早期版本因此
    返回 nan，而 dominant_rule 拿 nan 做比较恒为 False，相图会常数化成
    last_value。真值行改用全集准确率。
    """
    rows = {k: dict(agree=0, n=0, agree_disc=0, truth_disc=0, n_disc=0)
            for k in RULE_NAMES}
    for d in docs:
        assert not d.q_hist_k, "probe A 不支持 hist 查询"
        p = rule_predictions(d, offset)
        got = predictor.predict(d)
        for k in RULE_NAMES:
            if p[k] is None:
                continue
            r = rows[k]
            r["n"] += 1
            r["agree"] += int(p[k] == got)
            if k == TRUTH or p[k] == p[TRUTH]:
                continue
            r["n_disc"] += 1
            r["agree_disc"] += int(p[k] == got)
            r["truth_disc"] += int(p[TRUTH] == got)
    nan = float("nan")
    for r in rows.values():
        r["rate"] = r["agree"] / r["n"] if r["n"] else nan
        r["rate_disc"] = r["agree_disc"] / r["n_disc"] if r["n_disc"] else nan
        r["rate_truth_disc"] = (r["truth_disc"] / r["n_disc"]
                                if r["n_disc"] else nan)
    rows[TRUTH]["rate_disc"] = rows[TRUTH]["rate"]
    rows[TRUTH]["rate_truth_disc"] = rows[TRUTH]["rate"]
    return rows



def _margin(predictor, view, v_star: int, v_truth: int) -> float:
    a, b = predictor.logp(view, [v_star, v_truth])
    return a - b

def _mass(predictor, view, cands: Sequence[int]) -> float:
    """候选对占据的概率质量。OOD 探针的有效性上界：break_rarity 让答案值在
    前缀重复出现，是训练分布外的配置，若质量散到 {v*, truth} 之外，
    d_margin 度量的就不再是这两条规则的竞争。<0.5 视为该读数无效。
    合成 RulePredictor 的 logp 未归一化，此处返回值对它无意义，只用于真模型。"""
    import math
    return sum(math.exp(x) for x in predictor.logp(view, cands))

def causal(docs: Sequence[Doc], predictor, kind: str, offset: int,
           vocab: Vocab, cfg: CorpusCfg, seed: int = 0) -> dict:
    """dd_cond vs dd_all 量化选择偏置：编辑域与 ΔD 相关，可用子集的 ΔD 均值
    会偏离全体，须与读数并列报告。"""
    rng = random.Random(seed)
    _, tgt, sign = EDITS[kind]
    deltas, dd, masses = [], [], [] 
    for d in docs:
        ed = apply_edit(d, kind, vocab, cfg, rng, offset)
        if ed is None:
            continue
        dd.append(d.realized_delta)
        truth = r_last_value(d)
        deltas.append(_margin(predictor, ed, ed.v_star, truth)
                  - _margin(predictor, d, ed.v_star, truth))
        masses.append(_mass(predictor, ed, [ed.v_star, truth]))

    nan = float("nan")
    n = len(deltas)
    return dict(kind=kind, target=tgt, sign=sign, n=n,
                yield_rate=n / len(docs),
                d_margin=(sum(deltas) / n) if n else nan,
                frac_expected=(sum(sign * x > 0 for x in deltas) / n) if n else nan,
                dd_cond=(sum(dd) / len(dd)) if dd else nan,
                dd_all=sum(x.realized_delta for x in docs) / len(docs),
                mass_mean=(sum(masses) / n) if n else nan, 
                d_all=deltas,        # 逐篇 Δ。frac+ 与均值矛盾（R5/D5: Δ=+0.52 而
                     # frac+=0.56）说明效应由少数文档承载，均值不足以
                     # 描述分布。200 篇约 3KB/probe 点。
                )


def contamination(docs: Sequence[Doc], kind: str, offset: int, vocab: Vocab,
                  cfg: CorpusCfg, seed: int = 0) -> Tuple[Dict[str, float], int]:
    """编辑连带移动了哪些非目标规则的预测。目标规则应为 1.0，其余越低越好。"""
    rng = random.Random(seed)
    moved = {k: 0 for k in RULE_NAMES}
    n = 0
    for d in docs:
        ed = apply_edit(d, kind, vocab, cfg, rng, offset)
        if ed is None:
            continue
        n += 1
        pb, pe = rule_predictions(d, offset), rule_predictions(ed, offset)
        for k in RULE_NAMES:
            if pb[k] != pe[k]:
                moved[k] += 1
    nan = float("nan")
    return {k: (v / n if n else nan) for k, v in moved.items()}, n


# ---------------- 自检 ----------------

def probe_selfcheck(cfg: CorpusCfg, vocab: Vocab, n: int = 600,
                    verbose: bool = True) -> dict:
    docs = list(generate_corpus(vocab, cfg, n, seed_offset=1))
    offset = fit_position_offset(docs)
    for d in docs:               # 校验器必须放过所有 base 文档，否则它在误伤
        assert _order_ok(d.stmts), "_order_ok 拒绝了生成器产出的文档"
    rng = random.Random(0)
    stats: Dict[str, dict] = {}

    # ---- 旋钮 6：按 is_break 分层 ----
    # 逐编辑的断言只在非反转文档上跑，与 go_nogo 的读数域一致。理由不是
    # 「反转文档上标签份数不同」这么笼统：relabel_old 会把窗口里的末代值副本
    # 一起覆写成新值，标签份数从 n_new_kept 掉回 1，任何统一的期望份数都是错的。
    # 过滤到 base 之后，下面 vals.count(ed.answer)==1 那条原始断言不用放宽。
    base = [d for d in docs if not getattr(d, "is_break", False)]
    brk = [d for d in docs if getattr(d, "is_break", False)]
    assert base, "没有非反转文档，探针无域"
    if getattr(cfg, "p_break", 0.0) > 0.0:
        assert brk, (
            f"p_break={cfg.p_break} 但一篇反转文档都没有（n={n}）。两个可能："
            f"n 太小（p_break×n < 1），或 generator 没把 is_break 写进 Doc —— "
            f"后者的症状是多重性确实翻了（collide 的 RARITY|RECENCY 掉到 "
            f"1−p_break、covar 的 revQ≈p_break）而 Doc.is_break 恒 False。"
            f"判别式：answer_val_id 是否为 -1，nb_pair.py 的 check_inert 在 "
            f"p_break=0 下就能测出来。")
        # 非反转文档上式 1 必须精确成立。这是硬断言：它在 arm 里没被改动，
        # 目标函数与 Bayes 规则也没变，这正是 N− 作为"同任务对照"的依据。
        for d in base:
            assert r_rarity(d) == r_last_value(d), \
                "非反转文档上 rarity 与 last_value 分歧，式 1 被意外破坏"
        # 反转文档上别名应当破除，但存在有效损耗：窗口越界可能把末代值的副本
        # 丢到只剩钉在 p_final 那一份，counts 退化成 {老值:1, 末代值:1}，
        # rarity 的平票规则裁给更近者，于是又与 last_value 同指。这些文档的
        # 别名没破除，且在 N+ 下更糟 —— 标签取的是 q_history[-2]（老值），
        # 而实现出来的 argmin 指向末代值，该篇的标签不由任何一条候选规则产生，
        # 是训练信号里的纯噪声。故须量化而非断言掉，并让 collide.py 的
        # RECENCY|RARITY 对齐到 1-p_break×(1-degen) 而不是 1-p_break。
        degen = sum(1 for d in brk if r_rarity(d) == r_last_value(d)) / len(brk)
        stats["break"] = dict(
            frac=len(brk) / len(docs), degen=degen,
            n_old=sum(d.n_old_kept for d in brk) / len(brk),
            n_new=sum(d.n_new_kept for d in brk) / len(brk))
        assert degen < 0.02, (
            f"{degen:.1%} 的反转文档别名未破除（末代值副本被窗口越界丢尽）。"
            f"N+ 下这些篇的标签无规则可依，须先收窄 spread 或提高 n_stmts_lo")
        # 反转文档必须被 break_rarity 的域排除，否则读数混入训练分布内的配置
        assert all(apply_edit(d, "break_rarity", vocab, cfg, rng, offset) is None
                   for d in brk), "反转文档未被 break_rarity 域排除"

    for kind, (_, tgt, sign) in EDITS.items():
        ok = 0
        for d in base:
            ed = apply_edit(d, kind, vocab, cfg, rng, offset)
            if ed is None:
                continue
            ok += 1
            assert len(ed.tokens) == len(d.tokens), \
                f"{kind}: token 数 {len(d.tokens)} -> {len(ed.tokens)}"
            assert ed.answer_pos == d.answer_pos, f"{kind}: answer_pos 变了"
            assert ed.answer == d.answer, f"{kind}: 真值被改"
            assert ed.n_stmts == d.n_stmts, f"{kind}: 语句数变了"
            assert ed.v_star != r_last_value(d), f"{kind}: v* 与真值重合"
            assert _q_gap(ed) == _q_gap(d), \
                f"{kind}: q_gap 变了 {_q_gap(d)}->{_q_gap(ed)}"
            assert _order_ok(ed.stmts), f"{kind}: 编辑后时序非法"
            if kind == "shift_delta":
                assert ed.realized_delta > d.realized_delta, "shift 未生效"
            else:
                assert ed.realized_delta == d.realized_delta, f"{kind}: ΔD 变了"
            vals = [t for t in ed.tokens[:ed.answer_pos] if vocab.is_val(t)]
            if kind == "break_rarity":
    # 设计使然：编辑把老值副本改写成新值，答案值必然重复出现。
    # 这是该探针唯一的 OOD 维度，其余断言全部保留。
               assert vals.count(ed.answer) >= 2, \
        "break_rarity 未让答案值重复，编辑无效"
            else:
                assert vals.count(ed.answer) == 1, \
                f"{kind}: 答案值在前缀出现 {vals.count(ed.answer)} 次"
        stats[kind] = dict(yield_rate=ok / len(base), target=tgt, sign=sign)

    cov: Dict[str, float] = {}
    for rule, kinds in RULE_EDITS.items():
        hit = sum(1 for d in base
                  if any(apply_edit(d, k, vocab, cfg, rng, offset) is not None
                         for k in kinds))
        cov[rule] = hit / len(base)
    stats["coverage"] = cov

    # 机器自检 1：合成预测器必须被 attribute 读回它自己那条规则
    for rule in RULE_NAMES:
        rows = attribute(docs, RulePredictor(rule, offset), offset, vocab)
        assert rows[rule]["n_disc"] == 0 or rows[rule]["rate_disc"] > 0.99, \
            f"attribute 读不回 {rule}: {rows[rule]}"

    # 机器自检 2：Δ 是恒等式 —— 规则型 sign·2·sharp，跟踪型精确 0
    contam: Dict[str, Dict[str, float]] = {}
    for kind, (_, tgt, sign) in EDITS.items():
        if stats[kind]["yield_rate"] < YIELD_MIN:
            continue
        hot = causal(base, RulePredictor(tgt, offset), kind, offset, vocab, cfg)
        cold = causal(base, RulePredictor(TRUTH, offset), kind, offset, vocab, cfg)
        assert sign * hot["d_margin"] > 1.0, \
            f"{kind}: 规则型 Δ={hot['d_margin']:+.2f}（期望符号 {sign:+d}），编辑无力"
        assert abs(cold["d_margin"]) < 1e-9, \
            f"{kind}: 跟踪型 Δ={cold['d_margin']:+.3f}，编辑有副作用"
        assert hot["frac_expected"] > 0.99, \
            f"{kind}: 符号一致率仅 {hot['frac_expected']:.3f}"
        stats[kind].update(d_hot=hot["d_margin"], d_cold=cold["d_margin"],
                           dd_cond=hot["dd_cond"], dd_all=hot["dd_all"])
        moved, _ = contamination(docs, kind, offset, vocab, cfg)
        contam[kind] = moved
    stats["contam"] = contam

    ident = identifiability(docs, offset)
    stats["offset"] = {"v": offset}
    stats["ident"] = ident

    if verbose:
        print(f"[{cfg.name}] offset={offset}  规则覆盖: " +
              " ".join(f"{k}={v:.2f}" for k, v in cov.items()))
        for k in EDITS:
            v = stats[k]
            if "d_hot" in v:
                side = [f"{r}={contam[k][r]:.2f}" for r in RULE_NAMES
                        if r != v["target"] and contam[k][r] > CONTAM_WARN]
                print(f"    {k:<18} yield={v['yield_rate']:.2f} "
                      f"Δhot={v['d_hot']:+.1f} Δcold={v['d_cold']:+.3f} "
                      f"ΔD={v['dd_cond']:.1f}/{v['dd_all']:.1f}"
                      + ("  连带: " + " ".join(side) if side else ""))
            else:
                print(f"    {k:<18} yield={v['yield_rate']:.2f}  (域太小，跳过)")
        top = sorted(ident.items(), key=lambda kv: -kv[1])[:3]
        print("    规则碰撞率 top3: " + ", ".join(f"{a}={b:.3f}" for a, b in top))
    return stats


if __name__ == "__main__":
    spec = LangSpec()
    vocab = Vocab(spec)
    for r, d in [(1, 2), (1, 16), (12, 2), (12, 16), (3, 5)]:
        dlo, dhi = dd_band(d)
        probe_selfcheck(CorpusCfg(name=f"R{r}_D{d}", seed=0, p_update=0.5,
                                  max_updates=1, r_old_lo=r, r_old_hi=r,
                                  use_marker=False, delta_d_lo=dlo,
                                  delta_d_hi=dhi, p_hist_query=0.0), vocab)

    # 旋钮 6 的两个格（与 run plan 一致）。不跑这几行，本文件新加的分层断言
    # （式 1 在非反转文档上精确成立、degen 门、反转文档被 break_rarity 域排除）
    # 一次都不会被执行。
    # R_old=1 不能进这里：rb = r_old_hi = 1 不大于 k_old_break = 1，
    # validate_cfg 会拒（rarity 仍指向末代值，别名未破除）。
    print("\n== 旋钮 6：别名破除臂 ==")
    # k/rb 须满足 k+rb = R+1（反转 slot 与常规 slot 语句数相同）且 rb>k。
    # R3 只有 (1,3) 一种分法，rb=0 表示取 r_old_hi。
    # R16 取 k 最大的 (8,9)：老值存活的条件从「q_old 全部 16 个位置非负」松成
    # 「第 8 小的非负」。前者概率 (33/91)^16≈9e-8，64 次重试必失败，反转文档
    # 100% 走 clamp 回退，q_gap 从 5.30 塌到 1.71，构成绕过 query 的旁路。
    for r, d, k, rb in [(3, 8, 1, 0), (5, 8, 1, 0), (8, 8, 1, 0)]:
        dlo, dhi = dd_band(d)
        for tr in ("recency", "rarity"):
            cfg = CorpusCfg(name=f"nb{tr[:3]}_R{r}_D{d}", seed=0, p_update=0.5,
                            max_updates=1, r_old_lo=r, r_old_hi=r,
                            use_marker=False, delta_d_lo=dlo, delta_d_hi=dhi,
                            p_hist_query=0.0, n_stmts_lo=45, n_stmts_hi=55,
                            p_break=0.10, truth_rule=tr,
                            k_old_break=k, r_new_break=rb)
            st = probe_selfcheck(cfg, vocab, n=1200)
            b = st.get("break", {})
            if b:
                print(f"    反转 {b['frac']:.3f}  别名未破除(degen) "
                      f"{b['degen']:.4f}  老值/末代值存活 "
                      f"{b['n_old']:.2f}/{b['n_new']:.2f}")