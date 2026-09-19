"""Generate controlled assignment-language documents."""
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

from config import LangSpec, CorpusCfg, validate_cfg
from vocab import Vocab

STMT_TOKENS = 4          # ent attr val SEP
STMT_TOKENS_MARKED = 5   # upd ent attr val SEP


@dataclass
class Stmt:
    ent: int
    attr: int
    val: int
    is_update: bool


@dataclass
class Doc:
    tokens: List[int]
    answer: int
    answer_pos: int
    q_ent: int
    q_attr: int
    q_hist_k: Optional[int]
    realized_delta: int
    val_history: List[int]
    n_stmts: int
    n_slots: int
    upd_positions: List[int]
    n_tail_updates: int         # p_final 之后的 update 数（协变量，不强制）
    q_gap: int                  # q_final 到最近同 slot 老值的距离
    fill_gap_late: float        # 后半段填充 update 的同一距离均值
    adj_q: int                  # q_final 的前驱是否同 slot（0/1）
    adj_fill: float            # 填充 update 的同一比率
    fill_gap_near: float        # |p - p_final| <= 0.1*N 的填充 update 的同一距离
    w_st: int                   # 实际用的 q_old 窗口宽度（诊断用）
    fill_quint: List[float]     # 填充语句 update 占比，按最终位置五分位
    key_quint: List[float]      # 同上，但按 key 五分位（裁剪前的画布坐标）
    q_clamped: bool             # 回退到 clamp 放置（应 ≪1%，越界诊断用）
    q_kept: int                 # 实际进文档的 q_old 条数（越界丢弃后），realized R_old
    stmts: List[Stmt]           # 组装后的语句序列，探针据此做最小编辑
    # ---- 旋钮 6（p_break）。默认值让旧数据反序列化后语义不变 ----
    is_break: bool = False      # 本篇被查询 slot 的多重性是否被反转
    answer_val_id: int = -1     # 标签的 raw value id。N+ 的反转文档上它是老值，
                                # 与 val_history[-1] 不同，probe._answer_val 据此
                                # 取值而非从 val_history 反推。-1 表示旧数据。
    n_old_kept: int = 0         # q slot 内老值实际进文档的份数（越界丢弃后）
    n_new_kept: int = 0         # 同上，末代值。N− 的反转文档上标签在前缀出现
                                # 这么多次，selfcheck 第 3 条据此分层给期望值。


class _ValueDraw:
    """文档级值池。同一篇文档内每个 value id 至多归属一个 slot。

    这让"答案值在文档中只出现一次"成为结构事实而非 1-1/n_values 的概率事件，
    否则填充 slot 偶然撞上答案值时指标有本底噪声，两条 arm 不再严格可比。
    """

    def __init__(self, rng: random.Random, n_values: int):
        self._rng, self._n, self._used = rng, n_values, set()

    def take(self, k: int) -> List[int]:
        out = []
        tries = 0
        while len(out) < k:
            tries += 1
            if tries > 64 * k + 256:
                raise RuntimeError(
                f"值池耗尽：已用 {len(self._used)}/{self._n}，还需 "
                f"{k - len(out)} 个。每篇文档消耗约 1.5×slot 数个值，而 "
                f"slot 数 ≈ n_stmts/((1+p_update·R_old)/(1+spread))，"
                f"故 R_old 越小消耗越大（R_old=1 是最坏格，"
                f"n_stmts=55 时约需 120 个值）。提高 n_values。")
            v = self._rng.randrange(self._n)
            if v not in self._used:
                self._used.add(v)
                out.append(v)
        return out


def _build_slot(rng: random.Random, cfg: CorpusCfg, ent: int, attr: int,
                force_update: bool, draw: _ValueDraw,
                reps_old: Optional[int] = None, reps_final: int = 1
                ) -> Tuple[List[Stmt], List[int]]:
    """返回该 slot 的语句序列（时序）与值历史。默认被取代的值重复 R_old 次、
    末代值 1 份，这就是式 1 的别名：counts={v_old:R, v_new:1}，argmin count 与
    argmax pos 同指末代值。

    reps_old / reps_final 非默认时反转这个多重性（旋钮 6）。两处细节：

    last>0 的守卫：无 update 的填充 slot 只有一代值，它既是首代也是末代。
    若让它吃 reps_final，单例填充 slot 会被撑成 R 份，slot 数分布随 p_break
    变化；slot 数本身也是需要记录的格间协变量。

    is_update 的非对称不可避免：标记按世代打（i>0），整代重复 reps 次，故反转
    slot 有 R 条 update 而常规 slot 只有 1 条。想只标末代第一份的话 flags 会是
    [F,T,F,...]，违反 _order_ok 的单调不减，而 probe_selfcheck 断言 _order_ok
    必须放过所有 base 文档。因此 validate_cfg 在 p_break>0 时禁 use_marker
    （否则 UPD token 数泄漏 is_break），诊断侧则须按 is_break 分层统计
    update 密度，池化会报假阳性。

    RNG：reps_old 给定时跳过 randint（即便 r_old_lo==r_old_hi 它也消耗随机数），
    故反转文档与常规文档的流不同步。这不影响任何要求 —— p_break=0 时无反转
    文档、流与主网格逐比特相同；N+ 与 N− 在同 seed 下走的是同一条分支序列，
    前缀仍逐比特相同。
    """
    if force_update:
        n_upd = rng.randint(1, cfg.max_updates)
    else:
        n_upd = rng.randint(1, cfg.max_updates) if rng.random() < cfg.p_update else 0
    history = draw.take(n_upd + 1)
    stmts = []
    last = len(history) - 1
    for i, v in enumerate(history):
        if i == last:
            reps = reps_final if last > 0 else 1
        elif reps_old is not None:
            reps = reps_old
        else:
            reps = rng.randint(cfg.r_old_lo, cfg.r_old_hi)
        for _ in range(reps):
            stmts.append(Stmt(ent, attr, v, is_update=(i > 0)))
    return stmts, history


def _fresh_pair(rng: random.Random, spec: LangSpec, used: set) -> Tuple[int, int]:
    while True:
        e, a = rng.randrange(spec.n_entities), rng.randrange(spec.n_attrs)
        if (e, a) not in used:
            used.add((e, a))
            return e, a


def _keyed(stmts: List[Stmt], rng: random.Random, w: float,
           out: List[Tuple[float, Stmt]]) -> int:
    """给一个 slot 的语句分配排序键，保持 slot 内时序。返回落在画布内的语句数。

    锚点 u ~ U(0, 1+w)，老值键 = u - U(0,w)，越界（<0 或 >1）的语句丢弃。
    右侧留 w 余量 + 丢弃而非夹取，是让 update 键与老值键在 [0,1] 上同为均匀
    密度的唯一办法：老值在 t 的密度 = ∫_t^{t+w}(1/w)du = 1，与 update 相同，
    故 update 占比与位置无关。

    旧实现取 u~U(0,1) 且窗口 clamp 到 [0,u]：老值密度在 t→1 处按 (1-t)/w
    归零、在 t→0 处发散，末段只剩 update。秩不由自己的键决定而由全体键的
    经验分布决定，"u 均匀故秩均匀"是错的。EXTREME_A 实测 1.51 倍，且该偏差
    随 R_old 单调增强（老值质量占比 = R·p/(1+R·p)），与相图主轴共线。

    代价：靠前的 update 会丢掉部分老值，即"老值不在文档内的 update"。这是
    边界效应的正确去处 —— 它只影响文档开头，而 q_final 恒在 N-1-ΔD，
    两者不重叠；且它顺带打散了"重复出现的值必被取代"这条附带线索。
    """
    u = rng.uniform(0.0, 1.0 + w)
    kept = 0
    for k, st in zip(sorted(u - rng.uniform(0.0, w) for _ in range(len(stmts) - 1)),
                     stmts[:-1]):
        if 0.0 <= k <= 1.0:
            out.append((k, st))
            kept += 1
    if u <= 1.0:
        out.append((u, stmts[-1]))
        kept += 1
    return kept

def emit(seq: List[Stmt], q_ent: int, q_attr: int, hist_k: Optional[int],
         answer_val: int, vocab: Vocab, cfg: CorpusCfg) -> Tuple[List[int], int]:
    """语句序列 -> token 序列。返回 (tokens, answer_pos)。

    探针的最小编辑改 stmts 后走同一条发射路径，保证编辑前后除目标位点
    以外逐 token 相同。answer_pos 由前缀长度决定，故 token 数不变
    等价于 answer_pos 不变（probe_selfcheck 两个都查）。
    """
    toks: List[int] = []
    for st in seq:
        if cfg.use_marker and st.is_update:
            toks.append(vocab.UPD)
        toks += [vocab.ent(st.ent), vocab.attr(st.attr), vocab.val(st.val), vocab.SEP]
    toks += [vocab.QUERY, vocab.ent(q_ent), vocab.attr(q_attr)]
    if hist_k is not None:
        toks.append(vocab.time(hist_k))
    toks.append(vocab.ARROW)
    answer_pos = len(toks)
    toks.append(vocab.val(answer_val))
    return toks, answer_pos

def sample_document(vocab: Vocab, cfg: CorpusCfg, rng: random.Random) -> Doc:
    spec = vocab.spec
    draw = _ValueDraw(rng, spec.n_values)

    delta_d = rng.randint(cfg.delta_d_lo, cfg.delta_d_hi)
    # N 独立采样：长度是自由变量，不携带 ΔD / R_old 的信息
    n_stmts = rng.randint(cfg.n_stmts_lo, cfg.n_stmts_hi)
    p_final = n_stmts - 1 - delta_d

    used: set = set()
    q_ent, q_attr = _fresh_pair(rng, spec, used)
    rb = cfg.r_new_break or cfg.r_old_hi
    # is_break 在 q slot 之前抽。truth_rule 完全不消耗随机数，故同 seed 下
    # N+ 与 N− 的 is_break 序列相同、_build_slot 的实参相同、token 前缀逐比特
    # 相同，两条 arm 只在反转文档的标签 token 上不同。这是整个设计的地基
    # （见 pair 断言脚本）：探针只读 tokens[:answer_pos]，故两条 arm 的探针
    # 输入分布严格相同，读数差异只能来自目标函数。
    # p_break=0 时短路求值不调用 rng.random()，流与已发表的主网格逐比特一致。
    is_break = cfg.p_break > 0.0 and rng.random() < cfg.p_break
    if is_break:
        q_stmts, q_history = _build_slot(rng, cfg, q_ent, q_attr, True, draw,
                                         reps_old=cfg.k_old_break, reps_final=rb)
    else:
        q_stmts, q_history = _build_slot(rng, cfg, q_ent, q_attr, True, draw)
    # 反转 slot 是 [老值×k, 末代值×rb]，故 q_stmts[-1] 仍是末代值（钉在
    # p_final），而 q_old 变成 [老值×k, 末代值×(rb-1)] —— 窗口里混有末代值的
    # 副本。这正是别名被破除的形态：counts={老值:k, 末代值:rb}，argmin count
    # 指向老值而 argmax pos 仍指向末代值。
    q_final, q_old = q_stmts[-1], q_stmts[:-1]
    assert p_final >= len(q_old), (n_stmts, len(q_old), delta_d)


# ---- 填充语句：带键，稍后按键排序填空位 ----
# 多生成一些，裁剪推迟到 q_old 存活数已知之后（q_old 可能因越界被丢弃，
# 丢多少就要多填多少）。裁剪按 key 序切一条线，对所有 slot 一视同仁。
    budget0 = n_stmts - 1 - len(q_old)      # q_old 全存活时的填充预算
    exp_per_slot = (1.0 + cfg.p_update * (cfg.r_old_lo + cfg.r_old_hi) / 2.0) \
    / (1.0 + cfg.spread)
    need = budget0 + len(q_old) + 8         # top-up 余量 + 裁剪余量
    keyed: List[Tuple[float, Stmt]] = []
    while len(keyed) < need:
        for _ in range(max(1, int((need + 4 - len(keyed)) / exp_per_slot))):
            e, a = _fresh_pair(rng, spec, used)
            # 填充 slot 必须以同一比率反转（不变量 6）。否则「末代值被重复」
            # 就是定位被查询 slot 的完美判别式，模型无需读 query 即可答题。
            # fb 在 _build_slot 之前抽，与它内部的 n_upd 独立，故「有 update 的
            # slot 中被反转的比例」在两侧同为 p_break —— 无 update 的填充 slot
            # 只有一代值，_build_slot 的 last>0 守卫让 reps_final 不生效，反转
            # 对它是空操作，不进入这个比例的分母。
            fb = cfg.p_break > 0.0 and rng.random() < cfg.p_break
            s, _ = _build_slot(rng, cfg, e, a, False, draw,
                               reps_old=cfg.k_old_break if fb else None,
                               reps_final=rb if fb else 1)
            _keyed(s, rng, cfg.spread, keyed)
    keyed.sort(key=lambda t: t[0])
    del keyed[need:]        # 固定裁剪线，T 才不受批量生成过冲的影响

    kq = [0.0] * 5
    if keyed:
        step = len(keyed) / 5.0
        for b in range(5):
            seg = keyed[int(b * step):int((b + 1) * step)]
            kq[b] = (sum(st.is_update for _, st in seg) / len(seg)) if seg else float("nan")

# ---- 组装：q_final 钉死在 p_final，q_old 按填充侧的同一法则放置 ----
# w_st 不再是 spread×n_stmts，而是由实际画布密度反推：budget0 条填充语句
# 占 key∈[0,T]，故 key 宽 spread 对应 spread/T×n_stmts 个位置。硬编码
# spread×n_stmts 等于假定 T=1，任何裁剪都会让两侧坐标系错位（本轮 -14%）。
# 越界的 q_old 丢弃而非挤压：clamp 会把 R_old 个老值压进截短的窗口，
# 使 q_gap 随 R_old 系统偏小，而填充侧同处境是丢弃、gap 偏大。
    seq: List[Optional[Stmt]] = [None] * n_stmts
    seq[p_final] = q_final
    T = keyed[budget0 - 1][0]
    w_st = max(len(q_old) + 1, int(round(cfg.spread / T * n_stmts)))
    win = range(p_final - w_st, p_final)        # 可含负位置，越界即丢弃
    want = {st.val for st in q_old}
    q_clamped = False
    for _ in range(64):
        pairs = [(p, st) for p, st in
             zip(sorted(rng.sample(win, len(q_old))), q_old) if p >= 0]
        # 每个不同的老值至少留一份：全丢会使该篇没有冲突（hist 查询更会无解）
        if {st.val for _, st in pairs} == want:
            break
    else:
        lo = max(0, p_final - w_st)
        pairs = list(zip(sorted(rng.sample(range(lo, p_final), len(q_old))), q_old))
        q_clamped = True
    for p, st in pairs:
        seq[p] = st
    del keyed[n_stmts - 1 - len(pairs):]       # 丢了几条 q_old 就多填几条
    it = iter(st for _, st in keyed)
    for i in range(n_stmts):
        if seq[i] is None:
            seq[i] = next(it)
    assert seq[p_final] is q_final and all(s is not None for s in seq)

    # ---- q slot 内两代值的实际存活份数（越界丢弃之后）----
    # pairs 是 q_old 的存活者。常规文档它全是老值，故 n_new_kept 恒为 1
    # （只有钉在 p_final 的那份）。反转文档的 q_old 混有末代值的副本，两个量
    # 都要单独数：selfcheck 第 3 条用 n_new_kept 给 N− 的标签份数定期望值，
    # covar 用两者之比检查不变量 6。
    v_final = q_history[-1]
    n_new_kept = sum(1 for _, st in pairs if st.val == v_final) + 1
    n_old_kept = len(pairs) + 1 - n_new_kept
    # N+ 的标签是老值，它若被越界丢弃则标签不在文档内、该篇无解。want 重试
    # 循环（每个不同的老值至少留一份）与 clamp 回退（位置全非负故全存活）
    # 已经保证这点，此处把隐式保证变显式 —— 反转文档的 want 有两个元素而
    # 老值恰在 q_old[0]（拿最小的位置键，最易越界成负），比常规文档更吃紧。
    if is_break and cfg.truth_rule == "rarity":
        assert n_old_kept >= 1, (
            f"反转文档的老值全部越界丢弃，N+ 的标签不在文档内 "
            f"(n_stmts={n_stmts}, p_final={p_final}, w_st={w_st})")

    # ---- 查询类型 ----
    hist_k = None
    if cfg.p_hist_query > 0:
        if rng.random() < cfg.p_hist_query and len(q_history) >= 2:
            hist_k = rng.randint(1, min(len(q_history) - 1, spec.n_time_idx - 1))
        else:
            hist_k = 0
    # 标签。truth_rule 只改这里一个已抽好的值，不消耗随机数 —— 这是 N+/N−
    # 前缀逐比特相同的来源（见 is_break 处）。
    # N+（rarity）在反转文档上标签是老值 q_history[-2]：反转 slot 的 counts 是
    # {老值:k, 末代值:rb} 且 validate_cfg 强制 rb>k，故 argmin count 指向老值。
    # 非反转文档上两条规则同指末代值（式 1），标签与主网格完全一致。
    if cfg.truth_rule == "rarity" and is_break:
        answer_val = q_history[-2]
    else:
        answer_val = q_history[-1] if not hist_k else q_history[-1 - hist_k]

# ---- 发射 ----
    toks, answer_pos = emit(seq, q_ent, q_attr, hist_k, answer_val, vocab, cfg)
    assert len(toks) <= spec.ctx_len, (len(toks), spec.ctx_len)

    # ---- 结构统计。gap 与 adj 都须在 q_slot 与填充 slot 间同分布 ----
    def same(i: int, j: int) -> bool:
        return seq[i].ent == seq[j].ent and seq[i].attr == seq[j].attr

    fq = [0.0] * 5
    fill_idx = [i for i in range(n_stmts)
            if not (seq[i].ent == q_ent and seq[i].attr == q_attr)]
    if fill_idx:
        step = len(fill_idx) / 5.0
        for b in range(5):
            seg = fill_idx[int(b * step):int((b + 1) * step)]
            fq[b] = (sum(seq[i].is_update for i in seg) / len(seg)) if seg else float("nan")

    upd_pos = [i for i, s in enumerate(seq) if s.is_update]
    prev_q = [i for i in range(p_final) if same(i, p_final)]
    q_gap = p_final - max(prev_q) if prev_q else p_final
    adj_q = int(p_final > 0 and same(p_final - 1, p_final))

# sample_document 的结构统计段，gaps 之外再收一个近端桶
    gaps, gaps_near, adj = [], [], []
    near = max(2, int(0.1 * n_stmts))
    for p in upd_pos:
        if p == p_final:
            continue
        prev = [i for i in range(p) if same(i, p)]
        if prev and p >= p_final * 0.5:
            gaps.append(p - max(prev))
        if prev and abs(p - p_final) <= near:
            gaps_near.append(p - max(prev))
        if p > 0:
            adj.append(int(same(p - 1, p)))

    return Doc(
        tokens=toks, answer=vocab.val(answer_val), answer_pos=answer_pos,
        q_ent=q_ent, q_attr=q_attr, q_hist_k=hist_k,
        realized_delta=delta_d, val_history=q_history, n_stmts=n_stmts,
        n_slots=len({(s.ent, s.attr) for s in seq}),
        upd_positions=upd_pos,
        n_tail_updates=sum(1 for p in upd_pos if p > p_final),
        q_gap=q_gap,
fill_gap_late=(sum(gaps) / len(gaps)) if gaps else float("nan"),
fill_gap_near=(sum(gaps_near) / len(gaps_near)) if gaps_near else float("nan"),
adj_q=adj_q,
adj_fill=(sum(adj) / len(adj)) if adj else float("nan"),
w_st=w_st, fill_quint=fq, key_quint=kq,
q_clamped=q_clamped, q_kept=len(pairs), stmts=list(seq),
is_break=is_break, answer_val_id=answer_val,
n_old_kept=n_old_kept, n_new_kept=n_new_kept)


        

def generate_corpus(vocab: Vocab, cfg: CorpusCfg, n_docs: int, seed_offset: int = 0):
    """seed_offset 让评估集与训练集的 rng 流不相交：训练用 0，探针/评估用 1。
    同一 cfg 下两者的文档分布相同，但没有一篇重合。"""
    validate_cfg(cfg, vocab.spec)
    rng = random.Random(cfg.seed + 1_000_003 * seed_offset)
    for _ in range(n_docs):
        yield sample_document(vocab, cfg, rng)


def to_padded_batch(docs, vocab: Vocab):
    """右填充到 ctx_len + loss mask。训练用动态填充，见 train.py。"""
    L = vocab.spec.ctx_len
    ids, labels = [], []
    for d in docs:
        pad = L - len(d.tokens)
        ids.append(d.tokens + [vocab.PAD] * pad)
        labels.append(d.tokens + [-100] * pad)
    return ids, labels