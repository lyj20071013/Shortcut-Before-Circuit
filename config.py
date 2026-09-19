"""Dataclass definitions for the synthetic language and corpus."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LangSpec:
    """语言规模。跨所有配置固定，不作为旋钮。"""
    n_entities: int = 2000
    n_attrs: int = 8
    n_values: int = 512
    n_time_idx: int = 4        # @0..@3，仅在 hist query 开启时使用
    ctx_len: int = 600

    @property
    def n_bindings(self) -> int:
        """可能的 (e,a,v) 三元组数。远超模型容量是设计要求。"""
        return self.n_entities * self.n_attrs * self.n_values


@dataclass(frozen=True)
class CorpusCfg:
    """六个数据侧旋钮 + 文档形状随机化参数。"""
    name: str
    seed: int = 0

    # 旋钮 1：更新事件频率（非查询 slot 收到更新的概率）
    p_update: float = 0.5
    max_updates: int = 1

    # 旋钮 2：旧值冗余度 R_old ~ U[lo, hi]，作用于每个被取代的值
    r_old_lo: int = 1
    r_old_hi: int = 1

    # 旋钮 3：更新是否带显式标记
    use_marker: bool = False

    # 旋钮 4：查询 slot 的末次更新与查询之间的语句数 ΔD ~ U[lo, hi]
    # 现在是精确成立的硬约束，不再是"请求值"
    delta_d_lo: int = 1
    delta_d_hi: int = 8

    # 旋钮 5：历史索引查询的比例。>0 时所有查询都带 @k
    p_hist_query: float = 0.0

    # ---- 文档形状 ----
    # 语句总数直接采样，与 ΔD / R_old 解耦：这是长度随机化的唯一来源。
    # 下界必须容纳最坏情况 max_updates*r_old_hi + 1 + delta_d_hi，
    # 否则 N 会被条件化，长度重新与旋钮相关。validate_cfg 负责拦截。
    n_stmts_lo: int = 60
    n_stmts_hi: int = 100
    min_slots: int = 4          # 文档内最少不同 slot 数，保证实体多样性
    spread: float = 0.80     # slot 局部窗口宽度 = 语句数 × spread

    # 旋钮 6：别名破除。p_break 比例的文档把被查询 slot 的多重性反转成
    # {老值: k_old_break, 末代值: r_new_break}，于是 rarity 指向老值、
    # recency 指向末代值，式 1 的共延在这些文档上不成立。
    # truth_rule 决定反转文档的标签，且不消耗随机数：同 seed 下 N+ 与 N− 的
    # token 前缀逐比特相同，只有反转文档的标签 token 不同，故两条 arm 的读数
    # 差异只能来自目标函数。
    # p_break=0 且 truth_rule=recency 时 rng.random() 因短路求值不被调用，
    # 该配置与主网格逐比特相同 —— 已发表 75 个 run 的可复现性不受影响，且
    # N− 的剂量曲线在 p_break=0 处直接复用它们作为锚点。
    p_break: float = 0.0
    k_old_break: int = 1
    r_new_break: int = 0          # 0 -> 取 r_old_hi，令反转 slot 语句数与常规相同
    truth_rule: str = "recency"   # recency（N−，同任务对照）| rarity（N+，仪器正对照）

def validate_cfg(cfg: CorpusCfg, spec: LangSpec) -> None:
    """配置期校验。宁可生成前炸掉，也不要在数据里留静默失真。"""
    if cfg.truth_rule not in ("recency", "rarity"):
        raise ValueError(f"[{cfg.name}] truth_rule 只能是 recency | rarity")
    if not 0.0 <= cfg.p_break <= 1.0:
        raise ValueError(f"[{cfg.name}] p_break 是比例，须在 [0,1]")
    rb = cfg.r_new_break or cfg.r_old_hi
    if cfg.p_break > 0.0:
        if cfg.k_old_break < 1:
            raise ValueError(f"[{cfg.name}] k_old_break 须 ≥1：老值全删则 N+ "
                             f"的标签不在文档内")
        if rb <= cfg.k_old_break:
            raise ValueError(
                f"[{cfg.name}] 需 r_new_break({rb}) > k_old_break"
                f"({cfg.k_old_break})，否则 rarity 仍指向末代值，别名未破除")
        if cfg.use_marker:
            raise ValueError(
                f"[{cfg.name}] break arm 不支持 use_marker：反转 slot 的 "
                f"is_update 份数与常规 slot 不同（R 对 1，见 _order_ok 的单调性"
                f"要求），UPD token 数会随 is_break 变化，构成完美判别式")
        if cfg.p_hist_query > 0:
            raise ValueError(
                f"[{cfg.name}] break arm 不支持 hist 查询：@k 的真值是老值，"
                f"与 truth_rule=rarity 的标签定义冲突")
        if cfg.max_updates != 1:
            raise ValueError(
                f"[{cfg.name}] break arm 要求 max_updates=1：三代值下"
                f"「反转」有多种含义，rarity 的指向不再唯一")
    elif cfg.truth_rule == "rarity":
        raise ValueError(
            f"[{cfg.name}] p_break=0 时 rarity 与 recency 逐篇同指（式 1），"
            f"truth_rule='rarity' 是静默空操作。N+ 必须配 p_break>0；"
            f"剂量曲线的 p_break=0 锚点用 N−（truth_rule='recency'），"
            f"它与主网格逐比特相同。")

    # 反转 slot 的语句数 = k_old_break + rb。对称默认（k=1, rb=r_old_hi）下
    # 它等于常规 slot 的 r_old_hi+1，长度约束不变；Stage C 的 k_old_break=2
    # 会让它多 1，须由 n_stmts_lo 吸收。
    q_break = (cfg.k_old_break + rb) if cfg.p_break > 0.0 else 0
    worst_q = max(cfg.max_updates * cfg.r_old_hi + 1, q_break)
    need = worst_q + cfg.delta_d_hi + 4
    if cfg.n_stmts_lo < need:
        raise ValueError(
            f"[{cfg.name}] n_stmts_lo={cfg.n_stmts_lo} < {need}。"
            f"下界不足会让 N 被 ΔD/R_old 条件化，长度不再是独立变量。")
    stmt_cost = 5 if cfg.use_marker else 4
    query_cost = 5 + (1 if cfg.p_hist_query > 0 else 0)
    worst_tok = cfg.n_stmts_hi * stmt_cost + query_cost
    if worst_tok > spec.ctx_len:
        raise ValueError(
            f"[{cfg.name}] 最坏 token 数 {worst_tok} > ctx_len {spec.ctx_len}，"
            f"n_stmts_hi 应 ≤ {(spec.ctx_len - query_cost) // stmt_cost}")
    if cfg.delta_d_lo < 1:
        raise ValueError(f"[{cfg.name}] ΔD 必须 ≥1，ΔD=0 是退化情形")
    if cfg.delta_d_hi == cfg.delta_d_lo:
        raise ValueError(
            f"[{cfg.name}] ΔD 在格内不得为常数。ΔD 固定 ⇒ 答案恒在倒数第 "
            f"{1 + cfg.delta_d_lo} 条语句；unmarked 时每条恰 4 token，"
            f"'复制固定 token 偏移处的值'即 100% 正确规则，模型无需 slot 匹配，"
            f"整个网格会读成 no_tracking。用 dd_band() 生成区间。")
    if not 0.0 < cfg.spread <= 1.0:
        raise ValueError(f"[{cfg.name}] spread 是占全文比例，须在 (0,1]")

    # 反转 slot 的 q_old 是 [老值×k_old_break, 末代值×(rb-1)]，共 q_break-1 条，
    # 与常规 slot 同走一个窗口，故窗口宽度检查取两者最坏。
    q_len = max(cfg.max_updates * cfg.r_old_hi, q_break - 1)
    if cfg.spread * cfg.n_stmts_lo < 2 * q_len:
        raise ValueError(
        f"[{cfg.name}] spread×n_stmts_lo={cfg.spread * cfg.n_stmts_lo:.0f} "
        f"< 2×q_old={2 * q_len}。最短文档上 q_old 窗口会退化成紧贴 p_final "
        f"的实心块，q_gap 恒为 1，构成完美判别式。")

def dd_band(d: int, rel: float = 0.5) -> tuple:
    """轴值 d -> ΔD 的均匀支撑区间，均值 ≈ d。

    纯位置规则的上限 = 1/(hi-lo+1)：d=1 -> 0.50，d=2 -> 0.33，
    d=4 -> 0.20，d=8 -> 0.11，d=16 -> 0.06。
    低 ΔD 行的位置上限天然偏高（均值小必然导致质量集中），
    这是不可消除的，须作为格间协变量报告。d=1 行的 0.50 是否可接受，
    是需要你拍的设计决定：删掉该行，或改用 [2,4,8,16]。"""
    lo = max(1, int(round(d * (1 - rel))))
    hi = max(lo + 1, int(round(d * (1 + rel))))
    return lo, hi


# Legacy pilot configurations differ on four controls at once.
# Differences between these arms cannot be assigned to any single control.
# Use these for pipeline diagnostics, not identification of a learned rule.
EXTREME_A = CorpusCfg(
    name="freq_marked",
    p_update=0.9, r_old_lo=1, r_old_hi=2,
    use_marker=True, delta_d_lo=1, delta_d_hi=4,
    p_hist_query=0.0, seed=0,
)

EXTREME_B = CorpusCfg(
    name="rare_unmarked",
    p_update=0.1, r_old_lo=6, r_old_hi=12,
    use_marker=False, delta_d_lo=8, delta_d_hi=32,
    p_hist_query=0.0, seed=0,
)

# ---- 相图网格 ----
# R_old 低端加密：Schuster 报告单次重复即可翻转偏好，
# 相变若在 1→3 之间，均匀网格会完全错过。
R_OLD_GRID = [1, 2, 3, 5, 8, 12]
DELTA_D_GRID = [2, 3, 5, 8, 16]
PHASE_SEEDS = [0, 1, 2]


def phase_configs():
    """30 配置 × 3 种子 = 90 run。全部 p_hist_query=0：
    @k 语法会引入第二个 cue，与 use_marker 旋钮混淆，
    因此不进入相图主体。"""
    for r in R_OLD_GRID:
        for d in DELTA_D_GRID:
            for s in PHASE_SEEDS:
                dlo, dhi = dd_band(d)
                yield CorpusCfg(
                    name=f"R{r}_D{d}_s{s}",
                    seed=s,
                    p_update=0.5,
                    max_updates=1,
                    r_old_lo=r, r_old_hi=r,      # 定值，非区间，保证格子语义唯一
                    use_marker=False,
                    delta_d_lo=dlo, delta_d_hi=dhi,
                    p_hist_query=0.0,
                )


# ---- 旋钮 5 的独立对照 ----
# 从相图选代表格子，各配一个开历史索引的孪生配置。
HIST_CELLS = [(1, 5), (5, 5), (12, 5)]


def hist_configs():
    for r, d in HIST_CELLS:
        for s in PHASE_SEEDS:
            for p in (0.0, 0.5):
                dlo, dhi = dd_band(d)
                yield CorpusCfg(
                    name=f"hist{p}_R{r}_D{d}_s{s}",
                    seed=s,
                    p_update=0.9,
                    max_updates=3,               # @k 需要 ≥3 个不同值才有意义
                    r_old_lo=r, r_old_hi=r,
                    use_marker=False,
                    delta_d_lo=dlo, delta_d_hi=dhi,
                    p_hist_query=p,
                )