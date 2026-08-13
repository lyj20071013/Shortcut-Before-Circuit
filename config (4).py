"""受控合成语言的配置。所有随机性由 seed 控制。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LangSpec:
    """语言规模。跨所有配置固定，不作为旋钮。"""
    n_entities: int = 2000
    n_attrs: int = 8
    n_values: int = 2000
    n_time_idx: int = 4        # @0..@3，仅在 hist query 开启时使用
    ctx_len: int = 512

    @property
    def n_bindings(self) -> int:
        """可能的 (e,a,v) 三元组数。远超模型容量是设计要求。"""
        return self.n_entities * self.n_attrs * self.n_values


@dataclass(frozen=True)
class CorpusCfg:
    """五个数据侧旋钮 + 文档形状随机化参数。"""
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
    n_stmts_lo: int = 80
    n_stmts_hi: int = 140
    min_slots: int = 4          # 文档内最少不同 slot 数，保证实体多样性
    spread: float = 0.80     # slot 局部窗口宽度 = 语句数 × spread

def validate_cfg(cfg: CorpusCfg, spec: LangSpec) -> None:
    """配置期校验。宁可生成前炸掉，也不要在数据里留静默失真。"""
    worst_q = cfg.max_updates * cfg.r_old_hi + 1
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

    q_len = cfg.max_updates * cfg.r_old_hi
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


# 阶段 1 的两个极端配置。生死门只跑这两个。
# 注意：两者在 4 个旋钮上同时不同，arm 间差异不可归因到任何单一旋钮。
# 这里只用于验证"两端确实学到不同规则"，归因靠相图。
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