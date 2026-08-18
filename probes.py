"""规则归因探针套件 v2。

与 v1 的差异，均为已确认缺陷的修正：

1. _filler 增加 banned_vals。v1 从全池随机采干扰值，每条约 0.15% 撞上候选值，
   累计十余条约 2%，会污染 P2/P3 的概率读数。
2. P3 改为三候选且频次配平。v1 中 v_old 出现两次（早期赋值 + 末尾干扰），
   频次主导的模型会在 P3 上得高分，使 r_pos 无法与 r_freq 区分。
   现在 v_new / v_old / v_third 各出现恰好一次。
3. P4 改为「结构先采样、再双份渲染」，去掉 v1 里脆弱的 RNG 状态拷贝。
   带标记与不带标记两份现在共用逐字节相同的底层样本。
4. 新增 score_suite / bootstrap_ci，实现规格文档第 4 节定义的四维向量。

已知且刻意保留的局限：P2 中新值紧邻查询，因此 P2 不区分「近期规则」与
「抄最后一个值 token」。这个区分由 P3 承担。写论文时必须明说。

冻结纪律：本文件与生成的 jsonl 在第一个训练 run 之前 commit，之后不改。
"""
import hashlib
import json
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional

from config import LangSpec
from vocab import Vocab

N_PER_PROBE = 600      # 旧稿每格 N=25，SE≈10%，是教训三的直接来源
FREQ_REPS = 20         # P2 中旧值的重复次数
SUITE_SEED = 20260812

# 语句四元组：(ent, attr, val, is_update)
Stmt = Tuple[int, int, int, bool]


@dataclass
class ProbeItem:
    item_id: str
    probe: str
    prompt: List[int]                      # 到 ARROW 为止，不含答案
    candidates: Dict[str, int]             # 角色 -> value token
    expected_by_rule: Dict[str, str]       # 规则名 -> 该规则预测的角色
    meta: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------- 采样与渲染

def _sample_filler(vocab: Vocab, rng: random.Random, n: int,
                   banned_ents: set, banned_vals: set) -> List[Stmt]:
    """干扰语句。实体与值都避开候选，防止污染候选概率。"""
    out: List[Stmt] = []
    for _ in range(n):
        while True:
            e = rng.randrange(vocab.spec.n_entities)
            if e not in banned_ents:
                break
        while True:
            v = rng.randrange(vocab.spec.n_values)
            if v not in banned_vals:
                break
        a = rng.randrange(vocab.spec.n_attrs)
        out.append((e, a, v, False))
    return out


def _emit(vocab: Vocab, stmts: List[Stmt], marked: bool) -> List[int]:
    toks: List[int] = []
    for (e, a, v, is_upd) in stmts:
        if marked and is_upd:
            toks.append(vocab.UPD)
        toks += [vocab.ent(e), vocab.attr(a), vocab.val(v), vocab.SEP]
    return toks


def _emit_query(vocab: Vocab, e: int, a: int,
                hist_k: Optional[int] = None) -> List[int]:
    q = [vocab.QUERY, vocab.ent(e), vocab.attr(a)]
    if hist_k is not None:
        q.append(vocab.time(hist_k))
    q.append(vocab.ARROW)
    return q


# ---------------------------------------------------------------- P0 检索门

def probe_P0(vocab: Vocab, rng: random.Random, n=N_PER_PROBE,
             marked=False) -> List[ProbeItem]:
    """无冲突对照：单次赋值。测的是「能否从上下文取值」。
    这是门，不是规则维度。P0 不过 => 工程返工（回阶段 0 简化语言），
    不是方向死亡。distractor 从不在上下文出现，用于检验模型是否
    偏好上下文内 token。"""
    items = []
    for i in range(n):
        e = rng.randrange(vocab.spec.n_entities)
        a = rng.randrange(vocab.spec.n_attrs)
        v, v_absent = rng.sample(range(vocab.spec.n_values), 2)
        banned_v = {v, v_absent}
        stmts = _sample_filler(vocab, rng, rng.randint(4, 12), {e}, banned_v)
        stmts.append((e, a, v, False))
        stmts += _sample_filler(vocab, rng, rng.randint(1, 8), {e}, banned_v)
        toks = _emit(vocab, stmts, marked) + _emit_query(vocab, e, a)
        items.append(ProbeItem(
            item_id=f"P0_{i}", probe="P0", prompt=toks,
            candidates={"target": vocab.val(v), "absent": vocab.val(v_absent)},
            expected_by_rule={"retrieval": "target"},
            meta={"n_stmts": len(stmts)}))
    return items


# ---------------------------------------------------------------- P1 / P4

def _sample_p1_struct(vocab: Vocab, rng: random.Random) -> dict:
    """P1 结构。与渲染分离，使 P4 能对同一批样本渲染两份。"""
    e = rng.randrange(vocab.spec.n_entities)
    a = rng.randrange(vocab.spec.n_attrs)
    v_old, v_new = rng.sample(range(vocab.spec.n_values), 2)
    banned_v = {v_old, v_new}
    pre = _sample_filler(vocab, rng, rng.randint(3, 10), {e}, banned_v)
    mid = _sample_filler(vocab, rng, rng.randint(2, 8), {e}, banned_v)
    post = _sample_filler(vocab, rng, rng.randint(1, 6), {e}, banned_v)
    stmts = pre + [(e, a, v_old, False)] + mid + [(e, a, v_new, True)] + post
    return {"e": e, "a": a, "v_old": v_old, "v_new": v_new, "stmts": stmts}


def _render_p1(vocab: Vocab, st: dict, marked: bool,
               probe_name: str, item_id: str) -> ProbeItem:
    toks = _emit(vocab, st["stmts"], marked) \
        + _emit_query(vocab, st["e"], st["a"])
    return ProbeItem(
        item_id=item_id, probe=probe_name, prompt=toks,
        candidates={"new": vocab.val(st["v_new"]),
                    "old": vocab.val(st["v_old"])},
        expected_by_rule={"recency": "new", "frequency": "new"},
        meta={"n_stmts": len(st["stmts"]), "marked": int(marked)})


def probe_P1(vocab: Vocab, rng: random.Random, n=N_PER_PROBE,
             marked=False) -> List[ProbeItem]:
    """基本更新追踪：旧值一次在前、新值一次在后。频次为 1:1，
    因此近期与频次两条规则在此不冲突，都预测 new。
    P1 同时是 P2 的频次配平对照。"""
    return [_render_p1(vocab, _sample_p1_struct(vocab, rng), marked,
                       "P1", f"P1_{i}") for i in range(n)]


def probe_P4(vocab: Vocab, rng: random.Random, n=N_PER_PROBE
             ) -> List[ProbeItem]:
    """标记依赖度。同一批结构渲染两份：带 upd 与不带 upd。
    仅对 use_marker=True 训练出的模型有意义；unmarked 模型上
    r_cue 应标 N/A，因为它从未见过 upd token，读数是未训练
    token 的噪声，不是「低依赖」。"""
    structs = [_sample_p1_struct(vocab, rng) for _ in range(n)]
    out = []
    for i, st in enumerate(structs):
        out.append(_render_p1(vocab, st, True, "P4_marked", f"P4m_{i}"))
        out.append(_render_p1(vocab, st, False, "P4_unmarked", f"P4u_{i}"))
    return out


# ---------------------------------------------------------------- P2 频次

def probe_P2(vocab: Vocab, rng: random.Random, n=N_PER_PROBE,
             marked=False) -> List[ProbeItem]:
    """频次 vs 近期。旧值重复 FREQ_REPS 次，新值一次且紧邻查询。
    近期规则 -> new；频次规则 -> old。
    r_freq = P(old) / (P(old) + P(new))，与规格文档第 4 节一致。"""
    items = []
    for i in range(n):
        e = rng.randrange(vocab.spec.n_entities)
        a = rng.randrange(vocab.spec.n_attrs)
        v_old, v_new = rng.sample(range(vocab.spec.n_values), 2)
        banned_v = {v_old, v_new}
        stmts = _sample_filler(vocab, rng, rng.randint(2, 6), {e}, banned_v)
        for k in range(FREQ_REPS):
            stmts.append((e, a, v_old, False))
            if k % 4 == 3:      # 打散连续重复，避免退化成单个长 n-gram
                stmts += _sample_filler(vocab, rng, 1, {e}, banned_v)
        stmts.append((e, a, v_new, True))
        toks = _emit(vocab, stmts, marked) + _emit_query(vocab, e, a)
        items.append(ProbeItem(
            item_id=f"P2_{i}", probe="P2", prompt=toks,
            candidates={"new": vocab.val(v_new), "old": vocab.val(v_old)},
            expected_by_rule={"recency": "new", "frequency": "old"},
            meta={"n_stmts": len(stmts), "reps": FREQ_REPS}))
    return items


# ---------------------------------------------------------------- P3 位置

def probe_P3(vocab: Vocab, rng: random.Random, n=N_PER_PROBE,
             marked=False) -> List[ProbeItem]:
    """绝对位置 vs 逻辑追踪，频次严格配平。

    三个候选各出现恰好一次：
      v_old   -> e.a 的早期赋值        （早期位置）
      v_new   -> e.a 的最新赋值        （逻辑正确）
      v_third -> 末尾另一实体同属性的值（末尾位置）

    末尾干扰刻意用 a2 == a、e2 != e：同时检验位置捷径与
    「按属性类型抄」捷径，正是防捷径表要求的构造。

    r_pos = P(third) / (P(new) + P(old) + P(third))。
    v1 里末尾干扰复用 v_old，导致 v_old 出现两次、频次主导模型
    在 P3 上得高分，r_pos 与 r_freq 混淆，故修正。"""
    items = []
    for i in range(n):
        e, e2 = rng.sample(range(vocab.spec.n_entities), 2)
        a = rng.randrange(vocab.spec.n_attrs)
        v_old, v_new, v_third = rng.sample(range(vocab.spec.n_values), 3)
        banned_v = {v_old, v_new, v_third}
        banned_e = {e, e2}
        stmts = _sample_filler(vocab, rng, rng.randint(3, 8), banned_e, banned_v)
        stmts.append((e, a, v_old, False))
        stmts += _sample_filler(vocab, rng, rng.randint(2, 6), banned_e, banned_v)
        stmts.append((e, a, v_new, True))
        stmts += _sample_filler(vocab, rng, rng.randint(1, 3), banned_e, banned_v)
        stmts.append((e2, a, v_third, False))     # 同属性、不同实体、位于末尾
        toks = _emit(vocab, stmts, marked) + _emit_query(vocab, e, a)
        items.append(ProbeItem(
            item_id=f"P3_{i}", probe="P3", prompt=toks,
            candidates={"new": vocab.val(v_new), "old": vocab.val(v_old),
                        "third": vocab.val(v_third)},
            expected_by_rule={"recency": "new", "position_last": "third",
                              "position_first": "old"},
            meta={"n_stmts": len(stmts)}))
    return items


# ---------------------------------------------------------------- 套件

def build_suite(spec: LangSpec, seed: int = SUITE_SEED,
                marked: bool = False, include_p4: bool = True
                ) -> List[ProbeItem]:
    """marked 必须与被测模型的 use_marker 一致，否则 P0-P3 的
    upd token 分布与训练分布不符，读数不可比。"""
    vocab = Vocab(spec)
    rng = random.Random(seed)
    items: List[ProbeItem] = []
    items += probe_P0(vocab, rng, marked=marked)
    items += probe_P1(vocab, rng, marked=marked)
    items += probe_P2(vocab, rng, marked=marked)
    items += probe_P3(vocab, rng, marked=marked)
    if include_p4 and marked:
        items += probe_P4(vocab, rng)
    return items


def freeze(items: List[ProbeItem], path="probe_suite.jsonl") -> str:
    """写盘并返回内容哈希。哈希与 git commit 一并记进论文。"""
    h = hashlib.sha256()
    with open(path, "w") as f:
        for it in items:
            line = json.dumps(asdict(it), sort_keys=True)
            h.update(line.encode())
            f.write(line + "\n")
    return h.hexdigest()


def load_suite(path="probe_suite.jsonl") -> List[ProbeItem]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(ProbeItem(**json.loads(line)))
    return out


# ---------------------------------------------------------------- 打分

def _norm(probs: Dict[str, float], roles: List[str]) -> Dict[str, float]:
    z = sum(probs[r] for r in roles)
    if z <= 0:
        return {r: float("nan") for r in roles}
    return {r: probs[r] / z for r in roles}


def bootstrap_ci(vals: List[float], n_boot=10000, alpha=0.05,
                 rng: Optional[random.Random] = None):
    """百分位 bootstrap。Schuster 用 n=10000 + Holm-Bonferroni，
    多重比较校正在跨配置检验时另做。"""
    rng = rng or random.Random(0)
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    n = len(vals)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return sum(vals) / n, lo, hi


def score_suite(items: List[ProbeItem],
                probs: Dict[str, Dict[str, float]],
                n_boot=10000) -> dict:
    """probs[item_id][role] = 该 role 候选 token 的 next-token 概率
    （未归一化即可，函数内部按候选集归一化）。

    返回四维规则向量 + P0 门 + 每项的 bootstrap 区间。
    输出的是剖面，不是 accuracy。"""
    by_probe: Dict[str, List[ProbeItem]] = {}
    for it in items:
        by_probe.setdefault(it.probe, []).append(it)

    per_item: Dict[str, List[float]] = {}

    for it in by_probe.get("P0", []):
        p = _norm(probs[it.item_id], ["target", "absent"])
        per_item.setdefault("r_retrieval", []).append(p["target"])

    for it in by_probe.get("P1", []):
        p = _norm(probs[it.item_id], ["new", "old"])
        per_item.setdefault("r_base", []).append(p["new"])

    for it in by_probe.get("P2", []):
        p = _norm(probs[it.item_id], ["new", "old"])
        per_item.setdefault("r_freq", []).append(p["old"])

    for it in by_probe.get("P3", []):
        p = _norm(probs[it.item_id], ["new", "old", "third"])
        per_item.setdefault("r_pos", []).append(p["third"])
        per_item.setdefault("r_pos_first", []).append(p["old"])
        per_item.setdefault("r_track", []).append(p["new"])

    # P4：逐样本配对差值，不是两组均值相减
    marked = {it.item_id[4:]: it for it in by_probe.get("P4_marked", [])}
    unmarked = {it.item_id[4:]: it for it in by_probe.get("P4_unmarked", [])}
    for k in sorted(set(marked) & set(unmarked)):
        pm = _norm(probs[marked[k].item_id], ["new", "old"])["new"]
        pu = _norm(probs[unmarked[k].item_id], ["new", "old"])["new"]
        per_item.setdefault("r_cue", []).append(pm - pu)

    out = {}
    rng = random.Random(12345)
    for key, vals in per_item.items():
        m, lo, hi = bootstrap_ci(vals, n_boot=n_boot, rng=rng)
        out[key] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(vals)}
    return out


def discretize(profile: dict, gate=0.60, hi=0.70, lo=0.30) -> str:
    """相图格子标签。阈值是占位，阶段 1 后按实际分布重定，
    重定之后写进论文并锁死。"""
    if profile.get("r_retrieval", {}).get("mean", 0) < gate:
        return "no_retrieval"
    if profile.get("r_base", {}).get("mean", 0) < gate:
        return "no_tracking"
    f = profile.get("r_freq", {}).get("mean", float("nan"))
    p = profile.get("r_pos", {}).get("mean", float("nan"))
    if p > hi:
        return "position_dominated"
    if f > hi:
        return "frequency_dominated"
    if f < lo:
        return "recency_dominated"
    return "mixed"