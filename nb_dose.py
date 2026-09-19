"""Summarize disagreement-supervision dose arms from terminal caches."""
import argparse
import json
import math
import os
from collections import defaultdict

MASS_FLOOR = 0.50       # 与 go_nogo 同口径
NAN = float("nan")

# 正态样本极差的期望 d_n。range/d_n 是 σ 的估计量（app:power 用它把三 seed
# 的极差与十 seed 的比），但它只在拿不到逐 seed 值时才该用：极差随 n 增长，
# 用错 d_n 会把结论读反。本文件有逐 seed 值，所以一律直算 σ（sd1），
# range/d_n 只作副产物打印出来备核。
D_N = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
       7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}


def sd1(xs):
    """样本标准差，ddof=1 —— 与全文所有跨 run σ 同口径（app:drift 的约定）。"""
    n = len(xs)
    if n < 2:
        return NAN
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def load_rows(paths):
    """读若干 go_nogo 缓存。同 tag 后出现的覆盖先出现的。"""
    by_tag = {}
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"找不到 {p}（go_nogo 的缓存是 <txt>.jsonl）")
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "tag" in o:
                    by_tag[o["tag"]] = o
    return list(by_tag.values())


def filter_sched(rows, want):
    """只留下指定 lr schedule 的 run。剂量曲线必须单一 schedule。

    分组键是 (r_old, dd, truth_rule, p_break)，不含 schedule，所以同一剂量下
    不同 schedule 的 run 会被静默池化。实测踩过这个坑：p_break=0.03 那一档有
    两个 cos run 未逃逸、改 const 后逃逸，于是该档变成 const/cos/const 混合，
    算出极差 0.292 —— 而 §6.1 量过 schedule 效应的极差是 0.55，比 arm 里任何
    一档的极差（0.141–0.293）都大，混进去的数没有意义。

    不在分组键里加第五维而是在入口过滤，理由是曲线的 p_break=0 锚点就是已发表
    的 cos/16000 run，所以整条曲线**必须**是 cos。把 schedule 提到命令行让这
    件事显式，而不是藏在分组逻辑里。

    sched 字段是后加的，旧缓存没有它。缺失一律记为 "?" 并被丢弃（而不是默认
    成 cos）—— 那两个 const run 若被当成 cos 就正好复现了要修的那个 bug。
    丢弃数会打印出来，附带重算指令。
    """
    keep, drop = [], []
    for r in rows:
        s = r.get("sched") or "?"
        (keep if s == want else drop).append((r["tag"], s))
    tags = {t for t, _ in keep}
    return [r for r in rows if r["tag"] in tags], drop


def gate(rows):
    """与 go_nogo.report 同口径：state=retr 且格均值 mass≥0.5。

    剔除的行必须报出来而不是静默丢掉 —— arm 里最危险的失效模式是
    「N+ 没训起来」，它的症状恰好是被这道门挡掉，若不报就会看成缺数据。
    """
    keep, drop = [], []
    for r in rows:
        m = r.get("mass", NAN)
        if r.get("state") != "retr":
            drop.append((r["tag"], f"state={r.get('state')}"))
        elif m == m and m < MASS_FLOOR:
            drop.append((r["tag"], f"mass={m:.2f}"))
        else:
            keep.append(r)
    return keep, drop


def group(rows):
    """(r_old, dd, truth_rule, p_break) -> 按 seed 排序的行。

    p_break / truth_rule 缺失即已发表的主网格行，按 (0.0, recency) 兜底。
    """
    g = defaultdict(list)
    for r in rows:
        key = (r["r_old"], r["dd"],
               r.get("truth_rule") or "recency",
               float(r.get("p_break") or 0.0))
        g[key].append(r)
    for v in g.values():
        v.sort(key=lambda r: r.get("seed", 0))
    return g


def stats(rs, key="frac_positive"):
    """跨 seed 的极差与中位数，外加二项标准误作为噪声地板。

    极差要与 sqrt(0.25/n) 比：frac+ 是 n 篇文档上的符号比例，即便机制完全
    确定，三个 seed 的 frac+ 也会因文档抽样而相差约这个量级。极差收缩到
    地板附近就是「读数已复现」的定量含义。
    """
    # 默认值必须是 NAN 而不是让 .get 返回 None：None == None 为真，缺失的键会
    # 通过这道 NaN 过滤，随后 r[key] 抛 KeyError。已发表的主网格缓存行没有
    # p_rec / p_rar / rar_disc 三个键，而它们正是剂量曲线的 p_break=0 锚点。
    xs = [r[key] for r in rs if r.get(key, NAN) == r.get(key, NAN)]
    if not xs:
        return None
    ns = sorted(r.get("n", 0) for r in rs)
    n_med = ns[len(ns) // 2] if ns else 0
    srt = sorted(xs)
    return dict(n_seed=len(xs), lo=min(xs), hi=max(xs), rng=max(xs) - min(xs),
                med=srt[len(srt) // 2], n_doc=n_med, sd=sd1(xs),
                se=math.sqrt(0.25 / n_med) if n_med else NAN)


def fmt(v, w=6, p=3):
    return f"{'—':>{w}}" if v is None or v != v else f"{v:>{w}.{p}f}"


def report(g, r_old, dd):
    """一格的剂量表。N− 与 N+ 分开列，因为它们的期望方向不同。"""
    L = [f"格 R_old={r_old} ΔD={dd}", ""]
    for tr, name in (("recency", "N−  truth_rule=recency（同任务对照）"),
                     ("rarity", "N+  truth_rule=rarity（仪器正对照）")):
        doses = sorted(k[3] for k in g if k[:3] == (r_old, dd, tr))
        if not doses:
            L += [f"{name}：无数据", ""]
            continue
        L += [name, "",
              f"{'p_brk':>6} {'nSeed':>6} {'frac+ 逐 seed':>26} {'极差':>7} "
              f"{'中位':>7} {'SE':>6} {'medΔ':>8} {'pRec':>6} {'pRar':>6} "
              f"{'rarDsc':>7}"]
        for pb in doses:
            rs = g[(r_old, dd, tr, pb)]
            st = stats(rs)
            if st is None:
                # 该剂量点所有 run 的 frac+ 都是 NaN，即 n=0：break_rarity 在
                # 这些 run 上没有域。gate() 会放过它们（n=0 时 mass 也是 NaN，
                # 不触发 mass<0.5 那条），故必须在这里挡住，否则下面
                # st['n_seed'] 抛 TypeError。
                L.append(f"{pb:>6.2f} {'—':>6}   无有效读数（n=0，"
                         f"break_rarity 无域）")
                continue
            per = " ".join(f"{r['frac_positive']:.3f}" for r in rs
                           if r.get("frac_positive", NAN)
                           == r.get("frac_positive", NAN))
            md = stats(rs, "d_median")
            pr = stats(rs, "p_rec")
            pa = stats(rs, "p_rar")
            rd = stats(rs, "rar_disc")
            L.append(
                f"{pb:>6.2f} {st['n_seed']:>6} {per:>26} "
                f"{fmt(st['rng'], 7)} {fmt(st['med'], 7)} "
                f"{fmt(st['se'], 6)} {fmt(md['med'] if md else None, 8)} "
                f"{fmt(pr['med'] if pr else None, 6)} "
                f"{fmt(pa['med'] if pa else None, 6)} "
                f"{fmt(rd['med'] if rd else None, 7)}")
        L.append("")
    return L


def verdicts(g, r_old, dd):
    """预注册判据。包括「会推翻我们」的那几条 —— 这是自愿承担否证风险。

    刻意不把「N− 的 frac+ 偏离 0.5」单独判成推翻。理由：p_break=0.10 时仍有
    90% 的文档是共延的，目标函数只在 10% 上惩罚 rarity，压力很弱；完全收敛到
    Bayes 规则的模型在 N− 上应有 Δ=0，而残余偏离既可能是「没收敛完」也可能是
    「判据错了」，单点分不开。能分开的是趋势：随 p_break 上升，N− 的极差是否
    单调收窄、中位是否单调趋 0。故这里报趋势，不报单点。
    """
    L = ["预注册判据", ""]
    nm = sorted(k[3] for k in g if k[:3] == (r_old, dd, "recency"))
    np_ = sorted(k[3] for k in g if k[:3] == (r_old, dd, "rarity"))

    # 1. N− 的极差是否随剂量收窄。
    #
    # 只有 ≥2 个种子的剂量点能进单调性检查：单种子的极差按定义是 0.000，
    # 混进来会让序列变成 0.000 -> 0.293 -> 0.000 -> 0.196 -> 0.141 而误报
    # 「不单调」。实测就踩过这个坑（p_break=0.03 有两个 run 未逃逸被门剔除，
    # 只剩 1 个种子）。
    #
    # 比较基准不是二项地板。frac+ 的二项 SE 假设逐篇 Δ 是独立 Bernoulli，
    # 但 Δ 分布集中在 0 附近时，符号比例对分布的微小不对称极敏感，而那个
    # 不对称本身跨种子变化。论文 app:drift 已经量过正确的地板：同一个 run
    # 在逃逸后不同 checkpoint 之间的 frac+ 波动，sd 落在 0.006–0.303、
    # 中位 0.069。三个样本的期望极差是 1.693σ，故把极差换算成 σ 再与该带
    # 比较，才是「跨种子差异是否已经和同一个 run 自己的时间波动分不开」。
    # app:drift 的 within-run sd 带，ddof=1（旧值 0.069/0.303 是 ddof=0）。
    DRIFT_MED, DRIFT_HI = 0.074, 0.332
    usable = [(pb, s) for pb, s in
              ((pb, stats(g[(r_old, dd, "recency", pb)])) for pb in nm)
              if s and s["n_seed"] >= 2]
    if len(usable) >= 2:
        # σ 一律直算。range/d_n 用该点自己的 n（不是固定 d_3）并列出来备核：
        # 两者差得远就说明该剂量点的分布不接近正态，或有离群 seed。
        rngs = [(pb, s["rng"], s["sd"], s["rng"] / D_N.get(s["n_seed"], NAN),
                 s["n_seed"], s["se"]) for pb, s in usable]
        first, last = rngs[0], rngs[-1]
        mono = all(a[1] >= b[1] - 1e-9 for a, b in zip(rngs, rngs[1:]))
        L.append(f"[1] N− 极差（仅 ≥2 种子的剂量点，共 {len(rngs)} 个）")
        for pb, rg, sd, sg, ns, se in rngs:
            tag = ("带内" if sd <= DRIFT_HI else "带外")
            L.append(f"      p_break={pb:.2f}  n={ns:<2d} 极差 {rg:.3f}  "
                     f"σ={sd:.3f}（直算）/{sg:.3f}（极差/d_{ns}）  "
                     f"({tag}, within-run 中位 {DRIFT_MED:.3f} "
                     f"上界 {DRIFT_HI:.3f})")
        L.append(f"    单调收窄：{'是' if mono else '否'}")
        if mono and last[2] <= DRIFT_HI:
            L.append("    => 极差单调收窄，且末点的跨种子变异已落入同一个 run "
                     "自己的时间波动带内，与 within-run 漂移分不开。")
            L.append("       这比「收缩到二项地板」弱，但它是正确的陈述，且接上 "
                     "§5.6 的方差排序：种子那一项掉进了 within-run 那一档。")
        elif mono:
            L.append("    => 单调收窄但末点仍在 within-run 带外，剂量不足或"
                     "还有别的方差源。")
        elif last[1] > 0.3:
            L.append("    => 极差未收缩。若 N+ 也不收缩，见判据 [3]："
                     "this comparison alone does not separate noise, supervision response and structural confounds.")
    else:
        L.append(f"[1] 有 ≥2 种子的剂量点只有 {len(usable)} 个，无法判趋势。"
                 f"注意单种子点的极差是 0.000（定义所致），不可混入。")

    # 1b. 中心是否也随剂量移动。极差收缩说明「机制被定下来了」，中心移动说明
    #     「被定到了目标函数指的方向」。两者是独立的证据。
    #
    # 只在极差已落入 within-run 带的剂量点上做。目标函数无差别的那一档根本
    # 不存在「中心」这个量：p_break=0 的三个种子是 0.098/0.477/0.977，几乎
    # 均匀铺满 [0,1]，取中间那个得到 0.477 纯属偶然，它是散点的中位数而不是
    # 分布的中心。把它算进单调性检查会得到假阴性（实测就踩过：
    # 0.477@0.00 -> 0.498@0.01 报「不单调」，而 0.01 往后是严格单调的）。
    cen = [(pb, s["med"]) for pb, s in usable
           if s["rng"] / 1.693 <= DRIFT_HI]
    skipped = [pb for pb, s in usable if s["rng"] / 1.693 > DRIFT_HI]
    if len(cen) >= 2:
        mono_c = all(a[1] >= b[1] - 1e-9 for a, b in zip(cen, cen[1:]))
        L.append(f"[1b] frac+ 中位（仅极差已进 within-run 带的剂量点）："
                 + " -> ".join(f"{v:.3f}@{pb:.2f}" for pb, v in cen)
                 + f"　单调下降：{'是' if mono_c else '否'}")
        if skipped:
            L.append(f"     跳过 p_break=" + ",".join(f"{p:.2f}" for p in skipped)
                     + "：极差在带外，散点的中位数不是分布的中心，"
                     "目标函数在那里无差别、不存在「中心」这个量。")
        if mono_c:
            L.append("     => 中心也随剂量移动，方向与目标函数一致。这是与极差"
                     "收缩独立的第二个证据。")
            L.append("     注意混淆：p_break 越高，break_rarity 的编辑输出越接近"
                     "训练分布内的配置（反转文档就长那样，且在 N− 里标签是末代"
                     "值），会把 Δ 往负方向推。极差收缩不受影响（对三个种子等量"
                     "作用），但中心移动的解释二义。N+ 对此免疫（两条 arm 的编辑"
                     "分布逐比特相同），故它是这一条的对照。")
    else:
        L.append("[1b] 极差已进 within-run 带的剂量点不足 2 个，中心趋势不判。")

    # 2. N+ 是否给出可复现的强正信号（仪器正对照）
    if np_ and stats(g[(r_old, dd, "rarity", np_[-1])]) is None:
        L.append(f"[2] N+ (p_break={np_[-1]:.2f}) 无有效读数（n=0）："
                 f"仪器正对照缺失，N− 的 frac+ 无法与「探针已死」区分")
    elif np_:
        s = stats(g[(r_old, dd, "rarity", np_[-1])])
        pa = stats(g[(r_old, dd, "rarity", np_[-1])], "p_rar")
        L.append(f"[2] N+ (p_break={np_[-1]:.2f}) frac+ 中位 {fmt(s['med'],0)}"
                 f" 极差 {fmt(s['rng'],0)}；行为 pRar 中位 "
                 f"{fmt(pa['med'] if pa else None, 0)}")
        if pa and pa["med"] == pa["med"] and pa["med"] < 0.5:
            L.append("    => pRar 起不来：arm 没训起来（加预算或加 p_break），"
                     "这不是关于共延性的结果")
        elif s["med"] > 0.9 and s["rng"] < 0.1:
            L.append("    => 仪器能给出可复现的强正信号")
    else:
        L.append("[2] 无 N+ 数据，仪器正对照缺失 —— N− 的 frac+≈0.5 无法与"
                 "「探针已死」区分")

    # 3. 因果读数与行为标签是否一致。不一致 => 探针无效，全文结论作废
    bad = []
    for k, rs in g.items():
        if k[:2] != (r_old, dd) or k[3] <= 0.0:
            continue
        for r in rs:
            pr, pa = r.get("p_rec", NAN), r.get("p_rar", NAN)
            fp = r.get("frac_positive", NAN)
            if pa != pa or fp != fp:
                continue
            # N+ 应同时有 pRar 高与 frac+ 高；两者相悖即探针与行为脱节
            if k[2] == "rarity" and pa > 0.8 and fp < 0.5:
                bad.append(f"{r['tag']}: pRar={pa:.2f} 但 frac+={fp:.2f}")
            if k[2] == "recency" and pr == pr and pr > 0.8 and fp > 0.9:
                bad.append(f"{r['tag']}: pRec={pr:.2f} 但 frac+={fp:.2f}")
    L.append(f"[3] 因果读数与行为标签冲突的 run：{len(bad)}")
    for b in bad:
        L.append(f"      {b}")
    if bad:
        L.append("    => 探针与行为脱节，全文的因果读数结论作废。这是本臂"
                 "最强的否证入口，必须先查清再解释任何别的数字。")

    # 4. 旁路。反转文档在两处结构上与非反转文档不同（covar 实测）：
    #      antRbk  slot 内相邻间距比，R3=0.950 / R5=0.839
    #      keptBk  q slot 语句数恒为 R+1，非反转文档只有约 0.6R+1
    #    根因是重试循环只作用于被查询 slot，且 v_old 恰在 q_old[0] 拿最小位置，
    #    故「v_old 存活」等价于「q_old 全存活」，_order_ok 强制它不可绕开。
    #    危害：目标函数的压力施加在实际冗余度=R 的文档上，读数取自实际冗余度
    #    ≈0.6R 的文档，而 §5.4 已证效应随实际冗余度放大。模型可以学
    #    「R+1 条全在 → recency；缺了几条 → rarity」，满足目标函数而完全不改变
    #    非反转文档上的行为 —— 那样 N− 极差不收缩，会被判据 [1] 误读成
    #    「读数是噪声」。这种归因忽略了结构混淆：目标
    #    函数对反转文档的约束可能被捷径满足，未传递到常规文档。
    sw = [(r["tag"], r.get("sw_keep", NAN), r.get("sw_rar", NAN),
           r.get("sw_n", 0)) for k, rs in g.items() if k[3] > 0.0
          and k[:2] == (r_old, dd) for r in rs]
    sw = [x for x in sw if x[1] == x[1]]
    if not sw:
        L.append("[4] 无旁路检查数据（sw_keep 缺失）：该 arm 的 checkpoint 早于"
                 "break_diag 的 _swap_query，重训或忽略")
    else:
        worst = max(sw, key=lambda x: x[1])
        L.append(f"[4] 旁路检查 swKeep 最大 {worst[1]:.3f}（{worst[0]}，"
                 f"n={worst[3]}）")
        if worst[1] > 0.20:
            L.append("    => 模型在反转文档上没读 query，靠结构认出被查询 slot。"
                     "判据 [1] 的极差不收缩不可解释为「读数是噪声」，两者混淆。"
                     "须先消除结构差异（换更小的 R_old，或让填充侧也走重试循环）"
                     "再重跑本臂。")
        else:
            L.append("    => 模型读了 query，antRbk 与 keptBk 的结构差异未被利用。"
                     "this diagnostic alone does not causally identify the source of a range change.")
    return L


def plot(g, r_old, dd, path):
    """逐 seed 折线 + 跨 seed 极差阴影带。matplotlib 缺失时跳过，不算失败。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (装 matplotlib 出图；表格已写好，数字不受影响)")
        return False

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for tr, col, lab in (("recency", "tab:blue", "N$-$ (recency)"),
                         ("rarity", "tab:red", "N$+$ (rarity)")):
        doses = sorted(k[3] for k in g if k[:3] == (r_old, dd, tr))
        if not doses:
            continue
        xs, los, his = [], [], []
        thin = []                    # 只有 1 个种子的剂量点：画散点不画带
        per_seed = defaultdict(list)
        for pb in doses:
            rs = g[(r_old, dd, tr, pb)]
            s = stats(rs)
            if not s:
                continue
            # 散点全画，带只画 ≥2 种子的点。两者分开处理的理由：
            #
            # 带：lo==hi 时 fill_between 收缩成一条线，看起来像「带宽在该处
            # 归零又反弹」—— 非单调，而那纯粹是 n_seed=1 的伪影。
            #
            # 散点：不能跟着跳过。早先的版本用一个 continue 同时跳过两者，
            # 结果 p_break=0 的锚点在只有 seed 0 时整个从图里消失 —— 而它是
            # 「从 0.879 塌下来」那一段的起点，是整张图要讲的事。数据点是
            # 真实测量，不该因为兄弟种子缺席就隐形。
            for r in rs:
                if r.get("frac_positive") == r.get("frac_positive"):
                    per_seed[r.get("seed", 0)].append((pb, r["frac_positive"]))
            if s["n_seed"] < 2:
                thin.append(pb)
                continue
            xs.append(pb)
            los.append(s["lo"])
            his.append(s["hi"])
        if not per_seed:
            continue
        # 只护住带，不护住散点。带需要 ≥2 个剂量点才有面积（单点 fill_between
        # 什么都不画，无害）；散点只要有数据就该出现。早先这里是 `if not xs`，
        # 于是一条 arm 若全是单种子剂量点就整条消失 —— N+ 只有一档，任何时候
        # 它只跑了一个种子就会中招。
        if len(xs) >= 2:
            ax.fill_between(xs, los, his, color=col, alpha=0.18, lw=0)
        if thin:
            print(f"  {lab}: 剂量 {thin} 只有 1 个种子，画散点不画带")
        for sd, pts in sorted(per_seed.items()):
            pts.sort()
            ax.plot([p for p, _ in pts], [v for _, v in pts], "o-", color=col,
                    ms=3.5, lw=1.0, alpha=0.85,
                    label=lab if sd == min(per_seed) else None)
    ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel(r"$p_{\mathrm{break}}$  (fraction of non-aliased documents)")
    ax.set_ylabel(r"fraction of documents with $\Delta > 0$")
    ax.set_title(f"$R_{{\\mathrm{{old}}}}={r_old}$, $\\Delta D={dd}$",
                 fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+",
                    help="go_nogo 的缓存 jsonl，如 runs_g2/go_nogo.txt.jsonl "
                         "runs_nb/go_nogo.txt.jsonl")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--sched", default="cos",
                    help="只取该 lr schedule 的 run。曲线必须单一 schedule："
                         "p_break=0 的锚点是已发表的 cos/16000 run，故默认 cos")
    ap.add_argument("--txt", default="runs_nb/dose.txt")
    ap.add_argument("--fig", default="runs_nb/dose.pdf")
    a = ap.parse_args()

    rows = load_rows(a.caches)
    rows, sdrop = filter_sched(rows, a.sched)
    if sdrop:
        print(f"schedule 过滤（只留 {a.sched}）：丢弃 {len(sdrop)} 行")
        for t, s in sorted(sdrop):
            print(f"    {t}  sched={s}")
        if any(s == "?" for _, s in sdrop):
            print("    sched=? 是旧缓存（该字段后加）。重算：")
            print("    python go_nogo.py --pattern '<同一 pattern>' "
                  "--out <同一 out> --force")
    keep, drop = gate(rows)
    g = group(keep)

    L = [f"别名破除臂的剂量反应（{len(rows)} 行读入，{len(keep)} 行过门）", ""]
    L += report(g, a.r, a.d)
    L += verdicts(g, a.r, a.d)
    if drop:
        L += ["", f"未过门（state≠retr 或 mass<{MASS_FLOOR}）：{len(drop)}"]
        L += [f"    {t}  {why}" for t, why in sorted(drop)]
        L += ["注意：N+ 没训起来的症状恰好是被这道门挡掉。若被剔除的都是 N+，",
              "先查 pRar 与 accBrk，不要当成缺数据。"]
    txt = "\n".join(L)
    print(txt)
    os.makedirs(os.path.dirname(a.txt) or ".", exist_ok=True)
    with open(a.txt, "w", encoding='utf-8') as f:
        f.write(txt + "\n")
    print(f"\n已写入 {a.txt}")
    if plot(g, a.r, a.d, a.fig):
        print(f"已写入 {a.fig}")


if __name__ == "__main__":
    main()
