"""Launch or list the 75-run main grid."""
import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from typing import Optional

from config import CorpusCfg, LangSpec, dd_band, validate_cfg
from go_nogo import sign_test_p

R_OLDS = [3, 5, 8, 12, 16]
DDS = [2, 3, 5, 8, 16]
SEEDS = [0, 1, 2]
HEAD = [(16, 5), (16, 2), (16, 16), (3, 2), (3, 16), (5, 5)]   # 优先跑

FIXED = dict(steps=16000, n_values=512, n_entities=200,
             stmts_lo=45, stmts_hi=55, batch=256, lr=1e-3,
             sched="cos", wd=0.1, eval_every=1000, eval_docs=4000,
             workers=4)
TAG = "grid"
OUT_DEFAULT = "runs_g2"
YIELD_FLOOR = 0.05          # break_rarity 域低于此值：该格无 DV
MASS_FLOOR = 0.50           # 概率质量低于此值：OOD 探针失效，读数无意义
ACC_FLOOR = 0.99            # retrieval 态的 acc 门
COPY_FLOOR = 0.95           # retrieval 态的 copy_acc 门：检索回路是否建成
ESC_FLOOR = 0.95            # 逃逸步数 = copy_acc 首次达到此值的 step


def cells():
    """(r, d, seed) 列表。seed-major，HEAD 优先。"""
    grid = list(itertools.product(R_OLDS, DDS))
    rest = [c for c in grid if c not in HEAD]
    return [(r, d, s) for s in SEEDS for r, d in HEAD + rest]


def mk_cfg(r, d, seed):
    lo, hi = dd_band(d)
    return CorpusCfg(name=f"R{r}_D{d}", seed=seed, p_update=0.5,
                     max_updates=1, r_old_lo=r, r_old_hi=r,
                     use_marker=False, delta_d_lo=lo, delta_d_hi=hi,
                     p_hist_query=0.0,
                     n_stmts_lo=FIXED["stmts_lo"],
                     n_stmts_hi=FIXED["stmts_hi"])


def mk_spec():
    return LangSpec(n_values=FIXED["n_values"],
                    n_entities=FIXED["n_entities"])


def pos_ceil(d):
    """纯位置规则的正确率上限。unmarked 时每条语句恰 4 token、答案恒在倒数第
    1+ΔD 条，故「复制固定 token 偏移处的值」的命中率 = 1/|支撑集|。"""
    lo, hi = dd_band(d)
    return 1.0 / (hi - lo + 1)


def classify(acc, copy_acc, d, ceil=None, chance=None):
    """retr / posNN% / none。posNN% 是 acc 占 posCeil 的比例，≥90% 即可判定
    模型在用位置规则；此时 Δ 不可用。

    ceil=None 时用 pos_ceil(d)，即 1/|supp| —— 主网格的解析上限，成立依赖
    每条语句恰 4 token。已发表的 75 个 run 走这条路，行为逐比特不变。

    ceil 给了就用它。自然语言表层臂（nl_generator）句长可变，1/|supp| 不再
    是上限：变长语句让固定 token 偏移更难命中，实测上限低于解析值。那个数
    由 nl_collide.py 测出，写在 nl_collide.jsonl 的 pos_emp 字段。

    chance 是读数位置上的随机基线（NL 臂 = 1/|ADJ|）。给了就要求
    posCeil > chance 才承认位置态，理由见函数体内注释。主网格不给，
    因为它的 chance = 1/n_values 远低于 posCeil，这条从不触发。

    两个参数存在的理由都是两个臂必须共用一个分类器。分类器决定哪些 run 进
    网格（state=retr 才可用），若两臂各用一个，读数就不可并列 —— 而并列是
    NL 臂的唯一目的。
    """
    if acc != acc or copy_acc != copy_acc:
        return "?"
    if copy_acc >= COPY_FLOOR and acc >= ACC_FLOOR:
        return "retr"
    pc = pos_ceil(d) if ceil is None else ceil
    if pc <= 0 or pc != pc:
        return "none"
    # 位置态只在捷径优于随机时有意义。posCeil <= chance 的格子里，
    # "复制固定偏移处的值"比瞎猜还差，模型没有动机去学它，而
    # acc >= 0.5*posCeil 这个判据会把任何随机水平的模型判成 pos 并读出
    # posNNNN%。主网格的 chance 是 1/n_values≈0.002、posCeil 是它的
    # 30-170 倍，所以那里这条从不触发；NL 臂读数在 24 个 adj 上会触发。
    if chance is not None and pc <= chance:
        return "none"
    if copy_acc < 0.5 and acc >= 0.5 * pc:
        return f"pos{acc / pc:.0%}"
    return "none"


# ---------------- 预检 ----------------

def check(out_dir, n_probe: int = 300) -> bool:
    """25 格的 validate_cfg + break_rarity 域 + token 长度 + 协变量。
    纯 CPU。在烧掉 91 小时之前跑，任何一格失败都必须先修。"""
    from probe import apply_edit, fit_position_offset
    from generator import generate_corpus
    from vocab import Vocab

    spec = mk_spec()
    vocab = Vocab(spec)
    rows, bad = [], []
    print(f"预检 {len(R_OLDS) * len(DDS)} 格，每格 {n_probe} 篇 ...", flush=True)
    for r, d in itertools.product(R_OLDS, DDS):
        cfg = mk_cfg(r, d, 0)
        try:
            validate_cfg(cfg, spec)
        except Exception as e:
            bad.append(f"R{r}_D{d} validate_cfg: {e}")
            rows.append((r, d, "FAIL", 0.0, 0.0, 0, 0.0, 0.0, pos_ceil(d)))
            print(f"  R{r:>2} D{d:>2}  FAIL {e}", flush=True)
            continue
        try:
            docs = list(generate_corpus(vocab, cfg, n_probe, seed_offset=1))
        except Exception as e:
            bad.append(f"R{r}_D{d} 生成失败: {type(e).__name__}: {e}")
            rows.append((r, d, "GEN", 0.0, 0.0, 0, 0.0, 0.0, pos_ceil(d)))
            print(f"  R{r:>2} D{d:>2}  生成失败 {type(e).__name__}", flush=True)
            continue
        off = fit_position_offset(docs)
        rng = random.Random(0)
        n_br = sum(1 for x in docs
                   if apply_edit(x, "break_rarity", vocab, cfg, rng, off))
        y = n_br / len(docs)
        maxlen = max(len(x.tokens) for x in docs)
        ddr = sum(x.realized_delta for x in docs) / len(docs)
        slots = sum(x.n_slots for x in docs) / len(docs)
        kept = sum(x.q_kept for x in docs) / len(docs)
        rows.append((r, d, "ok", y, ddr, maxlen, slots, kept, pos_ceil(d)))
        print(f"  R{r:>2} D{d:>2}  yield={y:.3f} ΔD={ddr:.2f} maxTok={maxlen} "
              f"slots={slots:.1f} Rreal={kept:.2f} posCeil={pos_ceil(d):.3f}",
              flush=True)
        if y < YIELD_FLOOR:
            bad.append(f"R{r}_D{d} break_rarity yield={y:.3f}，该格无 DV")
        if maxlen > spec.ctx_len:
            bad.append(f"R{r}_D{d} maxTok={maxlen} > ctx_len={spec.ctx_len}")

    lines = [f"{'R':>3} {'ΔD':>3} {'cfg':>5} {'brYield':>8} {'ΔDreal':>7} "
             f"{'maxTok':>7} {'slots':>6} {'Rreal':>6} {'posCeil':>8}"]
    for r, d, st, y, ddr, ml, sl, kp, pc in rows:
        lines.append(f"{r:>3} {d:>3} {st:>5} {y:>8.3f} {ddr:>7.2f} "
                     f"{ml:>7} {sl:>6.1f} {kp:>6.2f} {pc:>8.3f}")
    lines += ["",
              "posCeil：纯位置规则的解析上限，低 ΔD 行天然偏高，训练早期会被",
              "模型吃满（实测吻合到 98%）。它是格间协变量而非 bug，须报告；",
              "固定带宽对照臂（ΔD ~ U[d,d+8]，posCeil 恒为 1/9）见附录。",
              "brYield 随 ΔD 下降是选择偏置：_keyed 越界丢弃率随窗口变宽而升，",
              "高 ΔD 端存活副本更少，读数子集的 Rreal 偏高。",
              "Rreal：q_old 实际进文档的条数均值，名义 R_old 的实现值。",
              ""]
    lines += (["预检失败："] + bad) if bad else ["预检全部通过。"]
    txt = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "precheck.txt")
    with open(path, "w") as f:
        f.write(txt + "\n")
    print("\n" + "\n".join(lines[-(len(bad) + 1):]))
    print(f"已写入 {path}")
    return not bad


# ---------------- 训练 ----------------

def cmd_for(r, d, s, out_dir):
    return [sys.executable, "-u", "train.py",
            "--r", str(r), "--d", str(d), "--seed", str(s),
            "--steps", str(FIXED["steps"]),
            "--n-values", str(FIXED["n_values"]),
            "--n-entities", str(FIXED["n_entities"]),
            "--stmts-lo", str(FIXED["stmts_lo"]),
            "--stmts-hi", str(FIXED["stmts_hi"]),
            "--batch", str(FIXED["batch"]), "--lr", str(FIXED["lr"]),
            "--sched", FIXED["sched"], "--wd", str(FIXED["wd"]),
            "--eval-every", str(FIXED["eval_every"]),
            "--eval-docs", str(FIXED["eval_docs"]),
            "--workers", str(FIXED["workers"]),
            "--out", out_dir, "--tag", TAG]


def run_all(out_dir, dry, only_seed):
    os.makedirs(os.path.join(out_dir, "log"), exist_ok=True)
    todo = [c for c in cells() if only_seed is None or c[2] == only_seed]
    if not dry:
        # 溯源：跑完几天后要能确认这批权重用的是哪套配置
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(dict(fixed=FIXED, r_olds=R_OLDS, dds=DDS, seeds=SEEDS,
                           tag=TAG, n_cells=len(todo),
                           started=time.strftime("%Y-%m-%d %H:%M:%S")),
                      f, indent=2, ensure_ascii=False)
    prog = os.path.join(out_dir, "sweep_progress.txt")
    t_all = time.time()
    done = skipped = failed = 0
    for i, (r, d, s) in enumerate(todo, 1):
        tag = f"R{r}_D{d}_s{s}_{TAG}"
        if os.path.exists(os.path.join(out_dir, f"{tag}.pt")):
            skipped += 1
            print(f"[{i}/{len(todo)}] 跳过 {tag}", flush=True)
            continue
        cmd = cmd_for(r, d, s, out_dir)
        if dry:
            print(" ".join(cmd))
            continue
        log = os.path.join(out_dir, "log", f"{tag}.txt")
        t0 = time.time()
        eta = ((time.time() - t_all) / done * (len(todo) - i + 1) / 3600
               if done else float("nan"))
        print(f"[{i}/{len(todo)}] {tag} 开始 {time.strftime('%m-%d %H:%M')} "
              f"剩余约 {eta:.1f}h", flush=True)
        with open(log, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        dt = (time.time() - t0) / 60
        done += 1
        failed += (rc != 0)
        line = (f"{tag} rc={rc} {dt:.1f}min "
                f"累计{(time.time() - t_all) / 3600:.1f}h")
        print("  " + line, flush=True)
        with open(prog, "a") as f:
            f.write(line + "\n")
        if rc != 0:
            print(f"  失败，见 {log}。继续下一格。", flush=True)
    print(f"\n完成 {done}，跳过 {skipped}，失败 {failed}，"
          f"总计 {(time.time() - t_all) / 3600:.1f}h", flush=True)


# ---------------- 汇总 ----------------

def read_run(jl):
    """返回 (最后一条 probe, 最后一条 eval, 逃逸步数)。
    逃逸步数 = copy_acc 首次 ≥ ESC_FLOOR 的 step，分辨率等于 eval_every。
    这是本设计的核心协变量：它随 R_old 系统变化，故「Δ 的跨格差异」与
    「收敛后训练量的跨格差异」在原始网格上无法完全分离，须一并报告。"""
    probe = ev = None
    esc = float("nan")
    with open(jl) as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue                      # 训练中途，末行可能不完整
            k = o.get("kind")
            if k == "probe":
                probe = o
            elif k == "eval":
                ev = o
                if esc != esc and o.get("copy_acc", 0.0) >= ESC_FLOOR:
                    esc = o["step"]
    return probe, ev, esc


def load_pos_emp(path: Optional[str]) -> dict:
    """读 nl_collide.jsonl -> {(r, d): pos_emp}。None 或文件不存在返回空字典，
    此时 classify 退回 1/|supp|，主网格行为不变。

    存在理由见 classify 的 docstring：两个臂必须共用一个分类器，否则读数
    不可并列。这个函数是 NL 臂把它测出的经验上限交给分类器的唯一通道。
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise SystemExit(
            f"--pos-emp 指向 {path}，不存在。先跑 nl_collide.py 生成它；"
            f"若要用解析上限就不要给这个参数。")
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            # chance 与 pos_emp 一起读。缺 chance 的旧文件退回 None，
            # 此时不施加随机基线下限（行为等同于只给 ceil）。
            out[(o["r_old"], o["dd"])] = (o["pos_emp"], o.get("chance"))
    if not out:
        raise SystemExit(f"{path} 里没有可用记录，检查 nl_collide 的输出。")
    return out


def collect(out_dir, txt, pos_emp=None):
    """相图数据表。DV 是 break_rarity 的 d_margin：观测型 dominant 在此设计下
    恒为 last_value=rarity（两者在训练分布上逐篇等价，attribute 的 n_disc=0、
    rate_disc=nan），故主图必须用因果读数。

    pos_emp：{(r,d): 经验 posCeil}。给了就用它替代 1/|supp|（NL 臂）。
    空字典或 None 时逐比特等同于已发表的行为。
    """
    pos_emp = pos_emp or {}
    rows, flags = [], []
    for r, d, s in cells():
        tag = f"R{r}_D{d}_s{s}_{TAG}"
        jl = os.path.join(out_dir, f"{tag}.jsonl")
        if not os.path.exists(jl):
            continue
        probe, ev, esc = read_run(jl)
        if not probe:
            continue
        c = (probe.get("causal") or {}).get("break_rarity") or {}
        nan = float("nan")
        ev = ev or {}
        acc, ca = ev.get("acc", nan), ev.get("copy_acc", nan)
        ceil, ch = pos_emp.get((r, d), (None, None))
        x = dict(r=r, d=d, seed=s, step=probe["step"],
                 state=classify(acc, ca, d, ceil, ch),
                 acc=acc, copy=ca, esc=esc, tail0=ev.get("acc_tail0", nan),
                 n=c.get("n", 0), y=c.get("yield_rate", nan),
                 dm=c.get("d_margin", nan), fx=c.get("frac_expected", nan),
                 mass=c.get("mass_mean", nan))
        rows.append(x)
        if x["state"] == "retr" and x["n"] and x["fx"] == x["fx"]:
            k = round(x["fx"] * x["n"])
            p = sign_test_p(k, x["n"])       # 从 go_nogo 导入
            if p > 0.05:
                flags.append(f"R{r}_D{d}_s{s} frac+={x['fx']:.2f} "
                     f"p={p:.2f} 方向不成立")
        if x["mass"] == x["mass"] and x["mass"] < MASS_FLOOR:
            flags.append(f"R{r}_D{d}_s{s} mass={x['mass']:.2f} 读数无效")
        if x["state"] == "retr" and x["fx"] == x["fx"] and 0.4 < x["fx"] < 0.6:
            flags.append(f"R{r}_D{d}_s{s} frac+={x['fx']:.2f} 方向随机")

    lines = [f"{'R':>3} {'ΔD':>3} {'s':>2} {'step':>6} {'state':>7} {'acc':>6} "
             f"{'copy':>6} {'esc':>6} {'tail0':>6} {'n':>5} {'yld':>5} "
             f"{'Δ':>8} {'frac+':>6} {'mass':>6}"]
    for x in rows:
        lines.append(
            f"{x['r']:>3} {x['d']:>3} {x['seed']:>2} {x['step']:>6} "
            f"{x['state']:>7} {x['acc']:>6.3f} {x['copy']:>6.3f} "
            f"{x['esc']:>6.0f} {x['tail0']:>6.3f} {x['n']:>5} {x['y']:>5.2f} "
            f"{x['dm']:>+8.3f} {x['fx']:>6.2f} {x['mass']:>6.2f}")

    # 相图形状。只聚合 retrieval 态：pos 态的 Δ 是位置规则副产物，方向恒为负
    agg = {}
    for x in rows:
        if x["state"] == "retr":
            agg.setdefault((x["r"], x["d"]), []).append(x["dm"])
    if agg:
        lines += ["", "格均值（仅 state=retr）  行=R_old 列=ΔD",
                  "      " + "".join(f"{d:>9}" for d in DDS)]
        for r in R_OLDS:
            cs = []
            for d in DDS:
                v = agg.get((r, d))
                cs.append(f"{sum(v) / len(v):>+9.2f}" if v else f"{'—':>9}")
            lines.append(f"R{r:>4} " + "".join(cs))

    esc_by_r = {}
    for x in rows:
        if x["esc"] == x["esc"]:
            esc_by_r.setdefault(x["r"], []).append(x["esc"])
    if esc_by_r:
        lines += ["", "逃逸步数（copy_acc 首达 0.95）按 R_old  ——  协变量，须报告"]
        for r in sorted(esc_by_r):
            v = esc_by_r[r]
            lines.append(f"R{r:>4}  均值 {sum(v) / len(v):>6.0f}  "
                         f"范围 {min(v):.0f}–{max(v):.0f}  n={len(v)}")

    lines += ["", f"完成 {len(rows)}/{len(cells())} run。",
              "Δ>0 rarity 型，Δ<0 frequency 型，|Δ|≈0 纯 recency。",
              "state=retr 才可用：pos 态无检索回路、Δ 是位置规则副产物，",
              "none 态什么都没学到。mass<0.50 读数无效，"
              "frac+ 落在 0.4–0.6 表示方向随机、均值无意义。"]
    lines += ([""] + ["警告：" + f for f in flags]) if flags else ["", "无警告。"]
    lines += ["", "raw: " + json.dumps(rows, ensure_ascii=False)]

    out = "\n".join(lines)
    print(out)
    os.makedirs(os.path.dirname(txt) or ".", exist_ok=True)
    with open(txt, "w") as f:
        f.write(out + "\n")
    print(f"\n已写入 {txt}")


def main():
    import signal
    # 管道被 head/less 提前关闭时静默退出，而非抛 BrokenPipeError
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只做预检")
    ap.add_argument("--collect", action="store_true", help="只汇总")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="只跑某个 seed")
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--txt", default=None, help="默认 <out>/grid.txt")
    ap.add_argument("--pos-emp", default=None,
                    help="nl_collide.jsonl 的路径。给了就用经验 posCeil 替代 "
                         "1/|supp|（自然语言表层臂）。不给则逐比特等同于"
                         "已发表行为。")
    a = ap.parse_args()
    txt = a.txt or os.path.join(a.out, "grid.txt")
    pe = load_pos_emp(a.pos_emp)
    if pe:
        print(f"[posCeil] 用经验值，{len(pe)} 格来自 {a.pos_emp}")

    if a.check:
        sys.exit(0 if check(a.out) else 1)
    if a.collect:
        collect(a.out, txt, pe)
        return
    if not a.skip_check and not check(a.out):
        print("\n预检未通过，不启动主网格。")
        sys.exit(1)
    run_all(a.out, a.dry_run, a.seed)
    if not a.dry_run:
        collect(a.out, txt, pe)


if __name__ == "__main__":
    main()