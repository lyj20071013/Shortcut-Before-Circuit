"""主网格驱动。相图 = R_old × ΔD，DV = break_rarity 的因果读数 d_margin。

用法：
  python sweep.py --check              30 格预检，纯 CPU，约 30 秒，必须先过
  python sweep.py --dry-run            打印将要执行的命令，不跑
  python sweep.py                      跑全网格（预检不过则拒绝启动）
  python sweep.py --collect            汇总已完成的 run，可随时跑，不干扰训练

设计要点：
- seed-major + 信息优先：seed 0 的四角与中心先跑，约 5.5 小时就能看出相图
  形状。若 Δ 在四角挤在同一量级，说明没有相变边界，及早止损而不是烧完 82 小时。
- 断点续跑：.pt 存在即跳过。SSH 掉线后重跑同一条命令即可接上。
- R_old=1 不在主网格：break_rarity 要求 q slot 有 ≥2 份老值副本可改写，
  R_old=1 时无域（预检实测 yield=0.000）。它是结构性零信号对照，
  相图上应画成空白而非蓝色，另作对照臂单跑。
- R_old 上限 16 而非 20：validate_cfg 要求 spread×n_stmts_lo ≥ 2×R_old，
  0.8×45=36 容得下 16 容不下 20。要上 20 得把 n_stmts_lo 提到 50。
- n_values=512 而非默认 2000 或 pilot 的 128：128 会让 _ValueDraw 死循环
  （R_old=2 时一篇文档约 79 个 slot、消耗约 120 个值），2000 则浪费算力。
  512 给约 4 倍余量，且已验证 1000 步内收敛。
"""
import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time

from config import CorpusCfg, LangSpec, dd_band, validate_cfg

R_OLDS = [2, 3, 5, 8, 12, 16]
DDS = [2, 3, 5, 8, 16]
SEEDS = [0, 1, 2]
HEAD = [(16, 5), (16, 2), (16, 16), (2, 2), (2, 16), (5, 5)]   # 优先跑

FIXED = dict(steps=12000, n_values=512, n_entities=200,
             stmts_lo=45, stmts_hi=55, batch=256, lr=1e-3,
             sched="cos", wd=0.1, eval_every=1000, eval_docs=4000,
             workers=4)
TAG = "grid"
YIELD_FLOOR = 0.05          # break_rarity 域低于此值：该格无 DV
MASS_FLOOR = 0.50           # 概率质量低于此值：OOD 探针失效，读数无意义
ACC_FLOOR = 0.99            # 低于此值：该格未收敛，须剔除或补步数


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


# ---------------- 预检 ----------------

def check(n_probe: int = 300) -> bool:
    """30 格的 validate_cfg + break_rarity 域 + token 长度。
    纯 CPU。在烧掉 82 小时之前跑，任何一格失败都必须先修。"""
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
            rows.append((r, d, "FAIL", 0.0, 0.0, 0, 0.0))
            print(f"  R{r:>2} D{d:>2}  FAIL {e}", flush=True)
            continue
        try:
            docs = list(generate_corpus(vocab, cfg, n_probe, seed_offset=1))
        except Exception as e:
            bad.append(f"R{r}_D{d} 生成失败: {type(e).__name__}: {e}")
            rows.append((r, d, "GEN", 0.0, 0.0, 0, 0.0))
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
        rows.append((r, d, "ok", y, ddr, maxlen, slots))
        print(f"  R{r:>2} D{d:>2}  yield={y:.3f} ΔD={ddr:.2f} "
              f"maxTok={maxlen} slots={slots:.1f}", flush=True)
        if y < YIELD_FLOOR:
            bad.append(f"R{r}_D{d} break_rarity yield={y:.3f}，该格无 DV")
        if maxlen > spec.ctx_len:
            bad.append(f"R{r}_D{d} maxTok={maxlen} > ctx_len={spec.ctx_len}")

    lines = [f"{'R':>3} {'ΔD':>3} {'cfg':>5} {'brYield':>8} {'ΔDreal':>7} "
             f"{'maxTok':>7} {'slots':>6}"]
    for r, d, st, y, ddr, ml, sl in rows:
        lines.append(f"{r:>3} {d:>3} {st:>5} {y:>8.3f} {ddr:>7.2f} "
                     f"{ml:>7} {sl:>6.1f}")
    lines += ["", "brYield 随 ΔD 下降是选择偏置：_keyed 的越界丢弃率随窗口",
              "变宽而上升，高 ΔD 端存活副本更少。低 R_old × 高 ΔD 的格子",
              "读数只来自约半数文档，条件化子集的 Rreal 偏高，须作协变量报告。",
              ""]
    lines += (["预检失败："] + bad) if bad else ["预检全部通过。"]
    txt = "\n".join(lines)
    os.makedirs("runs", exist_ok=True)
    with open("runs/precheck.txt", "w") as f:
        f.write(txt + "\n")
    print("\n" + "\n".join(lines[-(len(bad) + 1):]))
    print("已写入 runs/precheck.txt")
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

def collect(out_dir, txt):
    """相图数据表。DV 是 break_rarity 的 d_margin：观测型 dominant 在此设计下
    恒为 last_value=rarity（两者在训练分布上逐篇等价，attribute 的 n_disc=0、
    rate_disc=nan），故主图必须用因果读数。"""
    rows, flags = [], []
    for r, d, s in cells():
        tag = f"R{r}_D{d}_s{s}_{TAG}"
        jl = os.path.join(out_dir, f"{tag}.jsonl")
        if not os.path.exists(jl):
            continue
        probe = ev = None
        with open(jl) as f:
            for line in f:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # 训练中途，末行可能不完整
                if o.get("kind") == "probe":
                    probe = o
                elif o.get("kind") == "eval":
                    ev = o
        if not probe:
            continue
        c = (probe.get("causal") or {}).get("break_rarity") or {}
        nan = float("nan")
        x = dict(r=r, d=d, seed=s, step=probe["step"],
                 acc=(ev or {}).get("acc", nan),
                 tail0=(ev or {}).get("acc_tail0", nan),
                 n=c.get("n", 0), y=c.get("yield_rate", nan),
                 dm=c.get("d_margin", nan), fx=c.get("frac_expected", nan),
                 mass=c.get("mass_mean", nan))
        rows.append(x)
        if x["acc"] == x["acc"] and x["acc"] < ACC_FLOOR:
            flags.append(f"R{r}_D{d}_s{s} acc={x['acc']:.3f} 未收敛")
        if x["mass"] == x["mass"] and x["mass"] < MASS_FLOOR:
            flags.append(f"R{r}_D{d}_s{s} mass={x['mass']:.2f} 读数无效")
        if x["fx"] == x["fx"] and 0.4 < x["fx"] < 0.6:
            flags.append(f"R{r}_D{d}_s{s} frac+={x['fx']:.2f} 方向随机")

    lines = [f"{'R':>3} {'ΔD':>3} {'s':>2} {'step':>6} {'acc':>6} {'tail0':>6} "
             f"{'n':>5} {'yield':>6} {'Δ':>8} {'frac+':>6} {'mass':>6}"]
    for x in rows:
        lines.append(
            f"{x['r']:>3} {x['d']:>3} {x['seed']:>2} {x['step']:>6} "
            f"{x['acc']:>6.3f} {x['tail0']:>6.3f} {x['n']:>5} {x['y']:>6.2f} "
            f"{x['dm']:>+8.3f} {x['fx']:>6.2f} {x['mass']:>6.2f}")

    # seed 聚合，用于看相图形状
    agg = {}
    for x in rows:
        if x["acc"] == x["acc"] and x["acc"] >= ACC_FLOOR:
            agg.setdefault((x["r"], x["d"]), []).append(x["dm"])
    if agg:
        lines += ["", "格均值（仅 acc≥0.99 的 run）  行=R_old 列=ΔD",
                  "      " + "".join(f"{d:>9}" for d in DDS)]
        for r in R_OLDS:
            cs = []
            for d in DDS:
                v = agg.get((r, d))
                cs.append(f"{sum(v) / len(v):>+9.2f}" if v else f"{'—':>9}")
            lines.append(f"R{r:>4} " + "".join(cs))

    lines += ["", f"完成 {len(rows)}/{len(cells())} run。",
              "Δ>0 rarity 型，Δ<0 frequency 型，|Δ|≈0 纯 recency。",
              f"acc<{ACC_FLOOR} 未收敛；mass<{MASS_FLOOR} 读数无效；"
              "frac+ 在 0.4–0.6 方向随机。"]
    lines += ([""] + ["警告：" + f for f in flags]) if flags else ["", "无警告。"]
    lines += ["", "raw: " + json.dumps(rows, ensure_ascii=False)]

    out = "\n".join(lines)
    print(out)
    os.makedirs(os.path.dirname(txt) or ".", exist_ok=True)
    with open(txt, "w") as f:
        f.write(out + "\n")
    print(f"\n已写入 {txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只做预检")
    ap.add_argument("--collect", action="store_true", help="只汇总")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="只跑某个 seed")
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--txt", default="runs/grid.txt")
    a = ap.parse_args()

    if a.check:
        sys.exit(0 if check() else 1)
    if a.collect:
        collect(a.out, a.txt)
        return
    if not a.skip_check and not check():
        print("\n预检未通过，不启动主网格。")
        sys.exit(1)
    run_all(a.out, a.dry_run, a.seed)
    if not a.dry_run:
        collect(a.out, a.txt)


if __name__ == "__main__":
    main()