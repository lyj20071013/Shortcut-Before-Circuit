"""主网格驱动。相图 = R_old × ΔD，DV = break_rarity 的因果读数 d_margin。

用法：
  python sweep.py --check              25 格预检，纯 CPU，约 30 秒，必须先过
  python sweep.py --dry-run            打印将要执行的命令，不跑
  python sweep.py                      跑全网格（预检不过则拒绝启动）
  python sweep.py --collect            汇总已完成的 run，可随时跑，不干扰训练

产物写入 --out（默认 runs_g2），与第一轮的 runs/ 分开：runs/ 里的 R2 五格
是位置捷径瞬态的证据、R16/R5/R3 的 12000 步 checkpoint 是步数敏感性对照，
都要留。新目录同时让 .pt 存在检查不会误跳过（步数已从 12000 改为 16000）。

设计要点：
- seed-major + 信息优先：seed 0 的四角与中心先跑，约 7 小时就能看出相图形状。
  若 Δ 在四角挤在同一量级，说明没有相变边界，及早止损而不是烧完 91 小时。
- 断点续跑：.pt 存在即跳过。SSH 掉线后重跑同一条命令即可接上。
- n_values=512：pilot 的 128 会让 _ValueDraw 死循环（R_old 小时一篇文档约 79
  个 slot、消耗约 120 个值，而 _keyed 丢弃的语句已消耗的值不归还），默认 2000
  则浪费算力。512 给约 4 倍余量。
- R_old 上限 16 而非 20：validate_cfg 要求 spread×n_stmts_lo ≥ 2×R_old，
  0.8×45=36 容得下 16 容不下 20。要上 20 得把 n_stmts_lo 提到 50。

【位置捷径与三态，第一轮实测得到的核心约束】
训练分两阶段。模型先爬满纯位置规则（复制固定 token 偏移处的值）的解析上限
posCeil = 1/(dd_hi-dd_lo+1)，此时 copy_acc≈0，检索回路根本没形成；之后某一步
突然逃逸到 copy_acc≈1。实测：
  R2/D2  卡在 acc=0.326 vs posCeil=0.333（98%），12000 步未逃逸
  R3/D5  逃逸前 acc=0.141 vs posCeil=0.143（98%），step 6000 逃逸
  R2/D16 posCeil 仅 0.059，step 4000 逃逸
  R16/*  step 1000 前逃逸
逃逸时刻由 R_old 决定（复制监督密度），捷径收益由 ΔD 带宽决定。三条后果：

1. Δ 只在 retrieval 态有意义。position 态模型的 break_rarity 响应方向恒为负
   （R2/D2 Δ=-0.47 fx=0.05、R2/D8 Δ=-1.30 fx=0.02），若混进相图会在低 R_old
   角伪造出一个「frequency 型」区域。classify() 三态判读负责隔离，只有
   state=retr 的 run 进格均值。
2. R_old=2 在 16000 步内不逃逸，移出主网格（下界取 3，R3 已验证 step 6000
   逃逸、8000 达 acc=0.999）。R2 五格作为捷径瞬态证据进附录。
3. total_steps 12000→16000。收敛后训练量在 R16（约 15000 步）与 R3（约 8000
   步）之间的比值从 2.75 降到 1.9，减轻「比的是训练时长而非数据统计」这一
   混淆。逃逸步数本身作为协变量报告，见 collect 的 esc 列与末尾汇总。
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


def classify(acc, copy_acc, d):
    """retr / posNN% / none。posNN% 是 acc 占 posCeil 的比例，≥90% 即可判定
    模型在用位置规则；此时 Δ 不可用。"""
    if acc != acc or copy_acc != copy_acc:
        return "?"
    if copy_acc >= COPY_FLOOR and acc >= ACC_FLOOR:
        return "retr"
    if copy_acc < 0.5 and acc >= 0.5 * pos_ceil(d):
        return f"pos{acc / pos_ceil(d):.0%}"
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
        probe, ev, esc = read_run(jl)
        if not probe:
            continue
        c = (probe.get("causal") or {}).get("break_rarity") or {}
        nan = float("nan")
        ev = ev or {}
        acc, ca = ev.get("acc", nan), ev.get("copy_acc", nan)
        x = dict(r=r, d=d, seed=s, step=probe["step"], state=classify(acc, ca, d),
                 acc=acc, copy=ca, esc=esc, tail0=ev.get("acc_tail0", nan),
                 n=c.get("n", 0), y=c.get("yield_rate", nan),
                 dm=c.get("d_margin", nan), fx=c.get("frac_expected", nan),
                 mass=c.get("mass_mean", nan))
        rows.append(x)
        if x["state"] != "retr":
            flags.append(f"R{r}_D{d}_s{s} state={x['state']} acc={acc:.3f} "
                         f"copy={ca:.3f} Δ 不可用")
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
    a = ap.parse_args()
    txt = a.txt or os.path.join(a.out, "grid.txt")

    if a.check:
        sys.exit(0 if check(a.out) else 1)
    if a.collect:
        collect(a.out, txt)
        return
    if not a.skip_check and not check(a.out):
        print("\n预检未通过，不启动主网格。")
        sys.exit(1)
    run_all(a.out, a.dry_run, a.seed)
    if not a.dry_run:
        collect(a.out, txt)


if __name__ == "__main__":
    main()