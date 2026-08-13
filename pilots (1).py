"""诊断 pilot 的一键运行与汇总。

为什么跑这些：答案位每篇只有 1 个 token、占全 token loss 约 0.4%，看不出
学习进展。但语料里每篇约 50 个 val token 是"同 slot 上一次的值"的重复，
预测它们需要 (e,a) 匹配 + 复制，与答案位是同一条回路，且占 loss 约 45%。
copy_nll 因此是回路是否形成的直接读数，样本数比答案位多 50 倍。

每个 pilot 独立子进程：一个崩了不影响其余，GPU 显存也彻底释放。stdout 同时
落盘到 runs/<tag>.log（手机 SSH 滚屏会丢内容，日志必须落盘）。

判读统一用 copy_gain = chance_nll - copy_nll，即低于 chance 多少 nats。
n_values 不同的 pilot 绝对 NLL 不可比（ln2000=7.60 vs ln512=6.24），
只有 gain 可比。阈值 1.0 nat 是"回路开始形成"的保守线。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Option
BASE = ["--r", "3", "--d", "5", "--seed", "0", "--workers", "8",
        "--eval-docs", "1500", "--eval-every", "750",
        "--ctx-len", "1024"]        # 显式声明，不受 config.py 编辑影响

# tag -> (额外参数, 这个 pilot 在测什么)
PILOTS: Dict[str, tuple] = {
    "base": ([],
             "对照：当前正式配置，untied，batch 48"),
    "tie": (["--tie"],
            "假设 A：unembedding 与 embedding 的 2000 组行对齐太慢"),
    "v512": (["--n-values", "512"],
             "假设 A 的另一条路：值池缩到 512，保留 untied"),
    "bs192": (["--batch", "192", "--lr", "1e-3"],
              "吞吐上限：batch 翻 4 倍，看 sec_per_step 是否 < 3 倍"),
    # 兜底：前三个都不动时才跑。刻意简化到"必须能学会"，用于建立可学性下界。
    "floor": (["--n-values", "128", "--n-entities", "200",
               "--stmts-lo", "30", "--stmts-hi", "40",
               "--batch", "192", "--lr", "1e-3"],
              "兜底：极简配置，确认任务本身可学"),
}
DEFAULT_SET = ["tie", "v512", "bs192"]

COPY_GAIN_PASS = 1.0        # nats below chance
SEC_PER_STEP_PASS = 1.3     # batch 192 下的目标


def run_one(tag: str, steps: int, out_dir: str) -> int:
    extra, why = PILOTS[tag]
    cmd = [sys.executable, "-u", "train.py", *BASE, "--steps", str(steps),
           "--out", out_dir, "--tag", tag, *extra]
    log_path = os.path.join(out_dir, f"{tag}.log")
    print(f"\n{'=' * 70}\n[{tag}] {why}\n  {' '.join(cmd)}\n  日志 -> {log_path}\n{'=' * 70}",
          flush=True)
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(" ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        rc = p.wait()
        f.write(f"\nexit={rc} minutes={(time.time() - t0) / 60:.1f}\n")
    print(f"[{tag}] exit={rc} 用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    return rc


def read_jsonl(path: str) -> Optional[dict]:
    """取 meta、最后一条 eval、最后一条 train、最后一条 probe。"""
    if not os.path.exists(path):
        return None
    got: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            got[r.get("kind", "?")] = r
    return got or None


def find_jsonl(tag: str, out_dir: str) -> Optional[str]:
    for fn in os.listdir(out_dir):
        if fn.endswith(f"_{tag}.jsonl"):
            return os.path.join(out_dir, fn)
    return None


def summarize(tags: List[str], out_dir: str) -> str:
    hdr = (f"{'pilot':<8}{'params':>8}{'bs':>5}{'lr':>8}{'nVal':>6}"
           f"{'step':>7}{'s/step':>8}{'tok/s':>8}"
           f"{'copyNLL':>9}{'chance':>8}{'gain':>7}{'copyAcc':>8}"
           f"{'novel':>7}{'ansNLL':>8}{'rank':>7}{'acc':>7}  判读")
    lines = [hdr, "-" * len(hdr)]
    verdicts = {}
    for tag in tags:
        path = find_jsonl(tag, out_dir)
        got = read_jsonl(path) if path else None
        if not got or "eval" not in got:
            lines.append(f"{tag:<8}  (无 eval 记录，见 {tag}.log)")
            verdicts[tag] = "fail"
            continue
        m, e = got.get("meta", {}), got["eval"]
        t = got.get("train", {})
        spec = m.get("spec", {})
        nv = spec.get("n_values", "?")
        sps = t.get("sec_per_step", float("nan"))
        tok = t.get("tokens", 0) / max(1e-9, sps * max(1, t.get("step", 1)))
        ch = e.get("copy_chance_nll", e.get("chance_nll", float("nan")))
        cn = e.get("copy_nll", float("nan"))
        gain = ch - cn
        ok = gain > COPY_GAIN_PASS
        verdicts[tag] = "pass" if ok else "fail"
        mark = "回路在形成" if ok else "回路未起来"
        if tag == "bs192":
            mark += f" / 吞吐{'可接受' if sps < SEC_PER_STEP_PASS else '仍是瓶颈'}"
        lines.append(
            f"{tag:<8}{m.get('n_params', 0) / 1e6:>7.1f}M"
            f"{m.get('train', {}).get('batch_docs', 0):>5}"
            f"{m.get('train', {}).get('lr', 0):>8.1e}{nv:>6}"
            f"{e.get('step', 0):>7}{sps:>8.3f}{tok:>8.0f}"
            f"{cn:>9.3f}{ch:>8.2f}{gain:>+7.2f}{e.get('copy_acc', 0):>8.3f}"
            f"{e.get('novel_nll', 0):>7.2f}{e.get('ans_nll', 0):>8.3f}"
            f"{e.get('ans_rank', 0):>7.0f}{e.get('acc', 0):>7.3f}  {mark}")

    lines += ["", "判读口径：",
              f"  copy_gain = chance_nll - copy_nll > {COPY_GAIN_PASS} nat 视为回路开始形成。",
              "  novel_nll 是对照，新值不可预测，应恒等于 chance。",
              "  n_values 不同的 pilot 只能比 gain，不能比绝对 NLL。", ""]

    p = [k for k in ("tie", "v512") if verdicts.get(k) == "pass"]
    if "v512" in p:
        lines.append("下一步：锁定 n_values=512（保住 untied，§7 的 logit attribution 不受影响），"
                     "重跑 selfcheck 确认 bind 与答案均匀度。")
    elif "tie" in p:
        lines.append("下一步：接受 tie embedding，§7 的归因改用 attention pattern + "
                     "path patching（不依赖 unembedding 独立），方法节须声明该取舍。")
    elif verdicts:
        lines.append("下一步：两个假设都不成立，跑兜底 pilot 建立可学性下界："
                     "python pilots.py --only floor")
    if verdicts.get("bs192") == "pass":
        lines.append("吞吐：batch 192 有效，90 run 的预算按该 sec_per_step 重算。")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", type=str, default="runs")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=list(PILOTS), help="默认跑 tie / v512 / bs192")
    ap.add_argument("--summary-only", action="store_true",
                    help="不重跑，只从已有 jsonl 重新汇总")
    a = ap.parse_args()

    tags = a.only or DEFAULT_SET
    os.makedirs(a.out, exist_ok=True)
    if not a.summary_only:
        for tag in tags:
            try:
                run_one(tag, a.steps, a.out)
            except KeyboardInterrupt:
                print(f"\n[{tag}] 被中断，继续汇总已完成的部分", flush=True)
                break
            except Exception as ex:      # 一个 pilot 崩掉不该带走其余
                print(f"[{tag}] 启动失败: {ex}", flush=True)

    out = summarize(tags, a.out)
    print("\n" + out, flush=True)
    sp = os.path.join(a.out, "pilot_summary.txt")
    with open(sp, "w") as f:
        f.write(out + "\n")
    print(f"\n汇总已写入 {sp}")


if __name__ == "__main__":
    main()