"""生成 App B 的 γ 表 LaTeX 行，八行全部从 go_nogo 缓存读，不手抄。

三个校验，任一不过就在 stderr 报警：
  n 接近 400 —— 混进 --docs 200 的旧结果会让 frac+ 与论文表格差第三位小数
  mass 与 mOK —— mass<0.5 的行不出读数（那两个 γ=1 的 R3 格）
  两侧 gain 齐全 —— 缺一侧说明该格没有对照
中位数一律用 ungated 的 d_median，与 Table stats 的 median 列同约定。
gated 的 d_median_valid 在 mOK 低的格上是在极少数文档上算的（R3/D5 s0 gamma1
是 39 篇），不能与 ungated 混排。
"""
import argparse
import glob
import json
import os
import sys

CELLS = [("R3_D5", r"$R_{\mathrm{old}}=3, \Delta D=5$"),
         ("R16_D2", r"$R_{\mathrm{old}}=16, \Delta D=2$")]
SEEDS = [0, 1]
ARMS = [("grid", "2.0"), ("gamma1", "1.0")]
MASS_FLOOR = 0.50
N_MIN = 380


def load(dirs):
    out = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
            if "go_nogo" not in os.path.basename(p):
                continue
            for line in open(p):
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "tag" in o and "d_median" in o:
                    out[o["tag"]] = (o, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="含 go_nogo.txt.jsonl 的目录")
    a = ap.parse_args()
    got = load(a.dirs)
    print(f"% 从 {len(got)} 条 go_nogo 记录中提取", file=sys.stderr)

    bad = 0
    for key, label in CELLS:
        for seed in SEEDS:
            for arm, gain in ARMS:
                tag = f"{key}_s{seed}_{arm}"
                if tag not in got:
                    print(f"缺 {tag}", file=sys.stderr)
                    bad += 1
                    print(f"{label}, s{seed} & {gain} & "
                          r"\CHECK{missing} & & & & & \\")
                    continue
                o, src = got[tag]
                n = o["n"]
                yld = o.get("yield_rate") or 1.0
                docs = n / yld if yld > 0 else 0
                if docs < 380:
                    print(f"{tag}: n={n} yield={yld:.2f} → 约 {docs:.0f} 篇，"
                          f"不是 400 篇的跑法，重跑（源 {src}）", file=sys.stderr)
                    bad += 1
                esc = o.get("escape_step")
                esc_s = "none" if not (esc == esc) else f"{int(esc)}"
                mass = o["mass"]
                if mass >= MASS_FLOOR:
                    med = f"${o['d_median']:+.3f}$"
                    fr = f"{o['frac_positive']:.3f}"
                else:
                    med = fr = "---"
                    print(f"{tag}: mass={mass:.2f} < {MASS_FLOOR}，读数按 --- 输出",
                          file=sys.stderr)
                print(f"{label}, s{seed} & {gain} & {esc_s} & "
                      f"{o['acc']:.3f} & {o['copy_acc']:.3f} & {mass:.2f} & "
                      f"{med} & {fr} \\\\")
            # 同格两个种子之间空一行
        print(r"\addlinespace")
    if bad:
        print(f"\n{bad} 处问题，见上。", file=sys.stderr)
        sys.exit(1)
    print("八行齐，n 与 mass 均通过。", file=sys.stderr)


if __name__ == "__main__":
    main()

# python gamma_table.py runs_g2 /root/autodl-tmp/runs_g2_s1 /root/autodl-tmp/runs_gamma1 > gamma_rows.tex