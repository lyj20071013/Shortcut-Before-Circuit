"""Train a Transformer from scratch on rendered-language documents."""
import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import LM, ModelCfg
from nl_corpus import NLStream, nl_spec, tokenize

PAD = 0
COPY_FLOOR = 0.95      # 与 sweep.py 同值
ACC_FLOOR = 0.99


def collate(batch, ctx_len: int):
    """右填充到批内最长，不到 ctx_len —— 与 train.py 的动态填充一致。
    labels 的 pad 位置置 -100，model.forward 的 cross_entropy 会忽略。"""
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), PAD, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    apos = torch.zeros(len(batch), dtype=torch.long)
    for i, (t, p) in enumerate(batch):
        ids[i, :len(t)] = torch.tensor(t)
        lab[i, :len(t)] = torch.tensor(t)
        apos[i] = p
    return ids, lab, apos


def lr_at(step: int, total: int, lr: float, warmup: int = 500,
          lr_min_frac: float = 0.1, sched: str = "cos") -> float:
    """train.py:420-427 的逐行复制。两臂的调度必须逐比特相同，否则 σ 的比较
    混进调度差异 —— 而 App.~constlr 已经量过调度单独能移动读数 0.749。"""
    if step < warmup:
        return lr * (step + 1) / warmup
    if sched == "const":
        return lr
    t = (step - warmup) / max(1, total - warmup)
    return lr * (lr_min_frac + (1 - lr_min_frac) *
                 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


class Probe:
    """离线探针集 + break_rarity 读数。

    Δ = [logp(v*) - logp(truth)]_edit - [同]_base，与 probe._margin 同式。
    logp 在 adj_ids 上归一化 —— 对应 train.py:318-319 在 value 块内做
    log_softmax。两臂都是"答案 token 的候选集内部归一化"，只是取法不同。
    """

    def __init__(self, path: str, device: str):
        z = np.load(path)
        self.base = torch.tensor(z["base"].astype(np.int64))
        self.edit = torch.tensor(z["edit"].astype(np.int64))
        self.apos = torch.tensor(z["answer_pos"].astype(np.int64))
        self.eapos = torch.tensor(z["edit_answer_pos"].astype(np.int64))
        self.ans = torch.tensor(z["answer"].astype(np.int64))
        self.vst = torch.tensor(z["v_star"].astype(np.int64))
        self.blen = torch.tensor(z["base_len"].astype(np.int64))
        self.adj = torch.tensor(z["adj_ids"].astype(np.int64)).to(device)
        self.copy_pos = torch.tensor(z["copy_pos"].astype(np.int64))
        self.copy_rep = torch.tensor(z["copy_rep"].astype(np.int64))
        self.version = str(z["version"]) if "version" in z else "?"
        self.n = len(self.base)
        self.device = device
        # adj_ids 在词表里的位置 -> 它在 adj 序列里的下标。读数要把
        # answer/v_star 这两个词表 id 映射到 adj 内部下标才能索引 log_softmax。
        self.adj_index = {int(v): i for i, v in enumerate(z["adj_ids"])}

    def _lp(self, model, ids, apos):
        """answer_pos 上的 logits 行（全词表，未归一化）。

        取 logits[apos-1]：预测第 apos 个 token 的分布在第 apos-1 个位置上，
        与 train.py:313 的 ps.append(vpos - 1) 同一约定。

        返回全词表行而不是 adj 内部的 log_softmax，因为 mass 与 Δ 需要**不同
        的归一化域**：
          Δ    在 adj 集内部归一化。它是两个候选的 log-odds 差，域外的质量
               不影响差值，而 train.py:318-319 的 copy 诊断也是块内归一化。
          mass 必须在全词表上算。它的用途是"候选对是否占住了概率质量"
               （probe._mass：<0.5 视为读数无效）。若在 adj 集内归一化，
               模型把 99% 质量放在 'Alice' 这种非 adj token 上也看不出来 ——
               mass 恒高，门变成空的。
        先前这里只返回 adj 内部的 log_softmax，两个量共用它，于是 mass 门失效。
        """
        out = []
        bs = 64
        for i in range(0, len(ids), bs):
            b = ids[i:i + bs].to(self.device)
            p = apos[i:i + bs].to(self.device)
            logits, _ = model(b)
            sel = logits[torch.arange(len(b), device=self.device), p - 1]
            out.append(sel.float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def read(self, model) -> dict:
        model.eval()
        lg_b = self._lp(model, self.base, self.apos)     # 全词表 logits 行
        lg_e = self._lp(model, self.edit, self.eapos)
        model.train()

        # answer / v_star 是词表 id。它们必须都在 adj 集内 —— 两者都是某个
        # val_id 的 adj（nl_corpus 的 ADJ_OF），而 adj_ids 是前 n_values 个
        # adj 的词表 id。不在就是 build_probe 与 adj_ids 的 ADJ 顺序不一致，
        # 那会让读数静默错位到另一个 token 上。
        miss = [int(v) for v in list(self.ans) + list(self.vst)
                if int(v) not in self.adj_index]
        if miss:
            raise RuntimeError(
                f"{len(miss)} 个 answer/v_star 不在 adj_ids 内，例如 "
                f"{miss[:3]}。build_probe 与 adj_ids 用的 ADJ 顺序不一致，"
                f"或 adj_ids 的 [:n_values] 截断把它们切掉了。")

        ia = torch.tensor([self.adj_index[int(v)] for v in self.ans])
        iv = torch.tensor([self.adj_index[int(v)] for v in self.vst])
        r = torch.arange(len(ia))

        # Δ：在 adj 集内部归一化。log-odds 之差，域外质量不影响它。
        lp_b = torch.log_softmax(lg_b[:, self.adj.cpu()], -1)
        lp_e = torch.log_softmax(lg_e[:, self.adj.cpu()], -1)
        d = ((lp_e[r, iv] - lp_e[r, ia]) - (lp_b[r, iv] - lp_b[r, ia])).numpy()

        # mass：在**全词表**上算。probe._mass 的判据是"候选对占据的概率质量
        # <0.5 则该读数无效"，而这个判断只有在全词表上才有意义。
        pe = torch.softmax(lg_e, -1)
        mass = (pe[r, self.vst] + pe[r, self.ans]).numpy()

        # 准确率：全词表 argmax 命中答案 token。主网格的 acc 是 value 块内的
        # argmax（train.py:318-321），比这个宽松 —— 这里更严，故若 NL 臂的
        # acc 过 0.99，块内 acc 必然也过。
        acc = float((lg_b.argmax(-1) == self.ans).float().mean())
        # 块内版本一并报，与主网格同口径便于并列
        acc_blk = float((lp_b.argmax(-1) == ia).float().mean())
        # These training-time summaries use all valid edit pairs (ungated).
        # Preserve this schema consistently across seeds. nl_regate.py can
        # recompute mass-restricted terminal summaries from model weights.
        # Intermediate mass-restricted summaries require the corresponding
        # checkpoints and cannot be recovered from these aggregate fields.
        return dict(n=len(d), d_margin=float(np.mean(d)),
                    frac_expected=float(np.mean(d > 0)),
                    mass_mean=float(np.mean(mass)),
                    mass_min=float(np.min(mass)),
                    acc_base=acc_blk, acc_full=acc,
                    d_all=[float(x) for x in d])

    @torch.no_grad()
    def copy_diag(self, model) -> dict:
        """copy_acc 的 NL 类比：在"重复出现的值 token"上的 argmax 准确率。

        train.py:300-332 的 copy_acc 用 _val_slots(d, cfg) 枚举每个值位置并标
        是否重复；build_probe 把同一批位置存成 copy_pos / copy_rep（右填充 -1，
        因为 0 是合法位置）。所以这是同一个量，只是位置来自离线记录而不是
        现算。

        与主网格的差异：那里 argmax 在 512 个 value token 的连续区间上，这里
        在 150 个 adj 的 gather 集上。候选集小 3.4 倍，故 copy_acc 在 NL 臂上
        天然偏高 —— COPY_FLOOR=0.95 这个门因此偏松，要在文里注明。
        """
        model.eval()
        hit = tot = hit_n = tot_n = 0
        bs = 32
        for i in range(0, self.n, bs):
            b = self.base[i:i + bs].to(self.device)
            logits, _ = model(b)
            lg = logits[:, :, self.adj].argmax(-1)      # (B, L) adj 内部下标
            for j in range(len(b)):
                pos = self.copy_pos[i + j]
                rep = self.copy_rep[i + j]
                # 向量化：先掐掉哨兵与 p==0，再一次性比较。先前逐位置
                # int(lg[...].argmax()) 每次都同步一次 GPU，200 篇 × 约 60 个
                # 位置 = 一万两千次同步，单个探针点就要几秒。本次会话第五次
                # 同类浪费（kv_extract 4x、py_slots 2x、g3 60x、build_train
                # 的 40 万篇）。
                m = (pos >= 1)                          # p==0 时 p-1 越界
                if not bool(m.any()):
                    continue
                p = pos[m]
                r = rep[m]
                tgt = self.base[i + j][p]
                # 只算落在 adj 集内的目标。NL 词表里名字、功能词、带句点的
                # 形式都不是 adj，它们不是"值 token"，不进 copy 诊断。
                keep = torch.tensor([int(t) in self.adj_index for t in tgt])
                if not bool(keep.any()):
                    continue
                p, r, tgt = p[keep], r[keep], tgt[keep]
                want = torch.tensor([self.adj_index[int(t)] for t in tgt])
                # p 必须移到 device：lg 在 GPU 上，而用 CPU LongTensor 索引
                # CUDA 张量在部分 PyTorch 版本上抛 "indices should be either
                # on cpu or on the same device"。
                got = lg[j, (p - 1).to(self.device)].cpu()
                ok = (got == want)
                is_rep = (r == 1)
                hit += int(ok[is_rep].sum())
                tot += int(is_rep.sum())
                hit_n += int(ok[~is_rep].sum())
                tot_n += int((~is_rep).sum())
        model.train()
        nan = float("nan")
        return dict(copy_acc=hit / tot if tot else nan, n_copy=tot,
                    novel_acc=hit_n / tot_n if tot_n else nan, n_novel=tot_n)


def probe_steps(total: int, k: int = 7) -> List[int]:
    """log-spaced，含最后一步。与 TrainCfg.probe_points=7 一致。"""
    xs = sorted({1, total} | {int(round(total ** (i / (k - 1))))
                              for i in range(k)})
    return [x for x in xs if 1 <= x <= total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--sched", default="cos", choices=["cos", "const"])
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", default="nl_data")
    ap.add_argument("--out", default="runs_nl")
    ap.add_argument("--tag", default="nl")
    a = ap.parse_args()

    spec = nl_spec()
    with open(os.path.join(a.data, "vocab.json")) as f:
        vj = json.load(f)
    vocab, ver = vj["vocab"], vj["version"]
    print(f"词表 {len(vocab)}  语料版本 {ver}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (torch.bfloat16 if torch.cuda.is_bf16_supported()
           else torch.float16) if dev == "cuda" else torch.float32

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)

    # ModelCfg 只给 vocab_size 与 ctx_len，其余走默认 —— 主网格的 meta 记录
    # 显示它也是这样（d_model=512 n_layer=8 n_head=8 d_mlp=1376）。
    mc = ModelCfg(vocab_size=len(vocab), ctx_len=spec.ctx_len)
    model = LM(mc).to(dev)
    print(f"参数 {model.n_params():,}  非 embedding {model.n_params(False):,}")

    opt = torch.optim.AdamW(model.param_groups(a.wd), lr=a.lr,
                            betas=(0.9, 0.95))
    ds = NLStream(a.r, a.d, a.seed, vocab, spec)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers,
                    collate_fn=lambda b: collate(b, spec.ctx_len),
                    persistent_workers=a.workers > 0)

    # 探针集不带 seed：一份，全部训练 seed 共用（理由见 nl_corpus.main）。
    # 这样跨 seed 极差里没有 batch 项，与 App.~cross 的共享批次口径一致。
    pr = Probe(os.path.join(a.data, f"probe_R{a.r}_D{a.d}.npz"), dev)
    if pr.version != ver:
        raise SystemExit(
            f"探针集版本 {pr.version} != 词表版本 {ver}。两者必须来自同一次 "
            f"nl_corpus 运行，否则 adj_ids 的顺序可能不一致而读数静默错位。")
    print(f"探针 {pr.n} 篇  adj 候选 {len(pr.adj)}")

    os.makedirs(a.out, exist_ok=True)
    jl = os.path.join(a.out, f"R{a.r}_D{a.d}_s{a.seed}_{a.tag}.jsonl")
    pts = set(probe_steps(a.steps)) | set(
        range(a.eval_every, a.steps + 1, a.eval_every))

    with open(jl, "w") as f:
        f.write(json.dumps(dict(
            kind="meta", corpus_version=ver, vocab_size=len(vocab),
            spec=spec.__dict__, model=mc.__dict__,
            train=dict(total_steps=a.steps, batch_docs=a.batch, lr=a.lr,
                       wd=a.wd, sched=a.sched, warmup=a.warmup,
                       seed=a.seed, eval_every=a.eval_every),
            n_params=model.n_params(), device=dev, amp=str(amp),
            arm="nl_render")) + "\n")

    t0 = time.time()
    it = iter(dl)
    esc = None
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, a.steps, a.lr, a.warmup, sched=a.sched)
        ids, lab, _ = next(it)
        with torch.autocast(dev, dtype=amp, enabled=(dev == "cuda")):
            _, loss = model(ids.to(dev), labels=lab.to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step in pts:
            rd = pr.read(model)
            cd = pr.copy_diag(model)
            if esc is None and cd["copy_acc"] == cd["copy_acc"] \
                    and cd["copy_acc"] >= COPY_FLOOR:
                esc = step
            # 三态判定。ACC_FLOOR 先前定义了却没人用 —— 本次会话第五次同类
            # 死代码（--attr、rsweep、--rsweep 解析后无人读、G2 被算后丢弃）。
            # 位置态这里不判：它需要经验 posCeil，而那个数在 nl_collide.jsonl
            # 里，由 sweep.classify 消费。本文件只报 retr / not。
            retr = (cd["copy_acc"] >= COPY_FLOOR and rd["acc_base"] >= ACC_FLOOR)
            rec = dict(kind="probe", step=step, loss=float(loss),
                       lr=opt.param_groups[0]["lr"], esc=esc,
                       state="retr" if retr else "none",
                       causal=dict(break_rarity=rd), **cd)
            with open(jl, "a") as f:
                f.write(json.dumps(rec) + "\n")
                # 另写一条 eval 记录。sweep.collect 从 kind="eval" 取 acc 与
                # copy_acc（train.py 的 read_run 返回 probe/ev/esc 三个），
                # 只放在 probe 记录里会让它读成 nan。NL 臂大概率用自己的
                # 汇总器（constlr_read 那个形状），但两份都写的成本是零。
                f.write(json.dumps(dict(
                    kind="eval", step=step, acc=rd["acc_base"],
                    copy_acc=cd["copy_acc"], novel_acc=cd["novel_acc"],
                    n_copy=cd["n_copy"], n_novel=cd["n_novel"])) + "\n")
            print(f"step {step:>6}  loss {float(loss):.4f}  "
                  f"acc {rd['acc_base']:.4f}  copy {cd['copy_acc']:.4f}  "
                  f"mass {rd['mass_mean']:.3f}  frac+ {rd['frac_expected']:.3f}  "
                  f"Δ {rd['d_margin']:+.3f}  [{time.time() - t0:.0f}s]")

    torch.save(dict(model=model.state_dict(), cfg=mc.__dict__,
                    version=ver), jl.replace(".jsonl", ".pt"))
    print(f"\n完成。{jl}")
    print("读数判据：mass >= 0.5 才有定义（probe._mass 的同一门），")
    print("copy_acc >= 0.95 且 acc >= 0.99 才是 retrieval 态。")
    print("注意 copy_acc 的候选集是 150 个 adj 而主网格是 512 个 value，")
    print("故 0.95 这个门在 NL 臂上偏松 —— 要在文里注明。")


if __name__ == "__main__":
    main()
