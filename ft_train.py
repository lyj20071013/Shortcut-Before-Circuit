"""Fine-tune a pretrained model on the rendered synthetic task."""
import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch

MIRROR = "https://hf-mirror.com"
COPY_FLOOR = 0.95
ACC_FLOOR = 0.99
MASS_FLOOR = 0.5


def load_for_train(name: str, src: str):
    """fp32 主权重 + autocast 前向。**不要**用 ft_zeroshot.load —— 那个函数
    以 bf16 加载，对推理没问题，对训练会静默失败。

    bf16 只有 8 位尾数，相对精度约 2^-8 = 4e-3。权重典型量级 1e-2，而 lr=2e-5
    的单步更新在梯度 ~1 时给出 2e-3 的相对变化 —— 低于 bf16 的相对精度，更新
    被舍入吃掉。后果是模型看起来完全不动：acc 停在 0.22、Δ 停在零样本值。

    而那与"预训练先验决定了读数"在输出上**无法区分**。这个臂最重要的那个
    结论会被一个数值伪像伪造出来，所以这不是性能选择而是正确性前提。

    显存（0.5B）：fp32 权重 2GB + fp32 梯度 2GB + AdamW 两个态 4GB = 8GB，
    加激活与 logits 后约 12-14GB。64GB 卡上宽裕。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = name
    if src == "ms":
        from ft_tokcheck import MS_ID
        if name in MS_ID:
            from modelscope import snapshot_download
            path = snapshot_download(MS_ID[name])
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    m = m.to(dev)
    m.train()

    # autocast 的 dtype。只用 bf16，不退回 fp16 —— fp16 的指数范围比 fp32 窄，
    # 没有 GradScaler 会下溢，而主网格也不用 scaler（train.py:438-439 在
    # 不支持 bf16 时才退 fp16，那是它自己的取舍）。这里宁可全 fp32 慢一倍。
    amp = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) \
        else None
    print(f"[load] fp32 主权重"
          f"{'，bf16 autocast 前向' if amp else '，无 autocast（bf16 不支持）'}"
          f" —— bf16 主权重会让 2e-5 的更新被舍入吃掉")
    return tok, m, dev, amp


def lr_at(step: int, total: int, lr: float, warmup: int,
          min_frac: float = 0.1) -> float:
    """线性 warmup + 余弦。形状与 train.py:420-427 相同，只有 lr 的量级不同 ——
    这样两臂之间唯一变的是标量而不是调度形状。"""
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


class Probe:
    """离线探针集 -> Δ、mass、acc、copy_acc。

    归一化域的划分与 nl_train.Probe 相同，那个划分是修过一个 bug 的：
      Δ    在 adj 候选集内归一化（log-odds 之差，域外质量不影响它）
      mass 在**全词表**上算（probe._mass 的判据是"候选对占据的质量 < 0.5 则
           该读数无效"，只有全词表上才有意义。若在 adj 集内归一化，模型把
           99% 质量放在 'dull' 上也看不出来 —— 而 ft_zeroshot 实测零样本正是
           那个状态，所以这里不是假想的风险）
    """

    def __init__(self, path: str, dev: str, amp=None):
        self.amp = amp
        z = np.load(path, allow_pickle=False)
        self.z = z
        self.base = torch.tensor(z["base"].astype(np.int64))
        self.edit = torch.tensor(z["edit"].astype(np.int64))
        self.blen = torch.tensor(z["base_len"].astype(np.int64))
        self.elen = torch.tensor(z["edit_len"].astype(np.int64))
        self.apos = torch.tensor(z["answer_pos"].astype(np.int64))
        self.eapos = torch.tensor(z["edit_answer_pos"].astype(np.int64))
        self.ans = torch.tensor(z["answer"].astype(np.int64))
        self.vst = torch.tensor(z["v_star"].astype(np.int64))
        self.cpos = torch.tensor(z["copy_pos"].astype(np.int64))
        self.crep = torch.tensor(z["copy_rep"].astype(np.int64))
        self.pad = int(z["pad_id"])
        self.version = str(z["version"])
        self.model = str(z["model"])
        adj = z["adj_ids"].astype(np.int64)
        self.adj = torch.tensor(adj).to(dev)
        self.adj_cpu = torch.tensor(adj)
        self.aidx = {int(v): i for i, v in enumerate(adj)}
        self.n = len(self.base)
        self.dev = dev

        miss = [int(v) for v in list(self.ans) + list(self.vst)
                if int(v) not in self.aidx]
        if miss:
            raise RuntimeError(
                f"{len(miss)} 个 answer/v_star 不在 adj_ids 内。ft_data 与本次"
                f"运行的 ADJ 顺序不一致 —— 读数会错位到另一个 token 上。")

    def _rows(self, model, ids, lens, pos, bs: int):
        """answer_pos 上的全词表 logits 行。

        右填充 + causal，故真实末位不受右侧 pad 影响；但仍传 attention_mask，
        因为 HF 的实现对 pad 位置的处理不保证与 causal mask 等价。
        """
        out = []
        for i in range(0, len(ids), bs):
            b = ids[i:i + bs].to(self.dev)
            L = lens[i:i + bs].to(self.dev)
            p = pos[i:i + bs].to(self.dev)
            am = (torch.arange(b.shape[1], device=self.dev)[None, :] < L[:, None])
            with torch.no_grad(), torch.autocast(
                    self.dev, dtype=self.amp, enabled=self.amp is not None):
                o = model(input_ids=b, attention_mask=am.long())
            # .float() 在 autocast 之外：读数的算术全部 fp32，与主网格一致
            # （probe 那边 logits 也是 fp32，且 Δ 是两个小量之差，bf16 的
            # 4e-3 相对精度会淹没 0.01 量级的 Δ）。
            sel = o.logits[torch.arange(len(b), device=self.dev), p - 1]
            out.append(sel.float().cpu())
            del o
        return torch.cat(out)

    @torch.no_grad()
    def read(self, model, bs: int = 4) -> dict:
        model.eval()
        lb = self._rows(model, self.base, self.blen, self.apos, bs)
        le = self._rows(model, self.edit, self.elen, self.eapos, bs)
        model.train()

        ia = torch.tensor([self.aidx[int(v)] for v in self.ans])
        iv = torch.tensor([self.aidx[int(v)] for v in self.vst])
        r = torch.arange(len(ia))

        pb = torch.log_softmax(lb[:, self.adj_cpu], -1)
        pe = torch.log_softmax(le[:, self.adj_cpu], -1)
        d = ((pe[r, iv] - pe[r, ia]) - (pb[r, iv] - pb[r, ia])).numpy()

        se = torch.softmax(le, -1)
        mass = (se[r, self.vst] + se[r, self.ans]).numpy()

        # FT-specific summaries: d_median and frac_expected restrict to
        # edit pairs with candidate mass >= MASS_FLOOR. The *_ungated
        # fields retain all valid edit pairs. These populations differ;
        # select fields using the population specified for each analysis.
        ok = mass >= MASS_FLOOR
        dg = d[ok]
        nan = float("nan")
        return dict(n=len(d), n_gated=int(ok.sum()),
                    # FT mass-restricted summaries.
                    d_median=float(np.median(dg)) if len(dg) else nan,
                    frac_expected=float(np.mean(dg > 0)) if len(dg) else nan,
                    # Unrestricted summaries over all valid FT edit pairs.
                    d_median_ungated=float(np.median(d)),
                    d_margin_ungated=float(np.mean(d)),
                    frac_ungated=float(np.mean(d > 0)),
                    mass_mean=float(np.mean(mass)),
                    mass_ok=float(np.mean(ok)),
                    acc_base=float((pb.argmax(-1) == ia).float().mean()),
                    acc_full=float((lb.argmax(-1) == self.ans).float().mean()),
                    d_all=[float(x) for x in d],
                    mass_all=[float(x) for x in mass])

    @torch.no_grad()
    def copy_diag(self, model, bs: int = 2) -> dict:
        """重复出现的值 token 上的 argmax 准确率。位置来自 ft_data 的 BPE 映射。"""
        model.eval()
        hit = tot = hit_n = tot_n = 0
        for i in range(0, self.n, bs):
            b = self.base[i:i + bs].to(self.dev)
            L = self.blen[i:i + bs].to(self.dev)
            am = (torch.arange(b.shape[1], device=self.dev)[None, :] < L[:, None])
            with torch.autocast(self.dev, dtype=self.amp,
                                enabled=self.amp is not None):
                o = model(input_ids=b, attention_mask=am.long())
            # argmax 对 dtype 不敏感（只比大小），故这里不必升 fp32
            lg = o.logits[:, :, self.adj].argmax(-1).cpu()
            del o
            for j in range(len(b)):
                p, rp = self.cpos[i + j], self.crep[i + j]
                m = p >= 1
                if not bool(m.any()):
                    continue
                pp, rr = p[m], rp[m]
                tgt = self.base[i + j][pp]
                want = torch.tensor([self.aidx.get(int(t), -1) for t in tgt])
                keep = want >= 0
                if not bool(keep.any()):
                    continue
                ok = (lg[j, (pp[keep] - 1)] == want[keep])
                isr = (rr[keep] == 1)
                hit += int(ok[isr].sum())
                tot += int(isr.sum())
                hit_n += int(ok[~isr].sum())
                tot_n += int((~isr).sum())
        model.train()
        nan = float("nan")
        return dict(copy_acc=hit / tot if tot else nan, n_copy=tot,
                    novel_acc=hit_n / tot_n if tot_n else nan, n_novel=tot_n)


class Stream:
    """流式训练集，与 NLStream 同构但产出 BPE id。

    corpus seed = 1000*seed，与主网格 train.py:255 的 1000*seed + w 一致，只是
    这里没有 DataLoader worker（生成在主进程，w 恒为 0）。轮次递进 +7919。
    探针集用 777000，与 seed<=9 的训练流不交（见 ft_data）。

    单进程生成的代价：每步要 batch*accum = 32 篇，每篇一次 render + 一次
    tok.encode。若这成为瓶颈（GPU 利用率低），改成 DataLoader 并把 w 加回
    偏移式 —— 那时偏移必须是 1000*seed + w 而不是 seed + 1000 + w，后者会让
    相邻 seed 共享文档流（本会话在 NLStream 上犯过这个错，跨 seed sigma 会被
    压低，而 sigma 是这个臂要产出的量之一）。
    """

    def __init__(self, tok, r, d, seed, spec, ctx_len):
        self.tok, self.r, self.d, self.seed = tok, r, d, seed
        self.spec, self.ctx = spec, ctx_len

    def batches(self, bs: int):
        import dataclasses

        from generator import generate_corpus
        from nl_corpus import nl_corpus_cfg, tokenize
        from nl_render import render
        from vocab import Vocab

        from ft_data import canon

        v = Vocab(self.spec)
        cfg = nl_corpus_cfg(self.r, self.d, 1000 * self.seed)
        buf = []
        while True:
            for doc in generate_corpus(v, cfg, 4096):
                # canon(tokens) 而不是 rd["text"]：实测两者不等，而探针集用的
                # 是 canon。训练与探针必须喂同一个约定的串，否则模型在一种
                # 表面上训练、在另一种上被读 —— 那个差异会被算进 Δ。
                ids = self.tok.encode(canon(render(doc, v, tokenize)["tokens"]),
                                      add_special_tokens=False)
                if len(ids) > self.ctx:
                    continue
                buf.append(ids)
                if len(buf) >= bs:
                    yield buf
                    buf = []
            cfg = dataclasses.replace(cfg, seed=cfg.seed + 7919)


def collate(seqs, pad: int, dev: str):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        am[i, :len(s)] = 1
    lab = ids.masked_fill(am == 0, -100)
    return ids.to(dev), am.to(dev), lab.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--src", default="hf", choices=["hf", "ms"])
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--data", default="ft_data")
    ap.add_argument("--out", default="runs_ft")
    ap.add_argument("--tag", default="ft")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--d", type=int, default=8)
    a = ap.parse_args()
    if a.src == "hf" and not a.no_mirror:
        os.environ.setdefault("HF_ENDPOINT", MIRROR)

    from ft_pool import ft_spec, install, verify_install
    from nl_generator import values_needed

    meta = json.load(open(os.path.join(a.data, "meta.json")))
    tok, model, dev, amp = load_for_train(a.model, a.src)
    ver = install(tok, need=values_needed(55, 3), model_name=a.model)
    verify_install()
    if ver != meta["version"]:
        raise SystemExit(
            f"版本不符：ft_data 是 {meta['version']}，本次是 {ver}。"
            f"val_id -> adj 的映射不同，读数会错位。重跑 ft_data。")
    spec = ft_spec(meta["spec"]["ctx_len"])

    pr = Probe(os.path.join(a.data, f"probe_R{a.r}_D{a.d}.npz"), dev, amp)
    print(f"{a.model}  探针 {pr.n} 篇  候选 {len(pr.adj)}  版本 {pr.version}")
    print(f"lr {a.lr}  batch {a.batch}x{a.accum}={a.batch * a.accum}"
          f"  steps {a.steps}  seed {a.seed}")

    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)

    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters()
               if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd),
                             dict(params=nodecay, weight_decay=0.0)],
                            lr=a.lr, betas=(0.9, 0.95))

    os.makedirs(a.out, exist_ok=True)
    jl = os.path.join(a.out, f"R{a.r}_D{a.d}_s{a.seed}_{a.tag}.jsonl")
    with open(jl, "w") as f:
        f.write(json.dumps(dict(
            kind="meta", arm="ft", version=ver, model=a.model,
            spec=meta["spec"], n_values=meta["n_values"],
            train=dict(lr=a.lr, warmup=a.warmup, wd=a.wd, steps=a.steps,
                       batch=a.batch, accum=a.accum,
                       eff_batch=a.batch * a.accum, seed=a.seed),
            note="seed 只控数据顺序与 dropout；初始权重是预训练常量，"
                 "故对照 App.~dseed 而非主网格")) + "\n")

    # step 0：零样本基线。这个点是配对问题的锚，必须记。
    rd = pr.read(model)
    cd = pr.copy_diag(model)
    zs = dict(step=0, **rd, **cd)
    # Print mass-restricted FT summaries; parentheses contain unrestricted diagnostics.
    # n_gated 必须显示 —— massOK 低时门控量的样本很小，而那决定它可不可信。
    print(f"\nstep {0:>5}  acc {rd['acc_base']:.4f}  copy {cd['copy_acc']:.4f}"
          f"  mass {rd['mass_mean']:.3f}  massOK {rd['mass_ok']:.3f}"
          f"  n_gated {rd['n_gated']}"
          f"  frac+ {rd['frac_expected']:.3f}({rd['frac_ungated']:.3f})"
          f"  Δmed {rd['d_median']:+.3f}({rd['d_median_ungated']:+.3f})")
    with open(jl, "a") as f:
        f.write(json.dumps(dict(kind="probe", zeroshot=True, step=0,
                                causal=dict(break_rarity=rd), **cd)) + "\n")
    if rd["mass_ok"] < 0.05:
        print("  注意：零样本 mass 几乎全部低于 0.5，Δ 在这一点无定义。这是")
        print("  预期的（ft_zeroshot 实测质量集中在 'dull' 等先验词上），但若")
        print("  第一个训练探针点仍如此，就要停下 —— 见下面的 mass 门。")

    stream = Stream(tok, a.r, a.d, a.seed, spec, spec.ctx_len)
    gen = stream.batches(a.batch)
    t0 = time.time()
    esc = None
    hit_acc = None
    # 形成后 band 要这个。末尾的汇总先前引用了一个不存在的 probes，那是一个
    # 跑完一小时之后才抛的 NameError。
    probes: List[dict] = []
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, a.steps, a.lr, a.warmup)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.accum):
            ids, am, lab = collate(next(gen), pr.pad, dev)
            with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                out = model(input_ids=ids, attention_mask=am, labels=lab)
            # backward 在 autocast 之外：梯度累到 fp32 主权重的 .grad 上。
            # 不用 GradScaler —— bf16 的指数范围与 fp32 相同，不会下溢。
            (out.loss / a.accum).backward()
            tot += float(out.loss)
            del out
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % a.probe_every == 0 or step == a.steps:
            rd = pr.read(model)
            cd = pr.copy_diag(model)
            if esc is None and cd["copy_acc"] == cd["copy_acc"] \
                    and cd["copy_acc"] >= COPY_FLOOR:
                esc = step
            if hit_acc is None and rd["acc_base"] >= ACC_FLOOR:
                hit_acc = step
            print(f"step {step:>5}  loss {tot / a.accum:.4f}"
                  f"  acc {rd['acc_base']:.4f}  copy {cd['copy_acc']:.4f}"
                  f"  mass {rd['mass_mean']:.3f}  massOK {rd['mass_ok']:.3f}"
                  f"  n_gated {rd['n_gated']}"
                  f"  frac+ {rd['frac_expected']:.3f}({rd['frac_ungated']:.3f})"
                  f"  Δmed {rd['d_median']:+.3f}({rd['d_median_ungated']:+.3f})"
                  f"  [{time.time() - t0:.0f}s]")
            probes.append(dict(step=step, **{
                k: rd[k] for k in ("frac_expected", "d_median", "mass_ok",
                                   "n_gated", "acc_base", "frac_ungated",
                                   "d_median_ungated")}))
            with open(jl, "a") as f:
                f.write(json.dumps(dict(
                    kind="probe", step=step, loss=tot / a.accum,
                    lr=opt.param_groups[0]["lr"], esc=esc, acc_at=hit_acc,
                    causal=dict(break_rarity=rd), **cd)) + "\n")

            # mass 门在**固定步数**上查，不在第一个探针点上查。
            #
            # 先前写的是 step == a.probe_every，而那个条件与 --probe-every 耦合：
            # 用 --probe-every 50 时它在 step 50 触发，而那时权重才动了 50 步、
            # mass 本来就该低 —— run 会被无理由杀掉。实测 --probe-every 250 时
            # step 250 的 mass_ok 是 0.425，所以 250 是一个有依据的位置。
            if step == max(250, a.probe_every) and rd["mass_ok"] < 0.10:
                print("\n  停：第一个探针点 mass_ok < 0.10，Δ 在这个 run 上")
                print("  大概不会有定义。可能的原因：lr 太小（权重几乎没动）、")
                print("  或候选对不是模型的两个主要候选。先查 acc 有没有从")
                print(f"  {zs['acc_base']:.3f} 动过 —— 现在是 {rd['acc_base']:.3f}。")
                break

    # 存末态权重，不只是 cfg。
    #
    # nl_train 存了 state_dict，所以它那个未门控 bug 可以从 .pt 重算而不必重跑
    # 六小时。这里先前只存 cfg + version —— 若事后发现读数口径有问题，一小时
    # 的 run 无法挽回。0.5B fp32 约 2GB/seed，三个 seed 6GB，值这个价。
    torch.save(dict(model=model.state_dict(),
                    cfg=dict(lr=a.lr, seed=a.seed, steps=a.steps,
                             probe_every=a.probe_every),
                    version=ver, model_name=a.model),
               jl.replace(".jsonl", ".pt"))
    print(f"\n完成 {jl}")
    print(f"零样本 acc {zs['acc_base']:.4f} -> 末次 {rd['acc_base']:.4f}")

    # 形成后的 band，而不是只报末点。
    #
    # 理由：实测 acc 在 step 250 就到 0.99，故 2000 步里 1750 步都在形成之后，
    # 而 App.~drift 量过形成后漂移在 69 个 run 上的 within-run sigma 达 0.303
    # （第 1870 行）。只报末点会把一个漂移量当成读数 —— 那正是 App.~constlr
    # 第 1981-1987 行说的"single-checkpoint reading would have reversed its
    # conclusion"。
    post = [p for p in probes if hit_acc is not None and p["step"] >= hit_acc]
    if post:
        fr = [p["frac_expected"] for p in post
              if p["frac_expected"] == p["frac_expected"]]
        dm = [p["d_median"] for p in post
              if p["d_median"] == p["d_median"]]
        print(f"\n形成后 {len(post)} 个探针点（step >= {hit_acc}）")
        if fr:
            print(f"  frac+ 门控   {min(fr):.3f} .. {max(fr):.3f}"
                  f"   band {max(fr) - min(fr):.3f}   末点 {fr[-1]:.3f}")
        if dm:
            print(f"  Δ 中位 门控  {min(dm):+.3f} .. {max(dm):+.3f}"
                  f"   末点 {dm[-1]:+.3f}")
        print(f"  massOK       {min(p['mass_ok'] for p in post):.3f} .. "
              f"{max(p['mass_ok'] for p in post):.3f}")
        print(f"\n  这个 band 要与 App.~drift 的 0.303 上界并列报告。band 若")
        print("  与跨 seed 的分散同量级，则该臂分不开两者 —— 那是 App.~constlr")
        print("  第 1968-1971 行对恒定 lr 臂做的同一个判断。")

    print(f"\n零样本 frac+ {zs['frac_ungated']:.4f}（未门控，massOK "
          f"{zs['mass_ok']:.3f} 下门控值无意义）")
    if post and fr:
        print(f"配对位移     {fr[-1] - zs['frac_ungated']:+.4f}"
              f"   （门控末点 - 零样本未门控）")
        print("  注意两端口径不同：零样本时 94% 文档不过 mass 门，故只有未门控")
        print("  值存在。这个位移因此是方向性的，不是一个可并列的数值。")

    if hit_acc is None:
        print(f"\n  acc 从未达到 {ACC_FLOOR}。这个 run 不可解释 —— 分不清")
        print("  '先验决定了读数'与'预算太短、什么都没动'。加步数或提 lr，")
        print("  不要把它算进 sigma。")
    else:
        print(f"\n  acc 在 step {hit_acc} 达到 {ACC_FLOOR}，"
              f"copy 在 step {esc} 过 {COPY_FLOOR}。")
        if hit_acc <= a.probe_every:
            print(f"  形成发生在第一个探针点或更早。--probe-every "
                  f"{a.probe_every} 对这个臂太粗 —— 形成过程本身没有被采到，")
            print("  而那是 from-scratch 臂能给出而这个臂给不出的东西。若要")
            print("  它，用 --probe-every 25 重跑（探针成本约 20 秒/点）。")


if __name__ == "__main__":
    main()
