"""模型可学性阶梯。不含论文语义，只回答：model.py 的 LM 能不能做 induction，
2-token 键比 1-token 键难多少，词表规模的代价有多大。

三级，全部每序列重采样键->值映射（与正式语料一致：绑定不可记忆，
必须在上下文内匹配，不能靠参数记忆）：
  L1 1key : ... a v ... a ?     -> v    标准 induction
  L2 2key : ... e t v ... e t ? -> v    正式语料的复制位就是这个
  L3 2key_far : 同 L2，序列更长，键与目标的距离更远

键的首次出现在整条序列上随机散布（不是前 n_keys 个 slot 挤在一起）：
否则 repeat 位在序列中的位置是固定的、候选值也永远聚在开头，模型能靠
位置先验作弊，而正式语料没有这个结构。

三个必须一起看的读数：

  loss 的可动范围很窄。key token 的熵 ln(n_keys) 不可约。header 打印
  捷径/地板两条参考线，loss 只在这两条之间有意义。

  repeat nll 的第一个平台是 ln(n_keys)，不是 chance。模型会先学会"注意力
  铺到所有 value 位、抄出在场值的边际分布"。停在这条线上意味着 OV 复制
  通路已成形、QK 匹配还没有，是相变前的正常状态，不是 bug。

  prec 比 ind* 重要。ind* 是落在"同键前次出现的 value 位"的注意力质量，
  prec 是它占全部 value 位质量的比例。存在两套解：尖锐 induction 头
  （ind*→1.0），以及"弱偏置 + 读出端放大"（ind*≈0.14 也能到 99% 准确率，
  因为 softmax 前 OV/unembed 会放大 0.14 vs 0.07 的差）。用 ind* 的绝对
  阈值判断回路会在高准确率下给出错误否定 —— 这一点同样适用于正文 §7 的
  回路归因：那里必须用 logit attribution 和消融，不能用 attention mass。

  qk 是 softmax 前 attn logit 的 std。QK-norm 下起步 ≈ gain²，上界
  gain²·sqrt(d_head)，须显著大于 ln(ctx_len) 才可能形成尖锐匹配。
  关掉 QK-norm 时它会跑到 1e3 量级且锁死在无偏移的 key 匹配上。

判读顺序：
  自检失败                   => 数据构造或度量位对不上，与模型无关。
  acc 达标                   => 通过，看 t50 外推正式 run 的步数。
  卡在捷径平台、prec 不涨     => sharpening 没启动，调 --qk-gain。
  prec 在涨、acc 未达标       => 预算不足，加 --steps。
  prec 恒在 chance            => model.py 有实现 bug。

诊断 run 默认恒定 lr、wd=0：cosine 衰减会在相变刚起时掐掉 lr，而无限流
数据下 wd 不提供正则收益，只在恒定 lr 下把权重乘性收缩掉。
相变时间是重尾的，配置决策要用 --seeds 3 以上，单 seed 的倍数差不算证据。
"""
import argparse
import math
import statistics
import time

import torch

from model import LM, ModelCfg


def make_batch(B, n_keys, n_occ, key_len, v_key, v_attr, v_val, device, gen):
    """返回 (ids, repeat_mask, stride)。repeat_mask 标记该次出现的键此前是否已见过
    —— 只有这些位置的值可由上下文预测，是唯一有意义的度量位。

    首次出现的槽位随机散布：slot 0 必为首次，其余 n_keys-1 个首次在
    1..n_occ-1 里随机取；非首次槽位从"已引入的键"里均匀抽，保证每个 repeat
    位都有前驱、且 repeat 位置不是序列位置的确定函数。"""
    stride = key_len + 2                      # key(+attr) + val + SEP
    assert n_occ >= n_keys
    if key_len == 2:
        assert n_keys % 2 == 0 and v_attr >= 2
        ne = n_keys // 2
        ent = torch.rand(B, v_key, generator=gen, device=device).argsort(-1)[:, :ne] + 1
        ent = ent.repeat_interleave(2, dim=1)
        at = torch.rand(B, ne, v_attr, generator=gen, device=device).argsort(-1)[..., :2]
        at = at.reshape(B, n_keys) + 1 + v_key
    else:
        ent = torch.rand(B, v_key, generator=gen, device=device).argsort(-1)[:, :n_keys] + 1
        at = None
    vlo = 1 + v_key + v_attr
    val = torch.randint(0, v_val, (B, n_keys), generator=gen, device=device) + vlo

    first = torch.zeros(B, n_occ, dtype=torch.bool, device=device)
    first[:, 0] = True
    if n_keys > 1:
        pick = torch.rand(B, n_occ - 1, generator=gen,
                          device=device).argsort(-1)[:, :n_keys - 1] + 1
        first.scatter_(1, pick, True)
    order = torch.rand(B, n_keys, generator=gen, device=device).argsort(-1)
    n_intro = first.long().cumsum(-1)               # 含当前槽，>=1
    rank_first = (n_intro - 1).clamp_min(0)
    u = torch.rand(B, n_occ, generator=gen, device=device)
    rank_rep = (u * n_intro.float()).long().clamp_(max=n_keys - 1)
    occ = torch.gather(order, 1, torch.where(first, rank_first, rank_rep))
    rep = ~first

    ids = torch.zeros(B, n_occ * stride, dtype=torch.long, device=device)
    ids[:, 0::stride] = torch.gather(ent, 1, occ)
    if key_len == 2:
        ids[:, 1::stride] = torch.gather(at, 1, occ)
    ids[:, key_len::stride] = torch.gather(val, 1, occ)
    return ids, rep, stride


def selfcheck(n_keys, n_occ, key_len, v_key, v_attr, v_val, device, gen):
    """oracle 自检：只用 ids 与 repeat_mask 重放"回到上一次同键出现、抄它后面
    那个 token"这条规则。它必须在全部 repeat 位上 100% 命中，且 measure 取的
    logit 位必须正好预测这些位。任何一条不成立，学习曲线都不用看。"""
    ids, rep, stride = make_batch(8, n_keys, n_occ, key_len, v_key, v_attr,
                                  v_val, device, gen)
    B = ids.shape[0]
    hit = tot = 0
    rp = []
    for b in range(B):
        last = {}
        for i in range(n_occ):
            p = i * stride
            key = tuple(int(t) for t in ids[b, p:p + key_len])
            tgt = int(ids[b, p + key_len])
            if bool(rep[b, i]):
                tot += 1
                hit += int(last.get(key) == tgt)
                rp.append(i)
            else:
                assert key not in last, "repeat_mask 与实际首次出现不一致"
            last[key] = tgt
    assert tot > 0, "没有 repeat 位，度量为空"
    assert hit == tot, f"oracle 只有 {hit}/{tot}：键->值绑定在序列内不唯一"
    pos = torch.arange(n_occ, device=device) * stride + key_len
    vlo = 1 + v_key + v_attr
    assert bool(((ids[:, pos] >= vlo) & (ids[:, pos] < vlo + v_val)).all()), \
        "度量位不是 value token，logits 偏移错了"
    print(f"  自检: oracle {hit}/{tot} = 1.000  度量位对齐 OK  "
          f"repeat 位/序列 = {tot // B}  首个 repeat slot 分布 "
          f"min={min(rp)} 中位={int(statistics.median(rp))}")


@torch.no_grad()
def measure(model, ids, rep, stride, key_len, vlo, v_val):
    logits, _ = model(ids)
    pos = torch.arange(rep.shape[1], device=ids.device) * stride + key_len
    sel = logits[:, pos - 1, vlo:vlo + v_val].float()
    tgt = ids[:, pos] - vlo
    lp = torch.log_softmax(sel, -1)
    nll = -lp.gather(-1, tgt[..., None]).squeeze(-1)
    ok = (sel.argmax(-1) == tgt)
    return (ok[rep].float().mean().item(), nll[rep].mean().item(),
            ok[~rep].float().mean().item(), nll[~rep].mean().item())


# ---------------- 回路探针 ----------------

def _occ_masks(ids, n_occ, stride, key_len, device):
    """对每个 repeat 出现返回：
      mv   = 该键此前所有出现的 value 位（induction 落点。同键同序列内值恒定，
             故按集合求和而非只看最近一次）
      mk   = 该键此前所有出现的键末位（纯 token 匹配，是 induction 的中间态：
             匹配上了但 +1 偏移还没学出来）
      mval = query 之前所有 value 位（precision 的分母）
    query 取 p+key_len-1，即 logits 预测该 value 的那个位置。"""
    B, T = ids.shape
    bi, qp, mv, mk = [], [], [], []
    for b in range(B):
        seen = {}
        for i in range(n_occ):
            p = i * stride
            key = tuple(int(t) for t in ids[b, p:p + key_len])
            if key in seen:
                v = torch.zeros(T, dtype=torch.bool)
                k = torch.zeros(T, dtype=torch.bool)
                for pp in seen[key]:
                    v[pp + key_len] = True
                    k[pp + key_len - 1] = True
                bi.append(b)
                qp.append(p + key_len - 1)
                mv.append(v)
                mk.append(k)
            seen.setdefault(key, []).append(p)
    bi = torch.tensor(bi, device=device)
    qp = torch.tensor(qp, device=device)
    mv = torch.stack(mv).to(device)
    mk = torch.stack(mk).to(device)
    ar = torch.arange(T, device=device)
    valpos = ((ar % stride) == key_len)
    mval = valpos[None, :] & (ar[None, :] < qp[:, None])
    return bi, qp, mv, mk, mval


@torch.no_grad()
def task_probe(model, n_keys, n_occ, key_len, v_key, v_attr, v_val,
               device, gen, B=8):
    """在训练分布上量回路（不是随机序列：训练数据是严格 stride 周期的，
    模型的注意力与位置结构耦合，离分布探针会给错误否定）。"""
    ids, _, stride = make_batch(B, n_keys, n_occ, key_len, v_key, v_attr,
                                v_val, device, gen)
    cache = {"want_pattern": True}
    model.eval()
    model(ids, cache=cache)
    model.train()
    bi, qp, mv, mk, mval = _occ_masks(ids, n_occ, stride, key_len, device)
    T = ids.shape[1]
    t = torch.arange(1, T, device=device)
    nv = mv.sum(-1).float()
    ch_mass = float((nv / T).mean())
    ch_prec = float((nv / mval.sum(-1).float().clamp_min(1)).mean())
    rows = []
    for l in range(model.cfg.n_layer):
        att = cache[f"pattern.{l}"]                  # (B, H, T, T)
        a = att[bi, :, qp, :]                        # (N, H, T)
        ind = (a * mv[:, None, :]).sum(-1)
        bag = (a * mval[:, None, :]).sum(-1)
        prec = ind / bag.clamp_min(1e-6)
        km = (a * mk[:, None, :]).sum(-1)
        prev = att[:, :, t, t - 1].mean(-1).mean(0)
        rows.append((ind.mean(0).tolist(), prec.mean(0).tolist(),
                     bag.mean(0).tolist(), km.mean(0).tolist(),
                     prev.tolist(), cache[f"qk_std.{l}"]))
    return rows, ch_mass, ch_prec


def probe_best(rows, min_bag=0.05):
    """(最强 ind* 质量, 最强 prec)。prec 只在该头确有 value 指向时才可信，
    bag 低于 min_bag 的头不参与，否则分母噪声会给出虚高的 prec。"""
    bi = bp = 0.0
    for ind, prec, bag, *_ in rows:
        bi = max(bi, max(ind))
        for p, b in zip(prec, bag):
            if b >= min_bag:
                bp = max(bp, p)
    return bi, bp


def print_task_probe(rows, ch_mass, ch_prec):
    for l, (ind, prec, bag, km, prev, qk) in enumerate(rows):
        print(f"    L{l}: ind*=" + " ".join(f"{v:.2f}" for v in ind)
              + " | prec=" + " ".join(f"{v:.2f}" for v in prec)
              + " | kmatch=" + " ".join(f"{v:.2f}" for v in km)
              + " | prev=" + " ".join(f"{v:.2f}" for v in prev)
              + f" | qk={qk:.2f}")
    bi, bp = probe_best(rows)
    print(f"    最强 ind*={bi:.3f} (chance {ch_mass:.3f})  "
          f"最强 prec={bp:.3f} (chance {ch_prec:.3f})")
    return bi, bp


# ---------------- 训练 ----------------

def lr_at(step, steps, lr, warmup, sched, min_frac=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    if sched == "const":
        return lr
    t = (step - warmup) / max(1, steps - warmup)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def run(level, steps, B, lr, n_keys, n_occ, v_key, v_attr, v_val,
        d_model, n_layer, n_head, device, pos="rope", warmup=100, seed=0,
        qk_norm=True, qk_gain=2.0, wd=0.0, sched="const", stop_at=0.95,
        quiet=False):
    key_len = 1 if level == "1key" else 2
    if key_len == 1:
        v_attr = 0
    stride = key_len + 2
    T = n_occ * stride
    vocab = 1 + v_key + v_attr + v_val
    vlo = 1 + v_key + v_attr
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)

    mc = ModelCfg(vocab_size=vocab, ctx_len=T, d_model=d_model,
                  n_layer=n_layer, n_head=n_head, d_mlp=d_model * 8 // 3,
                  pos=pos, qk_norm=qk_norm, qk_norm_gain=qk_gain)
    model = LM(mc).to(device)
    opt = torch.optim.AdamW(model.param_groups(wd), lr=lr, betas=(0.9, 0.95))

    p_nov = n_keys / n_occ
    key_h = math.log(n_keys)
    chance = math.log(v_val)
    bag_nll = key_h
    floor = (key_h + p_nov * chance) / stride
    short = (key_h + (1 - p_nov) * bag_nll + p_nov * chance) / stride
    ceil = qk_gain ** 2 * math.sqrt(mc.d_head) if qk_norm else float("inf")
    print(f"\n[{level}/{pos}/s{seed}] vocab={vocab} T={T} keys={n_keys} "
          f"occ={n_occ} params={model.n_params()/1e6:.1f}M lr={lr} "
          f"sched={sched} wd={wd} qk_norm={qk_norm} qk_gain={qk_gain} "
          f"init_std={mc.init_std:.4f} tok/step={B*T/1e3:.0f}k")
    print(f"  参考线: repeat nll chance={chance:.2f} 捷径平台={bag_nll:.2f} | "
          f"loss 捷径={short:.3f} 地板={floor:.3f} | "
          f"qk 上界={ceil:.1f} 需要≳ln(T)={math.log(T):.1f}")
    if ceil < 2 * math.log(T):
        print("  ⚠ qk 上界不足 2·ln(T)：sharpening 可能无法完成，调大 --qk-gain")
    selfcheck(n_keys, n_occ, key_len, v_key, v_attr, v_val, device, gen)
    if not quiet:
        print("  初始回路:")
        print_task_probe(*task_probe(model, n_keys, n_occ, key_len, v_key,
                                     v_attr, v_val, device, gen))
    t0, best, best_step, t50 = time.time(), 0.0, 0, None
    peak_ind = peak_prec = 0.0
    every = max(1, steps // 12)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, steps, lr, warmup, sched)
        ids, rep, _ = make_batch(B, n_keys, n_occ, key_len, v_key, v_attr,
                                 v_val, device, gen)
        _, loss = model(ids, ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % every == 0 or step == steps:
            model.eval()
            eids, erep, _ = make_batch(256, n_keys, n_occ, key_len, v_key,
                                       v_attr, v_val, device, gen)
            ra, rn, na, nn_ = measure(model, eids, erep, stride, key_len, vlo, v_val)
            model.train()
            rows, cm, cp = task_probe(model, n_keys, n_occ, key_len, v_key,
                                      v_attr, v_val, device, gen)
            it, pr = probe_best(rows)
            qk = max(r[5] for r in rows)
            peak_ind, peak_prec = max(peak_ind, it), max(peak_prec, pr)
            if ra > best:
                best, best_step = ra, step
            if t50 is None and ra > 0.5:
                t50 = step
            flag = "" if rn < bag_nll - 0.05 else "  <捷径平台"
            print(f"  step {step:>5} loss {float(loss):.3f} "
                  f"tok {step*B*T/1e6:.0f}M | acc {ra:.3f} nll {rn:.3f}{flag} | "
                  f"novel {na:.3f}/{nn_:.2f} | ind* {it:.3f} prec {pr:.3f} "
                  f"qk {qk:.2f}")
            if ra > stop_at:
                print(f"  提前停止：acc {ra:.3f} > {stop_at}")
                break
    print("  末态回路:")
    print_task_probe(*task_probe(model, n_keys, n_occ, key_len, v_key,
                                 v_attr, v_val, device, gen))
    print(f"  [{level}/s{seed}] best acc={best:.3f} @ {best_step}  "
          f"t50={t50}  peak ind*={peak_ind:.3f} prec={peak_prec:.3f}  "
          f"{(time.time()-t0)/60:.1f} 分钟  {'PASS' if best > 0.9 else 'FAIL'}")
    return dict(acc=best, t50=t50, ind=peak_ind, prec=peak_prec)


def verdict(level, rs):
    acc = statistics.median(r["acc"] for r in rs)
    prec = max(r["prec"] for r in rs)
    t50s = [r["t50"] for r in rs if r["t50"]]
    n_ok = sum(r["acc"] > 0.9 for r in rs)
    line = (f"{level}: acc 中位={acc:.3f}  {n_ok}/{len(rs)} PASS  "
            f"peak prec={prec:.3f}")
    if t50s:
        line += f"  t50 {min(t50s)}–{max(t50s)} 中位 {int(statistics.median(t50s))}"
    print(line)
    if acc > 0.9:
        return True
    if prec > 0.5:
        print(f"  => 匹配已成形（prec {prec:.2f}）但 acc 未达标：预算不足，"
              f"加 --steps，不要改 model.py。")
    elif prec > 0.15:
        print(f"  => 匹配在成形中（prec {prec:.2f}）：相变还没走完，先加 --steps；"
              f"若 prec 长期不动再调 --qk-gain。")
    else:
        print(f"  => prec 贴近 chance：sharpening 没启动。按序试 "
              f"--qk-gain 3 / --qk-gain 4；仍无效才怀疑 model.py。")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+",
                    default=["1key", "2key", "2key_far"],
                    choices=["1key", "2key", "2key_far"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=1,
                    help="相变时间重尾，配置决策至少 3")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3,
                    help="非单调：4e-3 在 1key 上比 2e-3 差（见 runs/e2）")
    ap.add_argument("--sched", default="const", choices=["const", "cos"],
                    help="诊断用 const；cos 只在外推正式 run 步数时用")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--v-val", type=int, default=256,
                    help="值池大小。扫这个参数即可测'词表规模是否是瓶颈'")
    ap.add_argument("--v-key", type=int, default=256)
    ap.add_argument("--n-keys", type=int, default=None,
                    help="覆盖键数。捷径平台 = ln(n_keys)")
    ap.add_argument("--n-occ", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--pos", default="rope", choices=["rope", "nope", "learned"])
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--no-qk-norm", action="store_true", help="消融：关掉 QK-norm")
    ap.add_argument("--qk-gain", type=float, default=ModelCfg.qk_norm_gain)
    ap.add_argument("--stop-at", type=float, default=0.95)
    ap.add_argument("--easy", action="store_true",
                    help="最快回路：小词表、键更密。回归测试先跑这个")
    a = ap.parse_args()
    if a.easy:
        a.v_key, a.v_val = 64, 64
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for lv in a.levels:
        if a.easy:
            nk, no = (8, 48) if lv != "2key_far" else (16, 96)
        else:
            nk, no = (16, 40) if lv != "2key_far" else (32, 100)
        nk, no = (a.n_keys or nk), (a.n_occ or no)
        res[lv] = [run(lv, a.steps, a.batch, a.lr, nk, no, a.v_key, 8, a.v_val,
                       a.d_model, a.n_layer, a.n_head, dev, pos=a.pos,
                       warmup=a.warmup, seed=s, qk_norm=not a.no_qk_norm,
                       qk_gain=a.qk_gain, wd=a.wd, sched=a.sched,
                       stop_at=a.stop_at, quiet=(s > 0))
                   for s in range(a.seeds)]
    print("\n汇总:")
    ok = {lv: verdict(lv, rs) for lv, rs in res.items()}
    if all(ok.values()) and len(ok) == 3:
        print("=> 阶梯全通：实现没问题，剩下的是规模/算力问题。")
    elif ok.get("1key") and not ok.get("2key", True):
        print("=> L1 通过 L2 失败：2-token 键匹配是瓶颈，缩 n_entities 或加课程。")


if __name__ == "__main__":
    main()