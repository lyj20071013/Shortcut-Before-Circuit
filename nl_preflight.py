"""Check the rendered-language training and probe interfaces."""
import json
import os
import sys

import numpy as np
import torch

FAIL = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "nl_data"
    r, d = 3, 8

    # ---- 1. 纯算术：seed 分流。0 秒，而它是上一轮那个 bug 的判据 ----
    # 旧式 seed+1000+w 下 seed0 的 w3 与 seed1 的 w2 相同，但两者的 w0
    # 仍不同，所以"首篇不同"这种检查抓不到它。必须查整个 worker 集合。
    sets = {s: {1000 * s + w for w in range(4)} for s in range(10)}
    bad = [(a, b) for a in range(10) for b in range(a + 1, 10)
           if sets[a] & sets[b]]
    ck("十个 seed 的 worker 流两两不交", not bad, f"重叠 {bad[:3]}")

    PROBE_SEED = 777000
    reach = {1000 * s + w + 7919 * k
             for s in range(10) for w in range(4) for k in range(400)}
    ck("PROBE_SEED 不在训练流的可达集内", PROBE_SEED not in reach,
       f"probe={PROBE_SEED}")

    # ---- 2. 模块导入。抓语法错误（本会话我引入过一个 // 注释）----
    import nl_collide, nl_corpus, nl_gates, nl_generator, nl_render, nl_train
    ck("四个模块导入", True, f"len(ADJ)={len(nl_generator.ADJ)}")

    spec = nl_corpus.nl_spec()
    need = nl_generator.values_needed(55, 3)
    ck("ADJ 池够大", len(nl_generator.ADJ) >= spec.n_values,
       f"n_values={spec.n_values} <= len(ADJ)={len(nl_generator.ADJ)}")
    ck("n_values 不会耗尽 _ValueDraw", spec.n_values >= need,
       f"{spec.n_values} >= {need}")
    nl_render.check_pools(spec)
    ck("check_pools 通过", True)

    # 跨池不交。四个交集非空会让读数位置的 token 有歧义。
    A = set(nl_generator.ADJ)
    af = {x.split()[0] for x in nl_generator.ATTR}
    an = {x.split()[1] for x in nl_generator.ATTR}
    ck("ADJ 与其他池不交",
       not (A & af or A & an or A & set(nl_generator.NOUN)
            or A & set(nl_generator.NAMES)))

    # ---- 3. 文件：名字、版本、id 域 ----
    vp = os.path.join(data, "vocab.json")
    ck("vocab.json 存在", os.path.exists(vp), vp)
    if not os.path.exists(vp):
        return done()
    vj = json.load(open(vp))
    vocab, ver = vj["vocab"], vj["version"]
    ck("词表大小与 spec 相容", len(vocab) > spec.n_values,
       f"{len(vocab)} tokens, version={ver}")

    pp = os.path.join(data, f"probe_R{r}_D{d}.npz")
    ck("探针集用新文件名（无 _s{seed}）", os.path.exists(pp), pp)
    old = os.path.join(data, f"probe_R{r}_D{d}_s0.npz")
    ck("旧探针集已删除", not os.path.exists(old),
       "还在则可能被误用" if os.path.exists(old) else "")
    if not os.path.exists(pp):
        return done()

    z = np.load(pp)
    ck("探针集版本 == 词表版本", str(z["version"]) == ver,
       f"{str(z['version'])} vs {ver}")

    adj = z["adj_ids"]
    ck("adj_ids 数量 == n_values", len(adj) == spec.n_values,
       f"{len(adj)} vs {spec.n_values}")
    ck("adj_ids 全在词表值域内", bool(((adj >= 1) & (adj < len(vocab))).all()))
    ck("adj_ids 无重复", len(set(adj.tolist())) == len(adj))

    ai = set(adj.tolist())
    ck("answer 全在 adj_ids 内", set(z["answer"].tolist()) <= ai,
       f"越界 {len(set(z['answer'].tolist()) - ai)} 个")
    ck("v_star 全在 adj_ids 内", set(z["v_star"].tolist()) <= ai,
       f"越界 {len(set(z['v_star'].tolist()) - ai)} 个")
    ck("answer != v_star 逐篇成立",
       bool((z["answer"] != z["v_star"]).all()))

    cp, bl = z["copy_pos"], z["base_len"]
    ck("copy_pos 哨兵是 -1 且无位置 0",
       bool(((cp == -1) | (cp >= 1)).all()))
    ck("copy_pos 不越过 base_len",
       bool(((cp == -1) | (cp < bl[:, None])).all()))
    ck("copy_pos 非空", int((cp >= 1).sum()) > 0,
       f"n_copy_pos={int((cp >= 1).sum())}")

    b, e = z["base"], z["edit"]
    nb = (b != 0).sum(1)
    ne = (e != 0).sum(1)
    ck("G1 base/edit token 数逐篇相等", bool((nb == ne).all()),
       f"worst_diff={int(np.abs(nb - ne).max())}")
    ck("answer_pos 在 base 长度内",
       bool((z["answer_pos"] < bl).all()))

    # ---- 4. 训练流不含探针文档。上一轮我引入过这个污染 ----
    base_set = {tuple(x[x != 0].tolist()) for x in b}
    hit = 0
    for s in (0, 1, 2):
        st = nl_corpus.NLStream(r, d, s, vocab, spec)
        it = iter(st)
        for _ in range(40):
            toks, _ap = next(it)
            if tuple(toks) in base_set:
                hit += 1
    ck("训练流前 120 篇与探针集不交", hit == 0, f"重合 {hit} 篇")

    # ---- 5. 一次前反向 + 两个读数。30 秒，抓 device/shape/dtype ----
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from model import LM, ModelCfg
    mc = ModelCfg(vocab_size=len(vocab), ctx_len=spec.ctx_len)
    m = LM(mc).to(dev)
    ck("模型非 embedding 参数与主网格一致",
       m.n_params(False) == 25306624, f"{m.n_params(False):,}")

    st = nl_corpus.NLStream(r, d, 0, vocab, spec)
    it = iter(st)
    batch = [next(it) for _ in range(8)]
    ids, lab, _ = nl_train.collate(batch, spec.ctx_len)
    _, loss = m(ids.to(dev), labels=lab.to(dev))
    loss.backward()
    ck("前向 + 反向", torch.isfinite(loss).item(), f"loss={float(loss):.4f}")

    pr = nl_train.Probe(pp, dev)
    rd = pr.read(m)
    ck("Probe.read 跑通", np.isfinite(rd["d_margin"]),
       f"Δ={rd['d_margin']:+.4f} mass={rd['mass_mean']:.4f} "
       f"acc={rd['acc_base']:.4f}")
    cd = pr.copy_diag(m)
    ck("copy_diag 跑通且 n_copy 合理", cd["n_copy"] > 500,
       f"n_copy={cd['n_copy']} n_novel={cd['n_novel']} "
       f"copy={cd['copy_acc']:.4f}")

    # 未训练模型的 mass 应当很低。若这里就接近 1，说明 mass 算在了
    # adj 集内部而不是全词表 —— 那个 bug 会让 mass 门变成空的。
    ck("未训练模型的 mass 远低于 0.5（全词表归一化的证据）",
       rd["mass_mean"] < 0.1, f"mass={rd['mass_mean']:.5f}")
    done()


def done():
    print("\n" + "=" * 60)
    if FAIL:
        print(f"{len(FAIL)} 项失败，不要跑 nl_train：")
        for f in FAIL:
            print(f"  - {f}")
    else:
        print("全过。可以跑三个 seed。")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
