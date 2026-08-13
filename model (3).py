"""Decoder-only Transformer。25–51M 参数区间，单卡 90 run。

四个实质性设计选择，须在正文声明：

位置编码用 RoPE，不用 learned absolute。候选规则里有 position（复制固定
token 偏移处的值），而位置编码的选择直接决定这条规则有多容易学：learned
absolute 让模型能直接索引绝对位置，会人为抬高 position 的占比；RoPE 把
相对距离作为归纳偏置，而 ΔD 本身就是相对量。压制或抬高任一候选规则都会
让相图偏向伪结论，RoPE 是与自变量匹配且贴近现代 LM 实际配置的一侧。
附录应在少数格子上用 learned absolute 复跑作稳健性检查。

不 tie embedding。unembedding 独立是回路归因（§7）的前提：tied 时
"读入某 value" 与 "写出某 value" 共用同一方向，logit attribution 无法区分
搬运与生成。代价是多 2M 参数，可以接受。

无 dropout：语料是无限流式生成，每篇文档重新采样 (e,a)->v 绑定，
过拟合的对象不存在，dropout 只会拖慢收敛并给激活归因引入噪声。

QK-norm（q/k 各一个 per-head RMSNorm）。固定 init_std 在 d_model=256
下让 attn logit 起步 std ≈ 0.1，softmax 近乎均匀、梯度稀薄，induction
回路形成慢一个数量级（见 bench 的 1key）。QK-norm 把起步 logit 钉在
~1.0 且给出上界，不需要 soft cap。这是对所有格子一致施加的结构选择，
不与自变量交互；附录用 --no-qk-norm 在少数格子复跑。

初始化尺度随宽度走：std = 1/sqrt(d_model)，而非固定 0.02。残差写出
路径（attn.out / mlp.down）额外乘 1/sqrt(2·n_layer)。
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelCfg:
    vocab_size: int
    ctx_len: int = 1024
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_mlp: int = 1376          # ≈ 8/3·d_model，SwiGLU 三矩阵后与 4x GELU 等参
    rope_base: float = 10000.0
    init_std: Optional[float] = None      # None -> 1/sqrt(d_model)
    init_emb_std: Optional[float] = None  # None -> 1/sqrt(d_model)
    qk_norm: bool = True
    qk_norm_gain: float = 2.0   # QK-norm 增益初值。attn logit 起步 std ≈ gain^2；
                            # gain 从 1 长到 3 在 AdamW 下约需 2/lr 步，
                            # 这是 sharpening 的速率瓶颈，故留作可调
    tie_embed: bool = False
    pos: str = "rope"          # rope | nope | learned，诊断用

    def __post_init__(self):
        assert self.d_model % self.n_head == 0
        if self.init_std is None:
            self.init_std = 1.0 / math.sqrt(self.d_model)
        if self.init_emb_std is None:
            self.init_emb_std = 1.0 / math.sqrt(self.d_model)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                        + self.eps).type_as(x)


def _rope_tables(ctx: int, d_head: int, base: float, device):
    """始终 fp32 建表，用时再 cast：autocast 下 emb 出 fp32、Linear 出 bf16，
    按 x.dtype 建表会让 q/k 与 v 的 dtype 分叉。"""
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(ctx, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def _rope(x, cos, sin):
    # x: (B, H, T, D)
    T = x.shape[-2]
    c = cos[:T].to(x.dtype).unsqueeze(0).unsqueeze(0)
    s = sin[:T].to(x.dtype).unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelCfg, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.use_rope = (cfg.pos == "rope")
        self.qn = RMSNorm(cfg.d_head) if cfg.qk_norm else None
        self.kn = RMSNorm(cfg.d_head) if cfg.qk_norm else None

    def forward(self, x, cos, sin, cache: Optional[dict] = None):
        B, T, C = x.shape
        H, D = self.cfg.n_head, self.cfg.d_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        if self.qn is not None:               # per-head，归一化在 RoPE 之前
            q, k = self.qn(q), self.kn(k)
        if self.use_rope:
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        if cache is not None and cache.get("want_pattern"):
            # 显式路径，只在归因/诊断时走：SDPA 不返回注意力权重
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            sc = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(D)
            cache[f"qk_std.{self.layer}"] = float(sc.masked_select(mask).std())
            att = sc.masked_fill(~mask, float("-inf")).softmax(-1)
            cache[f"pattern.{self.layer}"] = att.detach()
            y = att.type_as(v) @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        if cache is not None:
            cache[f"z.{self.layer}"] = y.detach()
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.gate = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.down = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg, layer: int):
        super().__init__()
        self.layer = layer
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg, layer)
        self.n2, self.mlp = RMSNorm(cfg.d_model), MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        if cache is not None:
            cache[f"resid_pre.{self.layer}"] = x.detach()
        a = self.attn(self.n1(x), cos, sin, cache)
        if cache is not None:
            cache[f"attn_out.{self.layer}"] = a.detach()
        x = x + a
        m = self.mlp(self.n2(x))
        if cache is not None:
            cache[f"mlp_out.{self.layer}"] = m.detach()
        return x + m


class LM(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # wpe 必须在统一初始化之前建好：旧版建在 apply(_init) 之后，
        # 拿的是默认 N(0,1)，位置嵌入比 token 嵌入大 ~50x，token 身份被淹。
        self.wpe = nn.Embedding(cfg.ctx_len, cfg.d_model) \
            if cfg.pos == "learned" else None
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.nf = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._cos = self._sin = None
        self._init_weights()
        if cfg.tie_embed:
            self.head.weight = self.emb.weight

    def _init_weights(self):
        c = self.cfg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=c.init_std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=c.init_emb_std)
        if self.wpe is not None:              # 位置信号起步弱于 token 信号
            nn.init.normal_(self.wpe.weight, std=c.init_emb_std * 0.5)
        if c.qk_norm:
            for m in self.modules():
                if isinstance(m, Attention):
                    nn.init.constant_(m.qn.w, c.qk_norm_gain)
                    nn.init.constant_(m.kn.w, c.qk_norm_gain)
        for n, p in self.named_parameters():  # 残差写出路径按深度缩放
            if n.endswith(("out.weight", "down.weight")):
                nn.init.normal_(p, std=c.init_std / math.sqrt(2 * c.n_layer))

    def _tables(self, device):
        if self._cos is None or self._cos.device != device:
            self._cos, self._sin = _rope_tables(self.cfg.ctx_len, self.cfg.d_head,
                                                self.cfg.rope_base, device)
        return self._cos, self._sin

    def forward(self, idx: torch.Tensor, labels: Optional[torch.Tensor] = None,
                cache: Optional[dict] = None):
        """labels 用 -100 标记不计 loss 的位置（padding）。返回 (logits, loss)。
        loss 是全 token LM loss：更像预训练，且梯度信号远多于只训 answer。
        准确率另算，只在 answer token 上报（见 train.evaluate）。"""
        x = self.emb(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        cos, sin = self._tables(x.device)
        for b in self.blocks:
            x = b(x, cos, sin, cache)
        x = self.nf(x)
        if cache is not None:
            cache["resid_final"] = x.detach()
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss

    def param_groups(self, wd: float):
        """embedding 不做 weight decay：稀疏梯度下逐步衰减所有行，与小初始化
        叠加会把 token 身份继续压小。tie_embed 时 head 与 emb 同一张量，
        自动落入 nodecay（这一差异只出现在 tie 消融里，须在附录声明）。"""
        no_wd = {id(self.emb.weight)}
        if self.wpe is not None:
            no_wd.add(id(self.wpe.weight))
        decay, nodecay = [], []
        for p in self.parameters():
            (nodecay if (p.dim() < 2 or id(p) in no_wd) else decay).append(p)
        return [dict(params=decay, weight_decay=wd),
                dict(params=nodecay, weight_decay=0.0)]

    def n_params(self, embed: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not embed:
            n -= self.emb.weight.numel()
            if self.head.weight is not self.emb.weight:
                n -= self.head.weight.numel()
        return n