"""整数 token 直出，不经过 BPE。一个值 = 一个 token，是读答案分布的前提。"""
from config import LangSpec


class Vocab:
    def __init__(self, spec: LangSpec):
        self.spec = spec
        self.PAD = 0
        self.SEP = 1        # .
        self.QUERY = 2      # ?
        self.ARROW = 3      # ->
        self.UPD = 4        # 更新标记
        self.TIME0 = 5
        self.ENT0 = self.TIME0 + spec.n_time_idx
        self.ATTR0 = self.ENT0 + spec.n_entities
        self.VAL0 = self.ATTR0 + spec.n_attrs
        self.size = self.VAL0 + spec.n_values

    def ent(self, i: int) -> int:
        assert 0 <= i < self.spec.n_entities
        return self.ENT0 + i

    def attr(self, i: int) -> int:
        assert 0 <= i < self.spec.n_attrs
        return self.ATTR0 + i

    def val(self, i: int) -> int:
        assert 0 <= i < self.spec.n_values
        return self.VAL0 + i

    def time(self, k: int) -> int:
        assert 0 <= k < self.spec.n_time_idx
        return self.TIME0 + k

    def is_val(self, t: int) -> bool:
        return self.VAL0 <= t < self.size

    def val_index(self, t: int) -> int:
        assert self.is_val(t)
        return t - self.VAL0

    def decode(self, t: int) -> str:
        if t == self.PAD: return "<pad>"
        if t == self.SEP: return "."
        if t == self.QUERY: return "?"
        if t == self.ARROW: return "->"
        if t == self.UPD: return "upd"
        if t < self.ENT0: return f"@{t - self.TIME0}"
        if t < self.ATTR0: return f"e{t - self.ENT0}"
        if t < self.VAL0: return f"a{t - self.ATTR0}"
        return f"v{t - self.VAL0}"

    def render(self, toks) -> str:
        return " ".join(self.decode(t) for t in toks)