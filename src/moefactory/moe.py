"""标本本体：路由器 → top-k → 容量 → 派发 → 专家 → 合并。

**这是 ArchitectureFactory 那个 MoE 节点的放大镜版。** 那边一个 MoE 层只是
「一种装配」；这里它是全部 —— 因为本项目要看的不是「MoE 好不好」，而是
**路由系统内部发生了什么**。

## 三种路由方式，构成实验的对照骨架

| routing | 选谁 | 门控 | 它存在的理由 |
| --- | --- | --- | --- |
| ``learned`` | 路由器学出来的 | 学出来的 softmax | 被测对象 |
| ``static`` | ``token % n_experts`` | 均匀 | **负载完美均衡，却零分化** —— 证明「均衡 ≠ 分化」 |
| ``random`` | 均匀随机抽 | 均匀 | 地板：连 token 都不看 |

``static`` 这个对照是刻意设计的：它**按构造就是完全均衡的**（每个专家恰好拿到
1/E 的 token，``entropy_norm == 1.0``）。如果只看负载指标，它会是全场最优；
但它的专家每个都要同时学 4 个域的规则，分化度为零、loss 最差。
**「只看均衡会选错配置」这件事，就靠它来钉死。**

## 稀疏派发不是优化，是定义

专家 i 只对「被派发给它」的 token 做前向（``gather_rows`` → 专家 → ``scatter_rows``）。
把每个 token 送进每个专家再乘掩码，得到的是一个**稠密**网络 —— 那样
``active_params`` 就等于总参数量，MoE 的存在理由（同样参数、更少计算）就没了。
真实计算量 = ``top_k / n_experts`` 的专家参数，这一点会被单独记进指标。

## 容量：溢出即丢弃

每个专家每 batch 的容量 = ``capacity_factor × T × top_k / n_experts``。
超出容量的分配按门控值从低到高丢弃。**丢弃的 token 保留残差通路**
（``y = x + Σ``），所以丢弃不会让表示直接归零 —— 这正是真实实现的行为，
也让「容量太紧」表现为「路由白学了」而不是「模型崩了」。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .autograd import (
    Tensor,
    add,
    add_bias,
    constant,
    embed,
    gather1d,
    gather_rows,
    matmul,
    mean_rows,
    mul,
    rmsnorm,
    row_scale,
    scale,
    scatter_rows,
    silu,
    softmax_rows,
    sum_all,
    take_col,
)
from .rng import make_rng
from .schema import MoESpec


# ----------------------------------------------------------------------
# 路由轨迹（观测层的输入）
# ----------------------------------------------------------------------
@dataclass
class RouteTrace:
    """一次前向里发生的全部路由事实。**观测层只读它，不改它。**"""

    n_tokens: int
    gates: list[list[float]]        # [T][E] 门控概率
    assigned: list[list[int]]       # [T] → top_k 选中的专家（容量前）
    kept: list[list[int]]           # [T] → 通过容量的专家（容量后）
    load: list[int]                 # 每个专家**实际**接到的 token 数
    dropped: int                    # 被容量丢弃的分配数
    top1: list[int]                 # 每个 token 的主导专家（丢弃时退化为最高门控）
    dispatched: int                 # 实际做了前向的 (token, expert) 对数
    active_params: int              # 稀疏激活的参数（专家部分按实际派发比例计）
    comm_units: int                 # All-to-All 搬运的元素数

    @property
    def n_assignments(self) -> int:
        return sum(len(a) for a in self.assigned)

    @property
    def dropped_rate(self) -> float:
        n = self.n_assignments
        return (self.dropped / n) if n else 0.0


# ----------------------------------------------------------------------
# 专家
# ----------------------------------------------------------------------
def _size(t: Tensor) -> int:
    """张量的元素个数（1D 或 2D）。"""
    if not t.data:
        return 0
    if isinstance(t.data[0], list):
        return sum(len(row) for row in t.data)  # type: ignore[union-attr]
    return len(t.data)


class Expert:
    """一个专家 = 两层 MLP + SiLU。专家之间**不共享参数**，这是分化的物质基础。"""

    __slots__ = ("W1", "b1", "W2", "b2", "d_model", "d_ff")

    def __init__(self, d_model: int, d_ff: int, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2

    def forward(self, x: Tensor) -> Tensor:
        """``[k,d] -> [k,d]``。"""
        h = silu(add_bias(matmul(x, self.W1), self.b1))
        return add_bias(matmul(h, self.W2), self.b2)

    def params(self, prefix: str) -> list[tuple[str, Tensor]]:
        return [
            (f"{prefix}.W1", self.W1),
            (f"{prefix}.b1", self.b1),
            (f"{prefix}.W2", self.W2),
            (f"{prefix}.b2", self.b2),
        ]

    def flat_weights(self) -> list[float]:
        out: list[float] = []
        for row in self.W1.data:
            out.extend(row)
        for row in self.W2.data:
            out.extend(row)
        return out


# ----------------------------------------------------------------------
# 标本
# ----------------------------------------------------------------------
class MoEModel:
    """一个 MoE 标本：嵌入 → RMSNorm → MoE 块 → 残差 → 输出头。

    只有**一层** MoE 块，是刻意的：多层会把「哪一层的路由塌缩了」搅在一起，
    而本项目要看的正是单层路由系统本身。层数不是本项目的自变量。
    """

    def __init__(
        self,
        spec: MoESpec,
        vocab_size: int,
        seed: int,
        init_scale: float = 0.1,
    ) -> None:
        self.spec = spec.canonical()
        self.vocab_size = vocab_size
        self.seed = seed
        self.init_scale = init_scale
        self.d_model = self.spec.d_model
        self.d_ff = self.spec.d_ff
        self.n_experts = self.spec.n_experts

        rng = make_rng(seed, "init", self.spec.spec_id, vocab_size)

        def mat(rows: int, cols: int, name: str) -> Tensor:
            r = make_rng(seed, "init", self.spec.spec_id, name)
            return constant([[r.gauss(0.0, init_scale) for _ in range(cols)] for _ in range(rows)])

        def vec(n: int, name: str, fill: float | None = None) -> Tensor:
            if fill is not None:
                return constant([fill] * n)
            r = make_rng(seed, "init", self.spec.spec_id, name)
            return constant([r.gauss(0.0, init_scale) for _ in range(n)])

        self.emb = mat(vocab_size, self.d_model, "emb")
        self.norm_w = vec(self.d_model, "norm_w", fill=1.0)
        self.router_W = mat(self.d_model, self.n_experts, "router.W")
        self.router_b = vec(self.n_experts, "router.b", fill=0.0)
        self.head_W = mat(self.d_model, vocab_size, "head.W")
        self.head_b = vec(vocab_size, "head.b", fill=0.0)
        self.experts = [
            Expert(
                self.d_model,
                self.d_ff,
                mat(self.d_model, self.d_ff, f"expert{i}.W1"),
                vec(self.d_ff, f"expert{i}.b1", fill=0.0),
                mat(self.d_ff, self.d_model, f"expert{i}.W2"),
                vec(self.d_model, f"expert{i}.b2", fill=0.0),
            )
            for i in range(self.n_experts)
        ]
        # 保证「同配置必得同标本」：把 rng 消费掉再丢掉，避免上面几个 name 之外的漂移
        del rng

    # -- 参数 --------------------------------------------------------
    def parameters(self) -> list[tuple[str, Tensor]]:
        """**按名字排序**返回，顺序与遍历路径无关（否则梯度累积顺序会变成隐式变量）。"""
        out: list[tuple[str, Tensor]] = [
            ("emb", self.emb),
            ("norm_w", self.norm_w),
            ("router.W", self.router_W),
            ("router.b", self.router_b),
            ("head.W", self.head_W),
            ("head.b", self.head_b),
        ]
        for i, e in enumerate(self.experts):
            out.extend(e.params(f"expert{i}"))
        return sorted(out, key=lambda kv: kv[0])

    @property
    def num_params(self) -> int:
        return sum(_size(t) for _, t in self.parameters())

    @property
    def expert_params(self) -> int:
        return sum(_size(t) for n, t in self.parameters() if n.startswith("expert"))

    def zero_grad(self) -> None:
        for _, t in self.parameters():
            t.zero_grad()

    # -- 路由 --------------------------------------------------------
    def _random_gates(self, n_tokens: int) -> list[list[float]]:
        """``random`` 对照：均匀随机抽 k 个专家，门控均匀。**完全不看 token。**"""
        k = self.spec.top_k
        rng = make_rng(self.seed, "route-random", self.spec.spec_id, n_tokens)
        out = [[0.0] * self.n_experts for _ in range(n_tokens)]
        for t in range(n_tokens):
            for i in rng.sample(range(self.n_experts), k):
                out[t][i] = 1.0 / k
        return out

    def _static_gates(self, inputs: Sequence[int]) -> list[list[float]]:
        """``static`` 对照：``expert = token % n_experts``，门控均匀。

        **按构造完美均衡**（每个专家恰好 1/E 的 token），但不看域的边界，
        于是每个专家都要同时学全部域的规则。
        """
        k = self.spec.top_k
        out = [[0.0] * self.n_experts for _ in inputs]
        for t, tok in enumerate(inputs):
            start = tok % self.n_experts
            for j in range(k):
                out[t][(start + j) % self.n_experts] = 1.0 / k
        return out

    def _top_k(self, gates: Sequence[Sequence[float]]) -> list[list[int]]:
        k = self.spec.top_k
        out: list[list[int]] = []
        for row in gates:
            order = sorted(range(len(row)), key=lambda i: (-row[i], i))
            out.append(sorted(order[:k]))
        return out

    def _apply_capacity(
        self, gates: Sequence[Sequence[float]], assigned: Sequence[Sequence[int]], n_tokens: int
    ) -> tuple[list[list[int]], int]:
        cap = self.spec.capacity_per_expert(n_tokens)
        kept: list[list[int]] = [[] for _ in range(n_tokens)]
        dropped = 0
        for i in range(self.n_experts):
            cand = [t for t in range(n_tokens) if i in assigned[t]]
            if not cand:
                continue
            if math.isinf(cap):
                chosen = cand
            else:
                cand_sorted = sorted(cand, key=lambda t: (-gates[t][i], t))
                limit = max(1, int(math.floor(cap)))
                chosen = cand_sorted[:limit]
                dropped += len(cand) - len(chosen)
            for t in chosen:
                kept[t].append(i)
        return kept, dropped

    # -- 前向 --------------------------------------------------------
    def forward(
        self, inputs: Sequence[int], tags: Sequence[int], *, collect: bool = False
    ) -> tuple[Tensor, RouteTrace | None, Tensor | None]:
        """``([T], [T]) -> [T,V]``。

        **两条信息通路是刻意分开的：**

        - **路由器**看 ``rmsnorm(emb[c] + emb[tag])`` —— 它知道这是哪个任务；
        - **专家**只看 ``emb[c]`` —— 它不知道任务，只能靠「路由器把它派给了谁」。

        这一条是分化能成立的前提。若专家也能看到标签，一个专家就能通吃全部任务，
        分化会退化成装饰（这一点在标定阶段被实测推翻过一次，见 README 附录 C）。

        返回 ``(logits, trace, gates)``：``trace`` 只在 ``collect=True`` 时非空；
        ``gates`` 是**可导的门控张量**，辅助损失要它。对照组的路由不经过路由器，
        所以这里是 ``None`` —— 训练器因此完全不需要 ``if routing == ...`` 分支。
        """
        T = len(inputs)
        if T == 0:
            raise ValueError("空 batch 无法前向")
        if len(tags) != T:
            raise ValueError(f"tags 长度 {len(tags)} 与 inputs 长度 {T} 不一致")

        x_c = embed(self.emb, list(inputs))                       # 专家看到的
        x_r = rmsnorm(add(x_c, embed(self.emb, list(tags))), self.norm_w)  # 路由器看到的

        gates_t: Tensor | None
        if self.spec.routing == "learned":
            logits_r = add_bias(matmul(x_r, self.router_W), self.router_b)
            gates_t = softmax_rows(logits_r)
            gates = gates_t.data
        elif self.spec.routing == "static":
            gates = self._static_gates(inputs)
            gates_t = None
        else:
            gates = self._random_gates(T)
            gates_t = None

        assigned = self._top_k(gates)
        kept, dropped = self._apply_capacity(gates, assigned, T)

        # 稀疏派发：专家 i 只处理派发给它的 token，且只看到内容 token
        acc: Tensor | None = None
        dispatched = 0
        for i in range(self.n_experts):
            idx = [t for t in range(T) if i in kept[t]]
            if not idx:
                continue
            dispatched += len(idx)
            hi = self.experts[i].forward(gather_rows(x_c, idx))
            if gates_t is not None:
                wi = gather1d(take_col(gates_t, i), idx)
            else:
                wi = constant([gates[t][i] for t in idx])
            part = scatter_rows(row_scale(hi, wi), idx, T)
            acc = part if acc is None else add(acc, part)

        y = x_c if acc is None else add(x_c, acc)
        logits = add_bias(matmul(y, self.head_W), self.head_b)

        trace: RouteTrace | None = None
        if collect:
            load = [0] * self.n_experts
            top1: list[int] = []
            for t in range(T):
                if kept[t]:
                    for i in kept[t]:
                        load[i] += 1
                    # 主导专家 = **通过容量者中门控最高**的那个（不是索引最小的）
                    top1.append(max(kept[t], key=lambda i: (gates[t][i], -i)))
                else:
                    top1.append(max(range(self.n_experts), key=lambda i: (gates[t][i], -i)))
            trace = RouteTrace(
                n_tokens=T,
                gates=[list(r) for r in gates],
                assigned=[list(a) for a in assigned],
                kept=[list(kk) for kk in kept],
                load=load,
                dropped=dropped,
                top1=top1,
                dispatched=dispatched,
                active_params=self._active_params(dispatched, T),
                comm_units=dispatched * 2 * self.d_model,
            )
        return logits, trace, gates_t

    def _active_params(self, dispatched: int, n_tokens: int) -> int:
        """稀疏激活的参数量 = 共享部分 + 专家部分 × 实际派发比例。"""
        shared = self.num_params - self.expert_params
        ratio = (dispatched / (n_tokens * self.n_experts)) if n_tokens else 0.0
        return shared + int(round(self.expert_params * ratio))

    # -- 辅助损失 ----------------------------------------------------
    def aux_loss(self, gates_t: Tensor, load: Sequence[int]) -> Tensor:
        """Switch-Transformer 式负载均衡损失：``α · E · Σ_i f_i · P_i``。

        ``f_i`` 是**实际分配比例**（常数，来自派发），``P_i`` 是**平均门控概率**
        （可导）。梯度只经 ``P_i`` 回到路由器 —— 也就是说这条损失的作用是
        「把门控概率推向尚未被用满的专家」，而不是直接搬动 token。
        """
        E = self.n_experts
        total = sum(load)
        f = [(load[i] / total) if total else 0.0 for i in range(E)]
        P = mean_rows(gates_t)                      # [E]
        return scale(sum_all(mul(P, constant(f))), self.spec.aux_weight * E)

    # -- 分化度的物质指标 --------------------------------------------
    def expert_similarity(self) -> float:
        from .schema import mean_pairwise_cosine

        return mean_pairwise_cosine(e.flat_weights() for e in self.experts)

    def param_report(self) -> dict[str, int]:
        return {
            "total": self.num_params,
            "expert": self.expert_params,
            "shared": self.num_params - self.expert_params,
        }


def build_model(spec: MoESpec, vocab_size: int, seed: int, init_scale: float = 0.1) -> MoEModel:
    return MoEModel(spec, vocab_size, seed, init_scale)
