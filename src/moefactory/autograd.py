"""微型张量 autograd 引擎（零第三方依赖）。

**为什么手写。** 本项目要观测「token → expert 的动态路由究竟发生了什么」。
路由是一个**离散选择**过程，只有在梯度、初始化、算子实现都能逐位复核时，
「负载倾斜是路由器的选择还是初始化噪声」这类问题才答得清。用框架建模型，
得到的只是「某个版本在某一天的行为」。生产当然该用 PyTorch，故留了扩展点。

**与 ArchitectureFactory 的关系。** 本项目是它的一个节点纵向钻穿，因此
**共用同一套数值约定**（嵌套 list 张量、每算子自带 backward 闭包、内积一律
``math.sumprod``）。但这里**只保留 MoE 用得到的算子**：没有注意力、没有 SSM、
没有多头切分 —— 那些在 MoE 里一次也用不到，留着只是负债（见 README 附录 C）。

**设计取舍。**

- 张量 = 嵌套 ``list``（1D: ``list[float]``；2D: ``list[list[float]]``）。
- 反向传播 = 每个算子自带 backward 闭包 + 反向拓扑遍历。
- **内积一律走 :func:`math.sumprod`**（CPython 3.12+ 标准库，C 实现）。MoE 里
  专家前向是绝对热点（默认配置下占总 MACs 的 ~70%），这一处决定了整条流水线
  能不能在 1 分钟内跑完 24 个配置。
- 每个算子都配「解析梯度 == 数值梯度」的测试（``tests/test_autograd.py``）——
  梯度写错**不会报错**，只会让训练静默变差。

约定：``Tensor.data`` 与 ``Tensor.grad`` 同形；叶子用 :func:`constant` 造。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

#: C 速度的内积。``math.sumprod`` 自 CPython 3.12 起提供。
_dot = math.sumprod

__all__ = [
    "Tensor",
    "constant",
    "ones",
    "zeros",
    "matmul",
    "add",
    "add_bias",
    "mul",
    "scale",
    "row_scale",
    "take_col",
    "gather_rows",
    "gather1d",
    "scatter_rows",
    "sum_all",
    "sum_rows",
    "mean_rows",
    "silu",
    "embed",
    "rmsnorm",
    "softmax_rows",
    "softmax_cross_entropy",
]


# ----------------------------------------------------------------------
# 形状工具
# ----------------------------------------------------------------------
def _is2d(x: Sequence[Any]) -> bool:
    return bool(x) and isinstance(x[0], list)


def _zeros_like(x: Sequence[Any]) -> list[Any]:
    if _is2d(x):
        return [[0.0] * len(row) for row in x]  # type: ignore[union-attr]
    return [0.0] * len(x)


def _ones_like(x: Sequence[Any]) -> list[Any]:
    if _is2d(x):
        return [[1.0] * len(row) for row in x]  # type: ignore[union-attr]
    return [1.0] * len(x)


def _shape(x: Sequence[Any]) -> tuple[int, ...]:
    if _is2d(x):
        return (len(x), len(x[0]))  # type: ignore[arg-type]
    return (len(x),)


# ----------------------------------------------------------------------
# 张量
# ----------------------------------------------------------------------
class Tensor:
    """一个值 + 它的梯度 + 它从哪些父张量算来。"""

    __slots__ = ("data", "grad", "_parents", "_backward")

    def __init__(
        self,
        data: list[Any],
        parents: tuple["Tensor", ...] = (),
        backward: Callable[[list[Any]], None] | None = None,
    ) -> None:
        self.data = data
        self.grad = _zeros_like(data)
        self._parents = parents
        self._backward = backward

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape(self.data)

    def zero_grad(self) -> None:
        self.grad = _zeros_like(self.data)

    def backward(self) -> None:
        """从本张量出发做反向传播；本张量视作标量输出（grad=1）。"""
        topo: list[Tensor] = []
        seen: set[int] = set()

        def build(t: Tensor) -> None:
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                build(p)
            topo.append(t)

        build(self)
        self.grad = _ones_like(self.data)
        for t in reversed(topo):
            if t._backward is not None:
                t._backward(t.grad)

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        return f"Tensor(shape={self.shape})"


def constant(data: list[Any]) -> Tensor:
    """叶子张量（不接收梯度）。"""
    return Tensor(data)


def ones(n: int) -> Tensor:
    return Tensor([1.0] * n)


def zeros(n: int) -> Tensor:
    return Tensor([0.0] * n)


# ----------------------------------------------------------------------
# 线性代数
# ----------------------------------------------------------------------
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """``[m,k] @ [k,n] -> [m,n]``。前向与反向**都只由 sumprod 内积组成**。"""
    A, B = a.data, b.data
    m, k = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    if k != k2:
        raise ValueError(f"matmul 维度不匹配：{_shape(A)} @ {_shape(B)}")
    bcols = list(zip(*B))
    C = [[_dot(A[i], bcols[j]) for j in range(n)] for i in range(m)]
    out = Tensor(C, (a, b))

    def bw(g: list[Any]) -> None:
        acols = list(zip(*A))
        gcols = list(zip(*g))
        for i in range(m):
            gi = g[i]
            gai = a.grad[i]
            for t in range(k):
                gai[t] += _dot(gi, B[t])
        for t in range(k):
            at = acols[t]
            gbt = b.grad[t]
            for j in range(n):
                gbt[j] += _dot(at, gcols[j])

    out._backward = bw
    return out


def add(a: Tensor, b: Tensor) -> Tensor:
    """同形相加（2D 或 1D）。"""
    A, B = a.data, b.data
    if _is2d(A):
        m, n = len(A), len(A[0])
        C = [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]
    else:
        C = [A[i] + B[i] for i in range(len(A))]
    out = Tensor(C, (a, b))

    def bw(g: list[Any]) -> None:
        if _is2d(A):
            m, n = len(A), len(A[0])
            for i in range(m):
                for j in range(n):
                    a.grad[i][j] += g[i][j]
                    b.grad[i][j] += g[i][j]
        else:
            for i in range(len(A)):
                a.grad[i] += g[i]
                b.grad[i] += g[i]

    out._backward = bw
    return out


def add_bias(a: Tensor, b: Tensor) -> Tensor:
    """``[m,n] + [n]``（每行加同一个偏置）。"""
    A, B = a.data, b.data
    m, n = len(A), len(A[0])
    C = [[A[i][j] + B[j] for j in range(n)] for i in range(m)]
    out = Tensor(C, (a, b))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            gi = g[i]
            for j in range(n):
                a.grad[i][j] += gi[j]
                b.grad[j] += gi[j]

    out._backward = bw
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    """同形逐元素相乘（1D 或 2D）。门控加权就靠它。"""
    A, B = a.data, b.data
    if _is2d(A):
        m, n = len(A), len(A[0])
        C = [[A[i][j] * B[i][j] for j in range(n)] for i in range(m)]
    else:
        C = [A[i] * B[i] for i in range(len(A))]
    out = Tensor(C, (a, b))

    def bw(g: list[Any]) -> None:
        if _is2d(A):
            m, n = len(A), len(A[0])
            for i in range(m):
                for j in range(n):
                    a.grad[i][j] += g[i][j] * B[i][j]
                    b.grad[i][j] += g[i][j] * A[i][j]
        else:
            for i in range(len(A)):
                a.grad[i] += g[i] * B[i]
                b.grad[i] += g[i] * A[i]

    out._backward = bw
    return out


def scale(a: Tensor, s: float) -> Tensor:
    A = a.data
    if _is2d(A):
        C = [[v * s for v in row] for row in A]
    else:
        C = [v * s for v in A]
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        if _is2d(A):
            for i, row in enumerate(g):
                for j, v in enumerate(row):
                    a.grad[i][j] += v * s
        else:
            for i, v in enumerate(g):
                a.grad[i] += v * s

    out._backward = bw
    return out


def row_scale(a: Tensor, v: Tensor) -> Tensor:
    """``[m,n]`` 的第 i 行乘标量 ``v[i]``。**合并（combine）就靠它**。"""
    A, V = a.data, v.data
    m, n = len(A), len(A[0])
    C = [[A[i][j] * V[i] for j in range(n)] for i in range(m)]
    out = Tensor(C, (a, v))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            vi = V[i]
            s = 0.0
            for j in range(n):
                a.grad[i][j] += g[i][j] * vi
                s += g[i][j] * A[i][j]
            v.grad[i] += s

    out._backward = bw
    return out


def take_col(a: Tensor, j: int) -> Tensor:
    """取第 j 列 → ``[m]``。取「第 j 个专家对所有 token 的门控值」用。"""
    A = a.data
    m = len(A)
    C = [A[i][j] for i in range(m)]
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            a.grad[i][j] += g[i]

    out._backward = bw
    return out


# ----------------------------------------------------------------------
# 派发（dispatch）：MoE 的「只算被选中的 token」靠这三个算子
# ----------------------------------------------------------------------
def gather_rows(a: Tensor, idxs: Sequence[int]) -> Tensor:
    """按行取行：``[m,n] , [k] -> [k,n]``（允许重复、允许乱序）。

    **MoE 稀疏激活的实现基础。** 专家 i 只对「被派发给它」的那几个 token 做前向，
    所以真实计算量是 ``top_k / n_experts`` 而不是全部。一个把每个 token 都送进
    每个专家的「MoE」不是 MoE —— 稀疏激活本身就是它的存在理由。
    """
    A = a.data
    n = len(A[0]) if A else 0
    C = [list(A[i]) for i in idxs]
    out = Tensor(C, (a,))
    k = len(idxs)

    def bw(g: list[Any]) -> None:
        for r in range(k):
            gi = a.grad[idxs[r]]
            gr = g[r]
            for j in range(n):
                gi[j] += gr[j]

    out._backward = bw
    return out


def gather1d(a: Tensor, idxs: Sequence[int]) -> Tensor:
    """按索引取元素：``[m] , [k] -> [k]``。取被派发 token 的门控值用。"""
    A = a.data
    C = [A[i] for i in idxs]
    out = Tensor(C, (a,))
    k = len(idxs)

    def bw(g: list[Any]) -> None:
        for r in range(k):
            a.grad[idxs[r]] += g[r]

    out._backward = bw
    return out


def scatter_rows(a: Tensor, idxs: Sequence[int], m: int) -> Tensor:
    """把 ``[k,n]`` 的行放回 ``[m,n]`` 的指定位置，其余填零（gather_rows 的逆）。

    注意是**累加**语义：同一个下标出现多次时结果相加 —— 那正是 ``top_k`` 里
    同一个 token 被多个专家处理时需要的行为。
    """
    A = a.data
    k = len(A)
    n = len(A[0]) if A else 0
    C = [[0.0] * n for _ in range(m)]
    for r in range(k):
        row = C[idxs[r]]
        ar = A[r]
        for j in range(n):
            row[j] += ar[j]
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        for r in range(k):
            gr = a.grad[r]
            grow = g[idxs[r]]
            for j in range(n):
                gr[j] += grow[j]

    out._backward = bw
    return out


# ----------------------------------------------------------------------
# 归约
# ----------------------------------------------------------------------
def sum_all(a: Tensor) -> Tensor:
    """全元素求和 → 标量张量 ``[[s]]``。辅助损失要它。"""
    A = a.data
    s = 0.0
    if _is2d(A):
        for row in A:
            s += math.fsum(row)
    else:
        s = math.fsum(A)
    out = Tensor([[s]], (a,))

    def bw(g: list[Any]) -> None:
        gs = g[0][0]
        if _is2d(A):
            for i in range(len(A)):
                for j in range(len(A[0])):
                    a.grad[i][j] += gs
        else:
            for i in range(len(A)):
                a.grad[i] += gs

    out._backward = bw
    return out


def sum_rows(a: Tensor) -> Tensor:
    """按行求和 → ``[n]``。"""
    A = a.data
    m, n = len(A), len(A[0])
    C = [0.0] * n
    for i in range(m):
        for j in range(n):
            C[j] += A[i][j]
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            for j in range(n):
                a.grad[i][j] += g[j]

    out._backward = bw
    return out


def mean_rows(a: Tensor) -> Tensor:
    """按行求均值 → ``[n]``。``P_i``（专家 i 的平均门控概率）就靠它。"""
    A = a.data
    m, n = len(A), len(A[0])
    C = [0.0] * n
    for i in range(m):
        for j in range(n):
            C[j] += A[i][j]
    C = [v / m for v in C]
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            for j in range(n):
                a.grad[i][j] += g[j] / m

    out._backward = bw
    return out


# ----------------------------------------------------------------------
# 激活 / 归一化 / 嵌入
# ----------------------------------------------------------------------
def _sigmoid_scalar(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def silu(a: Tensor) -> Tensor:
    """``x * sigmoid(x)``。专家内部的非线性。"""
    A = a.data
    if _is2d(A):
        S = [[_sigmoid_scalar(v) for v in row] for row in A]
        C = [[A[i][j] * S[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    else:
        S = [_sigmoid_scalar(v) for v in A]
        C = [A[i] * S[i] for i in range(len(A))]
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        if _is2d(A):
            for i in range(len(A)):
                for j in range(len(A[0])):
                    s = S[i][j]
                    d = s * (1.0 + A[i][j] * (1.0 - s))
                    a.grad[i][j] += g[i][j] * d
        else:
            for i in range(len(A)):
                s = S[i]
                d = s * (1.0 + A[i] * (1.0 - s))
                a.grad[i] += g[i] * d

    out._backward = bw
    return out


def rmsnorm(a: Tensor, w: Tensor, eps: float = 1e-6) -> Tensor:
    """逐行 RMS 归一化（可学习缩放 ``w``）。MoE 块前置，稳住路由 logits 的尺度。"""
    A, W = a.data, w.data
    m, n = len(A), len(A[0])
    inv = [0.0] * m
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        ms = math.fsum(v * v for v in A[i]) / n
        s = 1.0 / math.sqrt(ms + eps)
        inv[i] = s
        row = A[i]
        for j in range(n):
            C[i][j] = row[j] * s * W[j]
    out = Tensor(C, (a, w))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            row = A[i]
            s = inv[i]
            # d/dx [ x*s*w ] ，s 依赖 x ⇒ 需要投影项
            dot_gs = 0.0
            for j in range(n):
                dot_gs += g[i][j] * W[j] * row[j]
            coef = s * s * s / n
            for j in range(n):
                a.grad[i][j] += g[i][j] * s * W[j] - coef * dot_gs * row[j]
                w.grad[j] += g[i][j] * row[j] * s

    out._backward = bw
    return out


def embed(weight: Tensor, ids: Sequence[int]) -> Tensor:
    """按 id 取行：``[V,d] , [L] -> [L,d]``。"""
    W = weight.data
    C = [list(W[i]) for i in ids]
    out = Tensor(C, (weight,))
    L = len(ids)
    d = len(W[0])

    def bw(g: list[Any]) -> None:
        for r in range(L):
            wi = weight.grad[ids[r]]
            for j in range(d):
                wi[j] += g[r][j]

    out._backward = bw
    return out


# ----------------------------------------------------------------------
# 路由 / 损失
# ----------------------------------------------------------------------
def softmax_rows(a: Tensor) -> Tensor:
    """逐行 softmax。路由器 logits → 门控概率。"""
    A = a.data
    m, n = len(A), len(A[0])
    P = [[0.0] * n for _ in range(m)]
    for i in range(m):
        row = A[i]
        mx = max(row)
        ex = [math.exp(v - mx) for v in row]
        s = math.fsum(ex)
        if s == 0.0:
            s = 1.0
        P[i] = [e / s for e in ex]
    out = Tensor(P, (a,))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            p = P[i]
            dot = math.fsum(g[i][j] * p[j] for j in range(n))
            for j in range(n):
                a.grad[i][j] += p[j] * (g[i][j] - dot)

    out._backward = bw
    return out


def softmax_cross_entropy(logits: Tensor, targets: Sequence[int]) -> Tensor:
    """逐行 softmax + NLL，返回**标量**张量（形状 ``[[loss]]``，已取平均）。"""
    A = logits.data
    m, V = len(A), len(A[0])
    probs: list[list[float]] = []
    loss = 0.0
    for i in range(m):
        row = A[i]
        mx = max(row)
        ex = [math.exp(v - mx) for v in row]
        s = math.fsum(ex)
        p = [e / s for e in ex]
        probs.append(p)
        loss += -math.log(max(p[targets[i]], 1e-12))
    loss /= m
    out = Tensor([[loss]], (logits,))

    def bw(g: list[Any]) -> None:
        gs = g[0][0]
        for i in range(m):
            t = targets[i]
            for j in range(V):
                logits.grad[i][j] += gs * (probs[i][j] - (1.0 if j == t else 0.0)) / m

    out._backward = bw
    return out
