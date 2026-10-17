"""数值引擎：**每个算子都要过「解析梯度 == 数值梯度」**。

梯度写错**不会报错**。它只会让训练静默变差、让结论变成「路由学不会任何东西」——
而这个项目里「路由学不会」恰恰是一个**合法结论**。两者在输出上长得一模一样，
所以唯一的防线就是这里：把每个算子的反向逐元素与中心差分对齐。

**踩过的坑（一次就够）。** 第一版校验脚本在循环里对共享张量调了 ``zero_grad()``，
于是数值梯度是对的、解析梯度被中途清零，15 个算子报出 11 个假 FAIL。教训是：
``backward()`` 之后**先把解析梯度快照下来**，再去做扰动（见 :func:`gradcheck`）。

**第二个坑**是权重：这里用固定权重的加权和把算子输出折成标量，而不是全 1 求和。
全 1 的 cotangent 会让「两个轴写反了」这类 bug 互相抵消 —— 加权和才照得出来。
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Sequence

import pytest

from moefactory import autograd as ag
from moefactory.autograd import (
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
    softmax_cross_entropy,
    softmax_rows,
    sum_all,
    sum_rows,
    take_col,
)

# ----------------------------------------------------------------------
# 校验器
# ----------------------------------------------------------------------
def _keys(t: Tensor) -> list[tuple[int, int | None]]:
    d = t.data
    if not d:
        return []
    if isinstance(d[0], list):
        return [(i, j) for i in range(len(d)) for j in range(len(d[0]))]  # type: ignore[arg-type]
    return [(i, None) for i in range(len(d))]


def _get(container: Sequence[Any], key: tuple[int, int | None]) -> float:
    i, j = key
    return container[i][j] if j is not None else container[i]  # type: ignore[index,return-value]


def _set(t: Tensor, key: tuple[int, int | None], value: float) -> None:
    i, j = key
    if j is not None:
        t.data[i][j] = value
    else:
        t.data[i] = value


def _weights(shape: tuple[int, ...], seed: int = 20261001) -> list[Any]:
    r = random.Random(seed)
    if len(shape) == 2:
        return [[r.uniform(-1.0, 1.0) for _ in range(shape[1])] for _ in range(shape[0])]
    return [r.uniform(-1.0, 1.0) for _ in range(shape[0])]


def _to_scalar(t: Tensor) -> Tensor:
    """折成标量张量。已经是 ``[[s]]`` 就直接用（softmax_cross_entropy / sum_all）。"""
    if t.shape == (1, 1):
        return t
    return sum_all(mul(t, constant(_weights(t.shape))))


def gradcheck(
    make_inputs: Callable[[], list[Tensor]],
    forward: Callable[[list[Tensor]], Tensor],
    *,
    eps: float = 1e-5,
    tol: float = 1e-4,
) -> float:
    """返回最大相对误差。调用方断言它 ``< tol``。"""
    ts = make_inputs()
    _to_scalar(forward(ts)).backward()
    # **先快照**：之后每次扰动都会重建一组输入，但解析梯度只取这一次。
    analytic = [copy.deepcopy(t.grad) for t in ts]

    worst = 0.0
    for a, t0 in enumerate(ts):
        for key in _keys(t0):
            base = _get(t0.data, key)
            vals: list[float] = []
            for delta in (eps, -eps):
                ts2 = make_inputs()
                _set(ts2[a], key, base + delta)
                vals.append(float(_to_scalar(forward(ts2)).data[0][0]))
            numeric = (vals[0] - vals[1]) / (2.0 * eps)
            got = _get(analytic[a], key)
            err = abs(numeric - got) / max(1.0, abs(numeric), abs(got))
            worst = max(worst, err)
    return worst


# ----------------------------------------------------------------------
# 每个算子的用例（make_inputs, forward）
# ----------------------------------------------------------------------
CHECKS: dict[str, tuple[Callable[[], list[Tensor]], Callable[[list[Tensor]], Tensor]]] = {
    "matmul": (
        lambda: [
            constant([[0.3, -0.7, 1.1], [0.2, 0.5, -0.9]]),
            constant([[0.1, 0.4], [-0.6, 0.2], [0.8, -0.3]]),
        ],
        lambda ts: matmul(ts[0], ts[1]),
    ),
    "add": (
        lambda: [
            constant([[0.3, -0.7], [0.2, 0.5]]),
            constant([[-1.1, 0.4], [0.9, -0.2]]),
        ],
        lambda ts: add(ts[0], ts[1]),
    ),
    "add_bias": (
        lambda: [constant([[0.3, -0.7, 1.1], [0.2, 0.5, -0.9]]), constant([0.4, -0.2, 0.6])],
        lambda ts: add_bias(ts[0], ts[1]),
    ),
    "mul": (
        lambda: [constant([[0.3, -0.7], [0.2, 0.5]]), constant([[-1.1, 0.4], [0.9, -0.2]])],
        lambda ts: mul(ts[0], ts[1]),
    ),
    "scale": (
        lambda: [constant([[0.3, -0.7], [0.2, 0.5]])],
        lambda ts: scale(ts[0], 1.7),
    ),
    "row_scale": (
        lambda: [constant([[0.3, -0.7], [0.2, 0.5], [0.1, 0.9]]), constant([1.3, -0.6, 0.8])],
        lambda ts: row_scale(ts[0], ts[1]),
    ),
    "take_col": (
        lambda: [constant([[0.3, -0.7, 1.1], [0.2, 0.5, -0.9]])],
        lambda ts: take_col(ts[0], 1),
    ),
    "gather_rows": (
        lambda: [constant([[0.3, -0.7], [0.2, 0.5], [0.1, 0.9]])],
        lambda ts: gather_rows(ts[0], [2, 0, 2]),
    ),
    "gather1d": (
        lambda: [constant([0.3, -0.7, 0.2, 0.5])],
        lambda ts: gather1d(ts[0], [3, 1, 3]),
    ),
    "scatter_rows": (
        lambda: [constant([[0.3, -0.7], [0.2, 0.5], [0.1, 0.9]])],
        lambda ts: scatter_rows(ts[0], [1, 0, 1], 3),
    ),
    "sum_all": (
        lambda: [constant([[0.3, -0.7], [0.2, 0.5]])],
        lambda ts: sum_all(ts[0]),
    ),
    "sum_rows": (
        lambda: [constant([[0.3, -0.7], [0.2, 0.5], [0.1, 0.9]])],
        lambda ts: sum_rows(ts[0]),
    ),
    "mean_rows": (
        lambda: [constant([[0.3, -0.7], [0.2, 0.5], [0.1, 0.9]])],
        lambda ts: mean_rows(ts[0]),
    ),
    "silu": (
        lambda: [constant([[-1.4, 0.2], [0.7, -0.3], [1.9, -2.2]])],
        lambda ts: silu(ts[0]),
    ),
    "rmsnorm": (
        lambda: [constant([[0.3, -0.7, 1.1], [0.2, 0.5, -0.9]]), constant([1.1, 0.7, 1.4])],
        lambda ts: rmsnorm(ts[0], ts[1]),
    ),
    "embed": (
        lambda: [constant([[0.3, -0.7, 1.1], [0.2, 0.5, -0.9], [0.8, -0.3, 0.4]])],
        lambda ts: embed(ts[0], [0, 2, 1, 2]),
    ),
    "softmax_rows": (
        lambda: [constant([[0.3, -0.7, 1.1], [0.2, 0.5, -0.9]])],
        lambda ts: softmax_rows(ts[0]),
    ),
    "softmax_cross_entropy": (
        lambda: [constant([[0.3, -0.7, 1.1], [0.2, 0.5, -0.9]])],
        lambda ts: softmax_cross_entropy(ts[0], [1, 0]),
    ),
}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_analytic_gradient_matches_numeric(name: str) -> None:
    make_inputs, forward = CHECKS[name]
    assert gradcheck(make_inputs, forward) < 1e-4, f"{name} 的反向与数值梯度不符"


def test_every_operator_is_covered() -> None:
    """``__all__`` 里的算子一个都不能漏 —— 漏掉的那个就是静默变差的那一个。"""
    non_ops = {"Tensor", "constant", "ones", "zeros"}
    assert set(CHECKS) == set(ag.__all__) - non_ops


def test_weighted_scalarisation_is_not_degenerate() -> None:
    """自检：加权和必须真的带上不同权重，否则这个校验器照不出轴写反的 bug。"""
    w = _weights((2, 3))
    assert len(set(v for row in w for v in row)) == 6


# ----------------------------------------------------------------------
# 形状 / 语义约定
# ----------------------------------------------------------------------
def test_sum_all_returns_scalar_tensor() -> None:
    t = sum_all(constant([[1.0, 2.0], [3.0, 4.0]]))
    assert t.shape == (1, 1)
    assert t.data[0][0] == pytest.approx(10.0)


def test_softmax_rows_sums_to_one() -> None:
    t = softmax_rows(constant([[0.3, -0.7, 1.1], [5.0, 5.0, 5.0]]))
    for row in t.data:
        assert sum(row) == pytest.approx(1.0)


def test_scatter_rows_accumulates_duplicates() -> None:
    """同一个下标出现多次时**相加** —— top_k 里同一个 token 被多个专家处理靠它。"""
    out = scatter_rows(constant([[1.0, 2.0], [10.0, 20.0]]), [0, 0], 2)
    assert out.data[0] == [11.0, 22.0]
    assert out.data[1] == [0.0, 0.0]


def test_gather_scatter_are_inverse() -> None:
    src = constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    idx = [2, 0, 1]
    assert scatter_rows(gather_rows(src, idx), idx, 3).data == src.data


def test_matmul_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        matmul(constant([[1.0, 2.0]]), constant([[1.0, 2.0]]))


def test_backward_accumulates_into_leaf_grad() -> None:
    """同一个叶子被用两次时梯度必须**累加**，不能被覆盖。"""
    x = constant([2.0, 3.0])
    mul(x, x).backward()
    assert x.grad == [4.0, 6.0]
