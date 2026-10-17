"""观测层：把一次前向的路由事实折算成可比指标。

**这一层是本项目的产物本身。** ArchitectureFactory 那边，MoE 只是一个装配选项；
这里，路由系统的每一个可观测侧面都要被量化、画成曲线、并与真值核对。

指标分三组，**不要混着看**：

| 组 | 指标 | 它回答什么 |
| --- | --- | --- |
| 负载 | ``entropy_norm`` / ``imbalance`` / ``load_cv`` | 活干得均不均匀 |
| 分化 | ``nmi`` / ``purity`` / ``expert_sim`` | 专家有没有形成职能 |
| 代价 | ``dropped_rate`` / ``comm_units`` / ``active_params`` | 这些换来多少开销 |

**为什么必须分三组。** 「均衡」与「分化」是两个独立的轴，而且会互相拉扯 ——
这正是本项目的核心命题。把它们揉成一个「路由质量」分数，就再也看不到张力了，
也就退化成「实现一个 MoE」。``static`` 对照专门用来证明这一点：它负载完美均衡
（``entropy_norm == 1.0``）而分化度为零。

**观测固定在评测集上做。** 训练 batch 每步都在换，路由统计会被 batch 组成污染，
曲线就不再是「路由器学到了什么」的证据。所以快照一律在**固定的测试集**上采集：
同一批 token、同一个顺序、每一步都一样。这样 ``stability``（同一 token 的路由
是否稳定）才有意义 —— 否则它只是在测量「换了一批数据」。
"""

from __future__ import annotations

from typing import Any, Sequence

from .corpus import Batch, Corpus
from .moe import MoEModel, RouteTrace
from .schema import (
    imbalance_ratio,
    load_cv,
    normalized_entropy,
    normalized_mutual_information,
    purity,
)


def domain_expert_matrix(corpus: Corpus, domains: Sequence[int], top1: Sequence[int]) -> list[list[int]]:
    """域 × 专家 的计数矩阵 —— **Sankey 的流就是它**。

    Sankey 的左列是域（token 按域分组），右列是专家，流量 = 被派发过去的 token 数。
    一眼就能看出「有没有塌缩」和「有没有分化」。
    """
    m = [[0] * corpus.n_domains for _ in range(corpus.n_domains)]
    for d, e in zip(domains, top1):
        m[d][e] += 1
    return m


def dominant_expert(matrix: Sequence[Sequence[int]]) -> list[int]:
    """每个域的**主导专家**（该域 token 去得最多的那个）。"""
    out: list[int] = []
    for row in matrix:
        best = 0
        for i, v in enumerate(row):
            if v > row[best]:
                best = i
        out.append(best)
    return out


def is_bijection(mapping: Sequence[int]) -> bool:
    """「域 → 主导专家」是不是一一对应。

    **只在 ``top_k == 1`` 时才是正确的真值判据。** ``top_k == 2`` 时一个域由
    **一对**专家共同承担，两个域的主导专家完全可能相同而依然各司其职 ——
    这条判据会把正确的结果判成失败（实测踩过，见 README 附录 C 的 bug 58）。
    多专家场合请用 :func:`primary_experts` / :func:`distinct_primary_sets`。
    """
    return len(set(mapping)) == len(mapping)


def primary_experts(row: Sequence[int], thresh: float = 0.2) -> tuple[int, ...]:
    """一个域的**主要专家集合**：承接该域比例 ≥ ``thresh`` 的专家。

    ``top_k=1`` 时通常只剩一个（退化成「主导专家」）；``top_k=2`` 时是一对。
    这个定义对两种 ``top_k`` 都成立，是判「域有没有被专门承担」的正确粒度。
    """
    tot = sum(row)
    if tot <= 0:
        return ()
    return tuple(i for i, v in enumerate(row) if v / tot >= thresh)


def distinct_primary_sets(matrix: Sequence[Sequence[int]], thresh: float = 0.2) -> bool:
    """每个域的主要专家集合两两不同 —— **本项目核对真值的方式**。

    对照组会失败得很干净：它们的行近似均匀，于是每个域的主要集合都是「全部专家」，
    四个域完全相同。这就是「没有分化」的可判定形式。
    """
    sets = [primary_experts(row, thresh) for row in matrix]
    if any(not s for s in sets):
        return False
    return len(set(sets)) == len(sets)


def observe(
    model: MoEModel,
    batch: Batch,
    trace: RouteTrace,
    corpus: Corpus,
    *,
    step: int,
    loss: float,
    acc: float,
    prev_top1: Sequence[int] | None = None,
    domain_correct: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """把一次前向折算成一份观测快照。**只读，不改模型、不改轨迹。**"""
    E = model.n_experts
    load = list(trace.load)
    total = sum(load)

    h, h_norm = normalized_entropy(load)
    matrix = domain_expert_matrix(corpus, batch.domains, trace.top1)
    dom = dominant_expert(matrix)
    pur, per_expert = purity(batch.domains, trace.top1, E)
    nmi = normalized_mutual_information(batch.domains, trace.top1)

    gate_max = 0.0
    if trace.gates:
        gate_max = sum(max(row) for row in trace.gates) / len(trace.gates)

    stability: float | None = None
    if prev_top1 is not None and len(prev_top1) == len(trace.top1):
        same = sum(1 for a, b in zip(prev_top1, trace.top1) if a == b)
        stability = same / len(trace.top1)

    snap: dict[str, Any] = {
        "step": step,
        "loss": loss,
        "acc": acc,
        # -- 负载 --
        "load": load,
        "load_frac": [(c / total) if total else 0.0 for c in load],
        "entropy": h,
        "entropy_norm": h_norm,
        "imbalance": imbalance_ratio(load),
        "load_cv": load_cv(load),
        # -- 分化 --
        "nmi": nmi,
        "purity": pur,
        "expert_domain": per_expert,
        "expert_sim": model.expert_similarity(),
        "gate_max_mean": gate_max,
        "stability": stability,
        # -- 代价 --
        "dropped": trace.dropped,
        "dropped_rate": trace.dropped_rate,
        "comm_units": trace.comm_units,
        "active_params": trace.active_params,
        "dispatched": trace.dispatched,
    }
    if domain_correct is not None:
        snap["domain_acc"] = _domain_accuracy(corpus, batch.domains, domain_correct)
    return snap


def _domain_accuracy(corpus: Corpus, domains: Sequence[int], correct: Sequence[bool]) -> list[float]:
    hit = [0] * corpus.n_domains
    tot = [0] * corpus.n_domains
    for d, ok in zip(domains, correct):
        tot[d] += 1
        if ok:
            hit[d] += 1
    return [(hit[d] / tot[d]) if tot[d] else 0.0 for d in range(corpus.n_domains)]


def summarize(snapshots: Sequence[dict[str, Any]], corpus: Corpus) -> dict[str, float]:
    """把快照序列收成**最终一行**指标（报告与矩阵表用的就是它）。"""
    if not snapshots:
        raise ValueError("没有快照可汇总")
    last = snapshots[-1]
    out: dict[str, float] = {
        "loss": last["loss"],
        "acc": last["acc"],
        "entropy_norm": last["entropy_norm"],
        "imbalance": last["imbalance"],
        "load_cv": last["load_cv"],
        "nmi": last["nmi"],
        "purity": last["purity"],
        "expert_sim": last["expert_sim"],
        "gate_max_mean": last["gate_max_mean"],
        "dropped_rate": last["dropped_rate"],
        "comm_units": float(last["comm_units"]),
        "active_params": float(last["active_params"]),
        "stability": float(last["stability"]) if last["stability"] is not None else 0.0,
    }
    accs = [s["acc"] for s in snapshots]
    out["acc_first"] = accs[0]
    out["acc_gain"] = accs[-1] - accs[0]
    out["loss_first"] = snapshots[0]["loss"]
    out["loss_drop"] = snapshots[0]["loss"] - last["loss"]
    if "domain_acc" in last:
        da = last["domain_acc"]
        out["domain_acc_min"] = min(da)
        out["domain_acc_mean"] = sum(da) / len(da)
    return out
