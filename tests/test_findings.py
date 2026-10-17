"""结论层：**每条结论都必须能被数据推翻**。

这里测的不是「结论是否成立」（那是 ``test_acceptance.py`` 用真实协议回答的问题），
而是**每条结论的判定逻辑**：喂进一组构造好的观测，它是否给出正确的是/否。

**为什么必须分开测。** 结论是算出来的（不是写死的字符串），所以判定条件里任何一个
符号写反、任何一个 ``origin`` 没排除，都会让报告把「没测出来」说成「测出来了」——
而报告看起来完全正常。这类 bug 在标定阶段真的出现过两次：

- ``grid_complete`` 把复核行也算进网格，于是「24 格」被数成「39 格」；
- ``truth_recovered`` 用「主导专家」判真值，``top_k=2`` 时把正确结果判成失败（坑 58）。
"""

from __future__ import annotations

import pytest

from moefactory.findings import _by_group, _rank, compute_findings, spearman
from moefactory.observe import dominant_expert
from moefactory.schema import MoESpec, RunRecord

FLOOR = 1.0
MARGIN = 0.1
THR = FLOOR - MARGIN  # 0.9

#: 每个域由一个不同专家独占 —— 「分化成功」的形态。
ONE_HOT = [[25, 0, 0, 0], [0, 25, 0, 0], [0, 0, 25, 0], [0, 0, 0, 25]]
#: 每行近似均匀 —— 「没有分化」的形态（对照组就是这样）。
UNIFORM = [[25, 25, 25, 25]] * 4
#: top_k=2 的形态：每个域由一对专家承担，集合两两不同。
PAIRS = [[60, 40, 0, 0], [0, 0, 70, 30], [30, 0, 0, 70], [0, 80, 20, 0]]


def _spec(*, routing: str = "learned", aux: float | None = None, top_k: int = 1) -> MoESpec:
    # 对照组的 aux 是 **N/A**，不是 0 —— 这里按 routing 给默认值，免得把「不适用」
    # 当成一个真实取值写进 spec_id（那会让同一份配置出现两个 id）。
    if aux is None:
        aux = 0.1 if routing == "learned" else 0.0
    return MoESpec(n_experts=4, top_k=top_k, d_model=12, d_ff=16,
                   routing=routing, aux_weight=aux, capacity_factor=2.0)


def _run(
    spec: MoESpec,
    *,
    loss: float,
    nmi: float = 0.0,
    entropy_norm: float = 0.9,
    purity_: float = 0.5,
    origin: str = "grid",
    seed: int = 1,
    matrix: list[list[int]] | None = None,
) -> RunRecord:
    m = ONE_HOT if matrix is None else matrix
    return RunRecord(
        spec_id=spec.spec_id,
        spec=spec.to_dict(),
        seed=seed,
        final={
            "loss": loss,
            "nmi": nmi,
            "entropy_norm": entropy_norm,
            "purity": purity_,
            "domain_blind_floor": FLOOR,
            "below_floor": 1.0 if loss < THR else 0.0,
        },
        snapshots=[],
        domain_expert=[list(r) for r in m],
        domain_dominant=dominant_expert(m),
        onset_step=None,
        params=1,
        flops=1,
        train_seconds=0.0,
        origin=origin,
    )


def _grid(n_learned: int, n_control: int) -> dict:
    return {
        "n_specs": n_learned + n_control,
        "groups": [
            {"name": "learned", "declared": n_learned, "axes": {}, "fixed": {}},
            {"name": "control", "declared": n_control, "axes": {}, "fixed": {}},
        ],
        "axes": ["aux_weight"],
    }


def _find(runs, *, grid=None, replication=None, expectations=None):
    return compute_findings(
        runs,
        floor=FLOOR,
        margin=MARGIN,
        grid=grid if grid is not None else _grid(4, 2),
        replication=replication,
        expectations=expectations if expectations is not None else {"min_specialized_frac": 0.6},
    )


def _get(findings, key):
    for f in findings:
        if f.key == key:
            return f
    return None


# ----------------------------------------------------------------------
# 秩相关（所有相关系数的底座）
# ----------------------------------------------------------------------
def test_spearman_perfect_monotone() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties() -> None:
    assert spearman([1, 1, 2, 2], [5, 5, 9, 9]) == pytest.approx(1.0)


def test_spearman_needs_three_points() -> None:
    assert spearman([1, 2], [2, 1]) == 0.0
    assert spearman([], []) == 0.0


def test_spearman_zero_variance() -> None:
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0


def test_rank_averages_ties() -> None:
    assert _rank([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


def test_by_group_splits_learned_and_control() -> None:
    learned = [_run(_spec(), loss=0.5)]
    control = [_run(_spec(routing="static"), loss=1.2)]
    a, b = _by_group(learned + control)
    assert a == learned and b == control


# ----------------------------------------------------------------------
# 1. 扇出完整性
# ----------------------------------------------------------------------
def test_grid_complete_counts_only_grid_rows() -> None:
    """**复核行不属于网格。** 混进去就会把 6 格数成 8 格（实测踩过）。"""
    runs = [_run(_spec(aux=a), loss=1.2) for a in (0.0, 0.1, 0.2, 0.5)]
    runs += [_run(_spec(routing="static"), loss=1.2), _run(_spec(routing="random"), loss=1.2)]
    runs += [_run(_spec(), loss=0.1, origin="replication", seed=2)]
    f = _get(_find(runs), "grid_complete")
    assert f.supported is True
    assert f.evidence["learned"] == "4/4"
    assert f.evidence["control"] == "2/2"
    assert f.evidence["grid_total"] == 6
    assert f.evidence["replication"] == 1


def test_grid_complete_fails_when_a_branch_is_missing() -> None:
    runs = [_run(_spec(aux=a), loss=1.2) for a in (0.0, 0.1, 0.2)]  # 少了一格
    runs += [_run(_spec(routing="static"), loss=1.2), _run(_spec(routing="random"), loss=1.2)]
    f = _get(_find(runs), "grid_complete")
    assert f.supported is False
    assert f.evidence["learned"] == "3/4"


# ----------------------------------------------------------------------
# 2. 对照组钉死在下界以上
# ----------------------------------------------------------------------
def test_control_pinned_at_floor_requires_all_control_rows_above() -> None:
    learned = [_run(_spec(), loss=0.2)]
    controls = [_run(_spec(routing="static"), loss=1.2), _run(_spec(routing="random"), loss=1.35)]
    f = _get(_find(learned + controls), "control_pinned_at_floor")
    assert f.supported is True
    assert f.evidence["count"] == "2/2"

    broken = [_run(_spec(routing="static"), loss=1.2), _run(_spec(routing="random"), loss=0.5)]
    f2 = _get(_find(learned + broken), "control_pinned_at_floor")
    assert f2.supported is False
    assert f2.evidence["count"] == "1/2"


def test_control_pinned_at_floor_fails_without_any_control_row() -> None:
    """没有对照组就没有「分离」可言 —— 空集不能算「全部满足」。"""
    f = _get(_find([_run(_spec(), loss=0.2)]), "control_pinned_at_floor")
    assert f.supported is False


def test_control_pinned_at_floor_uses_every_run_including_replication() -> None:
    """这条断言问的是「分离是否成立」，样本越多越好 ⇒ 复核行也要算进来。"""
    learned = [_run(_spec(), loss=0.2)]
    controls = [
        _run(_spec(routing="static"), loss=1.2),
        _run(_spec(routing="static"), loss=0.4, origin="replication", seed=2),
    ]
    f = _get(_find(learned + controls), "control_pinned_at_floor")
    assert f.supported is False
    assert f.evidence["count"] == "1/2"


# ----------------------------------------------------------------------
# 3. learned 突破下界
# ----------------------------------------------------------------------
def test_learned_breaks_floor_uses_the_configured_fraction() -> None:
    ok = [_run(_spec(aux=a), loss=0.2) for a in (0.0, 0.1, 0.2, 0.5)]
    f = _get(_find(ok + [_run(_spec(routing="static"), loss=1.2)]), "learned_breaks_floor")
    assert f.supported is True
    assert f.evidence["fraction"] == "100%"

    half = [_run(_spec(aux=a), loss=(0.2 if a < 0.2 else 1.2)) for a in (0.0, 0.1, 0.2, 0.5)]
    f2 = _get(_find(half + [_run(_spec(routing="static"), loss=1.2)]), "learned_breaks_floor")
    assert f2.supported is False
    assert f2.evidence["fraction"] == "50%"


def test_learned_breaks_floor_counts_replication_rows() -> None:
    grid = [_run(_spec(aux=a), loss=1.2) for a in (0.0, 0.1, 0.2, 0.5)]
    rep = [_run(_spec(), loss=0.1, origin="replication", seed=2),
           _run(_spec(), loss=0.15, origin="replication", seed=3)]
    f = _get(_find(grid + rep + [_run(_spec(routing="static"), loss=1.2)]), "learned_breaks_floor")
    assert f.evidence["count"] == "2/6"
    assert f.supported is False


# ----------------------------------------------------------------------
# 4. 均衡指标指向的方向是错的
# ----------------------------------------------------------------------
def _balance_case() -> list[RunRecord]:
    """构造一张「均衡 ↔ loss 正相关」的网格。

    learned 组内的均衡与 loss 刻意**不**单调（Spearman 约 0.45），
    对照组则占据均衡榜前两名而 loss 最差。这正是真实产物里的形态。
    """
    table = [
        # (entropy_norm, loss, nmi)
        (0.99, 0.30, 0.45),
        (0.98, 0.20, 0.55),
        (0.97, 0.30, 0.40),
        (0.96, 0.20, 0.50),
    ]
    runs = [
        _run(_spec(aux=a), loss=loss, nmi=nmi, entropy_norm=h)
        for a, (h, loss, nmi) in zip((0.0, 0.1, 0.2, 0.5), table)
    ]
    runs += [
        _run(_spec(routing="static"), loss=1.20, nmi=0.02, entropy_norm=1.0),
        _run(_spec(routing="random"), loss=1.30, nmi=0.03, entropy_norm=0.995),
    ]
    return runs


def test_balance_points_the_wrong_way_detects_the_trap() -> None:
    f = _get(_find(_balance_case()), "balance_points_the_wrong_way")
    assert f.supported is True
    assert f.evidence["most_balanced_spec"].endswith("static-aux0-cap2")
    assert f.evidence["most_balanced_entropy_norm"] == pytest.approx(1.0)
    assert f.evidence["spearman_balance_vs_loss_all"] > 0.3
    assert abs(f.evidence["spearman_balance_vs_loss_learned_only"]) < 0.5
    assert f.evidence["spearman_nmi_vs_loss"] < -0.5


def test_balance_points_the_wrong_way_fails_when_balance_helps() -> None:
    """反例：若均衡与 loss 负相关，这条结论必须**报不支持**，而不是硬说成立。"""
    runs = [
        _run(_spec(aux=a), loss=1.4 - 0.3 * i, nmi=0.1 * i, entropy_norm=0.9 + 0.02 * i)
        for i, a in enumerate((0.0, 0.1, 0.2, 0.5))
    ]
    runs += [
        _run(_spec(routing="static"), loss=1.5, nmi=0.0, entropy_norm=0.95),
        _run(_spec(routing="random"), loss=1.6, nmi=0.0, entropy_norm=0.96),
    ]
    f = _get(_find(runs), "balance_points_the_wrong_way")
    assert f.supported is False


# ----------------------------------------------------------------------
# 5. 辅助损失这一轴是平的
# ----------------------------------------------------------------------
def test_aux_axis_is_flat_when_aux_is_not_required() -> None:
    runs = [
        _run(_spec(aux=0.0), loss=0.25, entropy_norm=0.95),
        _run(_spec(aux=0.0), loss=0.30, entropy_norm=0.96),
        _run(_spec(aux=0.1), loss=0.28, entropy_norm=0.97),
        _run(_spec(aux=0.1), loss=0.26, entropy_norm=0.98),
        _run(_spec(aux=0.5), loss=0.24, entropy_norm=0.94),
        _run(_spec(aux=0.5), loss=0.32, entropy_norm=0.99),
    ]
    runs += [
        _run(_spec(routing="static"), loss=1.20, entropy_norm=1.0),
        _run(_spec(routing="random"), loss=1.30, entropy_norm=0.99),
    ]
    f = _get(_find(runs), "aux_axis_is_flat")
    assert f.supported is True
    assert f.evidence["aux0_break_floor"] == "2/2"
    assert float(f.evidence["aux_axis_spread"].rstrip("%")) < 35.0
    assert f.evidence["per_aux_mean_loss"].keys() == {"0", "0.1", "0.5"}


def test_aux_axis_is_flat_fails_when_aux_zero_cannot_break_the_floor() -> None:
    """反例：若关掉辅助损失就学不会，那这一轴就不是平的 —— 必须报不支持。"""
    runs = [
        _run(_spec(aux=0.0), loss=1.20),
        _run(_spec(aux=0.0), loss=1.25),
        _run(_spec(aux=0.5), loss=0.25),
    ]
    runs += [
        _run(_spec(routing="static"), loss=1.20),
        _run(_spec(routing="random"), loss=1.30),
    ]
    f = _get(_find(runs), "aux_axis_is_flat")
    assert f.supported is False
    assert f.evidence["aux0_break_floor"] == "0/2"


# ----------------------------------------------------------------------
# 6. 真值复原
# ----------------------------------------------------------------------
def test_truth_recovered_uses_primary_sets_not_dominant_experts() -> None:
    """``top_k=2`` 时一个域由一对专家承担 —— 用主导专家判真值会误判（坑 58）。"""
    best = _run(_spec(top_k=2), loss=0.15, matrix=PAIRS, purity_=0.64)
    control = _run(_spec(routing="static"), loss=1.20, matrix=UNIFORM)
    assert dominant_expert(PAIRS) == [0, 2, 3, 1]  # 恰好是一一对应，但这不是判据
    f = _get(_find([best, control] + [_run(_spec(aux=0.5), loss=1.2, matrix=UNIFORM)]),
             "truth_recovered")
    assert f.supported is True
    assert f.evidence["best_all_distinct"] is True
    assert f.evidence["control_all_distinct"] is False
    assert f.evidence["best_primary_sets"] == [[0, 1], [2, 3], [0, 3], [1, 2]]


def test_truth_recovered_fails_when_the_control_also_looks_specialised() -> None:
    best = _run(_spec(), loss=0.15, matrix=ONE_HOT)
    control = _run(_spec(routing="static"), loss=1.20, matrix=ONE_HOT)
    f = _get(_find([best, control]), "truth_recovered")
    assert f.supported is False


def test_truth_recovered_fails_without_a_run_below_the_floor() -> None:
    runs = [
        _run(_spec(), loss=1.2, matrix=ONE_HOT),
        _run(_spec(routing="static"), loss=1.2, matrix=UNIFORM),
    ]
    f = _get(_find(runs), "truth_recovered")
    assert f.supported is False
    assert f.evidence["best_spec"] is None


# ----------------------------------------------------------------------
# 7. 多种子复核
# ----------------------------------------------------------------------
def _rep(learned_success: int, learned_total: int, control_success: int, control_total: int) -> dict:
    return {
        "seeds": [1, 2, 3],
        "learned_success": learned_success,
        "learned_total": learned_total,
        "control_success": control_success,
        "control_total": control_total,
        "per_spec": [],
    }


def test_replication_separates_needs_zero_control_successes() -> None:
    runs = [_run(_spec(), loss=0.2), _run(_spec(routing="static"), loss=1.2)]
    f = _get(_find(runs, replication=_rep(6, 6, 0, 3)), "replication_separates")
    assert f.supported is True
    assert f.evidence["learned"] == "6/6"
    assert f.evidence["control"] == "0/3"

    f2 = _get(_find(runs, replication=_rep(6, 6, 1, 3)), "replication_separates")
    assert f2.supported is False

    f3 = _get(_find(runs, replication=_rep(5, 6, 0, 3)), "replication_separates")
    assert f3.supported is False


def test_replication_finding_is_absent_when_not_configured() -> None:
    runs = [_run(_spec(), loss=0.2)]
    assert _get(_find(runs, replication=None), "replication_separates") is None


# ----------------------------------------------------------------------
# 整体
# ----------------------------------------------------------------------
def test_findings_are_ordered_and_uniquely_keyed() -> None:
    runs = _balance_case()
    findings = _find(runs, replication=_rep(3, 3, 0, 1))
    keys = [f.key for f in findings]
    assert keys == [
        "grid_complete",
        "control_pinned_at_floor",
        "learned_breaks_floor",
        "balance_points_the_wrong_way",
        "aux_axis_is_flat",
        "truth_recovered",
        "replication_separates",
    ]
    assert len(keys) == len(set(keys))


def test_every_finding_carries_claim_and_evidence() -> None:
    """结论必须能被引用、被反驳：主张 + 证据 + 判定，一个都不能缺。"""
    for f in _find(_balance_case(), replication=_rep(3, 3, 0, 1)):
        assert isinstance(f.claim, str) and f.claim
        assert isinstance(f.supported, bool)
        assert isinstance(f.evidence, dict) and f.evidence
