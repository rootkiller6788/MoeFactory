"""观测层：把一次前向的路由事实折算成可比指标。

**指标分三组，测试也按三组分开写**（负载 / 分化 / 代价）。把它们揉成一个「路由质量」
分数，本项目的核心命题（均衡与分化是两个独立的轴）就再也测不出来了 ——
所以这里连一条「综合分」的测试都不该有。

另外两条口径必须在测试里固定下来，否则报告里的数字会漂：

- **NMI 用算术平均归一化**（不是 sqrt 或 min）；
- **``onset_step`` 用相邻步最大增幅**定位，而不是「首次越过阈值」——
  后者对阈值敏感，平曲线会给出假步号。
"""

from __future__ import annotations

import pytest

from moefactory.corpus import make_batch
from moefactory.moe import build_model
from moefactory.observe import (
    _domain_accuracy,
    distinct_primary_sets,
    domain_expert_matrix,
    dominant_expert,
    is_bijection,
    observe,
    primary_experts,
    summarize,
)
from moefactory.schema import (
    entropy_of,
    imbalance_ratio,
    load_cv,
    mean_pairwise_cosine,
    mutual_information,
    normalized_entropy,
    normalized_mutual_information,
    onset_step,
    purity,
)

LEARNED = "e4-k2-d12-f16-learned-aux0.1-cap2"


# ----------------------------------------------------------------------
# 负载组
# ----------------------------------------------------------------------
def test_normalized_entropy_of_a_perfect_partition_is_one() -> None:
    h, h_norm = normalized_entropy([10, 10, 10, 10])
    assert h == pytest.approx(1.3862943611, rel=1e-9)
    assert h_norm == pytest.approx(1.0)


def test_normalized_entropy_of_a_collapse_is_zero() -> None:
    h, h_norm = normalized_entropy([40, 0, 0, 0])
    assert h == pytest.approx(0.0)
    assert h_norm == pytest.approx(0.0)


def test_normalized_entropy_edge_cases() -> None:
    assert normalized_entropy([0, 0, 0, 0]) == (0.0, 0.0)
    assert normalized_entropy([7]) == (0.0, 0.0)  # 只有一个专家时 hmax 退化为 1


def test_imbalance_ratio_extremes() -> None:
    assert imbalance_ratio([10, 10, 10, 10]) == pytest.approx(1.0)
    assert imbalance_ratio([40, 0, 0, 0]) == pytest.approx(4.0)
    assert imbalance_ratio([0, 0]) == 1.0


def test_load_cv() -> None:
    assert load_cv([10, 10, 10, 10]) == pytest.approx(0.0)
    assert load_cv([40, 0, 0, 0]) == pytest.approx(1.7320508, rel=1e-6)
    assert load_cv([]) == 0.0


# ----------------------------------------------------------------------
# 分化组
# ----------------------------------------------------------------------
def test_mutual_information_is_zero_for_independent_labels() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    clusters = [0, 1, 0, 1, 0, 1, 0, 1]
    assert mutual_information(labels, clusters) == pytest.approx(0.0)
    assert normalized_mutual_information(labels, clusters) == pytest.approx(0.0)


def test_nmi_is_one_for_an_identical_partition() -> None:
    labels = [0, 0, 1, 1, 2, 2]
    assert normalized_mutual_information(labels, list(labels)) == pytest.approx(1.0)


def test_nmi_is_invariant_under_relabelling() -> None:
    labels = [0, 0, 1, 1, 2, 2]
    clusters = [2, 2, 0, 0, 1, 1]
    assert normalized_mutual_information(labels, clusters) == pytest.approx(1.0)


def test_nmi_uses_arithmetic_mean_normalisation() -> None:
    """口径固定：分母是 ``(H(X)+H(Y))/2``，不是 sqrt 也不是 min。"""
    labels = [0, 0, 1, 1]
    clusters = [0, 0, 1, 1]
    mi = mutual_information(labels, clusters)
    denom = (entropy_of(labels) + entropy_of(clusters)) / 2.0
    assert normalized_mutual_information(labels, clusters) == pytest.approx(mi / denom)


def test_nmi_edge_cases() -> None:
    assert normalized_mutual_information([], []) == 0.0
    assert normalized_mutual_information([1, 1, 1], [1, 1, 1]) == 0.0  # 分母为 0


def test_entropy_of() -> None:
    assert entropy_of([0, 0, 1, 1]) == pytest.approx(0.6931471806, rel=1e-9)
    assert entropy_of([]) == 0.0


def test_purity_is_load_weighted() -> None:
    # 专家 0 接 3 个域0 + 1 个域1（纯 0.75）；专家 1 全接域1（纯 1.0）
    labels = [0, 0, 0, 1, 1, 1, 1]
    clusters = [0, 0, 0, 0, 1, 1, 1]
    weighted, per = purity(labels, clusters, 2)
    assert per[0] == pytest.approx(0.75)
    assert per[1] == pytest.approx(1.0)
    assert weighted == pytest.approx((0.75 * 4 + 1.0 * 3) / 7)


def test_purity_of_an_empty_cluster_is_zero() -> None:
    weighted, per = purity([0, 0], [0, 0], 3)
    assert per[2] == 0.0
    assert weighted == pytest.approx(1.0)


def test_mean_pairwise_cosine() -> None:
    assert mean_pairwise_cosine([[1.0, 0.0], [1.0, 0.0]]) == pytest.approx(1.0)
    assert mean_pairwise_cosine([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(0.0)
    assert mean_pairwise_cosine([[1.0, 0.0]]) == 0.0  # 少于两个向量
    assert mean_pairwise_cosine([[0.0, 0.0], [1.0, 0.0]]) == 0.0  # 零向量跳过


def test_onset_step_uses_the_largest_jump() -> None:
    """**最大增幅**定位，不是首次越阈值 —— 平曲线必须返回 ``None``。"""
    assert onset_step([(0, 0.1), (10, 0.12), (20, 0.5), (30, 0.55)]) == 20
    assert onset_step([(0, 0.3), (10, 0.3), (20, 0.3)]) is None
    assert onset_step([(0, 0.5), (10, 0.4)]) is None  # 只降不升
    assert onset_step([(0, 0.5)]) is None
    assert onset_step([]) is None


# ----------------------------------------------------------------------
# Sankey 的原料与真值判据
# ----------------------------------------------------------------------
def test_domain_expert_matrix_counts_every_token(corpus) -> None:
    domains = [0, 0, 1, 1, 1]
    top1 = [0, 1, 2, 2, 3]
    m = domain_expert_matrix(corpus, domains, top1)
    assert len(m) == corpus.n_domains
    assert m[0] == [1, 1, 0, 0]
    assert m[1] == [0, 0, 2, 1]
    assert sum(sum(row) for row in m) == len(domains)


def test_dominant_expert_picks_the_biggest_cell() -> None:
    assert dominant_expert([[1, 5, 0, 0], [0, 0, 3, 3]]) == [1, 2]


def test_dominant_expert_ties_go_to_the_lowest_index() -> None:
    assert dominant_expert([[2, 2, 0, 0]]) == [0]


def test_is_bijection() -> None:
    assert is_bijection([0, 1, 2, 3]) is True
    assert is_bijection([0, 0, 1, 2]) is False
    assert is_bijection([]) is True


def test_primary_experts_threshold() -> None:
    row = [70, 20, 8, 2]
    assert primary_experts(row, 0.2) == (0, 1)
    assert primary_experts(row, 0.5) == (0,)
    assert primary_experts(row, 0.9) == ()
    assert primary_experts([0, 0, 0, 0]) == ()


def test_distinct_primary_sets_accepts_pairs() -> None:
    """``top_k=2`` 时一个域由**一对**专家承担 —— 判据必须容得下这件事。

    这正是「用主导专家判真值」会误判的地方（坑 58）：下面两行主导专家都是 0，
    但它们显然各司其职。
    """
    matrix = [
        [60, 40, 0, 0],   # 主导 0，主要集合 {0,1}
        [55, 45, 0, 0],   # 主导 0，主要集合 {0,1}  ← 与上一行相同
        [0, 0, 70, 30],   # 主要集合 {2,3}
        [30, 0, 0, 70],   # 主要集合 {0,3}
    ]
    assert dominant_expert(matrix) == [0, 0, 2, 3]
    assert is_bijection(dominant_expert(matrix)) is False  # 主导专家判据误判
    assert distinct_primary_sets(matrix) is False          # 但这两行确实重复
    fixed = [matrix[0], matrix[2], matrix[3], [0, 80, 20, 0]]
    assert distinct_primary_sets(fixed) is True


def test_distinct_primary_sets_rejects_uniform_rows() -> None:
    """对照组的表现：每行近似均匀 ⇒ 每域的主要集合都是「全部专家」，完全相同。"""
    uniform = [[25, 25, 25, 25]] * 4
    assert distinct_primary_sets(uniform) is False
    assert [primary_experts(r) for r in uniform] == [(0, 1, 2, 3)] * 4


def test_distinct_primary_sets_rejects_empty_rows() -> None:
    assert distinct_primary_sets([[0, 0, 0, 0], [1, 0, 0, 0]]) is False


# ----------------------------------------------------------------------
# observe / summarize
# ----------------------------------------------------------------------
def _observed(corpus, *, step: int, prev_top1=None):
    model = build_model(_spec(), corpus.vocab_size, 1)
    batch = make_batch(corpus, corpus.test)
    logits, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    from moefactory.autograd import softmax_cross_entropy

    loss = float(softmax_cross_entropy(logits, batch.targets).data[0][0])
    correct = [
        max(range(len(row)), key=lambda j: row[j]) == batch.targets[i]
        for i, row in enumerate(logits.data)
    ]
    snap = observe(model, batch, trace, corpus, step=step, loss=loss,
                   acc=sum(correct) / len(correct), prev_top1=prev_top1,
                   domain_correct=correct)
    return snap, trace.top1


def _spec():
    from moefactory.schema import MoESpec

    return MoESpec.from_spec_id(LEARNED)


def test_observe_returns_every_documented_field(corpus) -> None:
    snap, _ = _observed(corpus, step=0)
    expected = {
        "step", "loss", "acc", "load", "load_frac", "entropy", "entropy_norm",
        "imbalance", "load_cv", "nmi", "purity", "expert_domain", "expert_sim",
        "gate_max_mean", "stability", "dropped", "dropped_rate", "comm_units",
        "active_params", "dispatched", "domain_acc",
    }
    assert expected <= set(snap)
    assert len(snap["load_frac"]) == _spec().n_experts
    assert sum(snap["load_frac"]) == pytest.approx(1.0)
    assert len(snap["domain_acc"]) == corpus.n_domains


def test_observe_stability_is_none_without_a_previous_snapshot(corpus) -> None:
    snap, top1 = _observed(corpus, step=0)
    assert snap["stability"] is None
    same, _ = _observed(corpus, step=1, prev_top1=top1)
    assert same["stability"] == pytest.approx(1.0)


def test_observe_stability_detects_change(corpus) -> None:
    snap, top1 = _observed(corpus, step=0)
    flipped = [0 if e == 1 else 1 for e in top1]
    same, _ = _observed(corpus, step=1, prev_top1=flipped)
    assert same["stability"] == pytest.approx(0.0)


def test_observe_omits_domain_accuracy_when_not_supplied(corpus) -> None:
    model = build_model(_spec(), corpus.vocab_size, 1)
    batch = make_batch(corpus, corpus.test)
    _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    snap = observe(model, batch, trace, corpus, step=0, loss=1.0, acc=0.0)
    assert "domain_acc" not in snap


def test_domain_accuracy_helper(corpus) -> None:
    acc = _domain_accuracy(corpus, [0, 0, 1, 1], [True, False, True, True])
    assert acc[0] == pytest.approx(0.5)
    assert acc[1] == pytest.approx(1.0)
    assert acc[2] == 0.0


def test_summarize_uses_the_last_snapshot_and_the_first_acc(corpus) -> None:
    s0, top1 = _observed(corpus, step=0)
    s1, _ = _observed(corpus, step=10, prev_top1=top1)
    out = summarize([s0, s1], corpus)
    assert out["loss"] == pytest.approx(s1["loss"])
    assert out["loss_first"] == pytest.approx(s0["loss"])
    assert out["loss_drop"] == pytest.approx(s0["loss"] - s1["loss"])
    assert out["acc_first"] == pytest.approx(s0["acc"])
    assert out["acc_gain"] == pytest.approx(s1["acc"] - s0["acc"])
    assert 0.0 <= out["domain_acc_mean"] <= 1.0


def test_summarize_rejects_empty_input(corpus) -> None:
    with pytest.raises(ValueError):
        summarize([], corpus)
