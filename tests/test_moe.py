"""标本本体：路由 → top-k → 容量 → **稀疏**派发 → 专家 → 合并。

这里要钉住四件事，每一件都能独立地让整个项目变成废话：

1. **稀疏派发是真的。** 专家只对派发给它的 token 做前向。若把每个 token 都送进每个
   专家再乘掩码，``active_params`` 就等于总参数量 —— MoE 的存在理由就没了。
2. **两条信息通路是分开的。** 路由器看 ``emb[token]+emb[tag]``，专家只看
   ``emb[token]``。若专家也能看到标签，一个专家就能通吃全部域，分化退化成装饰。
3. **对照组真的不看 token。** ``static`` / ``random`` 的门控不经过路由器，
   所以它们**没有可导门控**（``gates_t is None``）—— 训练器因此不需要任何
   ``if routing == ...`` 分支。
4. **容量溢出按门控丢弃，且保留残差。** 丢弃不该表现为「模型崩了」。
"""

from __future__ import annotations

import pytest

from moefactory.corpus import make_batch
from moefactory.moe import MoEModel, RouteTrace, build_model
from moefactory.schema import ROUTINGS, MoESpec, imbalance_ratio, normalized_entropy

LEARNED = "e4-k2-d12-f16-learned-aux0.1-cap2"
STATIC = "e4-k1-d12-f16-static-aux0-cap2"
RANDOM = "e4-k1-d12-f16-random-aux0-cap2"


def _model(spec_id: str, corpus, seed: int = 1) -> MoEModel:
    return build_model(MoESpec.from_spec_id(spec_id), corpus.vocab_size, seed)


@pytest.fixture
def batch(corpus):
    return make_batch(corpus, corpus.test)


# ----------------------------------------------------------------------
# 路由方式：三条对照骨架
# ----------------------------------------------------------------------
def test_learned_gates_form_a_distribution_and_are_differentiable(corpus, batch) -> None:
    model = _model(LEARNED, corpus)
    logits, trace, gates = model.forward(batch.inputs, batch.tags, collect=True)
    assert gates is not None, "learned 路由必须给出可导门控（辅助损失要用）"
    assert trace is not None
    assert len(gates.data) == len(batch.inputs)
    for row in gates.data:
        assert sum(row) == pytest.approx(1.0)
        assert all(v >= 0.0 for v in row)


def test_control_routings_have_no_differentiable_gates(corpus, batch) -> None:
    """``static`` / ``random`` 不经过路由器 ⇒ ``gates_t is None``。

    正是这一条让训练器里**一个** ``if routing == ...`` 都不需要。
    """
    for spec_id in (STATIC, RANDOM):
        model = _model(spec_id, corpus)
        _, trace, gates = model.forward(batch.inputs, batch.tags, collect=True)
        assert gates is None
        assert trace is not None


def test_static_routing_is_balanced_by_construction(corpus, batch) -> None:
    """``token % n_experts`` 把 token 按残类摊平 —— **不看任务，因此不可能分化**。

    真实 batch 里各残类不是严格等频（内容 token 是随机抽的），所以实测
    ``entropy_norm`` 是 0.9973 而不是 1.0；严格相等由下面那条构造 batch 的测试钉住。
    """
    model = _model(STATIC, corpus)
    _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    assert trace is not None
    _, h_norm = normalized_entropy(trace.load)
    assert h_norm >= 0.99
    assert imbalance_ratio(trace.load) <= 1.1


def test_static_routing_is_exactly_balanced_on_a_balanced_batch(corpus) -> None:
    """构造一个各残类等频的 batch：负载必须**逐位相等**，``H == 1.0``。"""
    from moefactory.corpus import Batch

    n = 64
    inputs = [i % 4 for i in range(n)]
    batch = Batch(
        inputs=inputs,
        tags=[corpus.tag_of(i % 4) for i in range(n)],
        targets=[0] * n,
        domains=[i % 4 for i in range(n)],
        sample_ids=["synthetic"],
    )
    model = _model(STATIC, corpus)
    _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    assert trace is not None
    assert trace.load == [n // 4] * 4
    assert normalized_entropy(trace.load)[1] == pytest.approx(1.0)
    assert imbalance_ratio(trace.load) == pytest.approx(1.0)


def test_static_routing_never_changes_during_training(fast_cfg: dict, fast_corpus) -> None:
    """「按构造均衡」的真正含义：它的路由**与参数无关**，训练也改不动它。"""
    from moefactory.train import run_spec

    spec = MoESpec.from_spec_id("e4-k1-d12-f16-static-aux0-cap1")
    out = run_spec(spec, fast_corpus, fast_cfg, 1)
    first = out.snapshots[0]
    for snap in out.snapshots[1:]:
        assert snap["load"] == first["load"]
        assert snap["nmi"] == pytest.approx(first["nmi"])
        assert snap["purity"] == pytest.approx(first["purity"])


def test_static_routing_shows_no_specialisation(corpus, batch) -> None:
    """完美均衡 + 零分化：这就是「均衡 ≠ 分化」的可判定形式。"""
    from moefactory.observe import (
        distinct_primary_sets,
        domain_expert_matrix,
        normalized_mutual_information,
    )

    model = _model(STATIC, corpus)
    _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    nmi = normalized_mutual_information(batch.domains, trace.top1)
    assert nmi < 0.15
    matrix = domain_expert_matrix(corpus, batch.domains, trace.top1)
    # 每一行都近似均匀 ⇒ 每个域的主要专家集合都是「全部专家」，两两相同
    assert distinct_primary_sets(matrix) is False


def test_static_and_random_completely_ignore_the_task_tag(corpus, batch) -> None:
    """对照组完全不看标签：同一批输入换标签，输出**逐位不变**。"""
    for spec_id in (STATIC, RANDOM):
        model = _model(spec_id, corpus)
        a = model.forward(batch.inputs, batch.tags, collect=False)[0].data
        other = [corpus.tag_of((d + 1) % corpus.n_domains) for d in batch.domains]
        b = model.forward(batch.inputs, other, collect=False)[0].data
        assert a == b


def test_learned_routing_does_use_the_task_tag(corpus, batch) -> None:
    """被测对象必须看标签 —— 否则「路由器用上了域信息」无从谈起。"""
    model = _model(LEARNED, corpus)
    a = model.forward(batch.inputs, batch.tags, collect=False)[2].data
    other = [corpus.tag_of((d + 1) % corpus.n_domains) for d in batch.domains]
    b = model.forward(batch.inputs, other, collect=False)[2].data
    assert a != b


def test_random_routing_is_deterministic_per_seed(corpus, batch) -> None:
    m1, m2 = _model(RANDOM, corpus, 5), _model(RANDOM, corpus, 5)
    t1 = m1.forward(batch.inputs, batch.tags, collect=True)[1]
    t2 = m2.forward(batch.inputs, batch.tags, collect=True)[1]
    assert t1.top1 == t2.top1
    assert _model(RANDOM, corpus, 6).forward(batch.inputs, batch.tags, collect=True)[1].top1 != t1.top1


def test_all_declared_routings_can_be_built(corpus, batch) -> None:
    for routing in ROUTINGS:
        spec = MoESpec(n_experts=4, top_k=1, routing=routing)
        model = build_model(spec, corpus.vocab_size, 1)
        _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
        assert trace is not None and len(trace.top1) == len(batch.inputs)


# ----------------------------------------------------------------------
# top-k 选择
# ----------------------------------------------------------------------
def test_top_k_picks_the_highest_gates(corpus) -> None:
    gates = [[0.1, 0.5, 0.3, 0.1]]
    one = build_model(MoESpec(n_experts=4, top_k=1, routing="learned"), corpus.vocab_size, 1)
    assert one._top_k(gates) == [[1]]
    two = build_model(MoESpec(n_experts=4, top_k=2, routing="learned"), corpus.vocab_size, 1)
    assert two._top_k(gates) == [[1, 2]]


def test_top_k_ties_break_by_index(corpus) -> None:
    model = _model(LEARNED, corpus)
    assert model._top_k([[0.25, 0.25, 0.25, 0.25]]) == [[0, 1]]


def test_top1_is_the_highest_gate_among_kept(corpus, batch) -> None:
    model = _model(LEARNED, corpus)
    _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    assert trace is not None
    for t, kept in enumerate(trace.kept):
        if kept:
            assert trace.top1[t] in kept
            assert trace.gates[t][trace.top1[t]] == max(trace.gates[t][i] for i in kept)


# ----------------------------------------------------------------------
# 稀疏派发
# ----------------------------------------------------------------------
def test_dispatch_is_genuinely_sparse(corpus, batch) -> None:
    """``active_params`` 必须**小于**总参数量 —— 否则这就是个稠密网络。"""
    model = _model(LEARNED, corpus)
    spec = model.spec
    _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    assert trace is not None
    assert spec.top_k < spec.n_experts

    assert trace.dispatched == sum(len(k) for k in trace.kept)
    assert sum(trace.load) == trace.dispatched
    assert trace.dispatched <= len(batch.inputs) * spec.top_k
    assert trace.active_params < model.num_params
    assert trace.active_params >= model.num_params - model.expert_params
    assert trace.comm_units == trace.dispatched * 2 * model.d_model


def test_dense_top_k_reaches_full_param_count(corpus, batch) -> None:
    """边界：``top_k == n_experts`` 时稀疏性消失，激活参数等于全部参数。"""
    spec = MoESpec(n_experts=4, top_k=4, routing="learned", capacity_factor=1.0)
    model = build_model(spec, corpus.vocab_size, 1)
    _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    assert trace is not None
    assert trace.dispatched == len(batch.inputs) * 4
    assert trace.active_params == model.num_params


def test_parameters_are_returned_sorted_by_name(corpus) -> None:
    """顺序与遍历路径无关 —— 否则梯度累积顺序会变成一个隐式变量。"""
    names = [n for n, _ in _model(LEARNED, corpus).parameters()]
    assert names == sorted(names)
    assert "expert0.W1" in names and "router.W" in names


def test_expert_params_are_disjoint_from_shared(corpus) -> None:
    model = _model(LEARNED, corpus)
    assert 0 < model.expert_params < model.num_params
    report = model.param_report()
    assert report["total"] == report["expert"] + report["shared"] == model.num_params


def test_param_count_matches_the_model(corpus) -> None:
    """``spec.param_count`` 必须与模型实际参数量**逐位相等**。

    它是 ``explain`` 与「稀疏激活比例」的口径来源。曾经把它写成 ``2 * d_model``
    （以为有两个 RMS 缩放，实际只有一个），于是 ``explain`` 报的数比模型多出
    ``d_model`` —— 而没有任何地方会因此报错（坑 59）。
    """
    for spec_id in (LEARNED, STATIC, "e4-k1-d12-f16-learned-aux0-cap1"):
        model = _model(spec_id, corpus)
        assert model.num_params == model.spec.param_count(corpus.vocab_size)


# ----------------------------------------------------------------------
# 容量与丢弃
# ----------------------------------------------------------------------
def test_capacity_per_expert_formula() -> None:
    spec = MoESpec(n_experts=4, top_k=2, capacity_factor=1.5)
    assert spec.capacity_per_expert(100) == pytest.approx(1.5 * 100 * 2 / 4)
    assert MoESpec(n_experts=4, top_k=1, capacity_factor=0.0).capacity_per_expert(100) == float("inf")


def test_tight_capacity_drops_overflow(corpus, batch) -> None:
    spec = MoESpec(n_experts=4, top_k=1, routing="learned", aux_weight=0.1, capacity_factor=0.05)
    model = build_model(spec, corpus.vocab_size, 1)
    logits, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    assert trace is not None
    assert trace.dropped > 0
    assert trace.dropped_rate > 0.5
    assert trace.n_assignments == len(batch.inputs) * 1
    assert trace.dropped + trace.dispatched == trace.n_assignments
    # 丢弃保留残差通路 ⇒ 输出仍是有限值，不是「模型崩了」
    assert all(all(v == v for v in row) for row in logits.data)
    assert any(any(abs(v) > 0.0 for v in row) for row in logits.data)


def test_dropped_tokens_still_get_a_top1(corpus, batch) -> None:
    """全被丢弃的 token 也要有一个主导专家，否则 Sankey 会出现空洞。"""
    spec = MoESpec(n_experts=4, top_k=1, routing="learned", capacity_factor=0.01)
    model = build_model(spec, corpus.vocab_size, 1)
    _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    assert trace is not None
    assert len(trace.top1) == len(batch.inputs)
    assert all(0 <= e < spec.n_experts for e in trace.top1)


def test_unlimited_capacity_never_drops(corpus, batch) -> None:
    spec = MoESpec(n_experts=4, top_k=2, routing="learned", capacity_factor=0.0)
    model = build_model(spec, corpus.vocab_size, 1)
    _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    assert trace is not None
    assert trace.dropped == 0
    assert trace.dropped_rate == 0.0


def test_dropped_rate_uses_assignment_count(corpus, batch) -> None:
    spec = MoESpec(n_experts=4, top_k=2, routing="learned", capacity_factor=0.5)
    model = build_model(spec, corpus.vocab_size, 1)
    _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    assert trace is not None
    assert trace.n_assignments == len(batch.inputs) * 2
    assert trace.dropped_rate == pytest.approx(trace.dropped / trace.n_assignments)


# ----------------------------------------------------------------------
# 辅助损失
# ----------------------------------------------------------------------
def test_aux_loss_is_zero_when_weight_is_zero(corpus, batch) -> None:
    model = build_model(MoESpec(n_experts=4, top_k=2, routing="learned", aux_weight=0.0),
                        corpus.vocab_size, 1)
    _, trace, gates = model.forward(batch.inputs, batch.tags, collect=True)
    assert gates is not None and trace is not None
    assert float(model.aux_loss(gates, trace.load).data[0][0]) == 0.0


def test_aux_loss_is_scalar_and_pushes_gradient_into_the_gates(corpus, batch) -> None:
    model = _model(LEARNED, corpus)
    _, trace, gates = model.forward(batch.inputs, batch.tags, collect=True)
    assert gates is not None and trace is not None
    loss = model.aux_loss(gates, trace.load)
    assert loss.shape == (1, 1)
    assert float(loss.data[0][0]) > 0.0
    loss.backward()
    assert any(abs(v) > 0.0 for row in gates.grad for v in row)


def test_aux_loss_scales_with_the_weight(corpus, batch) -> None:
    """同一组门控与负载下，``α`` 翻十倍，损失必须恰好翻十倍。

    注意：**不能用两个不同 ``aux_weight`` 的模型来比** —— 它们的 ``spec_id`` 不同，
    初始化就不同，于是门控和负载都不一样，比出来的是「两个模型」而不是「一个权重」。
    """
    from dataclasses import replace

    model = build_model(MoESpec(n_experts=4, top_k=2, routing="learned", aux_weight=0.05),
                        corpus.vocab_size, 1)
    _, trace, gates = model.forward(batch.inputs, batch.tags, collect=True)
    assert gates is not None and trace is not None
    small = float(model.aux_loss(gates, trace.load).data[0][0])
    model.spec = replace(model.spec, aux_weight=0.5).canonical()
    big = float(model.aux_loss(gates, trace.load).data[0][0])
    assert small > 0.0
    assert big == pytest.approx(small * 10.0, rel=1e-12)


# ----------------------------------------------------------------------
# 物质指标 / 确定性 / 输入校验
# ----------------------------------------------------------------------
def test_expert_similarity_is_a_cosine_mean(corpus) -> None:
    model = _model(LEARNED, corpus)
    sim = model.expert_similarity()
    assert -1.0 <= sim <= 1.0


def test_experts_start_from_different_weights(corpus) -> None:
    model = _model(LEARNED, corpus)
    flats = [e.flat_weights() for e in model.experts]
    assert all(len(f) == model.d_ff * model.d_model * 2 for f in flats)
    assert len({tuple(f) for f in flats}) == model.n_experts


def test_model_is_deterministic_in_seed(corpus) -> None:
    a, b = _model(LEARNED, corpus, 3), _model(LEARNED, corpus, 3)
    assert a.emb.data == b.emb.data
    assert a.experts[0].W1.data == b.experts[0].W1.data
    c = _model(LEARNED, corpus, 4)
    assert c.emb.data != a.emb.data


def test_spec_id_changes_with_the_spec(corpus) -> None:
    assert _model(LEARNED, corpus).spec.spec_id == LEARNED
    assert _model(STATIC, corpus).spec.spec_id == STATIC


def test_forward_rejects_bad_input(corpus, batch) -> None:
    model = _model(LEARNED, corpus)
    with pytest.raises(ValueError):
        model.forward([], [], collect=True)
    with pytest.raises(ValueError):
        model.forward(batch.inputs, batch.tags[:-1], collect=True)


def test_spec_validation_rejects_impossible_configs() -> None:
    with pytest.raises(ValueError):
        MoESpec(routing="magic")
    with pytest.raises(ValueError):
        MoESpec(n_experts=4, top_k=5)
    with pytest.raises(ValueError):
        MoESpec(n_experts=1)
    with pytest.raises(ValueError):
        MoESpec(aux_weight=-0.1)
    with pytest.raises(ValueError):
        MoESpec(capacity_factor=-1.0)


def test_trace_helpers() -> None:
    t = RouteTrace(n_tokens=4, gates=[], assigned=[[0], [0, 1]], kept=[[0], []],
                   load=[1, 0], dropped=1, top1=[0, 1, 0, 0], dispatched=1,
                   active_params=1, comm_units=2)
    assert t.n_assignments == 3
    assert t.dropped_rate == pytest.approx(1 / 3)
