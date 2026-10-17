"""验收：**README 的验收标准本身**。

README 写的是：

> 能端到端完成「MoEFactory」，并在其「核心」（Expert Routing、负载均衡、可视化）上
> 给出可复现的量化结果：
>
> - 给出 Load imbalance / Entropy / Dropped Tokens 的量化曲线
> - Routing Sankey 能明确回答「专家是否形成了 specialization」

所以这个文件**不用 ``fast_cfg``**。削过的配置（20 步）跑出来的 loss ≈ 2.37，
远在下界 1.13 之上 —— 用它去断言「learned 突破下界」必然是假，而那条断言
恰恰是本项目的全部意义。这里用真实协议（120 步）跑 6 个配置，约 7 秒，跑一次。

**判据是「可计算下界」，不是 NMI 阈值。** 下界 ``E_c[H(后继|c)]`` 是从任务构造里
**推**出来的：一个看不见域的路由器对每个内容 token 只能押一个后继。于是

    loss < 下界 − 容差  ⇒  路由一定用上了域信息（即发生了分化）

这条比任何拍的阈值都硬，而且它**能被推翻**：对照组按构造完美均衡却永远达不到它。
"""

from __future__ import annotations

import math

import pytest

from moefactory.observe import distinct_primary_sets
from moefactory.report import ascii_sankey

LEARNED = [
    "e4-k2-d12-f16-learned-aux0.1-cap2",
    "e4-k1-d12-f16-learned-aux0.1-cap2",
    "e4-k1-d12-f16-learned-aux0.5-cap2",
    "e4-k1-d12-f16-learned-aux0-cap2",
]
CONTROL = [
    "e4-k1-d12-f16-static-aux0-cap2",
    "e4-k1-d12-f16-random-aux0-cap2",
]
#: 分化判据最强的那个配置（``distinct_primary_sets`` 在它上面成立）。
BEST = "e4-k1-d12-f16-learned-aux0.1-cap2"
SPECIALISED = "e4-k1-d12-f16-static-aux0-cap2"


@pytest.fixture(scope="session")
def floor(cfg: dict, corpus) -> float:
    return corpus.domain_blind_floor()


@pytest.fixture(scope="session")
def threshold(cfg: dict, floor: float) -> float:
    return floor - float(cfg["expectations"]["floor_margin"])


# ----------------------------------------------------------------------
# 验收 1：下界是**算出来的**，不是标定出来的
# ----------------------------------------------------------------------
def test_the_floor_is_computable_and_nontrivial(corpus, floor: float) -> None:
    """下界必须能从任务构造里推出来 —— 这是整条判据的地基。"""
    total = 0.0
    for c in range(corpus.n_content):
        succ = corpus.successors_of(c)
        counts: dict[int, int] = {}
        for s in succ:
            counts[s] = counts.get(s, 0) + 1
        n = len(succ)
        total += -math.fsum((v / n) * math.log(v / n) for v in counts.values())
    assert floor == pytest.approx(total / corpus.n_content, abs=1e-12)
    assert floor > 0.5, "下界太接近 0 的话「突破下界」就没有信息量"


def test_shared_vocabulary_is_what_makes_it_hard(corpus) -> None:
    """内容词表在域之间共享 ⇒ 同一个 token 在不同域有不同后继。

    这是「专家分化成为必需」的唯一理由：只看内容 token 无法知道该预测什么。
    实测默认语料平均每个内容 token 有 3.25 个不同后继（4 个域里偶尔撞值）。
    """
    counts = [len(set(corpus.successors_of(c))) for c in range(corpus.n_content)]
    assert min(counts) >= 2
    assert corpus.distinct_successor_ratio() > 0.75 * corpus.n_domains


# ----------------------------------------------------------------------
# 验收 2：对照组钉死在下界以上（判据能被推翻）
# ----------------------------------------------------------------------
def test_control_never_breaks_the_floor(acceptance_runs, threshold: float) -> None:
    """``static`` / ``random`` 的路由与任务无关 ⇒ 无论训练多久都学不会域规则。"""
    for spec_id in CONTROL:
        out = acceptance_runs[spec_id]
        assert out.final["loss"] >= threshold, (
            f"{spec_id} 突破了下界（loss={out.final['loss']:.4f} < {threshold:.4f}）—— "
            "若真如此，说明判据不成立或任务构造退化"
        )
        assert out.final["below_floor"] == 0.0


def test_control_shows_no_specialisation(acceptance_runs, cfg: dict) -> None:
    limit = float(cfg["expectations"]["control_nmi"])
    for spec_id in CONTROL:
        assert acceptance_runs[spec_id].final["nmi"] < limit
        assert distinct_primary_sets(acceptance_runs[spec_id].domain_expert) is False


# ----------------------------------------------------------------------
# 验收 3：learned 突破下界（这才是「路由用上了域信息」）
# ----------------------------------------------------------------------
def test_learned_breaks_the_floor(acceptance_runs, threshold: float) -> None:
    for spec_id in LEARNED:
        out = acceptance_runs[spec_id]
        assert out.final["loss"] < threshold, f"{spec_id} 没能突破下界"
        assert out.final["below_floor"] == 1.0
        assert out.final["loss"] < out.final["domain_blind_floor"]


def test_learned_specialises(acceptance_runs, cfg: dict) -> None:
    need = float(cfg["expectations"]["specialized_nmi"])
    for spec_id in LEARNED:
        assert acceptance_runs[spec_id].final["nmi"] >= need


def test_the_gap_between_learned_and_control_is_large(acceptance_runs) -> None:
    """不是「略好一点」：learned 的最差也远好过对照组的最好。"""
    worst_learned = max(acceptance_runs[s].final["loss"] for s in LEARNED)
    best_control = min(acceptance_runs[s].final["loss"] for s in CONTROL)
    assert worst_learned * 2.0 < best_control


def test_aux_weight_is_not_required_for_specialisation(acceptance_runs) -> None:
    """``aux=0``（完全关掉辅助损失）照样突破下界 —— 分化来自「路由会被任务梯度塑形」。"""
    assert acceptance_runs["e4-k1-d12-f16-learned-aux0-cap2"].final["below_floor"] == 1.0


# ----------------------------------------------------------------------
# 验收 4（核心命题）：均衡与分化是两个独立的轴
# ----------------------------------------------------------------------
def test_balance_does_not_imply_specialisation(acceptance_runs, cfg: dict) -> None:
    """**同一个均衡水平上，分化度可以相差数倍。**

    ``static`` 按构造完美均衡（H ≈ 1.0）而分化度为零；learned 也能达到同样的均衡，
    却明确分化了。所以「按均衡挑配置」会挑到那个结构上学不会任务的配置 ——
    这就是本项目要说的那句话，而且它是**可判定**的。
    """
    perfect = float(cfg["expectations"]["balance_perfect"])
    static = acceptance_runs[SPECIALISED]
    assert static.final["entropy_norm"] >= perfect
    assert static.final["nmi"] < float(cfg["expectations"]["control_nmi"])

    balanced_learned = [s for s in LEARNED if acceptance_runs[s].final["entropy_norm"] >= perfect]
    assert balanced_learned, "没有一个 learned 配置达到与 static 同等的均衡水平"
    for spec_id in balanced_learned:
        assert acceptance_runs[spec_id].final["nmi"] >= float(cfg["expectations"]["specialized_nmi"])


def test_specialisation_separates_by_a_large_factor(acceptance_runs) -> None:
    best_learned_nmi = max(acceptance_runs[s].final["nmi"] for s in LEARNED)
    worst_control_nmi = max(acceptance_runs[s].final["nmi"] for s in CONTROL)
    assert best_learned_nmi > 5.0 * worst_control_nmi


# ----------------------------------------------------------------------
# 验收 5：Routing Sankey 能明确回答「有没有形成 specialization」
# ----------------------------------------------------------------------
def test_learned_sankey_shows_domains_owned_by_distinct_experts(acceptance_runs) -> None:
    out = acceptance_runs[BEST]
    matrix = out.domain_expert
    shares = [max(row) / sum(row) for row in matrix]
    assert max(shares) >= 0.5, f"没有哪个域被某个专家主导：{shares}"


def test_control_sankey_shows_no_ownership(acceptance_runs) -> None:
    """对照组的每一行都近似均匀 —— 没有任何域「属于」某个专家。"""
    matrix = acceptance_runs[SPECIALISED].domain_expert
    for row in matrix:
        assert max(row) / sum(row) < 0.5


def test_learned_is_more_concentrated_than_control(acceptance_runs) -> None:
    def peak(spec_id: str) -> float:
        return max(max(r) / sum(r) for r in acceptance_runs[spec_id].domain_expert)

    assert peak(BEST) > peak(SPECIALISED) + 0.1


def test_ascii_sankey_renders_one_row_per_domain(acceptance_runs, corpus) -> None:
    out = acceptance_runs[BEST]
    experts = [f"E{i}" for i in range(len(out.domain_expert[0]))]
    text = ascii_sankey(out.domain_expert, experts)
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("域")]
    assert len(rows) == corpus.n_domains
    assert "→" in text


def test_ascii_sankey_answers_the_specialisation_question(acceptance_runs) -> None:
    """README 的验收标准就落在这一行输出上。"""
    out = acceptance_runs[BEST]
    experts = [f"E{i}" for i in range(len(out.domain_expert[0]))]
    learned_text = ascii_sankey(out.domain_expert, experts)
    control_out = acceptance_runs[SPECIALISED]
    control_text = ascii_sankey(control_out.domain_expert,
                                [f"E{i}" for i in range(len(control_out.domain_expert[0]))])
    assert "分化" in learned_text
    assert "分化" not in control_text
    assert "塌缩" not in learned_text


def test_truth_is_recovered_in_the_primary_expert_sets(acceptance_runs) -> None:
    assert distinct_primary_sets(acceptance_runs[BEST].domain_expert) is True
    for spec_id in CONTROL:
        assert distinct_primary_sets(acceptance_runs[spec_id].domain_expert) is False


# ----------------------------------------------------------------------
# 验收 6：量化曲线（Load imbalance / Entropy / Dropped Tokens）
# ----------------------------------------------------------------------
def test_every_run_has_a_quantitative_curve(acceptance_runs) -> None:
    for spec_id in LEARNED + CONTROL:
        snaps = acceptance_runs[spec_id].snapshots
        assert len(snaps) >= 4, f"{spec_id} 的曲线点太少"
        assert [s["step"] for s in snaps] == sorted(s["step"] for s in snaps)
        for s in snaps:
            for key in ("loss", "entropy_norm", "imbalance", "load_cv",
                        "nmi", "purity", "dropped_rate"):
                assert isinstance(s[key], float), f"{spec_id} 的快照缺少 {key}"


def test_every_run_starts_at_chance(corpus, acceptance_runs) -> None:
    """step 0 的 loss 必须贴着 ``ln(vocab)``、acc 贴着 0 —— 否则「训练前」就不干净。"""
    chance = math.log(corpus.vocab_size)
    for spec_id, out in acceptance_runs.items():
        first = out.snapshots[0]
        assert first["step"] == 0
        assert abs(first["loss"] - chance) < 0.15, spec_id
        assert first["acc"] < 0.25, spec_id


def test_learned_curve_actually_moves(acceptance_runs) -> None:
    """曲线必须**动**：loss 降下去、acc 升上来、分化是训练出来的。

    **不能拿 step 0 当「无分化」的基线。** 随机初始化的 tag 嵌入已经让路由带上
    标签信息，于是 step 0 的 nmi 可能相当高（实测某个配置是 0.38，比训练中途还高）。
    所以判据是「最终分化度远高于曲线上的最低点」，而不是「高于起点」。
    """
    for spec_id in LEARNED:
        snaps = acceptance_runs[spec_id].snapshots
        assert snaps[-1]["loss"] < snaps[0]["loss"], spec_id
        assert snaps[-1]["acc"] > snaps[0]["acc"], spec_id
        nmis = [s["nmi"] for s in snaps]
        assert snaps[-1]["nmi"] > 1.5 * min(nmis), spec_id


def test_stability_is_only_defined_after_the_first_snapshot(acceptance_runs) -> None:
    snaps = acceptance_runs[BEST].snapshots
    assert snaps[0]["stability"] is None
    assert all(s["stability"] is not None for s in snaps[1:])


def test_learned_has_a_specialisation_onset(acceptance_runs) -> None:
    """分化不是一开始就有的 —— 它有起始步，这正说明它是**学出来的**。"""
    out = acceptance_runs[BEST]
    assert out.onset_step is not None
    assert 0 < out.onset_step <= out.snapshots[-1]["step"]


def test_control_has_no_specialisation_onset(acceptance_runs) -> None:
    assert acceptance_runs[SPECIALISED].onset_step is None


# ----------------------------------------------------------------------
# 验收 7：可复现
# ----------------------------------------------------------------------
def test_every_run_is_internally_consistent(acceptance_runs) -> None:
    for spec_id, out in acceptance_runs.items():
        floor = out.final["domain_blind_floor"]
        assert out.final["below_floor"] == (1.0 if out.final["loss"] < floor - 0.1 else 0.0)
        assert out.params > 0 and out.flops > 0
        assert out.final["active_params"] < out.params, "稀疏激活必须真的稀疏"
        assert len(out.domain_dominant) == len(out.domain_expert)


def test_sankey_matrix_agrees_with_the_snapshot_count(acceptance_runs, corpus) -> None:
    """Sankey 的流量必须等于评测集的 token 数 —— 否则它在画另一件事。"""
    n_tokens = len(corpus.test) * (corpus.seq_len - 1)
    for spec_id, out in acceptance_runs.items():
        assert sum(sum(row) for row in out.domain_expert) == n_tokens, spec_id


def test_domain_expert_matrix_rebuilds_from_the_curve(acceptance_runs) -> None:
    """矩阵与快照同源：用最后一次快照的 top1 重建，必须与 ``domain_expert`` 一致。"""
    out = acceptance_runs[BEST]
    last = out.snapshots[-1]
    assert sum(last["load"]) == sum(sum(row) for row in out.domain_expert)
    assert len(out.domain_expert) == len(out.domain_expert[0])
