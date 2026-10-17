"""结论层：把观测折成**可执行结论**。

**测量型的产物是结论，不是曲线。** 一张 Sankey 只能让人「看出来」；结论要能被
引用、被反驳、被复现。所以每条结论都必须带：

- ``claim`` —— 一句可判定真假的话；
- ``supported`` —— 实测是否支持它；
- ``evidence`` —— 支撑它的具体数字（报告里直接打印，读者可以自己去核）。

**结论全部由数据算出来，不写死。** 一旦把「辅助损失会带来分化」写成常量字符串，
报告就变成宣传材料而不是测量结果 —— 这条纪律在标定阶段救过一次场：
实测结果是「辅助损失买到的是**均衡**，不是分化」，与最初的设计假设相反，
因为结论是算出来的，它才被如实报出来（见 README 附录 B）。
"""

from __future__ import annotations

from typing import Any, Sequence

from .observe import distinct_primary_sets, primary_experts
from .schema import Finding, RunRecord


def _rank(values: Sequence[float]) -> list[float]:
    """平均秩（并列取平均）。Spearman 需要它。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """秩相关。用秩而不是皮尔逊，是因为我们关心的是「单调趋势」，不是线性关系。"""
    n = len(xs)
    if n < 3:
        return 0.0
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def _by_group(runs: Sequence[RunRecord]) -> tuple[list[RunRecord], list[RunRecord]]:
    learned = [r for r in runs if r.spec["routing"] == "learned"]
    control = [r for r in runs if r.spec["routing"] != "learned"]
    return learned, control


def compute_findings(
    runs: Sequence[RunRecord],
    *,
    floor: float,
    margin: float,
    grid: dict[str, Any],
    replication: dict[str, Any] | None,
    expectations: dict[str, float],
) -> list[Finding]:
    """把全部观测折成一组结论。**顺序即报告里的呈现顺序。**

    统计口径分两层，**不要混**：

    - **网格行**（``origin == "grid"``）用于「网格完整性」「均衡 × 分化」「aux 轴」
      这类与扇出结构有关的断言 —— 复核行不属于网格，混进去会把 24 格数成 39 格。
    - **全部行**用于「对照组是否停在下界」「learned 是否突破下界」这类关于
      *分离是否成立* 的断言 —— 这里样本越多越好。
    """
    out: list[Finding] = []
    grid_runs = [r for r in runs if r.origin == "grid"]
    learned_all, control_all = _by_group(runs)
    learned, control = _by_group(grid_runs)
    thr = floor - margin
    specialized = [r for r in grid_runs if r.final["loss"] < thr]

    # 1. 扇出完整性 —— 图没漏分支，后面所有统计才有意义
    declared = {g["name"]: g["declared"] for g in grid["groups"]}
    per_group: dict[str, int] = {}
    for r in grid_runs:
        key = "learned" if r.spec["routing"] == "learned" else "control"
        per_group[key] = per_group.get(key, 0) + 1
    n_learned = per_group.get("learned", 0)
    n_control = per_group.get("control", 0)
    ok_grid = (
        n_learned == declared.get("learned", n_learned)
        and n_control == declared.get("control", n_control)
    )
    out.append(
        Finding(
            key="grid_complete",
            claim="扇出网格是完整的：每个组的格数 == 各轴取值数之积，没有漏分支。",
            supported=ok_grid,
            evidence={
                "learned": f"{n_learned}/{declared.get('learned')}",
                "control": f"{n_control}/{declared.get('control')}",
                "grid_total": len(grid_runs),
                "replication": len(runs) - len(grid_runs),
            },
        )
    )

    # 2. 对照组全部停在下界 —— 「按公式摊出来的均衡」学不会任务
    ctl_above = [r for r in control_all if r.final["loss"] >= thr]
    ctl_loss = [r.final["loss"] for r in control_all]
    ctl_nmi = [r.final["nmi"] for r in control_all]
    out.append(
        Finding(
            key="control_pinned_at_floor",
            claim=(
                f"对照组（static / random）**全部**停在下界以上：loss ≥ {thr:.3f}。"
                "它们的路由与任务无关，因此无论训练多久都不可能学会域规则。"
            ),
            supported=len(ctl_above) == len(control_all) and bool(control_all),
            evidence={
                "floor": round(floor, 4),
                "threshold": round(thr, 4),
                "count": f"{len(ctl_above)}/{len(control_all)}",
                "control_loss_range": (
                    f"{min(ctl_loss):.3f}~{max(ctl_loss):.3f}" if ctl_loss else "—"
                ),
                "control_nmi_max": round(max(ctl_nmi), 4) if ctl_nmi else 0.0,
            },
        )
    )

    # 3. learned 突破下界 —— 路由确实用上了域信息
    ok_frac = (
        len([r for r in learned_all if r.final["loss"] < thr]) / len(learned_all)
        if learned_all
        else 0.0
    )
    need = float(expectations.get("min_specialized_frac", 0.6))
    out.append(
        Finding(
            key="learned_breaks_floor",
            claim=(
                f"学出来的路由能突破可计算下界：loss < {thr:.3f} 的比例 ≥ {need:.0%}。"
                "这是「路由用上了域信息」的硬证据 —— 下界是从任务构造里推出来的，不是拍的阈值。"
            ),
            supported=ok_frac >= need,
            evidence={
                "fraction": f"{ok_frac:.0%}",
                "need": f"{need:.0%}",
                "count": f"{len([r for r in learned_all if r.final['loss'] < thr])}/{len(learned_all)}",
                "best_loss": round(min(r.final["loss"] for r in learned_all), 4) if learned_all else None,
                "best_spec": (
                    min(learned_all, key=lambda r: r.final["loss"]).spec_id if learned_all else None
                ),
            },
        )
    )

    # 4. **核心**：均衡指标指向的方向是错的
    best_balance = max(grid_runs, key=lambda r: r.final["entropy_norm"]) if grid_runs else None
    rho_bal_loss = spearman(
        [r.final["entropy_norm"] for r in grid_runs], [r.final["loss"] for r in grid_runs]
    )
    rho_bal_loss_lr = spearman(
        [r.final["entropy_norm"] for r in learned], [r.final["loss"] for r in learned]
    )
    rho_spec_loss = spearman(
        [r.final["nmi"] for r in grid_runs], [r.final["loss"] for r in grid_runs]
    )
    ok_trap = (
        best_balance is not None
        and best_balance.spec["routing"] != "learned"
        and rho_bal_loss > 0.3
        and abs(rho_bal_loss_lr) < 0.5
        and rho_spec_loss < -0.5
    )
    out.append(
        Finding(
            key="balance_points_the_wrong_way",
            claim=(
                "**均衡指标指向的方向是错的。** 全网格上「均衡 ↔ loss」是**正**相关"
                "（越均衡反而越差），learned 组内则几乎不相关；而「分化 ↔ loss」强负相关。"
                "若按均衡排序选配置，第一名是对照组 —— 它按构造完美均衡，loss 却是全场最差。"
            ),
            supported=ok_trap,
            evidence={
                "spearman_balance_vs_loss_all": round(rho_bal_loss, 3),
                "spearman_balance_vs_loss_learned_only": round(rho_bal_loss_lr, 3),
                "spearman_nmi_vs_loss": round(rho_spec_loss, 3),
                "most_balanced_spec": best_balance.spec_id if best_balance else None,
                "most_balanced_entropy_norm": (
                    round(best_balance.final["entropy_norm"], 4) if best_balance else None
                ),
                "most_balanced_loss": round(best_balance.final["loss"], 4) if best_balance else None,
                "most_balanced_nmi": round(best_balance.final["nmi"], 4) if best_balance else None,
            },
            note=(
                "均衡与分化是两个独立的轴。把它们揉成一个「路由质量」分数，"
                "这个陷阱就看不见了 —— 而按均衡排序恰好会选到那个结构上学不会任务的配置。"
            ),
        )
    )

    # 5. **辅助损失这一轴几乎是平的** —— 分化不靠它
    lr_runs = learned
    aux0 = [r for r in lr_runs if r.spec["aux_weight"] == 0.0]
    means: dict[float, float] = {}
    for aux in sorted({r.spec["aux_weight"] for r in lr_runs}):
        rs = [r for r in lr_runs if r.spec["aux_weight"] == aux]
        means[aux] = sum(r.final["loss"] for r in rs) / len(rs)
    aux0_loss = means.get(0.0, float("inf"))
    aux0_nmi = sum(r.final["nmi"] for r in aux0) / len(aux0) if aux0 else 0.0
    aux0_ok = sum(1 for r in aux0 if r.final["below_floor"] > 0.5)
    spread = ((max(means.values()) - min(means.values())) / min(means.values())) if means else 0.0
    ctl_mean = sum(r.final["loss"] for r in control) / len(control) if control else 0.0
    gap = (ctl_mean / min(means.values())) if means and min(means.values()) > 0 else 0.0
    rho_h = spearman(
        [r.spec["aux_weight"] for r in lr_runs], [r.final["entropy_norm"] for r in lr_runs]
    ) if lr_runs else 0.0
    ok_aux = bool(aux0) and aux0_ok == len(aux0) and spread < 0.35 and gap > 3.0
    out.append(
        Finding(
            key="aux_axis_is_flat",
            claim=(
                "**辅助损失不是分化的必要条件，这一轴几乎是平的。** ``aux=0`` 的全部格子都"
                "突破了下界；四个 aux 取值的平均 loss 落在一条窄带内，"
                "而 learned 与对照组之间差数倍。分化来自「路由会被任务梯度塑形」这件事本身。"
            ),
            supported=ok_aux,
            evidence={
                "aux0_break_floor": f"{aux0_ok}/{len(aux0)}",
                "per_aux_mean_loss": {f"{k:g}": round(v, 4) for k, v in means.items()},
                "aux_axis_spread": f"{spread:.1%}",
                "learned_vs_control_gap": f"{gap:.2f}×",
                "aux0_mean_nmi": round(aux0_nmi, 4),
                "spearman_aux_vs_balance": round(rho_h, 3),
            },
            note=(
                "这条与最初的设计假设相反（原以为 aux 是分化的必要条件）。"
                "更进一步的实测：aux 与均衡的关系也很弱且不单调（Spearman 仅 "
                f"{rho_h:.2f}，逐轴看有反向的格子），所以「辅助损失买到均衡」"
                "这个常见说法在本任务里也没站住。"
            ),
        )
    )

    # 6. 真值复原：每个域由**不同**的主要专家集合承担
    best = min(specialized, key=lambda r: r.final["loss"]) if specialized else None
    ctl_worst = max(control, key=lambda r: r.final["nmi"]) if control else None
    ok_bij = bool(best) and distinct_primary_sets(best.domain_expert)
    ctl_distinct = bool(ctl_worst) and distinct_primary_sets(ctl_worst.domain_expert)
    out.append(
        Finding(
            key="truth_recovered",
            claim=(
                "在突破下界的配置里，**每个域由不同的专家组合承担**（域 → 主要专家集合"
                "两两不同）；对照组则所有域的主要专家集合完全相同 —— 这就是「没有分化」"
                "的可判定形式。"
            ),
            supported=ok_bij and not ctl_distinct,
            evidence={
                "best_spec": best.spec_id if best else None,
                "best_loss": round(best.final["loss"], 4) if best else None,
                "best_primary_sets": (
                    [list(primary_experts(r)) for r in best.domain_expert] if best else None
                ),
                "best_all_distinct": ok_bij,
                "control_spec": ctl_worst.spec_id if ctl_worst else None,
                "control_primary_sets": (
                    [list(primary_experts(r)) for r in ctl_worst.domain_expert] if ctl_worst else None
                ),
                "control_all_distinct": ctl_distinct,
                "best_purity": round(best.final["purity"], 4) if best else None,
            },
            note=(
                "判据用「主要专家集合」而不是「主导专家」：top_k=2 时一个域由一对专家"
                "共同承担，两个域的主导专家可以相同而依然各司其职 —— 用主导专家会误判。"
            ),
        )
    )

    # 7. 多种子复核：对照组 0/N 全灭、learned N/N 全中
    if replication:
        ok_rep = (
            replication.get("control_success", 0) == 0
            and replication.get("learned_success", 0) == replication.get("learned_total", 0)
            and replication.get("learned_total", 0) > 0
        )
        out.append(
            Finding(
                key="replication_separates",
                claim=(
                    "多种子复核下分离依然成立：learned 全部突破下界，对照组一次都没有。"
                    "单次运行可能是运气，0/N 与 N/N 的对比才说明这是系统性的。"
                ),
                supported=ok_rep,
                evidence={
                    "learned": f"{replication.get('learned_success')}/{replication.get('learned_total')}",
                    "control": f"{replication.get('control_success')}/{replication.get('control_total')}",
                    "seeds": replication.get("seeds"),
                },
            )
        )
    return out
