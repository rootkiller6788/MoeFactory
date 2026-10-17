"""编排：展开网格 → 逐个训练观测 → 多种子复核 → 收拢 → 落盘。

**这个形状没有「选优」。** 网格里的每个格子都要跑、都要记、都要进矩阵 ——
因为产物是**证据**，不是冠军。任何一步「只保留最好的那个」都会把对比轴删掉，
让「均衡与分化的张力」这类结论无从谈起。

**只读命令从产物重建结果对象，不重跑流水线**（同坑 41）。本项目整条流水线
约 90 秒，若每条读命令都重算，用户会以为工具坏了。所以 ``load_result`` 是
一等公民：它把 ``runs.json`` / ``findings.json`` / ``version.json`` 读回来，
重建出与 ``run_pipeline`` 同形的结果对象。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .config import (
    expand_grid,
    expand_replication,
    expectations_of,
    floor_margin,
    grid_report,
    load_config,
    version_digest,
)
from .corpus import build_corpus
from .findings import compute_findings
from .lineage import Lineage, build_lineage, dumps as dumps_lineage
from .schema import Finding, MoESpec, RunRecord, Version, utcnow
from .train import protocol_digest, run_spec

#: 必需产物。缺任何一个，``load_result`` 必须返回 ``None``（不能「读一半」）。
OUT_FILES: tuple[str, ...] = (
    "runs.json",
    "matrix.csv",
    "curves.json",
    "sankey.json",
    "findings.json",
    "replication.json",
    "lineage.json",
    "version.json",
)

#: 只有实测耗时允许不可逐位复现。
TIMING_FIELDS: frozenset[str] = frozenset({"train_seconds"})

#: 矩阵表的列（顺序即 CSV 的列序）。
MATRIX_COLUMNS: tuple[str, ...] = (
    "group", "spec_id", "seed", "routing", "aux_weight", "top_k", "capacity_factor",
    "loss", "acc", "entropy_norm", "imbalance", "load_cv", "nmi", "purity",
    "expert_sim", "gate_max_mean", "stability", "dropped_rate", "comm_units",
    "active_params", "below_floor", "onset_step", "params", "train_seconds",
)


@dataclass
class PipelineResult:
    """测量型的产物：全部观测 + 结论 + 版本 + 图。**没有「最优解」字段。**"""

    runs: list[RunRecord]
    findings: list[Finding]
    version: Version
    grid: dict[str, Any]
    replication: dict[str, Any]
    corpus_meta: dict[str, Any]
    dag: Lineage
    objectives: tuple[str, ...] = ()

    # -- 查询 --------------------------------------------------------
    def find(self, spec_id: str, seed: int | None = None) -> RunRecord | None:
        for r in self.runs:
            if r.spec_id == spec_id and (seed is None or r.seed == seed):
                return r
        return None

    @property
    def grid_runs(self) -> list[RunRecord]:
        """只属于**扇出网格**的观测（不含多种子复核）。网格完整性等断言只看它。"""
        return [r for r in self.runs if r.origin == "grid"]

    @property
    def replication_runs(self) -> list[RunRecord]:
        return [r for r in self.runs if r.origin == "replication"]

    def by_routing(self, routing: str) -> list[RunRecord]:
        return [r for r in self.runs if r.spec["routing"] == routing]

    def by_group(self, group: str) -> list[RunRecord]:
        """``learned`` 组 = 学出来的路由；``control`` 组 = 对照组。"""
        if group == "learned":
            return self.by_routing("learned")
        if group == "control":
            return [r for r in self.runs if r.spec["routing"] != "learned"]
        raise KeyError(f"没有分组 {group}")

    @property
    def floor(self) -> float:
        return float(self.runs[0].final["domain_blind_floor"]) if self.runs else 0.0

    def best_by_loss(self) -> RunRecord | None:
        return min(self.runs, key=lambda r: r.final["loss"]) if self.runs else None

    def most_balanced(self) -> RunRecord | None:
        return max(self.runs, key=lambda r: r.final["entropy_norm"]) if self.runs else None

    def spec_ids(self) -> list[str]:
        return sorted({r.spec_id for r in self.runs})

    def finding(self, key: str) -> Finding | None:
        for f in self.findings:
            if f.key == key:
                return f
        return None


def _record(spec: MoESpec, seed: int, outcome: Any, origin: str = "grid") -> RunRecord:
    return RunRecord(
        spec_id=spec.spec_id,
        spec=spec.to_dict(),
        seed=seed,
        final=dict(outcome.final),
        snapshots=[dict(s) for s in outcome.snapshots],
        domain_expert=[list(row) for row in outcome.domain_expert],
        domain_dominant=list(outcome.domain_dominant),
        onset_step=outcome.onset_step,
        params=outcome.params,
        flops=outcome.flops,
        train_seconds=outcome.train_seconds,
        origin=origin,
        accepted=True,
        reason="",
    )


def _replication_summary(
    rows: list[RunRecord], expectations: dict[str, float]
) -> dict[str, Any]:
    """把多种子复核折成「成功次数」表。**对照组的 0/N 和 learned 的 N/N 一样重要。**"""
    groups: dict[str, list[RunRecord]] = {}
    for r in rows:
        groups.setdefault(r.spec_id, []).append(r)

    per_spec: list[dict[str, Any]] = []
    for spec_id in sorted(groups):
        rs = groups[spec_id]
        losses = [r.final["loss"] for r in rs]
        nmis = [r.final["nmi"] for r in rs]
        per_spec.append(
            {
                "spec_id": spec_id,
                "routing": rs[0].spec["routing"],
                "n": len(rs),
                "loss_mean": sum(losses) / len(losses),
                "loss_min": min(losses),
                "loss_max": max(losses),
                "nmi_mean": sum(nmis) / len(nmis),
                "nmi_min": min(nmis),
                "nmi_max": max(nmis),
                "success": sum(1 for r in rs if r.final["below_floor"] > 0.5),
                "losses": losses,
            }
        )

    learned = [p for p in per_spec if p["routing"] == "learned"]
    control = [p for p in per_spec if p["routing"] != "learned"]
    seeds = sorted({r.seed for r in rows})
    return {
        "seeds": seeds,
        "per_spec": per_spec,
        "learned_success": sum(p["success"] for p in learned),
        "learned_total": sum(p["n"] for p in learned),
        "control_success": sum(p["success"] for p in control),
        "control_total": sum(p["n"] for p in control),
        "expectations": expectations,
    }


def run_pipeline(
    cfg: dict,
    *,
    progress: Callable[[str, int, dict[str, Any]], None] | None = None,
) -> PipelineResult:
    """跑完整条流水线（不落盘）。``progress(group, index, final_metrics)`` 用来打印。"""
    corpus = build_corpus(cfg)
    specs = expand_grid(cfg)
    exp = expectations_of(cfg)
    margin = floor_margin(cfg)
    seed = int(cfg["seed"])

    runs: list[RunRecord] = []
    for i, spec in enumerate(specs, 1):
        outcome = run_spec(spec, corpus, cfg, seed)
        runs.append(_record(spec, seed, outcome, "grid"))
        if progress:
            progress("grid", i, runs[-1].final)

    rep_rows: list[RunRecord] = []
    for j, (spec, rseed) in enumerate(expand_replication(cfg), 1):
        outcome = run_spec(spec, corpus, cfg, rseed)
        rep_rows.append(_record(spec, rseed, outcome, "replication"))
        if progress:
            progress("replication", j, rep_rows[-1].final)

    replication = _replication_summary(rep_rows, exp) if rep_rows else {}
    grid = grid_report(cfg)
    all_spec_ids = [s.spec_id for s in specs] + [s.spec_id for s, _ in expand_replication(cfg)]
    digest = version_digest(cfg, corpus.digest, all_spec_ids)
    version = Version(
        version_id=f"v1-{digest}",
        digest=digest,
        created_at=utcnow(),
        corpus_digest=corpus.digest,
        protocol_tag=protocol_digest(cfg),
        grid_size=len(specs),
        counts={
            "grid": len(specs),
            "replication": len(rep_rows),
            "observations": len(runs) + len(rep_rows),
        },
    )

    # 结论要看到**全部**观测（网格 + 复核）：「对照组是否全灭」这类断言样本越多越好。
    findings = compute_findings(
        runs + rep_rows, floor=corpus.domain_blind_floor(), margin=margin,
        grid=grid, replication=replication, expectations=exp,
    )

    corpus_meta = corpus.to_dict(with_samples=False)
    run_keys = [(r.spec_id, r.seed) for r in runs] + [(r.spec_id, r.seed) for r in rep_rows]
    dag = build_lineage(
        corpus_meta,
        run_keys,
        n_train=len(corpus.train),
        n_test=len(corpus.test),
        version_id=version.version_id,
        grid=grid,
    )

    return PipelineResult(
        runs=runs + rep_rows,
        findings=findings,
        version=version,
        grid=grid,
        replication=replication,
        corpus_meta=corpus_meta,
        dag=dag,
        objectives=(),
    )


# ----------------------------------------------------------------------
# 落盘
# ----------------------------------------------------------------------
def _matrix_row(run: RunRecord, group: str) -> dict[str, Any]:
    s = run.spec
    row: dict[str, Any] = {
        "group": group,
        "spec_id": run.spec_id,
        "seed": run.seed,
        "routing": s["routing"],
        "aux_weight": s["aux_weight"],
        "top_k": s["top_k"],
        "capacity_factor": s["capacity_factor"],
        "onset_step": "" if run.onset_step is None else run.onset_step,
        "params": run.params,
        "train_seconds": round(run.train_seconds, 4),
    }
    for k in ("loss", "acc", "entropy_norm", "imbalance", "load_cv", "nmi", "purity",
              "expert_sim", "gate_max_mean", "stability", "dropped_rate", "comm_units",
              "active_params", "below_floor"):
        row[k] = round(float(run.final[k]), 6)
    return row


def _group_of(run: RunRecord) -> str:
    return "learned" if run.spec["routing"] == "learned" else "control"


def _curves(result: PipelineResult) -> dict[str, Any]:
    return {
        r.spec_id + f"#s{r.seed}": [
            {
                "step": s["step"], "loss": s["loss"], "acc": s["acc"],
                "entropy_norm": s["entropy_norm"], "imbalance": s["imbalance"],
                "nmi": s["nmi"], "purity": s["purity"],
                "dropped_rate": s["dropped_rate"], "stability": s["stability"],
            }
            for s in r.snapshots
        ]
        for r in result.runs
    }


def _sankey(result: PipelineResult) -> dict[str, Any]:
    """每个格子的「域 → 专家」流量矩阵。**Sankey 的原始数据。**

    左列 = 域（token 按域分组），右列 = 专家，流量 = 被派发过去的 token 数。
    一眼就能看出「塌缩」（全部涌向一个专家）与「分化」（每个域各有一个出口）。
    """
    n_domains = len(result.runs[0].domain_expert) if result.runs else 0
    out: dict[str, Any] = {"n_domains": n_domains}
    for r in result.runs:
        n_exp = len(r.domain_expert[0]) if r.domain_expert else 0
        out[r.spec_id + f"#s{r.seed}"] = {
            "domains": [f"域{d}" for d in range(len(r.domain_expert))],
            "experts": [f"E{i}" for i in range(n_exp)],
            "flow": r.domain_expert,
            "dominant": r.domain_dominant,
            "loss": round(r.final["loss"], 4),
            "nmi": round(r.final["nmi"], 4),
            "entropy_norm": round(r.final["entropy_norm"], 4),
        }
    return out


def write_outputs(result: PipelineResult, out_dir: str) -> None:
    """把全部产物落到 ``out_dir``。``report.md`` 由 ``report`` 模块写。"""
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "runs.json"), "w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in result.runs], fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    with open(os.path.join(out_dir, "matrix.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MATRIX_COLUMNS))
        w.writeheader()
        for r in result.runs:
            w.writerow(_matrix_row(r, _group_of(r)))

    with open(os.path.join(out_dir, "curves.json"), "w", encoding="utf-8") as fh:
        json.dump(_curves(result), fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    with open(os.path.join(out_dir, "sankey.json"), "w", encoding="utf-8") as fh:
        json.dump(_sankey(result), fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    with open(os.path.join(out_dir, "findings.json"), "w", encoding="utf-8") as fh:
        json.dump([f.to_dict() for f in result.findings], fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    with open(os.path.join(out_dir, "replication.json"), "w", encoding="utf-8") as fh:
        json.dump(result.replication, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    with open(os.path.join(out_dir, "version.json"), "w", encoding="utf-8") as fh:
        json.dump(result.version.to_dict(), fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    with open(os.path.join(out_dir, "lineage.json"), "w", encoding="utf-8") as fh:
        fh.write(dumps_lineage(result.dag))

    with open(os.path.join(out_dir, "lineage.dot"), "w", encoding="utf-8") as fh:
        fh.write(result.dag.to_dot())

    # report.md 依赖上面这些模块，放在最后写，避免循环导入
    from .report import render_report

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(render_report(result))


# ----------------------------------------------------------------------
# 只读重建
# ----------------------------------------------------------------------
def load_result(out_dir: str) -> PipelineResult | None:
    """从产物重建结果对象。**缺任何一个必需文件就返回 ``None``**，不能「读一半」。"""
    for name in OUT_FILES:
        if not os.path.exists(os.path.join(out_dir, name)):
            return None
    try:
        with open(os.path.join(out_dir, "runs.json"), encoding="utf-8") as fh:
            runs = [RunRecord.from_dict(d) for d in json.load(fh)]
        with open(os.path.join(out_dir, "findings.json"), encoding="utf-8") as fh:
            findings = [Finding.from_dict(d) for d in json.load(fh)]
        with open(os.path.join(out_dir, "version.json"), encoding="utf-8") as fh:
            version = Version.from_dict(json.load(fh))
        with open(os.path.join(out_dir, "replication.json"), encoding="utf-8") as fh:
            replication = json.load(fh)
        with open(os.path.join(out_dir, "lineage.json"), encoding="utf-8") as fh:
            dag = Lineage.from_dict(json.load(fh))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    corpus_meta: dict[str, Any] = {}
    lin_root = dag.nodes.get("corpus")
    if lin_root is not None:
        corpus_meta = dict(lin_root.meta)
    if not corpus_meta:
        return None

    # 网格自述存在收拢节点的 meta 里 —— 这样只读路径也能拿到**声明**的基数，
    # 「扇出完整性」这条不变量在从产物重建时同样可查（否则它退化成同义反复）。
    collect = dag.nodes.get("collect")
    grid = dict(collect.meta.get("grid", {})) if collect is not None else {}
    if not grid:
        grid = {"n_specs": version.grid_size, "groups": [], "axes": []}

    return PipelineResult(
        runs=runs,
        findings=findings,
        version=version,
        grid=grid,
        replication=replication,
        corpus_meta=corpus_meta,
        dag=dag,
        objectives=(),
    )


def load_config_or_default(path: str | None) -> dict:
    if path is None:
        path = os.path.join(os.getcwd(), "configs", "default.json")
    return load_config(path)
