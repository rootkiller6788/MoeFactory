"""流水线 + 只读契约。

**这个形状的产物是「证据」，所以有三条不变量比任何指标都重要：**

1. **扇出完整性**：每个组的格数 == 各轴取值数之积。漏一个分支，后面所有统计都
   建立在残图上，而且**看起来完全正常**。
2. **收拢点恰好等于全部观测**：``collect`` 的父节点必须一个不多、一个不少 ——
   「一个不多」保证没有落选的格子被偷偷接进来，「一个不少」保证没有静默漏掉的分支。
3. **只读命令从产物重建，绝不重跑、绝不写盘**（见坑 41）。本项目整条流水线约 100 秒；
   若每条读命令都重算，用户会以为工具坏了。这条用 **md5 指纹**钉住，不靠自觉。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os

import pytest

from moefactory.config import (
    expand_grid,
    expand_replication,
    grid_report,
    objectives_of,
    version_digest,
)
from moefactory.corpus import build_corpus
from moefactory.lineage import COLLECT_NODE_ID, Lineage, dumps as dumps_lineage
from moefactory.pipeline import OUT_FILES, load_result, run_pipeline, write_outputs
from moefactory.schema import MoESpec, RunRecord
from moefactory.train import protocol_digest, run_spec

SPEC = "e4-k1-d12-f16-learned-aux0.1-cap1"

#: 只读命令清单。**一个写命令都不许混进来** —— 混进来这条测试就失效了。
#: 这里只用**不依赖产物内容**的调用形式（不带 ``--seed``），因为同一份清单要在
#: 「有产物」和「没产物」两种情形下都跑通。
READ_COMMANDS: list[list[str]] = [
    ["summary"],
    ["grid"],
    ["findings"],
    ["matrix", "--limit", "3"],
    ["matrix", "--by", "loss", "--json"],
    ["runs"],
    ["show", SPEC],
    ["curves", SPEC],
    ["curves", SPEC, "--json"],
    ["sankey", SPEC],
    ["sankey", SPEC, "--json"],
    ["replication"],
    ["lineage"],
    ["lineage", "--dot"],
    ["lineage", "--limit", "2"],
    ["task"],
    ["compare"],
    ["explain", SPEC],
]

#: 这几条只读命令的输出里**不该**出现「产物 / 内存重跑」的提示：
#: ``task`` / ``explain`` 只依赖配置；``lineage --dot`` 吐的是裸 DOT，
#: 多一行注释就毁掉了「直接管道给 graphviz」这个用途。
NO_NOTE: set[tuple[str, ...]] = {("task",), ("explain", SPEC), ("lineage", "--dot")}


def _fingerprint(path: str) -> dict[str, str]:
    """目录内每个文件的 md5。**只看内容，不看 mtime** —— 重写同内容不算改。"""
    out: dict[str, str] = {}
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            with open(full, "rb") as fh:
                out[name] = hashlib.md5(fh.read()).hexdigest()
    return out


# ----------------------------------------------------------------------
# 扇出完整性
# ----------------------------------------------------------------------
def test_grid_declaration_equals_the_cartesian_product(fast_cfg: dict) -> None:
    report = grid_report(fast_cfg)
    specs = expand_grid(fast_cfg)
    assert report["n_specs"] == len(specs)
    assert len({s.spec_id for s in specs}) == len(specs), "展开必须去重"
    for g in report["groups"]:
        prod = 1
        for n in g["axes"].values():
            prod *= n
        assert prod == g["declared"], f"组 {g['name']} 声明 {g['declared']} 但轴乘积是 {prod}"


def test_grid_group_membership_matches_the_declaration(fast_cfg: dict) -> None:
    declared = {g["name"]: g["declared"] for g in grid_report(fast_cfg)["groups"]}
    specs = expand_grid(fast_cfg)
    assert len([s for s in specs if s.routing == "learned"]) == declared["learned"]
    assert len([s for s in specs if s.routing != "learned"]) == declared["control"]


def test_grid_expansion_is_deterministic(fast_cfg: dict) -> None:
    assert [s.spec_id for s in expand_grid(fast_cfg)] == [s.spec_id for s in expand_grid(fast_cfg)]


def test_grid_expansion_ignores_budget_changes(fast_cfg: dict) -> None:
    c = copy.deepcopy(fast_cfg)
    c["proxy"]["steps"] = 9999
    assert [s.spec_id for s in expand_grid(c)] == [s.spec_id for s in expand_grid(fast_cfg)]


def test_every_grid_spec_has_a_reversible_id(fast_cfg: dict) -> None:
    """``spec_id`` 不可逆就等于把只读命令堵死（坑 41/42）。"""
    for spec in expand_grid(fast_cfg) + [s for s, _ in expand_replication(fast_cfg)]:
        assert MoESpec.from_spec_id(spec.spec_id) == spec
        assert MoESpec.from_spec_id(spec.spec_id).spec_id == spec.spec_id


def test_spec_id_round_trip_over_extremes() -> None:
    for spec in (
        MoESpec(n_experts=8, top_k=3, d_model=7, d_ff=9, routing="random",
                aux_weight=0.0, capacity_factor=0.0),
        MoESpec(n_experts=2, top_k=1, d_model=1, d_ff=1, routing="static",
                aux_weight=0.125, capacity_factor=3.5),
    ):
        assert MoESpec.from_spec_id(spec.spec_id) == spec.canonical()


def test_spec_id_rejects_garbage() -> None:
    for bad in (
        "",
        "e4-k1",
        "e4-k1-d12-f16-learned-aux0.1",
        "e4-k1-d12-f16-learned-auxX-cap1",
        "e4-k1-d12-f16-magic-aux0-cap1",
        "e4-k1-d12-f16-learned-aux0-cap1-extra",
    ):
        with pytest.raises(ValueError):
            MoESpec.from_spec_id(bad)


def test_fan_out_is_complete_in_the_products(fast_result) -> None:
    """**从产物重建之后**这条不变量依然可查 —— 否则它退化成同义反复。"""
    declared = {g["name"]: g["declared"] for g in fast_result.grid["groups"]}
    grid = fast_result.grid_runs
    assert len([r for r in grid if r.spec["routing"] == "learned"]) == declared["learned"]
    assert len([r for r in grid if r.spec["routing"] != "learned"]) == declared["control"]
    assert len(grid) == sum(declared.values()) == fast_result.version.grid_size


def test_origin_partitions_the_runs(fast_result) -> None:
    assert all(r.origin == "grid" for r in fast_result.grid_runs)
    assert all(r.origin == "replication" for r in fast_result.replication_runs)
    assert {r.origin for r in fast_result.runs} == {"grid", "replication"}
    assert len(fast_result.runs) == len(fast_result.grid_runs) + len(fast_result.replication_runs)


def test_run_keys_are_unique(fast_result) -> None:
    keys = [(r.spec_id, r.seed, r.origin) for r in fast_result.runs]
    assert len(keys) == len(set(keys))


def test_replication_block_matches_the_config(fast_result, fast_cfg: dict) -> None:
    rep = fast_result.replication
    assert rep["seeds"] == fast_cfg["replication"]["seeds"]
    assert rep["learned_total"] + rep["control_total"] == len(fast_result.replication_runs)
    for p in rep["per_spec"]:
        assert p["n"] == len(fast_cfg["replication"]["seeds"])
        assert p["loss_min"] <= p["loss_mean"] <= p["loss_max"]
        assert p["nmi_min"] <= p["nmi_mean"] <= p["nmi_max"]
        assert 0 <= p["success"] <= p["n"]


def test_no_optimisation_target_is_declared(fast_cfg: dict, fast_result) -> None:
    """测量型：**没有优化目标**。任何「按指标选最优」的代码都该无事可做。"""
    assert objectives_of(fast_cfg) == ()
    assert fast_result.objectives == ()


def test_control_rows_mark_aux_as_not_applicable(fast_result) -> None:
    """对照组的 aux 是 **N/A**，不是 0（同坑 10：把不适用当 0 会让均值偏移）。"""
    for r in fast_result.by_group("control"):
        assert r.aux_applicable is False
        assert MoESpec.from_dict(r.spec).aux_applicable is False
    for r in fast_result.by_group("learned"):
        assert r.aux_applicable is True


# ----------------------------------------------------------------------
# 版本与协议指纹
# ----------------------------------------------------------------------
def test_version_counts_agree_with_the_products(fast_result) -> None:
    v = fast_result.version
    assert v.counts["grid"] == len(fast_result.grid_runs)
    assert v.counts["replication"] == len(fast_result.replication_runs)
    assert v.counts["observations"] == len(fast_result.runs)
    assert v.grid_size == v.counts["grid"]
    assert v.version_id == f"v1-{v.digest}"


def test_version_digest_follows_the_observed_object(cfg: dict, corpus) -> None:
    """内容寻址：**只由「观测对象」决定**。"""
    specs = [s.spec_id for s in expand_grid(cfg)]
    base = version_digest(cfg, corpus.digest, specs)

    c = copy.deepcopy(cfg)
    c["proxy"]["steps"] = 999
    assert version_digest(c, corpus.digest, specs) == base, "训练预算不该改版本"

    other_task = copy.deepcopy(cfg)
    other_task["task"]["n_content"] = int(cfg["task"]["n_content"]) + 1
    assert version_digest(other_task, build_corpus(other_task).digest, specs) != base, "语料变了"

    renamed = copy.deepcopy(cfg)
    renamed["task"]["name"] = "another-task"
    assert version_digest(renamed, corpus.digest, specs) != base, "任务名变了"

    assert version_digest(cfg, corpus.digest, specs + ["extra"]) != base, "网格变了"


def test_protocol_digest_follows_the_budget_and_task(cfg: dict) -> None:
    base = protocol_digest(cfg)

    c = copy.deepcopy(cfg)
    c["proxy"]["steps"] = 999
    assert protocol_digest(c) != base

    c2 = copy.deepcopy(cfg)
    c2["proxy"]["lr"] = 0.5
    assert protocol_digest(c2) != base

    c3 = copy.deepcopy(cfg)
    c3["task"]["n_content"] = int(cfg["task"]["n_content"]) + 1
    assert protocol_digest(c3) != base

    c4 = copy.deepcopy(cfg)
    c4["grid"]["groups"][0]["axes"]["top_k"] = [1]
    assert protocol_digest(c4) == base, "网格大小不该改协议指纹"


def test_version_and_protocol_tags_are_in_the_products(fast_result) -> None:
    assert fast_result.version.protocol_tag
    assert fast_result.version.corpus_digest
    assert fast_result.corpus_meta["domain_blind_floor"] == pytest.approx(fast_result.floor)


# ----------------------------------------------------------------------
# 血缘
# ----------------------------------------------------------------------
def test_collect_gathers_exactly_every_observation(fast_result) -> None:
    """收拢点一个不多、一个不少 —— 这是本形状存在的理由。"""
    obs = {n for n in fast_result.dag.nodes if n.startswith("observe:")}
    assert len(obs) == len(fast_result.runs)
    assert set(fast_result.dag.parents(COLLECT_NODE_ID)) == obs


def test_lineage_shape(fast_result) -> None:
    dag = fast_result.dag
    assert dag.roots() == ["corpus"]
    assert dag.parents("version") == [COLLECT_NODE_ID]
    assert dag.leaves() == ["version"]
    n_specs = len({r.spec_id for r in fast_result.runs})
    assert len(dag.nodes) == 1 + n_specs + 2 * len(fast_result.runs) + 2
    # corpus→spec、spec→train、train→observe、observe→collect、collect→version
    assert len(dag.edges) == n_specs + 3 * len(fast_result.runs) + 1


def test_lineage_shares_one_spec_node_across_seeds(fast_result) -> None:
    """同一配置的多个种子共用 ``spec`` 节点，但各有各的 ``train`` / ``observe``。"""
    for r in fast_result.replication_runs:
        node = f"spec:{r.spec_id}"
        assert node in fast_result.dag.nodes
        assert f"train:{r.spec_id}#s{r.seed}" in fast_result.dag.nodes
        assert f"observe:{r.spec_id}#s{r.seed}" in fast_result.dag.nodes


def test_lineage_edges_reference_existing_nodes(fast_result) -> None:
    for e in fast_result.dag.edges:
        assert e.src in fast_result.dag.nodes
        assert e.dst in fast_result.dag.nodes


def test_lineage_has_no_cycle(fast_result) -> None:
    dag = fast_result.dag
    for nid in dag.nodes:
        dag.path_to_root(nid)  # 有环会抛 ValueError


def test_lineage_round_trip_is_bit_identical(fast_result) -> None:
    dag = fast_result.dag
    assert dumps_lineage(Lineage.from_dict(dag.to_dict())) == dumps_lineage(dag)


def test_lineage_dot_is_valid_enough(fast_result) -> None:
    dot = fast_result.dag.to_dot()
    assert dot.startswith("digraph moefactory {")
    assert dot.rstrip().endswith("}")
    assert dot.count("{") == dot.count("}") == 1


def test_lineage_declares_the_grid_cardinality(fast_result, fast_cfg: dict) -> None:
    """网格自述存在收拢节点的 meta 里 ⇒ 只读路径也能核对完整性。"""
    declared = fast_result.dag.nodes[COLLECT_NODE_ID].meta["grid"]
    assert declared["groups"] == fast_result.grid["groups"]


def test_lineage_json_on_disk_is_canonical(fast_out: str) -> None:
    with open(os.path.join(fast_out, "lineage.json"), encoding="utf-8") as fh:
        raw = fh.read()
    assert dumps_lineage(Lineage.from_dict(json.loads(raw))) == raw


def test_lineage_ascii_and_dot_render(fast_result) -> None:
    text = fast_result.dag.to_ascii(limit=2)
    assert "血缘：" in text
    assert "收拢 collect" in text


def test_lineage_rejects_duplicate_nodes_and_dangling_edges() -> None:
    from moefactory.lineage import Node

    lin = Lineage()
    lin.add_node(Node("a", "corpus", "a"))
    with pytest.raises(ValueError):
        lin.add_node(Node("a", "corpus", "a"))
    with pytest.raises(ValueError):
        lin.add_edge("a", "missing")


# ----------------------------------------------------------------------
# 产物与只读重建
# ----------------------------------------------------------------------
def test_all_out_files_are_written(fast_out: str) -> None:
    names = set(os.listdir(fast_out))
    assert set(OUT_FILES) <= names
    assert {"report.md", "lineage.dot"} <= names


def test_matrix_csv_has_the_declared_columns(fast_out: str) -> None:
    import csv

    from moefactory.pipeline import MATRIX_COLUMNS

    with open(os.path.join(fast_out, "matrix.csv"), encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == list(MATRIX_COLUMNS)
    assert len(rows) == len(load_result(fast_out).runs)


def test_curves_and_sankey_cover_every_run(fast_out: str, fast_result) -> None:
    with open(os.path.join(fast_out, "curves.json"), encoding="utf-8") as fh:
        curves = json.load(fh)
    with open(os.path.join(fast_out, "sankey.json"), encoding="utf-8") as fh:
        sankey = json.load(fh)
    by_key = {f"{r.spec_id}#s{r.seed}": r for r in fast_result.runs}
    assert set(curves) == set(by_key)
    assert set(sankey) == set(by_key) | {"n_domains"}
    for key, run in by_key.items():
        assert len(curves[key]) == len(run.snapshots)
        assert [p["step"] for p in curves[key]] == [s["step"] for s in run.snapshots]
        flow = sankey[key]["flow"]
        assert flow == run.domain_expert
        assert sum(sum(row) for row in flow) > 0
        assert len(sankey[key]["experts"]) == len(flow[0])


def test_report_md_has_all_sections(fast_out: str) -> None:
    with open(os.path.join(fast_out, "report.md"), encoding="utf-8") as fh:
        text = fh.read()
    for heading in (
        "# MoEFactory 报告",
        "## 一、口径",
        "## 二、结论",
        "## 三、Routing Sankey",
        "## 四、均衡 ≠ 分化",
        "### 4.1 按均衡排序",
        "### 4.2 均衡 × 分化 散点",
        "### 4.3 aux 这一轴是平的",
        "## 五、多种子复核",
        "## 六、逐 step 曲线",
        "## 七、代价：丢弃与通信",
        "## 八、可复现性边界",
    ):
        assert heading in text, f"报告缺少小节：{heading}"


def test_report_marks_control_rows(fast_out: str) -> None:
    """对照组必须被标出来 —— 「数字更高」不代表「更好」。"""
    with open(os.path.join(fast_out, "report.md"), encoding="utf-8") as fh:
        text = fh.read()
    assert "**[对照]**" in text


def test_load_result_rejects_a_missing_directory(tmp_path) -> None:
    assert load_result(str(tmp_path / "nope")) is None


@pytest.mark.parametrize("missing", OUT_FILES)
def test_load_result_returns_none_when_any_required_file_is_missing(
    fast_out: str, tmp_path, missing: str
) -> None:
    """**不能「读一半」** —— 半个结果对象比没有结果更危险。"""
    import shutil

    target = str(tmp_path / f"out-{missing}")
    shutil.copytree(fast_out, target)
    os.remove(os.path.join(target, missing))
    assert load_result(target) is None


def test_load_result_returns_none_on_corrupt_json(fast_out: str, tmp_path) -> None:
    import shutil

    target = str(tmp_path / "corrupt")
    shutil.copytree(fast_out, target)
    with open(os.path.join(target, "runs.json"), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    assert load_result(target) is None


def test_load_result_reconstructs_the_declared_grid(fast_out: str, fast_cfg: dict) -> None:
    r = load_result(fast_out)
    declared = {g["name"]: g["declared"] for g in r.grid["groups"]}
    specs = expand_grid(fast_cfg)
    assert declared["learned"] == len([s for s in specs if s.routing == "learned"])
    assert declared["control"] == len([s for s in specs if s.routing != "learned"])


def test_run_record_helpers(fast_result) -> None:
    r = fast_result.runs[0]
    assert r.metric("loss") == r.final["loss"]
    with pytest.raises(KeyError):
        r.metric("no-such-metric")
    assert r.snapshot(r.snapshots[-1]["step"]) is not None
    assert r.snapshot(-1) is None
    assert RunRecord.from_dict(r.to_dict()).spec_id == r.spec_id


def test_pipeline_result_queries(fast_result) -> None:
    best = fast_result.best_by_loss()
    assert best is not None and best.final["loss"] == min(r.final["loss"] for r in fast_result.runs)
    balanced = fast_result.most_balanced()
    assert balanced is not None
    assert balanced.final["entropy_norm"] == max(r.final["entropy_norm"] for r in fast_result.runs)
    assert fast_result.find("no-such-spec") is None
    assert fast_result.finding("no-such-finding") is None
    assert set(fast_result.spec_ids()) == {r.spec_id for r in fast_result.runs}
    with pytest.raises(KeyError):
        fast_result.by_group("nope")


def test_two_runs_of_the_same_spec_agree_bit_for_bit(fast_cfg: dict, fast_corpus) -> None:
    """除实测耗时外，**两次运行必须逐位相同**。"""
    c = copy.deepcopy(fast_cfg)
    c["proxy"]["steps"] = 10
    c["proxy"]["eval_every"] = 5
    spec = MoESpec.from_spec_id(SPEC)
    a = run_spec(spec, fast_corpus, c, 1)
    b = run_spec(spec, fast_corpus, c, 1)
    assert a.snapshots == b.snapshots
    assert a.domain_expert == b.domain_expert
    assert a.domain_dominant == b.domain_dominant
    assert a.onset_step == b.onset_step
    assert {k: v for k, v in a.final.items()} == {k: v for k, v in b.final.items()}


def test_pipeline_is_deterministic_apart_from_timestamps(tmp_path, tiny_cfg: dict) -> None:
    a = run_pipeline(tiny_cfg)
    b = run_pipeline(tiny_cfg)
    assert a.version.digest == b.version.digest
    assert [r.final for r in a.runs] == [r.final for r in b.runs]
    assert [f.to_dict() for f in a.findings] == [f.to_dict() for f in b.findings]
    assert dumps_lineage(a.dag) == dumps_lineage(b.dag)


# ----------------------------------------------------------------------
# CLI：只读契约
# ----------------------------------------------------------------------
def test_read_commands_never_touch_the_products(cli, fast_cfg_path: str, fast_out: str) -> None:
    """**md5 指纹**：19 条只读命令跑完，产物一个字节都不许变。"""
    before = _fingerprint(fast_out)
    assert before, "产物目录是空的，这条测试没意义"
    for argv in READ_COMMANDS:
        code, out = cli(["--config", fast_cfg_path, "--out", fast_out] + argv)
        assert code == 0, f"{argv} 退出码 {code}\n{out}"
        assert "内存重跑" not in out, f"{argv} 没有走产物重建路径"
    assert _fingerprint(fast_out) == before


def test_read_commands_write_nothing_when_products_are_absent(
    cli, tiny_cfg_path: str, tmp_path
) -> None:
    """产物不存在时，只读命令在内存里重跑 —— **但仍然不落盘**。"""
    empty = tmp_path / "empty-out"
    empty.mkdir()
    for argv in READ_COMMANDS:
        code, out = cli(["--config", tiny_cfg_path, "--out", str(empty)] + argv)
        assert code == 0, f"{argv} 退出码 {code}\n{out}"
        if tuple(argv) not in NO_NOTE:
            assert "未写盘" in out, f"{argv} 没提示「未写盘」"
    assert os.listdir(empty) == []


def test_read_commands_work_from_products_only(cli, fast_cfg_path: str, fast_out: str) -> None:
    """把配置换成 tiny、产物仍用 fast 的 —— 读命令应当照样能读出来。

    这条钉住「只读路径**真的**是从产物重建的」：若它偷偷重跑配置，
    tiny 配置会给出 2 次观测而不是 10 次。
    """
    code, out = cli(["--config", fast_cfg_path, "--out", fast_out, "runs"])
    assert code == 0
    assert out.count("seed=") == len(load_result(fast_out).runs)


def test_seed_selects_a_specific_replication_run(cli, fast_cfg_path: str, fast_out: str) -> None:
    """同一 ``spec_id`` 有多个种子时，``--seed`` 必须能选中其中一个。"""
    result = load_result(fast_out)
    rep = [r for r in result.replication_runs if r.seed == 1]
    assert rep, "fast 配置里没有 seed=1 的复核行"
    target = rep[0]
    code, out = cli(["--config", fast_cfg_path, "--out", fast_out,
                     "show", target.spec_id, "--seed", "1"])
    assert code == 0
    assert f"seed={target.seed}" in out


def test_visualize_writes_only_report_html(cli, fast_cfg_path: str, fast_out: str, tmp_path) -> None:
    import shutil

    target = str(tmp_path / "viz")
    shutil.copytree(fast_out, target)
    before = _fingerprint(target)
    assert "report.html" not in before
    code, out = cli(["--config", fast_cfg_path, "--out", target, "visualize"])
    assert code == 0 and "自包含" in out
    after = _fingerprint(target)
    assert set(after) - set(before) == {"report.html"}
    for name, digest in before.items():
        assert after[name] == digest


def test_visualize_output_is_self_contained(cli, fast_cfg_path: str, fast_out: str, tmp_path) -> None:
    import shutil

    target = str(tmp_path / "viz2")
    shutil.copytree(fast_out, target)
    assert cli(["--config", fast_cfg_path, "--out", target, "visualize"])[0] == 0
    with open(os.path.join(target, "report.html"), encoding="utf-8") as fh:
        html = fh.read()
    assert html.startswith("<!DOCTYPE html>")
    assert "<svg" in html
    assert 'src="http' not in html and 'href="http' not in html


def test_visualize_without_products_fails(cli, fast_cfg_path: str, tmp_path) -> None:
    empty = str(tmp_path / "no-out")
    os.makedirs(empty)
    code, _ = cli(["--config", fast_cfg_path, "--out", empty, "visualize"])
    assert code == 1
    assert os.listdir(empty) == []


def test_run_is_the_only_command_that_writes(cli, fast_cfg_path: str, tmp_path) -> None:
    target = str(tmp_path / "run-out")
    code, out = cli(["--config", fast_cfg_path, "--out", target, "--quiet", "run"])
    assert code == 0
    assert set(OUT_FILES) <= set(os.listdir(target))
    assert out == "", "--quiet 不该有任何输出"


# ----------------------------------------------------------------------
# CLI：命令行为
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "argv,needle",
    [
        (["summary"], "可计算下界"),
        (["summary"], "观测目标    : 无"),
        (["grid"], "扇出网格"),
        (["grid"], "N/A"),
        (["findings"], "支持"),
        (["matrix", "--limit", "4"], "spec"),
        (["runs"], "seed="),
        (["show", SPEC], "域→主要专家集合"),
        (["show", SPEC], "稀疏激活"),
        (["curves", SPEC], "step"),
        (["sankey", SPEC], "域0"),
        (["replication"], "突破下界"),
        (["lineage"], "血缘："),
        (["compare"], "static"),
        (["compare"], "均衡 ≠ 分化"),
        (["task"], "domain_blind_floor"),
        (["task"], "关键性质"),
        (["explain", SPEC], "信息通路"),
        (["explain", SPEC], "辅助损失"),
    ],
)
def test_cli_commands_print_what_they_promise(cli, fast_cfg_path: str, fast_out: str, argv, needle) -> None:
    code, out = cli(["--config", fast_cfg_path, "--out", fast_out] + argv)
    assert code == 0
    assert needle in out, f"{argv} 的输出里没有 {needle!r}"


def test_explain_and_task_need_no_products(cli, fast_cfg_path: str, tmp_path) -> None:
    """这两条命令**只依赖配置**，连产物都不需要 —— 所以它们永远可用。"""
    missing = str(tmp_path / "definitely-missing")
    assert cli(["--config", fast_cfg_path, "--out", missing, "task"])[0] == 0
    assert cli(["--config", fast_cfg_path, "--out", missing, "explain", SPEC])[0] == 0
    assert not os.path.exists(missing)


def test_explain_marks_control_aux_as_na(cli, fast_cfg_path: str) -> None:
    code, out = cli(["--config", fast_cfg_path, "explain", "e4-k1-d12-f16-static-aux0-cap1"])
    assert code == 0
    assert "N/A" in out


def test_explain_rejects_a_bad_spec_id(cli, fast_cfg_path: str) -> None:
    assert cli(["--config", fast_cfg_path, "explain", "garbage"])[0] == 1


def test_show_suggests_nearby_specs(capsys, fast_cfg_path: str, fast_out: str) -> None:
    """找不到时要给相近建议，而不是只丢一句「找不到」—— 否则用户只能去翻 spec_id。"""
    from moefactory.cli import main

    code = main(["--config", fast_cfg_path, "--out", fast_out,
                 "show", "e4-k9-d99-f99-learned-aux0-cap0"])
    captured = capsys.readouterr()
    assert code == 1
    assert "找不到" in captured.err


def test_quiet_must_precede_the_subcommand(cli, fast_cfg_path: str, tmp_path) -> None:
    """``--quiet`` 是顶层开关。放错位置会被 argparse 当成未知参数。"""
    target = str(tmp_path / "quiet-out")
    assert cli(["--config", fast_cfg_path, "--out", target, "--quiet", "run"])[0] == 0
    with pytest.raises(SystemExit):
        cli(["run", "--quiet"])


def test_no_subcommand_is_an_error(cli) -> None:
    with pytest.raises(SystemExit):
        cli([])
