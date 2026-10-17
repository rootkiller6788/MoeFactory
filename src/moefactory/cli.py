"""命令行。**读命令不写盘**，写命令只有 ``run`` 与 ``visualize``。

这不是洁癖：``out/`` 里那份结果可能对应另一份配置，用户看一眼报告就会拿它下结论。
如果「看一眼」会顺手改写产物，结论与配置的对应关系就断了。所以：

- 读命令**从产物重建结果对象**（``load_result``），绝不重跑流水线；
- 产物缺失时，读命令在内存里跑一遍（仍然**不落盘**）—— 用户能立刻看到东西，
  但不会以为「看报告也会写文件」；
- ``visualize`` 是唯一的例外，它写 ``report.html``，并在 ``--out`` 之外单独存在。

命令清单（前 12 个只读）：

```
summary / grid / findings / matrix / runs / show / curves / sankey / replication
/ lineage / task / compare          ← 只读
run / visualize                     ← 写
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import expand_grid, grid_report, load_config, expectations_of
from .corpus import build_corpus
from .pipeline import OUT_FILES, PipelineResult, load_result, run_pipeline, write_outputs
from .observe import distinct_primary_sets, primary_experts
from .report import ascii_sankey, fmt
from .schema import MoESpec

DEFAULT_CONFIG = os.path.join("configs", "default.json")


# ----------------------------------------------------------------------
# 结果获取（读命令的唯一入口）
# ----------------------------------------------------------------------
def _result(args: argparse.Namespace) -> tuple[PipelineResult, dict, bool]:
    """返回 ``(结果, 配置, 是否来自产物)``。**永不落盘。**"""
    cfg = load_config(args.config)
    cached = load_result(args.out)
    if cached is not None:
        return cached, cfg, True
    return run_pipeline(cfg), cfg, False


def _note(from_products: bool) -> str:
    return "" if from_products else "  （产物不存在，本次在内存中重跑；未写盘）"


# ----------------------------------------------------------------------
# 只读命令
# ----------------------------------------------------------------------
def cmd_summary(args: argparse.Namespace) -> int:
    r, _, cached = _result(args)
    v = r.version
    print(f"版本        : {v.version_id}")
    print(f"digest      : {v.digest}")
    print(f"创建时间    : {v.created_at}  （不可逐位复现）")
    print(f"协议指纹    : {v.protocol_tag}")
    print(f"语料        : {r.corpus_meta.get('name')} "
          f"{r.corpus_meta.get('n_domains')}域 × {r.corpus_meta.get('n_content')}内容 "
          f"vocab={r.corpus_meta.get('vocab_size')} seq={r.corpus_meta.get('seq_len')}")
    print(f"语料摘要    : {v.corpus_digest}")
    print(f"可计算下界  : domain_blind_floor = {r.floor:.4f}")
    print(f"观测目标    : 无（测量型：只观测、不优化、不选优）")
    print(f"网格 / 复核 : {v.counts['grid']} / {v.counts['replication']}  → "
          f"共 {v.counts['observations']} 次观测")
    for g in r.grid.get("groups", []):
        print(f"  - {g['name']:8s} 声明 {g['declared']:3d} 格  轴 "
              f"{', '.join(f'{k}×{n}' for k, n in g['axes'].items())}")
    ok = sum(1 for f in r.findings if f.supported)
    print(f"结论        : {ok}/{len(r.findings)} 条被实测支持")
    print(f"来源        : {'产物' if cached else '内存重跑'}{_note(cached)}")
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    r, cfg, cached = _result(args)
    gr = r.grid
    print(f"扇出网格：{gr.get('n_specs')} 格{_note(cached)}")
    print(f"  轴：{', '.join(gr.get('axes', []))}")
    for g in gr.get("groups", []):
        prod = 1
        for n in g["axes"].values():
            prod *= n
        flag = "OK" if prod == g["declared"] else "不完整"
        print(f"  [{g['name']}] 声明 {g['declared']} = "
              f"{' × '.join(str(n) for n in g['axes'].values())}  {flag}")
        print(f"      固定：{g.get('fixed', {})}")
    print("\n  注：control 组的 aux 是 **N/A**（不是 0）—— 把不适用当 0 算进均值会让结论偏移。")
    return 0


def cmd_findings(args: argparse.Namespace) -> int:
    r, _, cached = _result(args)
    print(f"结论（{len(r.findings)} 条）{_note(cached)}\n")
    for i, f in enumerate(r.findings, 1):
        print(f"[{i}] {'支持  ' if f.supported else '不支持'} {f.claim}")
        for k, val in f.evidence.items():
            print(f"      {k} = {fmt(val)}")
        if f.note:
            print(f"      注：{f.note}")
        print()
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    r, _, cached = _result(args)
    seed0 = r.runs[0].seed if r.runs else 0
    rows = [x for x in r.runs if x.seed == seed0]
    rows.sort(key=lambda x: (-x.final["entropy_norm"] if args.by == "balance"
                             else x.final["loss"]))
    print(f"架构/路由矩阵（seed={seed0}，按 {'均衡' if args.by == 'balance' else 'loss'} 排序）"
          f"{_note(cached)}\n")
    print(f"  {'spec':44s} {'组':8s} {'H':>7s} {'imb':>6s} {'loss':>7s} "
          f"{'nmi':>6s} {'pur':>6s} {'drop':>6s} {'下界':>4s}")
    for x in rows[: args.limit]:
        print(f"  {x.spec_id:44s} {('learned' if x.spec['routing'] == 'learned' else 'control'):8s} "
              f"{x.final['entropy_norm']:7.4f} {x.final['imbalance']:6.2f} {x.final['loss']:7.3f} "
              f"{x.final['nmi']:6.3f} {x.final['purity']:6.3f} {x.final['dropped_rate']:6.3f} "
              f"{'是' if x.final['below_floor'] else '否':>4s}")
    if args.json:
        print(json.dumps([x.to_dict() for x in rows[: args.limit]], ensure_ascii=False, indent=2))
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    r, _, cached = _result(args)
    print(f"全部观测（{len(r.runs)} 次）{_note(cached)}\n")
    for x in r.runs:
        print(f"  {x.spec_id:44s} seed={x.seed:<9d} loss={x.final['loss']:.3f} "
              f"nmi={x.final['nmi']:.3f} H={x.final['entropy_norm']:.4f} "
              f"onset={x.onset_step}")
    return 0


def _pick(r: PipelineResult, spec_id: str, seed: int | None) -> Any:
    run = r.find(spec_id, seed)
    if run is not None:
        return run
    # 给相近建议，而不是只说「找不到」
    near = [s for s in r.spec_ids() if s[:6] == spec_id[:6]]
    print(f"找不到 spec_id={spec_id!r}", file=sys.stderr)
    if near:
        print("你是不是想找：", file=sys.stderr)
        for s in near[:6]:
            print(f"  {s}", file=sys.stderr)
    return None


def cmd_show(args: argparse.Namespace) -> int:
    r, _, cached = _result(args)
    run = _pick(r, args.spec_id, args.seed)
    if run is None:
        return 1
    print(f"{run.spec_id}  seed={run.seed}{_note(cached)}\n")
    print("  配置：", json.dumps(run.spec, ensure_ascii=False))
    print(f"  参数：{run.params}  （稀疏激活 {int(run.final['active_params'])}）")
    print(f"  观测：loss={run.final['loss']:.4f} acc={run.final['acc']:.4f} "
          f"下界={run.final['domain_blind_floor']:.4f} "
          f"突破下界={'是' if run.final['below_floor'] else '否'}")
    print(f"  负载：H={run.final['entropy_norm']:.4f} imb={run.final['imbalance']:.3f} "
          f"cv={run.final['load_cv']:.3f} 丢弃={run.final['dropped_rate']:.4f}")
    print(f"  分化：nmi={run.final['nmi']:.4f} purity={run.final['purity']:.4f} "
          f"专家相似度={run.final['expert_sim']:.4f}")
    print(f"  代价：通信={int(run.final['comm_units'])} 元素  MACs={run.flops}")
    print(f"  onset={run.onset_step}  耗时={run.train_seconds:.3f}s（不可复现字段）")
    sets = [list(primary_experts(r)) for r in run.domain_expert]
    print(f"  域→主导专家：{run.domain_dominant}")
    print(f"  域→主要专家集合（占比≥20%）：{sets}  "
          f"两两不同={'是' if distinct_primary_sets(run.domain_expert) else '否'}")
    return 0


def cmd_curves(args: argparse.Namespace) -> int:
    r, _, cached = _result(args)
    run = _pick(r, args.spec_id, args.seed)
    if run is None:
        return 1
    print(f"{run.spec_id} 的逐 step 曲线{_note(cached)}\n")
    print(f"  {'step':>5s} {'loss':>8s} {'acc':>7s} {'H':>7s} {'imb':>6s} "
          f"{'nmi':>7s} {'pur':>7s} {'drop':>7s} {'stab':>6s}")
    for s in run.snapshots:
        stab = "  —  " if s["stability"] is None else f"{s['stability']:6.2f}"
        print(f"  {s['step']:5d} {s['loss']:8.4f} {s['acc']:7.4f} {s['entropy_norm']:7.4f} "
              f"{s['imbalance']:6.2f} {s['nmi']:7.4f} {s['purity']:7.4f} "
              f"{s['dropped_rate']:7.4f} {stab}")
    if args.json:
        print(json.dumps(run.snapshots, ensure_ascii=False, indent=2))
    return 0


def cmd_sankey(args: argparse.Namespace) -> int:
    r, _, cached = _result(args)
    run = _pick(r, args.spec_id, args.seed)
    if run is None:
        return 1
    experts = [f"E{i}" for i in range(len(run.domain_expert[0]) if run.domain_expert else 0)]
    print(f"{run.spec_id} 的 Routing Sankey{_note(cached)}\n")
    print(ascii_sankey(run.domain_expert, experts))
    print(f"\n  loss={run.final['loss']:.4f}  H={run.final['entropy_norm']:.4f}  "
          f"nmi={run.final['nmi']:.4f}")
    if args.json:
        print(json.dumps({"flow": run.domain_expert, "dominant": run.domain_dominant},
                         ensure_ascii=False, indent=2))
    return 0


def cmd_replication(args: argparse.Namespace) -> int:
    r, _, cached = _result(args)
    rep = r.replication
    if not rep:
        print("未配置多种子复核")
        return 0
    print(f"多种子复核（种子 {rep['seeds']}）{_note(cached)}\n")
    print(f"  {'spec':44s} {'n':>2s} {'loss均值':>9s} {'min':>7s} {'max':>7s} "
          f"{'nmi均值':>8s} {'突破下界':>8s}")
    for p in rep["per_spec"]:
        print(f"  {p['spec_id']:44s} {p['n']:2d} {p['loss_mean']:9.3f} {p['loss_min']:7.3f} "
              f"{p['loss_max']:7.3f} {p['nmi_mean']:8.3f} {p['success']:>4d}/{p['n']:<3d}")
    print(f"\n  learned  {rep['learned_success']}/{rep['learned_total']} 突破下界")
    print(f"  对照组   {rep['control_success']}/{rep['control_total']} 突破下界")
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    r, _, cached = _result(args)
    if args.dot:
        print(r.dag.to_dot())
        return 0
    print(r.dag.to_ascii(limit=args.limit))
    print(f"{_note(cached)}" if not cached else "")
    return 0


def cmd_task(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    corpus = build_corpus(cfg)
    floor = corpus.domain_blind_floor()
    print(f"任务        : {corpus.name}（多任务 next-token）")
    print(f"vocab       : {corpus.vocab_size} = {corpus.n_content} 内容 token "
          f"+ {corpus.n_domains} 任务标签")
    print(f"序列        : [tag_d, c0, c1, ...]，c_{{i+1}} = perm_d(c_i)，长度 {corpus.seq_len}")
    print(f"样本        : train {len(corpus.train)} / test {len(corpus.test)}")
    print(f"\n**关键性质**：内容词表在域之间共享 ⇒ 同一个内容 token 在不同域有不同后继。")
    print(f"  平均不同后继数 = {corpus.distinct_successor_ratio():.2f} / {corpus.n_domains}")
    print(f"\n**可计算下界** domain_blind_floor = {floor:.4f}")
    print(f"  看不见域的路由器对每个 token 只能押一个后继，其最优损失就是这个数。")
    print(f"  ⇒ loss 显著低于它，就说明路由一定用上了域信息。")
    print(f"\n路由方式    : learned（被测）/ static（完美均衡但零分化）/ random（地板）")
    print(f"设计期望    : 每个域由一个专职专家承担（域→主导专家为一一对应）")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    r, _, cached = _result(args)
    seed0 = r.runs[0].seed if r.runs else 0
    print(f"三组路由方式对比（seed={seed0}）{_note(cached)}\n")
    print(f"  {'routing':10s} {'n':>2s} {'H均值':>8s} {'imb均值':>8s} {'loss均值':>9s} "
          f"{'nmi均值':>8s} {'突破下界':>8s}")
    for routing in ("learned", "static", "random"):
        rs = [x for x in r.runs if x.spec["routing"] == routing and x.seed == seed0]
        if not rs:
            continue
        n = len(rs)
        print(f"  {routing:10s} {n:2d} "
              f"{sum(x.final['entropy_norm'] for x in rs) / n:8.4f} "
              f"{sum(x.final['imbalance'] for x in rs) / n:8.2f} "
              f"{sum(x.final['loss'] for x in rs) / n:9.3f} "
              f"{sum(x.final['nmi'] for x in rs) / n:8.3f} "
              f"{sum(1 for x in rs if x.final['below_floor']):>4d}/{n:<3d}")
    print(f"\n  可计算下界 = {r.floor:.4f}")
    print("  注：static 按构造完美均衡（H≈1.0），却不可能突破下界 —— 均衡 ≠ 分化。")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """从 ``spec_id`` 展开一个标本的装配说明。**不需要任何产物、不训练。**"""
    try:
        spec = MoESpec.from_spec_id(args.spec_id)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    corpus = build_corpus(cfg)
    print(f"{spec.spec_id}\n")
    print(f"  专家数      : {spec.n_experts}")
    print(f"  top_k       : {spec.top_k}  （每个 token 派发给 {spec.top_k} 个专家）")
    print(f"  d_model/ffn : {spec.d_model} / {spec.d_ff}")
    print(f"  路由方式    : {spec.routing}")
    if spec.aux_applicable:
        print(f"  辅助损失    : α={spec.aux_weight:g}"
              f"{'（关闭：路由不会被推向均衡）' if spec.aux_weight == 0 else ''}")
    else:
        print(f"  辅助损失    : **N/A**（{spec.routing} 路由不经过路由器，这条轴对它不适用）")
    cap = spec.capacity_per_expert(corpus.seq_len * len(corpus.test))
    print(f"  容量因子    : {spec.capacity_factor:g} → 每专家每 batch 约 "
          f"{'不限' if cap == float('inf') else f'{cap:.1f} 个 token'}")
    print(f"  参数量      : {spec.param_count(corpus.vocab_size)}")
    print(f"\n  展开：embed[{corpus.vocab_size}×{spec.d_model}] → RMSNorm → "
          f"router[{spec.d_model}×{spec.n_experts}] → top-{spec.top_k} → 容量 → "
          f"派发到 {spec.n_experts} 个专家 → 合并 → 残差 → head[{spec.d_model}×{corpus.vocab_size}]")
    print(f"  信息通路：路由器看 emb[token]+emb[tag]；专家只看 emb[token]。")
    return 0


# ----------------------------------------------------------------------
# 写命令
# ----------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out = args.out

    def progress(group: str, i: int, final: dict[str, Any]) -> None:
        if args.quiet:
            return
        print(f"[{group}] {i} {final.get('loss', 0):.3f} "
              f"H={final.get('entropy_norm', 0):.3f} nmi={final.get('nmi', 0):.3f}",
              flush=True)

    result = run_pipeline(cfg, progress=progress)
    write_outputs(result, out)
    if not args.quiet:
        print(f"\n写入 {out}/：{', '.join(OUT_FILES)} + report.md + lineage.dot")
        print(f"版本 {result.version.version_id}　下界 {result.floor:.4f}")
        ok = sum(1 for f in result.findings if f.supported)
        print(f"结论 {ok}/{len(result.findings)} 条被支持")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    from .visualize import render_html

    cfg = load_config(args.config)
    result = load_result(args.out)
    if result is None:
        if args.quiet:
            return 1
        print("产物不存在或不完整，无法生成 report.html（先跑 run）", file=sys.stderr)
        return 1
    path = os.path.join(args.out, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(result))
    if not args.quiet:
        print(f"写入 {path}（自包含，无外部引用）")
    return 0


# ----------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------
#: 项目根：`src/{pkg}/cli.py` → 上三级。**所有相对路径默认值都锚到这里。**
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def _anchor_args(args: argparse.Namespace) -> None:
    """把 `--config` / `--out` / `--root` 的**相对路径**锚定到项目根。

    契约是 `cd <项目目录> && python main.py ...`，从项目目录里跑时锚不锚一个样
    —— 缺陷因此**长期隐形**。一旦从别处调用（例如跨项目的批量核对
    `python L5-xxx/main.py check`），`"configs/default.json"` 就指向了 cwd 下一个
    不存在的位置：严格 `load_config` 直接崩，宽松的会**静默回退 DEFAULTS**
    （更糟 —— 不报错，只是悄悄换了一套口径）。

    **在 `main()` 里紧跟 `parse_args` 锚一次**，比在每个调用点各锚一次可靠：
    漏掉一个调用点，就是一条只在某种调用方式下才现形的路径。

    绝对路径原样使用。语义与 `pytest` 的 rootdir / `cargo` / `npm --prefix` 一致。
    """
    for name in ("config", "out", "root"):
        value = getattr(args, name, None)
        if isinstance(value, str) and value:
            p = Path(value)
            if not p.is_absolute():
                setattr(args, name, str(_PKG_ROOT / p))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="moefactory",
        description="MoEFactory —— Token → Expert 的路由系统放大镜",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件路径；相对路径按项目根解析")
    p.add_argument("--out", default="out", help="产物目录（只有 run / visualize 会写它）；相对路径按项目根解析")
    p.add_argument("--quiet", action="store_true", help="安静模式")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary", help="版本 / 计数 / 下界 / 结论概览").set_defaults(func=cmd_summary)
    sub.add_parser("grid", help="扇出网格与完整性").set_defaults(func=cmd_grid)
    sub.add_parser("findings", help="逐条打印结论与证据").set_defaults(func=cmd_findings)

    m = sub.add_parser("matrix", help="对比矩阵（默认按均衡排序）")
    m.add_argument("--limit", type=int, default=20)
    m.add_argument("--by", choices=["balance", "loss"], default="balance")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_matrix)

    sub.add_parser("runs", help="全部观测一行一条").set_defaults(func=cmd_runs)

    s = sub.add_parser("show", help="一个格子的全部观测")
    s.add_argument("spec_id")
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(func=cmd_show)

    c = sub.add_parser("curves", help="逐 step 曲线")
    c.add_argument("spec_id")
    c.add_argument("--seed", type=int, default=None)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_curves)

    k = sub.add_parser("sankey", help="Routing Sankey（ASCII）")
    k.add_argument("spec_id")
    k.add_argument("--seed", type=int, default=None)
    k.add_argument("--json", action="store_true")
    k.set_defaults(func=cmd_sankey)

    sub.add_parser("replication", help="多种子复核").set_defaults(func=cmd_replication)

    l = sub.add_parser("lineage", help="血缘图（ASCII 或 DOT）")
    l.add_argument("--dot", action="store_true")
    l.add_argument("--limit", type=int, default=20)
    l.set_defaults(func=cmd_lineage)

    sub.add_parser("task", help="任务定义与可计算下界（不需要产物）").set_defaults(func=cmd_task)
    sub.add_parser("compare", help="三组路由方式对比").set_defaults(func=cmd_compare)

    e = sub.add_parser("explain", help="从一个 spec_id 展开装配说明（不需要产物）")
    e.add_argument("spec_id")
    e.set_defaults(func=cmd_explain)

    sub.add_parser("run", help="跑完整流水线并落盘").set_defaults(func=cmd_run)
    sub.add_parser("visualize", help="生成自包含 report.html").set_defaults(func=cmd_visualize)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _anchor_args(args)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
