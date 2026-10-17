"""报告：把观测与结论渲染成 markdown。

**报告的组织方式就是本项目的论点顺序：**

1. 口径（版本 / 协议 / 下界）—— 先说清「这些数字在什么条件下得到」；
2. 结论（每条带证据）—— 产物本身；
3. **Routing Sankey** —— README 里点名要看的东西，一眼看出塌缩与分化；
4. 均衡 vs 分化 —— 核心张力的两张表；
5. 多种子复核 —— 结论强度的来源；
6. 逐 step 曲线 —— 时间维度的证据；
7. 可复现性边界 —— 哪些数字必须逐位一致。

**ASCII 图不是凑数。** 这类项目的读者经常在终端里看报告；一张能直接在
终端里读出「所有 token 都涌向 E2」的 Sankey，比一张要开浏览器才能看的图有用。
自包含 HTML（``visualize``）是补充，不是替代。
"""

from __future__ import annotations

from typing import Any, Sequence

from .schema import RunRecord


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def _fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.{nd}f}"
    return str(v)


#: 公开别名：CLI 与报告必须用同一套数字格式，否则「命令行看到的」和
#: 「报告里写的」会不一致 —— 那是最难查的一类不一致。
fmt = _fmt


def _group_of(run: RunRecord) -> str:
    return "learned" if run.spec["routing"] == "learned" else "control"


def _tag(run: RunRecord) -> str:
    """对照组的名字后面加 ``[对照]``，提醒读者「它的数字更高不代表更好」。"""
    return f"`{run.spec_id}`" + (" **[对照]**" if _group_of(run) == "control" else "")


def bar(value: float, width: int = 24, char: str = "█") -> str:
    n = max(0, min(width, int(round(value * width))))
    return char * n + "·" * (width - n)


# ----------------------------------------------------------------------
# Sankey
# ----------------------------------------------------------------------
def ascii_sankey(
    flow: Sequence[Sequence[int]],
    experts: Sequence[str],
    *,
    width: int = 28,
    label: str = "",
) -> str:
    """把「域 → 专家」流量画成逐行堆叠条。**塌缩与分化一眼可见。**

    每行是一个域，条内按专家切段，段长 = 该域流向该专家的比例。
    末列标出主导专家与其占比 —— 占比接近 1 就是「这个域被一个专家独占」（分化），
    所有域的主导专家相同就是「塌缩」。
    """
    lines: list[str] = []
    if label:
        lines.append(f"  {label}")
    totals = [sum(row) for row in flow]
    grand = sum(totals) or 1
    for d, row in enumerate(flow):
        tot = totals[d] or 1
        seg = ""
        best, best_v = 0, -1
        for i, v in enumerate(row):
            if v > best_v:
                best, best_v = i, v
            n = int(round(v / tot * width))
            seg += ("0123456789"[(i % 10)]) * n
        seg = seg[:width].ljust(width, "·")
        share = best_v / tot if tot else 0.0
        lines.append(
            f"    域{d} ({totals[d]:>3d}) {seg}  → {experts[best]} {share:5.1%}"
            f"   占全场 {totals[d] / grand:5.1%}"
        )
    # 塌缩检测
    per_expert = [sum(row[i] for row in flow) for i in range(len(experts))]
    top = max(per_expert) / grand if grand else 0.0
    if top > 0.5:
        lines.append(f"    ⚠ 塌缩：{experts[per_expert.index(max(per_expert))]} 独占 {top:.1%} 的 token")
    doms = [max(range(len(row)), key=lambda i: row[i]) for row in flow]
    if len(set(doms)) == len(doms):
        lines.append(f"    分化：{len(doms)} 个域各自流向不同专家（一一对应）")
    return "\n".join(lines)


def ascii_scatter(rows: Sequence[tuple[float, float, str]], *, width: int = 46, height: int = 14) -> str:
    """通用二维散点：``(x, y, label)``。用于「均衡 × 分化」这张关键图。"""
    if not rows:
        return "  （无数据）"
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 - x0 < 1e-9:
        x1 = x0 + 1e-9
    if y1 - y0 < 1e-9:
        y1 = y0 + 1e-9
    grid = [[" "] * width for _ in range(height)]
    for x, y, lab in rows:
        cx = int((x - x0) / (x1 - x0) * (width - 1))
        cy = height - 1 - int((y - y0) / (y1 - y0) * (height - 1))
        ch = lab[0] if lab else "*"
        grid[cy][cx] = ch if grid[cy][cx] == " " else "+"
    out = [f"    y={y1:.3f} ┤" + "".join(grid[0])]
    for r in range(1, height):
        out.append(" " * 10 + "│" + "".join(grid[r]))
    out.append(f"    y={y0:.3f} └" + "─" * width)
    out.append(f"{'':12s}x={x0:.3f}{' ' * max(1, width - 14)}x={x1:.3f}")
    return "\n".join(out)


# ----------------------------------------------------------------------
# 各节
# ----------------------------------------------------------------------
def _header(result: Any) -> str:
    v = result.version
    cm = result.corpus_meta
    lines = [
        "# MoEFactory 报告",
        "",
        "> 本项目的产物是**证据**：路由系统在每个配置下的行为、以及可被反驳的结论。",
        "> 它**不输出「最优架构」** —— 那属于 L2-architecturefactory。",
        "",
        "## 一、口径",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 版本 | `{v.version_id}` |",
        f"| 内容摘要 | `{v.digest}` |",
        f"| 协议指纹 | `{v.protocol_tag}` |",
        f"| 创建时间 | {v.created_at}（唯一不可逐位复现的字段） |",
        f"| 语料 | {cm.get('name')}：{cm.get('n_domains')} 域 × {cm.get('n_content')} 内容 token，"
        f"vocab {cm.get('vocab_size')}，seq {cm.get('seq_len')} |",
        f"| 语料摘要 | `{v.corpus_digest}`（内容寻址：语料变则版本变） |",
        f"| 网格 | {v.counts['grid']} 格 + 复核 {v.counts['replication']} 次 = "
        f"{v.counts['observations']} 次观测 |",
        f"| **可计算下界** | `domain_blind_floor = {result.floor:.4f}` |",
        f"| 观测目标 | **无**（测量型：只观测、不优化、不选优） |",
        "",
        "**下界是怎么来的。** 内容词表在域之间共享，所以同一个内容 token 在不同域有不同后继。"
        "一个**看不见域**的路由器对每个 token 只能押一个后继，其最优损失就是"
        "「给定 token 时后继的熵」的平均值 —— 这个数**从任务构造里算出来**，不是拟合的。"
        "于是：**loss 显著低于它 ⟺ 路由一定用上了域信息。**",
        "",
    ]
    return "\n".join(lines)


def _findings(result: Any) -> str:
    lines = ["## 二、结论", "", "每条结论都由数据算出（`findings.py`），带证据、可被反驳。", ""]
    for i, f in enumerate(result.findings, 1):
        mark = "✅ 支持" if f.supported else "❌ 不支持"
        lines.append(f"### {i}. {f.claim}")
        lines.append("")
        lines.append(f"**判定：{mark}**")
        lines.append("")
        if f.evidence:
            lines.append("| 证据 | 值 |")
            lines.append("| --- | --- |")
            for k, v in f.evidence.items():
                lines.append(f"| `{k}` | {_fmt(v)} |")
            lines.append("")
        if f.note:
            lines.append(f"> {f.note}")
            lines.append("")
    return "\n".join(lines)


def _sankey_section(result: Any) -> str:
    lines = [
        "## 三、Routing Sankey",
        "",
        "左列是域，右列是专家；每行的条按「该域流向各专家的比例」切段，段上的数字是专家编号。",
        "**这一节回答 README 点名的问题：专家有没有形成 specialization。**",
        "",
    ]
    picks: list[RunRecord] = []
    learned = [r for r in result.by_group("learned") if r.seed == result.runs[0].seed]
    if learned:
        picks.append(min(learned, key=lambda r: r.final["loss"]))
        worst = max(learned, key=lambda r: r.final["loss"])
        if worst.spec_id != picks[0].spec_id:
            picks.append(worst)
    ctl = [r for r in result.by_group("control") if r.seed == result.runs[0].seed]
    if ctl:
        picks.append(min(ctl, key=lambda r: r.final["entropy_norm"]))
        picks.append(max(ctl, key=lambda r: r.final["loss"]))
    seen: set[str] = set()
    for r in picks:
        if r.spec_id in seen:
            continue
        seen.add(r.spec_id)
        experts = [f"E{i}" for i in range(len(r.domain_expert[0]) if r.domain_expert else 0)]
        head = (
            f"**{_tag(r)}**　loss={r.final['loss']:.3f}　"
            f"均衡 H={r.final['entropy_norm']:.3f}　分化 nmi={r.final['nmi']:.3f}　"
            f"purity={r.final['purity']:.3f}"
        )
        lines.append(head)
        lines.append("")
        lines.append("```")
        lines.append(ascii_sankey(r.domain_expert, experts))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _balance_vs_specialization(result: Any) -> str:
    seed0 = result.runs[0].seed
    rows = [r for r in result.runs if r.seed == seed0]
    lines = [
        "## 四、均衡 ≠ 分化",
        "",
        "**均衡** = 活干得均不均匀（`entropy_norm` 越接近 1 越均匀）；",
        "**分化** = 专家有没有形成职能（`nmi` / `purity` 越高越分化）。",
        "本项目的核心论点是：这是**两个独立的轴**，而且只看前者会选错。",
        "",
        "### 4.1 按均衡排序（前 8 名）",
        "",
        "| # | 配置 | 组 | H | imb | loss | nmi | 突破下界 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(sorted(rows, key=lambda r: -r.final["entropy_norm"])[:8], 1):
        lines.append(
            f"| {i} | `{r.spec_id}` | {_group_of(r)} | {r.final['entropy_norm']:.4f} | "
            f"{r.final['imbalance']:.2f} | {r.final['loss']:.3f} | {r.final['nmi']:.3f} | "
            f"{'是' if r.final['below_floor'] else '否'} |"
        )
    lines += [
        "",
        "> 如果只看「均衡」这一列，第一名是对照组 —— 它按构造完美均衡，却**在结构上"
        "不可能学会任务**（内容 token 的域信息不经过它，见 `moe.py` 的三行说明）。",
        "> 这就是「均衡 ≠ 分化」最干净的证据。",
        "",
        "### 4.2 均衡 × 分化 散点",
        "",
        "```",
    ]
    pts: list[tuple[float, float, str]] = []
    for r in rows:
        ch = "L" if _group_of(r) == "learned" else "C"
        pts.append((r.final["entropy_norm"], r.final["nmi"], ch))
    lines.append(ascii_scatter(pts))
    lines += [
        "```",
        "",
        "（横轴 = 均衡，纵轴 = 分化；`L` = learned 组，`C` = 对照组）",
        "",
        "### 4.3 aux 这一轴是平的：分化不靠它",
        "",
        "| 配置 | aux | H | imb | loss | nmi |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(
        [x for x in result.by_group("learned") if x.seed == seed0],
        key=lambda x: (x.spec["aux_weight"], x.spec["top_k"], x.spec["capacity_factor"]),
    ):
        lines.append(
            f"| `{r.spec_id}` | {r.spec['aux_weight']:g} | {r.final['entropy_norm']:.4f} | "
            f"{r.final['imbalance']:.2f} | {r.final['loss']:.3f} | {r.final['nmi']:.3f} |"
        )
    lines += [
        "",
        "**这张表是本项目最反直觉的一条。** `aux=0` 时分化照样发生 —— 也就是说"
        "**辅助损失不是分化的必要条件**。而且它与均衡的关系也很弱且不单调"
        "（逐轴看有反向的格子），所以连「辅助损失买到均衡」这个常见说法在本任务里也没站住。",
        "",
        "分化来自另一件事：**路由会被任务梯度塑形**。只要路由参与前向、梯度能回到它，"
        "它就会朝「把 token 送到擅长它的专家」方向走 —— 这与 aux 无关。",
        "aux 真正能影响的是**负载分布的形状**，而本项目的结论是："
        "那个形状与「路由有没有学到东西」几乎是两件不相干的事。",
        "",
    ]
    return "\n".join(lines)


def _replication_section(result: Any) -> str:
    rep = result.replication
    if not rep:
        return "## 五、多种子复核\n\n（未配置复核）\n"
    lines = [
        "## 五、多种子复核",
        "",
        "单次运行的结果可能是运气。下面把几个角落配置各跑多个种子，看分离是否稳定。",
        "",
        "| 配置 | 组 | n | loss 均值 | min | max | nmi 均值 | 突破下界 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in rep["per_spec"]:
        lines.append(
            f"| `{p['spec_id']}` | {p['routing']} | {p['n']} | {p['loss_mean']:.3f} | "
            f"{p['loss_min']:.3f} | {p['loss_max']:.3f} | {p['nmi_mean']:.3f} | "
            f"{p['success']}/{p['n']} |"
        )
    lines += [
        "",
        f"**learned 组：{rep['learned_success']}/{rep['learned_total']} 突破下界；"
        f"对照组：{rep['control_success']}/{rep['control_total']}。**",
        "",
        "对照组一次都没有突破 —— 这不是「训练不够」，而是结构性的：它们的路由"
        "与任务无关，训练再久也只能学会「给定 token 猜一个后继」，正好停在可计算下界上。",
        "",
    ]
    return "\n".join(lines)


def _curves_section(result: Any) -> str:
    seed0 = result.runs[0].seed
    lines = [
        "## 六、逐 step 曲线",
        "",
        "观测固定在**测试集**上（同一批 token、同一个顺序），所以曲线是「路由器学到了"
        "什么」的证据，而不是「这批数据是什么」。",
        "",
    ]
    picks = []
    learned = [r for r in result.by_group("learned") if r.seed == seed0]
    if learned:
        picks.append(min(learned, key=lambda r: r.final["loss"]))
    ctl = [r for r in result.by_group("control") if r.seed == seed0]
    if ctl:
        picks.append(min(ctl, key=lambda r: r.final["entropy_norm"]))
    for r in picks:
        lines.append(f"**{_tag(r)}**　onset（分化起始步）= {r.onset_step}")
        lines.append("")
        lines.append("| step | loss | acc | H | nmi | purity | 丢弃率 | 稳定性 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for s in r.snapshots:
            stab = "—" if s["stability"] is None else f"{s['stability']:.2f}"
            lines.append(
                f"| {s['step']} | {s['loss']:.3f} | {s['acc']:.3f} | {s['entropy_norm']:.3f} | "
                f"{s['nmi']:.3f} | {s['purity']:.3f} | {s['dropped_rate']:.3f} | {stab} |"
            )
        lines.append("")
        lines.append("```")
        lines.append("  loss " + bar(1.0 - min(1.0, r.snapshots[-1]["loss"] / 4.0), 40))
        lines.append("  nmi  " + bar(min(1.0, r.snapshots[-1]["nmi"]), 40))
        lines.append("```")
        lines.append("")
    lines.append("**onset 用「相邻 step 最大增幅」定位**，不用「首次超过阈值」——"
                 "后者对阈值敏感，平曲线会给出假步号。平曲线返回 `None`，表示"
                 "「没有在哪一步专门形成」。")
    lines.append("")
    return "\n".join(lines)


def _cost_section(result: Any) -> str:
    seed0 = result.runs[0].seed
    rows = [r for r in result.runs if r.seed == seed0]
    lines = [
        "## 七、代价：丢弃与通信",
        "",
        "| 配置 | cap | k | 丢弃率 | 通信量(元素) | 稀疏激活参数 | 总参数 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(rows, key=lambda r: -r.final["dropped_rate"])[:10]:
        lines.append(
            f"| `{r.spec_id}` | {r.spec['capacity_factor']:g} | {r.spec['top_k']} | "
            f"{r.final['dropped_rate']:.3f} | {int(r.final['comm_units'])} | "
            f"{int(r.final['active_params'])} | {r.params} |"
        )
    lines += [
        "",
        "**稀疏派发是真的稀疏。** 专家部分只对「被派发给它」的 token 做前向，所以"
        "``active_params`` 小于总参数量（`top_k / n_experts` 的量级）。"
        "把每个 token 送进每个专家再乘掩码得到的是稠密网络 —— 那样 MoE 就没有存在理由了。",
        "",
        "容量因子越紧，丢弃率越高；丢弃的 token 保留残差通路，所以表现为"
        "「路由白学了」而不是「模型崩了」。",
        "",
    ]
    return "\n".join(lines)


def _reproducibility(result: Any) -> str:
    return "\n".join(
        [
            "## 八、可复现性边界",
            "",
            "| 字段 | 是否逐位可复现 | 说明 |",
            "| --- | --- | --- |",
            "| `version_id` / `digest` | 是 | 内容寻址，语料 + 网格决定 |",
            "| 全部指标（loss / nmi / load / …） | 是 | 同配置同种子必得同值 |",
            "| `snapshots` | 是 | 观测固定在测试集上，无 batch 噪声 |",
            "| `train_seconds` | **否** | 唯一的实测耗时字段 |",
            "| `created_at` | **否** | 时间戳 |",
            "",
            "**唯一不可复现的部分被显式列出**，而不是「大概都一样」。"
            "测试里有一条专门钉住「除 `train_seconds` 外两次运行逐位相同」。",
            "",
        ]
    )


def render_report(result: Any) -> str:
    """渲染完整报告。**顺序即论点顺序。**"""
    parts = [
        _header(result),
        _findings(result),
        _sankey_section(result),
        _balance_vs_specialization(result),
        _replication_section(result),
        _curves_section(result),
        _cost_section(result),
        _reproducibility(result),
    ]
    return "\n".join(parts)
