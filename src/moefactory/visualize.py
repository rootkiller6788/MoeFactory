"""自包含 HTML：内联 SVG 的 Sankey + 热力图 + 曲线。**不引用任何外部资源。**

**为什么是自包含的。** 报告要能当文件发出去、离线打开、十年后还能看。任何
``http://`` 引用都会在某一天变成坏链，让「证据」失效。所以图全部由代码生成
内联 SVG，字体只用系统栈。

Sankey 是本项目的招牌图：左列域、右列专家、带宽 = 流量。它一眼回答
「专家有没有形成 specialization」，而且**看得出 loss 曲线看不出的塌缩**。
"""

from __future__ import annotations

import html
from typing import Any, Sequence

#: 与 IDE 主题无关的固定深色系：报告文件要能脱离主题独立存在。
_BG = "#0f1419"
_PANEL = "#161d26"
_FG = "#e6edf3"
_DIM = "#8b98a5"
_ACCENT = "#4ea1ff"
_WARN = "#ff7b72"
_OK = "#3fb950"
_EXPERT_COLORS = ("#4ea1ff", "#3fb950", "#d29922", "#bc8cff", "#ff7b72", "#39c5cf")


def _esc(s: Any) -> str:
    return html.escape(str(s))


# ----------------------------------------------------------------------
# Sankey
# ----------------------------------------------------------------------
def sankey_svg(
    flow: Sequence[Sequence[int]],
    *,
    width: int = 420,
    height: int = 220,
    pad: int = 8,
) -> str:
    """左列域 → 右列专家的流量图。带宽 = 该流向的 token 数。"""
    n_dom = len(flow)
    n_exp = len(flow[0]) if flow else 0
    total = sum(sum(r) for r in flow) or 1
    usable = height - 2 * pad - max(0, n_dom - 1) * 6
    usable_r = height - 2 * pad - max(0, n_exp - 1) * 6

    left_h = [sum(r) / total * usable for r in flow]
    right_h = [sum(flow[d][e] for d in range(n_dom)) / total * usable_r for e in range(n_exp)]

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">'
    ]
    # 左列
    y = pad
    ly: list[float] = []
    for d in range(n_dom):
        ly.append(y)
        parts.append(
            f'<rect x="{pad}" y="{y:.1f}" width="10" height="{left_h[d]:.1f}" '
            f'fill="{_EXPERT_COLORS[d % len(_EXPERT_COLORS)]}" rx="2"/>'
        )
        parts.append(
            f'<text x="{pad + 16}" y="{y + left_h[d] / 2 + 3:.1f}" fill="{_FG}" '
            f'font-size="10">域{d} ({int(sum(flow[d]))})</text>'
        )
        y += left_h[d] + 6
    # 右列
    y = pad
    ry: list[float] = []
    for e in range(n_exp):
        ry.append(y)
        parts.append(
            f'<rect x="{width - pad - 10}" y="{y:.1f}" width="10" height="{right_h[e]:.1f}" '
            f'fill="{_EXPERT_COLORS[e % len(_EXPERT_COLORS)]}" rx="2"/>'
        )
        parts.append(
            f'<text x="{width - pad - 16}" y="{y + right_h[e] / 2 + 3:.1f}" fill="{_FG}" '
            f'font-size="10" text-anchor="end">E{e}</text>'
        )
        y += right_h[e] + 6
    # 流带
    lcur = list(ly)
    rcur = list(ry)
    x0, x1 = pad + 10, width - pad - 10
    for d in range(n_dom):
        for e in range(n_exp):
            v = flow[d][e]
            if v <= 0:
                continue
            h = v / total * usable
            l0, r0 = lcur[d], rcur[e]
            lcur[d] += h
            rcur[e] += h
            c = (x1 - x0) * 0.42
            parts.append(
                f'<path d="M{x0},{l0:.1f} C{x0 + c},{l0:.1f} {x1 - c},{r0:.1f} {x1},{r0:.1f} '
                f'L{x1},{r0 + h:.1f} C{x1 - c},{r0 + h:.1f} {x0 + c},{l0 + h:.1f} {x0},{l0 + h:.1f} Z" '
                f'fill="{_EXPERT_COLORS[e % len(_EXPERT_COLORS)]}" opacity="0.34"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------
# 折线
# ----------------------------------------------------------------------
def line_svg(
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
    *,
    width: int = 460,
    height: int = 150,
    y0: float = 0.0,
    y1: float = 1.0,
) -> str:
    """多序列折线。``series`` = ``[(label, [(x, y), ...]), ...]``。"""
    pad = 26
    xs = [p[0] for _, pts in series for p in pts]
    if not xs:
        return "<svg/>"
    x0, x1 = min(xs), max(xs)
    if x1 - x0 < 1e-9:
        x1 = x0 + 1
    span = (y1 - y0) or 1.0

    def px(x: float) -> float:
        return pad + (x - x0) / (x1 - x0) * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - y0) / span * (height - 2 * pad)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="none"/>',
    ]
    for frac in (0.0, 0.5, 1.0):
        yy = y0 + frac * span
        parts.append(
            f'<line x1="{pad}" y1="{py(yy):.1f}" x2="{width - pad}" y2="{py(yy):.1f}" '
            f'stroke="{_DIM}" stroke-width="0.5" opacity="0.35"/>'
        )
        parts.append(
            f'<text x="2" y="{py(yy) + 3:.1f}" fill="{_DIM}" font-size="8">{yy:.2f}</text>'
        )
    for i, (label, pts) in enumerate(series):
        color = _EXPERT_COLORS[i % len(_EXPERT_COLORS)]
        d = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        parts.append(
            f'<text x="{width - pad}" y="{12 + i * 11}" fill="{color}" font-size="9" '
            f'text-anchor="end">{_esc(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------
# 散点
# ----------------------------------------------------------------------
def scatter_svg(
    points: Sequence[tuple[float, float, str, bool]],
    *,
    width: int = 460,
    height: int = 240,
    xlabel: str = "均衡 H",
    ylabel: str = "分化 nmi",
) -> str:
    """``(x, y, label, is_control)``。**均衡 × 分化**那张关键图。"""
    pad = 34
    if not points:
        return "<svg/>"
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    x0, x1 = x0 - 0.004, x1 + 0.004
    y0, y1 = min(0.0, y0 - 0.03), y1 + 0.03

    def px(x: float) -> float:
        return pad + (x - x0) / (x1 - x0) * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - y0) / (y1 - y0) * (height - 2 * pad)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">',
        f'<rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" '
        f'fill="none" stroke="{_DIM}" stroke-width="0.6" opacity="0.5"/>',
        f'<text x="{width / 2}" y="{height - 6}" fill="{_DIM}" font-size="9" '
        f'text-anchor="middle">{_esc(xlabel)}</text>',
        f'<text x="10" y="{height / 2}" fill="{_DIM}" font-size="9" '
        f'transform="rotate(-90 10 {height / 2})" text-anchor="middle">{_esc(ylabel)}</text>',
    ]
    for x, y, lab, ctl in points:
        color = _WARN if ctl else _ACCENT
        parts.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.6" fill="{color}" opacity="0.9"/>'
        )
    parts.append(
        f'<circle cx="{pad + 12}" cy="{pad + 10}" r="3.6" fill="{_WARN}"/>'
        f'<text x="{pad + 20}" y="{pad + 13}" fill="{_DIM}" font-size="9">对照组</text>'
        f'<circle cx="{pad + 74}" cy="{pad + 10}" r="3.6" fill="{_ACCENT}"/>'
        f'<text x="{pad + 82}" y="{pad + 13}" fill="{_DIM}" font-size="9">learned</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------
# 页面
# ----------------------------------------------------------------------
def _card(title: str, body: str, ok: bool | None = None) -> str:
    border = _OK if ok else (_WARN if ok is False else _DIM)
    badge = "" if ok is None else (f'<span style="color:{border}">{"支持" if ok else "不支持"}</span>')
    return (
        f'<div style="background:{_PANEL};border-left:3px solid {border};padding:10px 14px;'
        f'margin:8px 0;border-radius:4px">'
        f'<div style="font-weight:600;margin-bottom:4px">{_esc(title)} {badge}</div>'
        f'<div style="color:{_DIM};font-size:13px;line-height:1.6">{body}</div></div>'
    )


def render_html(result: Any) -> str:
    """整页 HTML。**内联 SVG、无外部引用**（测试会断言这一点）。"""
    v = result.version
    seed0 = result.runs[0].seed if result.runs else 0
    rows = [r for r in result.runs if r.seed == seed0]

    cards: list[str] = []
    for f in result.findings:
        ev = "　".join(f"<code>{_esc(k)}={_esc(val)}</code>" for k, val in f.evidence.items())
        cards.append(_card(f.claim, ev, f.supported))

    # Sankey 精选：最好的 learned + 最均衡的对照组
    picks: list[Any] = []
    learned = [r for r in rows if r.spec["routing"] == "learned"]
    control = [r for r in rows if r.spec["routing"] != "learned"]
    if learned:
        picks.append(min(learned, key=lambda r: r.final["loss"]))
    if control:
        picks.append(min(control, key=lambda r: r.final["entropy_norm"]))
    sankeys = []
    for r in picks:
        sankeys.append(
            f'<div style="background:{_PANEL};padding:12px;border-radius:6px;margin:8px 0">'
            f'<div style="font-size:13px;margin-bottom:6px"><code>{_esc(r.spec_id)}</code>'
            f'　loss=<b>{r.final["loss"]:.3f}</b>　H={r.final["entropy_norm"]:.3f}'
            f'　nmi={r.final["nmi"]:.3f}</div>'
            f'{sankey_svg(r.domain_expert)}</div>'
        )

    pts = [
        (r.final["entropy_norm"], r.final["nmi"], r.spec_id, r.spec["routing"] != "learned")
        for r in rows
    ]
    scatter = scatter_svg(pts)

    curves = []
    if picks:
        curves.append(
            line_svg(
                [
                    ("loss", [(s["step"], s["loss"]) for s in picks[0].snapshots]),
                ],
                y0=0.0,
                y1=max(1.0, max(s["loss"] for s in picks[0].snapshots) * 1.05),
            )
        )
        curves.append(
            line_svg(
                [
                    ("nmi", [(s["step"], s["nmi"]) for s in picks[0].snapshots]),
                    ("H", [(s["step"], s["entropy_norm"]) for s in picks[0].snapshots]),
                ],
                y0=0.0,
                y1=1.0,
            )
        )

    trs = []
    for r in sorted(rows, key=lambda r: -r.final["entropy_norm"]):
        trs.append(
            "<tr>"
            f'<td><code>{_esc(r.spec_id)}</code></td>'
            f'<td>{_esc(r.spec["routing"])}</td>'
            f'<td>{r.spec["aux_weight"]:g}</td>'
            f'<td>{r.final["entropy_norm"]:.4f}</td>'
            f'<td>{r.final["imbalance"]:.2f}</td>'
            f'<td>{r.final["loss"]:.3f}</td>'
            f'<td>{r.final["nmi"]:.3f}</td>'
            f'<td>{"是" if r.final["below_floor"] else "否"}</td>'
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MoEFactory 报告 {_esc(v.version_id)}</title></head>
<body style="margin:0;background:{_BG};color:{_FG};
font-family:system-ui,-apple-system,'Segoe UI',sans-serif">
<div style="max-width:900px;margin:0 auto;padding:28px 20px 60px">
<h1 style="font-size:22px;margin:0 0 4px">MoEFactory · 路由系统观测报告</h1>
<div style="color:{_DIM};font-size:13px;margin-bottom:18px">
版本 <code>{_esc(v.version_id)}</code>　协议 <code>{_esc(v.protocol_tag)}</code>　
语料 <code>{_esc(v.corpus_digest)}</code>　
可计算下界 <b>{result.floor:.4f}</b>　
{len(result.runs)} 次观测
</div>

<h2 style="font-size:16px;border-bottom:1px solid {_PANEL};padding-bottom:6px">结论</h2>
{''.join(cards)}

<h2 style="font-size:16px;border-bottom:1px solid {_PANEL};padding-bottom:6px;margin-top:24px">Routing Sankey</h2>
<div style="color:{_DIM};font-size:13px">左列域 → 右列专家，带宽 = 流量。一眼看出塌缩与分化。</div>
{''.join(sankeys)}

<h2 style="font-size:16px;border-bottom:1px solid {_PANEL};padding-bottom:6px;margin-top:24px">均衡 × 分化</h2>
<div style="background:{_PANEL};padding:12px;border-radius:6px">{scatter}</div>

<h2 style="font-size:16px;border-bottom:1px solid {_PANEL};padding-bottom:6px;margin-top:24px">曲线</h2>
<div style="background:{_PANEL};padding:12px;border-radius:6px">
{''.join(curves)}
</div>

<h2 style="font-size:16px;border-bottom:1px solid {_PANEL};padding-bottom:6px;margin-top:24px">矩阵</h2>
<table style="width:100%;border-collapse:collapse;font-size:12px">
<thead><tr style="color:{_DIM};text-align:left">
<th>spec</th><th>routing</th><th>aux</th><th>H</th><th>imb</th><th>loss</th><th>nmi</th><th>突破下界</th>
</tr></thead>
<tbody>{''.join(trs)}</tbody></table>

<div style="color:{_DIM};font-size:12px;margin-top:24px">
本页由 <code>python main.py visualize</code> 生成，自包含、无外部引用，可离线打开。
</div>
</div></body></html>
"""
