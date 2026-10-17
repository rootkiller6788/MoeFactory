"""血缘 DAG：把「语料 → 每个格子 → 收拢 → 版本」画成一张图。

**收拢点的入边是本形状的存在理由。** 图里有一个 ``collect`` 节点，它的父节点
必须**恰好**是全部 ``observe:*`` 节点 —— 一个不多、一个不少：

- 「一个不多」保证图没把落选/被丢弃的格子偷偷接进来；
- 「一个不少」保证每个格子真的被收拢了，没有静默漏掉的分支。

于是「这张图覆盖了哪些观测」看一张图就知道，不必翻报告。这条不变量由测试钉住。

与 Optimizer 型（DataMixtureFactory）的差别：那边 ``propose`` 节点的父是各父试验的
``evaluate`` 节点（候选是搜出来的）；这边**没有搜索**，格子是配置的笛卡尔积
**展开**出来的，所以每个 ``spec`` 节点直接挂在语料节点下。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

#: 收拢节点 id。报告与测试都引用它。
COLLECT_NODE_ID = "collect"
CORPUS_NODE_ID = "corpus"
VERSION_NODE_ID = "version"


@dataclass
class Node:
    node_id: str
    kind: str
    label: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "kind": self.kind, "label": self.label, "meta": self.meta}


@dataclass
class Edge:
    src: str
    dst: str
    kind: str = "feeds"

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "kind": self.kind}


@dataclass
class Lineage:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    # -- 构建 --------------------------------------------------------
    def add_node(self, node: Node) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"节点已存在：{node.node_id}")
        self.nodes[node.node_id] = node

    def add_edge(self, src: str, dst: str, kind: str = "feeds") -> None:
        for nid in (src, dst):
            if nid not in self.nodes:
                raise ValueError(f"边引用了不存在的节点：{nid}")
        self.edges.append(Edge(src, dst, kind))

    # -- 查询 --------------------------------------------------------
    def parents(self, node_id: str) -> list[str]:
        return sorted({e.src for e in self.edges if e.dst == node_id})

    def children(self, node_id: str) -> list[str]:
        return sorted({e.dst for e in self.edges if e.src == node_id})

    def path_to_root(self, node_id: str) -> list[str]:
        """到根的单条路径（沿第一条入边回溯）。用于「谱系是不是图里的真实路径」。"""
        path = [node_id]
        seen = {node_id}
        cur = node_id
        while True:
            ps = self.parents(cur)
            if not ps:
                break
            nxt = ps[0]
            if nxt in seen:
                raise ValueError(f"血缘里出现环：{nxt}")
            seen.add(nxt)
            path.append(nxt)
            cur = nxt
        return path

    def depth(self, node_id: str) -> int:
        return len(self.path_to_root(node_id)) - 1

    def roots(self) -> list[str]:
        return sorted(n for n in self.nodes if not self.parents(n))

    def leaves(self) -> list[str]:
        return sorted(n for n in self.nodes if not self.children(n))

    # -- 序列化 ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[k].to_dict() for k in sorted(self.nodes)],
            "edges": [e.to_dict() for e in sorted(self.edges, key=lambda e: (e.src, e.dst, e.kind))],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Lineage":
        lin = Lineage()
        for nd in d["nodes"]:
            lin.add_node(Node(nd["node_id"], nd["kind"], nd["label"], dict(nd.get("meta", {}))))
        for ed in d["edges"]:
            lin.add_edge(ed["src"], ed["dst"], ed.get("kind", "feeds"))
        return lin

    # -- 导出 --------------------------------------------------------
    def to_dot(self, highlight: Iterable[str] = ()) -> str:
        hl = set(highlight)
        lines = ["digraph moefactory {", '  rankdir=LR;', '  node [shape=box, fontname="monospace"];']
        for nid in sorted(self.nodes):
            n = self.nodes[nid]
            color = '"#c0392b"' if nid in hl else '"#2c3e50"'
            lines.append(f'  "{nid}" [label="{n.label}", color={color}];')
        for e in sorted(self.edges, key=lambda e: (e.src, e.dst)):
            lines.append(f'  "{e.src}" -> "{e.dst}" [label="{e.kind}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def to_ascii(self, limit: int = 40) -> str:
        """按层打印。格子多的时候只列前 ``limit`` 个。"""
        out: list[str] = []
        roots = self.roots()
        out.append(f"血缘：{len(self.nodes)} 节点 / {len(self.edges)} 边")
        for r in roots:
            out.append(f"  根 {r}")
        obs = sorted(n for n in self.nodes if n.startswith("observe:"))
        shown = obs[:limit]
        out.append(f"  observe 节点 {len(obs)} 个" + (f"（只列前 {limit} 个）" if len(obs) > limit else ""))
        for nid in shown:
            out.append(f"    {self.nodes[nid].label}  ←  " + ", ".join(self.parents(nid)))
        if len(obs) > limit:
            out.append(f"    … 另有 {len(obs) - limit} 个")
        out.append(f"  收拢 {COLLECT_NODE_ID} ← {len(self.parents(COLLECT_NODE_ID))} 个 observe 节点")
        out.append(f"  收尾 {VERSION_NODE_ID} ← {', '.join(self.parents(VERSION_NODE_ID))}")
        return "\n".join(out)


def spec_node(spec_id: str) -> str:
    return f"spec:{spec_id}"


def train_node(spec_id: str, seed: int) -> str:
    return f"train:{spec_id}#s{seed}"


def observe_node(spec_id: str, seed: int) -> str:
    return f"observe:{spec_id}#s{seed}"


def build_lineage(
    corpus_meta: dict[str, Any],
    run_keys: Sequence[tuple[str, int]],
    *,
    n_train: int,
    n_test: int,
    version_id: str,
    grid: dict[str, Any] | None = None,
) -> Lineage:
    """建图：``corpus → spec → train → observe``，全部 ``observe`` 收拢到 ``collect``。

    ``run_keys`` 是 ``(spec_id, seed)`` 列表，必须与落盘的运行**一一对应**。
    ``grid`` 是扇出完整性的**声明**（各轴取值数），存进收拢节点的 meta ——
    于是「从产物重建」时也能核对「格数 == 各轴取值数之积」，不必重跑配置展开。
    """
    lin = Lineage()
    lin.add_node(
        Node(
            CORPUS_NODE_ID,
            "corpus",
            f"corpus {corpus_meta.get('name')} ({corpus_meta.get('n_domains')}域)",
            dict(corpus_meta),
        )
    )
    for spec_id, seed in run_keys:
        sn = spec_node(spec_id)
        if sn not in lin.nodes:
            lin.add_node(Node(sn, "spec", spec_id, {"spec_id": spec_id}))
            lin.add_edge(CORPUS_NODE_ID, sn, "expands")
        tn = train_node(spec_id, seed)
        on = observe_node(spec_id, seed)
        lin.add_node(Node(tn, "train", f"train {spec_id} seed={seed}", {"seed": seed}))
        lin.add_node(Node(on, "observe", f"observe {spec_id} seed={seed}", {"seed": seed}))
        lin.add_edge(sn, tn, "trains")
        lin.add_edge(tn, on, "observes")

    lin.add_node(
        Node(
            COLLECT_NODE_ID,
            "collect",
            f"收拢 {len(run_keys)} 个观测",
            {"n_train": n_train, "n_test": n_test, "grid": grid or {}},
        )
    )
    for spec_id, seed in run_keys:
        lin.add_edge(observe_node(spec_id, seed), COLLECT_NODE_ID, "collects")
    lin.add_node(Node(VERSION_NODE_ID, "version", f"version {version_id}", {"version_id": version_id}))
    lin.add_edge(COLLECT_NODE_ID, VERSION_NODE_ID, "seals")
    return lin


def dumps(lin: Lineage) -> str:
    return json.dumps(lin.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
