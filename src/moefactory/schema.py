"""核心对象：配置、观测快照、网格运行记录、结论、版本。

**本项目是「测量型」（Microscope）**，所以这里没有 `Trial` / `Pareto` 这类
「谁更好」的对象。取而代之的是：

- :class:`MoESpec` —— 一个**标本配置**（路由器怎么路由、容量多紧、辅助损失多大）。
  它既是被观测的对象，也是扇出网格的一个坐标。
- :class:`RoutingSnapshot` —— 某一训练步上的**一次观测**（负载 / 熵 / 丢弃 /
  分化度 / 通信量）。曲线就是快照序列。
- :class:`RunRecord` —— 一个网格格子的全部观测 + 它在网格里的坐标。
- :class:`Finding` —— 一条**可执行结论**：主张 + 证据 + 是否被实测支持。

**关键设计：`spec_id` 必须可逆。** 报告、CLI、只读命令都靠它定位一个格子；
一旦不可逆，「从产物重建结果」这条只读契约就断了（见坑 41/42）。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

#: 可用的路由方式。三者构成实验的对照骨架（见 ``moe.py`` 的表）：
#: ``learned`` 是被测对象；``static``（``token % n_experts``）**按构造完美均衡却零分化**，
#: 用来钉死「均衡 ≠ 分化」；``random`` 是地板（连 token 都不看）。
ROUTINGS: tuple[str, ...] = ("learned", "static", "random")

#: 只有 learned 路由下「辅助损失」这个坐标才有意义。对照组的该轴是 **N/A**，
#: 不是 0 —— 把「不适用」当 0 算进均值会让结论偏移（同坑 10）。
AUX_APPLICABLE: frozenset[str] = frozenset({"learned"})

#: 允许不可逐位复现的字段（只有实测耗时）。
TIMING_FIELDS: frozenset[str] = frozenset({"train_seconds"})

#: 有方向的指标只有质量类；路由指标**没有方向**（均衡不等于更好，见 README 命题）。
METRIC_DIRECTIONS: dict[str, str] = {"loss": "min", "acc": "max"}


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _num(v: float) -> str:
    """把浮点写成可逆且稳定的字符串（``%g`` 去掉多余的零）。"""
    return f"{v:g}"


# ----------------------------------------------------------------------
# 标本配置
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MoESpec:
    """一个 MoE 标本的配置。**它是数据，不是代码分支。**

    ``routing`` / ``aux_weight`` / ``capacity_factor`` 决定路由器**怎么选**，
    ``n_experts`` / ``top_k`` / ``d_model`` / ``d_ff`` 决定**选给谁**。
    没有一处 ``if spec.mixer == ...`` 式的分支 —— 装配只查表。
    """

    n_experts: int = 4
    top_k: int = 1
    d_model: int = 12
    d_ff: int = 16
    routing: str = "learned"
    aux_weight: float = 0.05
    capacity_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.routing not in ROUTINGS:
            raise ValueError(f"未知 routing：{self.routing}（可选 {ROUTINGS}）")
        if self.top_k < 1 or self.top_k > self.n_experts:
            raise ValueError(f"top_k={self.top_k} 必须在 1..n_experts({self.n_experts}) 之间")
        if self.n_experts < 2:
            raise ValueError("n_experts 至少为 2，否则「路由」无从谈起")
        if self.d_model < 1 or self.d_ff < 1:
            raise ValueError("d_model / d_ff 必须为正")
        if self.aux_weight < 0.0:
            raise ValueError("aux_weight 不能为负")
        if self.capacity_factor < 0.0:
            raise ValueError("capacity_factor 不能为负（0 表示不限容量）")

    # -- 语义 --------------------------------------------------------
    @property
    def aux_applicable(self) -> bool:
        """辅助损失在这个标本上是否适用。对照组为 ``False``（N/A，不是 0）。"""
        return self.routing in AUX_APPLICABLE

    @property
    def capacity_is_limited(self) -> bool:
        return self.capacity_factor > 0.0

    def capacity_per_expert(self, n_tokens: int) -> float:
        """每个专家在每个 batch 里能接多少 token（不限容量时返回 ``inf``）。"""
        if not self.capacity_is_limited:
            return math.inf
        return self.capacity_factor * n_tokens * self.top_k / self.n_experts

    # -- 规范化 / 序列化 ---------------------------------------------
    def canonical(self) -> "MoESpec":
        """规范化：把浮点收成稳定表示（``-0.0`` → ``0.0``、去掉尾零）。

        搜索/展开出来的 spec 一律先过这里，否则「同一个标本两个 id」（同坑 51）。
        """
        return MoESpec(
            n_experts=int(self.n_experts),
            top_k=int(self.top_k),
            d_model=int(self.d_model),
            d_ff=int(self.d_ff),
            routing=str(self.routing),
            aux_weight=round(float(self.aux_weight), 6) + 0.0,
            capacity_factor=round(float(self.capacity_factor), 6) + 0.0,
        )

    @property
    def spec_id(self) -> str:
        s = self.canonical()
        return (
            f"e{s.n_experts}-k{s.top_k}-d{s.d_model}-f{s.d_ff}"
            f"-{s.routing}-aux{_num(s.aux_weight)}-cap{_num(s.capacity_factor)}"
        )

    @staticmethod
    def from_spec_id(text: str) -> "MoESpec":
        """``spec_id`` 的逆运算。不可逆就等于把只读命令堵死。"""
        try:
            parts = text.split("-")
            if len(parts) != 7:
                raise ValueError("字段数不对")
            e, k, d, f, routing, aux, cap = parts
            return MoESpec(
                n_experts=int(e[1:]),
                top_k=int(k[1:]),
                d_model=int(d[1:]),
                d_ff=int(f[1:]),
                routing=routing,
                aux_weight=float(aux[3:]),
                capacity_factor=float(cap[3:]),
            ).canonical()
        except (ValueError, IndexError) as exc:  # pragma: no cover - 错误路径
            raise ValueError(f"无法解析 spec_id：{text!r}（{exc}）") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.canonical())

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "MoESpec":
        return MoESpec(
            n_experts=d["n_experts"],
            top_k=d["top_k"],
            d_model=d["d_model"],
            d_ff=d["d_ff"],
            routing=d["routing"],
            aux_weight=d["aux_weight"],
            capacity_factor=d["capacity_factor"],
        ).canonical()

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    # -- 规模 --------------------------------------------------------
    def param_count(self, vocab_size: int) -> int:
        """参数量：嵌入 + **一个** RMS 缩放 + 路由器 + 全部专家 + 输出头。

        这个数必须与 ``MoEModel.num_params`` **逐位相等** —— 它是 ``explain``
        命令与「稀疏激活比例」的口径来源。曾经写成 ``2 * d_model``（以为有两个
        RMS 缩放，实际只有 ``norm_w`` 一个），于是 ``explain`` 报的参数量比模型
        实际多出 ``d_model``，而**没有任何地方会因此报错**（见 README 附录 C 的
        坑 59）。这条不变量现在由 ``tests/test_moe.py`` 钉住。
        """
        emb = vocab_size * self.d_model
        router = self.d_model * self.n_experts + self.n_experts
        expert = self.n_experts * (
            self.d_model * self.d_ff + self.d_ff + self.d_ff * self.d_model + self.d_model
        )
        head = self.d_model * vocab_size + vocab_size
        norms = self.d_model
        return emb + router + expert + head + norms


# ----------------------------------------------------------------------
# 观测
# ----------------------------------------------------------------------
@dataclass
class RoutingSnapshot:
    """某一训练步上的一次路由观测。曲线的每个点就是一个它。"""

    step: int
    loss: float
    acc: float
    load: list[int]                  # 每个专家接到的 token 数（含 top_k 重复计数）
    load_frac: list[float]
    entropy: float                   # -Σ p log p，p = load_frac
    entropy_norm: float              # 除以 log(n_experts)；1.0 = 完全均衡
    imbalance: float                 # max(load) / mean(load)；1.0 = 完全均衡
    load_cv: float                   # 负载变异系数
    dropped: int
    dropped_rate: float
    comm_units: int                  # all-to-all 的元素搬运量
    nmi: float                       # 域 ↔ 专家 的标准化互信息 = 分化度
    purity: float                    # 每个专家最主导域的占比，按负载加权
    expert_domain: list[float]       # 每个专家的「最主导域占比」
    gate_max_mean: float             # 平均最大门控值（路由有多自信）
    expert_sim: float                # 专家两两余弦相似度均值（越低越分化）
    stability: float | None          # 与上一快照相比，top-1 未变的 token 占比

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RoutingSnapshot":
        return RoutingSnapshot(**d)


@dataclass
class RunRecord:
    """网格里的一个格子：坐标 + 全部观测。"""

    spec_id: str
    spec: dict[str, Any]
    seed: int
    final: dict[str, float]
    snapshots: list[dict[str, Any]]
    domain_expert: list[list[int]]   # [n_domains][n_experts] 计数 → Sankey 的流
    domain_dominant: list[int]       # 每个域的主导专家（设计期望是一一对应）
    onset_step: int | None           # 分化起始步（相邻步最大增幅定位）
    params: int
    flops: int
    train_seconds: float
    #: 这条观测从哪来：``grid``（扇出网格，用默认种子）或 ``replication``（多种子复核）。
    #: **必须显式记下来**，否则「网格完整性」这类断言会把复核行也算进去 ——
    #: 实测就是这样把 24 格数成 39 格的。
    origin: str = "grid"
    accepted: bool = True
    reason: str = ""

    def metric(self, name: str) -> float:
        if name in self.final:
            return self.final[name]
        raise KeyError(f"{self.spec_id} 没有指标 {name}")

    def snapshot(self, step: int) -> dict[str, Any] | None:
        for s in self.snapshots:
            if s["step"] == step:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RunRecord":
        return RunRecord(**d)

    @property
    def aux_applicable(self) -> bool:
        return self.spec["routing"] in AUX_APPLICABLE


@dataclass
class Finding:
    """一条结论：主张 + 证据 + 是否被支持。

    **测量型的产物是「可执行结论」**，所以结论必须带上证据与判定，
    不能只写一句「看起来专家分化了」。
    """

    key: str
    claim: str
    supported: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Finding":
        return Finding(**d)


@dataclass
class Version:
    """内容寻址版本：语料 + 网格变了，版本就变；搜索/训练预算变了，版本**不变**。"""

    version_id: str
    digest: str
    created_at: str
    corpus_digest: str
    protocol_tag: str
    grid_size: int
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Version":
        return Version(**d)


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def normalized_entropy(counts: Sequence[int]) -> tuple[float, float]:
    """负载分布的熵与归一化熵（``1.0`` = 完全均衡）。"""
    total = sum(counts)
    if total <= 0:
        return 0.0, 0.0
    p = [c / total for c in counts]
    h = -math.fsum(pi * math.log(pi) for pi in p if pi > 0.0)
    hmax = math.log(len(counts)) if len(counts) > 1 else 1.0
    return h, (h / hmax if hmax > 0 else 0.0)


def imbalance_ratio(counts: Sequence[int]) -> float:
    """``max / mean``。``1.0`` = 完全均衡；``n_experts`` = 全压在一个专家上。"""
    total = sum(counts)
    if total <= 0:
        return 1.0
    mean = total / len(counts)
    return (max(counts) / mean) if mean > 0 else 1.0


def load_cv(counts: Sequence[int]) -> float:
    """负载变异系数（标准差 / 均值）。"""
    n = len(counts)
    total = sum(counts)
    if n == 0 or total <= 0:
        return 0.0
    mean = total / n
    var = math.fsum((c - mean) ** 2 for c in counts) / n
    return (math.sqrt(var) / mean) if mean > 0 else 0.0


def mutual_information(labels: Sequence[int], clusters: Sequence[int]) -> float:
    """域 ↔ 专家 的互信息（自然对数）。分化度的原料。"""
    n = len(labels)
    if n == 0:
        return 0.0
    joint: dict[tuple[int, int], int] = {}
    la: dict[int, int] = {}
    cb: dict[int, int] = {}
    for a, b in zip(labels, clusters):
        joint[(a, b)] = joint.get((a, b), 0) + 1
        la[a] = la.get(a, 0) + 1
        cb[b] = cb.get(b, 0) + 1
    mi = 0.0
    for (a, b), c in joint.items():
        pij = c / n
        pi = la[a] / n
        pj = cb[b] / n
        mi += pij * math.log(pij / (pi * pj))
    return mi


def entropy_of(labels: Sequence[int]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    counts: dict[int, int] = {}
    for a in labels:
        counts[a] = counts.get(a, 0) + 1
    return -math.fsum((c / n) * math.log(c / n) for c in counts.values())


def normalized_mutual_information(labels: Sequence[int], clusters: Sequence[int]) -> float:
    """NMI = I(X;Y) / mean(H(X), H(Y))。``0`` = 完全无关，``1`` = 完全一致。

    用**算术平均**归一化（不是 sqrt 或 min）：它对「专家数 ≠ 域数」不敏感，
    在本项目里域数恒等于专家数，三种归一化等价；选平均是为了让「专家数更多
    但只用到一部分」的配置不会因为分母变小而虚高。
    """
    mi = mutual_information(labels, clusters)
    ha = entropy_of(labels)
    hb = entropy_of(clusters)
    denom = (ha + hb) / 2.0
    if denom <= 0.0:
        return 0.0
    return mi / denom


def purity(labels: Sequence[int], clusters: Sequence[int], n_clusters: int) -> tuple[float, list[float]]:
    """按负载加权的「每个专家最主导域的占比」均值。``1.0`` = 每个专家纯一个域。"""
    per: dict[int, dict[int, int]] = {}
    for a, b in zip(labels, clusters):
        per.setdefault(b, {})
        per[b][a] = per[b].get(a, 0) + 1
    out = [0.0] * n_clusters
    total = 0
    weighted = 0.0
    for c, doms in per.items():
        c_total = sum(doms.values())
        best = max(doms.values()) / c_total if c_total else 0.0
        out[c] = best
        weighted += best * c_total
        total += c_total
    return (weighted / total if total else 0.0), out


def onset_step(series: Sequence[tuple[int, float]]) -> int | None:
    """用**相邻步最大增幅**定位「分化起始步」。

    不用「首次超过阈值」：那对阈值敏感，平曲线会给出假步号（同坑 6 的做法）。
    平曲线（没有明显台阶）返回 ``None``，表示「没有在哪一步专门形成」。
    """
    if len(series) < 2:
        return None
    best_gain = 0.0
    best_step: int | None = None
    for (s0, v0), (s1, v1) in zip(series, series[1:]):
        gain = v1 - v0
        if gain > best_gain:
            best_gain = gain
            best_step = s1
    return best_step if best_gain > 1e-9 else None


def mean_pairwise_cosine(vectors: Iterable[Sequence[float]]) -> float:
    """专家权重向量的两两余弦相似度均值。越低说明专家越「不像」。"""
    vs = [list(v) for v in vectors]
    n = len(vs)
    if n < 2:
        return 0.0
    norms = [math.sqrt(math.fsum(x * x for x in v)) for v in vs]
    vals: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] <= 0.0 or norms[j] <= 0.0:
                continue
            dot = math.fsum(a * b for a, b in zip(vs[i], vs[j]))
            vals.append(dot / (norms[i] * norms[j]))
    return (math.fsum(vals) / len(vals)) if vals else 0.0
