"""MoEFactory —— 把 ArchitectureFactory 的一个 MoE 节点纵向钻穿。

核心问题不是「实现 MoE」，而是：

    **Token → Expert 的动态路由系统究竟发生了什么？**

形状是**测量型（Microscope）**：被观测对象是一个训练中的 MoE 标本，扇出的是
**观测/配置网格**，产物是**路由证据**（逐 step 曲线 + 对比矩阵 + Routing Sankey
+ 可执行结论），而不是「一份数据集」或「一条 Pareto 前沿」。

导出核心对象，方便 ``from moefactory import ...``。
"""

from .autograd import Tensor
from .config import expand_grid, grid_report, load_config, validate_config
from .corpus import Corpus, Sample, build_corpus, make_batch, read_corpus, write_corpus
from .moe import MoEModel, RouteTrace, build_model
from .observe import (
    distinct_primary_sets,
    domain_expert_matrix,
    dominant_expert,
    is_bijection,
    observe,
    primary_experts,
    summarize,
)
from .schema import (
    AUX_APPLICABLE,
    METRIC_DIRECTIONS,
    ROUTINGS,
    TIMING_FIELDS,
    Finding,
    MoESpec,
    RoutingSnapshot,
    RunRecord,
    Version,
)
from .train import RunOutcome, TrainConfig, protocol_digest, run_spec, train_proxy

__all__ = [
    "Tensor",
    "load_config",
    "validate_config",
    "expand_grid",
    "grid_report",
    "Corpus",
    "Sample",
    "build_corpus",
    "make_batch",
    "write_corpus",
    "read_corpus",
    "MoEModel",
    "build_model",
    "RouteTrace",
    "observe",
    "summarize",
    "domain_expert_matrix",
    "dominant_expert",
    "is_bijection",
    "primary_experts",
    "distinct_primary_sets",
    "MoESpec",
    "RoutingSnapshot",
    "RunRecord",
    "Finding",
    "Version",
    "ROUTINGS",
    "AUX_APPLICABLE",
    "METRIC_DIRECTIONS",
    "TIMING_FIELDS",
    "TrainConfig",
    "RunOutcome",
    "protocol_digest",
    "run_spec",
    "train_proxy",
]

__version__ = "0.1.0"
