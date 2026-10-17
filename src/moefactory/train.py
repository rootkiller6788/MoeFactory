"""训练：让标本从「随机路由器」变成「学到了什么的路由器」。

**四条硬约束**（都来自同工作区 L2-architecturefactory 的实测，见 README 附录 C）：

1. **batch loss 取平均，不取和。** 取和会让 ``grad_clip`` 把梯度整体压掉 ——
   表现是「训练完全不动」，而且**不报错**。
2. **一步一个 batch。** 一步攒完整个 epoch 再更新，等于白烧 ``n/batch`` 倍算力
   换同样的效果。
3. **梯度裁剪在更新之前算全局范数**（不是每个参数各自裁）—— 各自裁会改变
   参数间的相对步长，把均衡性训练变成噪声。
4. **观测在固定的评测集上做**（见 ``observe.py`` 的说明）。

本项目与 L2-architecturefactory 的差别：那边比的是「不同架构谁更好」，这边
比的是「同一套路由机制在不同配置下的**行为**」。所以这里的训练器**不选优**，
只如实记录每一步的路由事实。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from .autograd import Tensor, add, softmax_cross_entropy
from .corpus import Batch, Corpus, iter_batches, make_batch
from .moe import MoEModel, build_model
from .observe import domain_expert_matrix, dominant_expert, observe, summarize
from .schema import MoESpec, onset_step


@dataclass(frozen=True)
class TrainConfig:
    """训练预算。**所有标本共享同一份**，否则「可比」就没了。"""

    steps: int = 40
    lr: float = 0.2
    batch_size: int = 10
    momentum: float = 0.9
    init_scale: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 5

    @staticmethod
    def from_config(cfg: dict) -> "TrainConfig":
        p = cfg.get("proxy", {})
        return TrainConfig(
            steps=int(p.get("steps", 40)),
            lr=float(p.get("lr", 0.2)),
            batch_size=int(p.get("batch_size", 10)),
            momentum=float(p.get("momentum", 0.9)),
            init_scale=float(p.get("init_scale", 0.1)),
            grad_clip=float(p.get("grad_clip", 1.0)),
            eval_every=int(p.get("eval_every", 5)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def protocol_digest(cfg: dict) -> str:
    """协议指纹：**只由训练预算与任务定义决定**。

    它必须随 ``proxy`` 变（预算改了，历史观测就不可比），也必须随 ``task`` 变
    （语料换了，读数就不是同一件事）。反过来，它**不该**随网格大小、输出目录变。
    """
    payload = {
        "proxy": TrainConfig.from_config(cfg).to_dict(),
        "task": {k: v for k, v in cfg.get("task", {}).items() if not k.startswith("_")},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------------
# 优化器
# ----------------------------------------------------------------------
def _iter_elems(t: Tensor) -> Sequence[tuple[int, int | None]]:
    d = t.data
    if not d:
        return []
    if isinstance(d[0], list):
        return [(i, j) for i in range(len(d)) for j in range(len(d[0]))]  # type: ignore[arg-type]
    return [(i, None) for i in range(len(d))]


# **这两组访问器必须分开。** 曾经把更新量加到了 ``t.grad`` 而不是 ``t.data`` ——
# 参数一动不动、loss 稳在 ln(V) 上、**不报任何错**。梯度类 bug 的典型形态：
# 它不会让程序崩溃，只会让结论变成「训练没用」。所以下面刻意分成 data_* / grad_*
# 两组命名，并且有一条测试专门断言「训练后参数确实变了」。
def _data_get(t: Tensor, key: tuple[int, int | None]) -> float:
    i, j = key
    return t.data[i][j] if j is not None else t.data[i]


def _data_add(t: Tensor, key: tuple[int, int | None], v: float) -> None:
    i, j = key
    if j is not None:
        t.data[i][j] += v
    else:
        t.data[i] += v


def _grad_get(t: Tensor, key: tuple[int, int | None]) -> float:
    i, j = key
    return t.grad[i][j] if j is not None else t.grad[i]


def global_grad_norm(model: MoEModel) -> float:
    """全局 L2 范数。裁剪前先算它，而不是逐参数裁 ——
    逐参数裁会改变参数之间的相对步长，把负载均衡训练变成噪声。"""
    sq = 0.0
    for _, t in model.parameters():
        for key in _iter_elems(t):
            g = _grad_get(t, key)
            sq += g * g
    return math.sqrt(sq)


def sgd_momentum_step(model: MoEModel, tc: TrainConfig, state: dict[str, dict[Any, float]]) -> float:
    """SGD + 动量。返回本步的梯度范数（裁剪前），供观测。"""
    gn = global_grad_norm(model)
    factor = min(1.0, tc.grad_clip / (gn + 1e-12)) if tc.grad_clip > 0 else 1.0
    for name, t in model.parameters():
        buf = state.setdefault(name, {})
        for key in _iter_elems(t):
            g = _grad_get(t, key) * factor
            v = buf.get(key, 0.0) * tc.momentum - tc.lr * g
            buf[key] = v
            _data_add(t, key, v)
    return gn


# ----------------------------------------------------------------------
# 成本
# ----------------------------------------------------------------------
def macs_per_token(spec: MoESpec, vocab_size: int) -> int:
    """前向每个 token 的乘加次数。**稀疏激活**：专家部分只算 ``top_k`` 个。"""
    router = spec.d_model * spec.n_experts
    experts = spec.top_k * (spec.d_model * spec.d_ff + spec.d_ff * spec.d_model)
    head = spec.d_model * vocab_size
    return router + experts + head


def estimate_training_macs(spec: MoESpec, corpus: Corpus, tc: TrainConfig) -> int:
    """整条训练的前向 MACs（反向按 2× 计），用于「代价」轴的粗略可比。"""
    per_step = tc.batch_size * (corpus.seq_len - 1)
    fwd = per_step * macs_per_token(spec, corpus.vocab_size) * tc.steps
    return int(fwd * 3)


# ----------------------------------------------------------------------
# 运行
# ----------------------------------------------------------------------
@dataclass
class RunOutcome:
    """一个格子的全部观测。**流水线只把它折成 ``RunRecord``，不做任何筛选。**"""

    snapshots: list[dict[str, Any]]
    final: dict[str, float]
    onset_step: int | None
    domain_expert: list[list[int]]
    domain_dominant: list[int]
    params: int
    flops: int
    train_seconds: float
    grad_norm_last: float


def _eval_batch(corpus: Corpus) -> Batch:
    """固定的评测 batch：**每一步都看同一批 token**，曲线才是路由器的证据。"""
    return make_batch(corpus, corpus.test)


def evaluate(
    model: MoEModel,
    corpus: Corpus,
    batch: Batch,
    *,
    step: int,
    prev_top1: Sequence[int] | None,
) -> tuple[dict[str, Any], list[int]]:
    """在固定评测集上做一次观测。"""
    logits, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    assert trace is not None
    loss_t = softmax_cross_entropy(logits, batch.targets)
    loss = float(loss_t.data[0][0])
    correct = [
        max(range(len(row)), key=lambda j: row[j]) == batch.targets[i]
        for i, row in enumerate(logits.data)
    ]
    acc = sum(1 for c in correct if c) / len(correct)
    snap = observe(
        model,
        batch,
        trace,
        corpus,
        step=step,
        loss=loss,
        acc=acc,
        prev_top1=prev_top1,
        domain_correct=correct,
    )
    return snap, trace.top1


def train_proxy(
    model: MoEModel,
    corpus: Corpus,
    cfg: dict,
    seed: int,
    *,
    progress: Callable[[int, dict[str, Any]], None] | None = None,
) -> RunOutcome:
    """训练一个标本并全程采集路由观测。

    ``progress`` 用来在 ``run`` 时打印行；不传就静默（测试里用）。
    """
    tc = TrainConfig.from_config(cfg)
    spec = model.spec
    batch_eval = _eval_batch(corpus)
    snapshots: list[dict[str, Any]] = []
    state: dict[str, dict[Any, float]] = {}
    t0 = time.perf_counter()

    prev_top1: list[int] | None = None
    snap, prev_top1 = evaluate(model, corpus, batch_eval, step=0, prev_top1=None)
    snapshots.append(snap)
    if progress:
        progress(0, snap)

    per_epoch = max(1, math.ceil(len(corpus.train) / tc.batch_size))
    grad_norm = 0.0
    for step in range(1, tc.steps + 1):
        epoch, within = divmod(step - 1, per_epoch)
        batch = list(iter_batches(corpus, corpus.train, tc.batch_size, seed, epoch))[within]

        model.zero_grad()
        logits, trace, gates = model.forward(batch.inputs, batch.tags, collect=True)
        assert trace is not None
        total = softmax_cross_entropy(logits, batch.targets)
        if gates is not None and spec.aux_weight > 0.0:
            total = add(total, model.aux_loss(gates, trace.load))
        total.backward()
        grad_norm = sgd_momentum_step(model, tc, state)

        if step % tc.eval_every == 0 or step == tc.steps:
            snap, prev_top1 = evaluate(model, corpus, batch_eval, step=step, prev_top1=prev_top1)
            snapshots.append(snap)
            if progress:
                progress(step, snap)

    elapsed = time.perf_counter() - t0
    final = summarize(snapshots, corpus)
    final["grad_norm"] = grad_norm
    # **可计算的下界**：看不见域的路由器能达到的最优损失。低于它（留一点采样容差）
    # ⇒ 路由一定用上了域信息。这个判据是从任务构造里**推**出来的，不是拍的阈值。
    floor = corpus.domain_blind_floor()
    margin = float(cfg.get("expectations", {}).get("floor_margin", 0.1))
    final["domain_blind_floor"] = floor
    final["below_floor"] = 1.0 if final["loss"] < floor - margin else 0.0
    # 直接用最后一次观测的 top1 重建矩阵 —— 不再多跑一次前向（同源，也更快）
    assert prev_top1 is not None
    matrix = domain_expert_matrix(corpus, batch_eval.domains, prev_top1)
    series = [(s["step"], s["purity"]) for s in snapshots]
    return RunOutcome(
        snapshots=snapshots,
        final=final,
        onset_step=onset_step(series),
        domain_expert=matrix,
        domain_dominant=dominant_expert(matrix),
        params=model.num_params,
        flops=estimate_training_macs(spec, corpus, tc),
        train_seconds=elapsed,
        grad_norm_last=grad_norm,
    )


def _last_top1(model: MoEModel, batch: Batch) -> list[int]:
    """再前向一次拿最终路由（只读，不改参数）。仅测试与调试用。"""
    _, trace, _ = model.forward(batch.inputs, batch.tags, collect=True)
    assert trace is not None
    return trace.top1


def run_spec(
    spec: MoESpec,
    corpus: Corpus,
    cfg: dict,
    seed: int,
    *,
    progress: Callable[[int, dict[str, Any]], None] | None = None,
) -> RunOutcome:
    """从配置到观测：建标本 → 训练 → 汇总。"""
    tc = TrainConfig.from_config(cfg)
    model = build_model(spec, corpus.vocab_size, seed, tc.init_scale)
    return train_proxy(model, corpus, cfg, seed, progress=progress)
