"""配置加载、校验、网格展开。

**网格展开是本项目的「扇出」节点**，所以它必须满足两条：

1. **每个组都是完整笛卡尔积**（``n == 各轴取值数之积``）—— 扇出不漏分支。
2. **展开确定且去重**（同配置必得同网格、同一标本只有一个 ``spec_id``）。

组的意义见 ``configs/default.json`` 的 ``_comment``：``learned`` 组是完整
笛卡尔积（aux × top_k × cap），``control`` 组是另一条完整笛卡尔积
（routing × top_k × cap），**control 组的 aux 是 N/A**。把不适用当 0 算进
均值，是这类实验最常见的静默偏差。
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from typing import Any

from .schema import ROUTINGS, MoESpec

#: 网格里允许出现的轴 → 它在 ``MoESpec`` 上的字段名（同名，但显式列出以防拼错）
AXIS_FIELDS: frozenset[str] = frozenset(
    {"n_experts", "top_k", "d_model", "d_ff", "routing", "aux_weight", "capacity_factor"}
)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_root"] = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """把配置错误挡在跑之前。**越早报错越便宜。**"""
    if "seed" not in cfg:
        raise ValueError("配置缺少 seed")
    t = cfg.get("task")
    if not isinstance(t, dict):
        raise ValueError("配置缺少 task")
    for key in ("n_domains", "n_content", "seq_len", "n_train", "n_test"):
        if key not in t:
            raise ValueError(f"task 缺少 {key}")
    if int(t["n_content"]) < 2:
        raise ValueError("n_content 至少为 2（否则造不出错排）")
    if int(t["seq_len"]) < 2:
        raise ValueError("seq_len 至少为 2（否则没有「下一个 token」）")

    p = cfg.get("proxy")
    if not isinstance(p, dict):
        raise ValueError("配置缺少 proxy")
    for key in ("steps", "lr", "batch_size", "momentum", "init_scale", "grad_clip", "eval_every"):
        if key not in p:
            raise ValueError(f"proxy 缺少 {key}")
    if int(p["steps"]) < 1 or int(p["batch_size"]) < 1 or int(p["eval_every"]) < 1:
        raise ValueError("proxy 的 steps / batch_size / eval_every 必须为正")

    g = cfg.get("grid")
    if not isinstance(g, dict) or "groups" not in g or not g["groups"]:
        raise ValueError("配置缺少 grid.groups")
    base = g.get("base", {})
    for key in base:
        if key not in AXIS_FIELDS:
            raise ValueError(f"grid.base 里有未知字段 {key}")
    for grp in g["groups"]:
        if "name" not in grp:
            raise ValueError("grid 组缺少 name")
        axes = grp.get("axes", {})
        if not axes:
            raise ValueError(f"网格组 {grp['name']} 没有轴")
        for axis, values in axes.items():
            if axis not in AXIS_FIELDS:
                raise ValueError(f"网格组 {grp['name']} 有未知轴 {axis}")
            if not isinstance(values, list) or not values:
                raise ValueError(f"轴 {axis} 的取值必须是非空列表")
        for key in grp.get("fixed", {}):
            if key not in AXIS_FIELDS:
                raise ValueError(f"网格组 {grp['name']} 的 fixed 里有未知字段 {key}")
        for key, val in grp.get("fixed", {}).items():
            if key == "routing" and val not in ROUTINGS:
                raise ValueError(f"未知 routing：{val}（可选 {ROUTINGS}）")


def spec_from_cfg(cfg: dict) -> MoESpec:
    """网格的「基准标本」（不含轴）。报告里当参考配置用。"""
    g = cfg["grid"]
    base = dict(g.get("base", {}))
    defaults = {"n_experts": 4, "top_k": 1, "d_model": 12, "d_ff": 16, "routing": "learned",
                "aux_weight": 0.05, "capacity_factor": 1.0}
    defaults.update(base)
    return MoESpec(**defaults).canonical()


def expand_grid(cfg: dict) -> list[MoESpec]:
    """把配置展开成标本列表。**确定、去重、每组都是完整笛卡尔积。**"""
    g = cfg["grid"]
    base = dict(g.get("base", {}))
    out: list[MoESpec] = []
    seen: set[str] = set()
    for grp in g["groups"]:
        fields = {**base, **grp.get("fixed", {})}
        axes = grp["axes"]
        names = list(axes.keys())
        for combo in itertools.product(*(axes[n] for n in names)):
            kw = dict(fields)
            kw.update(dict(zip(names, combo)))
            spec = MoESpec(**kw).canonical()
            if spec.spec_id in seen:
                continue
            seen.add(spec.spec_id)
            out.append(spec)
    return out


def grid_size_of_group(cfg: dict, name: str) -> int:
    for grp in cfg["grid"]["groups"]:
        if grp["name"] == name:
            n = 1
            for values in grp["axes"].values():
                n *= len(values)
            return n
    raise KeyError(f"没有网格组 {name}")


def grid_report(cfg: dict) -> dict[str, Any]:
    """扇出完整性的自述（报告与 CLI 都读它）。"""
    specs = expand_grid(cfg)
    groups: list[dict[str, Any]] = []
    for grp in cfg["grid"]["groups"]:
        declared = grid_size_of_group(cfg, grp["name"])
        groups.append(
            {
                "name": grp["name"],
                "declared": declared,
                "axes": {k: len(v) for k, v in grp["axes"].items()},
                "fixed": grp.get("fixed", {}),
            }
        )
    return {
        "n_specs": len(specs),
        "groups": groups,
        "axes": sorted({k for grp in cfg["grid"]["groups"] for k in grp["axes"]}),
    }


def objectives_of(cfg: dict) -> tuple[str, ...]:
    """本项目**没有优化目标** —— 它是测量型。

    保留这个函数是为了让只读命令与报告有一个统一的「本项目的产物是什么」的答案：
    返回空元组，任何试图「按指标排序挑最优」的代码都会立刻发现无事可做。
    """
    return ()


def expectations_of(cfg: dict) -> dict[str, float]:
    return {k: v for k, v in cfg.get("expectations", {}).items() if not k.startswith("_")}


def floor_margin(cfg: dict) -> float:
    """贴边比较的容差。``domain_blind_floor`` 是连续量，实测 loss 来自有限评测集。"""
    return float(cfg.get("expectations", {}).get("floor_margin", 0.1))


def expand_replication(cfg: dict) -> list[tuple[MoESpec, int]]:
    """多种子复核：``(spec, seed)`` 列表。

    **为什么要单独一块。** 网格里的每个格子只跑一个种子，于是「learned 突破了
    下界」可能只是运气。复核块把几个角落配置各跑多个种子，报出「分化成功次数」：
    对照组 ``0/N`` 与 learned ``N/N`` 的对比，才是结论强度的来源。
    """
    rep = cfg.get("replication") or {}
    seeds = [int(s) for s in rep.get("seeds", [])]
    specs = [MoESpec.from_spec_id(s) for s in rep.get("specs", [])]
    if not seeds or not specs:
        return []
    return [(sp, sd) for sp in specs for sd in seeds]


def version_digest(cfg: dict, corpus_digest: str, spec_ids: list[str]) -> str:
    """内容寻址版本：**只由「观测对象」决定**。

    语料变了、网格变了 → 版本变；训练预算变了 → 协议指纹变（``protocol_tag``）；
    输出目录变了 → 什么都不变。
    """
    payload = {
        "corpus": corpus_digest,
        "specs": sorted(spec_ids),
        "task": cfg["task"].get("name", ""),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_path(cfg: dict, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(cfg.get("_root", os.getcwd()), path)
