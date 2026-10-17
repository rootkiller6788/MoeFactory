"""测试夹具。

**一条纪律：贵的流水线只跑一次。** 本项目整条流水线（24 格 + 15 次复核）约 100 秒，
所以 ``fast_out`` / ``fast_result`` 都是 **session 级**：整套测试共享一次运行，
后续所有断言都从产物上读。

``fast_cfg`` 削的是 ``格子数 × 步数 × 批大小`` 这个**乘积**。这条教训来自同工作区
的 L2-architecturefactory：那边最初只削了样本数，把 ``steps`` 原样留着，结果
156 条测试跑了 301 秒 —— 削一个因子等于没削（见 skill ``python-pipeline-scaffold``
的「测试预算的削法」）。

**「下界可突破」这条命题不用 ``fast_cfg`` 证明。** 20 步的路由器还没学会任何东西
（实测 loss ≈ 2.37，下界 1.13），用它去断言「learned 突破下界」必然是假。
命题本身由 ``test_acceptance.py`` 用**真实协议**（120 步）单独跑一小撮配置来验证 ——
那是本项目的验收标准，不能被一个削过的配置糊过去。
"""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session", autouse=True)
def _chdir_root():
    """所有相对路径（configs/、out/）都以项目根为基准。"""
    old = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def cfg() -> dict:
    from moefactory.config import load_config

    return load_config(os.path.join(ROOT, "configs", "default.json"))


@pytest.fixture(scope="session")
def corpus(cfg: dict):
    """默认任务语料。造语料很便宜（纯标准库），所以不共享也行，但没必要重复造。"""
    from moefactory.corpus import build_corpus

    return build_corpus(cfg)


@pytest.fixture(scope="session")
def fast_cfg(cfg: dict) -> dict:
    """削过的配置：**只验证结构与契约，不验证质量**。

    四个因子一起削（格子数 × 步数 × 批大小 × 样本数）：
    网格 24→6 格、复核 15→4 次、步数 180→20、batch 10→6、序列 10→6。
    于是「跑一次流水线」从 ~100 秒降到 ~2 秒。
    """
    c = copy.deepcopy(cfg)
    c["task"]["n_train"] = 24
    c["task"]["n_test"] = 8
    c["task"]["seq_len"] = 6
    c["proxy"]["steps"] = 20
    c["proxy"]["batch_size"] = 6
    c["proxy"]["eval_every"] = 10
    # 两组都保留完整的笛卡尔积 —— 削的是每轴的取值数，不是「完整性」本身，
    # 否则 test_pipeline_cli 里那条「格数 == 各轴取值数之积」就变成同义反复。
    c["grid"]["groups"][0]["axes"] = {
        "aux_weight": [0.0, 0.1],
        "top_k": [1, 2],
        "capacity_factor": [1.0],
    }
    c["grid"]["groups"][1]["axes"] = {
        "routing": ["static", "random"],
        "top_k": [1],
        "capacity_factor": [1.0],
    }
    c["replication"]["specs"] = [
        "e4-k1-d12-f16-learned-aux0.1-cap1",
        "e4-k1-d12-f16-static-aux0-cap1",
    ]
    c["replication"]["seeds"] = [1, 2]
    return c


@pytest.fixture(scope="session")
def fast_corpus(fast_cfg: dict):
    from moefactory.corpus import build_corpus

    return build_corpus(fast_cfg)


@pytest.fixture(scope="session")
def fast_cfg_path(tmp_path_factory, fast_cfg: dict) -> str:
    """``--config`` 要一个路径，而 fast_cfg 是内存里的 dict。"""
    p = tmp_path_factory.mktemp("moefast") / "fast.json"
    p.write_text(json.dumps(fast_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


@pytest.fixture(scope="session")
def tiny_cfg(fast_cfg: dict) -> dict:
    """最小配置：2 格、0 复核、3 步。

    只用于「产物不存在时只读命令也不写盘」这类**需要反复重跑流水线**的契约测试 ——
    那边有近 20 条命令，用 ``fast_cfg``（每次约 2 秒）会把测试拖到 36 秒。
    这条正是 L2-architecturefactory 的教训：**契约测试的产物要共享或削到最小**。
    """
    c = copy.deepcopy(fast_cfg)
    c["proxy"]["steps"] = 3
    c["proxy"]["eval_every"] = 3
    c["grid"]["groups"][0]["axes"] = {"aux_weight": [0.1], "top_k": [1], "capacity_factor": [1.0]}
    c["grid"]["groups"][1]["axes"] = {"routing": ["static"], "top_k": [1], "capacity_factor": [1.0]}
    # 复核块留一格：否则 ``replication`` 命令会提前返回，测不到它的完整路径
    c["replication"]["specs"] = ["e4-k1-d12-f16-learned-aux0.1-cap1"]
    c["replication"]["seeds"] = [1]
    return c


@pytest.fixture(scope="session")
def tiny_cfg_path(tmp_path_factory, tiny_cfg: dict) -> str:
    p = tmp_path_factory.mktemp("moetiny") / "tiny.json"
    p.write_text(json.dumps(tiny_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


@pytest.fixture(scope="session")
def fast_out(tmp_path_factory, fast_cfg: dict) -> str:
    """把削过的流水线跑一次并落盘 —— 整套测试唯一的写盘动作（除了 CLI 契约测试）。"""
    from moefactory.pipeline import run_pipeline, write_outputs

    out = str(tmp_path_factory.mktemp("moefast-out"))
    write_outputs(run_pipeline(fast_cfg), out)
    return out


@pytest.fixture(scope="session")
def fast_result(fast_out: str):
    """**从产物重建**，而不是把内存对象留着 —— 顺手把只读重建路径也测了。"""
    from moefactory.pipeline import load_result

    r = load_result(fast_out)
    assert r is not None, "fast_out 的产物不完整，只读重建失败"
    return r


@pytest.fixture
def cli(capsys):
    """调用 CLI 并拿到 ``(退出码, stdout)``。"""
    from moefactory.cli import main

    def run(argv: list[str]):
        code = main(argv)
        return code, capsys.readouterr().out

    return run


@pytest.fixture(scope="session")
def acceptance_runs(cfg: dict, corpus):
    """验收命题用**真实协议**（120 步）跑的一小撮配置。**不能被削。**

    这里刻意只跑 6 个配置（约 7 秒），因为验收断言是「分离是否存在」，
    不是「整张网格长什么样」—— 后者由 ``out/`` 里那份完整产物回答。
    """
    from moefactory.schema import MoESpec
    from moefactory.train import run_spec

    c = copy.deepcopy(cfg)
    c["proxy"]["steps"] = 120
    c["proxy"]["eval_every"] = 30
    cases = [
        ("e4-k2-d12-f16-learned-aux0.1-cap2", 1),
        ("e4-k1-d12-f16-learned-aux0.1-cap2", 1),
        ("e4-k1-d12-f16-learned-aux0.5-cap2", 1),
        ("e4-k1-d12-f16-learned-aux0-cap2", 1),
        ("e4-k1-d12-f16-static-aux0-cap2", 1),
        ("e4-k1-d12-f16-random-aux0-cap2", 1),
    ]
    return {
        spec_id: run_spec(MoESpec.from_spec_id(spec_id), corpus, c, seed)
        for spec_id, seed in cases
    }
