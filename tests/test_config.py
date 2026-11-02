"""配置层的两条硬约束：完整笛卡尔积、展开确定且去重。"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from moefactory.config import expand_grid, grid_report, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def cfg() -> dict:
    return load_config(str(ROOT / "configs" / "default.json"))


def test_every_group_is_a_cartesian_product(cfg: dict) -> None:
    report = grid_report(cfg)
    assert report["n_specs"] == len(expand_grid(cfg))
    for group in report["groups"]:
        product = 1
        for n_values in group["axes"].values():
            product *= n_values
        assert group["declared"] == product, f"{group['name']} 不是完整笛卡尔积"


def test_expansion_is_deterministic(cfg: dict) -> None:
    first = [spec.spec_id for spec in expand_grid(cfg)]
    second = [spec.spec_id for spec in expand_grid(cfg)]
    assert first == second


def test_duplicate_groups_are_deduped(cfg: dict) -> None:
    doubled = copy.deepcopy(cfg)
    doubled["grid"]["groups"].append(copy.deepcopy(doubled["grid"]["groups"][0]))
    ids = [spec.spec_id for spec in expand_grid(doubled)]
    assert len(ids) == len(set(ids))
    assert len(ids) == len(expand_grid(cfg))


def test_unknown_axis_is_rejected(cfg: dict) -> None:
    bad = copy.deepcopy(cfg)
    bad["grid"]["groups"][0]["axes"]["bogus"] = [1]
    with pytest.raises(ValueError, match="未知轴"):
        validate_config(bad)


def test_missing_seed_is_rejected(cfg: dict) -> None:
    bad = copy.deepcopy(cfg)
    bad.pop("seed")
    with pytest.raises(ValueError, match="seed"):
        validate_config(bad)
