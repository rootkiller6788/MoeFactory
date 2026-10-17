"""版本号只有一个来源：``__init__.__version__`` 必须与 ``pyproject.toml`` 一致。"""

from __future__ import annotations

import tomllib
from pathlib import Path

import moefactory

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert moefactory.__version__ == data["project"]["version"]
