"""冒烟测试：所有子模块可导入，且公开导出名真实存在。"""

from __future__ import annotations

import importlib
import pkgutil

import moefactory


def test_all_submodules_import() -> None:
    modules = [
        info.name
        for info in pkgutil.iter_modules(moefactory.__path__, moefactory.__name__ + ".")
    ]
    assert modules, "没发现任何子模块"
    for name in modules:
        importlib.import_module(name)


def test_public_names_are_exported() -> None:
    missing = [name for name in moefactory.__all__ if not hasattr(moefactory, name)]
    assert not missing, f"__all__ 里这些名字不存在：{missing}"
