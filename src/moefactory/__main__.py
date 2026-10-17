"""``python -m moefactory`` 入口，等价于 ``python main.py``。

``main.py`` 需要仓库在 ``sys.path`` 上；以模块方式运行时包已经可导入，
所以这里只转交 CLI，不再重复处理路径。
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
