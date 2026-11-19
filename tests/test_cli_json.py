"""``grid --json`` 的输出必须能被脚本直接解析。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from moefactory import cli


class FakeResult:
    grid = {
        "n_specs": 2,
        "axes": ["top_k"],
        "groups": [
            {"name": "g", "declared": 2, "axes": {"top_k": 2}, "fixed": {}},
        ],
    }


def test_grid_json_is_parseable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_result", lambda args: (FakeResult(), {}, True))
    rc = cli.cmd_grid(SimpleNamespace(json=True))
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["n_specs"] == 2
    assert payload["groups"][0]["declared"] == 2
