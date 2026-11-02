# 贡献指南

## 环境

- CPython >= 3.12（下限由 `math.sumprod` 决定）。
- 核心链路**零第三方依赖**；只有测试需要 `pytest`：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## 三条硬纪律

1. **只读命令不写盘。** 读命令从 `out/` 的产物重建结果；产物缺失时在内存里重跑，
   仍然不落盘。写命令只有 `run` 与 `visualize`。
2. **结论由数据算出。** `findings.py` 里不要写死「辅助损失带来分化」这类文案；
   一旦写死，报告就从测量结果退化成宣传材料。
3. **阈值集中放 `configs/default.json`。** 代码里不硬编码阈值；新增网格轴时同步更新
   `config.AXIS_FIELDS`、配置校验与测试。

## 提交前

```bash
python -m pytest          # 不要再加 -q：pyproject.toml 已经带了
python main.py --version
```

提交信息用「类型: 说明」的形式（`feat` / `fix` / `test` / `docs` / `chore` / `ci`）。
