# 更新日志

## 0.1.0 — 2026-11-02

首个版本，测量型（Microscope）MoE 标本：

- 微型 autograd 张量引擎（18 个算子，含梯度校验）。
- 合成多任务语料（域真值可计算下界）。
- Router → Top-K → 容量 → 稀疏派发 → Combine 全链路。
- 观测层：负载 / 分化 / 通信代价三组指标。
- 24 格配置网格 + 15 次多种子复核，7 条结论全部由数据算出。
- 只读 CLI：从产物重建，不重跑流水线、不写盘。
- Routing Sankey（ASCII / JSON）与自包含 HTML 报告。
- `python -m moefactory` 入口与 `--version` 开关。
