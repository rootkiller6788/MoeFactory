# data/ —— 外部输入

本项目**不读外部输入**：MoEFactory 的世界 / 种群 / 证据都由 `configs/default.json` 的口径
在运行时自造，`src/moefactory/` 里没有任何地方引用 `data/`。
这个目录保留，是为了与工作区其余项目保持一致的目录结构。

如果你要接入真实输入：把 JSON 规格放进这个目录，并在 `configs/default.json` 里指向它。

## 接入约定（尚未实现读取分支）

- 编码 UTF-8，格式 JSON；顶层可以是数组，也可以是一行一个样本。
- 每条样本至少包含 `domain`（域真值）与 `tokens`（整数 token 序列）。
- token 必须落在当前词表口径内（内容词 + 域标签），否则 `build_corpus` 推导出的
  `vocab_size` 会与模型不一致。
- 当前版本**只**使用 `corpus.py` 的确定性合成器；`data/` 不参与运行时。
