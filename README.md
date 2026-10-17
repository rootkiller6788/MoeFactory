# MoEFactory

> 来源：factory-everything/factory-everything-项目总表.md ｜ 板块：factory-everything · L2 ｜ 落地约束：4060 8G 单机可跑

## 一、条目信息（原始字段，完整保留）

| 字段 | 内容 |
| --- | --- |
| 项目 | MoEFactory |
| 类型 | Factory |
| 真正有意思的核心 | Expert Routing、负载均衡、可视化 |

---

## 二、核心命题

**这个项目实际上是把 ArchitectureFactory 的一个节点纵向钻穿。**

核心问题不是"实现 MoE"，而是：

> Token → Expert 的动态路由系统究竟发生了什么？

## 三、整体架构

```
                 Tokens
                   │
                   ▼
                 Router
                   │
              Router Logits
                   │
                   ▼
                 Top-K
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
       E₁         E₂        ... Eₙ
         │         │          │
         └─────────┼──────────┘
                   ▼
                Combine
                   │
                   ▼
                 Output

          ─── Observation ───

Router
  │
  ├── Expert Load
  ├── Token Distribution
  ├── Routing Entropy
  ├── Expert Similarity
  ├── Dropped Tokens
  └── Communication Cost
```

## 四、闭环

```
Token
 ↓
Route
 ↓
Dispatch
 ↓
Expert Compute
 ↓
Combine
 ↓
Measure Balance
 ↓
Auxiliary Loss
 ↓
Update Router
 ↺
```

## 五、核心指标

```
Load_i   = #Tokens routed to Expert_i
Entropy  = -Σ_i p_i log p_i
```

以及：

- Expert specialization（专家是否分化出不同职能）
- Routing stability（同一 token 在不同 step 的路由是否稳定）
- Load imbalance（负载倾斜程度）
- Capacity overflow（超出容量的 token 被丢弃比例）
- Communication cost（All-to-All 通信开销）

## 六、最终 UI 最值得做的不是 loss curve，而是 Routing Sankey

```
Tokens                    Experts

Python ────────────────→ E1
Rust   ────────────────→ E1
Math   ────────→ E3
French ─────────────────→ E5
SQL    ────────→ E2

        Routing Sankey
```

**一眼看出 Expert 有没有形成 specialization。** 如果所有 token 都涌向同一个专家，Sankey 会立刻暴露这个塌缩——这是 loss 曲线看不出来的问题。

## 七、阶段性落地计划

- **M0｜路由骨架**：实现 Router + Top-K + Dispatch + Combine 前向，跑通一次（非自动）。
- **M1｜观测层**：实现 Expert Load / Entropy / Dropped Tokens / Communication Cost 全套指标采集。
- **M2｜均衡与闭环**：接入 Auxiliary Loss 与容量因子，做负载均衡的参数搜索。
- **M3｜Routing Sankey 与发布**：实现路由可视化面板（Sankey + 热力图），输出 specialization 分析报告。

## 八、验收标准

能端到端完成「MoEFactory」，并在其「核心」（Expert Routing、负载均衡、可视化）上给出可复现的量化结果：

- 给出 Load imbalance / Entropy / Dropped Tokens 的量化曲线
- Routing Sankey 能明确回答"专家是否形成了 specialization"

---

_本 README 的「条目信息」字段来自 `factory-everything-项目总表.md`，架构与流水线为 L2 深化设计稿；可按实际进展修订。_

---

# 附录 A｜骨架、快速开始、CLI 契约、产物

## A.1 目录

```
L2-moefactory/
├── configs/default.json      所有阈值与网格都在这里（代码里不硬编码任何阈值）
├── src/moefactory/
│   ├── rng.py               确定性随机（sha256 派生种子，同配置必得同随机流）
│   ├── autograd.py          微型张量引擎：18 个算子，每个都有梯度校验
│   ├── schema.py            MoESpec / RoutingSnapshot / RunRecord / Finding / Version
│   ├── corpus.py            带域真值的多任务 next-token 语料 + **可计算下界**
│   ├── moe.py               标本本体：路由 → top-k → 容量 → 稀疏派发 → 合并
│   ├── observe.py           观测层：负载 / 分化 / 代价 三组指标（不要混着看）
│   ├── train.py             代理训练（一步一 batch、全局范数裁剪、固定评测集观测）
│   ├── findings.py          把观测折成**可执行结论**（7 条，判定条件全部由数据算）
│   ├── config.py            网格展开（每组完整笛卡尔积）+ 版本/协议指纹
│   ├── lineage.py           血缘 DAG（收拢点必须恰好等于全部观测）
│   ├── pipeline.py          编排 + 落盘 + **只读重建**（load_result）
│   ├── report.py            markdown 报告（含 ASCII Sankey 与散点）
│   ├── visualize.py         自包含 HTML（内联 SVG，零外部引用）
│   └── cli.py               14 个子命令（12 只读 + 2 写）
├── tests/                   243 条测试，约 18 秒
├── main.py                  入口（等价于 python -m moefactory.cli）
└── out/                     产物（见 A.4）
```

**形状**：测量型（Microscope）。被观测对象是一个训练中的 MoE 标本；扇出的是
**观测/配置网格**；产物是**路由证据**。这里没有 `Trial`、没有 `Pareto`、没有「最优解」——
`objectives_of()` 返回空元组，任何「按指标排序挑最优」的代码都会立刻发现无事可做。

## A.2 快速开始

```bash
# 1. 先看任务本身：不需要任何产物
python main.py task
python main.py explain e4-k2-d12-f16-learned-aux0.1-cap2

# 2. 跑完整条流水线（24 格 + 15 次复核，本机约 100 秒）
python main.py run

# 3. 看结论（只读，从产物重建，不会重跑）
python main.py summary
python main.py findings
python main.py compare
python main.py sankey e4-k2-d12-f16-learned-aux0.1-cap2
python main.py matrix --by loss

# 4. 自包含 HTML（唯一额外的写命令）
python main.py visualize        # → out/report.html

# 5. 测试
python -m pytest                # 243 passed，约 18 秒
                                # pyproject.toml 里已经默认 -q，不要再加一个
```

## A.3 CLI 契约

| 类别 | 命令 | 作用 |
| --- | --- | --- |
| 只读 | `summary` | 版本 / 计数 / 下界 / 结论概览 |
| 只读 | `grid` | 扇出网格与完整性（每组格数 == 各轴取值数之积） |
| 只读 | `findings` | 逐条打印结论与证据 |
| 只读 | `matrix` | 对比矩阵（默认按**均衡**排序 —— 刻意暴露那个陷阱） |
| 只读 | `runs` | 全部观测一行一条 |
| 只读 | `show <spec_id> [--seed]` | 一个格子的全部观测 |
| 只读 | `curves <spec_id> [--seed]` | 逐 step 曲线 |
| 只读 | `sankey <spec_id> [--seed]` | Routing Sankey（ASCII） |
| 只读 | `replication` | 多种子复核：成功次数表 |
| 只读 | `lineage [--dot]` | 血缘图（ASCII 或裸 DOT） |
| 只读 | `task` | 任务定义与可计算下界（**不需要产物**） |
| 只读 | `compare` | 三种路由方式对比 |
| 只读 | `explain <spec_id>` | 从 `spec_id` 展开装配说明（**不需要产物**） |
| 写 | `run` | 跑流水线并落盘 |
| 写 | `visualize` | 写 `out/report.html` |

**只读契约（三条，都由测试钉住）：**

1. 只读命令**从产物重建结果对象**（`load_result`），不重跑流水线 —— 整条流水线 100 秒，
   若每条读命令都重算，用户会以为工具坏了。
2. 产物缺失时，只读命令在内存里重跑，但**仍然不落盘**，并打印「未写盘」。
3. `run` 与 `visualize` 是仅有的两个写命令。

第 1、2 条由 `tests/test_pipeline_cli.py` 用 **md5 指纹**验证：19 条只读命令跑完，
`out/` 里每个文件一个字节都不许变。`--quiet` 是**顶层**开关，必须写在子命令之前。

## A.4 产物清单

| 文件 | 内容 |
| --- | --- |
| `runs.json` | 全部观测（39 条 RunRecord，含逐 step 快照与域×专家矩阵） |
| `matrix.csv` | 对比矩阵（24 列，一行一个格子） |
| `curves.json` | 每个格子的逐 step 曲线（loss / acc / H / imb / nmi / purity / drop / stability） |
| `sankey.json` | 每个格子的「域 → 专家」流量矩阵 |
| `findings.json` | 7 条结论：主张 + 证据 + 是否被支持 |
| `replication.json` | 多种子复核的成功次数表 |
| `lineage.json` | 血缘 DAG（105 节点 / 142 边） |
| `lineage.dot` | 同一张图的 graphviz 版本 |
| `version.json` | 内容寻址版本（`v1-ebdecf5d39f4`） |
| `report.md` | markdown 报告（八节，顺序即论点顺序） |
| `report.html` | 自包含 HTML（由 `visualize` 生成） |

**`load_result` 缺任何一个必需文件都返回 `None`** —— 不能「读一半」。半个结果对象
比没有结果更危险。

---

# 附录 B｜环境、标定、设计期望 vs 实测、结论

## B.1 环境与实测开销

- **零第三方依赖**，只用标准库；`requires-python >= 3.12`（下限由 `math.sumprod` 决定）。
  专家前向占总 MACs 的约 70%，内积走 C 实现是「24 个配置能在一分半内跑完」的前提。
- 本机实测：CPython 3.13.14 / pytest 9.1.1。
- 整条流水线：**100.0 秒**（24 格 + 15 次复核 = 39 次训练，每次 180 步）。
- 测试套件：**243 条，17.6 秒**。

## B.2 标定：为什么是 180 步

所有标本共享同一份训练预算（可比性的前提），所以这个数只能标定一次。取
`e4-k2-d12-f16-learned-aux0.1-cap2` 与 `e4-k1-d12-f16-static-aux0-cap2`（seed 1），
只看「是否突破可计算下界」（下界 1.1264，容差 0.1 ⇒ 阈值 1.0264）：

| steps | learned loss | learned 突破? | static loss | static 突破? |
| --- | --- | --- | --- | --- |
| 20 | 2.377 | 否 | 2.263 | 否 |
| 40 | 1.396 | 否 | 1.438 | 否 |
| 60 | 1.137 | 否 | 1.231 | 否 |
| 80 | 0.776 | **是** | 1.097 | 否 |
| 120 | 0.360 | 是 | 1.106 | 否 |
| 180（出货协议） | 0.075 | 是 | 1.200 | 否 |

读法：**分离在 80 步左右出现，180 步时留出约 1.05 的余量**（阈值 1.0264 对
learned 的最好成绩 0.0753）。120 步虽然也成立，但分离度只有 180 步的约 1/5；
240 步以上收益递减而耗时线性增长。对照组在任何步数下都停在下界以上 —— 这正是判据
「能被推翻」的证明。

## B.3 设计期望 vs 实测

| # | 设计期望 | 实测 | 判定 |
| --- | --- | --- | --- |
| 1 | 辅助损失是专家分化的**必要条件** | `aux=0` 的 4 个格子**全部**突破下界；四个 aux 取值的平均 loss 落在 17.8% 的窄带内 | **被推翻** |
| 2 | 辅助损失买到**负载均衡** | Spearman(aux, H) 仅 **0.303**，逐轴看有反向的格子 | 不成立 |
| 3 | 越均衡的配置越好 | 全网格 Spearman(H, loss) = **+0.552**（越均衡反而越差）；按均衡排序第一名是 `static`（H=0.9999，loss=1.1709，全场最差之一） | **被推翻** |
| 4 | 「域 → 主导专家」应是一一对应 | `top_k=2` 时主导专家是 `[2,0,2,1]`（有重复），但**主要专家集合**两两不同 | 判据本身要改（坑 58） |
| 5 | 专家的信息只能来自「路由器把它派给了谁」 | 对照组完全不看标签 ⇒ 同一批输入换标签，输出**逐位不变** | 成立 |
| 6 | 同配置必得同读数 | 两次运行的快照、矩阵、onset **逐位相同**（唯一例外是 `train_seconds`） | 成立 |
| 7 | 容量越紧越差 | 容量因子 1.0 时丢弃 0~20%，learned 仍 100% 突破下界；丢弃的 token 保留残差通路 | 成立 |

第 1~3 条是最有价值的：它们都是**被实测推翻**的设计假设。能推翻自己，靠的是
「结论全部由数据算出，不写死字符串」这条纪律 —— 一旦把「辅助损失带来分化」写成
常量文案，报告就变成宣传材料了。

## B.4 结论（7 条，全部被实测支持）

| key | 主张 | 关键证据 |
| --- | --- | --- |
| `grid_complete` | 扇出网格完整，没有漏分支 | learned 16/16，control 8/8，合计 24 格 + 15 次复核 |
| `control_pinned_at_floor` | 对照组**全部**停在下界以上 | 14/14；loss 1.135~1.200；nmi 最大 0.035 |
| `learned_breaks_floor` | 学出来的路由能突破可计算下界 | 25/25（要求 ≥60%）；最好 0.0753 |
| `balance_points_the_wrong_way` | **均衡指标指向的方向是错的** | ρ(均衡, loss)=+0.552（learned 组内 +0.185）；ρ(分化, loss)=**−0.768** |
| `aux_axis_is_flat` | 辅助损失不是分化的必要条件 | `aux=0` 4/4 突破；aux 轴展开 17.8% vs learned/control 差距 **4.99×** |
| `truth_recovered` | 每个域由不同的专家组合承担 | 最好格子的主要集合 `[[2,3],[0],[0,2,3],[1,2]]` 两两不同；对照组全部相同 |
| `replication_separates` | 多种子复核下分离依然成立 | learned **9/9** vs 对照组 **0/6**（种子 1/2/3） |

## B.5 代表读数与成本

| | `e4-k2-...-learned-aux0.1-cap2`（最好） | `e4-k1-...-static-aux0-cap2`（对照） |
| --- | --- | --- |
| loss / acc | **0.0753** / 0.9444 | 1.2000 / 0.3403 |
| 可计算下界 | 1.1264 | 1.1264 |
| 突破下界 | 是 | 否 |
| H（均衡） | 0.9924 | **0.9973** |
| imbalance | 1.222 | **1.083** |
| nmi（分化） | **0.4602** | 0.0115 |
| purity | 0.6389 | 0.3125 |
| 专家相似度 | −0.0114 | 0.0054 |
| 分化起始步 | 90 | 无（平曲线） |
| 参数 / 稀疏激活 | 2112 / **1288**（61%） | 2112 / **876**（41%） |
| 通信量 | 6912 元素 | 3456 元素 |

**这两列就是整个项目的那句话**：对照组在「均衡」这一列**更漂亮**，在「loss」那一列
**差 16 倍**。把两者揉成一个「路由质量」分数，这个陷阱就看不见了。

---

# 附录 C｜坑（58–74）

编号承接同工作区前序项目（1–57 见 L2-architecturefactory 的附录）。

### 58. `top_k=2` 时用「主导专家」判真值会误判

**症状**：真值复原这条结论报**不支持**，而模型明明已经突破下界、acc 0.94。
最好的格子里「域 → 主导专家」是 `[2,0,2,1]`，两个域的主导专家相同。

**原因**：`top_k=2` 时一个域由**一对**专家共同承担。两个域可以各用一对不同的专家，
却让 argmax 落在同一个专家上 —— 用「主导专家一一对应」判真值，会把正确的结果判成失败。

**修法**：判据改成**主要专家集合**（承接该域比例 ≥ 20% 的专家），对 `top_k=1` 与
`top_k=2` 都成立。实测最好格子的集合是 `[[2,3],[0],[0,2,3],[1,2]]`（两两不同），
对照组是 `[[0,1,2],[0,1,2],[0,1,2],[0]]`（大量重复）。`is_bijection` 保留但只在
`top_k==1` 时标注为可用。

### 59. `param_count` 多报了一个 RMS 缩放

**症状**：`explain` 报的参数量比模型实际多出 `d_model`（12）。**没有任何地方会报错。**

**原因**：`MoESpec.param_count` 里写了 `norms = 2 * d_model`（以为有两个 RMS 缩放），
而模型只有 `norm_w` 一个。

**修法**：改成 `norms = d_model`，并加一条测试断言
`spec.param_count(vocab) == model.num_params` 逐位相等。这个数是 `explain` 与
「稀疏激活比例」的口径来源，偏一点就会让「MoE 省了多少算力」这个说法失真。

### 60. 优化器把更新量写进了 `.grad`

**症状**：loss 稳在 `ln(16) = 2.7726` 上一动不动，acc 约 0.01，**不报任何错**。

**原因**：更新步把 `v` 加到了 `t.grad` 而不是 `t.data` —— 参数从未改变，而梯度每步
被覆盖，于是「训练」变成了空转。

**修法**：把访问器拆成 `_data_get` / `_data_add` / `_grad_get` 三组命名，并在
`tests/test_autograd.py` 里加一条**专门的回归测试**：「一步更新后参数确实变了，
且优化器没有改写 `grad`」。梯度类 bug 的典型形态就是它不崩溃，只让结论变成
「训练没用」—— 而「训练没用」在这个项目里是一个**合法结论**，两者输出上一模一样。

### 61. 任务构造让对照组反而更强（设计缺陷，不是代码 bug）

**症状**：标定第一版里 `static` 路由 loss 0.39，learned 路由 1.44 —— 对照把被测对象
打得一败涂地，整个命题不成立。

**原因**：最初给每个域分配了**不相交的内容词表**。于是「同一个 token 只出现在一个域里」，
任何稳定的划分（包括 `token % 4`）都足以学会任务 —— **专家分化不是必需的，只是装饰**。

**修法**：重新设计任务，三条一起改：内容词表在域之间**共享**；路由器看
`emb[token] + emb[tag]`、专家**只看** `emb[token]`；规则用随机**错排**而不是循环移位。
改完之后 `static`/`random` 被钉在下界上（1.09~1.20），learned 到 0.075。
**这是本项目最重要的一次修正**：它说明「让分化成为必需」必须由任务构造保证，
不能指望损失函数自己长出来。

### 62. 数值梯度校验脚本在循环里清零了梯度

**症状**：15 个算子里报出 11 个「解析梯度与数值梯度不符」（例如 `matmul` 数值 0.8
对解析 0.0），看起来引擎烂透了。

**原因**：校验脚本在扰动循环内部对共享张量调了 `zero_grad()`，把已经算好的解析梯度
中途清掉了。

**修法**：`backward()` 之后**先快照**解析梯度，再去做中心差分。修好之后 18 个算子
全部 OK。教训：**校验器本身也会错，而且错起来会集体误报**——先怀疑校验器。

### 63. 「网格完整性」把复核行也算进网格

**症状**：`grid_complete` 报 `learned = 25/16, control = 14/8` —— 24 格的网格被数成 39 格。

**原因**：多种子复核的观测与网格观测混在一个 `runs` 列表里，按 routing 分组时
把复核行也算了进去。

**修法**：给 `RunRecord` 加 `origin` 字段（`grid` / `replication`），
`PipelineResult.grid_runs` 只返回网格行。统计口径分两层：**网格行**用于「扇出完整性」
这类结构断言，**全部行**用于「分离是否成立」这类断言（那里样本越多越好）。
字段上的注释写明了这次事故。

### 64. `replication` 计数报 0

**症状**：`grid_complete` 的证据里 `replication = 0`，而复核明明跑了 15 次。

**原因**：`run_pipeline` 调 `compute_findings` 时只传了 `runs`，没传 `rep_rows`。

**修法**：传 `runs + rep_rows`（配合坑 63 的 `origin` 字段，结构断言依然只看网格行）。

### 65. 一条结论被实测推翻：「辅助损失买到均衡」

**症状**：第 5 条结论的判定条件不满足 —— Spearman(aux, H) 只有 0.303，逐轴看还有
反向的格子。

**原因**：最初按「aux 是分化的必要条件」写判定，实测发现 aux 与**均衡**的关系也很弱，
于是连「aux 买均衡」这个更弱的说法也站不住。

**修法**：把结论改写成 `aux_axis_is_flat`：「辅助损失不是分化的必要条件，这一轴几乎是平的」，
并把 `note` 里如实记下「连『aux 买均衡』也没站住」。**结论是算出来的，所以它能被推翻**；
若写死成常量字符串，这里就会变成一句漂亮的假话。

### 66. 一条结论的相关系数符号是反的

**症状**：第 4 条结论原假设「均衡 ↔ loss 不相关（ρ ≈ 0）」，实测 ρ = **+0.552**。

**原因**：假设本身就错了 —— 全网格上均衡与 loss 是**正**相关（越均衡越差），
因为完美均衡的那个配置恰好是学不会任务的对照组。

**修法**：结论改写成 `balance_points_the_wrong_way`，用三个相关系数一起说：
全网格 +0.552、learned 组内 +0.185、分化↔loss −0.768。**比原来的说法更强**。

### 67. `ROUTINGS` 少了 `static`

**症状**：`ValueError: 未知 routing：static` —— 设计文档里明明有三种路由。

**原因**：`schema.ROUTINGS` 只写了 `learned` 与 `random`。

**修法**：补成 `("learned", "static", "random")`，并加一张表说明每种方式**为什么存在**
（learned 是被测对象；static 按构造完美均衡却零分化，用来钉死「均衡 ≠ 分化」；
random 是地板）。三条路由不是可选特性，是**实验的对照骨架**。

### 68. `--quiet` 放在子命令之后

**症状**：`unrecognized arguments: --quiet`。

**原因**：`--quiet` 是顶层开关，argparse 的子解析器不认它。

**修法**：写在子命令之前（`main.py --quiet run`），并加一条测试同时断言
「放对位置能跑」与「放错位置会报错」。

### 69. `from .report import _fmt` 导入了私有名

**症状**：CLI 与报告各自维护数字格式，迟早会不一致 —— 而「命令行看到的」和
「报告里写的」不一样是最难查的一类不一致。

**修法**：在 `report.py` 里加公开别名 `fmt = _fmt`，并注明「CLI 与报告必须共用
同一套数字格式」。

### 70. `load_result` 里残留了一句无意义表达式，且网格声明在只读路径上丢了

**症状**：从产物重建时崩溃（`version.seed_of_first()` 根本不存在）；修掉之后
`grid` 变成空字典，「扇出完整性」在只读路径上退化成同义反复。

**修法**：`build_lineage` 增加 `grid` 参数，把网格的**声明**（各轴取值数）存进收拢节点
的 `meta`；`load_result` 从 `collect` 节点的 meta 里把它读回来。于是「从产物重建之后
仍能核对格数 == 各轴取值之积」。

### 71. 用两个不同 `aux_weight` 的模型去比辅助损失

**症状**：测试断言「α 翻十倍，损失翻十倍」失败（0.5219 vs 0.5167）。

**原因**：两个模型的 `spec_id` 不同 ⇒ 初始化不同 ⇒ 门控与负载都不一样。
比出来的是「两个模型」，不是「一个权重」。

**修法**：同一个模型、同一组门控与负载，只把 `spec.aux_weight` 换掉再算一次
（`dataclasses.replace`）。**比较实验里唯一变量必须是你要比的那个变量** —— 这条在
测试代码里同样成立。

### 72. `step 0` 的分化度不能当基线

**症状**：断言「nmi 随训练上升」失败 —— 有个配置 step 0 的 nmi 是 0.38，
比训练中途还高，最终才 0.29。

**原因**：`tag` 的嵌入是随机初始化的，于是**训练前**路由就已经带上标签信息，
`nmi` 因此可能虚高。它什么都没学。

**修法**：判据改成「最终分化度 > 曲线上最低点的 1.5 倍」（分化是训练出来的），
并单独加一条测试断言 step 0 的 loss 贴着 `ln(vocab)`、acc 贴着 0 ——
把「起点干净」这件事本身变成断言。

### 73. `distinct_successor_ratio` 不是满值

**症状**：断言「平均每个内容 token 有 4 个不同后继」失败，实测是 **3.25/4**。

**原因**：域规则是**随机**错排，两个域偶尔会把同一个 token 映射到同一个值
（例如某 token 的两个域都指向 3）。这是概率，不是 bug。

**修法**：断言改成三条可判定的性质：每个 token 至少 2 个不同后继、平均值 > 0.75×域数、
至少 1/4 的 token 达到满值。**不要把「期望的典型值」当成「必须成立的性质」** ——
随机构造出来的东西，断言要写它的分布性质，不要写某个具体数字。

### 74. 文档给的命令把自己的输出吞掉了

**症状**：按快速开始敲 `python -m pytest -q`，屏幕上只有四行点号，**看不到
`243 passed in 17.20s`**，于是「测试到底过了没有」只能靠退出码猜。

**原因**：`pyproject.toml` 的 `addopts = "-q"` 已经带了 `-q`，命令行再给一个就成了
`-qq`；pytest 在 `-qq` 下会省掉末尾的汇总行。这不是 pytest 的怪癖，是**同一个开关
被给了两遍**。

**修法**：文档里的命令改成 `python -m pytest`，并在注释里写清「不要再加 `-q`」。
顺带记住：凡是「配置里已有默认值」的开关，文档示例都只应该给**非默认**的那部分 ——
否则读者抄下来的命令和作者跑的命令不是同一条。

---

## 附：本项目的测试预算

`tests/conftest.py` 的 `fast_cfg` 削的是 `格子数 × 步数 × 批大小 × 样本数` 这个
**乘积**（24 格→6 格、180 步→20 步、batch 10→6、seq 10→6），于是「跑一次流水线」
从 100 秒降到约 2 秒。这条教训来自同工作区的 L2-architecturefactory：那边最初
只削了样本数、把 `steps` 原样留着，结果 156 条测试跑了 301 秒 —— **削一个因子等于没削**。

同时遵守两条配套纪律：

- **贵的产物共享出去。** `fast_out` / `fast_result` 都是 session 级的，整套测试只
  跑一次流水线；`acceptance_runs`（真实协议 120 步，6 个配置，约 7 秒）也只在
  `test_acceptance.py` 里共享一次。
- **契约测试用最小配置。** 「产物不存在时只读命令也不写盘」这条要跑 19 条命令，
  每条都会在内存里重跑流水线；用 `fast_cfg` 会拖到 36 秒，所以另建了一个
  `tiny_cfg`（2 格 + 1 次复核 + 3 步）。

**但验收命题不用削过的配置。** 20 步的路由器 loss ≈ 2.37，远在下界之上 ——
用它断言「learned 突破下界」必然是假。所以 `test_acceptance.py` 用真实协议单独跑
6 个配置：那是本项目的验收标准，不能被一个被削过的配置糊过去。

