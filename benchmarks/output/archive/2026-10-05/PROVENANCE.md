# 2026-10-05 三方召回基准 · 数据归档与出处

本目录保存 2026-10-05 那轮"切分并入事件"修复前后的**全部原始跑分**，以及每个数据点的
出处（命令、口径、修复前后）。基准脚本：`benchmarks/three_way_recall.py`；
模型 DeepSeek deepseek-v4-flash + DashScope `text-embedding-v4`；上下文预算 = 8 条消息。

## 文件

| 文件 | 内容 |
|---|---|
| `recall_results_prefix.json` | **修复前**全部跑分（备份于修复前，含 `8-0/16-0/24-0` 旧值、`xs-*`、`d-8-0`、`d-16-0`） |
| `recall_results_postfix.json` | **修复后**全部跑分（含 A/B 两组、细节、跨会话、3 样本；key 带 tag 区分口径） |
| `recall_results_premerge.json`、`recall_results_baseline.json` | 更早的历史跑分（切分+画像合并实验期），一并留档 |

## 口径（两种）

- **门控不阻塞**：`slice_value=0.0` / `long_term_value=0.0` / `trigger=8 条` / `τ_m=60`（压缩时间尺度）。
  让机制在压缩时间的基准里密集触发，用来量**每次事件的单位成本上限**。
- **生产口径**：`--production-gates`（`slice=0.7` / `long_term=0.9` / `τ_m=600` / `τ=43200`）
  + `--trigger-threshold 50` + `--virtual-turn-seconds 35`（每轮 35 虚拟秒 → 25 轮 ≈ 875 s ≈ 成熟阈值 869 s）。
  这是中间件在真实会话节奏下的行为。**没有虚拟时钟时，生产门控在压缩基准里没有意义**
  （消息只有几秒大 → 门永不成熟；或门一开就再不复位）。

## 数据点出处

| 结果 key | 维度 | 口径 | 修复 | 命令（`benchmarks/` 下） |
|---|---|---|---|---|
| `8-0` `16-0` `24-0` | 事实针 | 门控不阻塞 | 后 | `python three_way_recall.py --lengths 8,16,24 --samples 1 --embeddings dashscope` （跑前清 `output/tmp`） |
| `24-0-gated` | 事实针 | 生产 | 后 | `--lengths 24 --samples 1 --production-gates --trigger-threshold 50 --virtual-turn-seconds 35 --tag=-gated --fresh` |
| `d-8-0-postfix` `d-16-0-postfix` `d-24-0-postfix` | 细节 8 条 | 门控不阻塞 | 后 | `--details --lengths 8,16,24 --tag=-postfix --fresh` |
| `d-16-0-gated` `d-24-0-gated` | 细节 8 条 | 生产 | 后 | `--details --lengths 16,24 --production-gates --trigger-threshold 50 --virtual-turn-seconds 35 --tag=-gated --fresh` |
| `d-24-{0,1,2}-gated3` | 细节 8 条 | 生产 · **3 样本** | 后 | `--details --lengths 24 --samples 3 --production-gates --trigger-threshold 50 --virtual-turn-seconds 35 --tag=-gated3 --fresh` |
| `xs-16-0-postfix` `xs-24-0-postfix` | 跨会话 | — | 后 | `--cross-session --lengths 16,24 --tag=-postfix --fresh` |
| `24-0-fair` | 事实针 | 生产 · **两边同阈值 50 条** | 后 | `--lengths 24 --production-gates --trigger-threshold 50 --summarize-trigger 50 --virtual-turn-seconds 35 --tag=-fair --fresh` |
| `d-24-{0,1,2}-fair3` | 细节 8 条 | 生产 · 同阈值 50 条 · 3 样本 | 后 | `--details --lengths 24 --samples 3 --production-gates --trigger-threshold 50 --summarize-trigger 50 --virtual-turn-seconds 35 --tag=-fair3 --fresh` |
| `d-24-{0,1,2}-diag3` | 细节 8 条（带诊断） | 同上 | 后 | 同上 + `--tag=-diag3`（输出片段库 / 注入 id / 主题 / 最终提示词） |
| `d-24-{0,1,2}-stable6` | 细节 8 条 | 同上 + `--top-k 6` | 后 | 同上 + `--top-k 6 --tag=-stable6` |
| `d-24-{0,1,2}-fixed3` | 细节 8 条 | 同上 + `--retrieve-k 20 --top-k 4 --always-inject-types identity,preference` | 后 | 同上 + `--retrieve-k 20 --top-k 4 --always-inject-types identity,preference --tag=-fixed3`（含记忆体积计量） |
| `d-24-{0,1,2}-default3` | 细节 8 条 | 同上 · **默认值**（不预筛 + 保底，top_k=3） | 后 | 同上（**不加任何选片覆盖参数**）`--tag=-default3` |
| `xs-8-0` `xs-16-0` `xs-24-0`、`d-8-0` `d-16-0`（无 tag） | 跨会话 / 细节 | — | **前** | 上一轮会话（见 `recall_results_prefix.json`） |

## 结论摘要（修复后）

| 维度 | 口径 | BMDM | SummarizationMiddleware | BMDM/官方 |
|---|---|---|---|---|
| 事实针 24 轮 | 门控不阻塞 | 3/4 | 3/4 | 2.38× 输入 · 1.30× 调用 |
| 事实针 24 轮 | 生产 · **两边同阈值 50 条** | 3/4 | 3/4 | **1.05× 输入 · 1.03× 调用** |
| 细节 24 轮 | 门控不阻塞 | 8/8 | 2/8 | 2.55× 输入 · 1.29× 调用 || 细节 24 轮 | 生产 · 同阈值 50 条 · 3 样本（默认选片） | 6.67/8（4·8·8） | 7.33/8（8·7·7） | 0.70× 输入 · 1.02× 调用 |
| 细节 24 轮 | 同上 · **+ `retrieve_k=20` + `always_inject_types=identity,preference`** | **8.0/8（8·8·8）** | 7.67/8（8·8·7） | 记忆正文 230 vs 270 tok |
| 跨会话 16/24 轮 | — | **3/4** | 0/4（结构性） | — |

¹ 已用**同阈值对照组**（`24-0-fair`，两边窗口预算都设 50 条）定论：两边都只触发 1 次事件，
BMDM 每次事件 2 次内部调用（切分 + 调参）vs 官方 1 次 → 总量基本相等（1.03× 调用 / 1.05× 输入）。
此前"调用更少"（0.83×）来自两边阈值不对等（BMDM 50 条 vs 官方 10 条），是**阈值假象**，已更正。
单位成本 3:1（每次事件，含归纳）仍是唯一稳定可报的架构结论。

**归一化口径（跨口径可比的那一栏）**：BMDM 每次事件 = 切分 + 调参 +（片段成熟时）归纳 = 门控不阻塞
口径实测 **3.0** 次结构化调用；官方每次摘要事件 = **1.0** 次。事件频率由各自窗口预算决定
（5/7/9 次 vs 7/9/12 次）。**总量 = 单位成本 × 频率，两者必须分开报。**

**修复前的失真来源**（详见主项目 `记忆中间件修改记录_2026-10-05.md`）：① 成熟度门被设成 0.0；
② 切分当时独立于主触发事件执行 → 窗口不复位时每轮调一次切分模型（占内部调用一半以上）。

**另有两处对官方（SummarizationMiddleware）的手脚，已于本日末修正**（影响上表所有细节行）：

1. 官方摘要此前误用聊天模型，被 `max_tokens=100` 截断 —— 会话越长截得越狠。
2. 触发预算两边不等（BMDM 50 条 vs 官方 10 条）。

修正后重跑的**同阈值对照**（`24-0-fair`、`d-24-{0,1,2}-fair3`）才是可比口径：
事实针 1.05× 输入 / 1.03× 调用（召回持平）；细节 官方 7.33/8 vs BMDM 6.67/8、输入 0.70×。
**上表中一切"门控不阻塞 / 阈值不等"的细节行只作历史留档，不得再引用。**
