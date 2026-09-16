# BalancedMultiDimensionMemory（衡忆多维认知架构）

一个 **LangGraph `AgentMiddleware`**，基于艾宾浩斯遗忘曲线、多维加权评分、LLM 动态调参与增量画像归纳，为 LLM Agent 提供长期记忆。

> 从生产项目 [fireflymall-ai-customer-service](https://github.com/fufuxiaokeai/fireflymall-ai-customer-service)（LangGraph + DeepSeek 智能客服）解耦提炼的独立版本。**开箱即离线**——测试与演示零外部服务；Redis / RabbitMQ / 真实嵌入均为可选插件。

---

## 为什么需要它

普通对话上下文会遗忘。官方 `SummarizationMiddleware` 把旧轮次压成摘要——有用，但丢失了可检索的细节，也构建不了"用户模型"。本中间件实现**三层记忆**，参照人类记忆的工作方式：

```
┌────────────────────────────────────────────────────────────────────┐
│ 1. 工作记忆（短期）                                                 │
│    带时间戳的 messages 列表 —— 实时对话                              │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ 记忆成熟度 M(Δt) ≥ 阈值（艾宾浩斯）
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ 2. 暂存库（中期）—— 向量 RAG                                       │
│    对话切片，携带元数据：主题 / 类型 / 时间 / 巩固次数               │
│    按需检索，成功复用后"再巩固"                                      │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ 基于游标的增量总结
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ 3. 长期记忆库 —— 结构化用户画像（UserProfile）                      │
│    事实 / 偏好 / 目标 / 约束 …… 按用户独立存储                       │
└────────────────────────────────────────────────────────────────────┘
```

当上下文接近预算时，中间件检索最相关的片段**注入系统提示词**（核心 → 画像 → 片段，按 K-V 缓存前缀友好顺序排列），并截断原始历史——上下文保持有界，而关键事实留存。

## 核心公式

| 量 | 公式 | 含义 |
|---|---|---|
| 成熟度 | `M(Δt) = 1 − exp(−(Δt/τ_m)^c_m)` | 最早消息"熟透"→ 触发切片 |
| 时间衰减 | `T(m) = exp(−(Δt/τ)^c)` | 艾宾浩斯遗忘曲线；τ ≈ 衰减到 37% 的时间 |
| 固有重要性 | `F(m) = clamp(w₀ + w₁·Σ(vᵢ·I_type(i)) + w₂·(1−exp(−refresh·k)), 0, 1)` | 类型先天重要性 + 巩固次数 |
| 语义相关 | `R(m, q) = max(0, cos(m, q) − θ_min)` | 与当前主题的语义契合度 |
| 综合得分 | `S(m) = α·R + β·T + γ·F + δ` | top-K 注入的排序依据 |

α/β/γ/δ 与 w₀/w₁/w₂ 由 **LLM 按当前对话主题动态调参**（结构化输出，α+β+γ=1、w₀+w₁+w₂=1），权重随用户讨论内容自适应。

## 快速开始

```bash
pip install memory-middleware           # 从 PyPI 安装
# 或源码安装：
git clone https://github.com/fufuxiaokeai/memory-middleware.git && cd memory-middleware
pip install -e ".[dev]"
python examples/quickstart_offline.py   # 完全离线，脚本化假模型
pytest                                  # 79 个离线测试，零服务，<1s
```

接入真实模型（DeepSeek）：

```python
from memory_middleware import BalancedMultiDimensionMemory, MemoryConfig

memory = BalancedMultiDimensionMemory(MemoryConfig.from_vocation('customer service'))

agent = create_agent(
    model, tools,
    middleware=[memory],                 # 与 SummarizationMiddleware 同一插槽
)
```

运行时要求：`runtime.context.user_id`（或 `state['user_id']`）、`runtime.store`（任意 LangGraph store，如 `InMemoryStore`）、state 通道 `messages` / `new_msg_idx` / `user_profile` / `system_prompt`。

**消息时间戳自包含**：任何缺失 `time` 字段的消息会在首次见到时自动补点（视为"当前时间"），
艾宾浩斯成熟度/衰减机制无需任何外部打点钩子即可工作；已有时间戳**永远不会被覆盖**——
**需要精确时间戳时，请在你自己管线入口打点（上游模式），外部时间戳永远优先**。
兜底的偏差方向是保守的：未知年龄视为"现在"，只会推迟（不会提前）切片/衰减/总结；
每次兜底补点都会记 debug 日志。上游项目同时在其消息入口节点打点，两者幂等、可共存。

完整带 **token/费用统计** 的示例见 `examples/quickstart_real.py`（deepseek-v4-flash 上 4 轮对话 ≈ ¥0.003）。

## 依赖解耦

一切外部依赖都是**可注入的窄协议 + 离线默认实现**——测试与演示零服务即可运行：

| 关注点 | 协议/注入点 | 离线默认 | 生产插件 |
|---|---|---|---|
| 片段序号游标 | `KVStore`（`aget`/`aset`） | `MemoryKVStore`（进程内） | `RedisKVStore`（原子、多进程） |
| 失败保底 | `ErrorRecovery`（`on_split_error`/`on_summary_error`） | `NullRecovery`（仅记日志） | `RabbitMQRecovery`（持久队列 + 可选告警回调） |
| 向量嵌入 | langchain `Embeddings` | `HashEmbeddings`（确定性、离线） | DashScope / OpenAI / Ollama |
| 向量库 | `VectorStore` 接口 | `SQLiteVecStore`（本地文件 / `:memory:`） | Milvus / ES（同一接口） |
| 四个模型调用 | 模型实例或 `model_factory` | 脚本化假模型 | 任意 `init_chat_model` 提供商（DeepSeek/OpenAI/…） |
| token 计数 | `TokenCounter` | 字数/3.3 启发式 | tiktoken / transformers / 官方 usage |
| 主系统提示词 | `MemoryConfig.initial_prompt` | — | 应用自带提示词 |

如实说明的局限（文档明示，不隐藏）：
- `MemoryKVStore` 仅单进程——多进程部署请用 `RedisKVStore`
- `NullRecovery` 会丢弃失败的切分/总结载荷（该轮跳过，Agent 继续工作）——需要重试请挂 `RabbitMQRecovery`
- `HashEmbeddings` 语义质量低——生产召回请换真实嵌入模型

## 测试策略

**离线套件（默认 `pytest`，无需 API key）：**
- Layer 0 —— 纯数学：艾宾浩斯公式、clamp/参数约束、画像合并、scope 解析、提示词排序
- Layer 1 —— 假 LLM + 内存存储的组件测试：切片流程、增量总结（游标不回头）、检索注入、跨用户隔离、降级路径、并发访问
- 并发测试还抓到过一个真实 bug（共享 sqlite 连接跨用户竞争）——该 bug 在上游项目中同样存在

**集成套件（`pytest -m integration`，需要 `DEEPSEEK_API_KEY`）：**
- 真实模型端到端：片段落库、画像写入、片段注入、回答提及早期事实
- 逐调用 token 与费用统计——基于官方 `usage.prompt_cache_hit_tokens / prompt_cache_miss_tokens` 字段，按 DeepSeek 现行价计费（¥0.02 / ¥1 / ¥2 每百万 token，`memory_middleware.cost` 可配置）

## 实测数据

以下全部为真实运行数据（非估算）。缓存账目使用 DeepSeek 官方计费字段
`usage.prompt_cache_hit_tokens / prompt_cache_miss_tokens`
（命中 ¥0.02/M、未命中 ¥1/M、输出 ¥2/M，`memory_middleware/cost.py` 可配置）。

**离线套件** —— 79 个测试，约 1 秒，零服务、零 API key。

**真实 DeepSeek 端到端（deepseek-v4-flash，4 轮对话）**
（集成测试 `tests/integration/test_cost_tracking.py`，2026-08-16 运行）：

| 指标 | 数值 |
|---|---|
| LLM 调用次数 | 13 —— 其中 **4 次为主 Agent 对话调用**（对话主循环），**9 次为中间件内部调用**（主题切分 / 增量总结 / 参数调优） |
| 输入 token | 8,276 |
| 输出 token | 1,009 |
| 平均缓存命中率 | 62.1%（中间件内部切分/总结调用：86–93%） |
| 总费用 | ¥0.0033 |

逐调用账目来自官方 `usage` 字段，按 ¥0.02 / ¥1 / ¥2 每百万 token 计价
（`memory_middleware/cost.py`）。主 Agent 与中间件内部调用在同一采集会话中，
均经过同一用量采集回调（见 `tests/integration/conftest.py`）。

**召回证据** —— 真实 5 轮对话后（第 1 轮声明姓名与爱好），**主 Agent 对话调用**的最终回复：

> 张三先生，我记得您上次提到游泳爱好。考虑到您的运动场景，我为您推荐 IP68 防水手机…

### 三方对比（针-草垛）

评测脚本：`benchmarks/three_way_recall.py`（确定性事实针 + 闲聊草垛 + 提问，
三种配置共用完全相同的对话文本；上下文预算 = 8 条消息；真实 DeepSeek + DashScope 嵌入）。
召回判定 = 回答包含事实关键词。

| 会话长度 | 无记忆基线 | SummarizationMiddleware | BMDM |
|---|---|---|---|
| 8 轮 | 0/4 | 3/4 | 3/4 |
| 16 轮 | 0/4 | 3/4 | 2/4 |
| 24 轮 | 0/4 | 3/4 | 3/4 |

![长期记忆召回基准](benchmarks/output/recall_benchmark.png)

24 轮时的输入成本：基线 3,582 token / 31 次调用 · Summarization 11,809 / 40 ·
BMDM 66,751 / 99（约为 Summarization 的 3 倍——三遍架构（切分/总结/对话）的代价）。

### 跨会话召回（会话内基准看不到的维度）

SummarizationMiddleware 的摘要活在会话 state 里——**全新线程从零开始**。
BMDM 把画像与片段持久化在 store。会话 1 陈述事实，会话 2（新线程，
暖场闲聊 + 同样的 4 条提问）提问：

| 会话1 长度 | 无记忆基线 | SummarizationMiddleware | BMDM |
|---|---|---|---|
| 8 轮 | 0/4 | 0/4 | **3/4** |
| 16 轮 | 0/4 | 0/4 | **3/4** |
| 24 轮 | 0/4 | 0/4 | **3/4** |

![跨会话召回基准](benchmarks/output/cross_session_recall.png)

如实解读：
- 没有跨会话机制，新会话**什么都记不住**——两种基线在所有长度下都是 0/4，符合设计。
- BMDM 在新会话中召回 3/4（姓名/职业/爱好经持久化画像；"周三下午要开会"虽已入画像
  但未在回答中浮现——与会话内看到的约束细节缺口一致）。
- 值得知道的特性：记忆注入只在窗口达到触发阈值（8 条消息）后生效——新会话前几轮
  没有记忆辅助，这是设计使然。
- 观察到的数据质量问题：增量画像合并只做精确去重，`user_constraints` 在多次切片后
  累积了 7~14 条同一约束的不同措辞（语义去重或列表上限是后续改进点）。

### 细节保留（"总结不丢细节"的实测）

8 条低显著度细节（猫名/订单号/卡尾号/生日/机型/空闲时段/爱吃的/住址）在对话开头陈述，
末尾逐条提问：

| 会话长度 | 无记忆基线 | SummarizationMiddleware | BMDM |
|---|---|---|---|
| 8 轮 | 0/8 | 2/8 | **4/8** |
| 16 轮 | 0/8 | 2/8 | **5/8** |
| 24 轮 | 0/8 | 1/8 | **4/8** |

![细节保留基准](benchmarks/output/detail_retention.png)

如实解读：
- 摘要对细节的召回**随长度退化**（2→2→1）：对话越长，压缩丢掉的低显著度细节越多。
- BMDM 稳定在 4~5/8：片段**原文存储**、持续可检索，差距随对话长度拉大。
- 两种机制都丢纯数字类细节（订单号/卡尾号/生日/空闲时段）：摘要直接压缩掉，
  BMDM 的主题门控检索也不稳定（真实边界，不隐藏）。
- 24 轮时成本：基线 5,492 token/40 次 · Summarization 15,842/52 · BMDM 90,270/128——
  此规模下细节保留约为 Summarization 的 6 倍成本。

如实解读：
- 严格的 8 条消息预算下，**无记忆基线全部遗忘**——第 1 轮的事实直接消失。
- 两种记忆机制都保住 ~75% 召回；16 轮时 BMDM 掉到 50%（一个事实在检索波动中丢失——
  此规模下单样本的噪声）。
- 两个机制丢的是**同一个**事实（"每周三下午要开会"这类约束细节）：摘要会丢掉这类
  细节，画像的约束字段也不稳定。
- BMDM 的差异化价值在短程图表之外：**按用户的结构化画像 + 可主题检索的片段**
  （vs 一段不透明的摘要文本），以及跨会话持久化。首版为每长度单样本，
  加大样本量/拉长会话可拉开差距。
- 数据文件：`benchmarks/output/recall_results.json`（`--chart-only` 可重新出图）。

### 为什么中间件调用不带对话历史（实测）

这是一个看似反直觉的设计决策——携带历史会**提高**缓存命中率，却**提高**账单。实测于 deepseek-v4-flash，8 组 × 6 次调用，输入成本比值（B = 携带"忽略"历史前缀，A = 无状态基线）：

| 调用类型 | 冷启动（首次写入） | 温热重放（缓存已预热） |
|---|---|---|
| 数学调参 | B/A = 1.82× | 0.59× |
| 片段切分 | B/A = 1.89× | 0.71× |
| 画像总结 | B/A = 2.30× | **1.90×** |

- **冷启动**（生产常态：对话内容不会字节级重复——中间件处理的是互不重叠的增量）：历史 token 首次出现按全价计费，之后才享受折扣读。B 恒亏。
- **温热重放**（TTL 内重复发送相同请求）：历史摊销后 B 在"历史很短"时可能小赢——但生产历史无限增长，2% 的读成本随之增长。
- **陷阱**：温热运行中 B-summary 命中率达 93.3%（A-summary 仅 0%），成本却仍贵 1.9 倍。**命中率 ≠ 成本。**
- 实测缓存机制：DeepSeek 按 128-token 单元从开头缓存；末尾一个单元即使输入字节相同也按全价计费。

结论：三个中间件调用保持无状态；K-V 缓存前缀思维用在真正见效的地方——静态系统提示词（跨调用、跨用户共享缓存）与主对话的增长历史。

## 配置

`MemoryConfig` 关键字段（完整见 `memory_middleware/config.py`）：

| 字段 | 默认值 | 含义 |
|---|---|---|
| `model_name` / `model_kwargs` | `deepseek-v4-flash` | 三个内部调用所用模型 |
| `vocation` | customer service | τ_m / τ / c / 阈值预设（collaborative creation / customer service / accompany / 自定义） |
| `pattern` / `trigger_threshold` | fraction / 0.8 | 检索注入的触发方式（上下文预算比例 / token 数 / 消息条数） |
| `rag_db_path` | `memory_fragments.db` | 本地 sqlite 文件（可用 `:memory:`） |
| `retrieve_k` / `top_k` | 6 / 3 | 检索候选数 → 注入 top-K |

## 上游项目

本包是从 [fireflymall-ai-customer-service](https://github.com/lijia-ming/fireflymall-ai-customer-service)
（生产级智能客服 Agent，LangGraph + DeepSeek）中解耦提炼的记忆中间件。上游仓库保留
业务耦合版本（Redis / RabbitMQ / DashScope 接线），并含可复现的缓存率基准
（`bench_cache_rate.py`）——上表数据即出自该基准。

### 精炼版 vs 上游完整实现——你会失去什么

本包是**解耦子集**，不是超集。生产使用前请知晓差异：

| 关注点 | 本包（离线默认） | 上游生产接线 |
|---|---|---|
| 失败保底 | `NullRecovery`——失败轮次跳过 | RabbitMQ 持久队列 + 邮件告警（`memory_rag.py`） |
| 片段游标 | `MemoryKVStore`——仅单进程 | Redis（多进程安全） |
| 向量嵌入 | `HashEmbeddings`——离线、语义质量低 | DashScope `text-embedding-v4` |
| token 计数 | 字数/3.3 启发式 | tiktoken / transformers |

上游完整实现位于：`Tools/middleware/memory/time_memory.py`
（BalancedMultiDimensionMemory 全量实现）、`memory_rag.py`（切分/增量总结/RabbitMQ 保底）、
`Tools/middleware/compose.py`（中间件洋葱链组合）。需要生产级韧性时，请直接使用上游项目，
或在本包的注入点上接入对应实现。

## License

MIT
