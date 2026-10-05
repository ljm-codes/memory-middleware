"""配置：构造器注入式（上游项目 fireflymall-ai-customer-service，
https://github.com/lijia-ming/fireflymall-ai-customer-service —— 通过 load_config.config
全局读取，独立版改为显式配置）。

示例：
    from memory_middleware import MemoryConfig, VocationParams
    config = MemoryConfig(
        model_name='deepseek-v4-flash',
        model_kwargs={'temperature': 1.3, 'extra_body': {'thinking': {'type': 'disabled'}}},
        vocation=VocationParams(name='customer service'),
        pattern='fraction',
        trigger_threshold=0.8,
    )
"""
from dataclasses import dataclass, field
from typing import Any, Optional

# 记忆类型的基础重要性（自我参照效应 Rogers et al., 1977 等）
TYPE_SCORE_MAP: dict[str, float] = {
    # 自我相关记忆具有自我参照效应，回忆率最高
    'identity': 0.95,
    # 决策记忆涉及脚本记忆，高重复调用性
    'decision': 0.85,
    # 偏好属于内隐态度，稳定性高但检索频率中等
    'preference': 0.80,
    # 语义记忆，稳定但情感附着低
    'fact': 0.60,
    # 情景记忆，易受时间衰减影响最大
    'episode': 0.40,
    # 无结构闲聊，遗忘曲线最陡
    'chat': 0.15,
}


@dataclass
class VocationParams:
    """职业场景参数（艾宾浩斯公式的时间尺度等）"""
    name: str = 'customer service'
    tau_m: float = 600.0        # M(Δt) 时间尺度（秒），控制切片固化速度
    c_m: float = 0.5            # M(Δt) 曲线形状
    tau: float = 43200.0        # T(m) 时间尺度（秒），43200 = 12h
    c_t: float = 0.5            # T(m) 曲线形状
    slice_value: float = 0.7    # 切片触发阈值
    long_term_value: float = 0.9  # 归纳（总结为长期记忆）触发阈值


VOCATION_PRESETS: dict[str, dict] = {
    'collaborative creation': dict(
        tau_m=3600, c_m=1.0, tau=604800, c_t=0.7, slice_value=0.9, long_term_value=0.98),
    'customer service': dict(
        tau_m=600, c_m=0.5, tau=43200, c_t=0.5, slice_value=0.7, long_term_value=0.9),
    'accompany': dict(
        tau_m=7200, c_m=0.8, tau=86400, c_t=0.5, slice_value=0.8, long_term_value=0.95),
}


@dataclass
class MemoryConfig:
    """BalancedMultiDimensionMemory 的完整配置（替代上游项目 config.yaml 的 model.summary 段）"""
    # ---- 模型 ----
    model_name: str = 'deepseek-v4-flash'
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    # ---- 记忆公式参数 ----
    vocation: VocationParams = field(default_factory=VocationParams)

    # ---- 总结触发 ----
    pattern: str = 'fraction'             # fraction | tokens | messages
    trigger_threshold: float = 0.8
    max_input_tokens: int = 1_000_000     # fraction 模式下的上下文预算（1m 等价解析见 _get_model_max_tokens）

    # ---- 向量暂存库 ----
    rag_db_path: str = 'memory_fragments.db'   # sqlite 文件路径，':memory:' 也可
    rag_table: str = 'memory_fragments'
    embeddings: Optional[Any] = None           # langchain Embeddings；None 时用内置 HashEmbeddings（离线）

    # ---- 检索 ----
    # 候选池大小。None = 不限（该用户的全部片段都参与打分）——推荐值：
    # 预筛按"与当前主题的语义相似度"取候选，而它无从判断片段内容的价值；细节往往扎堆在
    # 少数片段里，主题一偏就把它们整块挡在门外（实测：召回在 3/8~8/8 之间跳）。
    # 放开预筛几乎零成本：sqlite-vec 本就是全表算距离再取前 k，而打分是本地算术。
    retrieve_k: Optional[int] = None
    top_k: int = 4             # 注入 top-N（实测 3→4 可把细节召回的样本间波动收成零方差）
    top_k_max_per_theme: Optional[int] = None  # 注入集合里单主题上限（None=不限；防同主题占满名额）
    # 保底类型：这些类型的片段优先占席（与 TYPE_SCORE_MAP 的最高两档 0.95/0.80 一致）。
    # 动机：检索预筛按主题语义取候选，主题一偏，identity/preference 片段会整块落选——
    # 而它们正是公式自己声明的最重要记忆。
    always_inject_types: tuple = ('identity', 'preference')
    strengthen_k: float = 0.5  # F(m) 中巩固次数的衰减系数

    # ---- 画像列表字段去重（膨胀控制：同义变体如"不吃香菜"/"忌香菜"累积膨胀） ----
    profile_dedup_threshold: float = 0.85   # 语义去重余弦阈值（真实嵌入同义约束 0.85+，宁漏不去）
    profile_semantic_min_items: int = 5     # 精确去重后达到此条数才调嵌入 API（小列表零成本）
    profile_list_max: Optional[int] = 10    # 列表字段上限，超出保留新值截断（None 不限制）

    # ---- 提示词 ----
    initial_prompt: str = ''   # 主系统提示词（上游项目从 agent.main_agent 导入，这里注入）

    @classmethod
    def from_vocation(cls, name: str, **overrides: Any) -> "MemoryConfig":
        """按职业预设快速构建（collaborative creation / customer service / accompany）"""
        preset = VOCATION_PRESETS.get(name)
        if preset is None:
            raise ValueError(
                f"未知职业场景 {name!r}，可选：{list(VOCATION_PRESETS)}，自定义请直接构造 VocationParams")
        return cls(vocation=VocationParams(name=name, **preset), **overrides)
