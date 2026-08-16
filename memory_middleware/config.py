"""配置：构造器注入式（上游项目 fireflymall-ai-customer-service，
https://github.com/fufuxiaokeai/fireflymall-ai-customer-service —— 通过 load_config.config
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
    retrieve_k: int = 6        # 检索候选数
    top_k: int = 3             # 注入 top-N
    strengthen_k: float = 0.5  # F(m) 中巩固次数的衰减系数

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
