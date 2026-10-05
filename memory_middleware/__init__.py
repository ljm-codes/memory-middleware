"""BalancedMultiDimensionMemory —— 面向 LLM 长期记忆的 LangGraph AgentMiddleware。

- 三层记忆：工作记忆 → 暂存库（向量 RAG）→ 长期记忆（用户画像）
- 艾宾浩斯衰减 + 多维加权评分 + LLM 动态调参
- 所有外部依赖（Redis/RabbitMQ/DashScope/主模型）均为可注入依赖，默认离线可跑
"""

__version__ = "0.1.0"

from .config import MemoryConfig, VocationParams, TYPE_SCORE_MAP
from .middleware import BalancedMultiDimensionMemory
from .models import (
    MemoryFragments,
    MemoryFragmentsMetadata,
    SummaryMemoryAi,
    SummaryMemoryFragmentsConfig,
    TimeMemoryFormulaParam,
    UserProfile,
)
from .spliter import MemoryFragmentsAiSpliter
from .rag import FragmentsMemoryRAG
from .storage.kv import KVStore, MemoryKVStore, RedisKVStore
from .storage.vectors import FragmentIdSource, HashEmbeddings, SQLiteVecStore
from .recovery import ErrorRecovery, NullRecovery, RabbitMQRecovery

__all__ = [
    "BalancedMultiDimensionMemory",
    "MemoryConfig",
    "VocationParams",
    "TYPE_SCORE_MAP",
    "MemoryFragments",
    "MemoryFragmentsMetadata",
    "SummaryMemoryAi",
    "SummaryMemoryFragmentsConfig",
    "TimeMemoryFormulaParam",
    "UserProfile",
    "MemoryFragmentsAiSpliter",
    "FragmentsMemoryRAG",
    "KVStore",
    "MemoryKVStore",
    "RedisKVStore",
    "HashEmbeddings",
    "SQLiteVecStore",
    "FragmentIdSource",
    "ErrorRecovery",
    "NullRecovery",
    "RabbitMQRecovery",
]
