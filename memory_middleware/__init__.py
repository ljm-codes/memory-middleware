"""BalancedMultiDimensionMemory —— 面向 LLM 长期记忆的 LangGraph AgentMiddleware。

- 三层记忆：工作记忆 → 暂存库（向量 RAG）→ 长期记忆（用户画像）
- 艾宾浩斯衰减 + 多维加权评分 + LLM 动态调参
- 所有外部依赖（Redis/RabbitMQ/DashScope/主模型）均为可注入依赖，默认离线可跑
"""

def _detect_version() -> str:
    """版本从安装元数据读（单一来源 = pyproject.toml），避免手写副本漂移。

    此前这里硬编码过 `__version__ = "0.1.0"`，而 pyproject 已到 0.2.0——
    `memory_middleware.__version__` 与实际安装版本长期不一致。源码直接 import（未安装）时退回占位串。
    """
    try:
        from importlib.metadata import version
        return version('memory-middleware')
    except Exception:
        return '0.0.0+unknown'


__version__ = _detect_version()

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
