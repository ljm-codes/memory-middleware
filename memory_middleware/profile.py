"""用户画像的增量合并。

增量总结只产出"游标之后新片段"的画像，需要与既有画像合并：
- 列表字段：去重合并（新值优先）；可选语义去重（注入 embedding 时），
  否则退化为精确去重；可选列表上限（max_items），超出时保留新值截断
- 字典字段：递归合并
- 标量字段：仅当新值非空且不是模型默认值时才覆盖（防止增量总结把未提及字段重置为默认值）

膨胀背景：增量总结每轮都可能产出同一约束（如 user_constraints）的不同措辞，
精确去重无法识别同义变体（"不吃香菜" vs "忌香菜"），长期累积会膨胀到 7~14 条。
语义去重依赖注入的 embedding（生产接真实嵌入模型）；离线默认（embedding=None）
不做语义去重——HashEmbeddings 无语义，宁可不删不可误删。

成本控制：语义去重只在列表"已经膨胀"（精确去重后 ≥ semantic_min_items 条）
时才调用嵌入 API；小列表直接精确去重，零 API 调用。嵌入比 LLM 便宜约 1 个数量级，
且只在低频的增量总结时触发（对比数学调参是每次检索都调 LLM）。
"""
import math
from typing import Any, Optional

from pydantic_core import PydanticUndefined

from .models import UserProfile


def field_default_value(field_name: str):
    """获取 UserProfile 字段的默认值，用于判断增量总结中哪些字段未被提及"""
    fld = UserProfile.model_fields.get(field_name)
    if fld is None:
        return None
    if fld.default is not PydanticUndefined:
        return fld.default
    if fld.default_factory is not None:
        return fld.default_factory()
    return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度（两向量长度不一致按 0 处理，避免下游 embedding 维度漂移静默出错）"""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _semantic_merge_list(
        old: list[str], new: list[str], embedding, similarity_threshold: float) -> list[str]:
    """语义去重合并（新值优先，结果新在前旧在后，与精确去重的顺序语义一致）：
    - new 内部两两相似（≥ 阈值）只保留先出现的项
    - old 中与 new 任意项相似的视为同义变体，丢弃（新措辞刷新旧措辞）
    - 其余 old 项追加在后
    """
    new_vecs = embedding.embed_documents(new)
    old_vecs = embedding.embed_documents(old) if old else []
    merged: list[str] = []
    merged_vecs: list[list[float]] = []
    for item, vec in zip(new, new_vecs):
        if any(_cosine_similarity(vec, mvec) >= similarity_threshold for mvec in merged_vecs):
            continue
        merged.append(item)
        merged_vecs.append(vec)
    for item, vec in zip(old, old_vecs):
        if any(_cosine_similarity(vec, mvec) >= similarity_threshold for mvec in merged_vecs):
            continue
        merged.append(item)
        merged_vecs.append(vec)
    return merged


def merge_user_profile(
        old: dict, new: dict, *,
        embedding: Any = None,
        similarity_threshold: float = 0.85,
        semantic_min_items: int = 5,
        max_items: Optional[int] = None,
) -> dict:
    """将增量总结的新画像合并进旧画像（不修改入参，返回新 dict）。

    embedding：langchain Embeddings 实例（embed_documents）；None 时列表字段仅精确去重。
    similarity_threshold：余弦相似度阈值，同义变体的判定标准（真实嵌入模型
        同义约束通常 0.85+，取保守值宁漏不去）。
    semantic_min_items：语义去重的最小列表规模——精确去重后不足此条数的
        列表直接走精确去重，不调用嵌入 API（小列表无膨胀风险，省成本）。
    max_items：列表字段上限，超出保留新值截断（None 不限制）。
    """
    merged = dict(old)
    for key, new_val in new.items():
        old_val = merged.get(key)
        if isinstance(new_val, list):
            old_items = old_val if isinstance(old_val, list) else []
            exact_combined = list(new_val)
            for item in old_items:
                if item not in exact_combined:
                    exact_combined.append(item)
            if embedding is not None and len(exact_combined) >= semantic_min_items:
                combined = _semantic_merge_list(old_items, new_val, embedding, similarity_threshold)
            else:
                combined = exact_combined
            if max_items is not None and len(combined) > max_items:
                combined = combined[:max_items]
            merged[key] = combined
        elif isinstance(new_val, dict):
            merged[key] = {**(old_val if isinstance(old_val, dict) else {}), **new_val}
        else:
            default = field_default_value(key)
            if new_val not in (None, '', default):
                merged[key] = new_val
    return merged
