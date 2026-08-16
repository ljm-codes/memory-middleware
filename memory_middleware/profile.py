"""用户画像的增量合并。

增量总结只产出"游标之后新片段"的画像，需要与既有画像合并：
- 列表字段：去重合并（新值优先）
- 字典字段：递归合并
- 标量字段：仅当新值非空且不是模型默认值时才覆盖（防止增量总结把未提及字段重置为默认值）
"""
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


def merge_user_profile(old: dict, new: dict) -> dict:
    """将增量总结的新画像合并进旧画像（不修改入参，返回新 dict）"""
    merged = dict(old)
    for key, new_val in new.items():
        old_val = merged.get(key)
        if isinstance(new_val, list):
            combined = list(new_val)
            if isinstance(old_val, list):
                for item in old_val:
                    if item not in combined:
                        combined.append(item)
            merged[key] = combined
        elif isinstance(new_val, dict):
            merged[key] = {**(old_val if isinstance(old_val, dict) else {}), **new_val}
        else:
            default = field_default_value(key)
            if new_val not in (None, '', default):
                merged[key] = new_val
    return merged
