"""token 与费用统计（测试与基准用）。

依据：DeepSeek 官方 API 每响应返回 usage.prompt_cache_hit_tokens /
prompt_cache_miss_tokens（计费字段：命中按折扣价、未命中按全价）。

注意：with_structured_output 返回的 pydantic 实例不携带 response_metadata，
需要从原始模型响应（AIMessage）提取 usage —— 见 tests/integration/ 的采集方式。
"""
from typing import Optional

# deepseek-v4-flash 人民币现行价（¥/1M tokens，2026-08-17 峰谷价生效前）
# 峰谷价（8-17 起）：空闲 ¥0.05/1.5/4.5，高峰(北京时间 9-12、14-18) ¥0.10/3.0/9.0
# 传入自定义 pricing 覆盖即可，命中/未命中比值不变，相对结论不变。
DEFAULT_PRICING_CNY = {
    'hit': 0.02,    # 缓存命中输入
    'miss': 1.0,    # 缓存未命中输入
    'out': 2.0,     # 输出
}


def extract_usage(message) -> dict:
    """从模型响应（AIMessage / 含 response_metadata 的对象）提取计费用量字段"""
    meta = getattr(message, 'response_metadata', None) or {}
    usage = meta.get('token_usage') or meta.get('usage') or {}
    return {
        'hit': int(usage.get('prompt_cache_hit_tokens', 0) or 0),
        'miss': int(usage.get('prompt_cache_miss_tokens', 0) or 0),
        'out': int(usage.get('completion_tokens', 0) or 0),
    }


def cost_cny(usage: dict, pricing: Optional[dict] = None) -> float:
    """按当前定价计算费用（¥）。pricing 形如 {'hit': .., 'miss': .., 'out': ..}"""
    p = {**DEFAULT_PRICING_CNY, **(pricing or {})}
    return (usage['hit'] * p['hit'] + usage['miss'] * p['miss'] + usage['out'] * p['out']) / 1e6


def hit_rate(usage: dict) -> float:
    """缓存命中率 = hit / (hit + miss)（仅输入）"""
    total = usage['hit'] + usage['miss']
    return usage['hit'] / total if total else 0.0
