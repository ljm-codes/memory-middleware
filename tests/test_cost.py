# -*- coding: utf-8 -*-
"""Layer 0：token/费用统计（cost.py）"""
from types import SimpleNamespace

from memory_middleware.cost import cost_cny, extract_usage, hit_rate


def _msg_with_usage(hit, miss, out):
    return SimpleNamespace(response_metadata={
        'token_usage': {
            'prompt_cache_hit_tokens': hit,
            'prompt_cache_miss_tokens': miss,
            'completion_tokens': out,
        }
    })


class TestExtractUsage:
    def test_extract_deepseek_fields(self):
        usage = extract_usage(_msg_with_usage(hit=100, miss=50, out=20))
        assert usage == {'hit': 100, 'miss': 50, 'out': 20}

    def test_missing_metadata_returns_zeros(self):
        assert extract_usage(SimpleNamespace(response_metadata=None)) == {'hit': 0, 'miss': 0, 'out': 0}

    def test_usage_metadata_fallback(self):
        msg = SimpleNamespace(response_metadata={'usage': {'prompt_cache_hit_tokens': 1}})
        assert extract_usage(msg)['hit'] == 1


class TestCost:
    def test_cost_formula(self):
        # 100 miss × ¥1/M + 100 hit × ¥0.02/M + 50 out × ¥2/M
        usage = {'hit': 100, 'miss': 100, 'out': 50}
        assert cost_cny(usage) == pytest.approx((100 + 2 + 100) / 1e6)

    def test_custom_pricing(self):
        usage = {'hit': 0, 'miss': 1000, 'out': 0}
        assert cost_cny(usage, {'miss': 1.5}) == pytest.approx(1500 / 1e6)

    def test_hit_rate(self):
        assert hit_rate({'hit': 90, 'miss': 10, 'out': 0}) == pytest.approx(0.9)
        assert hit_rate({'hit': 0, 'miss': 0, 'out': 5}) == 0.0


import pytest  # noqa: E402  （保持导入放在文档说明后）
