# -*- coding: utf-8 -*-
"""集成：token 与费用统计（用户要求：测试同时计算 token 与费用）。

依据：DeepSeek 官方 usage 字段 prompt_cache_hit_tokens / prompt_cache_miss_tokens
（命中按 ¥0.02/M、未命中 ¥1/M、输出 ¥2/M 计价，见 memory_middleware.cost）。
"""
import pytest

from memory_middleware.cost import cost_cny, hit_rate

pytestmark = pytest.mark.integration

TURNS = [
    '你好，我叫李四，是做数据分析的。',
    '我想问一下订单什么时候能到？',
    '还有发票的事情也帮我确认下。',
    '谢谢，另外我周末喜欢爬山。',
]


class TestCostTracking:
    def test_usage_collected_and_priced(self, run_conversation, usage_handler):
        """全链路跑完：usage 被采集，token/费用可按现行价计算"""
        run_conversation(TURNS, max_replies=4)

        usages = usage_handler.usage_list
        assert len(usages) >= 4, "至少应有 4 次模型调用（每轮 1 次 + 中间件内部调用）"

        total_in = sum(u['hit'] + u['miss'] for u in usages)
        total_out = sum(u['out'] for u in usages)
        total_cost = sum(cost_cny(u) for u in usages)
        avg_hit_rate = sum(hit_rate(u) for u in usages) / len(usages)

        assert total_in > 0
        assert total_out > 0
        assert total_cost > 0

        print(f"\n===== 费用统计（{len(usages)} 次调用，现行价 ¥0.02/1/2 per 1M tokens）=====")
        for i, u in enumerate(usages, 1):
            print(f"  调用{i:>2}: 输入 {u['hit'] + u['miss']:>6} tok"
                  f"（命中 {u['hit']:>5} / 未命中 {u['miss']:>5}）"
                  f" | 输出 {u['out']:>4} tok | 命中率 {hit_rate(u) * 100:5.1f}%"
                  f" | ¥{cost_cny(u):.6f}")
        print(f"  合计: 输入 {total_in} tok / 输出 {total_out} tok"
              f" / 平均命中率 {avg_hit_rate * 100:.1f}% / ¥{total_cost:.6f}")
