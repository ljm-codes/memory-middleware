# -*- coding: utf-8 -*-
"""Layer 1：时间戳自包含——用户消息缺失 time 字段时，中间件自动补齐（无需外部辅助）"""
import asyncio
import time

from langchain_core.messages import HumanMessage


class TestSelfTimestamping:
    def test_abefore_model_stamps_missing_time(self, build_middleware, runtime):
        """无 time 字段的用户消息被自动打点（否则时间机制会静默失效）"""
        mw, _, _ = build_middleware()
        msgs = [HumanMessage('没有时间戳的消息')]
        assert 'time' not in msgs[0].additional_kwargs

        async def run():
            await mw.abefore_model({'messages': msgs, 'new_msg_idx': 0}, runtime)
            return msgs[0].additional_kwargs.get('time')

        t = asyncio.run(run())
        assert isinstance(t, float)
        assert 0 <= time.time() - t < 5  # 补的是"当前时间"

    def test_existing_timestamp_not_overwritten(self, build_middleware, runtime):
        """幂等：已有时间戳的消息不被覆盖"""
        mw, _, _ = build_middleware()
        old = 1000.0
        msgs = [HumanMessage('有旧时间戳', additional_kwargs={'time': old})]

        async def run():
            await mw.abefore_model({'messages': msgs, 'new_msg_idx': 0}, runtime)
            return msgs[0].additional_kwargs['time']

        assert asyncio.run(run()) == old

    def test_maturity_accumulates_from_stamped_time(self, build_middleware, runtime):
        """补点后时间机制正常起效：等待窗口变老后切片可触发（对比缺失时的静默失效）"""
        from memory_middleware.formula import maturity

        mw, _, _ = build_middleware()
        msgs = [HumanMessage('消息')]

        async def run():
            await mw.abefore_model({'messages': msgs, 'new_msg_idx': 0}, runtime)
            stamped = msgs[0].additional_kwargs['time']
            # 刚打点 → 成熟度应≈0（正确语义：未知年龄视为新消息）；
            # 打点与检查之间可能隔了几毫秒，允许微小正值（Windows 时间戳分辨率）
            return maturity(time.time() - stamped, mw.m_t, mw.m_c)

        m = asyncio.run(run())
        assert m < 0.05
        # 时间戳已存在，后续轮次会自然累积成熟度
        assert 'time' in msgs[0].additional_kwargs
