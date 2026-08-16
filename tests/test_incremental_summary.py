# -*- coding: utf-8 -*-
"""Layer 1：增量画像总结（归纳游标 + 画像合并 + 只总结新片段）"""
import asyncio
import time

from langchain_core.messages import HumanMessage

from helpers import make_fragment
from memory_middleware.models import UserProfile


def _mature_messages(age=10_000):
    now = time.time()
    return [HumanMessage('对话消息', additional_kwargs={'time': now - age})]


class TestIncrementalSummary:
    def test_summarizes_and_advances_cursor(self, build_middleware, runtime):
        """成熟新片段 → 总结一次 → 画像落 store → 游标=最大 fid"""
        mw, vec_store, _ = build_middleware(
            summary_result=UserProfile(user_name='张三', user_hobby=['读书']))
        # 预置 2 个已成熟片段
        docs = [make_fragment('u1', 1, age=10_000), make_fragment('u1', 2, age=9_000)]

        async def run():
            await vec_store.aadd_documents(docs)
            return await mw._memory_slice(_mature_messages(), user_id='u1', runtime=runtime)

        assert asyncio.run(run()) is True
        # 总结模型只被调用一次
        assert len(mw.memory_fragments_rag.summary_llm.calls) == 1
        # 画像落库 + 游标推进
        item = asyncio.run(runtime.store.aget(('long-term', 'user_profile',), 'u1'))
        profile = item.value['user_profile']
        assert profile['user_name'] == '张三'
        assert profile['user_hobby'] == ['读书']
        assert profile['last_summarized_id'] == 2

    def test_no_new_fragments_no_llm_call(self, build_middleware, runtime):
        """游标已到最新：不调用总结模型（增量设计的核心成本优化）"""
        mw, vec_store, _ = build_middleware(summary_result=UserProfile(user_name='张三'))
        docs = [make_fragment('u1', 1, age=10_000)]

        async def run():
            await vec_store.aadd_documents(docs)
            # 第一次：总结并推进游标
            await mw._memory_slice(_mature_messages(), user_id='u1', runtime=runtime)
            calls_after_first = len(mw.memory_fragments_rag.summary_llm.calls)
            # 第二次：无新片段，不应再调用
            await mw._memory_slice(_mature_messages(), user_id='u1', runtime=runtime)
            return calls_after_first, len(mw.memory_fragments_rag.summary_llm.calls)

        c1, c2 = asyncio.run(run())
        assert c1 == 1 and c2 == 1

    def test_not_mature_no_summary(self, build_middleware, runtime):
        """片段存在但未成熟（M < 阈值）：不总结"""
        mw, vec_store, _ = build_middleware(summary_result=UserProfile(user_name='张三'))
        docs = [make_fragment('u1', 1, age=10)]  # 10 秒太新

        async def run():
            await vec_store.aadd_documents(docs)
            return await mw._memory_slice(_mature_messages(age=10), user_id='u1', runtime=runtime)

        assert asyncio.run(run()) is False
        assert len(mw.memory_fragments_rag.summary_llm.calls) == 0

    def test_merge_keeps_old_profile_fields(self, build_middleware, runtime):
        """旧画像中"新总结未提及"的字段必须保留（增量合并）"""
        mw, vec_store, _ = build_middleware(
            summary_result=UserProfile(user_hobby=['读书']))
        docs = [make_fragment('u1', 1, age=10_000)]

        async def run():
            # 旧画像：已有姓名与爱好
            await runtime.store.aput(
                ('long-term', 'user_profile',), 'u1',
                {'user_profile': {'user_name': '旧名', 'user_hobby': ['游泳'],
                                  'last_summarized_id': 0}})
            await vec_store.aadd_documents(docs)
            await mw._memory_slice(_mature_messages(), user_id='u1', runtime=runtime)
            item = await runtime.store.aget(('long-term', 'user_profile',), 'u1')
            return item.value['user_profile']

        profile = asyncio.run(run())
        assert profile['user_name'] == '旧名'   # 新总结未提及 → 保留
        # 列表合并：新值优先，旧值补尾（与上游项目 merge_user_profile 行为一致）
        assert profile['user_hobby'] == ['读书', '游泳']

    def test_summary_failure_recovery(self, build_middleware, runtime):
        """总结 LLM 失败：返回 None、走 ErrorRecovery、画像不落库"""
        from helpers import FakeSummaryLLM, SpyRecovery
        from memory_middleware.models import UserProfile

        recovery = SpyRecovery()
        mw, vec_store, _ = build_middleware(recovery=recovery)
        mw.memory_fragments_rag.summary_llm = FakeSummaryLLM(
            result=UserProfile(), raise_error=RuntimeError('总结 LLM 挂了'))
        docs = [make_fragment('u1', 1, age=10_000)]

        async def run():
            await vec_store.aadd_documents(docs)
            return await mw._memory_slice(_mature_messages(), user_id='u1', runtime=runtime)

        assert asyncio.run(run()) is True  # 切片照常触发（总结失败不影响切片判断）
        assert len(recovery.summary_errors) == 1
        assert recovery.summary_errors[0][0] == 'u1'
        assert recovery.summary_errors[0][2] == 1  # last_summary_id
        # 画像未落库
        assert asyncio.run(runtime.store.aget(('long-term', 'user_profile',), 'u1')) is None
