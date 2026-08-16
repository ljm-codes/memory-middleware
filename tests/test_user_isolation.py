# -*- coding: utf-8 -*-
"""Layer 1：跨用户隔离（检索过滤 + 画像隔离 + 并发锁）"""
import asyncio
import time

from langchain_core.messages import HumanMessage

from helpers import make_fragment


class TestRetrievalIsolation:
    def test_query_filtered_by_user(self, build_middleware, vec_store):
        """检索必须按 user_id 过滤：u1 查不到 u2 的片段"""
        mw, vec_store, _ = build_middleware()

        async def run():
            await vec_store.aadd_documents([
                make_fragment('u1', 1, content='u1 的片段'),
                make_fragment('u2', 1, content='u2 的片段'),
            ])
            docs = await mw.memory_fragments_rag.query_context_distance(
                '片段', k=5, user_id='u1')
            return docs

        docs = asyncio.run(run())
        assert len(docs) == 1
        assert docs[0][0].metadata['user_id'] == 'u1'

    def test_injection_only_own_fragments(self, build_middleware, make_request, make_handler):
        """检索注入只带本用户片段"""
        mw, vec_store, _ = build_middleware()
        mw.memory_spliter.dialogue_theme_by_user['u1'] = '手机'

        async def setup():
            await vec_store.aadd_documents([
                make_fragment('u1', 1, content='u1 的手机片段', theme='手机'),
                make_fragment('u2', 1, content='u2 的物流片段', theme='物流'),
            ])
        asyncio.run(setup())
        mw._user_retrieve_state['u1'] = True
        msgs = [HumanMessage('问题', additional_kwargs={'time': time.time()})]
        captured = []
        request = make_request(msgs)

        async def run():
            await mw.awrap_model_call(request, make_handler(captured))
            return captured[0].system_message.content

        sys_content = asyncio.run(run())
        assert 'u1 的手机片段' in sys_content
        assert 'u2 的物流片段' not in sys_content


class TestProfileIsolation:
    def test_profiles_separate_per_user(self, build_middleware, runtime, runtime_u2):
        """画像按用户独立存储"""
        mw, _, _ = build_middleware()

        async def run():
            await runtime.store.aput(
                ('long-term', 'user_profile',), 'u1',
                {'user_profile': {'user_name': 'u1 用户', 'last_summarized_id': 1}})
            await runtime.store.aput(
                ('long-term', 'user_profile',), 'u2',
                {'user_profile': {'user_name': 'u2 用户', 'last_summarized_id': 3}})
            p1 = await runtime.store.aget(('long-term', 'user_profile',), 'u1')
            p2 = await runtime.store.aget(('long-term', 'user_profile',), 'u2')
            return p1.value['user_profile'], p2.value['user_profile']

        p1, p2 = asyncio.run(run())
        assert p1['user_name'] == 'u1 用户' and p1['last_summarized_id'] == 1
        assert p2['user_name'] == 'u2 用户' and p2['last_summarized_id'] == 3


class TestConcurrentUsers:
    def test_concurrent_abefore_model_no_cross_talk(self, build_middleware, runtime, runtime_u2, kv_store):
        """两个用户并发触发切片：各自游标互不干扰"""
        mw, vec_store, _ = build_middleware()
        now = time.time()
        msgs_u1 = [HumanMessage('u1 的消息', additional_kwargs={'time': now - 2000})]
        msgs_u2 = [HumanMessage('u2 的消息', additional_kwargs={'time': now - 2000})]

        async def run():
            await asyncio.gather(
                mw.abefore_model({'messages': msgs_u1, 'new_msg_idx': 0}, runtime),
                mw.abefore_model({'messages': msgs_u2, 'new_msg_idx': 0}, runtime_u2),
            )
            c1 = await kv_store.aget('memory_fragments:u1')
            c2 = await kv_store.aget('memory_fragments:u2')
            return c1, c2

        c1, c2 = asyncio.run(run())
        assert c1 == '1'
        assert c2 == '1'
        # 两用户的片段独立落库
        rows_u1 = vec_store.fetch_fragments_since('u1', 0)
        rows_u2 = vec_store.fetch_fragments_since('u2', 0)
        assert len(rows_u1) == 1 and len(rows_u2) == 1
        assert rows_u1[0][1]['user_id'] == 'u1'
        assert rows_u2[0][1]['user_id'] == 'u2'
