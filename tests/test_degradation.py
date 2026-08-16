# -*- coding: utf-8 -*-
"""Layer 1：降级路径（检索失败 → 普通调用兜底，不崩、不二次调用）"""
import asyncio
import time

from langchain_core.messages import HumanMessage

from helpers import FakeParamLLM, make_fragment
from memory_middleware.models import TimeMemoryFormulaParam


class TestModelCallDegradation:
    def test_param_llm_failure_falls_back_to_handler(self, build_middleware, make_request, make_handler):
        """调参 LLM 失败：降级为普通调用，handler 用原始请求执行，不抛异常"""
        mw, vec_store, _ = build_middleware()
        mw.memory_spliter.dialogue_theme_by_user['u1'] = '手机'

        async def setup():
            await vec_store.aadd_documents([make_fragment('u1', 1, age=100)])
        asyncio.run(setup())

        mw._user_retrieve_state['u1'] = True
        mw._summary_llm = FakeParamLLM(
            result=TimeMemoryFormulaParam(), raise_error=RuntimeError('调参模型挂了'))

        msgs = [HumanMessage('问题', additional_kwargs={'time': time.time()})]
        captured = []
        request = make_request(msgs)

        async def run():
            result = await mw.awrap_model_call(request, make_handler(captured))
            return result, mw._user_retrieve_state['u1']

        result, retrieve_state = asyncio.run(run())
        assert captured  # handler 被调用
        assert result.result[0].content == 'ok'  # 兜底回复
        assert retrieve_state is False  # 检索状态被重置

    def test_retrieval_exception_inside_handler_path(self, build_middleware, make_request, make_handler):
        """检索链路任意异常：awrap_model_call 的 except 分支兜底"""
        mw, _, _ = build_middleware()
        mw._user_retrieve_state['u1'] = True
        mw.memory_spliter.dialogue_theme_by_user['u1'] = '手机'
        # 无片段 → _extract_fragments_by_time 内部提前返回 None（正常路径），
        # 这里强制制造异常：直接让 query 抛错
        original_query = mw.memory_fragments_rag.query_context_distance

        async def broken_query(*args, **kwargs):
            raise ConnectionError('向量库连接失败')

        mw.memory_fragments_rag.query_context_distance = broken_query
        msgs = [HumanMessage('问题', additional_kwargs={'time': time.time()})]
        captured = []
        request = make_request(msgs)

        async def run():
            result = await mw.awrap_model_call(request, make_handler(captured))
            return result

        result = asyncio.run(run())
        assert captured
        assert result.result[0].content == 'ok'
        mw.memory_fragments_rag.query_context_distance = original_query

    def test_extract_fragments_no_theme_returns_none(self, build_middleware, make_request, make_handler):
        """无当前主题：不触发调参 LLM，直接按普通调用处理"""
        mw, _, _ = build_middleware()
        mw._user_retrieve_state['u1'] = True  # 有检索状态但主题为空
        msgs = [HumanMessage('问题', additional_kwargs={'time': time.time()})]
        captured = []
        request = make_request(msgs)

        async def run():
            await mw.awrap_model_call(request, make_handler(captured))
            return len(mw._summary_llm.calls)

        assert asyncio.run(run()) == 0  # 调参模型未被调用
        assert len(captured) == 1
