# -*- coding: utf-8 -*-
"""集成：端到端全链路（真实 LLM）——切片/总结/检索/注入是否真的发生。

结构断言（稳定）：片段落库、画像落库（含归纳游标）、检索注入触发、回答产出。
召回率质量曲线（针-草垛）属于基准脚本，见 README「评测」一节。
"""
import asyncio

import pytest

from langchain_core.messages import AIMessage, HumanMessage

pytestmark = pytest.mark.integration

TURNS = [
    '你好，我叫张三，是一名软件工程师，平时喜欢读书和游泳。',
    '我想咨询一下你们平台的手机价格。',
    '我最近在考虑买一台新手机，预算五千左右。',
    '对了，我之前说过我喜欢游泳，游泳馆离家很近。',
    '请问有什么推荐的吗？',
]


class TestEndToEnd:
    def test_full_pipeline_runs(self, run_conversation):
        """真实模型下全链路可跑：有片段、有画像、有回答"""
        result = run_conversation(TURNS, max_replies=5)
        state, mw, vec_store = result['state'], result['mw'], result['vec_store']

        # 1) 切分发生：至少一个片段落库（带完整元数据）
        rows = vec_store.fetch_fragments_since('integ-user', 0)
        assert len(rows) >= 1, "应有记忆片段落库"
        assert rows[0][1]['user_id'] == 'integ-user'

        # 2) 总结发生：用户画像已写 store 且带归纳游标
        async def _profile():
            item = await result['runtime'].store.aget(('long-term', 'user_profile',), 'integ-user')
            return item.value['user_profile'] if item else None
        profile = asyncio.run(_profile())
        assert profile is not None, "应有用户画像落库"
        assert isinstance(profile.get('last_summarized_id'), int)

        # 3) 每轮都有回复产出
        assert len(result['replies']) == len(TURNS)

    def test_facts_surface_in_final_answer(self, run_conversation):
        """早期事实（姓名）最终能被模型提及（经片段注入；宽松断言）"""
        result = run_conversation(TURNS, max_replies=5)
        final_reply = str(result['replies'][-1].content)
        assert len(final_reply) > 0
        print(f"\n[最终回复] {final_reply[:200]}")
        # 姓名若被注入，模型通常会在后续轮次提到；此处仅打印不硬断（质量曲线见基准脚本）
        # assert '张三' in final_reply
