# -*- coding: utf-8 -*-
"""Layer 1：检索注入（提示词重写 + new_msg_idx 游标 + 强化落库）"""
import asyncio
import time

from langchain_core.messages import AIMessage, HumanMessage

from helpers import make_fragment


def _setup_retrievable(mw, vec_store, user_id='u1', theme='手机'):
    """预置该用户的可检索片段 + 当前对话主题"""
    mw.memory_spliter.dialogue_theme_by_user[user_id] = theme

    async def _add():
        await vec_store.aadd_documents([make_fragment(user_id, 1, theme=theme, age=100)])
    asyncio.run(_add())


class TestRetrievePath:
    def test_prompt_rewritten_with_fragments(self, build_middleware, make_request, make_handler):
        """检索触发轮：系统提示词被重写（核心+画像+片段），消息截断到最后一个人类消息"""
        mw, vec_store, _ = build_middleware(initial_prompt='核心提示词')
        _setup_retrievable(mw, vec_store)
        mw._user_retrieve_state['u1'] = True

        msgs = [
            HumanMessage('第一条', additional_kwargs={'time': time.time() - 10}),
            AIMessage('回复一', additional_kwargs={'time': time.time() - 9}),
            HumanMessage('当前问题', additional_kwargs={'time': time.time()}),
        ]
        captured = []
        request = make_request(msgs, state={
            'messages': msgs, 'new_msg_idx': 0,
            'system_prompt': '核心提示词', 'user_profile': '画像：喜欢手机',
        })

        async def run():
            result = await mw.awrap_model_call(request, make_handler(captured))
            return result

        result = asyncio.run(run())
        assert captured, "handler 必须被调用"
        new_request = captured[0]
        # 提示词重写：核心 + 画像 + 片段
        sys_content = new_request.system_message.content
        assert '核心提示词' in sys_content
        assert '画像：喜欢手机' in sys_content
        assert '用户喜欢讨论手机话题' in sys_content  # 片段内容被注入
        # 消息只保留最后一个人类消息（截断历史）
        assert len(new_request.messages) == 1
        assert new_request.messages[0].content == '当前问题'
        # 返回 ExtendedModelResponse 携带状态更新
        assert result.command is not None
        update = result.command.update
        assert update['new_msg_idx'] == 2
        assert '画像：喜欢手机' in update['system_prompt']

    def test_strengthen_persisted_after_success(self, build_middleware, make_request, make_handler):
        """模型调用成功后：片段巩固次数 +1 并落库"""
        mw, vec_store, _ = build_middleware()
        _setup_retrievable(mw, vec_store)
        mw._user_retrieve_state['u1'] = True
        msgs = [HumanMessage('问题', additional_kwargs={'time': time.time()})]
        request = make_request(msgs)

        async def run():
            await mw.awrap_model_call(request, make_handler())
            rows = vec_store.fetch_fragments_since('u1', 0)
            return rows

        rows = asyncio.run(run())
        assert rows[0][1]['strengthen_num'] == 1  # 0 → 1

    def test_no_fragments_resets_retrieve_state(self, build_middleware, make_request, make_handler):
        """库里无片段：不注入、重置检索状态、按普通调用处理"""
        mw, vec_store, _ = build_middleware()
        mw.memory_spliter.dialogue_theme_by_user['u1'] = '手机'  # 有主题但库里没片段
        mw._user_retrieve_state['u1'] = True
        msgs = [HumanMessage('问题', additional_kwargs={'time': time.time()})]
        request = make_request(msgs, state={
            'messages': msgs, 'new_msg_idx': 0,
            'system_prompt': 'SYS', 'user_profile': ''})

        async def run():
            await mw.awrap_model_call(request, make_handler())
            return mw._user_retrieve_state['u1']

        assert asyncio.run(run()) is False
        # 有主题但无片段 → 调参模型未被调用（提前 return None）
        assert len(mw._summary_llm.calls) == 0


class TestNormalPath:
    def test_non_retrieve_override(self, build_middleware, make_request, make_handler):
        """非检索轮：system_prompt 与 new_msg_idx 生效的普通调用"""
        mw, _, _ = build_middleware()
        msgs = [HumanMessage('一', additional_kwargs={'time': time.time() - 5}),
                HumanMessage('二', additional_kwargs={'time': time.time()})]
        captured = []
        request = make_request(msgs, state={
            'messages': msgs, 'new_msg_idx': 1,
            'system_prompt': '系统提示', 'user_profile': ''})

        async def run():
            await mw.awrap_model_call(request, make_handler(captured))
            return captured[0]

        new_request = asyncio.run(run())
        assert new_request.system_message.content == '系统提示'
        assert [m.content for m in new_request.messages] == ['二']
