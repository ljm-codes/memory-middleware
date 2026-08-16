# -*- coding: utf-8 -*-
"""真实 API 快速开始：DeepSeek 主模型 + 全链路记忆（约 10 次调用，几分钱）。

运行前提：主项目根目录 .env 已配置 DEEPSEEK_API_KEY（或已导出环境变量）。
运行：python examples/quickstart_real.py
"""
import asyncio
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).resolve().parents[2] / '.env')

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore

from memory_middleware import (
    BalancedMultiDimensionMemory,
    FragmentsMemoryRAG,
    MemoryConfig,
    MemoryFragmentsAiSpliter,
    VocationParams,
)
from memory_middleware.cost import cost_cny
from memory_middleware.models import SummaryMemoryAi, TimeMemoryFormulaParam, UserProfile
from memory_middleware.recovery import NullRecovery
from memory_middleware.storage.kv import MemoryKVStore
from memory_middleware.storage.vectors import HashEmbeddings, SQLiteVecStore


class UsageHandler(BaseCallbackHandler):
    def __init__(self):
        self.usages = []

    def on_llm_end(self, response, **kwargs):
        usage = (getattr(response, 'llm_output', None) or {}).get('token_usage') or {}
        if usage:
            self.usages.append({
                'hit': int(usage.get('prompt_cache_hit_tokens', 0) or 0),
                'miss': int(usage.get('prompt_cache_miss_tokens', 0) or 0),
                'out': int(usage.get('completion_tokens', 0) or 0),
            })


def make_model(usage_handler, schema=None):
    llm = init_chat_model(
        'deepseek-v4-flash', temperature=1.3,
        extra_body={'thinking': {'type': 'disabled'}})
    if schema is not None:
        llm = llm.with_structured_output(schema)
    return llm.with_config({'callbacks': [usage_handler]})


async def main():
    usage = UsageHandler()
    chat_model = make_model(usage)

    config = MemoryConfig(
        pattern='messages',
        trigger_threshold=3,
        vocation=VocationParams(
            tau_m=60, c_m=0.5, tau=3600, c_t=0.5,
            slice_value=0.0, long_term_value=0.0),
        rag_db_path='quickstart_real.db',
        initial_prompt='你是一个智能客服，请基于用户画像与记忆片段简洁回答。',
    )
    vec_store = SQLiteVecStore(config.rag_db_path, HashEmbeddings(dim=256))
    mw = BalancedMultiDimensionMemory(
        config=config,
        summary_llm=make_model(usage, TimeMemoryFormulaParam),
        spliter=MemoryFragmentsAiSpliter(
            split_llm=make_model(usage, SummaryMemoryAi), kv_store=MemoryKVStore()),
        rag=FragmentsMemoryRAG(
            summary_llm=make_model(usage, UserProfile), vector_store=vec_store),
        recovery=NullRecovery(),
    )

    from types import SimpleNamespace
    runtime = SimpleNamespace(
        store=InMemoryStore(), context=SimpleNamespace(user_id='real-demo'))

    turns = [
        '你好，我叫王五，是一名教师，平时喜欢摄影。',
        '我想了解一下你们平台的相机。',
        '预算大概八千左右，有什么推荐？',
    ]
    state = {'messages': [], 'new_msg_idx': 0,
             'system_prompt': config.initial_prompt, 'user_profile': ''}

    async def handler(request):
        msgs = ([request.system_message] if request.system_message else []) + request.messages
        out = await chat_model.ainvoke(msgs)
        state['messages'].append(AIMessage(
            content=out.content, additional_kwargs={'time': time.time()}))
        return ModelResponse(result=[out])

    for text in turns:
        state['messages'].append(HumanMessage(text, additional_kwargs={'time': time.time()}))
        await mw.abefore_model(state, runtime)
        request = ModelRequest(
            model=object(), tools=[],
            system_message=SystemMessage(content=state['system_prompt']),
            messages=state['messages'], state=state, runtime=runtime)
        result = await mw.awrap_model_call(request, handler)
        if getattr(result, 'command', None) is not None:
            state.update(result.command.update)
        await mw.aafter_model(state, runtime)
        print(f"客服：{out.content[:80]}...")

    item = await runtime.store.aget(('long-term', 'user_profile',), 'real-demo')
    print(f"\n[用户画像] {item.value['user_profile'] if item else {}}")
    total = sum(cost_cny(u) for u in usage.usages)
    print(f"[费用] {len(usage.usages)} 次调用，输入 "
          f"{sum(u['hit'] + u['miss'] for u in usage.usages)} tok，总费用 ¥{total:.6f}")


if __name__ == '__main__':
    asyncio.run(main())
