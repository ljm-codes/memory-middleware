# -*- coding: utf-8 -*-
"""集成测试 fixtures：真实 DeepSeek 模型 + 用量采集。

默认跳过：必须显式 `python -m pytest -m integration` 才会执行；
且需要 .env 中 DEEPSEEK_API_KEY（缺失时优雅跳过）。
"""
import os
import pathlib
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # 包根目录

from dotenv import load_dotenv
# .env 位于主项目根目录（memory_middleware 的上级）
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / '.env')

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memory_middleware import (
    BalancedMultiDimensionMemory,
    FragmentsMemoryRAG,
    MemoryConfig,
    MemoryFragmentsAiSpliter,
    VocationParams,
)
from memory_middleware.models import SummaryMemoryAi, TimeMemoryFormulaParam, UserProfile
from memory_middleware.recovery import NullRecovery
from memory_middleware.storage.kv import MemoryKVStore
from memory_middleware.storage.vectors import HashEmbeddings, SQLiteVecStore


def pytest_collection_modifyitems(config, items):
    """默认不跑集成测试：只有 -m 显式包含 integration 才执行"""
    marker_opt = config.getoption("-m") or ""
    if "integration" not in marker_opt:
        skip = pytest.mark.skip(reason="集成测试需显式指定 -m integration（真实 LLM API）")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


if not os.environ.get('DEEPSEEK_API_KEY'):
    pytest.skip("未找到 DEEPSEEK_API_KEY，跳过集成测试（需要 .env 配置）", allow_module_level=True)


class UsageHandler(BaseCallbackHandler):
    """采集每次 LLM 调用的 usage（含 prompt_cache_hit/miss_tokens，见 memory_middleware.cost）。

    每条记录带 `tag`：main=主对话，split/summary/param=中间件内部三类调用——
    这样"总量里有多少花在中间件自身"可以直接从采集结果分组，不必靠输入长度猜。
    """

    def __init__(self):
        self.usage_list: list[dict] = []

    def on_llm_end(self, response, **kwargs):
        llm_output = getattr(response, 'llm_output', None) or {}
        usage = llm_output.get('token_usage') or {}
        if usage:
            tags = kwargs.get('tags') or []
            self.usage_list.append({
                'hit': int(usage.get('prompt_cache_hit_tokens', 0) or 0),
                'miss': int(usage.get('prompt_cache_miss_tokens', 0) or 0),
                'out': int(usage.get('completion_tokens', 0) or 0),
                'tag': next((t for t in tags if t in ('main', 'split', 'summary', 'param')), '?'),
            })

    def by_tag(self) -> dict:
        """按 tag 汇总：{tag: {'calls': n, 'in': tok, 'out': tok, 'cost': ¥}}"""
        from memory_middleware.cost import cost_cny
        agg: dict[str, dict] = {}
        for u in self.usage_list:
            a = agg.setdefault(u['tag'], {'calls': 0, 'in': 0, 'out': 0, 'cost': 0.0})
            a['calls'] += 1
            a['in'] += u['hit'] + u['miss']
            a['out'] += u['out']
            a['cost'] += cost_cny(u)
        return agg


def _make_model(usage_handler, schema=None, tag='main'):
    """真实 DeepSeek 模型（挂用量采集回调）；schema 非空时包结构化输出。

    注意顺序：with_structured_output 在前、with_config 包最外层——
    实测顺序相反时，结构化输出路径会丢 usage 回调（usage 采不全）。
    """
    llm = init_chat_model(
        'deepseek-v4-flash',
        temperature=1.3,
        extra_body={'thinking': {'type': 'disabled'}},
    )
    if schema is not None:
        llm = llm.with_structured_output(schema)
    return llm.with_config({'callbacks': [usage_handler], 'tags': [tag]})


@pytest.fixture(scope='module')
def usage_handler():
    return UsageHandler()


@pytest.fixture(scope='module')
def models(usage_handler):
    """四路真实模型：主对话（chat）/切分/总结/调参"""
    return {
        'chat': _make_model(usage_handler, tag='main'),
        'split': _make_model(usage_handler, SummaryMemoryAi, tag='split'),
        'summary': _make_model(usage_handler, UserProfile, tag='summary'),
        'param': _make_model(usage_handler, TimeMemoryFormulaParam, tag='param'),
    }


@pytest.fixture
def runtime():
    from langgraph.store.memory import InMemoryStore
    return SimpleNamespace(store=InMemoryStore(), context=SimpleNamespace(user_id='integ-user'))


@pytest.fixture
def build_middleware(tmp_path, models, runtime):
    """构造挂真实模型的中间件；返回 (middleware, chat_model, 词典句柄)"""
    def _build():
        vec_store = SQLiteVecStore(str(tmp_path / 'mem.db'), HashEmbeddings(dim=256))
        rec = NullRecovery()
        mw = BalancedMultiDimensionMemory(
            config=MemoryConfig(
                pattern='messages',
                trigger_threshold=3,
                vocation=VocationParams(
                    tau_m=60, c_m=0.5, tau=3600, c_t=0.5,
                    slice_value=0.0, long_term_value=0.0),
                rag_db_path=str(tmp_path / 'mem.db'),
                initial_prompt='你是一个智能客服，请基于用户画像与记忆片段简洁回答。',
                retrieve_k=6,
                top_k=3,
            ),
            summary_llm=models['param'],
            spliter=MemoryFragmentsAiSpliter(
                split_llm=models['split'], kv_store=MemoryKVStore(), recovery=rec),
            rag=FragmentsMemoryRAG(
                summary_llm=models['summary'], vector_store=vec_store, recovery=rec),
            recovery=rec,
        )
        return mw, models['chat'], vec_store
    return _build


@pytest.fixture
def run_conversation(build_middleware, runtime):
    """最小对话循环：逐轮驱动 abefore_model → awrap_model_call → aafter_model，
    等价于 langgraph 图内中间件行为（手动应用 Command 状态更新）。"""
    def _run(turns: list[str], max_replies: int = 8) -> dict:
        mw, chat_model, vec_store = build_middleware()
        state = {
            'messages': [],
            'new_msg_idx': 0,
            'system_prompt': mw.config.initial_prompt,
            'user_profile': '',
        }

        async def handler(request):
            msgs = ([request.system_message] if request.system_message else []) + request.messages
            out = await chat_model.ainvoke(msgs)
            state['messages'].append(AIMessage(
                content=out.content, additional_kwargs={'time': time.time()}))
            return ModelResponse(result=[out])

        async def _run_async():
            for i, text in enumerate(turns):
                state['messages'].append(HumanMessage(text, additional_kwargs={'time': time.time()}))
                await mw.abefore_model(state, runtime)
                request = ModelRequest(
                    model=object(), tools=[],
                    system_message=SystemMessage(content=state['system_prompt']),
                    messages=state['messages'], state=state, runtime=runtime,
                )
                result = await mw.awrap_model_call(request, handler)
                # 手动应用 ExtendedModelResponse 携带的 Command 状态更新
                if getattr(result, 'command', None) is not None:
                    state.update(result.command.update)
                await mw.aafter_model(state, runtime)
            return state

        import asyncio
        state = asyncio.run(_run_async())
        return {
            'state': state,
            'mw': mw,
            'vec_store': vec_store,
            'runtime': runtime,
            'replies': [m for m in state['messages'] if isinstance(m, AIMessage)],
        }

    return _run
