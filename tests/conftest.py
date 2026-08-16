# -*- coding: utf-8 -*-
"""共享 fixtures：假 LLM、确定性嵌入、内存 KV、临时向量库、假 runtime。

离线原则：本文件不导入任何外部服务（无 Redis/RabbitMQ/DashScope/DeepSeek）。
"""
import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # 保证包可导入
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))     # 保证 helpers 可导入

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from helpers import FakeParamLLM, FakeSplitLLM, FakeSummaryLLM, SpyRecovery, make_fragment

from memory_middleware import (
    BalancedMultiDimensionMemory,
    FragmentsMemoryRAG,
    MemoryConfig,
    MemoryFragmentsAiSpliter,
)
from memory_middleware.models import (
    SummaryMemoryAi,
    SummaryMemoryFragmentsConfig,
    TimeMemoryFormulaParam,
    UserProfile,
)
from memory_middleware.recovery import NullRecovery
from memory_middleware.storage.kv import MemoryKVStore
from memory_middleware.storage.vectors import HashEmbeddings, SQLiteVecStore


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kv_store():
    return MemoryKVStore()


@pytest.fixture
def embeddings():
    return HashEmbeddings(dim=16)


@pytest.fixture
def vec_store(tmp_path, embeddings):
    store = SQLiteVecStore(str(tmp_path / 'test.db'), embeddings)
    yield store
    store.close()


@pytest.fixture
def runtime():
    """假 langgraph runtime：InMemoryStore + user_id 上下文"""
    from langgraph.store.memory import InMemoryStore
    return SimpleNamespace(
        store=InMemoryStore(),
        context=SimpleNamespace(user_id='u1'),
    )


@pytest.fixture
def runtime_u2(runtime):
    """第二个用户（共享同一个 store，用于隔离测试）"""
    return SimpleNamespace(store=runtime.store, context=SimpleNamespace(user_id='u2'))


@pytest.fixture
def make_request(runtime):
    def _make(messages, state=None, system='SYS'):
        # ModelRequest 是普通类（非 pydantic）：runtime/state 可传任意对象
        return ModelRequest(
            model=object(),
            tools=[],
            system_message=SystemMessage(content=system),
            messages=messages,
            state=state or {
                'messages': messages,
                'new_msg_idx': 0,
                'system_prompt': system,
                'user_profile': '',
            },
            runtime=runtime,
        )
    return _make


@pytest.fixture
def make_handler():
    def _make(capture=None):
        async def handler(request):
            if capture is not None:
                capture.append(request)
            return ModelResponse(result=[AIMessage(content='ok')])
        return handler
    return _make


@pytest.fixture
def build_middleware(tmp_path, kv_store, embeddings):
    """构造注入全假的中间件；返回 (middleware, vec_store, recovery)"""
    def _build(
            split_result=None,
            summary_result=None,
            param_result=None,
            recovery=None,
            initial_prompt='SYS',
            **cfg_overrides,
    ):
        config = MemoryConfig(
            rag_db_path=str(tmp_path / 'mem.db'),
            initial_prompt=initial_prompt,
            **cfg_overrides,
        )
        vec_store = SQLiteVecStore(config.rag_db_path, embeddings, table=config.rag_table)
        rec = recovery or NullRecovery()
        mw = BalancedMultiDimensionMemory(
            config=config,
            summary_llm=FakeParamLLM(param_result or TimeMemoryFormulaParam()),
            spliter=MemoryFragmentsAiSpliter(
                split_llm=FakeSplitLLM(
                    split_result or SummaryMemoryAi(
                        theme_num=1,
                        config=[SummaryMemoryFragmentsConfig(theme='测试主题', type=['chat'], scope='0-1')],
                        current_theme='测试主题',
                    )),
                kv_store=kv_store,
                recovery=rec,
            ),
            rag=FragmentsMemoryRAG(
                summary_llm=FakeSummaryLLM(summary_result or UserProfile()),
                vector_store=vec_store,
                recovery=rec,
            ),
            kv_store=kv_store,
            recovery=rec,
        )
        return mw, vec_store, rec
    return _build


