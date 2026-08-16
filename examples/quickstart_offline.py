# -*- coding: utf-8 -*-
"""离线快速开始：不依赖任何外部服务/API key，用脚本化假模型跑通全链路。

运行：python examples/quickstart_offline.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage

from memory_middleware import (
    BalancedMultiDimensionMemory,
    MemoryConfig,
    MemoryFragmentsAiSpliter,
    FragmentsMemoryRAG,
    VocationParams,
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


class ScriptedLLM:
    """脚本化假模型：按提示词内容返回固定的结构化输出"""

    def __init__(self, result):
        self.result = result

    async def ainvoke(self, messages):
        return self.result


def main():
    config = MemoryConfig(
        pattern='messages',
        trigger_threshold=3,  # 低阈值：几轮内即可看到检索注入
        vocation=VocationParams(
            tau_m=60, c_m=0.5, tau=3600, c_t=0.5,
            slice_value=0.0, long_term_value=0.0),
        rag_db_path='quickstart_offline.db',
        initial_prompt='你是智能客服，请基于记忆片段与用户画像回答。',
    )

    # ---- 注入假模型与离线存储（生产请替换为真实模型 + Redis + 真实嵌入） ----
    split_llm = ScriptedLLM(SummaryMemoryAi(
        theme_num=1,
        config=[SummaryMemoryFragmentsConfig(theme='个人信息', type=['identity', 'preference'], scope='0-2')],
        current_theme='个人信息',
    ))
    summary_llm = ScriptedLLM(UserProfile(user_name='张三', user_hobby=['读书', '游泳']))
    param_llm = ScriptedLLM(TimeMemoryFormulaParam())
    kv_store = MemoryKVStore()
    vec_store = SQLiteVecStore(config.rag_db_path, HashEmbeddings(dim=256))

    mw = BalancedMultiDimensionMemory(
        config=config,
        summary_llm=param_llm,
        spliter=MemoryFragmentsAiSpliter(split_llm=split_llm, kv_store=kv_store),
        rag=FragmentsMemoryRAG(summary_llm=summary_llm, vector_store=vec_store),
        recovery=NullRecovery(),
        kv_store=kv_store,
    )

    # ---- 最小运行时（真实 langgraph 中由 runtime 提供） ----
    from langgraph.store.memory import InMemoryStore
    from types import SimpleNamespace
    store = InMemoryStore()
    runtime = SimpleNamespace(store=store, context=SimpleNamespace(user_id='demo-user'))

    turns = ['我叫张三，喜欢读书和游泳', '我想买一台手机', '预算五千左右']
    state = {'messages': [], 'new_msg_idx': 0,
             'system_prompt': config.initial_prompt, 'user_profile': ''}

    import asyncio

    async def run():
        for text in turns:
            state['messages'].append(HumanMessage(text, additional_kwargs={'time': time.time()}))
            await mw.abefore_model(state, runtime)
            print(f"[切片/总结] 片段游标={await kv_store.aget('memory_fragments:demo-user')}")

        # 画像
        item = await store.aget(('long-term', 'user_profile',), 'demo-user')
        profile = item.value['user_profile'] if item else {}
        print(f"\n[用户画像] {profile}")
        print(f"[暂存片段] {vec_store.fetch_fragments_since('demo-user', 0)}")

    asyncio.run(run())
    print("\n离线全链路跑通 ✔（生产接入：真实模型 + RedisKVStore + 真实嵌入）")


if __name__ == '__main__':
    main()
