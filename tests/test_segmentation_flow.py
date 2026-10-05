# -*- coding: utf-8 -*-
"""Layer 1：记忆切片流程（成熟度触发 → 主题切分 → 落库 → 游标续接）"""
import asyncio
import time

from langchain_core.messages import AIMessage, HumanMessage


def _mature_messages(n=4, age=2000):
    """构造最早消息已成熟（M(2000s)≥0.7，customer service 参数）的对话"""
    now = time.time()
    msgs = []
    for i in range(n):
        msg = HumanMessage(f'消息{i}', additional_kwargs={'time': now - (n - i) * 500})
        if i % 2 == 1:
            msg = AIMessage(f'回复{i}', additional_kwargs={'time': now - (n - i) * 500})
        msgs.append(msg)
    # 最早消息 age=2000s
    msgs[0].additional_kwargs['time'] = now - age
    return msgs


class TestSliceTrigger:
    def test_abefore_model_slices_and_stores(self, build_middleware, kv_store, runtime):
        """事件触发（窗口达预算 + 记忆成熟）后：切分模型被调用、片段落库、游标推进

        切分已并入主触发事件（见 middleware.abefore_model），只满足成熟度不放行切分，
        因此这里显式给一个会触发的窗口预算。
        """
        mw, vec_store, _ = build_middleware(pattern='messages', trigger_threshold=4)
        msgs = _mature_messages()

        async def run():
            result = await mw.abefore_model(
                {'messages': msgs, 'new_msg_idx': 0}, runtime)
            return result

        assert asyncio.run(run()) is None
        # 切分模型被调用
        assert len(mw.memory_spliter.split_llm.calls) == 1
        # 片段落库（fid=1），元数据完整
        rows = vec_store.fetch_fragments_since('u1', 0)
        assert len(rows) == 1
        text, meta, fid = rows[0]
        assert fid == 1
        assert meta['user_id'] == 'u1'
        assert meta['id'] == 1
        assert meta['theme'] == '测试主题'
        assert meta['strengthen_num'] == 0
        assert '消息0' in text  # [0, 1) → 左闭右开取第一条
        # KV 游标推进
        assert asyncio.run(kv_store.aget('memory_fragments:u1')) == '1'

    def test_not_mature_no_slice(self, build_middleware, runtime):
        """消息未成熟（Δ 很小）时不触发切分，模型不被调用"""
        mw, _, _ = build_middleware()
        now = time.time()
        msgs = [HumanMessage('新消息', additional_kwargs={'time': now - 1})]

        async def run():
            return await mw.abefore_model({'messages': msgs, 'new_msg_idx': 0}, runtime)

        assert asyncio.run(run()) is None
        assert len(mw.memory_spliter.split_llm.calls) == 0


class TestCursorContinuation:
    def test_atext_to_document_incremental(self, build_middleware, kv_store):
        """游标续接：第二次切分只处理游标之后的新消息，id 继续递增"""
        mw, vec_store, _ = build_middleware()
        msgs1 = _mature_messages(4)
        msgs2 = _mature_messages(6)

        async def run():
            docs1 = await mw.memory_spliter.atext_to_document(text=msgs1, user='u1')
            assert docs1 is not None and len(docs1) == 1
            await mw.memory_fragments_rag.add(docs1)
            # 游标已推进到 1：第二轮只切游标后的新消息
            docs2 = await mw.memory_spliter.atext_to_document(text=msgs2, user='u1')
            return docs2

        docs2 = asyncio.run(run())
        assert docs2 is not None and len(docs2) == 1
        assert docs2[0].id == 'u1-2'
        assert docs2[0].metadata['id'] == 2
        assert asyncio.run(kv_store.aget('memory_fragments:u1')) == '2'

    def test_fresh_kv_continues_ids_from_store(self, build_middleware):
        """KV 计数器丢失（进程重启 / Redis 无持久化）后：从向量库持久高水位接着编，不撞 UNIQUE 约束"""
        from memory_middleware.storage.kv import MemoryKVStore

        mw, vec_store, _ = build_middleware()
        msgs1 = _mature_messages(4)
        msgs2 = _mature_messages(6)

        async def run():
            docs1 = await mw.memory_spliter.atext_to_document(text=msgs1, user='u1')
            await mw.memory_fragments_rag.add(docs1)
            # 模拟进程重启：KV 换新、进程内状态清空，只剩向量库里的持久高水位
            mw.memory_spliter._kv_store = MemoryKVStore()
            mw.memory_spliter._id_reconciled.clear()
            mw.memory_spliter._id_high_water.clear()
            docs2 = await mw.memory_spliter.atext_to_document(text=msgs2, user='u1')
            await mw.memory_fragments_rag.add(docs2)   # 修复前这里抛 UNIQUE constraint failed
            return docs1, docs2

        docs1, docs2 = asyncio.run(run())
        assert [d.id for d in docs1] == ['u1-1']
        assert docs2 is not None and [d.id for d in docs2] == ['u1-2']
        # 消息偏移也从持久高水位自愈：第二次只切游标之后的 2 条新消息
        assert asyncio.run(mw.memory_spliter._kv_store.aget('memory_msg_offset:u1')) == '6'

    def test_concurrent_slices_get_disjoint_ids(self, build_middleware):
        """并发切片：INCRBY 原子预留保证两批片段拿到的 id 段不重叠（多进程下同理）"""
        from memory_middleware.models import SummaryMemoryAi, SummaryMemoryFragmentsConfig

        split_result = SummaryMemoryAi(
            theme_num=2,
            config=[SummaryMemoryFragmentsConfig(theme='主题A', type=['chat'], start_idx=0, end_idx=2),
                    SummaryMemoryFragmentsConfig(theme='主题B', type=['chat'], start_idx=2, end_idx=4)],
            current_theme='主题A')
        mw, vec_store, _ = build_middleware(split_result=split_result)
        msgs = _mature_messages(4)

        async def run():
            return await asyncio.gather(
                mw.memory_spliter.atext_to_document(text=msgs, user='u1'),
                mw.memory_spliter.atext_to_document(text=msgs, user='u1'),
            )

        a, b = asyncio.run(run())
        ids = [d.metadata['id'] for d in (a or [])] + [d.metadata['id'] for d in (b or [])]
        assert ids, "至少一批应产出片段"
        assert len(ids) == len(set(ids)), f"id 段重叠: {ids}"   # 修复目标：不重叠
        assert min(ids) >= 1

    def test_multi_fragment_slice_message_offset(self, build_middleware, kv_store):
        """双游标回归：一次切片产出多片段时，消息偏移必须独立于片段数推进。

        此前片段数≠消息数会让下一轮切分窗口错位（片段丢失/重复切片）。
        """
        from memory_middleware.models import SummaryMemoryAi, SummaryMemoryFragmentsConfig

        split_result = SummaryMemoryAi(
            theme_num=2,
            config=[SummaryMemoryFragmentsConfig(theme='主题A', type=['chat'], start_idx=0, end_idx=2),
                    SummaryMemoryFragmentsConfig(theme='主题B', type=['chat'], start_idx=2, end_idx=4)],
            current_theme='主题A')
        mw, vec_store, _ = build_middleware(split_result=split_result)
        msgs1 = _mature_messages(4)
        msgs2 = _mature_messages(6)

        async def run():
            docs1 = await mw.memory_spliter.atext_to_document(text=msgs1, user='u1')
            await mw.memory_fragments_rag.add(docs1)
            docs2 = await mw.memory_spliter.atext_to_document(text=msgs2, user='u1')
            return docs1, docs2

        docs1, docs2 = asyncio.run(run())
        # 第一次切片：4 条消息 → 2 个片段（片段数 2 ≠ 消息数 4）
        assert [d.metadata['id'] for d in docs1] == [1, 2]
        # 第二次切片：消息偏移=4 → 窗口是 msgs2[4:]（第 4、5 条），内容不能错位
        assert len(docs2) == 1
        assert docs2[0].metadata['id'] == 3
        assert '消息4' in docs2[0].page_content
        # 两个游标分离
        assert asyncio.run(kv_store.aget('memory_msg_offset:u1')) == '6'
        assert asyncio.run(kv_store.aget('memory_fragments:u1')) == '3'

    def test_new_session_resets_msg_offset(self, build_middleware, kv_store):
        """跨会话回归：新会话消息列表比游标短时偏移归零（否则切分窗口为空、片段丢失）"""
        mw, vec_store, _ = build_middleware()
        msgs1 = _mature_messages(4)  # 会话1：4 条消息 → offset=4
        msgs2 = _mature_messages(2)  # 会话2：新列表只有 2 条 < 4 → 应重置偏移

        async def run():
            docs1 = await mw.memory_spliter.atext_to_document(text=msgs1, user='u1')
            await mw.memory_fragments_rag.add(docs1)
            docs2 = await mw.memory_spliter.atext_to_document(text=msgs2, user='u1')
            return docs1, docs2

        docs1, docs2 = asyncio.run(run())
        assert len(docs1) == 1 and docs1[0].metadata['id'] == 1
        # 会话2 切的是自己的第 0 条消息（而非空窗口）
        assert len(docs2) == 1
        assert docs2[0].metadata['id'] == 2  # 片段 id 游标继续递增
        assert '消息0' in docs2[0].page_content
        assert asyncio.run(kv_store.aget('memory_msg_offset:u1')) == '2'

    def test_split_failure_returns_none(self, build_middleware, kv_store):
        """切分 LLM 失败：返回 None、走 ErrorRecovery、不落库、不抛异常"""
        from helpers import FakeSplitLLM, SpyRecovery
        from memory_middleware.models import SummaryMemoryAi
        from memory_middleware.spliter import MemoryFragmentsAiSpliter

        recovery = SpyRecovery()
        mw, vec_store, _ = build_middleware(recovery=recovery)
        # 替换成失败模型
        mw.memory_spliter.split_llm = FakeSplitLLM(
            result=SummaryMemoryAi(), raise_error=RuntimeError('LLM 挂了'))

        async def run():
            docs = await mw.memory_spliter.atext_to_document(text=_mature_messages(4), user='u1')
            return docs

        assert asyncio.run(run()) is None
        assert len(recovery.split_errors) == 1
        assert recovery.split_errors[0][0] == 'u1'
        assert isinstance(recovery.split_errors[0][2], RuntimeError)
        assert asyncio.run(kv_store.aget('memory_fragments:u1')) is None

    def test_split_none_result_returns_none(self, build_middleware, kv_store):
        """结构化输出解析失败：模型**返回 None 而不抛异常**——同样按失败降级，不崩在 .current_theme

        2026-10-05 基准实测踩到：None 落在 try/except 之外 → AttributeError 掀掉整轮对话。
        """
        from helpers import FakeSplitLLM, SpyRecovery

        recovery = SpyRecovery()
        mw, vec_store, _ = build_middleware(recovery=recovery)
        mw.memory_spliter.split_llm = FakeSplitLLM(None)   # 不抛异常，只返回 None

        docs = asyncio.run(mw.memory_spliter.atext_to_document(text=_mature_messages(4), user='u1'))

        assert docs is None
        assert len(recovery.split_errors) == 1
        assert recovery.split_errors[0][0] == 'u1'
        assert '空结果' in str(recovery.split_errors[0][2])
        assert asyncio.run(kv_store.aget('memory_fragments:u1')) is None
        assert not mw.memory_spliter.dialogue_theme_by_user.get('u1'), '失败时不该写入主题'
