"""暂存库检索与增量画像总结（FragmentsMemoryRAG）。

与上游项目 fireflymall-ai-customer-service
（https://github.com/lijia-ming/fireflymall-ai-customer-service）
memory_rag.py 的 FragmentsMemoryRAG 一致，差异：
向量库 → VectorStore 注入；RabbitMQ 保底 → ErrorRecovery 注入；
模型名参数 → 构造时注入的 model_factory。
"""
import logging
import time
from asyncio import Lock
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.retry import ExponentialJitterParams

from .formula import maturity
from .models import UserProfile
from .recovery import ErrorRecovery, NullRecovery

logger = logging.getLogger(__name__)

# 与上游项目 memory_rag.py 的总结提示词内容一致（仅去除源码缩进）
DEFAULT_SUMMARY_PROMPT = """\
你现在是一个专业的总结专家，你需要根据用户的新增对话片段来总结出相对应的用户画像。
注意：
    - 只总结片段中明确出现的用户信息，不要猜测。
    - 片段中没有涉及的字段，保持默认值或空值即可（合并时会保留旧画像中的对应信息）。
"""


class FragmentsMemoryRAG:
    """暂存库（向量 RAG）的检索、巩固与增量总结"""

    def __init__(
            self,
            summary_llm: Optional[Any] = None,
            vector_store: Optional[Any] = None,
            recovery: Optional[ErrorRecovery] = None,
            prompt: str = DEFAULT_SUMMARY_PROMPT,
            model_name: str = 'deepseek-v4-flash',
            model_kwargs: Optional[dict] = None,
            model_factory: Optional[Callable[[], Any]] = None,
    ):
        # 注入的模型需支持结构化输出（ainvoke 返回 UserProfile）；None 时按工厂惰性创建
        self.summary_llm = summary_llm
        self.vector_store = vector_store
        self._recovery = recovery or NullRecovery()
        self.prompt = prompt
        self._model_name = model_name
        self._model_kwargs = model_kwargs or {}
        self._model_factory = model_factory
        self._init_lock = Lock()

    def _create_model(self):
        if self._model_factory is not None:
            return self._model_factory()
        return init_chat_model(self._model_name, **self._model_kwargs)

    async def create_summary_agent(self):
        if self.summary_llm is None:
            async with self._init_lock:
                if self.summary_llm is None:
                    _init_llm = self._create_model()
                    self.summary_llm = _init_llm.with_structured_output(UserProfile).with_retry(
                        retry_if_exception_type=(Exception,),
                        wait_exponential_jitter=True,
                        stop_after_attempt=3,
                        exponential_jitter_params=ExponentialJitterParams(
                            initial=2.0,
                            max=10,
                            exp_base=2,
                            jitter=1.0,
                        ),
                    )

    async def add(self, memory: Optional[List[Document]]) -> None:
        if not memory:
            logger.error("记忆片段不能为空")
            return None
        await self.vector_store.aadd_documents(memory)

    async def query_context_distance(
            self,
            query: str,
            k: Optional[int] = None,
            user_id: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """检索与 query 最相似的记忆片段，按 user_id 过滤，避免跨用户泄露。

        k=None 表示不预筛（该用户全部片段都作为候选）——推荐做法，理由见 MemoryConfig.retrieve_k。
        """
        doc_scores = await self.vector_store.asimilarity_search_with_score(
            query=query, k=k, filter={'user_id': user_id})
        if not doc_scores:
            return list()
        # sqlite-vec 返回的 score 为距离，cosine 距离 = 1 - 余弦相似度
        return [(doc, 1 - score) for doc, score in doc_scores]

    async def update_config_strengthen(self, ids: List[str], documents: List[Document]):
        """RAG 不支持直接更新：先删再添，仅用于更新片段的巩固次数"""
        await self.vector_store.adelete(ids=ids)
        await self.vector_store.aadd_documents(documents)

    async def long_memory_summary_by_time(
            self,
            m_t: float,
            m_c: float,
            long_term_value: float,
            user_id: str,
            last_summary_id: int = 0,
    ) -> Optional[Tuple[dict, int]]:
        """对归纳游标之后、且最早片段已成熟的新片段做增量总结。

        返回：(增量画像 dict, 新游标)；无新片段/未成熟/LLM 失败时返回 None。
        """
        await self.create_summary_agent()
        result = self._select_fragments_by_time(m_t, m_c, long_term_value, user_id, last_summary_id)
        if not result:
            return None

        text, new_cursor = result
        try:
            profile = await self.summary_llm.ainvoke([
                SystemMessage(content=self.prompt),
                HumanMessage(content=f"目前的新增对话片段如下：\n{text}"),
            ])
        except Exception as e:
            logger.error(f"在进行总结分片时，出现异常：{e}")
            await self._recovery.on_summary_error(user_id, text, new_cursor, e)
            # 返回 None 而非保底：UserProfile 字段复杂且涉及隐私，无法可靠兜底
            profile = None
        if not profile:
            return None
        profile_json = profile.model_dump(mode='json')
        profile_json['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"用户画像（增量总结）：{profile_json}")
        return profile_json, new_cursor

    def _select_fragments_by_time(
            self, m_t: float, m_c: float, long_term_value: float,
            user_id: str, last_summary_id: int,
    ) -> Optional[Tuple[str, int]]:
        """游标之后的新片段中，最早片段已成熟（M ≥ long_term_value）才触发总结"""
        rows = self.vector_store.fetch_fragments_since(user_id, last_summary_id)
        if not rows:
            return None
        first_text, first_meta, _ = rows[0]
        memory_time = first_meta.get('time', time.time())
        m = maturity(time.time() - memory_time, m_t, m_c)
        if m < long_term_value:
            return None
        text_list = [t for t, _, _ in rows]
        return '\n'.join(text_list), rows[-1][2]
