"""片段切分（MemoryFragmentsAiSpliter）：把工作记忆中的消息按主题切成暂存库片段。

与上游项目 fireflymall-ai-customer-service
（https://github.com/lijia-ming/fireflymall-ai-customer-service）
memory_rag.py 的 MemoryFragmentsAiSpliter 一致，差异：
Redis 游标 → KVStore 注入；RabbitMQ 保底 → ErrorRecovery 注入。
"""
import logging
import time
from typing import Any, Callable, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables.retry import ExponentialJitterParams

from .models import MemoryFragments, MemoryFragmentsMetadata, SummaryMemoryAi
from .recovery import ErrorRecovery, NullRecovery
from .storage.kv import KVStore, MemoryKVStore
from .storage.vectors import FragmentIdSource

logger = logging.getLogger(__name__)

# 与上游项目 memory_rag.py 的切分提示词内容一致（仅去除源码缩进）
DEFAULT_SPLIT_PROMPT = """\
你现在是一个专业的摘要模型，你的任务是将一个长对话文本按照不同主题进行切分区域，
并输出每个片段的主题、可能出现的类型、片段的内容等。
注意：
    - 输出的主题，你自己决定。（但不宜过长，尽量在30字以内）
    - 只支持的类型（绝对不允许出现除了以下类型的其他类型）（一个主题允许存在多个类型）：
        - identity: 包含用户身份信息（姓名、职业等）
        - preference: 包含用户偏好、习惯
        - decision: 包含关键决策、承诺
        - fact: 包含客观事实、技术细节
        - episode: 描述某个经历或任务过程
        - chat: 一般性闲聊，长期价值低
    - 类型判定只看内容本身，不看语气：用户陈述的个人信息 / 偏好 / 约束（姓名、职业、生日、机型、
      订单号、卡尾号、作息、饮食、住址等）一律标 identity / preference / fact，
      绝不能因为夹在闲聊里就标成 chat —— chat 只用于确实没有信息量的寒暄。
    - 片段范围：必须按照主题划分区域，且不允许跨主题。
      用 start_idx 与 end_idx 两个整数表示，区间为左闭右开 [start_idx, end_idx)，即覆盖索引 start_idx ~ end_idx-1 的消息；
      如 start_idx=0、end_idx=50 表示索引 0~49 的消息；要表达单条消息，用 start_idx=0、end_idx=1
    - 每次输出要保证 主题总数 与 记忆片段配置列表 的长度一致。
    - 一个消息会对应相对应的类型并且包含消息的索引号（从0开始），
      若消息过长，将会截取字符，一般发生在tool类型的消息中，
        如：
        对应的消息索引0，ai:你好
        对应的消息索引1，user:你好
        对应的消息索引2，tool:你好
        ....
"""


class MemoryFragmentsAiSpliter:
    """对话消息 → 按主题切分的记忆片段（含主题/类型/时间/巩固次数元数据）"""

    def __init__(
            self,
            split_llm: Optional[Any] = None,
            kv_store: Optional[KVStore] = None,
            recovery: Optional[ErrorRecovery] = None,
            prompt: str = DEFAULT_SPLIT_PROMPT,
            model_name: str = 'deepseek-v4-flash',
            model_kwargs: Optional[dict] = None,
            model_factory: Optional[Callable[[], Any]] = None,
            id_source: Optional[FragmentIdSource] = None,
    ):
        # 注入的模型需支持结构化输出（ainvoke 返回 SummaryMemoryAi）；None 时按工厂惰性创建
        self.split_llm = split_llm
        self._kv_store = kv_store or MemoryKVStore()
        self._recovery = recovery or NullRecovery()
        self.prompt = prompt
        self._model_name = model_name
        self._model_kwargs = model_kwargs or {}
        self._model_factory = model_factory
        # 片段 id / 消息偏移的持久化高水位来源（KV 计数器丢失后的兜底；不注入则行为与原来一致）
        self.id_source = id_source
        self._id_reconciled: set[str] = set()      # 本进程已向持久层校正过基数的用户
        self._id_high_water: dict[str, int] = dict()  # 进程内高水位：KV 中途被清也不复用 id
        # 每个用户当前对话主题（切分模型产出，检索调参用）
        self.dialogue_theme_by_user: dict[str, str] = dict()

    def _create_model(self):
        if self._model_factory is not None:
            return self._model_factory()
        return init_chat_model(self._model_name, **self._model_kwargs)

    def create_agent(self):
        if not self.split_llm:
            initial_model = self._create_model()
            self.split_llm = initial_model.with_structured_output(SummaryMemoryAi).with_retry(
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

    async def asplit_text(
            self,
            messages: list[BaseMessage],
            user_id: str,
            current_index: Optional[int] = 0,
    ) -> List[MemoryFragments] | None:
        """对 current_index 之后的新消息做主题切分；LLM 失败时走 ErrorRecovery 并返回 None"""
        self.create_agent()
        if not self.split_llm:
            raise ValueError("摘要模型未初始化")
        self.dialogue_theme_by_user.setdefault(user_id, '')
        # current_index = 已切分的消息数（消息偏移，见 atext_to_document 的双游标说明）
        messages = messages[current_index:]
        text = []
        content_text = []
        for i, message in enumerate(messages):
            # content_text 必须与 messages 一一对应：空 content 的消息也要占位，
            # 否则索引错位会导致 LLM 返回的 start_idx/end_idx 切错内容
            if isinstance(message, AIMessage):
                role = 'ai'
                content = message.content
                if not content and message.tool_calls:
                    content = '（调用了工具 ' + ', '.join(tc['name'] for tc in message.tool_calls) + '，无文本内容）'
            elif isinstance(message, ToolMessage):
                role = 'tool'
                content = message.content if isinstance(message.content, str) else str(message.content)
            elif isinstance(message, HumanMessage):
                role = 'user'
                content = message.content
            else:
                role = 'other'
                content = message.content if message.content else '（非对话消息）'
            if not content:
                content = '（消息为空）'
            if role == 'tool' and len(content) > 100:
                content = content[:100] + '...'
            text.append(f'对应的消息索引{i}，{role}:{content}')
            content_text.append(content)
        prompt = "请将以下文本按照不同主题进行切分区域：\n" + "\n".join(text)
        try:
            splited_texts = await self.split_llm.ainvoke([
                SystemMessage(content=self.prompt),
                HumanMessage(content=prompt),
            ])
        except Exception as e:
            logger.error(f"在记忆分片时出现了异常：{e}")
            await self._recovery.on_split_error(user_id, messages, e)
            # 返回 None 而非保底切分：SummaryMemoryAi 涉及类型与主题，无法可靠兜底
            return None
        if splited_texts is None:
            # with_structured_output 解析失败时返回 None（**不抛异常**）——必须走同一条失败路径：
            # 否则下面取 .current_theme 会抛 AttributeError，把整轮对话掀掉（2026-10-05 基准实测踩到）
            err = ValueError('切分模型返回空结果（结构化输出解析失败）')
            logger.error(f"在记忆分片时出现空结果：{err}")
            await self._recovery.on_split_error(user_id, messages, err)
            return None
        self.dialogue_theme_by_user[user_id] = splited_texts.current_theme
        theme_num = splited_texts.theme_num
        conf = splited_texts.config
        return self.split_text(theme_num, conf, content_text, messages)

    async def atext_to_document(
            self,
            memory_fragments: Optional[List[MemoryFragments]] = None,
            text: Optional[list[BaseMessage]] = None,
            user: Optional[str] = None,
    ) -> List[Document] | None:
        """切分结果落库为带元数据的 Document。

        两个游标（此前共用一个键导致片段数≠消息数时窗口错位）：
        - memory_fragments:{user}：片段 id 基数（每片段 +1，用于文档 id 与增量归纳游标）
        - memory_msg_offset:{user}：已切分的消息数（切分窗口偏移，每成功切片推进到 len(text)）
        text 必须传"完整消息列表"（从会话开头），偏移由消息游标控制，保证全局单调。
        """
        fragment_base = await self.fragment_id_base(user)
        raw_offset = await self._kv_store.aget(f"memory_msg_offset:{user}")
        msg_offset = int(raw_offset or '0')
        if raw_offset is None:
            # KV 丢了：从片段元数据的持久高水位恢复切分进度，避免同一会话被从 0 重切出重复片段
            probed = await self._probed_ids(user)
            if probed is not None:
                msg_offset = max(msg_offset, probed[1])
        # 新会话检测：消息列表比游标短 → 游标属于上一个会话的消息流，重置为 0。
        # （偏移语义是"本消息流已切分的消息数"；同一会话内 state['messages'] 只增不减）
        if len(text or []) < msg_offset:
            msg_offset = 0
        if not memory_fragments:
            memory_fragments = await self.asplit_text(text, user, msg_offset)
        if not memory_fragments:
            return None
        # 原子预留：先把 KV 计数器抬到持久高水位（KV 丢失或落后时），再一次性 INCRBY 拿 n 个 id。
        # 多进程下每个进程各自可能做一次校正，但 INCRBY 保证各进程拿到的 id 段互不重叠。
        if fragment_base > int(await self._kv_store.aget(f"memory_fragments:{user}") or '0'):
            await self._kv_store.aset(f"memory_fragments:{user}", str(fragment_base))
        last_id = await self._kv_store.aincr(f"memory_fragments:{user}", len(memory_fragments))
        documents = []
        fragment_id = last_id - len(memory_fragments)
        for fragment in memory_fragments:
            fragment_id += 1
            metadata = fragment.config.model_dump()
            # 用户归属与片段序号：跨用户检索过滤（user_id）与增量归纳游标（id）
            metadata['user_id'] = user
            metadata['id'] = fragment_id
            # 文档 id 需全局唯一（sqlite 表内有 UNIQUE 约束），带 user 前缀避免多用户撞 id
            document = Document(
                id=f"{user}-{fragment_id}",
                page_content=fragment.content,
                metadata=metadata,
            )
            documents.append(document)
        if documents:
            # 最后一片记下"切到哪了"：KV 丢失后据此恢复消息偏移（旧数据无此键 → 取 0，行为不变）
            documents[-1].metadata['msg_end'] = len(text or [])
        # 计数器已由 aincr 原子写回（值 = last_id），这里只更新进程内高水位
        self._id_high_water[user] = last_id
        # 消息偏移 = 完整消息列表长度（切分失败时不推进，下轮重试）
        await self._kv_store.aset(f"memory_msg_offset:{user}", str(len(text or [])))
        return documents

    async def _probed_ids(self, user: str) -> Optional[tuple]:
        """向持久化来源探一次 (最大片段 id, 最大已切分消息数)；未注入来源或查询失败返回 None"""
        if self.id_source is None:
            return None
        try:
            return self.id_source.max_fragment_ids(user)
        except Exception as e:
            logger.warning(f"片段高水位兜底查询失败，沿用 KV 计数器: {e}")
            return None

    async def fragment_id_base(self, user: str) -> int:
        """片段 id 基数：KV 快路径 + 持久高水位兜底，只抬高不回退。

        触发兜底查询的条件（其余情况零额外查询）：
        - KV 键缺失或为 0（计数器丢失的失败信号）
        - 本进程尚未校正过该用户（覆盖"KV 存在但落后"，如 Redis 恢复了旧快照）
        """
        raw = await self._kv_store.aget(f"memory_fragments:{user}")
        base = int(raw or '0')
        if raw is None or base == 0 or user not in self._id_reconciled:
            probed = await self._probed_ids(user)
            if probed is not None:
                base = max(base, probed[0])
                self._id_reconciled.add(user)
        return max(base, self._id_high_water.get(user, 0))

    @staticmethod
    def split_text(theme_num, conf, text: list[str], messages: list[BaseMessage]) -> List[MemoryFragments]:
        """把切分模型的输出（主题数+配置列表）翻译为记忆片段。

        防御性处理：theme_num 与 conf 长度不一致时截断；与消息条数相关的越界跳过该片段
        （负索引与零宽/反向区间已由 SummaryMemoryFragmentsConfig 的模型校验兜底）。
        """
        if theme_num <= 0:
            logger.warning(f"长主题数量必须大于0，当前主题数量为{theme_num}")
            return list()
        if theme_num != len(conf):
            logger.error(f"长主题数量与配置数量不一致，当前主题数量为{theme_num}，"
                         f"配置数量为{len(conf)}，已截断为 {min(theme_num, len(conf))} 处理")
        memory_fragments = []
        valid_count = min(theme_num, len(conf))
        for i in range(valid_count):
            # 负索引、零宽/反向区间已在 SummaryMemoryFragmentsConfig 的模型校验里兜底，
            # 这里只处理与消息条数相关的越界
            start = conf[i].start_idx
            end = min(conf[i].end_idx, len(messages))
            if start >= len(messages) or start >= end:
                logger.warning(f"第 {i} 个片段范围无效: [{start}, {end})，"
                               f"消息数 {len(messages)}，跳过该片段")
                continue

            content = "\n".join(text[start:end])
            # 取片段内最后一条消息的时间
            msg_end_time = messages[end - 1].additional_kwargs.get('time', time.time())
            metadata = MemoryFragmentsMetadata(
                theme=conf[i].theme,
                type=conf[i].type,
                time=msg_end_time,
                strengthen_num=0,
            )
            memory_fragments.append(MemoryFragments(
                config=metadata,
                content=content,
            ))
        return memory_fragments
