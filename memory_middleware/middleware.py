"""衡忆多维认知架构（BalancedMultiDimensionMemory）：LangGraph AgentMiddleware。

上游项目：https://github.com/lijia-ming/fireflymall-ai-customer-service

三层记忆：
    1. 工作记忆（短期）：带时间戳的 messages 列表
    2. 暂存库（待巩固的中间层）：从工作记忆卸载的对话切片（向量 RAG），
       含情境标签（主题/类型/时间/巩固次数）
    3. 长期记忆库（经巩固的知识层）：结构化用户画像（UserProfile）

核心机制：
    - 记忆成熟度 M(Δt)=1-exp(-(Δt/τ_m)^c_m) 达到阈值后触发切片
    - 切片由 LLM 按主题切分，落暂存库（向量 + 元数据）
    - 检索评分 S(m)=α·R+β·T+γ·F+δ（LLM 按当前主题动态调参 α/β/γ/δ/w0/w1/w2）
    - top-K 片段注入系统提示词，成功注入后该片段"再巩固"（刷新时间戳、增加 F 值）
    - 归纳游标（last_summarized_id）增量总结：只对游标后的新片段总结并合并进旧画像

与上游项目 time_memory.py 的差异（解耦点）：
    - 配置：config.yaml 全局 → MemoryConfig 构造器注入
    - 主系统提示词：agent.main_agent 导入 → initial_prompt 注入
    - 片段游标：Redis → KVStore 注入（默认 MemoryKVStore 单进程）
    - 向量库：DashScope + sqlite 文件 → SQLiteVecStore(embedding 注入，默认 HashEmbeddings 离线)
    - 失败保底：RabbitMQ → ErrorRecovery 注入（默认 NullRecovery）
    - token 计数：tiktoken/transformers → TokenCounter 注入（默认 字数/3.3）
"""
import logging
import math
import time
from asyncio import Lock
from textwrap import dedent
from typing import Any, Awaitable, Callable, Optional

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse, ToolCallRequest
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from typing_extensions import override

from .config import MemoryConfig, TYPE_SCORE_MAP
from .formula import (importance, maturity, relevance_scores, score, time_decay,
                      type_score_union)
from .models import TimeMemoryFormulaParam
from .profile import merge_user_profile
from .prompt import SystemPromptOperation
from .rag import FragmentsMemoryRAG
from .spliter import MemoryFragmentsAiSpliter
from .storage.kv import KVStore, MemoryKVStore
from .storage.vectors import FragmentIdSource, HashEmbeddings, SQLiteVecStore
from .token_counter import TokenCounter, default_token_counter

logger = logging.getLogger(__name__)

_METERAGE_MAP = {
    'K': 10 ** 3,
    'M': 10 ** 6,
    'B': 10 ** 9,
}

_PARAM_CACHE_MAX = 256   # 调参缓存条数上限（FIFO 淘汰）

DEFAULT_MATH_PROMPT = dedent("""\
    你现在是一个专业的数学专家，你需要通过当前聊天主题来调整对于公式的参数。
    公式：S(m)=α*R(m,q)+β*T(m)+γ*F(m)+δ
    其中：
        S(m) -> 记忆片段得分
        R(m, q) ：语义相关性 -> 决定是否需要
            R = max(0, (cos(m,q) - 本批候选中位余弦) / (本批最高余弦 - 本批中位余弦))
            即在**本次候选集合内**做相对刻度归一化：低于中位数的记 0，最高者记 1
        T(m) ：结合艾宾浩斯曲线的参数化时间衰减 -> 决定是否"过期"
        F(m) ：固有重要性 -> 决定是否重要
        而F(m) 的公式为：clamp(w0 + w1*u + w2*(1-exp(-refresh_count(m)*k)), 0, 1)
            其中 u = 1 - Π(1 - v_i)：类型分的**概率并集**（多类型递增、边际递减、有界不饱和）
            refresh_count(m) ：记忆m的刷新次数，初始值为0，每次刷新增加1
            w0, w1, w2：超参数，根据实际情况调整
        α, β, γ, δ ：超参数，根据实际情况调整

    因此，你需要根据当前聊天主题，调整w0, w1, w2, α, β, γ, δ的值。

    参数限制：
        参数 建议范围         说明
        w0  [0.05, 0.15]    基础重要性，确保任何记忆片段都不会因零分而被完全遗忘。值不宜过大，否则会稀释类型和巩固的区分度。
        w1  [0.5, 0.8]      类型重要性的权重，是 F(m) 的主要贡献项。因为类型是记忆最稳定的属性，决定其先天重要性。
        w2  [0.05, 0.2]     巩固次数的权重，是记忆的后天强化项。值不宜过大，否则频繁访问的片段会过度压制重要但很少被检索的关键信息。每次巩固的增量很小（如每次 +0.02），需要通过多次巩固才能显著提升。

        α   [0, 1]          语义相关性权重
        β   [0, 1]          时间衰减权重
        γ   [0, 1]          固有重要性权重
        δ   [-0.1, 0.2]     基线保留偏移, 独立于归一化约束，仅用于微调

    必须满足：
        w0+w1+w2=1
        α+β+γ=1
    """).strip('\n')


async def _load_user_profile(user_id: str, runtime) -> dict:
    """读取用户既有的长期记忆画像（含归纳游标），失败时按空画像处理"""
    if runtime.store is None:
        return {}
    try:
        item = await runtime.store.aget(('long-term', 'user_profile',), user_id)
    except Exception as e:
        logger.warning(f"读取用户画像失败，按空画像处理: {e}")
        return {}
    if item is None:
        return {}
    return item.value.get('user_profile', {}) or {}


class BalancedMultiDimensionMemory(AgentMiddleware):
    """结合 AI 模型与数学公式的长期记忆中间件：

    多维加权评分 + 可调艾宾浩斯衰减 + 主题分片 + 权重制衡 + 分层记忆 + 防上下文分裂。

    挂载方式（与官方 SummarizationMiddleware 同为 AgentMiddleware）：
        from langchain.agents import create_agent
        agent = create_agent(model, tools, middleware=[BalancedMultiDimensionMemory(config)])

    运行时要求（langgraph）：
        - runtime.context 提供 user_id（或 state['user_id']）
        - runtime.store 提供长期记忆（如 InMemoryStore / PostgresStore），
          键空间 ('long-term', 'user_profile') / ('long-term', 'user_person')
        - state 至少包含 messages / new_msg_idx / user_profile / system_prompt 通道
    """

    def __init__(
            self,
            config: Optional[MemoryConfig] = None,
            summary_llm: Optional[Any] = None,
            spliter: Optional[MemoryFragmentsAiSpliter] = None,
            rag: Optional[FragmentsMemoryRAG] = None,
            kv_store: Optional[KVStore] = None,
            recovery: Optional[Any] = None,
            token_counter: Optional[TokenCounter] = None,
            math_prompt: str = DEFAULT_MATH_PROMPT,
    ):
        super().__init__()
        self.config = config or MemoryConfig()
        self.math_agent_prompt = math_prompt

        v = self.config.vocation
        self.m_t = v.tau_m
        self.m_c = v.c_m
        self.t_t = v.tau
        self.t_c = v.c_t
        self.slice_value = v.slice_value
        self.long_term_value = v.long_term_value

        # ---- 校验总结触发配置 ----
        pattern = self.config.pattern
        trigger = self.config.trigger_threshold
        if pattern not in ('fraction', 'tokens', 'messages'):
            raise ValueError("总结模式必须为 fraction, tokens, messages 中的一个")
        if pattern == 'fraction':
            if not 0 <= trigger <= 1:
                raise ValueError("对于 fraction 模式，总结阈值必须在 0 到 1 之间")
            if trigger < 0.7:
                import warnings as _warnings
                _warnings.warn("对于 fraction 模式，总结阈值过低，建议设置为 0.7 或以上，以避免反复总结")
        elif pattern == 'tokens':
            if not isinstance(trigger, int):
                raise ValueError("对于 tokens 模式，总结阈值必须为整数")
            if trigger <= 0:
                raise ValueError("对于 tokens 模式，总结阈值必须大于 0")
            if trigger < 10000:
                import warnings as _warnings
                _warnings.warn("对于 tokens 模式，总结阈值过低，建议设置为 10000 或以上，以避免反复总结")
        elif pattern == 'messages':
            if not isinstance(trigger, int):
                raise ValueError("对于 messages 模式，总结阈值必须为整数")
            if trigger <= 0:
                raise ValueError("对于 messages 模式，总结阈值必须大于 0")
            if trigger < 50:
                import warnings as _warnings
                _warnings.warn("对于 messages 模式，总结阈值过低，建议设置为 50 或以上，以避免反复总结")

        self.max_token = self._get_model_max_tokens()
        self.token_counter = token_counter or default_token_counter()

        # ---- 子组件：全部支持注入（离线默认实现保证无服务也能跑） ----
        self._kv_store = kv_store or MemoryKVStore()
        self._recovery = recovery

        def _model_factory():
            return init_chat_model(self.config.model_name, **self.config.model_kwargs)

        if rag is None:
            embeddings = self.config.embeddings or HashEmbeddings()
            vec_store = SQLiteVecStore(
                self.config.rag_db_path, embeddings, table=self.config.rag_table)
            rag = FragmentsMemoryRAG(
                vector_store=vec_store, recovery=self._recovery, model_factory=_model_factory)
        self.memory_fragments_rag = rag

        self.memory_spliter = spliter or MemoryFragmentsAiSpliter(
            kv_store=self._kv_store, recovery=self._recovery, model_factory=_model_factory)
        # 片段 id 撞库兜底：注入式 spliter（bench / tests / examples 都走这条）也自动接上持久高水位来源
        if self.memory_spliter.id_source is None:
            vec_store = getattr(rag, 'vector_store', None)
            if isinstance(vec_store, FragmentIdSource):
                self.memory_spliter.id_source = vec_store

        self._summary_llm = summary_llm
        # 按 user 隔离的提示词操作实例，避免多用户并发时互相覆盖 memory_fragments
        self._prompt_operations: dict[str, SystemPromptOperation] = dict()
        self._lock: dict[str, Lock] = dict()
        self._user_retrieve_state: dict[str, bool] = dict()
        # 调参缓存：(user_id, 主题) → 参数。主题由切分模型产出、只在切片时变，
        # 同一主题下的多次注入不必重复调 LLM（顺带省掉一次结构化调用）
        self._param_cache: dict[tuple[str, str], Any] = dict()
        # 打分明细（诊断用）：每次注入事件记一条，含调参值与每个候选片段的 R/T/F 及各维加权贡献，
        # 用来回答"多维评分里到底是哪一维在决定入选"。有界保留，不进任何生产路径。
        self._scoring_log: list[dict] = []
        self._init_lock = Lock()

    # ------------------------------------------------------------------
    # 钩子实现
    # ------------------------------------------------------------------

    async def _ensure_summary_agent(self):
        if self._summary_llm is None:
            async with self._init_lock:
                if self._summary_llm is None:
                    initial_model = init_chat_model(self.config.model_name, **self.config.model_kwargs)
                    self._summary_llm = initial_model.with_structured_output(TimeMemoryFormulaParam)

    @staticmethod
    def _get_user_id(runtime, state=None) -> str:
        ctx = getattr(runtime, 'context', None)
        uid = getattr(ctx, 'user_id', None)
        if uid is None and state is not None:
            uid = state.get('user_id')
        if uid is None:
            raise ValueError("无法获取 user_id：请在 runtime.context 上提供 user_id（或 state['user_id']）")
        return uid

    @override
    async def abefore_model(self, state, runtime) -> dict[str, Any] | None:
        await self._ensure_summary_agent()
        new_idx = state.get('new_msg_idx', 0)
        messages = state['messages'][new_idx:]

        # 自包含时间戳：用户消息（或任何消息）缺失 time 字段时在此补齐。
        # 上游项目由外部 msg_handle 入口打点；独立版不依赖外部辅助——
        # 缺失视为"当前时间"，时间机制（成熟度/衰减/总结）从首次见到起算。
        # 幂等：已有时间戳的消息不被覆盖。
        now = time.time()
        for msg in messages:
            if 'time' not in msg.additional_kwargs:
                msg.additional_kwargs['time'] = now
                logger.debug(f"消息缺失 time 字段，已按首次见到时间补点: {msg.content[:50]!r}")

        user_id = self._get_user_id(runtime, state)
        self._lock.setdefault(user_id, Lock())

        async with self._lock[user_id]:
            self._user_retrieve_state.setdefault(user_id, False)

            # 记忆切片（成熟度触发）
            if not await self._memory_slice(messages, user_id=user_id, runtime=runtime):
                return None

            current_tokens = self._calculate_current_token(messages, user_id)

            if self._is_primary_triggered(current_tokens, len(messages)):
                # 事件 = 窗口达到预算。awrap_model_call 会把上下文截断到最近一条人类消息，
                # 因此固化必须与事件同刻发生，否则这段内容既不在上下文、也不在暂存库。
                # 切分不再是独立动作：一次事件 = 切分 + 调参 +（片段成熟时）归纳。
                # text 必须传完整消息列表（消息偏移游标在 spliter 内部管理，见 atext_to_document 说明）
                if memory_fragments := await self.memory_spliter.atext_to_document(text=state['messages'], user=user_id):
                    try:
                        await self.memory_fragments_rag.add(memory_fragments)
                    except Exception as e:
                        # 落库失败不该掀掉整轮对话：游标已推进，本批片段弃掉（留 id 空洞但不会重复）
                        logger.error(f"记忆片段落库失败，跳过本批: {e}")
                self._user_retrieve_state[user_id] = True
                return None

            return None

    @override
    async def awrap_model_call(
            self,
            request: ModelRequest[Any],
            handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        user_id = self._get_user_id(request.runtime, request.state)
        system_prompt = request.state.get('system_prompt')
        msg_idx = request.state.get('new_msg_idx')
        logger.info(f"用户ID：{user_id}")
        try:
            if not self._user_retrieve_state.get(user_id, False):
                if system_prompt is not None and msg_idx is not None:
                    new_request = request.override(
                        system_message=SystemMessage(content=system_prompt),
                        messages=request.messages[msg_idx:],
                    )
                    return await handler(new_request)
                else:
                    return await handler(request)
            # 检索提取、重写提示词
            related_fragments = await self._extract_fragments_by_time(user_id)
            if not related_fragments:
                # 当前无可注入的记忆片段：重置检索状态，按普通调用处理
                self._user_retrieve_state[user_id] = False
                if system_prompt is not None and msg_idx is not None:
                    new_request = request.override(
                        system_message=SystemMessage(content=system_prompt),
                        messages=request.messages[msg_idx:],
                    )
                    return await handler(new_request)
                else:
                    return await handler(request)

            # 按 user 隔离提示词操作实例，避免多用户并发时互相覆盖
            if user_id not in self._prompt_operations:
                self._prompt_operations[user_id] = SystemPromptOperation(
                    initial_prompt=self.config.initial_prompt)
            prompt_ops = self._prompt_operations[user_id]
            new_documents, old_ids = prompt_ops.add_memorys([x[0] for x in related_fragments])
            # 用户画像作为独立、每轮新鲜的层
            prompt_ops.prompt.profile = request.state.get('user_profile') or ''
            new_prompt = prompt_ops.get_prompt()
            logger.info(f"重写后的提示词为：{new_prompt}")

            total_messages = len(request.messages)
            for msg in reversed(request.messages):
                total_messages -= 1
                if isinstance(msg, HumanMessage):
                    break
            if total_messages < 0:
                total_messages = len(request.messages) - 1

            new_request = request.override(
                system_message=SystemMessage(content=new_prompt),
                messages=request.messages[total_messages:],
            )
            self._user_retrieve_state[user_id] = False

            response = await handler(new_request)
            # 模型调用成功后才将强化落库，避免失败轮次误记巩固；
            # 强化失败属于尽力而为的后置操作，不应触发外层降级导致模型被二次调用
            try:
                await self.memory_fragments_rag.update_config_strengthen(
                    ids=old_ids, documents=new_documents)
            except Exception as e:
                logger.error(f"记忆强化落库失败（不影响本次回复）: {e}")
            request.state['new_msg_idx'] = total_messages
            state_update = {"new_msg_idx": total_messages, "system_prompt": new_prompt}
            return ExtendedModelResponse(model_response=response, command=Command(update=state_update))
        except Exception as e:
            self._user_retrieve_state[user_id] = False
            logger.error(f"记忆检索失败，降级为普通调用: {e}")
            return await handler(request)

    @override
    async def aafter_model(self, state, runtime) -> dict[str, Any] | None:
        """为 AI 消息添加时间戳"""
        messages = state['messages']
        messages = list(reversed(messages))
        for msg in messages:
            if isinstance(msg, AIMessage) and 'time' not in msg.additional_kwargs:
                msg.additional_kwargs['time'] = time.time()
                break
        return None

    @override
    async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """在工具执行完成的当刻为 ToolMessage 打上真实时间戳，
        避免在下一轮的 abefore_model 中补打 time.time() 造成时间偏差"""
        result = await handler(request)
        if isinstance(result, ToolMessage) and 'time' not in result.additional_kwargs:
            result.additional_kwargs['time'] = time.time()
        return result

    # ------------------------------------------------------------------
    # 内部逻辑
    # ------------------------------------------------------------------

    def _get_model_max_tokens(self) -> Optional[int]:
        """解析 max_input_tokens（支持 '1m' 这类带单位的写法）"""
        value = self.config.max_input_tokens
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                last_symbol = value[-1].upper()
                num_tokens = int(value[:-1])
                if last_symbol in _METERAGE_MAP:
                    return num_tokens * _METERAGE_MAP[last_symbol]
            except ValueError:
                pass
        raise ValueError(f"max_input_tokens 无效: {value!r}")

    def _calculate_current_token(self, messages: list[BaseMessage], user_id: str = None) -> int:
        if user_id is None:
            raise ValueError("user_id cannot be None")
        return math.ceil(self.token_counter(messages))

    def _is_primary_triggered(self, segment_tokens: int, segment_len: int) -> bool:
        """是否触发一次检索注入（工作记忆达到上下文预算）"""
        if self.config.pattern == 'fraction':
            return segment_tokens >= self.max_token * self.config.trigger_threshold
        elif self.config.pattern == 'tokens':
            return segment_tokens >= self.config.trigger_threshold
        elif self.config.pattern == 'messages':
            return segment_len >= self.config.trigger_threshold
        return False

    @staticmethod
    async def _summary_to_long_term(res_json: dict, user_id: str, runtime):
        """总结写入长期记忆（store），文件类内容不在本中间件范围"""
        if runtime.store is None:
            logger.warning("未配置 langgraph store，跳过长期记忆存储")
            return
        if res_json.get('user_person'):
            await runtime.store.aput(
                ('long-term', 'user_person',),
                user_id,
                {'user_person': res_json.get('user_person')}
            )
        await runtime.store.aput(
            ('long-term', 'user_profile',),
            user_id,
            {'user_profile': res_json}
        )

    async def _memory_slice(self, messages: list[BaseMessage], user_id: str, runtime):
        """记忆切片判断 + 增量归纳（成熟片段总结为长期记忆画像）"""
        last_msg = messages[0]
        last_time = last_msg.additional_kwargs.get('time', time.time())
        m = maturity(time.time() - last_time, self.m_t, self.m_c)

        # 增量归纳：读取既有画像的游标，只对游标之后的新片段做总结，并合并进旧画像
        existing_profile = await _load_user_profile(user_id, runtime)
        last_summary_id = existing_profile.get('last_summarized_id', 0)
        if profile := await self.memory_fragments_rag.long_memory_summary_by_time(
                self.m_t, self.m_c, self.long_term_value,
                user_id=user_id, last_summary_id=last_summary_id):
            profile_json, new_cursor = profile
            # 列表字段语义去重（注入真实 embedding 才启用；None 时退化为精确去重，不调 API）
            merged = merge_user_profile(
                existing_profile, profile_json,
                embedding=self.config.embeddings,
                similarity_threshold=self.config.profile_dedup_threshold,
                semantic_min_items=self.config.profile_semantic_min_items,
                max_items=self.config.profile_list_max,
            )
            merged['last_summarized_id'] = new_cursor
            await self._summary_to_long_term(merged, user_id, runtime)
        logger.info(f"记忆切片概率：{m}")
        return m >= self.slice_value

    async def _extract_fragments_by_time(self, user_id: str):
        """检索候选片段 + LLM 动态调参 + S(m) 评分排序，取 top-K 注入"""
        current_theme_by_user = self.memory_spliter.dialogue_theme_by_user.get(user_id, '')
        if not current_theme_by_user:
            return None
        all_fragments_by_theme = await self.memory_fragments_rag.query_context_distance(
            current_theme_by_user, k=self.config.retrieve_k, user_id=user_id)
        if not all_fragments_by_theme:
            # 库里还没有该用户的片段时，跳过 LLM 调参调用，避免每轮白烧一次模型
            return None
        cache_key = (user_id, current_theme_by_user)
        params_class = self._param_cache.get(cache_key)
        if params_class is None:
            params_class = await self._summary_llm.ainvoke([
                SystemMessage(content=self.math_agent_prompt),
                HumanMessage(content=f"当前用户{user_id}的对话主题：{current_theme_by_user}"),
            ])
            if len(self._param_cache) >= _PARAM_CACHE_MAX:
                self._param_cache.pop(next(iter(self._param_cache)))  # FIFO 淘汰，防无界增长
            self._param_cache[cache_key] = params_class
        w0 = params_class.w0
        w1 = params_class.w1
        w2 = params_class.w2
        alpha = params_class.alpha
        beta = params_class.beta
        gamma = params_class.gamma
        delta = params_class.delta

        retrieve_fragments = []
        score_rows = []
        # 相关度改为候选内相对刻度（中位数锚定，见 formula.relevance_scores 的取舍说明）：
        # 旧做法 max(0, cos − 0.5) 的绝对值不可跨主题/跨嵌入模型比较，且实测 αR 几乎不区分候选
        relevances = relevance_scores([c for _, c in all_fragments_by_theme])
        for (document, cosine), r in zip(all_fragments_by_theme, relevances):
            document_time = document.metadata['time']
            type_list = document.metadata['type']
            strengthen_num = document.metadata['strengthen_num']
            t = time_decay(time.time() - document_time, self.t_t, self.t_c)
            f = importance(w0, w1, w2, type_score_union(type_list, TYPE_SCORE_MAP),
                           strengthen_num, k=self.config.strengthen_k)
            s_m = score(alpha, r, beta, t, gamma, f, delta)
            retrieve_fragments.append((document, s_m))
            score_rows.append({
                'id': (document.metadata or {}).get('id'),
                'theme': (document.metadata or {}).get('theme', ''),
                'type': type_list,
                'strengthen_num': strengthen_num,
                'age_s': round(time.time() - document_time, 1),
                'cos': round(cosine, 4),      # 原始余弦：让相关度的标定可事后复算
                'r': round(r, 4), 't': round(t, 4), 'f': round(f, 4),
                'alpha_r': round(alpha * r, 4), 'beta_t': round(beta * t, 4),
                'gamma_f': round(gamma * f, 4), 's': round(s_m, 4),
            })
        retrieve_fragments.sort(key=lambda x: x[1], reverse=True)
        picked = self._select_top_k(retrieve_fragments)
        picked_ids = {(d.metadata or {}).get('id') for d, _ in picked}
        self._record_scoring(params_class, score_rows, picked_ids)
        return picked

    def _record_scoring(self, params_class, rows: list, picked_ids: set) -> None:
        """记一次打分明细（诊断用，有界保留最近 20 次事件）。

        回答"多维评分里哪一维在决定入选"：rows 是本次事件的全部候选及其 R/T/F 与各维加权贡献，
        picked 标出最终占席的片段（含 always_inject_types 保底与主题名额的干预结果）。
        """
        self._scoring_log.append({
            'params': {k: getattr(params_class, k, None)
                       for k in ('alpha', 'beta', 'gamma', 'delta', 'w0', 'w1', 'w2')},
            'candidates': [{**row, 'picked': row['id'] in picked_ids} for row in rows],
        })
        if len(self._scoring_log) > 20:
            self._scoring_log.pop(0)

    def _select_top_k(self, scored: list) -> list:
        """注入选片。

        - `always_inject_types`：这些类型的片段优先占席（保底），组内仍按 S(m) 降序。
          动机：检索预筛按主题语义取候选，主题一偏，其它主题的 identity/preference 片段会整块落选
          ——而它们正是 TYPE_SCORE_MAP 里价值最高的类型。
        - `top_k_max_per_theme`：单主题名额上限（防同主题占满），名额没填满时放开补齐。
        """
        limit = self.config.top_k
        priority = set(self.config.always_inject_types or ())
        if priority:
            def _is_priority(doc):
                return bool(priority & set((doc.metadata or {}).get('type') or []))
            scored = sorted(scored, key=lambda x: (not _is_priority(x[0]), -x[1]))
        cap = self.config.top_k_max_per_theme
        if not cap:
            return scored[:limit]
        picked, per_theme = [], {}
        for doc, s_m in scored:
            theme = (doc.metadata or {}).get('theme', '')
            if per_theme.get(theme, 0) >= cap:
                continue
            picked.append((doc, s_m))
            per_theme[theme] = per_theme.get(theme, 0) + 1
            if len(picked) >= limit:
                return picked
        if len(picked) < limit:   # 单主题占比过高 → 放开上限补齐
            chosen = {id(d) for d, _ in picked}
            for doc, s_m in scored:
                if len(picked) >= limit:
                    break
                if id(doc) not in chosen:
                    picked.append((doc, s_m))
        return picked
