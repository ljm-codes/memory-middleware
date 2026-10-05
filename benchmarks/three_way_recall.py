# -*- coding: utf-8 -*-
"""三方召回对比基准（针-草垛）：BalancedMultiDimensionMemory vs 官方 SummarizationMiddleware vs 无记忆基线

设计：
- 每轮对话 = 3 条"事实针"（姓名/职业/爱好/约束）+ L 条闲聊草垛 + 4 条召回提问
- 三种配置用完全相同的对话文本与驱动循环：
    无记忆基线：上下文只保留最近 BUDGET 条消息（有界上下文）
    SummarizationMiddleware：超过阈值后把旧消息压成摘要（官方中间件）
    BMDM：切片落向量库 + 增量画像 + 检索注入（本项目中间件）
- 命中判定：答案包含事实关键词即算召回成功

用法：
    python benchmarks/three_way_recall.py --lengths 8,16,24 --samples 1 --embeddings dashscope
    python benchmarks/three_way_recall.py --chart-only   # 用已保存结果重新出图

结果落盘：benchmarks/output/recall_results.json（增量保存，可断点续跑）
依赖 .env：DEEPSEEK_API_KEY（必需）；DASHSCOPE_API_KEY（--embeddings dashscope 时）

留档规则（2026-10-05 起）：
1. 每条记录带 `_meta`（版本 / 提交 / 运行时刻 / 完整口径参数）——答"这条数据是哪版哪时什么参数跑的"。
2. **判定输入与判定结果一起留档**：`judge_input = {keywords, answers}`，判定是"答案里有没有这个关键词"。
   没有答案原文，判定就无法事后复核，换了判分口径只能重跑（2026-10-05 就吃过这个亏：判分对空格敏感，
   历史答案没存，只能重跑；`--rejudge` 可在不跑模型的前提下用新口径重判已留档的答案）。
3. 判定前做空白归一化（`_norm`）——只消格式差异，不放宽语义。
"""
import argparse
import asyncio
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / '.env')

# 基准跑分强制关闭 LangSmith 追踪：免费额度有限，开着会拖慢并持续报"超过请求上限"
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.store.memory import InMemoryStore

from memory_middleware import (
    BalancedMultiDimensionMemory,
    FragmentsMemoryRAG,
    MemoryConfig,
    MemoryFragmentsAiSpliter,
    VocationParams,
)
from memory_middleware.models import SummaryMemoryAi, TimeMemoryFormulaParam, UserProfile
from memory_middleware.recovery import NullRecovery
from memory_middleware.token_counter import default_token_counter
from memory_middleware.storage.kv import MemoryKVStore
from memory_middleware.storage.vectors import HashEmbeddings, SQLiteVecStore

BUDGET = 8          # 有界上下文预算（消息条数）：基线截断 / summarize 触发 / BMDM 检索触发
CHAT_MAX_TOKENS = 100  # 加速：聊天回复限长（不影响召回判定）
OUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'

# ---------------------------------------------------------------------------
# 评测数据（确定性生成）
# ---------------------------------------------------------------------------

FACT_SETS = [
    {   # sample 0
        'name': '李明轩',
        'occupation': '数据分析师',
        'hobby': '攀岩',
        'constraint': '每周三下午要开会',
        'turns': [
            '你好，我叫李明轩，是一名数据分析师。',
            '我的爱好是攀岩，周末经常去岩馆。',
            '对了，我每周三下午都有例会，那个时段不太方便。',
        ],
    },
    {   # sample 1
        'name': '陈雅雯',
        'occupation': '产品经理',
        'hobby': '观鸟',
        'constraint': '每个月第一周出差',
        'turns': [
            '你好，我是陈雅雯，做产品经理的。',
            '我平时喜欢观鸟，家里有个望远镜。',
            '提醒一下，我每个月第一周都在出差。',
        ],
    },
]

QUESTIONS = [
    '请问我叫什么名字？',
    '我的职业是什么？',
    '我的爱好是什么？',
    '我什么时候不方便？',
]

# 细节保留评测：8 条低显著度细节（摘要压缩最容易丢失的信息类型：
# 具体数字/名称/尾号等），逐条提问检验"总结后不丢失细节"。
DETAIL_SETS = [
    {
        'turns': [
            '我家养了一只猫，叫小白。',
            '我上周下的订单号是 20260815-0042。',
            '退款请退到我尾号 3366 的银行卡。',
            '我的生日是 3 月 14 日。',
            '我的手机是华为 Mate 60。',
            '我一般晚上十点之后才有空。',
            '我特别喜欢重庆火锅。',
            '我住在朝阳区望京街道。',
        ],
        'values': ['小白', '20260815-0042', '3366', '3 月 14 日', 'Mate 60', '晚上十点', '重庆火锅', '望京'],
        'questions': [
            '我家猫叫什么名字？', '我的订单号是多少？', '退款退到哪张卡？', '我生日是哪天？',
            '我用的是什么手机？', '我什么时候有空？', '我喜欢吃什么？', '我住在哪里？',
        ],
    },
]

def build_detail_conversation(length: int, sample: int) -> list[str]:
    """细节评测：8 条细节（开头声明）+ L 条草垛 + 8 条细节提问"""
    d = DETAIL_SETS[sample % len(DETAIL_SETS)]
    turns = list(d['turns'])
    turns.extend(_FILLER[i % len(_FILLER)] for i in range(length))
    turns.extend(d['questions'])
    return turns


def detail_values(sample: int) -> list[str]:
    return DETAIL_SETS[sample % len(DETAIL_SETS)]['values']

_FILLER = [
    '你们商店几点开门？', '营业时间是早上九点到晚上十点。',
    '可以货到付款吗？', '支持货到付款，也可以在线支付。',
    '发货一般要多久？', '现货一般两天内发出。',
    '支持七天无理由退货吗？', '支持的，不影响二次销售即可。',
    '有会员折扣吗？', '有，注册会员享九八折。',
    '可以开发票吗？', '可以，下单备注抬头即可。',
    '你们有没有实体店？', '我们以线上为主，部分城市有体验店。',
    '客服几点下班？', '客服值班到晚上十点。',
    '退货运费谁出？', '质量问题我们承担运费。',
    '有什么优惠活动吗？', '最近有满三百减五十的活动。',
    '可以预约上门取件吗？', '可以，售后页面可预约。',
    '保修期是多久？', '整机保修一年。',
    '能先看下样品吗？', '线上商城有实拍图。',
    '支付方式有哪些？', '支持微信、支付宝和银行卡。',
]


def build_conversation(length: int, sample: int) -> list[str]:
    """事实针 + L 条草垛 + 提问（确定性）"""
    facts = FACT_SETS[sample % len(FACT_SETS)]
    turns = list(facts['turns'])
    turns.extend(_FILLER[i % len(_FILLER)] for i in range(length))
    turns.extend(QUESTIONS)
    return turns


def fact_values(sample: int) -> list[str]:
    facts = FACT_SETS[sample % len(FACT_SETS)]
    return [facts['name'], facts['occupation'], facts['hobby'], facts['constraint']]


# ---------------------------------------------------------------------------
# 模型与用量采集
# ---------------------------------------------------------------------------

class UsageHandler(BaseCallbackHandler):
    def __init__(self):
        self.usages: list[dict] = []

    def on_llm_end(self, response, **kwargs):
        usage = (getattr(response, 'llm_output', None) or {}).get('token_usage') or {}
        if usage:
            self.usages.append({
                'hit': int(usage.get('prompt_cache_hit_tokens', 0) or 0),
                'miss': int(usage.get('prompt_cache_miss_tokens', 0) or 0),
                'out': int(usage.get('completion_tokens', 0) or 0),
            })


def _make_model(handler, schema=None, max_tokens=None, temperature=0.2):
    """结构化输出模型用低温（0.2）：切分/总结/调参是评测链路，需要稳定输出；
    主对话模型（max_tokens 限长）另用 0.8。"""
    kwargs = {'temperature': temperature, 'extra_body': {'thinking': {'type': 'disabled'}}}
    if max_tokens:
        kwargs['max_tokens'] = max_tokens
    llm = init_chat_model('deepseek-v4-flash', **kwargs)
    if schema is not None:
        llm = llm.with_structured_output(schema)
    return llm.with_config({'callbacks': [handler]})


# ---------------------------------------------------------------------------
# 虚拟时钟（--virtual-turn-seconds）
# 成熟度门 M(Δt)、时间衰减 T(m) 都是分钟/小时量级，压缩时间的基准里
# 用真实秒表测不出生产语义（门要么永不成熟、要么一开就每轮跑）。
# 开虚拟时钟后：消息打点、maturity、time_decay 全部走虚拟时间，
# 而"每轮推进多少秒"就是这次测量里"用户一轮对话花了多久"的定义。
# ---------------------------------------------------------------------------

class ClockShim:
    """每轮推进固定秒数的虚拟时钟（提供 .time()，用于顶替模块内的 time 模块）"""

    def __init__(self, start: float, seconds_per_turn: float):
        self.now = start
        self.step = seconds_per_turn

    def time(self) -> float:
        return self.now

    def tick(self) -> None:
        self.now += self.step


CLOCK: ClockShim | None = None
BMDM_OVERRIDES: dict = {}   # BMDM 门控/节奏参数覆盖（CLI：--production-gates 等）
SUMMARIZE_OVERRIDES: dict = {}  # 官方摘要的窗口预算覆盖（CLI：--summarize-trigger 等，用于同阈值对照）


def install_virtual_clock(seconds_per_turn: float) -> ClockShim:
    global CLOCK
    from memory_middleware import middleware as _mw, spliter as _sp, rag as _rag
    CLOCK = ClockShim(start=time.time(), seconds_per_turn=seconds_per_turn)
    for mod in (_mw, _sp, _rag):
        mod.time = CLOCK
    return CLOCK


def _now() -> float:
    return CLOCK.time() if CLOCK else time.time()


def _tick() -> None:
    if CLOCK:
        CLOCK.tick()


# 记忆内容体积计量：两边都按包内默认口径（字数/3.3）算正文 token，
# 口径与中间件内部的 token_counter 一致，避免"用官方 usage 比估算值"的错配。
_TOKEN_COUNTER = default_token_counter()


def _count_text(text: str) -> int:
    return math.ceil(_TOKEN_COUNTER([AIMessage(content=text)])) if text else 0


def summarize_usage(runner, handler, hits, turns: int) -> dict:
    """归档一次运行：总量 + 按事件归一化（主对话恰好每轮 1 次调用 → 内部调用 = 总调用 − 轮数）"""
    calls = len(handler.usages)
    internal = calls - turns
    events = getattr(runner, 'events', 0)
    return {
        'recall': sum(hits),
        'hits': [bool(h) for h in hits],
        'input_tokens': sum(u['hit'] + u['miss'] for u in handler.usages),
        'output_tokens': sum(u['out'] for u in handler.usages),
        'llm_calls': calls,
        'turns': turns,
        'events': events,
        'internal_calls': internal,
        'internal_per_turn': round(internal / turns, 3) if turns else None,
        'internal_per_event': round(internal / events, 3) if events else None,
    }


# ---------------------------------------------------------------------------
# 驱动循环：三种配置共用同一结构与对话文本
# ---------------------------------------------------------------------------

class ConvRunner:
    """最小对话循环（等价于 langgraph 图内行为，手动应用状态更新）"""

    def __init__(self, handler, chat_model):
        self.handler = handler
        self.chat_model = chat_model
        self.state = {
            'messages': [],
            'new_msg_idx': 0,
            'system_prompt': '你是智能客服，请简洁回答用户的问题。',
            'user_profile': '',
        }
        self.replies: list[str] = []
        self.events = 0   # BMDM：检索注入次数；SummarizationMiddleware：摘要触发次数

    def _apply_state_update(self, update):
        if not update:
            return
        for key, value in update.items():
            if key == 'messages' and isinstance(value, list):
                self.state['messages'] = [m for m in value if not isinstance(m, RemoveMessage)]
            else:
                self.state[key] = value

    async def _chat_call(self, messages_for_model):
        """直接调聊天模型并记录回复（基线/summarize 用）"""
        out = await self.chat_model.ainvoke(messages_for_model)
        text = str(out.content)
        self.replies.append(text)
        self.state['messages'].append(AIMessage(
            content=text, additional_kwargs={'time': _now()}))
        return out

    async def turn(self, text: str):
        raise NotImplementedError

    def ask_questions(self, values: list[str]) -> list[bool]:
        """最后 len(values) 条回复中是否包含事实关键词（4 条或 8 条评测通用）。

        匹配前做**空白归一化**：`'3 月 14 日'` 与 `'3月14日'`、`'Mate 60'` 与 `'Mate60'` 视为同一答案。
        动机（2026-10-05 实测）：关键词带空格，模型换个写法就被判成未召回——那是**判分假阴性**，
        不是记忆或模型的缺陷。归一化只消掉格式差异，不放宽语义（不接受"三月十四日"这类改写）。
        """
        last_replies = [_norm(r) for r in self.replies[-len(values):]]
        return [any(_norm(v) in r for r in last_replies) for v in values]


class BaselineRunner(ConvRunner):
    """无记忆基线：上下文只保留最近 BUDGET 条消息"""

    async def turn(self, text: str):
        _tick()
        self.state['messages'].append(HumanMessage(text, additional_kwargs={'time': _now()}))
        recent = self.state['messages'][-BUDGET:]
        await self._chat_call([SystemMessage(content=self.state['system_prompt']), *recent])


class SummarizeRunner(ConvRunner):
    """官方 SummarizationMiddleware：超过阈值把旧消息压成摘要"""

    SUMMARY_MARKER = 'Here is a summary of the conversation to date:'   # langchain 源码固定前缀

    def __init__(self, handler, chat_model):
        super().__init__(handler, chat_model)
        cfg = dict(trigger=BUDGET + 2, keep=4, max_tokens=None)
        cfg.update(SUMMARIZE_OVERRIDES)
        # 摘要模型与聊天模型分离：早期版本共用 max_tokens=100 的聊天模型，摘要被截断到 100 token
        # （对官方不利）。这里给摘要单独一个不设上限、低温的模型——两边都按各自最佳状态比。
        summary_model = _make_model(handler, max_tokens=cfg['max_tokens'], temperature=0.2)
        self.mw = SummarizationMiddleware(
            model=summary_model,
            trigger=('messages', cfg['trigger']),
            keep=('messages', cfg['keep']),
        )
        self._runtime = SimpleNamespace(store=None, context=SimpleNamespace(user_id='bench'))
        self.summary_texts: list[str] = []   # 每次摘要的正文（体积计量的原料）

    async def turn(self, text: str):
        _tick()
        self.state['messages'].append(HumanMessage(text, additional_kwargs={'time': _now()}))
        update = await self.mw.abefore_model(self.state, self._runtime)
        self._apply_state_update(update)
        if update:
            self.events += 1   # 官方中间件真正触发了摘要
            self._capture_summary()
        await self._chat_call([
            SystemMessage(content=self.state['system_prompt']), *self.state['messages']])

    def _capture_summary(self):
        """取出刚写入的摘要正文（官方以带固定前缀的 HumanMessage 插入上下文）"""
        for msg in self.state['messages']:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.startswith(self.SUMMARY_MARKER):
                self.summary_texts.append(content[len(self.SUMMARY_MARKER):].strip())

    def memory_payload(self) -> dict:
        """记忆内容体积：官方写入上下文的摘要正文 token（最后一次为准，累计值另给）"""
        return {
            'summary_tokens_last': _count_text(self.summary_texts[-1]) if self.summary_texts else 0,
            'summary_tokens_total': sum(_count_text(t) for t in self.summary_texts),
            'summary_texts': [t[:400] for t in self.summary_texts],
        }


class BMDMRunner(ConvRunner):
    """本项目中间件：切片 + 增量画像 + 检索注入"""

    def __init__(self, handler, chat_model, tmp_dir: pathlib.Path, embeddings,
                 store=None, kv_store=None):
        super().__init__(handler, chat_model)
        # 默认是"门控不阻塞 + 压缩时间尺度"（A 组）；B 组用 --production-gates + 虚拟时钟跑生产语义
        cfg = dict(trigger_threshold=BUDGET, slice_value=0.0, long_term_value=0.0,
                   tau_m=60.0, c_m=0.5, tau=3600.0, c_t=0.5)
        cfg.update(BMDM_OVERRIDES)
        config = MemoryConfig(
            pattern='messages',
            trigger_threshold=cfg['trigger_threshold'],
            vocation=VocationParams(
                tau_m=cfg['tau_m'], c_m=cfg['c_m'], tau=cfg['tau'], c_t=cfg['c_t'],
                slice_value=cfg['slice_value'], long_term_value=cfg['long_term_value']),
            rag_db_path=str(tmp_dir / 'mem.db'),
            initial_prompt='你是智能客服，请基于用户画像与记忆片段简洁回答。',
            retrieve_k=cfg.get('retrieve_k'),
            top_k=cfg.get('top_k', 4),
            top_k_max_per_theme=cfg.get('top_k_max_per_theme'),
            always_inject_types=cfg.get('always_inject_types', ('identity', 'preference')),
        )
        vec_store = SQLiteVecStore(config.rag_db_path, embeddings)
        self.vec_store = vec_store  # 诊断/报告用
        # 跨会话评测：会话2 通过 store=/kv_store= 复用会话1 的长期记忆与片段游标
        # （生产里对应 Redis 游标跨会话持久；不共享会导致片段 id 从 0 重计撞 UNIQUE）
        self.kv_store = kv_store or MemoryKVStore()
        self.mw = BalancedMultiDimensionMemory(
            config=config,
            summary_llm=_make_model(handler, TimeMemoryFormulaParam),
            spliter=MemoryFragmentsAiSpliter(
                split_llm=_make_model(handler, SummaryMemoryAi),
                kv_store=self.kv_store,
            ),
            rag=FragmentsMemoryRAG(
                summary_llm=_make_model(handler, UserProfile),
                vector_store=vec_store,
            ),
            recovery=NullRecovery(),
        )
        self.store = store or InMemoryStore()
        self._runtime = SimpleNamespace(
            store=self.store, context=SimpleNamespace(user_id='bench'))

    async def _bmdm_handler(self, request):
        """awrap_model_call 的 handler：真实聊天模型 + 记录回复"""
        msgs = ([request.system_message] if request.system_message else []) + request.messages
        out = await self.chat_model.ainvoke(msgs)
        text = str(out.content)
        self.replies.append(text)
        self.state['messages'].append(AIMessage(
            content=text, additional_kwargs={'time': _now()}))
        return ModelResponse(result=[out])

    async def _refresh_user_profile(self):
        """每轮从 store 刷新 user_profile（真实应用由 msg_handle 入口完成；
        基准 runner 需模拟，否则画像层永远为空、BMDM 召回被低估）"""
        item = await self._runtime.store.aget(('long-term', 'user_profile',), 'bench')
        if item is None:
            self.state['user_profile'] = ''
            return
        profile = item.value['user_profile'] or {}
        lines = []
        for key, value in profile.items():
            if key in ('last_summarized_id', 'last_updated'):
                continue
            if value in (None, '', [], {}):
                continue
            lines.append(f"{key}: {value}")
        self.state['user_profile'] = '\n'.join(lines)

    MEMORY_MARKER = '以下为相关片段：'   # SystemPrompt.__str__ 里片段层的分隔标记

    def memory_payload(self) -> dict:
        """记忆内容体积：注入的片段正文 token + 画像 token（分开计，画像层官方没有对应物）"""
        prompt = self.state.get('system_prompt') or ''
        core = getattr(self.mw.config, 'initial_prompt', '') or ''
        body = prompt[len(core):] if core and prompt.startswith(core) else prompt
        profile, fragments = body.split(self.MEMORY_MARKER, 1) if self.MEMORY_MARKER in body else (body, '')
        return {
            'injected_tokens': _count_text(fragments),
            'profile_tokens': _count_text(profile),
            'injected_chars': len(fragments),
            'injected_preview': fragments[:400],
        }

    def diagnostics(self) -> dict:
        """排查用（召回波动诊断）：片段库全貌 + 实际注入过的片段 id + 检索主题 + 最终系统提示词"""
        rows = self.vec_store.fetch_fragments_since('bench', 0)
        ops = self.mw._prompt_operations.get('bench')
        injected = []
        if ops is not None:
            injected = [getattr(d, 'metadata', {}).get('id') for d in
                        getattr(ops.prompt, 'memory_fragments', [])]
        return {
            'theme': self.mw.memory_spliter.dialogue_theme_by_user.get('bench', ''),
            'fragments': [{'id': fid, 'theme': (meta or {}).get('theme', '')} for _, meta, fid in rows],
            'injected_ids': injected,
            'system_prompt': self.state.get('system_prompt', ''),
            # 每次注入事件的打分表（含全部事件，不只最后一次）：R/T/F 各维与加权贡献
            'scoring_log': list(self.mw._scoring_log),
        }

    async def turn(self, text: str):
        _tick()
        self.state['messages'].append(HumanMessage(text, additional_kwargs={'time': _now()}))
        await self.mw.abefore_model(self.state, self._runtime)
        await self._refresh_user_profile()
        request = ModelRequest(
            model=object(), tools=[],
            system_message=SystemMessage(content=self.state['system_prompt']),
            messages=self.state['messages'], state=self.state, runtime=self._runtime,
        )
        result = await self.mw.awrap_model_call(request, self._bmdm_handler)
        if getattr(result, 'command', None) is not None:
            self.state.update(result.command.update)
            self.events += 1   # 一次检索注入 = 一次记忆事件
        await self.mw.aafter_model(self.state, self._runtime)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def make_embeddings(kind: str):
    if kind == 'dashscope' and os.environ.get('DASHSCOPE_API_KEY'):
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings(model='text-embedding-v4')
    print('[注意] 使用 HashEmbeddings（离线，语义质量低）——建议 --embeddings dashscope')
    return HashEmbeddings(dim=256)


async def _run_configs(turns: list[str], values: list[str], tmp_dir: pathlib.Path,
                       embeddings) -> dict:
    """对给定对话与提问跑三种配置，返回各配置召回 + 用量"""
    out = {}
    for name, cls in [('baseline', BaselineRunner),
                      ('summarize', SummarizeRunner),
                      ('bmdm', BMDMRunner)]:
        handler = UsageHandler()
        chat_model = _make_model(handler, max_tokens=CHAT_MAX_TOKENS, temperature=0.8)
        if name == 'bmdm':
            runner = cls(handler, chat_model, tmp_dir, embeddings)
        else:
            runner = cls(handler, chat_model)
        for text in turns:
            await runner.turn(text)
        hits = runner.ask_questions(values)
        out[name] = summarize_usage(runner, handler, hits, len(turns))
        # 判定输入与判定结果一起留档：没有答案原文，判定就无法事后复核/改判
        # （2026-10-05 踩过——判分对空格敏感，而历史答案没存 → 只能重跑，历史数字救不回来）
        out[name]['judge_input'] = {'keywords': list(values),
                                    'answers': runner.replies[-len(values):]}
        m = out[name]
        print(f"    {name:<10} 召回 {sum(hits)}/{len(values)} {hits} | "
              f"调用 {m['llm_calls']}（内部 {m['internal_calls']}）事件 {m['events']} | "
              f"每次事件内部调用 {m['internal_per_event']}")
        if name in ('bmdm', 'summarize'):
            payload = runner.memory_payload()
            out[name]['memory_payload'] = payload
            if name == 'bmdm':
                # 打分表同时进 JSON：留档可解析（日志里另有一份人读的文本）
                out[name]['scoring_log'] = list(runner.mw._scoring_log)
                print(f"      [记忆体积] 注入片段 {payload['injected_tokens']} tok"
                      f"（{payload['injected_chars']} 字符）· 画像 {payload['profile_tokens']} tok")
            else:
                print(f"      [记忆体积] 摘要正文 末次 {payload['summary_tokens_last']} tok · "
                      f"累计 {payload['summary_tokens_total']} tok（{len(payload['summary_texts'])} 次）")
                for t in payload['summary_texts'][-1:]:
                    print(f"      [摘要正文] {t[:300]!r}")
        if name == 'bmdm':
            d = runner.diagnostics()
            print(f"      [诊断] 检索主题={d['theme']!r}")
            print(f"      [诊断] 片段库={d['fragments']}")
            print(f"      [诊断] 注入过的片段 id={d['injected_ids']}")
            print_scoring_log(d.get('scoring_log') or [])
            print(f"      [诊断] 最终系统提示词=\n{d['system_prompt']}")
    return out


async def run_detail_bench(length: int, sample: int, embeddings) -> dict:
    """细节保留评测：8 条低显著度细节 + 草垛 + 8 条细节提问（同会话内）"""
    turns = build_detail_conversation(length, sample)
    values = detail_values(sample)
    tmp_dir = OUT_DIR / 'tmp' / f'd_len{length}_s{sample}'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [细节 length={length} sample={sample}]")
    return await _run_configs(turns, values, tmp_dir, embeddings)


async def run_one(length: int, sample: int, embeddings) -> dict:
    """一个 (长度, 样本) 跑三种配置；每种配置独立用量采集（成本归因干净）"""
    turns = build_conversation(length, sample)
    values = fact_values(sample)
    # 每个 (长度, 样本) 独立向量库目录：并发跑多个进程时避免片段 id 撞唯一约束
    tmp_dir = OUT_DIR / 'tmp' / f'len{length}_s{sample}'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [length={length} sample={sample}]")
    return await _run_configs(turns, values, tmp_dir, embeddings)


async def run_cross_session(length: int, sample: int, embeddings) -> dict:
    """跨会话评测：会话1 埋事实（不提问），会话2 新线程提问。

    - baseline / SummarizationMiddleware：无跨会话机制（摘要存于会话 state，新线程即空）
    - BMDM：会话2 复用同一长期记忆 store（画像 + 暂存片段），验证跨会话持久化
    """
    turns1 = build_conversation(length, sample)[:-4]  # 去掉提问，只埋事实 + 草垛
    values = fact_values(sample)
    tmp_dir = OUT_DIR / 'tmp' / f'xs_len{length}_s{sample}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def _make_runner(name, handler, chat_model, store=None, kv_store=None):
        if name == 'bmdm':
            return BMDMRunner(handler, chat_model, tmp_dir, embeddings,
                              store=store, kv_store=kv_store)
        return {'baseline': BaselineRunner, 'summarize': SummarizeRunner}[name](
            handler, chat_model)

    # ---- 会话1：埋事实（各配置正常运转；BMDM 把画像/片段/游标写入持久态） ----
    bmdm_store, bmdm_kv = None, None
    for name in ('baseline', 'summarize', 'bmdm'):
        handler = UsageHandler()
        chat_model = _make_model(handler, max_tokens=CHAT_MAX_TOKENS, temperature=0.8)
        runner = _make_runner(name, handler, chat_model)
        for text in turns1:
            await runner.turn(text)
        if name == 'bmdm':
            bmdm_store = runner.store
            bmdm_kv = runner.kv_store

    # 会话1 结束后的 BMDM 持久态（诊断/报告用）
    if bmdm_store is not None:
        item = await bmdm_store.aget(('long-term', 'user_profile',), 'bench')
        profile = item.value['user_profile'] if item else None
        print(f"  [会话1结束] BMDM 画像: {profile}")

    # ---- 会话2：全新会话（BMDM 复用 store + 片段游标，其余全新） ----
    # 注意：记忆注入的触发条件是"窗口 ≥ BUDGET 条消息"，会话2 若只有 4 个提问
    # （7 条消息）将永远差一条、注入不生效——需 4 条暖场闲聊把窗口推到触发点，
    # 让 4 条提问都发生在注入生效之后（这也如实反映了"短会话不注入记忆"的特性）。
    turns2 = [_FILLER[i % len(_FILLER)] for i in range(4)] + QUESTIONS
    out = {}
    for name in ('baseline', 'summarize', 'bmdm'):
        handler = UsageHandler()
        chat_model = _make_model(handler, max_tokens=CHAT_MAX_TOKENS, temperature=0.8)
        store = bmdm_store if name == 'bmdm' else None
        kv = bmdm_kv if name == 'bmdm' else None
        runner = _make_runner(name, handler, chat_model, store=store, kv_store=kv)
        for text in turns2:
            await runner.turn(text)
        if name == 'bmdm':
            rows = runner.vec_store.fetch_fragments_since('bench', 0)
            print(f"  [会话2结束] BMDM 主题={runner.mw.memory_spliter.dialogue_theme_by_user!r} "
                  f"片段数={len(rows)} 检索状态={runner.mw._user_retrieve_state!r}")
            print(f"  [会话2结束] BMDM 最后提示词: {runner.state['system_prompt'][:300]!r}")
        hits = runner.ask_questions(values)
        out[name] = summarize_usage(runner, handler, hits, len(turns2))
        out[name]['judge_input'] = {'keywords': list(values),
                                    'answers': runner.replies[-len(values):]}
        m = out[name]
        print(f"  [跨会话 length={length} sample={sample}] {name:<10} 召回 {sum(hits)}/4 {hits} | "
              f"调用 {m['llm_calls']}（内部 {m['internal_calls']}）事件 {m['events']}")
    return out


_WS_RE = re.compile(r'\s+')


def _norm(s: str) -> str:
    """判分归一化：去掉全部空白（只消格式差异，不放宽语义）"""
    return _WS_RE.sub('', s or '')


def _pkg_version() -> str:
    """独立包版本（跑分 _meta 戳；取不到就 unknown）"""
    try:
        from importlib.metadata import version
        return version('memory-middleware')
    except Exception:
        return 'unknown'


def _git_commit() -> str | None:
    """当前仓库短 SHA（跑分 _meta 戳；取不到就 None）"""
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=OUT_DIR.parent.parent, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def run_meta(tag: str) -> dict:
    """本次跑分的「版本 + 时间 + 口径」戳。

    写在**每条记录内部**而不是文件头：save_results 是 existing.update(results)，
    不同 key 会在不同时间被重跑覆盖——记录级戳才答得上"这条数据是哪版、哪时、什么参数跑的"。
    """
    return {
        'version': _pkg_version(),
        'git_commit': _git_commit(),
        'run_at': datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),
        'tag': tag,
        'bmdm_config': {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in BMDM_OVERRIDES.items()},
        'summarize_config': dict(SUMMARIZE_OVERRIDES),
    }


def print_scoring_log(scoring_log: list) -> None:
    """打印每次注入事件的打分表：调参值 + 候选的 R/T/F 与各维加权贡献（`*` = 入选）。

    回答"多维评分里哪一维在决定入选"——纯诊断输出，不参与任何判定。
    """
    if not scoring_log:
        return
    print(f"      [诊断] 打分表（{len(scoring_log)} 次注入事件；αR/βT/γF 为该维的实际加权贡献）")
    for i, ev in enumerate(scoring_log, 1):
        p = ev['params']
        print(f"        事件{i}: α={p['alpha']} β={p['beta']} γ={p['gamma']} δ={p['delta']} "
              f"| w0={p['w0']} w1={p['w1']} w2={p['w2']}")
        for c in sorted(ev['candidates'], key=lambda x: -x['s']):
            mark = '*' if c['picked'] else ' '
            print(f"          {mark} id={c['id']} [{str(c['theme'])[:14]}] "
                  f"{','.join(c['type'] or [])[:18]:<18} str={c['strengthen_num']} "
                  f"age={c['age_s']:>7.1f}s | R={c['r']:.3f} T={c['t']:.3f} F={c['f']:.3f} | "
                  f"αR={c['alpha_r']:.3f} βT={c['beta_t']:.3f} γF={c['gamma_f']:.3f} → S={c['s']:.3f}")


def save_results(results: dict, tag: str = ''):
    """增量保存：本次跑的键盖新戳，已有键连同它们的旧戳原样保留"""
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / 'recall_results.json'
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
    meta = run_meta(tag)
    for record in results.values():
        if isinstance(record, dict) and '_meta' not in record:
            record['_meta'] = meta
    existing.update(results)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')


def _parse_kinds(results: dict) -> dict[str, set[int]]:
    """解析结果键 → {评测类型: {长度集合}}。键形如 '8-0' / 'xs-8-0' / 'd-8-0'"""
    kinds: dict[str, set[int]] = {}
    for k in results.keys():
        parts = k.split('-')
        if parts[0] in ('xs', 'd'):
            kind, length = parts[0], int(parts[1])
        else:
            kind, length = '', int(parts[0])
        kinds.setdefault(kind, set()).add(length)
    return kinds


def _key_matches(k: str, prefix: str, length: int, tag: str) -> bool:
    """结果键形如 {prefix}{length}-{sample}{tag}。

    tag='' 只匹配**无后缀**的原始跑分（避免把不同门控口径平均进同一条折线）；
    给定 tag 时只匹配该口径（如 tag='-fair3' 匹配 d-24-0-fair3）。
    """
    head = f'{prefix}{length}-'
    if not k.startswith(head):
        return False
    rest = k[len(head):]
    return rest.endswith(tag) if tag else '-' not in rest


def load_results() -> dict:
    """读取已保存的全部跑分（出图用：出图必须看全量，不能只看本次运行的 key）"""
    path = OUT_DIR / 'recall_results.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def make_chart_for(results: dict, kind: str = '', questions: int = 4,
                   out_name: str = 'recall_benchmark.png', tag: str = ''):
    """通用出图：折线（常规/细节）或分组柱状（跨会话）。配色为 dataviz 验证过的 1-3 号槽位。

    tag 指定口径后缀（''=无后缀的原始跑分；'-gated'/'-fair3' 等=该口径），
    **不同口径绝不混进同一条折线**。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lengths = sorted(_parse_kinds(results).get(kind, []))
    if not lengths:
        return
    prefix = f'{kind}-' if kind else ''
    configs = ['bmdm', 'summarize', 'baseline']
    labels = {'bmdm': 'BMDM (ours)', 'summarize': 'SummarizationMiddleware', 'baseline': 'No-memory baseline'}
    colors = {'bmdm': '#2a78d6', 'summarize': '#eb6834', 'baseline': '#1baf7a'}
    # 同一长度上三条线可能取同值（如 24 轮 BMDM 与官方都是 75%）：不同点型 + 主序列点更大，
    # 避免被后画的线完全盖住
    markers = {'bmdm': 'o', 'summarize': 's', 'baseline': '^'}
    marker_sizes = {'bmdm': 12, 'summarize': 7, 'baseline': 9}
    INK_SECONDARY = '#52514e'  # 文字用墨色，不用系列色（dataviz 规范）

    def recall_for(cfg, length):
        runs = [v[cfg] for k, v in results.items()
                if _key_matches(k, prefix, length, tag) and cfg in v]
        if not runs:
            return None
        return sum(r['recall'] for r in runs) / (questions * len(runs)) * 100

    # 该 tag 下一份数据都没有 → 不覆盖已有图（否则会用空图盖掉别的口径出的图）
    if not any(recall_for(cfg, l) is not None for cfg in configs for l in lengths):
        print(f'（跳过 {out_name}：tag={tag!r} 下无数据）')
        return

    fig, ax = plt.subplots(figsize=(7, 4.6))
    if kind == 'xs':
        # 跨会话：分组柱状图
        width = 0.26
        for i, cfg in enumerate(configs):
            vals = [recall_for(cfg, length) for length in lengths]
            if all(v is None for v in vals):
                continue
            vals = [v or 0 for v in vals]
            xs = [x + (i - 1) * width for x in range(len(lengths))]
            bars = ax.bar(xs, vals, width=width, label=labels[cfg], color=colors[cfg])
            for bar, v in zip(bars, vals):
                ax.annotate(f'{v:.0f}%', (bar.get_x() + bar.get_width() / 2, v),
                            textcoords='offset points', xytext=(0, 4),
                            ha='center', fontsize=9, color=INK_SECONDARY)
        ax.set_xticks(range(len(lengths)))
        ax.set_xticklabels([f'{l} turns' for l in lengths])
        ax.set_ylabel(f'Recall in session 2 ({questions} facts) (%)')
        ax.set_title('Cross-session recall: facts from session 1,\n'
                     'questions in a brand-new session')
    else:
        for cfg in configs:
            pts = [(l, recall_for(cfg, l)) for l in lengths]
            pts = [(l, v) for l, v in pts if v is not None]
            if not pts:
                continue
            xs_, ys = zip(*pts)
            ax.plot(xs_, ys, marker=markers[cfg], markersize=marker_sizes[cfg], linewidth=2,
                    label=labels[cfg], color=colors[cfg])
            # 选择性直接标注：只标主序列（bmdm）与基线（baseline）的端点
            if cfg in ('bmdm', 'baseline'):
                ax.annotate(f'{ys[-1]:.0f}%', (xs_[-1], ys[-1]),
                            textcoords='offset points', xytext=(0, 9),
                            ha='center', fontsize=9, color=INK_SECONDARY)
        ax.set_xlabel('Conversation length (turns)')
        ax.set_ylabel(f'Fact recall rate ({questions} facts) (%)')
        if kind == 'd':
            ax.set_title('Detail retention: BMDM vs baselines\n'
                         '(8 low-salience facts · matched trigger budget · 3 samples)')
        else:
            ax.set_title('Long-term memory recall: BMDM vs baselines\n'
                         '(needle-in-haystack, 4 facts, context budget = 8 messages)')

    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False)
    ax.grid(axis='y' if kind == 'xs' else 'both', alpha=0.25, linewidth=0.8, color='#e1e0d9')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=160)
    print(f'图表已保存：{out}')


def make_all_charts(results: dict, tag: str = ''):
    """为结果中存在的评测类型出图（只画指定口径 tag 的那一档）"""
    specs = {
        '': dict(questions=4, out_name='recall_benchmark.png'),
        'xs': dict(questions=4, out_name='cross_session_recall.png'),
        'd': dict(questions=8, out_name='detail_retention.png'),
    }
    for kind in _parse_kinds(results):
        spec = specs.get(kind)
        if spec:
            make_chart_for(results, kind=kind, tag=tag, **spec)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lengths', default='8,16,24')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--embeddings', default='dashscope', choices=['dashscope', 'hash'])
    parser.add_argument('--chart-only', action='store_true')
    parser.add_argument('--cross-session', action='store_true',
                        help='跨会话评测：会话1 埋事实，会话2 新线程提问')
    parser.add_argument('--details', action='store_true',
                        help='细节保留评测：8 条低显著度细节（订单号/卡尾号等）')
    parser.add_argument('--virtual-turn-seconds', type=float, default=0.0,
                        help='虚拟时钟：每轮推进的虚拟秒数（0 = 真实时间）。'
                             '用生产门控（τ_m 分钟量级）测时必须开启，否则门要么永不成熟、要么一开每轮跑')
    parser.add_argument('--production-gates', action='store_true',
                        help='BMDM 生产门控预设：slice=0.7 / long_term=0.9 / tau_m=600 / tau=43200')
    parser.add_argument('--slice-value', type=float, default=None, help='BMDM 切片成熟度门')
    parser.add_argument('--long-term-value', type=float, default=None, help='BMDM 归纳门')
    parser.add_argument('--trigger-threshold', type=int, default=None, help='BMDM 主触发窗口预算（条消息）')
    parser.add_argument('--tau-m', type=float, default=None, help='BMDM 成熟度时间尺度（秒）')
    parser.add_argument('--tau', type=float, default=None, help='BMDM 时间衰减尺度（秒）')
    parser.add_argument('--top-k', type=int, default=None, help='BMDM 注入片段数（默认 3）')
    parser.add_argument('--retrieve-k', type=int, default=None, help='BMDM 检索候选数（默认 6）')
    parser.add_argument('--max-per-theme', type=int, default=None,
                        help='BMDM 注入集合里单主题上限（默认不限；防同主题占满名额）')
    parser.add_argument('--always-inject-types', default=None,
                        help='BMDM 保底类型（逗号分隔，如 identity,preference）：这些类型的片段优先占席')
    parser.add_argument('--summarize-trigger', type=int, default=None,
                        help='官方摘要的窗口预算（条消息）。与 --trigger-threshold 取同值即为同阈值对照')
    parser.add_argument('--summarize-keep', type=int, default=None, help='官方摘要保留的最近消息条数')
    parser.add_argument('--summarize-max-tokens', type=int, default=None,
                        help='官方摘要输出上限（默认不限；早期版本误用聊天模型的 100）')
    parser.add_argument('--tag', default='', help='结果 key 后缀（区分不同门控口径，避免互相覆盖）')
    parser.add_argument('--rejudge', action='store_true',
                        help='不跑模型，用当前判分规则重判已留档的答案（judge_input）并报告差异；'
                             '只对 2026-10-05 之后留了答案原文的记录有效')
    parser.add_argument('--no-chart', action='store_true',
                        help='跑完不重画图表。默认会用**本次 tag** 重画对应图——诊断/临时跑分请加这个开关，'
                             '否则会把已发布的 detail_retention.png 覆盖成本次单样本的口径'
                             '（恢复：--chart-only --tag=-fixed3）')
    parser.add_argument('--fresh', action='store_true',
                        help='跑之前清空 benchmarks/output/tmp：片段 id 计数器在内存 KV 里、向量库是持久的，'
                             '复用旧库会让 id 从 1 重算并撞 UNIQUE 约束（生产中 Redis 计数丢失同理）')
    args = parser.parse_args()

    if args.chart_only:
        chart_only(args.tag)
        return
    if args.rejudge:
        rejudge()
        return

    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('[错误] 需要 DEEPSEEK_API_KEY（主项目 .env）')

    if args.virtual_turn_seconds > 0:
        install_virtual_clock(args.virtual_turn_seconds)
        print(f'[虚拟时钟] 每轮推进 {args.virtual_turn_seconds:.0f} 秒'
              f'（消息打点 / 成熟度 M(Δt) / 时间衰减 T(m) 均走虚拟时间）')

    gates: dict = {}
    if args.production_gates:
        gates.update(slice_value=0.7, long_term_value=0.9, tau_m=600.0, tau=43200.0)
    for name, value in (('slice_value', args.slice_value),
                        ('long_term_value', args.long_term_value),
                        ('trigger_threshold', args.trigger_threshold),
                        ('tau_m', args.tau_m), ('tau', args.tau),
                        ('top_k', args.top_k), ('retrieve_k', args.retrieve_k),
                        ('top_k_max_per_theme', args.max_per_theme)):
        if value is not None:
            gates[name] = value
    if args.always_inject_types:
        gates['always_inject_types'] = tuple(
            t.strip() for t in args.always_inject_types.split(',') if t.strip())
    if gates:
        BMDM_OVERRIDES.update(gates)
        print(f'[BMDM 门控] {gates}')

    summarize: dict = {}
    for name, value in (('trigger', args.summarize_trigger),
                        ('keep', args.summarize_keep),
                        ('max_tokens', args.summarize_max_tokens)):
        if value is not None:
            summarize[name] = value
    if summarize:
        SUMMARIZE_OVERRIDES.update(summarize)
        print(f'[官方摘要] {summarize}（与 --trigger-threshold 同值即为同阈值对照）')

    embeddings = make_embeddings(args.embeddings)
    if args.fresh:
        import shutil
        tmp_root = OUT_DIR / 'tmp'
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        print('[fresh] 已清空 benchmarks/output/tmp（各配置从空库起跑）')
    lengths = [int(x) for x in args.lengths.split(',')]
    results = {}
    for length in lengths:
        for sample in range(args.samples):
            if args.cross_session:
                key = f'xs-{length}-{sample}' + args.tag
                print(f'===== 跨会话：会话1 长度={length}，样本={sample} =====')
                results[key] = await run_cross_session(length, sample, embeddings)
            elif args.details:
                key = f'd-{length}-{sample}' + args.tag
                print(f'===== 细节保留：长度={length}，样本={sample} =====')
                results[key] = await run_detail_bench(length, sample, embeddings)
            else:
                key = f'{length}-{sample}' + args.tag
                print(f'===== 会话长度={length}，样本={sample} =====')
                results[key] = await run_one(length, sample, embeddings)
            save_results(results, args.tag)  # 增量保存：中断后可 --chart-only 或续跑

    print('\n===== 汇总 =====')
    for key, value in results.items():
        print(f"  {key}: " + ", ".join(
            f"{cfg}: {v['recall']}/{len(v['hits'])} ({v['input_tokens']} in, {v['llm_calls']} calls)"
            for cfg, v in value.items() if cfg in ('baseline', 'summarize', 'bmdm')))
    # 出图看全量（含历史口径），但只画本次 tag 那一档——不同口径绝不混进同一条折线
    if args.no_chart:
        print('（--no-chart：跳过出图）')
    else:
        make_all_charts(load_results(), tag=args.tag)


def rejudge() -> None:
    """用**当前判分规则**重判已留档的答案（不跑模型，零成本）。

    只对带 `judge_input` 的记录有效——2026-10-05 之前的跑分没存答案原文，无法改判，只能重跑。
    本命令只出报告、不改数据（避免无意中改写归档）。
    """
    results = load_results()
    total, changed = 0, []
    for key, rec in results.items():
        if not isinstance(rec, dict):
            continue
        for cfg in ('baseline', 'summarize', 'bmdm'):
            v = rec.get(cfg)
            if not isinstance(v, dict):
                continue
            ji = v.get('judge_input')
            if not ji:
                continue
            total += 1
            old = v.get('hits')
            new = [any(_norm(k) in _norm(a) for a in ji['answers']) for k in ji['keywords']]
            if old != new:
                changed.append((key, cfg, old, new))
    print(f'可重判记录 {total} 条（{len(results)} 个 key）')
    if not total:
        print('（没有带 judge_input 的记录：2026-10-05 之前的跑分未留档答案原文，只能重跑）')
    print(f'按当前判分规则（空白归一化）判定发生变化的：{len(changed)} 条')
    for key, cfg, old, new in changed:
        o = ''.join('1' if h else '0' for h in old)
        n = ''.join('1' if h else '0' for h in new)
        print(f'  {key} [{cfg}] {sum(old)}→{sum(new)} ({o} → {n})')


def chart_only(tag: str = ''):
    results = load_results()
    if not results:
        raise SystemExit(f'无结果文件：{OUT_DIR / "recall_results.json"}，请先运行基准')
    make_all_charts(results, tag=tag)


if __name__ == '__main__':
    asyncio.run(main())
