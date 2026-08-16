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
"""
import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / '.env')

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
            content=text, additional_kwargs={'time': time.time()}))
        return out

    async def turn(self, text: str):
        raise NotImplementedError

    def ask_questions(self, values: list[str]) -> list[bool]:
        """最后 len(values) 条回复中是否包含事实关键词（4 条或 8 条评测通用）"""
        last_replies = self.replies[-len(values):]
        return [any(v in r for r in last_replies) for v in values]


class BaselineRunner(ConvRunner):
    """无记忆基线：上下文只保留最近 BUDGET 条消息"""

    async def turn(self, text: str):
        self.state['messages'].append(HumanMessage(text, additional_kwargs={'time': time.time()}))
        recent = self.state['messages'][-BUDGET:]
        await self._chat_call([SystemMessage(content=self.state['system_prompt']), *recent])


class SummarizeRunner(ConvRunner):
    """官方 SummarizationMiddleware：超过阈值把旧消息压成摘要"""

    def __init__(self, handler, chat_model):
        super().__init__(handler, chat_model)
        self.mw = SummarizationMiddleware(
            model=chat_model,
            trigger=('messages', BUDGET + 2),
            keep=('messages', 4),
        )
        self._runtime = SimpleNamespace(store=None, context=SimpleNamespace(user_id='bench'))

    async def turn(self, text: str):
        self.state['messages'].append(HumanMessage(text, additional_kwargs={'time': time.time()}))
        update = await self.mw.abefore_model(self.state, self._runtime)
        self._apply_state_update(update)
        await self._chat_call([
            SystemMessage(content=self.state['system_prompt']), *self.state['messages']])


class BMDMRunner(ConvRunner):
    """本项目中间件：切片 + 增量画像 + 检索注入"""

    def __init__(self, handler, chat_model, tmp_dir: pathlib.Path, embeddings,
                 store=None, kv_store=None):
        super().__init__(handler, chat_model)
        config = MemoryConfig(
            pattern='messages',
            trigger_threshold=BUDGET,
            vocation=VocationParams(
                tau_m=60, c_m=0.5, tau=3600, c_t=0.5,
                slice_value=0.0, long_term_value=0.0),
            rag_db_path=str(tmp_dir / 'mem.db'),
            initial_prompt='你是智能客服，请基于用户画像与记忆片段简洁回答。',
            retrieve_k=6,
            top_k=3,
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
            content=text, additional_kwargs={'time': time.time()}))
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

    async def turn(self, text: str):
        self.state['messages'].append(HumanMessage(text, additional_kwargs={'time': time.time()}))
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
        out[name] = {
            'recall': sum(hits),
            'hits': [bool(h) for h in hits],
            'input_tokens': sum(u['hit'] + u['miss'] for u in handler.usages),
            'output_tokens': sum(u['out'] for u in handler.usages),
            'llm_calls': len(handler.usages),
        }
        print(f"    {name:<10} 召回 {sum(hits)}/{len(values)} {hits}")
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
        out[name] = {
            'recall': sum(hits),
            'hits': [bool(h) for h in hits],
            'input_tokens': sum(u['hit'] + u['miss'] for u in handler.usages),
            'output_tokens': sum(u['out'] for u in handler.usages),
            'llm_calls': len(handler.usages),
        }
        print(f"  [跨会话 length={length} sample={sample}] {name:<10} 召回 {sum(hits)}/4 {hits}")
    return out


def save_results(results: dict):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / 'recall_results.json'
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
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


def make_chart_for(results: dict, kind: str = '', questions: int = 4,
                   out_name: str = 'recall_benchmark.png'):
    """通用出图：折线（常规/细节）或分组柱状（跨会话）。配色为 dataviz 验证过的 1-3 号槽位。"""
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
    INK_SECONDARY = '#52514e'  # 文字用墨色，不用系列色（dataviz 规范）

    def recall_for(cfg, length):
        runs = [v[cfg] for k, v in results.items()
                if k.startswith(f'{prefix}{length}-') and cfg in v]
        return sum(r['recall'] for r in runs) / (questions * len(runs)) * 100

    fig, ax = plt.subplots(figsize=(7, 4.6))
    if kind == 'xs':
        # 跨会话：分组柱状图
        width = 0.26
        for i, cfg in enumerate(configs):
            vals = [recall_for(cfg, length) for length in lengths]
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
            ys = [recall_for(cfg, length) for length in lengths]
            ax.plot(lengths, ys, marker='o', markersize=8, linewidth=2,
                    label=labels[cfg], color=colors[cfg])
            # 选择性直接标注：只标主序列（bmdm）与基线（baseline）的端点
            if cfg in ('bmdm', 'baseline'):
                ax.annotate(f'{ys[-1]:.0f}%', (lengths[-1], ys[-1]),
                            textcoords='offset points', xytext=(0, 9),
                            ha='center', fontsize=9, color=INK_SECONDARY)
        ax.set_xlabel('Conversation length (turns)')
        ax.set_ylabel(f'Fact recall rate ({questions} facts) (%)')
        if kind == 'd':
            ax.set_title('Detail retention: BMDM vs baselines\n'
                         '(8 low-salience facts: numbers, names, tails — '
                         'the details summaries tend to drop)')
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


def make_all_charts(results: dict):
    """为结果中存在的评测类型出图"""
    specs = {
        '': dict(questions=4, out_name='recall_benchmark.png'),
        'xs': dict(questions=4, out_name='cross_session_recall.png'),
        'd': dict(questions=8, out_name='detail_retention.png'),
    }
    for kind in _parse_kinds(results):
        spec = specs.get(kind)
        if spec:
            make_chart_for(results, kind=kind, **spec)


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
    args = parser.parse_args()

    if args.chart_only:
        chart_only()
        return

    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('[错误] 需要 DEEPSEEK_API_KEY（主项目 .env）')

    embeddings = make_embeddings(args.embeddings)
    lengths = [int(x) for x in args.lengths.split(',')]
    results = {}
    for length in lengths:
        for sample in range(args.samples):
            if args.cross_session:
                key = f'xs-{length}-{sample}'
                print(f'===== 跨会话：会话1 长度={length}，样本={sample} =====')
                results[key] = await run_cross_session(length, sample, embeddings)
            elif args.details:
                key = f'd-{length}-{sample}'
                print(f'===== 细节保留：长度={length}，样本={sample} =====')
                results[key] = await run_detail_bench(length, sample, embeddings)
            else:
                key = f'{length}-{sample}'
                print(f'===== 会话长度={length}，样本={sample} =====')
                results[key] = await run_one(length, sample, embeddings)
            save_results(results)  # 增量保存：中断后可 --chart-only 或续跑

    print('\n===== 汇总 =====')
    for key, value in results.items():
        print(f"  {key}: " + ", ".join(
            f"{cfg}: {v['recall']}/{len(v['hits'])} ({v['input_tokens']} in, {v['llm_calls']} calls)"
            for cfg, v in value.items()))
    make_all_charts(results)


def chart_only():
    path = OUT_DIR / 'recall_results.json'
    if not path.exists():
        raise SystemExit(f'无结果文件：{path}，请先运行基准')
    results = json.loads(path.read_text(encoding='utf-8'))
    make_all_charts(results)


if __name__ == '__main__':
    asyncio.run(main())
