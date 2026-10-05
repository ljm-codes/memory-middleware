# -*- coding: utf-8 -*-
"""Layer 0：split_text（切分输出 → 记忆片段）的防御性解析测试"""
import time

from langchain_core.messages import AIMessage, HumanMessage

from memory_middleware.spliter import MemoryFragmentsAiSpliter


def _messages(n=4, tool_call_at=None):
    """构造 n 条带时间戳的消息；tool_call_at 指定位置放一条纯工具调用的 AIMessage"""
    msgs = []
    now = time.time()
    for i in range(n):
        if tool_call_at is not None and i == tool_call_at:
            msgs.append(AIMessage(
                content='',
                tool_calls=[{'name': 'search_goods', 'args': {'q': '手机'}, 'id': 't1'}],
                additional_kwargs={'time': now - (n - i)},
            ))
        elif i % 2 == 0:
            msgs.append(HumanMessage(f'用户消息{i}', additional_kwargs={'time': now - (n - i)}))
        else:
            msgs.append(AIMessage(f'AI回复{i}', additional_kwargs={'time': now - (n - i)}))
    return msgs


def _conf(start_idx, end_idx, theme='主题', types=None):
    from memory_middleware.models import SummaryMemoryFragmentsConfig
    return SummaryMemoryFragmentsConfig(theme=theme, type=types or ['chat'],
                                        start_idx=start_idx, end_idx=end_idx)


class TestValidScopes:
    def test_basic_split(self):
        msgs = _messages(4)
        text = [f'对应的消息索引{i}' for i in range(4)]
        result = MemoryFragmentsAiSpliter.split_text(1, [_conf(0, 2)], text, msgs)
        assert len(result) == 1
        assert result[0].config.theme == '主题'
        assert result[0].config.time == msgs[1].additional_kwargs['time']  # 片段内最后一条消息
        assert result[0].content == '\n'.join(text[0:2])  # 左闭右开


class TestDefensiveParsing:
    def test_zero_width_range_takes_single_message(self):
        # 零宽/反向区间由模型校验兜底为 end_idx = start_idx + 1：只取 start_idx 那一条
        msgs = _messages(4)
        text = [f'i{i}' for i in range(4)]
        for start, end in ((1, 1), (1, 0), (1, -5)):
            result = MemoryFragmentsAiSpliter.split_text(1, [_conf(start, end)], text, msgs)
            assert len(result) == 1, (start, end)
            assert result[0].content == text[1]
            assert result[0].config.time == msgs[1].additional_kwargs['time']

    def test_negative_start_clamped_to_zero(self):
        # 负索引由模型校验归零
        msgs = _messages(4)
        text = [f'i{i}' for i in range(4)]
        result = MemoryFragmentsAiSpliter.split_text(1, [_conf(-3, 1)], text, msgs)
        assert len(result) == 1
        assert result[0].content == text[0]

    def test_out_of_bounds_end_clamped(self):
        # LLM 越界：end_idx 超过消息末尾时钳到 len(messages)
        msgs = _messages(4)
        text = [f'i{i}' for i in range(4)]
        result = MemoryFragmentsAiSpliter.split_text(1, [_conf(0, 99)], text, msgs)
        assert len(result) == 1
        assert result[0].content == '\n'.join(text[0:4])

    def test_start_beyond_messages_skipped(self):
        # start_idx 越界（依赖消息条数，模型校验管不了）→ 跳过而不是崩溃
        msgs = _messages(4)
        text = [f'i{i}' for i in range(4)]
        result = MemoryFragmentsAiSpliter.split_text(1, [_conf(9, 12)], text, msgs)
        assert result == []

    def test_theme_num_mismatch_truncated(self):
        # 主题数与配置数不一致：截断处理而不是抛异常
        msgs = _messages(4)
        text = [f'i{i}' for i in range(4)]
        result = MemoryFragmentsAiSpliter.split_text(
            2, [_conf(0, 1), _conf(1, 2), _conf(2, 4)], text, msgs)
        assert len(result) == 2

    def test_zero_theme_num_empty(self):
        msgs = _messages(4)
        text = [f'i{i}' for i in range(4)]
        assert MemoryFragmentsAiSpliter.split_text(0, [], text, msgs) == []


class TestIndexAlignment:
    def test_tool_call_placeholder_keeps_alignment(self):
        # 纯工具调用（空 content）必须占位，否则 LLM 返回的消息索引错位
        msgs = _messages(4, tool_call_at=1)
        text = [f'对应的消息索引{i}，ai:xxx' if i == 1 else f'i{i}' for i in range(4)]
        result = MemoryFragmentsAiSpliter.split_text(1, [_conf(1, 3)], text, msgs)
        assert len(result) == 1
        # 片段时间取 messages[end-1] = 索引 2 的消息
        assert result[0].config.time == msgs[2].additional_kwargs['time']
