# -*- coding: utf-8 -*-
"""共享测试工具（不从 conftest 导入，避免与 tests/integration/conftest.py 命名冲突）"""
import time

from langchain_core.documents import Document


class _FakeLLM:
    """基类：记录调用，返回固定结果；可配置抛出异常"""

    def __init__(self, result=None, raise_error=None):
        self.result = result
        self.raise_error = raise_error
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.raise_error is not None:
            raise self.raise_error
        return self.result


class FakeSplitLLM(_FakeLLM):
    """切分模型：返回固定的 SummaryMemoryAi"""


class FakeSummaryLLM(_FakeLLM):
    """总结模型：返回固定的 UserProfile"""


class FakeParamLLM(_FakeLLM):
    """调参模型：返回固定的 TimeMemoryFormulaParam"""


class SpyRecovery:
    """记录错误回调的假保底机制"""

    def __init__(self):
        self.split_errors = []
        self.summary_errors = []

    async def on_split_error(self, user_id, messages, error):
        self.split_errors.append((user_id, list(messages), error))

    async def on_summary_error(self, user_id, text, last_summary_id, error):
        self.summary_errors.append((user_id, text, last_summary_id, error))


def make_fragment(user_id: str, fid: int, content: str = '用户喜欢讨论手机话题',
                  theme: str = '手机', types=None, age: float = 100.0):
    """构造带完整元数据的记忆片段 Document（id 全局唯一：user-fid）"""
    return Document(
        id=f"{user_id}-{fid}",
        page_content=content,
        metadata={
            'theme': theme,
            'type': types or ['chat'],
            'time': time.time() - age,
            'strengthen_num': 0,
            'user_id': user_id,
            'id': fid,
        },
    )
