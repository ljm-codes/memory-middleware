# -*- coding: utf-8 -*-
"""共享测试工具（不从 conftest 导入，避免与 tests/integration/conftest.py 命名冲突）"""
import time

from langchain_core.documents import Document


class DictEmbeddings:
    """测试用可控嵌入：文本 → 预置向量；未注册文本返回零向量（与任何向量 cos=0）。

    向量均为单位向量，cosine 值即点积，方便直接断言相似/不相似。
    """

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []  # 记录每次 embed_documents 的入参，验证 API 触发次数

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [self.mapping.get(t, [0.0, 0.0, 0.0, 0.0]) for t in texts]

    def embed_query(self, text):
        return self.mapping.get(text, [0.0, 0.0, 0.0, 0.0])


class ExplodingEmbeddings:
    """一旦被调用就抛错：用于验证小列表场景下嵌入 API 不应被触发"""

    def embed_documents(self, texts):
        raise AssertionError("不应调用嵌入 API")

    def embed_query(self, text):
        raise AssertionError("不应调用嵌入 API")


# 单位向量：同义对 cos≥0.98，无关对 cos=0，临界对 cos=0.8（< 0.85 阈值）
SYNONYM_MAP = {
    '不吃香菜': [1.0, 0.0, 0.0, 0.0],
    '忌香菜': [0.99, 0.141, 0.0, 0.0],
    '不要香菜': [0.98, 0.199, 0.0, 0.0],
    '少糖': [0.0, 1.0, 0.0, 0.0],
    '不要太甜': [0.1, 0.995, 0.0, 0.0],
    '喜欢拍照': [0.0, 0.0, 1.0, 0.0],
    '周末爬山': [0.0, 0.0, 0.0, 1.0],
    '少放盐': [0.8, 0.6, 0.0, 0.0],  # 与"不吃香菜" cos=0.8，低于阈值
}


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
