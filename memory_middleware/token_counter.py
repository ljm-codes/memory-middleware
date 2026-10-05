"""token 估算：可注入的 TokenCounter。

上游项目（fireflymall-ai-customer-service，https://github.com/lijia-ming/fireflymall-ai-customer-service）
用 tiktoken/transformers 精确计数；独立版默认使用"字数/3.3"启发式估算，
测试与轻量场景足够，精确计数请注入自定义实现。
"""
from typing import Iterable, Protocol

from langchain_core.messages import MessageLikeRepresentation


class TokenCounter(Protocol):
    def __call__(self, messages: Iterable[MessageLikeRepresentation]) -> int:
        ...


def default_token_counter() -> TokenCounter:
    """字数 / 3.3 约等于 1 token（与上游项目 TokenCalculatorFactory 的兜底口径一致）"""

    def count_token(messages: Iterable[MessageLikeRepresentation]) -> int:
        total = 0
        for message in messages:
            total += len(message.content) / 3.3
        return total

    return count_token
