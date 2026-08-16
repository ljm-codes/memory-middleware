"""失败保底机制抽象：切分/总结 LLM 失败时的恢复钩子。

上游项目（fireflymall-ai-customer-service，https://github.com/fufuxiaokeai/fireflymall-ai-customer-service）
原机制依赖 RabbitMQ（持久队列重试 + 邮件告警）；独立版拆为可插拔协议：
- NullRecovery（默认）：失败记日志、返回 None（上层跳过该轮），离线可用
- RabbitMQRecovery（可选）：与项目版一致——失败时把原始数据投递到持久队列，
  由恢复消费者在 LLM 恢复后重新处理；未提供时请用 NullRecovery
"""
import asyncio
import logging
from typing import Any, Callable, List, Optional, Protocol, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel as PydanticModel
from pydantic import Field, field_validator

logger = logging.getLogger(__name__)


class ErrorRecovery(Protocol):
    async def on_split_error(self, user_id: str, messages: List[BaseMessage], error: Exception) -> None:
        ...

    async def on_summary_error(self, user_id: str, text: str, last_summary_id: int, error: Exception) -> None:
        ...


class NullRecovery:
    """离线默认：只记日志。失败轮次由调用方跳过，不影响后续消息处理。"""

    async def on_split_error(self, user_id: str, messages: List[BaseMessage], error: Exception) -> None:
        logger.error(f"记忆切分失败（user_id={user_id}）: {error}")

    async def on_summary_error(self, user_id: str, text: str, last_summary_id: int, error: Exception) -> None:
        logger.error(f"画像总结失败（user_id={user_id}）: {error}")


class ErrorFragmentsData(PydanticModel):
    """投递到恢复队列的载荷（与上游项目 memory_rag.ErrorFragmentsData 一致）"""
    type: str = Field(default='spliter', description="错误引发的源头：spliter | summary")
    user_id: str = Field(default='-1', description="用户ID")
    text: Union[str, List[Any]] = Field(default='', description="原始消息列表（spliter）或未区分的文本（summary）")
    last_summary_id: Optional[int] = Field(default=None, description='增量总结游标，供消费者恢复进度')

    @field_validator('text', mode='before')
    @classmethod
    def _restore_messages(cls, v):
        """pydantic 反序列化 List[BaseMessage] 按 type 字段还原类型"""
        if isinstance(v, list) and v and all(isinstance(item, dict) for item in v):
            msg_type_map = {
                'human': HumanMessage,
                'ai': AIMessage,
                'tool': ToolMessage,
                'system': SystemMessage,
            }
            restored = []
            for item in v:
                msg_type = item.get('type', '')
                msg_cls = msg_type_map.get(msg_type, BaseMessage)
                fields = {k: val for k, val in item.items() if k != 'type'}
                try:
                    restored.append(msg_cls(**fields))
                except Exception as e:
                    logger.warning(f"恢复消息 {item} 时出错: {e}")
                    restored.append(BaseMessage(**{**fields, 'type': msg_type}))
            return restored
        return v


class RabbitMQRecovery:
    """生产可选保底：切分/总结失败时把原始数据发布到持久队列，LLM 恢复后重试。

    用法：
        recovery = RabbitMQRecovery(
            url='amqp://user:pass@host:5672/',
            notifier=lambda msg: print(msg),   # 可选：超时/异常告警回调
        )
        middleware = BalancedMultiDimensionMemory(recovery=recovery)

    注意：需要 `pip install aio-pika`。恢复消费者端请参考上游项目
    memory_rag.py 的 _recover_split / _recover_summary（按 type 字段分发处理）。
    """

    def __init__(
            self,
            url: str,
            exchange: str = 'rag_error',
            split_queue: str = 'uncut_text',
            summary_queue: str = 'unclassified_fragments',
            routing_prefix: str = 'data',
            notifier: Optional[Callable[[str], None]] = None,
    ):
        self.url = url
        self.exchange = exchange
        self.split_queue = split_queue
        self.summary_queue = summary_queue
        self.routing_prefix = routing_prefix
        self.notifier = notifier
        self._conn = None
        self._channel = None
        self._lock = asyncio.Lock()

    async def _ensure_ready(self):
        if self._channel is not None:
            return self._channel
        async with self._lock:
            if self._channel is not None:
                return self._channel
            from aio_pika import ExchangeType, connect_robust
            self._conn = await connect_robust(self.url)
            self._channel = await self._conn.channel()
            exchange = await self._channel.declare_exchange(
                self.exchange, ExchangeType.DIRECT, durable=True)
            for queue_name in (self.split_queue, self.summary_queue):
                queue = await self._channel.declare_queue(queue_name, durable=True)
                await queue.bind(exchange, routing_key=f"{queue_name}_{self.routing_prefix}")
            return exchange

    async def _publish(self, queue_name: str, payload: ErrorFragmentsData):
        from aio_pika import DeliveryMode, Message
        exchange = await self._ensure_ready()
        await exchange.publish(
            Message(
                body=payload.model_dump_json().encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                content_type='application/json',
            ),
            routing_key=f"{queue_name}_{self.routing_prefix}",
        )

    async def on_split_error(self, user_id: str, messages: List[BaseMessage], error: Exception) -> None:
        try:
            payload = ErrorFragmentsData(
                type='spliter',
                user_id=user_id,
                text=[m.model_dump(mode='json') for m in messages],
            )
            await self._publish(self.split_queue, payload)
        except Exception as e:
            logger.error(f"投递切分失败消息到 RabbitMQ 失败: {e}")
        if self.notifier:
            self.notifier(f"记忆切分异常（user_id={user_id}）: {error}")

    async def on_summary_error(self, user_id: str, text: str, last_summary_id: int, error: Exception) -> None:
        try:
            payload = ErrorFragmentsData(
                type='summary',
                user_id=user_id,
                text=text,
                last_summary_id=last_summary_id,
            )
            await self._publish(self.summary_queue, payload)
        except Exception as e:
            logger.error(f"投递总结失败消息到 RabbitMQ 失败: {e}")
        if self.notifier:
            self.notifier(f"画像总结异常（user_id={user_id}）: {error}")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._channel = None
