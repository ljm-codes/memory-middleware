"""KV 存储抽象：上游项目（fireflymall-ai-customer-service，
https://github.com/fufuxiaokeai/fireflymall-ai-customer-service）的 Redis
只承担"片段序号游标"（memory_fragments:{user_id} 计数器）。

- MemoryKVStore：单进程离线实现（测试/演示/单进程部署），不保证多进程一致
- RedisKVStore  ：生产实现，包装任意 async redis 兼容客户端（如 redis.asyncio）
"""
from typing import Optional, Protocol


class KVStore(Protocol):
    async def aget(self, key: str) -> Optional[str]: ...

    async def aset(self, key: str, value: str, ex: Optional[int] = None) -> None: ...


class MemoryKVStore:
    """单进程内存实现：测试与离线演示用。多进程部署请使用 RedisKVStore。"""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def aget(self, key: str) -> Optional[str]:
        return self._data.get(key)

    async def aset(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self._data[key] = value


class RedisKVStore:
    """生产实现：包装 redis.asyncio 客户端（连接池由调用方管理）。

    示例：
        import redis.asyncio as aioredis
        con = aioredis.Redis(connection_pool=...)
        kv = RedisKVStore(con)
    """

    def __init__(self, con) -> None:
        self._con = con

    async def aget(self, key: str) -> Optional[str]:
        return await self._con.get(key)

    async def aset(self, key: str, value: str, ex: Optional[int] = None) -> None:
        await self._con.set(key, value, ex=ex)
