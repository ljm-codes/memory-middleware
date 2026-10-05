"""向量暂存库：sqlite-vec 本地实现 + 确定性离线嵌入。

- CustomizeSQLiteVec：从上游项目 fireflymall-ai-customer-service
  （https://github.com/lijia-ming/fireflymall-ai-customer-service）
  Tools/middleware/memory/customize_sqlite_vec.py 提炼（仅替换了日志为标准库，其余逻辑一致）
- HashEmbeddings：确定性哈希嵌入（离线可用，语义质量低，生产请换真实嵌入）
- SQLiteVecStore：中间件使用的窄接口（增/删/检索 + 归纳游标查询）
"""
import asyncio
import hashlib
import json
import logging
import math
import sqlite3
import struct
import uuid
import warnings
from typing import Any, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


def _serialize_f32(vector: List[float]) -> bytes:
    """将浮点数列表序列化为紧凑的"原始字节"格式（sqlite-vec 要求）"""
    return struct.pack("%sf" % len(vector), *vector)


class CustomizeSQLiteVec(VectorStore):
    """自定义的 SQLiteVec 向量数据库（本地文件，无外部服务）"""

    def __init__(
            self,
            table: str,
            connection: Optional[sqlite3.Connection],
            embedding: Embeddings,
            db_file: str = "vec.db",
            is_async: Optional[bool] = False,
    ):
        try:
            import sqlite_vec  # noqa
        except ImportError:
            raise ImportError(
                "Could not import sqlite-vec python package. "
                "Please install it with `pip install sqlite-vec`."
            )

        if not connection:
            connection = self.create_connection(db_file, is_async)

        if not isinstance(embedding, Embeddings):
            warnings.warn("embeddings input must be Embeddings object.")

        self._connection = connection
        self._table = table
        self._embedding = embedding

        self.create_table_if_not_exists()

    def create_table_if_not_exists(self) -> None:
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table}
            (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE,
                text TEXT,
                metadata BLOB,
                text_embedding BLOB
            )
            ;
            """
        )
        self._connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self._table}_vec USING vec0(
                rowid INTEGER PRIMARY KEY,
                text_embedding float[{self.get_dimensionality()}]
                distance_metric=cosine
            )
            ;
            """
        )
        self._connection.execute(
            f"""
                CREATE TRIGGER IF NOT EXISTS {self._table}_embed_text
                AFTER INSERT ON {self._table}
                BEGIN
                    INSERT INTO {self._table}_vec(rowid, text_embedding)
                    VALUES (new.rowid, new.text_embedding)
                    ;
                END;
            """
        )
        self._connection.execute(
            f"""
                CREATE TRIGGER IF NOT EXISTS {self._table}_delete_vec
                AFTER DELETE ON {self._table}
                BEGIN
                    DELETE FROM {self._table}_vec WHERE rowid = old.rowid;
                END;
                """
        )
        self._connection.commit()

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        max_id = self._connection.execute(
            f"SELECT max(rowid) as rowid FROM {self._table}"
        ).fetchone()["rowid"]
        if max_id is None:  # no text added yet
            max_id = 0

        embeds = self._embedding.embed_documents(list(texts))
        if not metadatas:
            metadatas = [{} for _ in texts]
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        data_input = [
            (id, text, json.dumps(metadata), _serialize_f32(embed))
            for id, text, metadata, embed in zip(ids, texts, metadatas, embeds)
        ]
        self._connection.executemany(
            f"INSERT INTO {self._table}(id, text, metadata, text_embedding) VALUES (?,?,?,?)",
            data_input,
        )
        self._connection.commit()
        results = self._connection.execute(
            f"SELECT rowid FROM {self._table} WHERE rowid > {max_id}"
        )
        return [str(row["rowid"]) for row in results]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if not ids:
            raise ValueError("Must provide ids to delete.")
        placeholders = ",".join(["?" for _ in ids])
        self._connection.execute(
            f"DELETE FROM {self._table} WHERE id IN ({placeholders})", ids
        )
        self._connection.commit()
        return True

    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """按元数据过滤的余弦相似检索（filter 用 json_extract 匹配 metadata 键）"""
        filter_clause = ""
        params: list = [_serialize_f32(embedding), k]
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(f"json_extract(e.metadata, '$.{key}') = ?")
                params.append(value)
            if conditions:
                filter_clause = " AND " + " AND ".join(conditions)

        sql_query = f"""
            SELECT
                id,
                text,
                metadata,
                distance
            FROM {self._table} AS e
            INNER JOIN {self._table}_vec AS v on v.rowid = e.rowid
            WHERE
                v.text_embedding MATCH ?
                AND k = ?
                {filter_clause}
            ORDER BY distance
        """
        cursor = self._connection.cursor()
        cursor.execute(sql_query, params)
        results = cursor.fetchall()

        documents = []
        for row in results:
            metadata = json.loads(row["metadata"]) or {}
            doc = Document(page_content=row["text"], metadata=metadata, id=row["id"])
            documents.append((doc, row["distance"]))

        return documents

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k
        )
        return [doc for doc, _ in documents]

    def similarity_search_with_score(
            self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return documents

    def similarity_search_by_vector(
            self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k
        )
        return [doc for doc, _ in documents]

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            table: str = "langchain",
            db_file: str = "vec.db",
            **kwargs: Any,
    ) -> "CustomizeSQLiteVec":
        connection = cls.create_connection(db_file)
        vec = cls(
            table=table, connection=connection, db_file=db_file, embedding=embedding
        )
        vec.add_texts(texts=texts, metadatas=metadatas)
        return vec

    @staticmethod
    def create_connection(db_file: str, is_async: bool = False) -> sqlite3.Connection:
        import sqlite3
        import sqlite_vec

        check = not is_async
        connection = sqlite3.connect(db_file, check_same_thread=check)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        return connection

    def get_dimensionality(self) -> int:
        dummy_text = "This is a dummy text"
        dummy_embedding = self._embedding.embed_query(dummy_text)
        return len(dummy_embedding)


class HashEmbeddings(Embeddings):
    """确定性哈希嵌入：纯离线（演示/测试），语义质量低，生产请接入真实嵌入模型。

    基于字符双元组 hash 到固定维向量并归一化；同一文本始终得到同一向量。
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        text = text.lower()
        if len(text) < 2:
            h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
            vec[h % self.dim] = 1.0
        else:
            for i in range(len(text) - 1):
                pair = text[i:i + 2]
                h = int(hashlib.md5(pair.encode('utf-8')).hexdigest(), 16)
                vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


@runtime_checkable
class FragmentIdSource(Protocol):
    """片段 id / 消息偏移的持久化高水位来源（KV 计数器丢失后自愈用）"""

    def max_fragment_ids(self, user_id: str) -> Tuple[int, int]:
        """返回 (最大片段 id, 最大已切分消息数)；无数据时 (0, 0)"""


class SQLiteVecStore:
    """中间件使用的向量库窄接口：sqlite-vec 本地实现（文件/内存库均可）。

    同时持有两个连接：
    - 写入/检索连接（CustomizeSQLiteVec，langchain VectorStore 的 async 默认实现）
    - 只读连接（归纳游标查询 fetch_fragments_since 用）
    """

    def __init__(self, db_path: str, embedding: Embeddings, table: str = 'memory_fragments'):
        self._vec = CustomizeSQLiteVec(
            table=table,
            connection=CustomizeSQLiteVec.create_connection(db_path, is_async=True),
            embedding=embedding,
            db_file=db_path,
            is_async=True,
        )
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA busy_timeout=5000")
        self._table = table
        # sqlite 单连接不支持并发操作：中间件的锁是"按用户"的，跨用户写必须在此串行化
        self._lock = asyncio.Lock()

    async def aadd_documents(self, documents: List[Document]) -> None:
        async with self._lock:
            await self._vec.aadd_documents(documents)

    async def asimilarity_search_with_score(
            self, query: str, k: Optional[int] = None, filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        async with self._lock:
            if k is None:
                # 不限候选 = 返回该用户全部片段。sqlite-vec 本就是全表算距离再取前 k，
                # 差别只是返回多少行，向量计算成本不变。
                k = 4096
            return await self._vec.asimilarity_search_with_score(query=query, k=k, filter=filter)

    async def adelete(self, ids: List[str]) -> None:
        async with self._lock:
            await self._vec.adelete(ids=ids)

    def fetch_fragments_since(self, user_id: str, last_summary_id: int) -> List[Tuple[str, dict, int]]:
        """归纳游标之后的新片段 [(text, metadata, fid)]，按 fid 升序（增量总结的原材料）"""
        cursor = self._con.cursor()
        cursor.execute(
            f"select text, metadata, cast(json_extract(metadata, '$.id') as integer) as fid "
            f"from {self._table} "
            f"where json_extract(metadata, '$.user_id') = ? "
            f"and cast(json_extract(metadata, '$.id') as integer) > ? "
            f"order by fid",
            (user_id, last_summary_id),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [(r['text'], json.loads(r['metadata']) or {}, r['fid']) for r in rows]

    def max_fragment_ids(self, user_id: str) -> Tuple[int, int]:
        """该用户片段的高水位：(最大片段 id, 最大已切分消息数)。

        片段 id 计数器活在 KV 里、片段本体在这里；KV 一丢（进程重启 / Redis 无持久化 /
        换库重跑）计数器就从 1 重算并撞 UNIQUE 约束，故用库里的持久高水位兜底。
        msg_end 由写入端打在每次切分的最后一片 metadata 上；旧数据没这个键 → 取 0，
        行为与兜底前一致，无需迁移。
        """
        cursor = self._con.cursor()
        cursor.execute(
            f"select max(cast(json_extract(metadata, '$.id') as integer)) as max_fid, "
            f"max(cast(json_extract(metadata, '$.msg_end') as integer)) as max_offset "
            f"from {self._table} where json_extract(metadata, '$.user_id') = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        return (int(row['max_fid'] or 0), int(row['max_offset'] or 0))

    def close(self) -> None:
        try:
            self._con.close()
        finally:
            self._vec._connection.close()
