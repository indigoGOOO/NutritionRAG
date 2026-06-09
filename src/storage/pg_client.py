"""PostgreSQL 连接器

存储结构化数据和KV对（EAV Schema）
支持CRUD操作和批量操作
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch

from config.settings import PG_CONFIG
from src.indexing.models import ContentChunk

logger = logging.getLogger(__name__)


class PostgreSQLClient:
    """PostgreSQL 客户端 - 存储结构化数据和KV对"""

    def __init__(self, config: dict | None = None):
        """
        初始化PostgreSQL连接

        Args:
            config: 数据库配置（如果为None则使用settings中的配置）
        """
        self.config = config or PG_CONFIG
        self.conn = None
        self.connect()

    def connect(self):
        """建立数据库连接"""
        try:
            self.conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                database=self.config["database"],
                user=self.config["user"],
                password=self.config["password"],
            )
            logger.info(f"PostgreSQL连接成功: {self.config['host']}:{self.config['port']}")
        except Exception as e:
            logger.error(f"PostgreSQL连接失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("PostgreSQL连接已关闭")

    def init_tables(self):
        """初始化数据库表"""
        with self.conn.cursor() as cur:
            # 创建chunks表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    chunk_type VARCHAR(50),
                    doc_category VARCHAR(50),
                    source_doc_id VARCHAR(255),
                    token_count INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(source_doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
                CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks(doc_category);
            """)

            # 创建KV对表（EAV Schema）
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kv_pairs (
                    kv_id SERIAL PRIMARY KEY,
                    entity_id VARCHAR(255) NOT NULL,
                    entity_type VARCHAR(50),
                    attribute VARCHAR(255) NOT NULL,
                    value JSONB NOT NULL,
                    source_chunk_id INTEGER REFERENCES chunks(chunk_id),
                    confidence FLOAT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_kv_entity ON kv_pairs(entity_id);
                CREATE INDEX IF NOT EXISTS idx_kv_attribute ON kv_pairs(attribute);
                CREATE INDEX IF NOT EXISTS idx_kv_type ON kv_pairs(entity_type);
            """)

            # 创建关系三元组表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS triples (
                    triple_id SERIAL PRIMARY KEY,
                    subject VARCHAR(255) NOT NULL,
                    predicate VARCHAR(255) NOT NULL,
                    object VARCHAR(255) NOT NULL,
                    source_chunk_id INTEGER REFERENCES chunks(chunk_id),
                    confidence FLOAT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
                CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
                CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    alias_id SERIAL PRIMARY KEY,
                    canonical_entity_id VARCHAR(255) NOT NULL,
                    alias VARCHAR(255) NOT NULL,
                    alias_type VARCHAR(50),
                    language VARCHAR(20),
                    source VARCHAR(255),
                    confidence FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (canonical_entity_id, alias)
                );
                CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias ON entity_aliases(alias);
                CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical
                    ON entity_aliases(canonical_entity_id);
            """)

            self.conn.commit()
            logger.info("数据库表初始化完成")

    def init_runtime_tables(self):
        """Initialize request-level runtime observability tables."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_runtime_logs (
                    id SERIAL PRIMARY KEY,
                    request_id VARCHAR(64) UNIQUE NOT NULL,
                    session_id VARCHAR(255),
                    user_id VARCHAR(255),
                    query TEXT,
                    intent VARCHAR(100),
                    intent_confidence FLOAT,
                    personalization_mode VARCHAR(50),
                    private_content_required BOOLEAN,
                    private_content_found BOOLEAN,
                    planned_routes JSONB,
                    executed_routes JSONB,
                    fallback_routes JSONB,
                    route_status JSONB,
                    route_errors JSONB,
                    route_decision JSONB,
                    trace JSONB,
                    evidence_summary JSONB,
                    answer TEXT,
                    citations JSONB,
                    latency_ms INTEGER,
                    status VARCHAR(50),
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_agent_runtime_request
                    ON agent_runtime_logs(request_id);
                CREATE INDEX IF NOT EXISTS idx_agent_runtime_session
                    ON agent_runtime_logs(session_id);
                CREATE INDEX IF NOT EXISTS idx_agent_runtime_user
                    ON agent_runtime_logs(user_id);
                CREATE INDEX IF NOT EXISTS idx_agent_runtime_created
                    ON agent_runtime_logs(created_at);
            """)
            self.conn.commit()

    def insert_runtime_log(self, log: dict[str, Any]) -> int | None:
        """Insert one request-level runtime log snapshot."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agent_runtime_logs (
                    request_id,
                    session_id,
                    user_id,
                    query,
                    intent,
                    intent_confidence,
                    personalization_mode,
                    private_content_required,
                    private_content_found,
                    planned_routes,
                    executed_routes,
                    fallback_routes,
                    route_status,
                    route_errors,
                    route_decision,
                    trace,
                    evidence_summary,
                    answer,
                    citations,
                    latency_ms,
                    status,
                    error
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (request_id)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    error = EXCLUDED.error,
                    latency_ms = EXCLUDED.latency_ms,
                    answer = EXCLUDED.answer,
                    citations = EXCLUDED.citations
                RETURNING id;
            """, (
                log.get("request_id"),
                log.get("session_id"),
                log.get("user_id"),
                log.get("query"),
                log.get("intent"),
                log.get("intent_confidence"),
                log.get("personalization_mode"),
                log.get("private_content_required"),
                log.get("private_content_found"),
                json.dumps(log.get("planned_routes", []), ensure_ascii=False),
                json.dumps(log.get("executed_routes", []), ensure_ascii=False),
                json.dumps(log.get("fallback_routes", []), ensure_ascii=False),
                json.dumps(log.get("route_status", {}), ensure_ascii=False),
                json.dumps(log.get("route_errors", []), ensure_ascii=False),
                json.dumps(log.get("route_decision", {}), ensure_ascii=False),
                json.dumps(log.get("trace", {}), ensure_ascii=False),
                json.dumps(log.get("evidence_summary", []), ensure_ascii=False),
                log.get("answer"),
                json.dumps(log.get("citations", []), ensure_ascii=False),
                log.get("latency_ms"),
                log.get("status"),
                log.get("error"),
            ))
            row = cur.fetchone()
            self.conn.commit()
            return row[0] if row else None

    def insert_chunk(self, chunk: ContentChunk) -> int:
        """
        插入单个chunk

        Args:
            chunk: ContentChunk对象

        Returns:
            插入的chunk_id
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chunks (content, chunk_type, doc_category, source_doc_id, token_count, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING chunk_id;
            """, (
                chunk.content,
                chunk.chunk_type,
                chunk.doc_category.value if chunk.doc_category else None,
                chunk.source_doc_id,
                chunk.token_count,
                json.dumps(chunk.metadata, ensure_ascii=False),
            ))
            chunk_id = cur.fetchone()[0]
            self.conn.commit()
            return chunk_id

    def insert_chunks_batch(self, chunks: list[ContentChunk]) -> list[int]:
        """
        批量插入chunks

        Args:
            chunks: ContentChunk列表

        Returns:
            插入的chunk_id列表
        """
        chunk_ids = []
        with self.conn.cursor() as cur:
            for chunk in chunks:
                cur.execute("""
                    INSERT INTO chunks (content, chunk_type, doc_category, source_doc_id, token_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING chunk_id;
                """, (
                    chunk.content,
                    chunk.chunk_type,
                    chunk.doc_category.value if chunk.doc_category else None,
                    chunk.source_doc_id,
                    chunk.token_count,
                json.dumps(chunk.metadata, ensure_ascii=False),
                ))
                chunk_id = cur.fetchone()[0]
                chunk_ids.append(chunk_id)
            self.conn.commit()
        return chunk_ids

    def insert_kv_pair(
        self,
        entity_id: str,
        entity_type: str,
        attribute: str,
        value: Any,
        source_chunk_id: int | None = None,
        confidence: float | None = None,
        metadata: dict | None = None,
    ) -> int:
        """
        插入KV对（EAV Schema）

        Args:
            entity_id: 实体ID
            entity_type: 实体类型
            attribute: 属性名
            value: 属性值
            source_chunk_id: 源chunk ID
            confidence: 置信度
            metadata: 元数据

        Returns:
            插入的kv_id
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO kv_pairs (entity_id, entity_type, attribute, value, source_chunk_id, confidence, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING kv_id;
            """, (
                entity_id,
                entity_type,
                attribute,
                json.dumps(value, ensure_ascii=False),
                source_chunk_id,
                confidence,
                json.dumps(metadata, ensure_ascii=False) if metadata else None,
            ))
            kv_id = cur.fetchone()[0]
            self.conn.commit()
            return kv_id

    def insert_kv_pairs_batch(self, kv_pairs: list[dict]) -> list[int]:
        """
        批量插入KV对

        Args:
            kv_pairs: KV对列表，每个元素包含：
                - entity_id: 实体ID
                - entity_type: 实体类型
                - attribute: 属性名
                - value: 属性值
                - source_chunk_id: 源chunk ID（可选）
                - confidence: 置信度（可选）
                - metadata: 元数据（可选）

        Returns:
            插入的kv_id列表
        """
        kv_ids = []
        with self.conn.cursor() as cur:
            for kv in kv_pairs:
                cur.execute("""
                    INSERT INTO kv_pairs (entity_id, entity_type, attribute, value, source_chunk_id, confidence, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING kv_id;
                """, (
                    kv.get("entity_id"),
                    kv.get("entity_type"),
                    kv.get("attribute"),
                    json.dumps(kv.get("value"), ensure_ascii=False),
                    kv.get("source_chunk_id"),
                    kv.get("confidence"),
                    json.dumps(kv.get("metadata"), ensure_ascii=False) if kv.get("metadata") else None,
                ))
                kv_id = cur.fetchone()[0]
                kv_ids.append(kv_id)
            self.conn.commit()
        return kv_ids

    def insert_triple(
        self,
        subject: str,
        predicate: str,
        object_: str,
        source_chunk_id: int | None = None,
        confidence: float | None = None,
        metadata: dict | None = None,
    ) -> int:
        """
        插入关系三元组

        Args:
            subject: 主体
            predicate: 谓词
            object_: 客体
            source_chunk_id: 源chunk ID
            confidence: 置信度
            metadata: 元数据

        Returns:
            插入的triple_id
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO triples (subject, predicate, object, source_chunk_id, confidence, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING triple_id;
            """, (
                subject,
                predicate,
                object_,
                source_chunk_id,
                confidence,
                json.dumps(metadata, ensure_ascii=False) if metadata else None,
            ))
            triple_id = cur.fetchone()[0]
            self.conn.commit()
            return triple_id

    def insert_triples_batch(self, triples: list[dict]) -> list[int]:
        """
        批量插入关系三元组

        Args:
            triples: 三元组列表，每个元素包含：
                - subject: 主体
                - predicate: 谓词
                - object: 客体
                - source_chunk_id: 源chunk ID（可选）
                - confidence: 置信度（可选）
                - metadata: 元数据（可选）

        Returns:
            插入的triple_id列表
        """
        triple_ids = []
        with self.conn.cursor() as cur:
            for triple in triples:
                cur.execute("""
                    INSERT INTO triples (subject, predicate, object, source_chunk_id, confidence, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING triple_id;
                """, (
                    triple.get("subject"),
                    triple.get("predicate"),
                    triple.get("object"),
                    triple.get("source_chunk_id"),
                    triple.get("confidence"),
                    json.dumps(triple.get("metadata"), ensure_ascii=False) if triple.get("metadata") else None,
                ))
                triple_id = cur.fetchone()[0]
                triple_ids.append(triple_id)
            self.conn.commit()
        return triple_ids

    def insert_entity_alias(
        self,
        canonical_entity_id: str,
        alias: str,
        alias_type: str = "name",
        language: str = "",
        source: str = "",
        confidence: float = 1.0,
    ) -> int | None:
        """Insert or update one entity alias."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO entity_aliases
                    (canonical_entity_id, alias, alias_type, language, source, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (canonical_entity_id, alias)
                DO UPDATE SET
                    alias_type = EXCLUDED.alias_type,
                    language = EXCLUDED.language,
                    source = EXCLUDED.source,
                    confidence = GREATEST(entity_aliases.confidence, EXCLUDED.confidence)
                RETURNING alias_id;
            """, (
                canonical_entity_id,
                alias,
                alias_type,
                language,
                source,
                confidence,
            ))
            row = cur.fetchone()
            self.conn.commit()
            return row[0] if row else None

    def insert_entity_aliases_batch(self, aliases: list[dict]) -> list[int]:
        """Batch insert or update entity aliases."""
        alias_ids = []
        with self.conn.cursor() as cur:
            for item in aliases:
                cur.execute("""
                    INSERT INTO entity_aliases
                        (canonical_entity_id, alias, alias_type, language, source, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (canonical_entity_id, alias)
                    DO UPDATE SET
                        alias_type = EXCLUDED.alias_type,
                        language = EXCLUDED.language,
                        source = EXCLUDED.source,
                        confidence = GREATEST(entity_aliases.confidence, EXCLUDED.confidence)
                    RETURNING alias_id;
                """, (
                    item.get("canonical_entity_id"),
                    item.get("alias"),
                    item.get("alias_type", "name"),
                    item.get("language", ""),
                    item.get("source", ""),
                    item.get("confidence", 1.0),
                ))
                row = cur.fetchone()
                if row:
                    alias_ids.append(row[0])
            self.conn.commit()
        return alias_ids

    def resolve_entity_alias(self, alias: str) -> str | None:
        """Resolve an alias to its canonical entity id."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT canonical_entity_id
                FROM entity_aliases
                WHERE alias = %s
                ORDER BY confidence DESC, alias_id ASC
                LIMIT 1;
            """, (alias,))
            row = cur.fetchone()
            return row["canonical_entity_id"] if row else None

    def query_chunks_by_doc_id(self, doc_id: str) -> list[dict]:
        """
        按文档ID查询chunks

        Args:
            doc_id: 文档ID

        Returns:
            chunk列表
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM chunks WHERE source_doc_id = %s ORDER BY chunk_id;
            """, (doc_id,))
            return cur.fetchall()

    def query_kv_by_entity(self, entity_id: str) -> list[dict]:
        """
        按实体ID查询KV对

        Args:
            entity_id: 实体ID

        Returns:
            KV对列表
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM kv_pairs WHERE entity_id = %s ORDER BY kv_id;
            """, (entity_id,))
            return cur.fetchall()

    def query_triples_by_subject(self, subject: str) -> list[dict]:
        """
        按主体查询三元组

        Args:
            subject: 主体

        Returns:
            三元组列表
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM triples WHERE subject = %s ORDER BY triple_id;
            """, (subject,))
            return cur.fetchall()

    def query_triples_by_object(self, object_: str) -> list[dict]:
        """
        按客体查询三元组（反向查找）

        Args:
            object_: 客体

        Returns:
            三元组列表
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM triples WHERE object = %s ORDER BY triple_id;
            """, (object_,))
            return cur.fetchall()

    def query_triples_by_predicate(self, predicate: str, limit: int = 50) -> list[dict]:
        """
        按谓词查询三元组

        Args:
            predicate: 谓词
            limit: 最大返回数

        Returns:
            三元组列表
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM triples WHERE predicate = %s ORDER BY triple_id LIMIT %s;
            """, (predicate, limit))
            return cur.fetchall()

    def query_triples_by_entity(self, entity: str) -> list[dict]:
        """
        按实体双向查询三元组（subject 或 object 匹配）

        Args:
            entity: 实体名称

        Returns:
            三元组列表
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM triples
                WHERE subject = %s OR object = %s
                ORDER BY triple_id;
            """, (entity, entity))
            return cur.fetchall()

    def query_triples_by_entities(self, entities: list[str], limit: int = 50) -> list[dict]:
        """
        批量按实体查询三元组（双向）

        Args:
            entities: 实体名称列表
            limit: 最大返回数

        Returns:
            三元组列表
        """
        if not entities:
            return []
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT ON (subject, predicate, object) *
                FROM triples
                WHERE subject = ANY(%s) OR object = ANY(%s)
                ORDER BY subject, predicate, object
                LIMIT %s;
            """, (entities, entities, limit))
            return cur.fetchall()

    def query_kv_by_entity_batch(self, entity_ids: list[str]) -> list[dict]:
        """
        批量按实体ID查询KV对

        Args:
            entity_ids: 实体ID列表

        Returns:
            KV对列表
        """
        if not entity_ids:
            return []
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT ON (entity_id, attribute) *
                FROM kv_pairs WHERE entity_id = ANY(%s)
                ORDER BY entity_id, attribute;
            """, (entity_ids,))
            return cur.fetchall()

    def query_relation_report(self, entity_id: str) -> dict:
        """聚合实体在 PG 中的完整关系报告

        Args:
            entity_id: 实体名

        Returns:
            {
                "entity": entity_id,
                "type": entity_type 或 "unknown",
                "attributes": {attr: value, ...},  来自 kv_pairs
                "relations_as_subject": [(predicate, object), ...], 来自 triples
                "relations_as_object": [(predicate, subject), ...],  来自 triples
            }
        """
        kvs = self.query_kv_by_entity(entity_id)
        triples_out = self.query_triples_by_subject(entity_id)
        triples_in = self.query_triples_by_object(entity_id)

        entity_type = "unknown"
        attributes = {}
        for kv in kvs:
            if kv["attribute"] == "_entity_type":
                entity_type = kv.get("value", {}).get("type", "unknown")
            # Merge all attributes into a flat dict
            val = kv.get("value", {})
            if isinstance(val, dict):
                attrs = val.get("attributes", val)
                if isinstance(attrs, dict):
                    attributes.update(attrs)

        return {
            "entity": entity_id,
            "type": entity_type,
            "attributes": attributes,
            "relations_as_subject": [
                (r["predicate"], r["object"]) for r in triples_out
            ],
            "relations_as_object": [
                (r["predicate"], r["subject"]) for r in triples_in
            ],
        }

    def delete_chunk(self, chunk_id: int):
        """删除chunk"""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE chunk_id = %s;", (chunk_id,))
            self.conn.commit()

    def delete_chunks_by_doc_id(self, doc_id: str):
        """按文档ID删除chunks"""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source_doc_id = %s;", (doc_id,))
            self.conn.commit()
