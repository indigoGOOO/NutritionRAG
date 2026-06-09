"""历史高质量问答记忆 - Milvus + PostgreSQL

定位：这是问答经验缓存，不是权威知识库。主知识来源仍然是文档、
结构化数据库和图谱；历史QA只在相似问题出现时作为上下文参考。

策略：
- 写入前做质量评估，避免把低质量/失败回答沉淀为长期记忆
- 写入前做相似问题查重，高相似时合并旧记录而不是重复新增
- 检索命中后按相似度和意图给出复用策略：direct_reuse / context_only / ignore
- 医疗、疾病、个性化饮食类默认只作为上下文，不直接复用
"""
# tags意图类型
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import datetime
from typing import Any

from config.settings import (
    KNOWLEDGE_CONTEXT_THRESHOLD,
    KNOWLEDGE_DEDUPE_THRESHOLD,
    KNOWLEDGE_MIN_ANSWER_CHARS,
    KNOWLEDGE_MIN_QUALITY_SCORE,
    KNOWLEDGE_REUSE_THRESHOLD,
    KNOWLEDGE_SAFE_REUSE_INTENTS,
    KNOWLEDGE_UNSAFE_REUSE_INTENTS,
)
from src.memory.base import BaseMemory, KnowledgeEntry, MemoryItem
from src.storage.milvus_client import MilvusClient
from src.storage.pg_client import PostgreSQLClient

logger = logging.getLogger(__name__)

MEMORY_COLLECTION = "knowledge_memory"
EMBEDDING_DIM = 384


class KnowledgeMemory(BaseMemory):
    """历史高质量问答缓存"""

    def __init__(
        self,
        pg: PostgreSQLClient,
        milvus: MilvusClient,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = KNOWLEDGE_CONTEXT_THRESHOLD,
    ):
        self.pg = pg
        self._milvus = milvus
        self.embedding_model = embedding_model
        self._encoder = None
        self.similarity_threshold = similarity_threshold
        self._milvus_ready = False

        self._init_pg_table()
        logger.debug("[KM] Milvus 将在首次读写时初始化")

    @property
    def encoder(self):
        """延迟加载向量模型；不可用时调用处会用fallback向量。"""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder

    # ---- 初始化 ----

    def _init_pg_table(self):
        with self.pg.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_memory (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    entities JSONB DEFAULT '[]',
                    tags JSONB DEFAULT '[]',
                    hit_count INTEGER DEFAULT 0,
                    rating FLOAT DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_km_tags ON knowledge_memory USING gin(tags);
                CREATE INDEX IF NOT EXISTS idx_km_hit ON knowledge_memory(hit_count DESC);
            """)
            cur.execute(
                "ALTER TABLE knowledge_memory ADD COLUMN IF NOT EXISTS quality_score FLOAT DEFAULT 0.0;"
            )
            cur.execute(
                "ALTER TABLE knowledge_memory ADD COLUMN IF NOT EXISTS source_citations JSONB DEFAULT '[]';"
            )
            cur.execute(
                "ALTER TABLE knowledge_memory ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"
            )
        self.pg.conn.commit()

    def _ensure_milvus(self) -> bool:
        """延迟初始化 Milvus 集合；不可用则降级为仅 PG 模式。"""
        if self._milvus_ready:
            return True

        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

        try:
            self._milvus_collection = Collection(MEMORY_COLLECTION)
            logger.info(f"[KM] 连接已有集合: {MEMORY_COLLECTION}")
            self._milvus_ready = True
            return True
        except Exception:
            try:
                fields = [
                    FieldSchema(name="km_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1024),
                    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                ]
                schema = CollectionSchema(fields, description="Q&A知识记忆")
                collection = Collection(name=MEMORY_COLLECTION, schema=schema)
                collection.create_index(
                    "dense_vector",
                    {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}},
                )
                self._milvus_collection = collection
                self._milvus_ready = True
                logger.info(f"[KM] 创建集合: {MEMORY_COLLECTION}")
                return True
            except Exception as e:
                logger.warning(f"[KM] Milvus 不可用，降级为仅 PG 模式: {e}")
                self._milvus_ready = False
                return False

    # ---- 写入 ----

    def store_qa(
        self,
        question: str,
        answer: str,
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        citations: list[dict] | None = None,
        evidence_count: int = 0,
    ) -> int | None:
        """存储一条高质量问答；质量不足时返回 None。"""
        entities = entities or []
        tags = tags or []
        citations = citations or []

        quality = self.evaluate_quality(answer, citations, tags, evidence_count)
        if quality["score"] < KNOWLEDGE_MIN_QUALITY_SCORE:
            logger.info(
                "[KM] 问答质量不足，不写入长期记忆: score=%.2f reasons=%s",
                quality["score"],
                quality["reasons"],
            )
            return None

        duplicate = self.find_duplicate(question)
        if duplicate:
            km_id = int(duplicate.metadata["km_id"])
            self._merge_existing(
                km_id=km_id,
                question=question,
                answer=answer,
                entities=entities,
                tags=tags,
                citations=citations,
                quality_score=quality["score"],
            )
            logger.info(f"[KM] 合并相似问答: id={km_id} score={duplicate.score:.3f}")
            return km_id

        with self.pg.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO knowledge_memory
                   (question, answer, entities, tags, quality_score, source_citations)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    question,
                    answer,
                    json.dumps(entities, ensure_ascii=False),
                    json.dumps(tags, ensure_ascii=False),
                    quality["score"],
                    json.dumps(citations, ensure_ascii=False),
                ),
            )
            km_id = cur.fetchone()[0]
        self.pg.conn.commit()

        if self._ensure_milvus():
            try:
                vector = self._encode_query(question)
                self._milvus_collection.insert([[km_id], [question], [vector]])
                self._milvus_collection.flush()
            except Exception as e:
                logger.warning(f"[KM] Milvus 写入失败: {e}")

        logger.info(f"[KM] 存储问答: id={km_id} score={quality['score']:.2f} q={question[:40]}...")
        return km_id

    def evaluate_quality(
        self,
        answer: str,
        citations: list[dict] | None = None,
        tags: list[str] | None = None,
        evidence_count: int = 0,
    ) -> dict:
        """综合多信号判断问答是否值得长期保存。"""
        citations = citations or []
        tags = tags or []
        score = 0.0
        reasons = []

        if citations:
            score += min(0.35, 0.2 + 0.05 * len(citations))
            reasons.append("has_citations")
        else:
            reasons.append("no_citations")

        if len(answer.strip()) >= KNOWLEDGE_MIN_ANSWER_CHARS:
            score += 0.25
            reasons.append("sufficient_length")
        else:
            score -= 0.2
            reasons.append("too_short")

        if evidence_count > 0:
            score += min(0.15, evidence_count * 0.03)
            reasons.append("has_evidence")

        if tags and tags[0] not in {"", "general"}:
            score += 0.1
            reasons.append("specific_intent")

        if self._looks_low_quality(answer):
            score -= 0.4
            reasons.append("low_quality_phrase")

        if any(tag in KNOWLEDGE_UNSAFE_REUSE_INTENTS for tag in tags):
            score -= 0.05
            reasons.append("sensitive_intent")

        return {"score": max(0.0, min(1.0, score)), "reasons": reasons}

    def find_duplicate(self, question: str) -> MemoryItem | None:
        """写入前查重，高相似问题合并到旧记录。"""
        candidates = self.search(
            question,
            limit=1,
            threshold=KNOWLEDGE_DEDUPE_THRESHOLD,
            update_hit=False,
        )
        return candidates[0] if candidates else None

    # ---- 检索 ----

    def search(
        self,
        query: str,
        limit: int = 5,
        threshold: float | None = None,
        update_hit: bool = True,
    ) -> list[MemoryItem]:
        """检索与 query 相似的历史问答。"""
        threshold = self.similarity_threshold if threshold is None else threshold
        if not self._ensure_milvus():
            return self._search_fallback(query, limit)

        try:
            vector = self._encode_query(query)
            self._milvus_collection.load()

            results = self._milvus_collection.search(
                data=[vector],
                anns_field="dense_vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=limit,
                output_fields=["km_id", "question"],
            )

            items: list[MemoryItem] = []
            for hit in results[0]:
                score = float(hit.score) #milvus返回的相似度score
                if score < threshold:
                    continue

                km_id = int(hit.id)
                row = self._get_by_id(km_id)
                if not row:
                    continue

                if update_hit:
                    self._increment_hit(km_id)
                tags = row.get("tags", [])
                items.append(
                    MemoryItem(
                        id=f"km_{km_id}",
                        content=row["answer"],
                        metadata={
                            "question": row["question"],
                            "source": "knowledge",
                            "km_id": km_id,
                            "tags": tags,
                            "quality_score": row.get("quality_score", 0.0),
                            "source_citations": row.get("source_citations", []),
                            "reuse_policy": self.get_reuse_policy(score, tags), #相似度+是否安全意图
                        },
                        score=score,
                    )
                )

            return items
        except Exception as e:
            logger.warning(f"[KM] Milvus 检索失败，降级: {e}")
            return self._search_fallback(query, limit)

    def find_reusable_answer(self, query: str, intent: str | None = None) -> MemoryItem | None:
        """寻找可直接复用的历史答案；仅允许低风险意图。"""
        candidates = self.search(query, limit=1, threshold=KNOWLEDGE_REUSE_THRESHOLD)
        if not candidates:
            return None

        item = candidates[0]
        tags = item.metadata.get("tags", [])
        effective_intent = intent or (tags[0] if tags else "")
        if not self.is_safe_reuse_intent(effective_intent):
            item.metadata["reuse_policy"] = "context_only"
            return None

        item.metadata["reuse_policy"] = "direct_reuse"
        return item

    @staticmethod
    def get_reuse_policy(score: float, tags: list[str]) -> str:
        intent = tags[0] if tags else ""
        if score >= KNOWLEDGE_REUSE_THRESHOLD and KnowledgeMemory.is_safe_reuse_intent(intent):
            return "direct_reuse"
        if score >= KNOWLEDGE_CONTEXT_THRESHOLD:
            return "context_only"
        return "ignore"

    @staticmethod
    def is_safe_reuse_intent(intent: str | None) -> bool:
        if not intent or intent in KNOWLEDGE_UNSAFE_REUSE_INTENTS:
            return False
        return intent in KNOWLEDGE_SAFE_REUSE_INTENTS

    def _search_fallback(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """Milvus 不可用时的降级：PG 关键词匹配，仅作为上下文。"""
        try:
            with self.pg.conn.cursor() as cur:
                cur.execute(
                    """SELECT * FROM knowledge_memory
                       WHERE question LIKE %s
                       ORDER BY hit_count DESC LIMIT %s""",
                    (f"%{query[:50]}%", limit),
                )
                rows = cur.fetchall()
                return [
                    MemoryItem(
                        id=f"km_{row[0]}",
                        content=row[2],
                        metadata={
                            "question": row[1],
                            "source": "knowledge",
                            "km_id": row[0],
                            "reuse_policy": "context_only",
                        },
                        score=0.5,
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.warning(f"[KM] 降级检索也失败: {e}")
            return []

    def search_with_question(self, query: str, limit: int = 5) -> list[KnowledgeEntry]:
        """检索并返回完整 KnowledgeEntry 对象"""
        items = self.search(query, limit)
        entries = []
        for item in items:
            row = self._get_by_id(item.metadata.get("km_id", 0))
            if row:
                entries.append(self._dict_to_entry(row))
        return entries

    # ---- 管理 ----

    def rate(self, km_id: int, rating: float):
        with self.pg.conn.cursor() as cur:
            cur.execute("UPDATE knowledge_memory SET rating = %s WHERE id = %s", (rating, km_id))
        self.pg.conn.commit()

    def get_popular(self, limit: int = 10) -> list[KnowledgeEntry]:
        with self.pg.conn.cursor() as cur:
            cur.execute("SELECT * FROM knowledge_memory ORDER BY hit_count DESC LIMIT %s", (limit,))
            return [self._row_to_entry(r) for r in cur.fetchall()]

    def remove(self, item_id: str) -> bool:
        try:
            km_id = int(item_id.replace("km_", ""))
            with self.pg.conn.cursor() as cur:
                cur.execute("DELETE FROM knowledge_memory WHERE id = %s", (km_id,))
            self.pg.conn.commit()
            if self._milvus_ready:
                try:
                    self._milvus_collection.delete(f"km_id == {km_id}")
                except Exception:
                    pass
            return True
        except (ValueError, Exception) as e:
            logger.warning(f"[KM] 删除失败: {e}")
            return False

    # ---- 基类接口 ----

    def add(self, item: MemoryItem) -> str:
        meta = item.metadata or {}
        km_id = self.store_qa(
            question=meta.get("question", ""),
            answer=item.content,
            entities=meta.get("entities"),
            tags=meta.get("tags"),
            citations=meta.get("citations"),
            evidence_count=meta.get("evidence_count", 0),
        )
        return f"km_{km_id}" if km_id is not None else ""

    # ---- 内部 ----

    def _merge_existing(
        self,
        km_id: int,
        question: str,
        answer: str,
        entities: list[str],
        tags: list[str],
        citations: list[dict],
        quality_score: float,
    ):
        row = self._get_by_id(km_id)
        if not row:
            return

        merged_entities = self._merge_list(row.get("entities", []), entities)
        merged_tags = self._merge_list(row.get("tags", []), tags)
        merged_citations = self._merge_citations(row.get("source_citations", []), citations)

        old_quality = float(row.get("quality_score", 0.0) or 0.0)
        chosen_answer = answer if quality_score >= old_quality else row["answer"]
        chosen_question = question if quality_score >= old_quality else row["question"]

        with self.pg.conn.cursor() as cur:
            cur.execute(
                """UPDATE knowledge_memory
                   SET question = %s,
                       answer = %s,
                       entities = %s,
                       tags = %s,
                       quality_score = GREATEST(quality_score, %s),
                       source_citations = %s,
                       hit_count = hit_count + 1,
                       updated_at = CURRENT_TIMESTAMP,
                       last_accessed = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (
                    chosen_question,
                    chosen_answer,
                    json.dumps(merged_entities, ensure_ascii=False),
                    json.dumps(merged_tags, ensure_ascii=False),
                    quality_score,
                    json.dumps(merged_citations, ensure_ascii=False),
                    km_id,
                ),
            )
        self.pg.conn.commit()

    def _get_by_id(self, km_id: int) -> dict | None:
        with self.pg.conn.cursor() as cur:
            cur.execute("SELECT * FROM knowledge_memory WHERE id = %s", (km_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def _increment_hit(self, km_id: int):
        with self.pg.conn.cursor() as cur:
            cur.execute(
                "UPDATE knowledge_memory SET hit_count = hit_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = %s",
                (km_id,),
            )
        self.pg.conn.commit()

    def _encode_query(self, query: str) -> list[float]:
        try:
            return self.encoder.encode(query).tolist()
        except Exception as e:
            logger.warning(f"[KM] 向量模型不可用，使用fallback向量: {e}")
            return self._fallback_vector(query)

    @staticmethod
    def _fallback_vector(text: str, dimension: int = EMBEDDING_DIM) -> list[float]:
        vector = [0.0] * dimension
        tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", text) or [text]
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    @staticmethod
    def _row_to_dict(row: tuple) -> dict:
        col_names = [
            "id",
            "question",
            "answer",
            "entities",
            "tags",
            "hit_count",
            "rating",
            "created_at",
            "last_accessed",
            "quality_score",
            "source_citations",
            "updated_at",
        ]
        d = dict(zip(col_names, row))
        for field in ("entities", "tags", "source_citations"):
            if isinstance(d.get(field), str):
                d[field] = json.loads(d[field])
        return d

    @staticmethod
    def _row_to_entry(row: tuple) -> KnowledgeEntry:
        return KnowledgeMemory._dict_to_entry(KnowledgeMemory._row_to_dict(row))

    @staticmethod
    def _dict_to_entry(d: dict) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=str(d["id"]),
            question=d["question"],
            answer=d["answer"],
            entities=d.get("entities", []),
            tags=d.get("tags", []),
            hit_count=d.get("hit_count", 0),
            rating=d.get("rating", 0.0),
            created_at=d.get("created_at", datetime.now()),
            last_accessed=d.get("last_accessed", datetime.now()),
        )

    @staticmethod
    def _looks_low_quality(answer: str) -> bool:
        bad_patterns = ["抱歉", "生成遇到问题", "图片分析失败", "信息不足", "无法回答", "不知道"]
        return any(pattern in answer for pattern in bad_patterns)

    @staticmethod
    def _merge_list(old: list, new: list) -> list:
        merged = []
        for item in [*(old or []), *(new or [])]:
            if item and item not in merged:
                merged.append(item)
        return merged

    @staticmethod
    def _merge_citations(old: list[dict], new: list[dict]) -> list[dict]:
        merged = []
        seen = set()
        for item in [*(old or []), *(new or [])]:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                merged.append(item)
        return merged[:20]
