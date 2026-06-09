"""存储管理器 - 协调 PostgreSQL 和 Milvus 操作

统一接口用于：
1. 存储chunks到PostgreSQL和Milvus
2. 存储KV对到PostgreSQL
3. 存储关系三元组到PostgreSQL
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.indexing.models import ContentChunk, PipelineResult
from src.storage.milvus_client import MilvusClient
from src.storage.pg_client import PostgreSQLClient

logger = logging.getLogger(__name__)


class StorageManager:
    """存储管理器 - 协调多数据库操作"""

    def __init__(
        self,
        pg_config: dict | None = None,
        milvus_config: dict | None = None,
    ):
        """
        初始化存储管理器

        Args:
            pg_config: PostgreSQL配置
            milvus_config: Milvus配置
        """
        self.pg_client = PostgreSQLClient(pg_config)
        self.milvus_client = MilvusClient(milvus_config)

    def init_all(self, embedding_dim: int = 384):
        """初始化所有数据库"""
        logger.info("初始化所有存储层...")
        self.pg_client.init_tables()
        self.milvus_client.init_collection(embedding_dim)
        logger.info("所有存储层初始化完成")

    def close_all(self):
        """关闭所有数据库连接"""
        self.pg_client.close()
        self.milvus_client.close()

    def store_chunk(
        self,
        chunk: ContentChunk,
        dense_vector: list[float],
        sparse_vector: dict | None = None,
    ) -> int:
        """存储单个chunk到PostgreSQL和Milvus"""
        try:
            chunk_id = self.pg_client.insert_chunk(chunk)
            self.milvus_client.insert_chunk(
                chunk_id=chunk_id,
                content=chunk.content,
                chunk_type=chunk.chunk_type,
                doc_category=chunk.doc_category.value if chunk.doc_category else None,
                source_doc_id=chunk.source_doc_id,
                token_count=chunk.token_count,
                dense_vector=dense_vector,
                sparse_vector=json.dumps(sparse_vector) if sparse_vector else None,
            )
            logger.debug(f"存储chunk: chunk_id={chunk_id}")
            return chunk_id
        except Exception as e:
            logger.error(f"存储chunk失败: {e}")
            raise

    def store_chunks_batch(
        self,
        chunks: list[ContentChunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict] | None = None,
    ) -> list[int]:
        """批量存储chunks"""
        try:
            chunk_ids = self.pg_client.insert_chunks_batch(chunks)
            milvus_data = []
            for i, chunk_id in enumerate(chunk_ids):
                milvus_data.append({
                    "chunk_id": chunk_id,
                    "content": chunks[i].content,
                    "chunk_type": chunks[i].chunk_type,
                    "doc_category": chunks[i].doc_category.value if chunks[i].doc_category else None,
                    "source_doc_id": chunks[i].source_doc_id,
                    "token_count": chunks[i].token_count,
                    "dense_vector": dense_vectors[i],
                    "sparse_vector": json.dumps(sparse_vectors[i]) if sparse_vectors and i < len(sparse_vectors) else None,
                })
            self.milvus_client.insert_chunks_batch(milvus_data)
            logger.info(f"批量存储{len(chunk_ids)}个chunks")
            return chunk_ids
        except Exception as e:
            logger.error(f"批量存储chunks失败: {e}")
            raise

    def store_kv_pair(
        self,
        entity_id: str,
        entity_type: str,
        attribute: str,
        value: Any,
        source_chunk_id: int | None = None,
        confidence: float | None = None,
    ) -> int:
        """存储KV对到PostgreSQL"""
        try:
            kv_id = self.pg_client.insert_kv_pair(
                entity_id=entity_id,
                entity_type=entity_type,
                attribute=attribute,
                value=value,
                source_chunk_id=source_chunk_id,
                confidence=confidence,
            )
            logger.debug(f"存储KV对: {entity_id}.{attribute}")
            return kv_id
        except Exception as e:
            logger.error(f"存储KV对失败: {e}")
            raise

    def store_kv_pairs_batch(self, kv_pairs: list[dict]) -> list[int]:
        """批量存储KV对"""
        try:
            kv_ids = self.pg_client.insert_kv_pairs_batch(kv_pairs)
            logger.info(f"批量存储{len(kv_ids)}个KV对")
            return kv_ids
        except Exception as e:
            logger.error(f"批量存储KV对失败: {e}")
            raise

    def store_triple(
        self,
        subject: str,
        predicate: str,
        object_: str,
        source_chunk_id: int | None = None,
        confidence: float | None = None,
    ) -> int:
        """存储关系三元组到PostgreSQL"""
        try:
            triple_id = self.pg_client.insert_triple(
                subject=subject,
                predicate=predicate,
                object_=object_,
                source_chunk_id=source_chunk_id,
                confidence=confidence,
            )
            logger.debug(f"存储三元组: {subject} -[{predicate}]-> {object_}")
            return triple_id
        except Exception as e:
            logger.error(f"存储三元组失败: {e}")
            raise

    def store_triples_batch(self, triples: list[dict]) -> list[int]:
        """批量存储关系三元组到PostgreSQL"""
        try:
            triple_ids = self.pg_client.insert_triples_batch(triples)
            logger.info(f"批量存储{len(triple_ids)}个三元组")
            return triple_ids
        except Exception as e:
            logger.error(f"批量存储三元组失败: {e}")
            raise

    def store_pipeline_result(self, result: PipelineResult) -> dict:
        """将索引管线结果写入 PostgreSQL / Milvus。"""
        if result.errors:
            logger.warning("管线结果包含错误，仍尝试写入可用数据: %s", result.errors)

        chunk_id_map: dict[str, int] = {}
        chunk_ids: list[int] = []

        if result.chunks:
            embedding_by_chunk_id = {
                item.get("chunk_id"): item for item in result.embeddings
            }
            dense_vectors = []
            sparse_vectors = []

            for chunk in result.chunks:
                embedding = embedding_by_chunk_id.get(chunk.chunk_id)
                if embedding is None:
                    raise ValueError(f"缺少chunk向量: {chunk.chunk_id}")
                dense_vectors.append(embedding["dense_vector"])
                sparse_vectors.append(embedding.get("sparse_vector") or {})

            chunk_ids = self.store_chunks_batch(
                chunks=result.chunks,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
            )
            chunk_id_map = {
                chunk.chunk_id: chunk_ids[i]
                for i, chunk in enumerate(result.chunks)
            }

        kv_payload = []
        for kv in result.kv_pairs:
            kv_payload.append({
                "entity_id": kv.key,
                "entity_type": kv.entity_type,
                "attribute": "attributes",
                "value": kv.value,
                "source_chunk_id": chunk_id_map.get(kv.source_chunk_id),
                "confidence": kv.value.get("confidence") if isinstance(kv.value, dict) else None,
                "metadata": {"source_doc_id": kv.source_doc_id},
            })

        triple_payload = []
        for triple in result.triples:
            triple_payload.append({
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "source_chunk_id": chunk_id_map.get(triple.source_chunk_id),
                "confidence": triple.properties.get("confidence") if triple.properties else None,
                "metadata": {
                    **(triple.properties or {}),
                    "subject_type": triple.subject_type,
                    "object_type": triple.object_type,
                },
            })

        kv_ids = self.store_kv_pairs_batch(kv_payload) if kv_payload else []
        triple_ids = self.store_triples_batch(triple_payload) if triple_payload else []

        try:
            self.milvus_client.flush()
        except Exception as e:
            logger.warning("Milvus flush失败: %s", e)

        return {
            "chunks": len(chunk_ids),
            "kv_pairs": len(kv_ids),
            "triples": len(triple_ids),
            "chunk_id_map": chunk_id_map,
        }

    def search_chunks(
        self,
        query_vector: list[float],
        query_terms: dict[str, float] | None = None,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[dict]:
        """混合检索chunks"""
        try:
            results = self.milvus_client.hybrid_search(
                query_vector=query_vector,
                query_terms=query_terms,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
            logger.debug(f"混合检索完成: 返回{len(results)}个结果")
            return results
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise

    def get_chunk_by_id(self, chunk_id: int) -> dict | None:
        """获取chunk详情"""
        try:
            with self.pg_client.conn.cursor() as cur:
                cur.execute("SELECT * FROM chunks WHERE chunk_id = %s;", (chunk_id,))
                result = cur.fetchone()
                if result:
                    return {
                        "chunk_id": result[0],
                        "content": result[1],
                        "chunk_type": result[2],
                        "doc_category": result[3],
                        "source_doc_id": result[4],
                        "token_count": result[5],
                        "metadata": json.loads(result[6]) if result[6] else {},
                    }
                return None
        except Exception as e:
            logger.error(f"获取chunk失败: {e}")
            raise

    def delete_document(self, doc_id: str):
        """删除文档的所有数据"""
        try:
            logger.info(f"删除文档: {doc_id}")
            self.pg_client.delete_chunks_by_doc_id(doc_id)
            self.milvus_client.delete_chunks_by_doc_id(doc_id)
            logger.info(f"文档删除完成: {doc_id}")
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            raise

    def get_statistics(self) -> dict:
        """获取存储统计信息"""
        try:
            with self.pg_client.conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM chunks;")
                chunk_count = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM kv_pairs;")
                kv_count = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM triples;")
                triple_count = cur.fetchone()[0]
            return {
                "postgresql": {
                    "chunks": chunk_count,
                    "kv_pairs": kv_count,
                    "triples": triple_count,
                },
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            raise

    def get_entity_info(self, entity_id: str) -> dict | None:
        """获取实体信息（从 PG triples 和 kv_pairs）"""
        try:
            report = self.pg_client.query_relation_report(entity_id)
            return report
        except Exception as e:
            logger.error(f"获取实体信息失败: {e}")
            raise
