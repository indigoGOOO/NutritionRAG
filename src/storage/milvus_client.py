"""Milvus 连接器

向量数据库 - 支持dense和sparse向量的混合检索
"""

from __future__ import annotations

import logging
import hashlib
from typing import Any, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from config.settings import MILVUS_CONFIG

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus 客户端 - 向量检索"""

    def __init__(self, config: dict | None = None):
        """
        初始化Milvus连接

        Args:
            config: Milvus配置（如果为None则使用settings中的配置）
        """
        self.config = config or MILVUS_CONFIG
        self.collection_name = self.config.get("collection", "nutrition_chunks")
        self.collection = None
        self.connect()

    def connect(self):
        """建立Milvus连接"""
        try:
            connections.connect(
                alias="default",
                host=self.config["host"],
                port=self.config["port"],
            )
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
            logger.info(f"Milvus连接成功: {self.config['host']}:{self.config['port']}")
        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            raise

    def close(self):
        """关闭Milvus连接"""
        try:
            connections.disconnect(alias="default")
            logger.info("Milvus连接已关闭")
        except Exception as e:
            logger.error(f"关闭Milvus连接失败: {e}")

    def init_collection(self, embedding_dim: int = 384):
        """
        初始化集合

        Args:
            embedding_dim: embedding维度
        """
        try:
            # 定义字段
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="doc_category", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="source_doc_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="token_count", dtype=DataType.INT32),
                # Dense向量（semantic embedding）
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
                # Sparse向量（BM25）- 使用JSON存储
                FieldSchema(name="sparse_vector", dtype=DataType.VARCHAR, max_length=65535),
            ]

            # 创建schema
            schema = CollectionSchema(
                fields=fields,
                description="营养RAG chunks集合 - 支持dense和sparse混合检索",
            )

            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"Milvus集合已存在，直接复用: {self.collection_name}")
            else:
                # 创建集合
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using="default",
                )

                # 创建索引
                self.collection.create_index(
                    field_name="dense_vector",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": "L2",
                        "params": {"nlist": 128},
                    },
                )

            self.collection.load()

            logger.info(f"Milvus集合初始化完成: {self.collection_name}")
        except Exception as e:
            logger.error(f"初始化Milvus集合失败: {e}")
            raise

    def insert_chunk(
        self,
        chunk_id: int,
        content: str,
        chunk_type: str,
        doc_category: str,
        source_doc_id: str,
        token_count: int,
        dense_vector: list[float],
        sparse_vector: str | None = None,
    ):
        """
        插入单个chunk的向量

        Args:
            chunk_id: chunk ID
            content: chunk内容
            chunk_type: chunk类型
            doc_category: 文档类别
            source_doc_id: 源文档ID
            token_count: token数
            dense_vector: dense向量
            sparse_vector: sparse向量（JSON字符串）
        """
        try:
            self._ensure_collection()
            data = [
                [chunk_id],
                [content],
                [chunk_type],
                [doc_category],
                [source_doc_id],
                [token_count],
                [dense_vector],
                [sparse_vector or "{}"],
            ]

            self.collection.insert(data)
            logger.debug(f"插入chunk向量: chunk_id={chunk_id}")
        except Exception as e:
            logger.error(f"插入chunk向量失败: {e}")
            raise

    def insert_chunks_batch(self, chunks_data: list[dict]):
        """
        批量插入chunks的向量

        Args:
            chunks_data: chunk数据列表，每个元素包含：
                - chunk_id: chunk ID
                - content: chunk内容
                - chunk_type: chunk类型
                - doc_category: 文档类别
                - source_doc_id: 源文档ID
                - token_count: token数
                - dense_vector: dense向量
                - sparse_vector: sparse向量（可选）
        """
        try:
            self._ensure_collection()
            chunk_ids = []
            contents = []
            chunk_types = []
            doc_categories = []
            source_doc_ids = []
            token_counts = []
            dense_vectors = []
            sparse_vectors = []

            for chunk in chunks_data:
                chunk_ids.append(chunk["chunk_id"])
                contents.append(chunk["content"])
                chunk_types.append(chunk["chunk_type"])
                doc_categories.append(chunk["doc_category"])
                source_doc_ids.append(chunk["source_doc_id"])
                token_counts.append(chunk["token_count"])
                dense_vectors.append(chunk["dense_vector"])
                sparse_vectors.append(chunk.get("sparse_vector", "{}"))

            data = [
                chunk_ids,
                contents,
                chunk_types,
                doc_categories,
                source_doc_ids,
                token_counts,
                dense_vectors,
                sparse_vectors,
            ]

            self.collection.insert(data)
            logger.info(f"批量插入{len(chunks_data)}个chunk向量")
        except Exception as e:
            logger.error(f"批量插入chunk向量失败: {e}")
            raise

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """
        Dense向量检索

        Args:
            query_vector: 查询向量
            top_k: 返回结果数
            filter_expr: 过滤表达式

        Returns:
            检索结果列表
        """
        try:
            self._ensure_collection()
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }

            results = self.collection.search(
                data=[query_vector],
                anns_field="dense_vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=[
                    "chunk_id",
                    "content",
                    "chunk_type",
                    "doc_category",
                    "source_doc_id",
                    "token_count",
                ],
            )

            return self._format_search_results(results)
        except Exception as e:
            logger.error(f"Dense向量检索失败: {e}")
            raise

    def search_sparse(
        self,
        query_terms: dict[str, float],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """
        Sparse向量检索（BM25）

        Args:
            query_terms: 查询词及其权重 {term: weight}
            top_k: 返回结果数
            filter_expr: 过滤表达式

        Returns:
            检索结果列表
        """
        try:
            self._ensure_collection()
            import json

            # 构建sparse向量查询
            # 这里需要实现BM25相似度计算
            # 暂时使用简单的term匹配
            results = []

            # 获取所有chunks
            all_results = self.collection.query(
                expr=filter_expr or "chunk_id > 0",
                output_fields=[
                    "chunk_id",
                    "content",
                    "chunk_type",
                    "doc_category",
                    "source_doc_id",
                    "token_count",
                    "sparse_vector",
                ],
                limit=10000,
            )

            # 计算BM25分数
            hashed_query_terms = {
                self._stable_token_id(term): weight
                for term, weight in query_terms.items()
            }

            for result in all_results:
                score = 0.0
                sparse_vec = json.loads(result.get("sparse_vector", "{}"))

                for token_id, weight in hashed_query_terms.items():
                    if token_id in sparse_vec:
                        score += sparse_vec[token_id] * weight

                if score > 0:
                    results.append({
                        "chunk_id": result["chunk_id"],
                        "content": result["content"],
                        "chunk_type": result["chunk_type"],
                        "doc_category": result["doc_category"],
                        "source_doc_id": result.get("source_doc_id"),
                        "token_count": result["token_count"],
                        "score": score,
                    })

            # 按分数排序
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Sparse向量检索失败: {e}")
            raise

    def hybrid_search(
        self,
        query_vector: list[float],
        query_terms: dict[str, float] | None = None,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """
        混合检索（Dense + Sparse）

        Args:
            query_vector: 查询向量
            query_terms: 查询词及其权重（可选）
            top_k: 返回结果数
            dense_weight: dense权重
            sparse_weight: sparse权重
            filter_expr: 过滤表达式

        Returns:
            检索结果列表
        """
        try:
            # Dense检索
            dense_results = self.search_dense(query_vector, top_k=top_k*2, filter_expr=filter_expr)

            # Sparse检索
            sparse_results = []
            if query_terms:
                sparse_results = self.search_sparse(query_terms, top_k=top_k*2, filter_expr=filter_expr)

            # 融合结果（RRF或加权）
            merged_results = self._merge_results(
                dense_results,
                sparse_results,
                dense_weight,
                sparse_weight,
                top_k,
            )

            return merged_results
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise

    def _format_search_results(self, results) -> list[dict]:
        """格式化搜索结果"""
        formatted = []
        for result in results:
            for hit in result:
                formatted.append({
                    "chunk_id": hit.id,
                    "content": hit.entity.get("content"),
                    "chunk_type": hit.entity.get("chunk_type"),
                    "doc_category": hit.entity.get("doc_category"),
                    "source_doc_id": hit.entity.get("source_doc_id"),
                    "token_count": hit.entity.get("token_count"),
                    "score": hit.score,
                })
        return formatted

    def _merge_results(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        dense_weight: float,
        sparse_weight: float,
        top_k: int,
    ) -> list[dict]:
        """
        融合Dense和Sparse检索结果

        使用加权融合策略
        """
        merged = {}

        # 添加dense结果
        for i, result in enumerate(dense_results):
            chunk_id = result["chunk_id"]
            # 归一化分数到[0, 1]
            normalized_score = 1.0 / (1.0 + result.get("score", 0))
            if chunk_id not in merged:
                merged[chunk_id] = result.copy()
                merged[chunk_id]["final_score"] = 0.0
            merged[chunk_id]["final_score"] += normalized_score * dense_weight

        # 添加sparse结果
        if sparse_results:
            max_sparse_score = max(r.get("score", 0) for r in sparse_results) or 1.0
            for result in sparse_results:
                chunk_id = result["chunk_id"]
                normalized_score = result.get("score", 0) / max_sparse_score
                if chunk_id not in merged:
                    merged[chunk_id] = result.copy()
                    merged[chunk_id]["final_score"] = 0.0
                merged[chunk_id]["final_score"] += normalized_score * sparse_weight

        # 排序并返回top_k
        sorted_results = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)
        return sorted_results[:top_k]

    def delete_chunk(self, chunk_id: int):
        """删除chunk"""
        try:
            self._ensure_collection()
            self.collection.delete(expr=f"chunk_id == {chunk_id}")
            logger.debug(f"删除chunk向量: chunk_id={chunk_id}")
        except Exception as e:
            logger.error(f"删除chunk向量失败: {e}")
            raise

    def delete_chunks_by_doc_id(self, doc_id: str):
        """按文档ID删除chunks"""
        try:
            self._ensure_collection()
            self.collection.delete(expr=f"source_doc_id == '{doc_id}'")
            logger.info(f"删除文档chunks: doc_id={doc_id}")
        except Exception as e:
            logger.error(f"删除文档chunks失败: {e}")
            raise

    def flush(self):
        """刷新集合"""
        try:
            self._ensure_collection()
            self.collection.flush()
            logger.debug("Milvus集合已刷新")
        except Exception as e:
            logger.error(f"刷新Milvus集合失败: {e}")

    def _ensure_collection(self):
        """确保当前客户端已绑定可用集合。"""
        if self.collection is not None:
            return
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            return
        raise RuntimeError(f"Milvus集合不存在，请先调用 init_collection(): {self.collection_name}")

    @staticmethod
    def _stable_token_id(token: str) -> str:
        """生成与EmbeddingService一致的稀疏向量token id。"""
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return str(int(digest[:8], 16) % 100000)
