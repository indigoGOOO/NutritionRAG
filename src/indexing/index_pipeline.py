"""索引管道编排（改进版）

串联所有步骤：路由→解析→清洗→分类→分块→KV提取→图谱构建→向量化。
这是索引管线的主入口，协调所有组件完成完整的文档处理流程。

改进：
- 集成ChunkRouter，实现智能分块路由
- 支持多种chunk用途（检索/存储/展示）
- 添加chunk质量评估和后处理
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.indexing.chunk_router import ChunkPurpose, ChunkRouter
from src.indexing.content_classifier import ContentClassifier
from src.indexing.data_cleaner import DataCleaner
from src.indexing.document_router import DocumentRouter
from src.indexing.embedding_service_v2 import EmbeddingService
from src.indexing.graph_builder import GraphBuilder
from src.indexing.kv_extractor_v2 import KVExtractor
from src.indexing.llm_client import BaseLLMClient, get_default_client
from src.indexing.models import (
    ContentChunk,
    DocCategory,
    PipelineResult,
    UnifiedDocument,
)

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """索引管道主编排器（改进版）"""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self._llm_client = llm_client
        self.router = DocumentRouter(llm_client=self.llm_client)
        self.cleaner = DataCleaner()
        self.classifier = ContentClassifier(llm_client=self.llm_client)
        self.chunk_router = ChunkRouter()  # ← 新增：智能分块路由器
        self.embedding_service = EmbeddingService()

        self._kv_extractor: KVExtractor | None = None
        self._graph_builder: GraphBuilder | None = None

    @property
    def llm_client(self) -> BaseLLMClient:
        if self._llm_client is None:
            self._llm_client = get_default_client()
        return self._llm_client

    @property
    def kv_extractor(self) -> KVExtractor:
        if self._kv_extractor is None:
            self._kv_extractor = KVExtractor(self.llm_client)
        return self._kv_extractor

    @property
    def graph_builder(self) -> GraphBuilder:
        if self._graph_builder is None:
            self._graph_builder = GraphBuilder(self.llm_client)
        return self._graph_builder

    def run(self, input_path: Path, chunk_purpose: ChunkPurpose = ChunkPurpose.RETRIEVAL) -> PipelineResult:
        """执行完整索引管线"""
        start_time = time.time()
        result = PipelineResult(doc_id="")

        try:
            # Step 1: 路由与解析
            logger.info(f"[1/7] 路由解析: {input_path.name}")
            document = self.router.route(input_path)
            result.doc_id = document.doc_id

            # Step 2: 数据清洗
            logger.info(f"[2/7] 数据清洗")
            document = self.cleaner.clean(document)

            # Step 3: 内容分类
            logger.info(f"[3/7] 内容分类")
            document = self.classifier.classify_and_set(document)

            # Step 4: 智能分块路由（改进）
            logger.info(f"[4/7] 智能分块路由 (类型: {document.doc_category.value}, 用途: {chunk_purpose.value})")
            chunks = self.chunk_router.route(document, purpose=chunk_purpose)

            # Step 5: KV提取
            logger.info(f"[5/7] KV提取 ({len(chunks)} chunks)")
            kv_pairs = self.kv_extractor.extract(chunks)
            result.kv_pairs = kv_pairs

            # Step 6: 图谱构建
            logger.info(f"[6/7] 图谱构建")
            triples = self.graph_builder.build(chunks, kv_pairs)
            result.triples = triples

            # Step 7: 向量化
            logger.info(f"[7/7] 向量化")
            embeddings = self.embedding_service.embed_chunks(chunks)
            result.chunks = chunks

        except Exception as e:
            logger.error(f"管道执行失败: {e}", exc_info=True)
            result.errors.append(str(e))

        elapsed = time.time() - start_time
        result.stats = {
            "elapsed_seconds": round(elapsed, 2),
            "chunk_count": len(result.chunks),
            "kv_count": len(result.kv_pairs),
            "triple_count": len(result.triples),
            "error_count": len(result.errors),
        }

        logger.info(
            f"管道完成: {result.stats['chunk_count']} chunks, "
            f"{result.stats['kv_count']} KV, "
            f"{result.stats['triple_count']} triples, "
            f"耗时 {elapsed:.1f}s"
        )

        return result

    def run_text(
        self,
        text: str,
        source_name: str = "user_input",
        chunk_purpose: ChunkPurpose = ChunkPurpose.RETRIEVAL,
    ) -> PipelineResult:
        """处理纯文本输入（对话、表单等）"""
        start_time = time.time()
        result = PipelineResult(doc_id="")

        try:
            document = self.router.route_text(text, source_name=source_name)
            result.doc_id = document.doc_id

            document = self.cleaner.clean(document)
            document = self.classifier.classify_and_set(document)
            chunks = self.chunk_router.route(document, purpose=chunk_purpose)

            kv_pairs = self.kv_extractor.extract(chunks)
            result.kv_pairs = kv_pairs

            triples = self.graph_builder.build(chunks, kv_pairs)
            result.triples = triples

            self.embedding_service.embed_chunks(chunks)
            result.chunks = chunks

        except Exception as e:
            logger.error(f"文本管道执行失败: {e}", exc_info=True)
            result.errors.append(str(e))

        elapsed = time.time() - start_time
        result.stats = {
            "elapsed_seconds": round(elapsed, 2),
            "chunk_count": len(result.chunks),
            "kv_count": len(result.kv_pairs),
            "triple_count": len(result.triples),
        }

        return result

    def run_batch(
        self,
        directory: Path,
        recursive: bool = True,
        chunk_purpose: ChunkPurpose = ChunkPurpose.RETRIEVAL,
    ) -> list[PipelineResult]:
        """批量处理目录下的所有文档"""
        results = []
        documents = self.router.route_batch(directory, recursive=recursive)

        for document in documents:
            try:
                document = self.cleaner.clean(document)
                document = self.classifier.classify_and_set(document)
                chunks = self.chunk_router.route(document, purpose=chunk_purpose)

                kv_pairs = self.kv_extractor.extract(chunks)
                triples = self.graph_builder.build(chunks, kv_pairs)
                self.embedding_service.embed_chunks(chunks)

                result = PipelineResult(
                    doc_id=document.doc_id,
                    chunks=chunks,
                    kv_pairs=kv_pairs,
                    triples=triples,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"文档处理失败 {document.doc_id[:8]}: {e}")
                results.append(PipelineResult(
                    doc_id=document.doc_id,
                    errors=[str(e)],
                ))

        logger.info(f"批量处理完成: {len(results)} 个文档")
        return results
