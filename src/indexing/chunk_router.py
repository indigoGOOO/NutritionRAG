"""Chunk Router - 简化的分块路由器

改进设计（按CHUNKER_REDESIGN_FINAL.md）：
1. 文档级路由：根据doc_category选择专业化chunker
2. 用途适配：根据chunk用途（检索/存储/展示）调整策略

注意：
- 各个Chunker内部已集成SemanticChunker，自己处理二次分块
- Router只负责路由和用途优化，不负责块级处理
- 块级处理逻辑已移到各个Chunker内部
"""

from __future__ import annotations

import logging
from enum import Enum

from src.indexing.models import (
    ContentChunk,
    DocCategory,
    UnifiedDocument,
)

logger = logging.getLogger(__name__)


class ChunkPurpose(str, Enum):
    """Chunk的用途"""
    RETRIEVAL = "retrieval"  # 用于向量检索
    STORAGE = "storage"      # 用于数据库存储
    DISPLAY = "display"      # 用于前端展示
    ANALYSIS = "analysis"    # 用于数据分析


class ChunkRouter:
    """简化的分块路由器 - 只负责路由和用途优化"""

    def route(
        self,
        document: UnifiedDocument,
        purpose: ChunkPurpose = ChunkPurpose.RETRIEVAL,
    ) -> list[ContentChunk]:
        """
        智能路由：根据文档类型和用途选择最优分块策略

        Args:
            document: 统一文档格式
            purpose: chunk的用途

        Returns:
            处理后的chunks列表
        """
        logger.info(
            f"Chunk路由: doc_category={document.doc_category.value}, "
            f"purpose={purpose.value}"
        )

        # 第一步：文档级路由 - 选择专业化chunker
        chunker = self._get_chunker(document.doc_category)

        # 第二步：调用Chunker（Chunker内部已处理二次分块）
        chunks = chunker.chunk(document)

        # 第三步：根据用途优化
        final_chunks = self._optimize_for_purpose(chunks, purpose)

        logger.info(
            f"Chunk路由完成: {len(chunks)} → {len(final_chunks)} chunks"
        )
        return final_chunks

    def _get_chunker(self, category: DocCategory):
        """
        获取对应的Chunker

        各个Chunker内部已集成SemanticChunker，自己处理二次分块
        """
        from src.indexing.chunkers.daily_record_chunker import DailyRecordChunker
        from src.indexing.chunkers.medical_chunker import MedicalChunker
        from src.indexing.chunkers.nutrition_chunker import NutritionChunker
        from src.indexing.chunkers.personal_chunker import PersonalChunker
        from src.indexing.chunkers.recipe_chunker import RecipeChunker
        from src.indexing.chunkers.semantic_chunker_v2 import SemanticChunker

        chunker_map = {
            DocCategory.PERSONAL: PersonalChunker,
            DocCategory.DAILY: DailyRecordChunker,
            DocCategory.NUTRITION: NutritionChunker,
            DocCategory.RECIPE: RecipeChunker,
            DocCategory.MEDICAL: MedicalChunker,
            DocCategory.UNKNOWN: SemanticChunker,
        }

        chunker_class = chunker_map.get(category, SemanticChunker)
        return chunker_class()

    def _optimize_for_purpose(
        self,
        chunks: list[ContentChunk],
        purpose: ChunkPurpose,
    ) -> list[ContentChunk]:
        """
        根据用途优化chunks

        Args:
            chunks: 原始chunks列表
            purpose: chunk的用途

        Returns:
            优化后的chunks列表
        """
        if purpose == ChunkPurpose.RETRIEVAL:
            return self._optimize_for_retrieval(chunks)
        elif purpose == ChunkPurpose.STORAGE:
            return self._optimize_for_storage(chunks)
        elif purpose == ChunkPurpose.DISPLAY:
            return self._optimize_for_display(chunks)
        else:
            return chunks

    def _optimize_for_retrieval(self, chunks: list[ContentChunk]) -> list[ContentChunk]:
        """优化用于向量检索"""
        # 检索优化：
        # 1. 过滤掉太短的chunk（无足够上下文）
        # 2. 添加检索相关的元数据
        result = []
        for chunk in chunks:
            if chunk.token_count >= 50:  # 最小长度阈值
                chunk.metadata["optimized_for"] = "retrieval"
                result.append(chunk)

        if not result and chunks:
            for chunk in chunks:
                chunk.metadata["optimized_for"] = "retrieval"
                chunk.metadata["kept_short_chunk"] = True
            return chunks

        logger.debug(f"检索优化: {len(chunks)} → {len(result)} chunks")
        return result

    def _optimize_for_storage(self, chunks: list[ContentChunk]) -> list[ContentChunk]:
        """优化用于数据库存储"""
        # 存储优化：
        # 1. 添加完整的元数据
        # 2. 确保chunk的独立性
        # 3. 添加版本信息
        for chunk in chunks:
            chunk.metadata["optimized_for"] = "storage"
            chunk.metadata["version"] = "1.0"

        logger.debug(f"存储优化: {len(chunks)} chunks")
        return chunks

    def _optimize_for_display(self, chunks: list[ContentChunk]) -> list[ContentChunk]:
        """优化用于前端展示"""
        # 展示优化：
        # 1. 添加格式化信息
        # 2. 添加可读性相关的元数据
        # 3. 确保chunk的完整性
        for chunk in chunks:
            chunk.metadata["optimized_for"] = "display"
            chunk.metadata["display_format"] = self._infer_display_format(chunk)

        logger.debug(f"展示优化: {len(chunks)} chunks")
        return chunks

    @staticmethod
    def _infer_display_format(chunk: ContentChunk) -> str:
        """推断chunk的展示格式"""
        if chunk.chunk_type == "table":
            return "table"
        elif chunk.chunk_type == "list":
            return "list"
        elif chunk.chunk_type == "recipe_step":
            return "step"
        elif chunk.chunk_type == "recipe_ingredients":
            return "ingredients"
        else:
            return "text"

        if quality == ChunkQuality.EXCELLENT:
            # 质量很好，无需处理
            return [chunk]
        elif quality == ChunkQuality.GOOD:
            # 质量不错，可能需要微调
            return [chunk]
        elif quality == ChunkQuality.FAIR:
            # 质量一般，需要二次处理
            return self._refine_chunk(chunk)
        else:  # POOR
            # 质量差，需要重新分块
            return self._resplit_chunk(chunk)

    def _assess_chunk_quality(self, chunk: ContentChunk) -> ChunkQuality:
        """
        评估chunk质量

        评估维度：
        1. 长度合理性：是否在合理范围内
        2. 语义完整性：是否包含完整的语义单位
        3. 结构清晰性：是否有清晰的结构
        """
        token_count = chunk.token_count

        # 维度1：长度合理性
        if token_count < 50:
            return ChunkQuality.POOR  # 太短
        elif token_count > 1000:
            return ChunkQuality.FAIR  # 太长
        elif 100 <= token_count <= 512:
            return ChunkQuality.EXCELLENT  # 理想范围
        else:
            return ChunkQuality.GOOD  # 可接受范围

    def _refine_chunk(self, chunk: ContentChunk) -> list[ContentChunk]:
        """微调chunk（质量一般）"""
        # 可能的微调：
        # 1. 添加上下文
        # 2. 调整边界
        # 3. 添加元数据
        chunk.metadata["refined"] = True
        return [chunk]

    def _resplit_chunk(self, chunk: ContentChunk) -> list[ContentChunk]:
        """重新分块（质量差）"""
        from src.indexing.chunkers.semantic_chunker_v2 import SemanticChunker

        semantic_chunker = SemanticChunker()
        sub_texts = semantic_chunker.split_text(chunk.content)

        result = []
        for sub_text in sub_texts:
            result.append(
                ContentChunk(
                    content=sub_text,
                    chunk_type=chunk.chunk_type,
                    doc_category=chunk.doc_category,
                    source_doc_id=chunk.source_doc_id,
                    source_block_ids=chunk.source_block_ids,
                    token_count=semantic_chunker._count_tokens(sub_text),
                    metadata={**chunk.metadata, "resplit": True},
                )
            )
        return result

    def _split_nutrition_chunk(self, chunk: ContentChunk) -> list[ContentChunk]:
        """按营养素类别切分营养文本"""
        # 识别营养素类别边界
        import re

        nutrient_keywords = [
            "热量", "蛋白质", "脂肪", "碳水化合物",
            "维生素", "矿物质", "纤维", "钠", "钙", "铁"
        ]

        # 按营养素关键词切分
        pattern = "|".join(nutrient_keywords)
        parts = re.split(f"({pattern})", chunk.content)

        result = []
        current_text = ""

        for part in parts:
            if part in nutrient_keywords:
                if current_text:
                    result.append(current_text)
                current_text = part
            else:
                current_text += part

        if current_text:
            result.append(current_text)

        # 转为ContentChunk
        chunks = []
        for text in result:
            if text.strip():
                chunks.append(
                    ContentChunk(
                        content=text,
                        chunk_type=chunk.chunk_type,
                        doc_category=chunk.doc_category,
                        source_doc_id=chunk.source_doc_id,
                        source_block_ids=chunk.source_block_ids,
                        token_count=len(text) // 4,  # 粗略估算
                        metadata={**chunk.metadata, "split_by": "nutrient"},
                    )
                )

        return chunks if chunks else [chunk]

    def _post_process_chunks(
        self, chunks: list[ContentChunk], purpose: ChunkPurpose
    ) -> list[ContentChunk]:
        """
        第三步：质量评估和后处理

        根据用途调整chunk：
        - RETRIEVAL：优化用于向量检索
        - STORAGE：优化用于数据库存储
        - DISPLAY：优化用于前端展示
        """
        if purpose == ChunkPurpose.RETRIEVAL:
            return self._optimize_for_retrieval(chunks)
        elif purpose == ChunkPurpose.STORAGE:
            return self._optimize_for_storage(chunks)
        elif purpose == ChunkPurpose.DISPLAY:
            return self._optimize_for_display(chunks)
        else:
            return chunks

    def _optimize_for_retrieval(self, chunks: list[ContentChunk]) -> list[ContentChunk]:
        """优化用于向量检索"""
        # 检索优化：
        # 1. 确保chunk有足够的上下文
        # 2. 添加检索相关的元数据
        # 3. 过滤掉太短的chunk
        result = []
        for chunk in chunks:
            if chunk.token_count >= 50:  # 最小长度
                chunk.metadata["optimized_for"] = "retrieval"
                result.append(chunk)
        if not result and chunks:
            for chunk in chunks:
                chunk.metadata["optimized_for"] = "retrieval"
                chunk.metadata["kept_short_chunk"] = True
            return chunks
        return result

    def _optimize_for_storage(self, chunks: list[ContentChunk]) -> list[ContentChunk]:
        """优化用于数据库存储"""
        # 存储优化：
        # 1. 添加完整的元数据
        # 2. 确保chunk的独立性
        # 3. 添加版本信息
        for chunk in chunks:
            chunk.metadata["optimized_for"] = "storage"
            chunk.metadata["version"] = "1.0"
        return chunks

    def _optimize_for_display(self, chunks: list[ContentChunk]) -> list[ContentChunk]:
        """优化用于前端展示"""
        # 展示优化：
        # 1. 添加格式化信息
        # 2. 添加可读性相关的元数据
        # 3. 确保chunk的完整性
        for chunk in chunks:
            chunk.metadata["optimized_for"] = "display"
            chunk.metadata["display_format"] = self._infer_display_format(chunk)
        return chunks

    @staticmethod
    def _infer_display_format(chunk: ContentChunk) -> str:
        """推断chunk的展示格式"""
        if chunk.chunk_type == "table":
            return "table"
        elif chunk.chunk_type == "list":
            return "list"
        elif chunk.chunk_type == "recipe_step":
            return "step"
        else:
            return "text"
