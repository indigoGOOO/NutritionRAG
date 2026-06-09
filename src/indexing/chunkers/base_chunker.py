"""分块器工具类和协议

提供通用的分块工具方法和Chunker协议定义。
"""

from __future__ import annotations

from typing import Protocol

from src.indexing.models import ContentChunk, UnifiedDocument


class Chunker(Protocol):
    """Chunker协议 - 任何实现chunk()方法的类都是Chunker"""

    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        """将文档分块为ContentChunk列表"""
        ...


class ChunkerUtils:
    """Chunker工具类 - 提供通用方法"""

    @staticmethod
    def count_tokens(text: str) -> int:
        """计算token数"""
        try:
            import tiktoken
            tokenizer = tiktoken.get_encoding("cl100k_base")
            return len(tokenizer.encode(text))
        except Exception:
            # 降级方案：粗略估算
            return int(len(text) * 0.7)

    @staticmethod
    def should_split(chunk: ContentChunk, threshold: int = 512) -> bool:
        """判断是否需要分块"""
        return chunk.token_count > threshold

    @staticmethod
    def create_sub_chunk(
        parent_chunk: ContentChunk,
        content: str,
        split_by: str = "semantic",
    ) -> ContentChunk:
        """创建子chunk"""
        return ContentChunk(
            content=content,
            chunk_type=parent_chunk.chunk_type,
            doc_category=parent_chunk.doc_category,
            source_doc_id=parent_chunk.source_doc_id,
            source_block_ids=parent_chunk.source_block_ids,
            token_count=ChunkerUtils.count_tokens(content),
            metadata={**parent_chunk.metadata, "split_by": split_by},
        )
