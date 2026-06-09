"""改进的语义分块器 - 相似度断点识别

问题2改进：不再用token数硬切，而是用相邻句子的相似度识别断点。
相似度低于阈值的地方作为切分点，保持语义连贯性。
"""

from __future__ import annotations

import logging
import hashlib
import re
from typing import Any

import numpy as np

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    SEMANTIC_SIMILARITY_THRESHOLD,
)
from src.indexing.models import (
    BlockType,
    ContentChunk,
    UnifiedDocument,
)

logger = logging.getLogger(__name__)


class SemanticChunker:
    """改进的语义分块器 - 基于相似度的断点识别"""

    # 中文句子分隔符
    SENTENCE_SPLITTERS = re.compile(r"(?<=[。！？；\n])")

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
        min_chunk_size: int = MIN_CHUNK_SIZE,
        max_chunk_size: int = MAX_CHUNK_SIZE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min(min_chunk_size, chunk_size)
        self.max_chunk_size = chunk_size if chunk_size != CHUNK_SIZE and max_chunk_size == MAX_CHUNK_SIZE else max_chunk_size

        self._embedding_model = None
        self._tokenizer = None

    @property
    def embedding_model(self):
        """延迟加载embedding模型"""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            from config.settings import EMBEDDING_MODEL

            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    @property
    def tokenizer(self):
        """延迟加载tokenizer"""
        if self._tokenizer is None:
            try:
                import tiktoken

                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._tokenizer = None
        return self._tokenizer

    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        """将文档按语义分块"""
        chunks = []

        for block in document.blocks:
            if block.block_type in (BlockType.TEXT, BlockType.LIST):
                text = block.content if isinstance(block.content, str) else ""
                if not text:
                    continue
                block_chunks = self._split_text_by_similarity(text)
                for chunk_text in block_chunks:
                    chunks.append(
                        ContentChunk(
                            content=chunk_text,
                            chunk_type=block.block_type.value,
                            doc_category=document.doc_category,
                            source_doc_id=document.doc_id,
                            source_block_ids=[block.block_id],
                            token_count=self._count_tokens(chunk_text),
                        )
                    )
            elif block.block_type == BlockType.TABLE:
                table_text = document._table_to_text(block.content)
                chunks.append(
                    ContentChunk(
                        content=table_text,
                        chunk_type="table",
                        doc_category=document.doc_category,
                        source_doc_id=document.doc_id,
                        source_block_ids=[block.block_id],
                        token_count=self._count_tokens(table_text),
                    )
                )

        return chunks

    def split_text(self, text: str) -> list[str]:
        """
        将文本按语义分块（供其他chunker调用）

        Args:
            text: 要分块的文本

        Returns:
            分块后的文本列表
        """
        return self._split_text_by_similarity(text)

    def _split_text_by_similarity(self, text: str) -> list[str]:
        """
        问题2改进：基于相似度的断点识别

        算法：
        1. 按句子分割文本
        2. 生成每个句子的embedding
        3. 计算相邻句子的cosine相似度
        4. 相似度 < threshold → 标记为断点
        5. 在断点处切分，同时保持overlap
        """
        sentences = self.SENTENCE_SPLITTERS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text] if text.strip() else []

        if len(sentences) == 1:
            return sentences

        # 生成句子向量
        embeddings = self._encode_sentences(sentences)

        # 计算相邻句子的相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # 识别断点（相似度低于阈值）
        breakpoints = [0]  # 第一个句子总是起点
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(sentences))  # 最后一个句子总是终点

        # 根据断点生成chunks
        chunks = []
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]

            # 合并句子
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = "".join(chunk_sentences)

            # 检查chunk大小
            chunk_tokens = self._count_tokens(chunk_text)

            if chunk_tokens > self.max_chunk_size:
                # 超过最大大小，递归切分
                sub_chunks = self._hard_split(chunk_text)
                chunks.extend(sub_chunks)
            elif chunk_tokens >= self.min_chunk_size:
                chunks.append(chunk_text)
            else:
                # 太小，尝试与下一个chunk合并
                if chunks:
                    chunks[-1] += chunk_text
                else:
                    chunks.append(chunk_text)

        # 添加overlap
        chunks_with_overlap = self._add_overlap(chunks)
        return chunks_with_overlap

    def _hard_split(self, text: str) -> list[str]:
        """对超长文本按token数硬切"""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            chunks = []
            for i in range(0, len(tokens), self.max_chunk_size):
                chunk_tokens = tokens[i : i + self.max_chunk_size]
                chunks.append(self.tokenizer.decode(chunk_tokens))
            return chunks
        else:
            # fallback: 按字符数估算
            char_limit = int(self.max_chunk_size * 0.7)
            return [text[i : i + char_limit] for i in range(0, len(text), char_limit)]

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """为chunks添加overlap"""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            # 从前一个chunk的末尾提取overlap部分
            prev_chunk = chunks[i - 1]
            overlap_text = self._get_overlap_text(prev_chunk)
            result.append(overlap_text + chunks[i])

        return result

    def _get_overlap_text(self, text: str) -> str:
        """从文本末尾提取overlap部分"""
        overlap_tokens = 0
        overlap_text = ""

        # 从末尾向前遍历
        sentences = self.SENTENCE_SPLITTERS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in reversed(sentences):
            s_tokens = self._count_tokens(sentence)
            if overlap_tokens + s_tokens > self.chunk_overlap:
                break
            overlap_text = sentence + overlap_text
            overlap_tokens += s_tokens

        return overlap_text

    def _count_tokens(self, text: str) -> int:
        """计算token数"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # fallback估算
        return int(len(text) * 0.7)

    def _encode_sentences(self, sentences: list[str]):
        """生成句向量；模型不可用时使用本地确定性fallback。"""
        try:
            return self.embedding_model.encode(sentences, normalize_embeddings=True)
        except Exception as e:
            logger.warning("SemanticChunker embedding模型不可用，使用fallback句向量: %s", e)
            return np.array([self._fallback_sentence_vector(s) for s in sentences])

    @staticmethod
    def _fallback_sentence_vector(sentence: str, dimension: int = 384) -> list[float]:
        vector = [0.0] * dimension
        tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", sentence) or [sentence]

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign

        norm = float(np.linalg.norm(vector)) or 1.0
        return [v / norm for v in vector]
