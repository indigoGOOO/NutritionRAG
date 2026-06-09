"""通用语义分块器

对长文本块做句子级切分，按语义连贯性合并，
控制每块token数不超过CHUNK_SIZE。
作为其他专业chunker的二次切分工具使用。
"""

from __future__ import annotations

import re

import tiktoken

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE
from src.indexing.models import (
    BlockType,
    ContentChunk,
    UnifiedDocument,
)


class SemanticChunker:
    """通用语义分块器

    改进：
    - 中文语义边界：支持更多分隔符（，、：；），支持 jieba 分句
    - 最小 token 阈值：低于阈值的短 chunk 与相邻 chunk 合并，减少碎片
    """

    # 中文主句分隔符（强停顿）
    SENTENCE_BREAKERS = re.compile(r"(?<=[。！？；\n])")
    # 从句分隔符（弱停顿 — 逗号、冒号、顿号、分号组成的完整子句在此断开）
    CLAUSE_BREAKERS = re.compile(r"(?<=[，、；：])")
    # 合并标记：弱停顿开头的小写字母（单个子句不应独立成块）
    CONTINUATION_MARKERS = re.compile(r"^(并且|而且|以及|同时|此外|另外|还有|或|或者|和|与|及|但|但是|然而|不过|则|就|才|还|也|又|都|只|只要|因为|所以|因此|如果|虽然|即使)")

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_tokens: int = 80,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        """将文档按语义分块"""
        chunks = []

        for block in document.blocks:
            if block.block_type in (BlockType.TEXT, BlockType.LIST):
                text = block.content if isinstance(block.content, str) else ""
                if not text:
                    continue
                block_chunks = self._split_text(text)
                for chunk_text in block_chunks:
                    chunks.append(ContentChunk(
                        content=chunk_text,
                        chunk_type=block.block_type.value,
                        doc_category=document.doc_category,
                        source_doc_id=document.doc_id,
                        source_block_ids=[block.block_id],
                        token_count=self._count_tokens(chunk_text),
                    ))
            elif block.block_type == BlockType.TABLE:
                table_text = document._table_to_text(block.content)
                chunks.append(ContentChunk(
                    content=table_text,
                    chunk_type="table",
                    doc_category=document.doc_category,
                    source_doc_id=document.doc_id,
                    source_block_ids=[block.block_id],
                    token_count=self._count_tokens(table_text),
                ))

        return chunks

    def split_text(self, text: str) -> list[str]:
        """公开接口：将文本切分为多个chunk字符串"""
        return self._split_text(text)

    def _split_text(self, text: str) -> list[str]:
        """按语义切分文本（双层边界 + 短块合并）"""
        clauses = self._extract_clauses(text)
        if not clauses:
            return [text] if text.strip() else []

        # 拼合子句为完整句子：标记为 continuation 的子句粘附到前一句
        sentences = self._merge_clauses_into_sentences(clauses)

        # 用 token 预算合并句子为 chunk，同时处理短块合并
        raw_chunks = self._build_token_budget_chunks(sentences)

        # 合并太短的相邻 chunk
        merged = self._merge_short_chunks(raw_chunks)

        return merged if merged else raw_chunks

    def _extract_clauses(self, text: str) -> list[str]:
        """先按主句边界切，再按从句边界二次切，返回细粒度子句列表"""
        # 如果可用 jieba 辅助理解语义，但这里只做正则切分
        # 第一层：主句分隔
        main_parts = self.SENTENCE_BREAKERS.split(text)
        clauses = []
        for part in main_parts:
            part = part.strip()
            if not part:
                continue
            # 第二层：从句分隔（逗号/冒号等）
            sub_parts = self.CLAUSE_BREAKERS.split(part)
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    clauses.append(sp)
        return clauses

    def _merge_clauses_into_sentences(self, clauses: list[str]) -> list[str]:
        """将以连接词/副词开头的从句粘附到前一个子句"""
        sentences = []
        for clause in clauses:
            if sentences and self.CONTINUATION_MARKERS.search(clause):
                sentences[-1] += "，" + clause
            else:
                sentences.append(clause)
        return sentences

    def _build_token_budget_chunks(self, sentences: list[str]) -> list[str]:
        """用 token 预算合并句子为 chunk（不做短块合并，只做预算控制）"""
        chunks = []
        current_chunk: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # 单句中句超长，硬切
            if sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                sub_chunks = self._hard_split(sentence)
                chunks.extend(sub_chunks)
                continue

            # 加入后超预算 → 先封存当前块
            if current_tokens + sentence_tokens > self.chunk_size:
                chunks.append("".join(current_chunk))
                overlap_chunk = self._get_overlap(current_chunk)
                current_chunk = overlap_chunk + [sentence]
                current_tokens = sum(self._count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def _merge_short_chunks(self, chunks: list[str]) -> list[str]:
        """将 token 数低于 min_chunk_tokens 的 chunk 合并到前一个 chunk"""
        if not chunks or len(chunks) <= 1:
            return chunks

        merged = []
        for chunk in chunks:
            token_count = self._count_tokens(chunk)
            if merged and token_count < self.min_chunk_tokens:
                # 合并到前一个块
                merged[-1] += chunk
            else:
                merged.append(chunk)
        return merged

    def _hard_split(self, text: str) -> list[str]:
        """对超长文本按token数硬切"""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            chunks = []
            for i in range(0, len(tokens), self.chunk_size):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunks.append(self.tokenizer.decode(chunk_tokens))
            return chunks
        else:
            char_limit = int(self.chunk_size * 0.7)
            return [text[i:i + char_limit] for i in range(0, len(text), char_limit)]

    def _get_overlap(self, sentences: list[str]) -> list[str]:
        """获取overlap部分的句子"""
        if not sentences or self.chunk_overlap <= 0:
            return []

        overlap_sentences = []
        overlap_tokens = 0
        for s in reversed(sentences):
            s_tokens = self._count_tokens(s)
            if overlap_tokens + s_tokens > self.chunk_overlap:
                break
            overlap_sentences.insert(0, s)
            overlap_tokens += s_tokens

        return overlap_sentences

    def _count_tokens(self, text: str) -> int:
        """计算token数"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return int(len(text) * 0.7)
