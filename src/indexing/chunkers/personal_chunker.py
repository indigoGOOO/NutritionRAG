"""个人数据分块器

个人数据（个人信息、偏好、过敏源等）按用户ID聚合，
保持完整的用户画像上下文。
支持表单输入（直接结构化）和对话提取（合并到画像chunk）。

改进：内部集成SemanticChunker处理长文本
"""

from __future__ import annotations

import re

from src.indexing.chunkers.base_chunker import ChunkerUtils
from src.indexing.chunkers.semantic_chunker_v2 import SemanticChunker
from src.indexing.models import (
    BlockType,
    ContentChunk,
    DocCategory,
    UnifiedDocument,
)


class PersonalChunker:
    """个人数据分块器 - 按用户ID聚合"""

    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.utils = ChunkerUtils()

    # 个人数据关键模式
    USER_ID_PATTERNS = [
        re.compile(r"(用户[ID编号]?\s*[:：]\s*)([a-zA-Z0-9_-]+)", re.IGNORECASE),
        re.compile(r"(user[_\s]?id\s*[:：]\s*)([a-zA-Z0-9_-]+)", re.IGNORECASE),
    ]

    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        """将个人数据按用户聚合为一个或多个chunk"""
        chunks = []

        # 提取用户ID
        user_id = self._extract_user_id(document.text_content) or document.doc_id[:8]

        # 按数据类型分组
        sections = self._group_by_data_type(document)

        for section_type, blocks_content in sections.items():
            if not blocks_content.strip():
                continue

            section_chunk = ContentChunk(
                content=blocks_content,
                chunk_type=f"personal_{section_type}",
                doc_category=DocCategory.PERSONAL,
                source_doc_id=document.doc_id,
                source_block_ids=[b.block_id for b in document.blocks],
                token_count=self.utils.count_tokens(blocks_content),
                metadata={
                    "user_id": user_id,
                    "section": section_type,
                },
            )

            # 评估质量，决定是否需要分块
            if self.utils.should_split(section_chunk):
                # 长文本：使用语义分块
                sub_chunks = self._semantic_split(section_chunk)
                chunks.extend(sub_chunks)
            else:
                # 短文本：保持原样
                chunks.append(section_chunk)

        # 如果没有分组，按整体处理
        if not chunks and document.text_content.strip():
            full_chunk = ContentChunk(
                content=document.text_content,
                chunk_type="personal_profile",
                doc_category=DocCategory.PERSONAL,
                source_doc_id=document.doc_id,
                source_block_ids=[b.block_id for b in document.blocks],
                token_count=self.utils.count_tokens(document.text_content),
                metadata={"user_id": user_id},
            )

            # 评估质量
            if self.utils.should_split(full_chunk):
                chunks.extend(self._semantic_split(full_chunk))
            else:
                chunks.append(full_chunk)

        return chunks

    def _semantic_split(self, chunk: ContentChunk) -> list[ContentChunk]:
        """使用SemanticChunker进行语义分块"""
        sub_texts = self.semantic_chunker.split_text(chunk.content)

        return [
            self.utils.create_sub_chunk(chunk, sub_text)
            for sub_text in sub_texts
        ]

    def _extract_user_id(self, text: str) -> str | None:
        """从文本中提取用户ID"""
        for pattern in self.USER_ID_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(2)
        return None

    def _group_by_data_type(self, document: UnifiedDocument) -> dict[str, str]:
        """按数据类型分组"""
        sections: dict[str, list[str]] = {
            "profile": [],  # 基本信息
            "preference": [],  # 偏好
            "allergy": [],  # 过敏源
            "health": [],  # 健康指标
        }

        current_section = "profile"
        allergy_keywords = ["过敏", "不耐受", "禁忌"]
        preference_keywords = ["喜欢", "偏好", "口味", "不喜欢"]
        health_keywords = ["血压", "血糖", "血脂", "BMI", "体重", "身高"]

        for block in document.blocks:
            if block.block_type == BlockType.TEXT and isinstance(block.content, str):
                text = block.content

                if any(k in text for k in allergy_keywords):
                    current_section = "allergy"
                elif any(k in text for k in preference_keywords):
                    current_section = "preference"
                elif any(k in text for k in health_keywords):
                    current_section = "health"

                sections[current_section].append(text)

        return {k: "\n".join(v) for k, v in sections.items()}

