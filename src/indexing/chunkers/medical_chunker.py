"""医学建议分块器

医学建议/膳食指南按病症/建议类型聚合，
保持适用人群和限制条件的上下文。

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


class MedicalChunker:
    """医学建议分块器 - 按病症/建议类型聚合"""

    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.utils = ChunkerUtils()

    # 病症关键词
    CONDITION_KEYWORDS = [
        "糖尿病", "高血压", "高血脂", "痛风", "肾病", "肝病",
        "胃病", "心脏病", "肥胖", "贫血", "骨质疏松", "肿瘤", "癌症",
    ]

    # 建议类型关键词
    ADVICE_KEYWORDS = [
        "建议", "禁忌", "适宜", "不宜", "推荐", "每日", "摄入量",
        "注意事项", "警告", "警告", "医嘱", "处方", "治疗",
    ]

    # 适用人群关键词
    CROWD_KEYWORDS = [
        "患者", "儿童", "孕妇", "老年人", "青少年", "婴幼儿",
        "运动员", "素食者", "减肥", "增重", "术后",
    ]

    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        """按病症和建议类型将医学文档分块"""
        chunks = []

        # 分析文档中的病症
        conditions = self._extract_conditions(document.text_content)
        crowds = self._extract_crowds(document.text_content)

        # 按内容类型分组
        sections = self._group_by_advice_type(document)

        for section_type, content_list in sections.items():
            if not content_list:
                continue

            full_content = "\n\n".join(content_list)

            medical_chunk = ContentChunk(
                content=full_content,
                chunk_type=f"medical_{section_type}",
                doc_category=DocCategory.MEDICAL,
                source_doc_id=document.doc_id,
                source_block_ids=[b.block_id for b in document.blocks],
                token_count=self.utils.count_tokens(full_content),
                metadata={
                    "conditions": conditions,
                    "target_crowds": crowds,
                },
            )

            # 评估质量，决定是否需要分块
            if self.utils.should_split(medical_chunk):
                # 长文本：使用语义分块
                sub_chunks = self._semantic_split(medical_chunk)
                chunks.extend(sub_chunks)
            else:
                # 短文本：保持原样
                chunks.append(medical_chunk)

        # 如果没有识别到结构，按整体处理
        if not chunks and document.text_content.strip():
            full_chunk = ContentChunk(
                content=document.text_content,
                chunk_type="medical_general",
                doc_category=DocCategory.MEDICAL,
                source_doc_id=document.doc_id,
                source_block_ids=[b.block_id for b in document.blocks],
                token_count=self.utils.count_tokens(document.text_content),
                metadata={
                    "conditions": conditions,
                    "target_crowds": crowds,
                },
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

    def _extract_conditions(self, text: str) -> list[str]:
        """提取文本中的病症关键词"""
        conditions = []
        for keyword in self.CONDITION_KEYWORDS:
            if keyword in text:
                conditions.append(keyword)
        return list(dict.fromkeys(conditions))

    def _extract_crowds(self, text: str) -> list[str]:
        """提取适用人群关键词"""
        crowds = []
        for keyword in self.CROWD_KEYWORDS:
            if keyword in text:
                crowds.append(keyword)
        return list(dict.fromkeys(crowds))

    def _group_by_advice_type(self, document: UnifiedDocument) -> dict[str, list[str]]:
        """按建议类型分组内容"""
        sections: dict[str, list[str]] = {
            "contraindication": [],  # 禁忌
            "recommendation": [],   # 推荐
            "dosage": [],           # 用量
            "general": [],          # 一般建议
        }

        for block in document.blocks:
            if block.block_type != BlockType.TEXT or not isinstance(block.content, str):
                continue

            text = block.content

            if any(k in text for k in ["禁忌", "不宜", "禁止", "不可"]):
                sections["contraindication"].append(text)
            elif any(k in text for k in ["建议", "推荐", "适宜", "每日", "摄入"]):
                sections["recommendation"].append(text)
            elif any(k in text for k in ["用量", "剂量", "克", "毫克", "ml", "ml"]):
                sections["dosage"].append(text)
            else:
                sections["general"].append(text)

        return sections

