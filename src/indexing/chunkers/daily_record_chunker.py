"""每日记录分块器

每日记录（饮食日志、健康指标、活动等）按日期聚合，
保持日间变化趋势的上下文。

改进：内部集成SemanticChunker处理长文本
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from src.indexing.chunkers.base_chunker import ChunkerUtils
from src.indexing.chunkers.semantic_chunker_v2 import SemanticChunker
from src.indexing.models import (
    BlockType,
    ContentChunk,
    DocCategory,
    TableData,
    UnifiedDocument,
)


class DailyRecordChunker:
    """每日记录分块器 - 按日期聚合"""

    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.utils = ChunkerUtils()

    # 日期模式
    DATE_PATTERNS = [
        re.compile(r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)"),
        re.compile(r"(\d{4}\d{2}\d{2})"),
        re.compile(r"(今|昨|前|明)天"),
    ]

    MEAL_KEYWORDS = ["早餐", "午餐", "晚餐", "加餐", "早", "午", "晚", "宵夜"]

    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        """按日期将每日记录分块"""
        chunks = []

        # 尝试提取日期
        dates = self._extract_dates(document.text_content)
        if not dates:
            dates = [datetime.now().strftime("%Y-%m-%d")]

        # 按日期分组
        date_groups: dict[str, list[str]] = {d: [] for d in dates}

        for block in document.blocks:
            block_text = self._get_block_text(block)
            if not block_text:
                continue

            # 识别块中的日期
            block_dates = self._extract_dates(block_text)
            target_date = block_dates[0] if block_dates else dates[0]

            date_groups.setdefault(target_date, []).append(block_text)

        # 生成每个日期的chunk
        for date, contents in date_groups.items():
            if not contents:
                continue

            full_content = "\n".join(contents)

            # 分析当天包含的餐次
            meals = self._extract_meals(full_content)

            daily_chunk = ContentChunk(
                content=full_content,
                chunk_type="daily_record",
                doc_category=DocCategory.DAILY,
                source_doc_id=document.doc_id,
                source_block_ids=[b.block_id for b in document.blocks],
                token_count=self.utils.count_tokens(full_content),
                metadata={
                    "date": date,
                    "meals": meals,
                },
            )

            # 评估质量，决定是否需要分块
            if self.utils.should_split(daily_chunk):
                # 长文本：使用语义分块
                sub_chunks = self._semantic_split(daily_chunk)
                chunks.extend(sub_chunks)
            else:
                # 短文本：保持原样
                chunks.append(daily_chunk)

        return chunks

    def _semantic_split(self, chunk: ContentChunk) -> list[ContentChunk]:
        """使用SemanticChunker进行语义分块"""
        sub_texts = self.semantic_chunker.split_text(chunk.content)

        return [
            self.utils.create_sub_chunk(chunk, sub_text)
            for sub_text in sub_texts
        ]

    def _extract_dates(self, text: str) -> list[str]:
        """提取文本中的所有日期"""
        dates = []
        for pattern in self.DATE_PATTERNS:
            matches = pattern.findall(text)
            dates.extend(matches)
        return list(dict.fromkeys(dates))  # 去重保持顺序

    def _extract_meals(self, text: str) -> list[str]:
        """识别文本中的餐次"""
        meals = []
        for keyword in self.MEAL_KEYWORDS:
            if keyword in text:
                if keyword == "早餐":
                    meals.append("breakfast")
                elif keyword == "午餐":
                    meals.append("lunch")
                elif keyword == "晚餐":
                    meals.append("dinner")
                elif keyword == "加餐":
                    meals.append("snack")
        return list(dict.fromkeys(meals))

    def _get_block_text(self, block: Any) -> str:
        """获取block的文本内容"""
        if block.block_type == BlockType.TEXT and isinstance(block.content, str):
            return block.content
        elif block.block_type == BlockType.TABLE and isinstance(block.content, TableData):
            table = block.content
            parts = []
            if table.caption:
                parts.append(table.caption)
            if table.headers:
                parts.append(" | ".join(table.headers))
            for row in table.rows:
                parts.append(" | ".join(row))
            return "\n".join(parts)
        elif block.block_type == BlockType.LIST and isinstance(block.content, str):
            return block.content
        return ""

