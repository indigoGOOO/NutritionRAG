"""营养成分分块器

营养成分表按营养素类别聚合，
保持营养素之间的关联上下文。
表格结构优先保留。

改进：内部集成SemanticChunker处理长文本
"""

from __future__ import annotations

from src.indexing.chunkers.base_chunker import ChunkerUtils
from src.indexing.chunkers.semantic_chunker_v2 import SemanticChunker
from src.indexing.models import (
    BlockType,
    ContentChunk,
    DocCategory,
    TableData,
    UnifiedDocument,
)


class NutritionChunker:
    """营养成分分块器 - 按营养素类别聚合"""

    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.utils = ChunkerUtils()

    # 营养素分类
    NUTRIENT_CATEGORIES = {
        "macro": ["能量", "热量", "蛋白质", "脂肪", "碳水化合物", "膳食纤维", "糖"],
        "vitamin": ["维生素A", "维生素B", "维生素C", "维生素D", "维生素E", "维生素K", "叶酸", "烟酸"],
        "mineral": ["钙", "铁", "锌", "镁", "钾", "钠", "磷", "硒", "铜", "碘"],
        "other": ["胆固醇", "水分", "灰分", "酒精"],
    }

    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        """按营养素类别将成分表分块"""
        chunks = []

        # 优先处理表格（保持原样）
        table_chunks = self._chunk_tables(document)
        chunks.extend(table_chunks)

        # 处理非表格文本（可能需要分块）
        text_chunks = self._chunk_text(document)
        chunks.extend(text_chunks)

        return chunks

    def _chunk_tables(self, document: UnifiedDocument) -> list[ContentChunk]:
        """处理表格类营养数据"""
        chunks = []

        for block in document.blocks:
            if block.block_type != BlockType.TABLE or not isinstance(block.content, TableData):
                continue

            table = block.content

            # 尝试识别营养素类别
            category = self._identify_category(table)
            table_text = self._table_to_text(table)

            chunk = ContentChunk(
                content=table_text,
                chunk_type=f"nutrition_{category}",
                doc_category=DocCategory.NUTRITION,
                source_doc_id=document.doc_id,
                source_block_ids=[block.block_id],
                token_count=self.utils.count_tokens(table_text),
                metadata={
                    "category": category,
                    "headers": table.headers,
                    "row_count": len(table.rows),
                },
            )
            chunks.append(chunk)

        return chunks

    def _chunk_text(self, document: UnifiedDocument) -> list[ContentChunk]:
        """处理非表格营养文本"""
        chunks = []
        text_parts = []

        for block in document.blocks:
            if block.block_type == BlockType.TEXT and isinstance(block.content, str):
                # 检查是否含营养素关键词
                if self._contains_nutrient_keyword(block.content):
                    text_parts.append(block.content)
            elif block.block_type == BlockType.LIST and isinstance(block.content, str):
                text_parts.append(block.content)

        if text_parts:
            full_text = "\n\n".join(text_parts)
            text_chunk = ContentChunk(
                content=full_text,
                chunk_type="nutrition_text",
                doc_category=DocCategory.NUTRITION,
                source_doc_id=document.doc_id,
                source_block_ids=[b.block_id for b in document.blocks],
                token_count=self.utils.count_tokens(full_text),
            )

            # 评估质量，决定是否需要分块
            if self.utils.should_split(text_chunk):
                # 长文本：按营养素类别分块
                sub_chunks = self._split_by_nutrient(text_chunk)
                chunks.extend(sub_chunks)
            else:
                # 短文本：保持原样
                chunks.append(text_chunk)

        return chunks

    def _split_by_nutrient(self, chunk: ContentChunk) -> list[ContentChunk]:
        """按营养素类别分块"""
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
                chunks.append(self.utils.create_sub_chunk(chunk, text, split_by="nutrient"))

        return chunks if chunks else [chunk]

    def _identify_category(self, table: TableData) -> str:
        """识别表格所属的营养素类别"""
        all_headers = " ".join(table.headers) + " " + " ".join(
            " ".join(row) for row in table.rows[:5]  # 只检查前5行
        )

        for category, keywords in self.NUTRIENT_CATEGORIES.items():
            for keyword in keywords:
                if keyword in all_headers:
                    return category

        return "macro"  # 默认大类

    def _contains_nutrient_keyword(self, text: str) -> bool:
        """检查文本是否含营养素关键词"""
        all_keywords = []
        for keywords in self.NUTRIENT_CATEGORIES.values():
            all_keywords.extend(keywords)
        return any(k in text for k in all_keywords)

    @staticmethod
    def _table_to_text(table: TableData) -> str:
        lines = []
        if table.caption:
            lines.append(f"【{table.caption}】")
        if table.headers:
            lines.append(" | ".join(table.headers))
            lines.append("-" * len(" | ".join(table.headers)))
        for row in table.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)

