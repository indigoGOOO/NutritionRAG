"""菜谱分块器

菜谱数据按步骤与配料解耦分块：
- 块1：基础信息（菜名、难度、时间）
- 块2：配料列表（带用量）
- 块3-N：各步骤详情（带配料引用）

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
    TableData,
    UnifiedDocument,
)


class RecipeChunker:
    """菜谱分块器 - 步骤/配料解耦"""

    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.utils = ChunkerUtils()

    # 步骤模式
    STEP_PATTERNS = [
        re.compile(r"^步?骤?\s*(\d+)[：:：]?"),
        re.compile(r"^(\d+)[\.、]\s*"),
        re.compile(r"^第\s*(\d+)\s*步"),
    ]

    # 基础信息关键词
    INFO_KEYWORDS = ["菜名", "名称", "用时", "难度", "适合", "人群", "简介"]
    INGREDIENT_KEYWORDS = ["食材", "配料", "原料", "调料", "用量"]

    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        """将菜谱解耦为多个关联chunk，并处理长文本"""
        # 提取各部分内容
        basic_info = self._extract_basic_info(document)
        ingredients = self._extract_ingredients(document)
        steps = self._extract_steps(document)

        recipe_name = self._extract_recipe_name(document)
        chunks = []

        # 块1：基础信息
        if basic_info:
            basic_chunk = ContentChunk(
                content=basic_info,
                chunk_type="recipe_basic",
                doc_category=DocCategory.RECIPE,
                source_doc_id=document.doc_id,
                source_block_ids=[b.block_id for b in document.blocks],
                token_count=self.utils.count_tokens(basic_info),
                metadata={"recipe_name": recipe_name},
            )
            # 基础信息通常较短，不需要分块
            chunks.append(basic_chunk)

        # 块2：配料列表
        if ingredients:
            ingredient_chunk = ContentChunk(
                content=ingredients,
                chunk_type="recipe_ingredients",
                doc_category=DocCategory.RECIPE,
                source_doc_id=document.doc_id,
                source_block_ids=[b.block_id for b in document.blocks],
                token_count=self.utils.count_tokens(ingredients),
                metadata={"recipe_name": recipe_name},
            )
            # 配料列表是结构化的，保持原样
            chunks.append(ingredient_chunk)

        # 块3-N：各步骤（可能需要分块）
        for i, step in enumerate(steps, 1):
            step_chunk = ContentChunk(
                content=step,
                chunk_type="recipe_step",
                doc_category=DocCategory.RECIPE,
                source_doc_id=document.doc_id,
                source_block_ids=[b.block_id for b in document.blocks],
                token_count=self.utils.count_tokens(step),
                metadata={
                    "recipe_name": recipe_name,
                    "step_number": i,
                    "total_steps": len(steps),
                },
            )

            # 评估步骤质量，决定是否需要分块
            if self.utils.should_split(step_chunk):
                # 长步骤：使用语义分块
                sub_chunks = self._semantic_split(step_chunk)
                chunks.extend(sub_chunks)
            else:
                # 短步骤：保持原样
                chunks.append(step_chunk)

        # 如果没有识别到结构，按整体处理
        if not chunks and document.text_content.strip():
            full_chunk = ContentChunk(
                content=document.text_content,
                chunk_type="recipe_full",
                doc_category=DocCategory.RECIPE,
                source_doc_id=document.doc_id,
                source_block_ids=[b.block_id for b in document.blocks],
                token_count=self.utils.count_tokens(document.text_content),
                metadata={"recipe_name": recipe_name},
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

    def _extract_recipe_name(self, document: UnifiedDocument) -> str:
        """提取菜谱名称"""
        for block in document.blocks:
            if block.block_type == BlockType.TEXT and isinstance(block.content, str):
                # 第一个短行可能是标题
                lines = block.content.split("\n")
                for line in lines[:3]:
                    line = line.strip()
                    if line and len(line) < 20 and not line.startswith("#"):
                        return line
        return document.metadata.title or "未知菜谱"

    def _extract_basic_info(self, document: UnifiedDocument) -> str:
        """提取基础信息"""
        info_parts = []
        found_info = False

        for block in document.blocks:
            text = self._get_block_text(block)
            if not text:
                continue

            # 检查是否含基础信息关键词
            if any(kw in text for kw in self.INFO_KEYWORDS):
                info_parts.append(text)
                found_info = True

        return "\n\n".join(info_parts) if info_parts else ""

    def _extract_ingredients(self, document: UnifiedDocument) -> str:
        """提取配料列表"""
        ingredient_parts = []

        for block in document.blocks:
            text = self._get_block_text(block)
            if not text:
                continue

            if any(kw in text for kw in self.INGREDIENT_KEYWORDS):
                ingredient_parts.append(text)

        return "\n\n".join(ingredient_parts) if ingredient_parts else ""

    def _extract_steps(self, document: UnifiedDocument) -> list[str]:
        """提取烹饪步骤"""
        steps = []
        current_step = ""

        for block in document.blocks:
            text = self._get_block_text(block)
            if not text:
                continue

            # 检测步骤标记
            step_match = None
            for pattern in self.STEP_PATTERNS:
                m = pattern.match(text.strip())
                if m:
                    step_match = m.group(0)
                    break

            if step_match:
                if current_step:
                    steps.append(current_step.strip())
                current_step = text
            else:
                # 尝试在文本内检测步骤
                lines = text.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    for pattern in self.STEP_PATTERNS:
                        if pattern.match(line):
                            if current_step:
                                steps.append(current_step.strip())
                            current_step = line
                            break
                    else:
                        current_step += "\n" + line

        if current_step.strip():
            steps.append(current_step.strip())

        return steps

    def _get_block_text(self, block) -> str:
        """获取block的文本内容"""
        if block.block_type == BlockType.TEXT and isinstance(block.content, str):
            return block.content
        elif block.block_type == BlockType.TABLE and isinstance(block.content, TableData):
            table = block.content
            parts = []
            if table.headers:
                parts.append(" | ".join(table.headers))
            for row in table.rows:
                parts.append(" | ".join(row))
            return "\n".join(parts)
        elif block.block_type == BlockType.LIST and isinstance(block.content, str):
            return block.content
        return ""

