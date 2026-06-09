"""文本解析器

处理纯文本(.txt)和Markdown(.md)文件，
识别其中的结构（标题、列表、表格）并转为UnifiedDocument。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.indexing.models import (
    BlockMetadata,
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    SourceType,
    TableData,
    UnifiedDocument,
)

logger = logging.getLogger(__name__)


class TextParser:
    """纯文本和Markdown解析器"""

    # Markdown表格行的正则
    TABLE_ROW_RE = re.compile(r"^\|(.+)\|$")
    TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$")
    # Markdown标题
    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
    # 列表项
    LIST_RE = re.compile(r"^[\s]*[-*+]\s+(.+)$")

    def parse(self, file_path: Path) -> UnifiedDocument:
        """解析文本文件为统一文档格式"""
        logger.info(f"文本解析: {file_path.name}")

        text = file_path.read_text(encoding="utf-8", errors="replace")
        blocks = self._parse_content(text)

        metadata = DocumentMetadata(
            source_path=str(file_path),
            source_type=SourceType.TEXT,
            title=self._extract_title(blocks),
            file_size_bytes=file_path.stat().st_size,
        )

        return UnifiedDocument(blocks=blocks, metadata=metadata)

    def parse_from_string(self, text: str, source_name: str = "input") -> UnifiedDocument:
        """从字符串解析（用于对话文本等非文件输入）"""
        blocks = self._parse_content(text)
        metadata = DocumentMetadata(
            source_path=source_name,
            source_type=SourceType.TEXT,
        )
        return UnifiedDocument(blocks=blocks, metadata=metadata)

    def _parse_content(self, text: str) -> list[DocumentBlock]:
        """将文本内容解析为blocks列表"""
        lines = text.split("\n")
        blocks: list[DocumentBlock] = []
        current_section: list[str] = []
        current_list: list[str] = []
        position = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            # 检测表格
            if self.TABLE_ROW_RE.match(line):
                # 先保存之前的文本
                if current_section:
                    blocks.append(self._make_text_block(current_section, position))
                    current_section = []
                    position += 1
                if current_list:
                    blocks.append(self._make_list_block(current_list, position))
                    current_list = []
                    position += 1

                table_block, consumed = self._parse_table_block(lines, i, position)
                if table_block:
                    blocks.append(table_block)
                    position += 1
                i += consumed
                continue

            # 检测标题 - 作为段落分隔符
            heading_match = self.HEADING_RE.match(line)
            if heading_match:
                if current_section:
                    blocks.append(self._make_text_block(current_section, position))
                    current_section = []
                    position += 1
                if current_list:
                    blocks.append(self._make_list_block(current_list, position))
                    current_list = []
                    position += 1
                current_section.append(line)
                i += 1
                continue

            # 检测列表
            list_match = self.LIST_RE.match(line)
            if list_match:
                if current_section:
                    blocks.append(self._make_text_block(current_section, position))
                    current_section = []
                    position += 1
                current_list.append(list_match.group(1).strip())
                i += 1
                continue

            # 普通文本
            if current_list:
                blocks.append(self._make_list_block(current_list, position))
                current_list = []
                position += 1

            # 空行作为段落分隔
            if not line.strip():
                if current_section:
                    blocks.append(self._make_text_block(current_section, position))
                    current_section = []
                    position += 1
            else:
                current_section.append(line)

            i += 1

        # 处理剩余内容
        if current_section:
            blocks.append(self._make_text_block(current_section, position))
        if current_list:
            blocks.append(self._make_list_block(current_list, position + 1))

        return blocks

    def _parse_table_block(
        self, lines: list[str], start: int, position: int
    ) -> tuple[DocumentBlock | None, int]:
        """解析Markdown表格"""
        table_lines = []
        i = start

        while i < len(lines) and (self.TABLE_ROW_RE.match(lines[i]) or self.TABLE_SEP_RE.match(lines[i])):
            table_lines.append(lines[i])
            i += 1

        if len(table_lines) < 2:
            return None, 1

        # 解析表头和数据行
        headers = []
        rows = []
        for idx, tl in enumerate(table_lines):
            if self.TABLE_SEP_RE.match(tl):
                continue
            cells = [c.strip() for c in tl.strip("|").split("|")]
            if idx == 0:
                headers = cells
            else:
                rows.append(cells)

        table_data = TableData(headers=headers, rows=rows)
        block = DocumentBlock(
            block_type=BlockType.TABLE,
            content=table_data,
            metadata=BlockMetadata(position=position),
        )
        return block, i - start

    @staticmethod
    def _make_text_block(lines: list[str], position: int) -> DocumentBlock:
        return DocumentBlock(
            block_type=BlockType.TEXT,
            content="\n".join(lines).strip(),
            metadata=BlockMetadata(position=position),
        )

    @staticmethod
    def _make_list_block(items: list[str], position: int) -> DocumentBlock:
        content = "\n".join(f"- {item}" for item in items)
        return DocumentBlock(
            block_type=BlockType.LIST,
            content=content,
            metadata=BlockMetadata(position=position),
        )

    @staticmethod
    def _extract_title(blocks: list[DocumentBlock]) -> str:
        """从第一个标题块提取文档标题"""
        for block in blocks:
            if block.block_type == BlockType.TEXT and isinstance(block.content, str):
                match = re.match(r"^#{1,2}\s+(.+)$", block.content, re.MULTILINE)
                if match:
                    return match.group(1).strip()
        return ""
