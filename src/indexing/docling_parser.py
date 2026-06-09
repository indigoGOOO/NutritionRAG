"""Docling解析器

统一处理PDF和图片文档，利用Docling提取文本、表格、图片。
所有非纯文本文档都经过此解析器转为UnifiedDocument。
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.indexing.models import (
    BlockMetadata,
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    ImageData,
    SourceType,
    TableData,
    UnifiedDocument,
)

logger = logging.getLogger(__name__)


class DoclingParser:
    """基于Docling的文档解析器，支持PDF和图片"""

    def __init__(self):
        self._converter = None

    @property
    def converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    def parse(self, file_path: Path) -> UnifiedDocument:
        """解析PDF或图片文件为统一文档格式"""
        logger.info(f"Docling解析: {file_path.name}")

        suffix = file_path.suffix.lower()
        source_type = SourceType.IMAGE if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff") else SourceType.PDF

        result = self.converter.convert(str(file_path))
        doc = result.document

        blocks = []
        page_count = 0

        for i, element in enumerate(doc.iterate_items()):
            item = element[1] if isinstance(element, tuple) else element
            block = self._convert_element(item, position=i)
            if block:
                blocks.append(block)
                if block.metadata.page_number and block.metadata.page_number > page_count:
                    page_count = block.metadata.page_number

        metadata = DocumentMetadata(
            source_path=str(file_path),
            source_type=source_type,
            title=self._extract_title(doc),
            page_count=page_count,
            file_size_bytes=file_path.stat().st_size,
        )

        return UnifiedDocument(
            blocks=blocks,
            metadata=metadata,
        )

    def _convert_element(self, item, position: int) -> DocumentBlock | None:
        """将Docling元素转为DocumentBlock"""
        from docling.datamodel.document import TableItem, TextItem, PictureItem

        page_no = getattr(item, "prov", [{}])
        if page_no and hasattr(page_no[0], "page_no"):
            page_number = page_no[0].page_no
        else:
            page_number = None

        block_meta = BlockMetadata(page_number=page_number, position=position)

        if isinstance(item, TextItem):
            text = item.text.strip()
            if not text:
                return None
            return DocumentBlock(
                block_type=BlockType.TEXT,
                content=text,
                metadata=block_meta,
            )

        elif isinstance(item, TableItem):
            table_data = self._parse_table(item)
            if table_data:
                return DocumentBlock(
                    block_type=BlockType.TABLE,
                    content=table_data,
                    metadata=block_meta,
                )

        elif isinstance(item, PictureItem):
            image_data = ImageData(
                description=getattr(item, "caption", "") or "",
                ocr_text=getattr(item, "text", "") or "",
            )
            return DocumentBlock(
                block_type=BlockType.IMAGE,
                content=image_data,
                metadata=block_meta,
            )

        return None

    def _parse_table(self, table_item) -> TableData | None:
        """解析表格元素为结构化TableData"""
        try:
            df = table_item.export_to_dataframe()
            headers = [str(col) for col in df.columns.tolist()]
            rows = [[str(cell) for cell in row] for row in df.values.tolist()]
            caption = getattr(table_item, "caption", "") or ""
            return TableData(headers=headers, rows=rows, caption=caption)
        except Exception as e:
            logger.warning(f"表格解析失败: {e}")
            text = getattr(table_item, "text", "")
            if text:
                return TableData(headers=[], rows=[[text]], caption="")
            return None

    @staticmethod
    def _extract_title(doc) -> str:
        """从文档中提取标题"""
        if hasattr(doc, "title") and doc.title:
            return doc.title
        for element in doc.iterate_items():
            item = element[1] if isinstance(element, tuple) else element
            if hasattr(item, "label") and "title" in str(getattr(item, "label", "")).lower():
                return getattr(item, "text", "")
        return ""
