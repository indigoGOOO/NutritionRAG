"""数据清洗模块

对UnifiedDocument中的所有blocks进行统一清洗：
- 去广告、水印
- 去乱码字符
- 去页眉页脚
- 表格修复
- 空内容过滤
"""

from __future__ import annotations

import logging
import re

from src.indexing.models import (
    BlockType,
    DocumentBlock,
    ImageData,
    TableData,
    UnifiedDocument,
)

logger = logging.getLogger(__name__)


class DataCleaner:
    """文档数据清洗器"""

    # 广告/水印常见模式
    AD_PATTERNS = [
        re.compile(r"(扫码|关注|公众号|微信号|抖音号|小红书)", re.IGNORECASE),
        re.compile(r"(版权所有|copyright|all rights reserved)", re.IGNORECASE),
        re.compile(r"(www\.\S+\.(com|cn|net|org))", re.IGNORECASE),
        re.compile(r"(免费领取|限时优惠|点击购买|立即下单)", re.IGNORECASE),
        re.compile(r"(转载请注明|禁止转载|侵权必究)"),
    ]

    # 页眉页脚模式
    HEADER_FOOTER_PATTERNS = [
        re.compile(r"^第?\s*\d+\s*页$"),
        re.compile(r"^Page\s*\d+", re.IGNORECASE),
        re.compile(r"^\d+\s*/\s*\d+$"),
        re.compile(r"^[-—]+\s*\d+\s*[-—]+$"),
    ]

    # 乱码字符范围（保留中文、英文、数字、常见标点）
    VALID_CHAR_RE = re.compile(
        r"[^一-鿿　-〿＀-￯"
        r"a-zA-Z0-9\s"
        r"，。！？、；：""''（）【】《》·…—\-\.\,\!\?\;\:\"\'\(\)\[\]\<\>\/\\\+\=\*\#\@\%\&\^\~\|]"
    )

    def clean(self, document: UnifiedDocument) -> UnifiedDocument:
        """清洗文档中的所有blocks"""
        cleaned_blocks = []

        for block in document.blocks:
            cleaned = self._clean_block(block)
            if cleaned and self._is_meaningful(cleaned):
                cleaned_blocks.append(cleaned)

        removed_count = len(document.blocks) - len(cleaned_blocks)
        if removed_count > 0:
            logger.info(f"清洗完成: 移除 {removed_count} 个无效块")

        document.blocks = cleaned_blocks
        return document

    def _clean_block(self, block: DocumentBlock) -> DocumentBlock | None:
        """清洗单个block"""
        if block.block_type == BlockType.TEXT:
            return self._clean_text_block(block)
        elif block.block_type == BlockType.TABLE:
            return self._clean_table_block(block)
        elif block.block_type == BlockType.IMAGE:
            return self._clean_image_block(block)
        elif block.block_type == BlockType.LIST:
            return self._clean_text_block(block)
        return block

    def _clean_text_block(self, block: DocumentBlock) -> DocumentBlock | None:
        """清洗文本块"""
        if not isinstance(block.content, str):
            return None

        text = block.content

        # 去页眉页脚
        if self._is_header_footer(text):
            return None

        # 去广告
        if self._is_advertisement(text):
            return None

        # 去乱码
        text = self._remove_garbled(text)

        # 去多余空白
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.strip()

        if not text:
            return None

        block.content = text
        return block

    def _clean_table_block(self, block: DocumentBlock) -> DocumentBlock | None:
        """清洗表格块"""
        if not isinstance(block.content, TableData):
            return None

        table = block.content

        # 过滤空行
        table.rows = [row for row in table.rows if any(cell.strip() for cell in row)]

        # 清洗单元格内容
        table.headers = [self._clean_cell(h) for h in table.headers]
        table.rows = [[self._clean_cell(cell) for cell in row] for row in table.rows]

        # 表格至少要有1行数据
        if not table.rows and not table.headers:
            return None

        block.content = table
        return block

    def _clean_image_block(self, block: DocumentBlock) -> DocumentBlock | None:
        """清洗图片块"""
        if not isinstance(block.content, ImageData):
            return None

        image = block.content
        if image.description:
            image.description = image.description.strip()
        if image.ocr_text:
            image.ocr_text = self._remove_garbled(image.ocr_text).strip()

        # 图片至少要有描述或OCR文本
        if not image.description and not image.ocr_text and not image.image_bytes:
            return None

        block.content = image
        return block

    def _is_advertisement(self, text: str) -> bool:
        """判断是否为广告内容"""
        for pattern in self.AD_PATTERNS:
            if pattern.search(text):
                # 短文本中出现广告关键词，大概率是广告
                if len(text) < 100:
                    return True
        return False

    def _is_header_footer(self, text: str) -> bool:
        """判断是否为页眉页脚"""
        text = text.strip()
        if len(text) > 50:
            return False
        for pattern in self.HEADER_FOOTER_PATTERNS:
            if pattern.match(text):
                return True
        return False

    def _remove_garbled(self, text: str) -> str:
        """移除乱码字符"""
        # 计算乱码比例，如果超过30%则整段丢弃
        garbled_chars = self.VALID_CHAR_RE.findall(text)
        if len(text) > 0 and len(garbled_chars) / len(text) > 0.3:
            logger.debug(f"乱码比例过高，丢弃: {text[:50]}...")
            return ""
        # 否则只移除乱码字符
        return self.VALID_CHAR_RE.sub("", text)

    @staticmethod
    def _clean_cell(cell: str) -> str:
        """清洗表格单元格"""
        cell = cell.strip()
        cell = re.sub(r"\s+", " ", cell)
        return cell

    @staticmethod
    def _is_meaningful(block: DocumentBlock) -> bool:
        """判断block是否有实际内容"""
        if block.block_type == BlockType.TEXT and isinstance(block.content, str):
            return len(block.content.strip()) > 2
        elif block.block_type == BlockType.LIST and isinstance(block.content, str):
            return len(block.content.strip()) > 2
        elif block.block_type == BlockType.TABLE and isinstance(block.content, TableData):
            return bool(block.content.rows or block.content.headers)
        elif block.block_type == BlockType.IMAGE:
            return True
        return False
