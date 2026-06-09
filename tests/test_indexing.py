"""文本解析器和文档路由器测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.text_parser import TextParser
from src.indexing.models import BlockType, TableData


class TestTextParser:
    """TextParser单元测试"""

    def setup_method(self):
        self.parser = TextParser()

    def test_parse_plain_text(self):
        text = "这是一段普通文本。\n\n这是第二段。"
        doc = self.parser.parse_from_string(text)
        assert len(doc.blocks) == 2
        assert doc.blocks[0].block_type == BlockType.TEXT
        assert "普通文本" in doc.blocks[0].content

    def test_parse_markdown_table(self):
        text = """| 食材 | 热量 | 蛋白质 |
| --- | --- | --- |
| 鸡蛋 | 144kcal | 13.3g |
| 牛奶 | 54kcal | 3.0g |"""
        doc = self.parser.parse_from_string(text)
        table_blocks = [b for b in doc.blocks if b.block_type == BlockType.TABLE]
        assert len(table_blocks) == 1
        table = table_blocks[0].content
        assert isinstance(table, TableData)
        assert "食材" in table.headers
        assert len(table.rows) == 2

    def test_parse_markdown_list(self):
        text = """配料清单：
- 鸡蛋 2个
- 面粉 200g
- 牛奶 100ml"""
        doc = self.parser.parse_from_string(text)
        list_blocks = [b for b in doc.blocks if b.block_type == BlockType.LIST]
        assert len(list_blocks) == 1
        assert "鸡蛋" in list_blocks[0].content

    def test_parse_heading_splits_sections(self):
        text = """# 菜谱标题

这是简介。

## 配料

- 食材A
- 食材B"""
        doc = self.parser.parse_from_string(text)
        assert len(doc.blocks) >= 3

    def test_empty_input(self):
        doc = self.parser.parse_from_string("")
        assert len(doc.blocks) == 0


class TestDataCleaner:
    """DataCleaner单元测试"""

    def setup_method(self):
        from src.indexing.data_cleaner import DataCleaner
        self.cleaner = DataCleaner()

    def test_remove_advertisement(self):
        from src.indexing.models import DocumentBlock, BlockMetadata, UnifiedDocument, DocumentMetadata

        ad_block = DocumentBlock(
            block_type=BlockType.TEXT,
            content="关注公众号获取更多",
            metadata=BlockMetadata(position=0),
        )
        normal_block = DocumentBlock(
            block_type=BlockType.TEXT,
            content="鸡蛋含有丰富的蛋白质，每100克含13.3克蛋白质。",
            metadata=BlockMetadata(position=1),
        )
        doc = UnifiedDocument(blocks=[ad_block, normal_block], metadata=DocumentMetadata())
        cleaned = self.cleaner.clean(doc)
        assert len(cleaned.blocks) == 1
        assert "蛋白质" in cleaned.blocks[0].content

    def test_remove_page_number(self):
        from src.indexing.models import DocumentBlock, BlockMetadata, UnifiedDocument, DocumentMetadata

        page_block = DocumentBlock(
            block_type=BlockType.TEXT,
            content="第 3 页",
            metadata=BlockMetadata(position=0),
        )
        doc = UnifiedDocument(blocks=[page_block], metadata=DocumentMetadata())
        cleaned = self.cleaner.clean(doc)
        assert len(cleaned.blocks) == 0


class TestContentClassifier:
    """ContentClassifier单元测试"""

    def setup_method(self):
        from src.indexing.content_classifier import ContentClassifier
        self.classifier = ContentClassifier()

    def test_classify_nutrition(self):
        from src.indexing.models import DocumentBlock, BlockMetadata, UnifiedDocument, DocumentMetadata, DocCategory

        block = DocumentBlock(
            block_type=BlockType.TEXT,
            content="每100克含热量144千卡，蛋白质13.3克，脂肪8.8克，碳水化合物2.8克。",
            metadata=BlockMetadata(position=0),
        )
        doc = UnifiedDocument(blocks=[block], metadata=DocumentMetadata())
        category = self.classifier.classify(doc)
        assert category == DocCategory.NUTRITION

    def test_classify_recipe(self):
        from src.indexing.models import DocumentBlock, BlockMetadata, UnifiedDocument, DocumentMetadata, DocCategory

        block = DocumentBlock(
            block_type=BlockType.TEXT,
            content="步骤1：将鸡蛋打散。步骤2：加入面粉搅拌。步骤3：倒入锅中煎至金黄。",
            metadata=BlockMetadata(position=0),
        )
        doc = UnifiedDocument(blocks=[block], metadata=DocumentMetadata())
        category = self.classifier.classify(doc)
        assert category == DocCategory.RECIPE
