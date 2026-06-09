"""分块器测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.models import (
    BlockMetadata,
    BlockType,
    ContentChunk,
    DocCategory,
    DocumentBlock,
    DocumentMetadata,
    TableData,
    UnifiedDocument,
)


def _make_doc(text: str, category: DocCategory = DocCategory.UNKNOWN) -> UnifiedDocument:
    """辅助函数：创建测试文档"""
    block = DocumentBlock(
        block_type=BlockType.TEXT,
        content=text,
        metadata=BlockMetadata(position=0),
    )
    return UnifiedDocument(
        doc_category=category,
        blocks=[block],
        metadata=DocumentMetadata(),
    )


class TestSemanticChunker:
    """SemanticChunker测试"""

    def setup_method(self):
        from src.indexing.chunkers.semantic_chunker_v2 import SemanticChunker
        self.chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)

    def test_short_text_single_chunk(self):
        doc = _make_doc("这是一段短文本。")
        chunks = self.chunker.chunk(doc)
        assert len(chunks) == 1
        assert "短文本" in chunks[0].content

    def test_long_text_multiple_chunks(self):
        long_text = "这是一段很长的文本。" * 50
        doc = _make_doc(long_text)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) > 1

    def test_empty_document(self):
        doc = _make_doc("")
        chunks = self.chunker.chunk(doc)
        assert len(chunks) == 0


class TestRecipeChunker:
    """RecipeChunker测试"""

    def setup_method(self):
        from src.indexing.chunkers.recipe_chunker import RecipeChunker
        self.chunker = RecipeChunker()

    def test_extract_steps(self):
        text = """番茄炒蛋

配料：
- 番茄 2个
- 鸡蛋 3个
- 盐 适量

步骤1：将鸡蛋打散加盐搅拌均匀
步骤2：番茄切块备用
步骤3：热锅倒油，倒入蛋液炒至凝固"""

        blocks = [
            DocumentBlock(block_type=BlockType.TEXT, content=text, metadata=BlockMetadata(position=0)),
        ]
        doc = UnifiedDocument(
            doc_category=DocCategory.RECIPE,
            blocks=blocks,
            metadata=DocumentMetadata(),
        )
        chunks = self.chunker.chunk(doc)
        step_chunks = [c for c in chunks if c.chunk_type == "recipe_step"]
        assert len(step_chunks) >= 2


class TestNutritionChunker:
    """NutritionChunker测试"""

    def setup_method(self):
        from src.indexing.chunkers.nutrition_chunker import NutritionChunker
        self.chunker = NutritionChunker()

    def test_table_chunk(self):
        table = TableData(
            headers=["营养素", "含量", "单位"],
            rows=[
                ["蛋白质", "13.3", "g"],
                ["脂肪", "8.8", "g"],
                ["碳水化合物", "2.8", "g"],
            ],
            caption="鸡蛋营养成分表",
        )
        blocks = [
            DocumentBlock(block_type=BlockType.TABLE, content=table, metadata=BlockMetadata(position=0)),
        ]
        doc = UnifiedDocument(
            doc_category=DocCategory.NUTRITION,
            blocks=blocks,
            metadata=DocumentMetadata(),
        )
        chunks = self.chunker.chunk(doc)
        assert len(chunks) >= 1
        assert "蛋白质" in chunks[0].content


class TestPersonalChunker:
    """PersonalChunker测试"""

    def setup_method(self):
        from src.indexing.chunkers.personal_chunker import PersonalChunker
        self.chunker = PersonalChunker()

    def test_personal_data(self):
        text = """用户ID: user_001
姓名：张三
年龄：30岁
过敏源：海鲜、花生
偏好：喜欢清淡口味，不喜欢辣"""

        doc = _make_doc(text, DocCategory.PERSONAL)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) >= 1
        # 应该包含过敏信息
        all_content = " ".join(c.content for c in chunks)
        assert "海鲜" in all_content
