"""统一数据模型定义

所有解析器的输出和管道中间数据都使用这些模型，
确保解析层和下游处理完全解耦。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"


class DocCategory(str, Enum):
    PERSONAL = "personal"
    DAILY = "daily"
    NUTRITION = "nutrition"
    RECIPE = "recipe"
    MEDICAL = "medical"
    UNKNOWN = "unknown"


# ===== 文档块级模型 =====


@dataclass
class TableData:
    """表格结构化数据"""

    headers: list[str]
    rows: list[list[str]]
    caption: str = ""


@dataclass
class ImageData:
    """图片数据"""

    image_bytes: bytes | None = None
    image_path: str = ""
    description: str = ""
    ocr_text: str = ""


@dataclass
class BlockMetadata:
    """块级元数据"""

    page_number: int | None = None
    position: int = 0
    confidence: float = 1.0


@dataclass
class DocumentBlock:
    """文档中的一个内容块"""

    block_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    block_type: BlockType = BlockType.TEXT
    content: str | TableData | ImageData = ""
    metadata: BlockMetadata = field(default_factory=BlockMetadata)


# ===== 文档级模型 =====


@dataclass
class DocumentMetadata:
    """文档级元数据"""

    source_path: str = ""
    source_type: SourceType = SourceType.TEXT
    title: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    page_count: int = 0
    file_size_bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedDocument:
    """统一文档格式 - 所有解析器的输出目标"""

    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_category: DocCategory = DocCategory.UNKNOWN
    blocks: list[DocumentBlock] = field(default_factory=list)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    @property
    def text_content(self) -> str:
        """获取所有文本块的拼接内容"""
        parts = []
        for block in self.blocks:
            if block.block_type == BlockType.TEXT and isinstance(block.content, str):
                parts.append(block.content)
            elif block.block_type == BlockType.TABLE and isinstance(block.content, TableData):
                parts.append(self._table_to_text(block.content))
        return "\n\n".join(parts)

    @staticmethod
    def _table_to_text(table: TableData) -> str:
        lines = []
        if table.caption:
            lines.append(table.caption)
        if table.headers:
            lines.append(" | ".join(table.headers))
            lines.append("-" * (len(" | ".join(table.headers))))
        for row in table.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)


# ===== Chunk级模型 =====


@dataclass
class ContentChunk:
    """分块后的内容单元 - 准备写入向量库"""

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    chunk_type: str = ""
    doc_category: DocCategory = DocCategory.UNKNOWN
    source_doc_id: str = ""
    source_block_ids: list[str] = field(default_factory=list)
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ===== KV 和图谱模型 =====


@dataclass
class KVPair:
    """知识图谱键值对 - 写入PostgreSQL"""

    key: str = ""
    value: dict[str, Any] = field(default_factory=dict)
    entity_type: str = ""
    source_chunk_id: str = ""
    source_doc_id: str = ""


@dataclass
class GraphTriple:
    """关系三元组 - 写入Neo4j"""

    subject: str = ""
    subject_type: str = ""
    predicate: str = ""
    object: str = ""
    object_type: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    source_chunk_id: str = ""


# ===== 管道结果模型 =====


@dataclass
class PipelineResult:
    """索引管道的完整输出"""

    doc_id: str = ""
    chunks: list[ContentChunk] = field(default_factory=list)
    kv_pairs: list[KVPair] = field(default_factory=list)
    triples: list[GraphTriple] = field(default_factory=list)
    embeddings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
