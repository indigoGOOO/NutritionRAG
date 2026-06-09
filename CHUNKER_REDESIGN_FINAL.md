# Chunker 架构重设计 - 完整改进方案

## 核心问题

### 问题1：BaseChunker 冗余

```python
# 现有代码
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        ...

class RecipeChunker(BaseChunker):
    def chunk(self, document):
        # 实现
        pass
```

**问题**：
- BaseChunker 只定义了一个方法，没有提供任何通用逻辑
- 继承关系不必要，反而增加了耦合
- 各个 Chunker 的实现差异很大，不适合继承

### 问题2：SemanticChunker 位置不对

```python
# 现有代码在 ChunkRouter 中
def _resplit_chunk(self, chunk):
    semantic_chunker = SemanticChunker()  # ← 在Router中创建
    sub_texts = semantic_chunker.split_text(chunk.content)
```

**问题**：
- SemanticChunker 应该是每个专业化 Chunker 的**内部工具**
- 不应该在 Router 中作为备选方案
- 每个 Chunker 应该自己决定何时调用

### 问题3：ChunkRouter 职责过重

```python
# 现有代码
class ChunkRouter:
    def route(self, document):
        # 1. 路由
        chunker = self._get_document_level_chunker(...)
        chunks = chunker.chunk(document)
        
        # 2. 块级路由
        for chunk in chunks:
            processed = self._route_chunk(chunk)  # ← 处理逻辑
        
        # 3. 质量评估
        quality = self._assess_chunk_quality(chunk)  # ← 评估逻辑
        
        # 4. 后处理
        final_chunks = self._post_process_chunks(...)  # ← 优化逻辑
```

**问题**：
- Router 既做路由，又做处理、评估、优化
- 违反单一职责原则
- 难以维护和扩展

## 改进方案

### 方案1：删除 BaseChunker，使用 Protocol

```python
# src/indexing/chunkers/base_chunker.py
from typing import Protocol

class Chunker(Protocol):
    """Chunker 协议 - 任何实现 chunk() 方法的类都是 Chunker"""
    
    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        """将文档分块为 ContentChunk 列表"""
        ...


class ChunkerUtils:
    """Chunker 工具类 - 提供通用方法"""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """计算 token 数"""
        try:
            import tiktoken
            tokenizer = tiktoken.get_encoding("cl100k_base")
            return len(tokenizer.encode(text))
        except:
            return int(len(text) * 0.7)
    
    @staticmethod
    def should_split(chunk: ContentChunk, threshold: int = 512) -> bool:
        """判断是否需要分块"""
        return chunk.token_count > threshold
    
    @staticmethod
    def create_sub_chunk(
        parent_chunk: ContentChunk,
        content: str,
        split_by: str = "semantic",
    ) -> ContentChunk:
        """创建子 chunk"""
        return ContentChunk(
            content=content,
            chunk_type=parent_chunk.chunk_type,
            doc_category=parent_chunk.doc_category,
            source_doc_id=parent_chunk.source_doc_id,
            source_block_ids=parent_chunk.source_block_ids,
            token_count=ChunkerUtils.count_tokens(content),
            metadata={**parent_chunk.metadata, "split_by": split_by},
        )
```

### 方案2：在各个 Chunker 内部集成 SemanticChunker

```python
# src/indexing/chunkers/recipe_chunker.py
from src.indexing.chunkers.base_chunker import ChunkerUtils
from src.indexing.chunkers.semantic_chunker_v2 import SemanticChunker

class RecipeChunker:
    """菜谱分块器 - 内部集成 SemanticChunker"""
    
    def __init__(self):
        self.semantic_chunker = SemanticChunker()  # ← 内部集成
        self.utils = ChunkerUtils()
    
    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        """将菜谱解耦为多个关联 chunk"""
        # 第一步：提取菜谱结构
        chunks = self._extract_recipe_structure(document)
        
        # 第二步：处理 chunks（内部处理二次分块）
        return self._process_chunks(chunks)
    
    def _process_chunks(self, chunks: list[ContentChunk]) -> list[ContentChunk]:
        """处理 chunks：评估质量，决定是否分块"""
        result = []
        
        for chunk in chunks:
            # 保留类型：不分块
            if chunk.chunk_type in ["recipe_ingredients", "recipe_basic", "table"]:
                result.append(chunk)
            # 长文本：分块
            elif self.utils.should_split(chunk):
                sub_chunks = self._semantic_split(chunk)
                result.extend(sub_chunks)
            # 其他：保持原样
            else:
                result.append(chunk)
        
        return result
    
    def _semantic_split(self, chunk: ContentChunk) -> list[ContentChunk]:
        """使用 SemanticChunker 进行语义分块"""
        sub_texts = self.semantic_chunker.split_text(chunk.content)
        
        return [
            self.utils.create_sub_chunk(chunk, sub_text)
            for sub_text in sub_texts
        ]
    
    def _extract_recipe_structure(self, document: UnifiedDocument) -> list[ContentChunk]:
        """提取菜谱结构 - 保持原有逻辑"""
        # ... 原有实现
        pass
```

### 方案3：简化 ChunkRouter - 只负责路由和用途优化

```python
# src/indexing/chunk_router.py（改进版）
class ChunkRouter:
    """简化的 Chunk Router - 只负责路由和用途优化"""
    
    def route(
        self,
        document: UnifiedDocument,
        purpose: ChunkPurpose = ChunkPurpose.RETRIEVAL,
    ) -> list[ContentChunk]:
        """智能路由"""
        
        # 第一步：获取对应的 Chunker
        chunker = self._get_chunker(document.doc_category)
        
        # 第二步：调用 Chunker（Chunker 内部已处理二次分块）
        chunks = chunker.chunk(document)
        
        # 第三步：根据用途优化（可选）
        final_chunks = self._optimize_for_purpose(chunks, purpose)
        
        return final_chunks
    
    def _get_chunker(self, category: DocCategory):
        """获取对应的 Chunker"""
        from src.indexing.chunkers.daily_record_chunker import DailyRecordChunker
        from src.indexing.chunkers.medical_chunker import MedicalChunker
        from src.indexing.chunkers.nutrition_chunker import NutritionChunker
        from src.indexing.chunkers.personal_chunker import PersonalChunker
        from src.indexing.chunkers.recipe_chunker import RecipeChunker
        from src.indexing.chunkers.semantic_chunker_v2 import SemanticChunker
        
        chunker_map = {
            DocCategory.PERSONAL: PersonalChunker,
            DocCategory.DAILY: DailyRecordChunker,
            DocCategory.NUTRITION: NutritionChunker,
            DocCategory.RECIPE: RecipeChunker,
            DocCategory.MEDICAL: MedicalChunker,
            DocCategory.UNKNOWN: SemanticChunker,
        }
        
        chunker_class = chunker_map.get(category, SemanticChunker)
        return chunker_class()
    
    def _optimize_for_purpose(
        self,
        chunks: list[ContentChunk],
        purpose: ChunkPurpose,
    ) -> list[ContentChunk]:
        """根据用途优化（简化版）"""
        if purpose == ChunkPurpose.RETRIEVAL:
            # 检索用：过滤掉太短的 chunk
            return [c for c in chunks if c.token_count >= 50]
        elif purpose == ChunkPurpose.STORAGE:
            # 存储用：添加元数据
            for chunk in chunks:
                chunk.metadata["optimized_for"] = "storage"
            return chunks
        else:
            return chunks
```

## 文件结构对比

### 改进前

```
src/indexing/chunkers/
├── base_chunker.py          # 基类（几乎没用）
├── semantic_chunker.py      # 通用语义分块
├── semantic_chunker_v2.py   # 改进版语义分块
├── recipe_chunker.py        # 菜谱分块（不处理长文本）
├── nutrition_chunker.py     # 营养分块（不处理长文本）
├── personal_chunker.py      # 个人数据分块
├── daily_record_chunker.py  # 每日记录分块
└── medical_chunker.py       # 医学分块

chunk_router.py             # 在这里处理二次分块（不合理）
```

### 改进后

```
src/indexing/chunkers/
├── base_chunker.py          # 改为 ChunkerUtils 工具类 + Protocol
├── semantic_chunker_v2.py   # 通用语义分块（被各 Chunker 调用）
├── recipe_chunker.py        # 菜谱分块（内部调用 SemanticChunker）
├── nutrition_chunker.py     # 营养分块（内部调用 SemanticChunker）
├── personal_chunker.py      # 个人数据分块（内部调用 SemanticChunker）
├── daily_record_chunker.py  # 每日记录分块（内部调用 SemanticChunker）
└── medical_chunker.py       # 医学分块（内部调用 SemanticChunker）

chunk_router.py             # 只负责路由和用途优化
```

## 优势对比

| 方面 | 改进前 | 改进后 |
|------|--------|--------|
| **职责清晰** | Router 混合了路由和处理 | 每个 Chunker 自己处理，Router 只路由 |
| **代码复用** | SemanticChunker 在 Router 中被调用 | 各 Chunker 可独立使用 SemanticChunker |
| **灵活性** | 固定的处理流程 | 每个 Chunker 可自定义处理策略 |
| **可维护性** | 修改处理逻辑需要改 Router | 修改逻辑只需改对应 Chunker |
| **可测试性** | 难以单独测试 Chunker | 可独立测试每个 Chunker |
| **扩展性** | 添加新 Chunker 需要改 Router | 新 Chunker 自己处理，无需改 Router |
| **代码行数** | Router: 388 行 | Router: ~50 行 |

## 实现步骤

### Step 1：改进 base_chunker.py

```python
# src/indexing/chunkers/base_chunker.py
from typing import Protocol
from src.indexing.models import ContentChunk, UnifiedDocument

class Chunker(Protocol):
    """Chunker 协议"""
    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        ...

class ChunkerUtils:
    """Chunker 工具类"""
    # ... 实现如上
```

### Step 2：改进各个 Chunker

以 RecipeChunker 为例，添加内部 SemanticChunker 集成。

### Step 3：简化 ChunkRouter

删除所有处理逻辑，只保留路由和用途优化。

## 总结

你的建议非常正确：

1. **BaseChunker 可以删除或改为工具类** ✅
   - 改为 ChunkerUtils，提供通用方法
   - 使用 Protocol 定义接口

2. **在各个 Chunker 内部调用 SemanticChunker** ✅
   - 每个 Chunker 自己决定何时分块
   - 更灵活，更易维护

3. **ChunkRouter 只负责路由** ✅
   - 不再处理二次分块
   - 职责清晰，代码简洁

这样的架构更符合**单一职责原则**和**开闭原则**。
