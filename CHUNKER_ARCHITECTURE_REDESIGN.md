# Chunker 架构重设计

## 你的观察是对的！

### 当前问题

```
现有架构：
├── BaseChunker（基类）
│   └── 只定义了一个抽象方法 chunk()
│
├── 各个专业化Chunker
│   ├── RecipeChunker
│   ├── NutritionChunker
│   ├── PersonalChunker
│   ├── DailyRecordChunker
│   └── MedicalChunker
│
└── SemanticChunker（通用语义分块）
    └── 在ChunkRouter中被调用

问题：
1. BaseChunker太简单，几乎没有价值
2. 各个Chunker各自为政，没有统一的二次处理机制
3. SemanticChunker的调用分散在ChunkRouter中
4. 无法复用通用的分块逻辑
```

### 你的建议

```
改进方案：
1. 删除BaseChunker（或简化为工具类）
2. 在各个Chunker内部集成SemanticChunker
3. 每个Chunker自己决定何时调用SemanticChunker
4. ChunkRouter只负责路由，不负责二次处理
```

## 改进方案详解

### 方案1：删除BaseChunker，使用Protocol（推荐）

```python
# 之前：BaseChunker基类
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        ...

# 之后：使用Protocol（鸭子类型）
from typing import Protocol

class Chunker(Protocol):
    """Chunker协议 - 任何实现chunk()方法的类都是Chunker"""
    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        ...
```

**优点**：
- 更灵活，不需要继承
- Python风格更Pythonic
- 减少不必要的类层级

### 方案2：在各个Chunker内部集成SemanticChunker

```python
# RecipeChunker改进版
class RecipeChunker:
    def __init__(self):
        self.semantic_chunker = SemanticChunkerV2()  # ← 内部集成
    
    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        # 第一步：提取菜谱结构
        chunks = self._extract_recipe_structure(document)
        
        # 第二步：对每个chunk进行质量评估和处理
        processed_chunks = []
        for chunk in chunks:
            if self._should_split(chunk):
                # 对长文本进行语义分块
                sub_chunks = self._semantic_split(chunk)
                processed_chunks.extend(sub_chunks)
            else:
                # 保持原样（表格、配料列表等）
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _should_split(self, chunk: ContentChunk) -> bool:
        """判断是否需要语义分块"""
        # 表格、配料列表不分块
        if chunk.chunk_type in ["recipe_ingredients", "table"]:
            return False
        
        # 长文本才分块
        if chunk.token_count > 512:
            return True
        
        return False
    
    def _semantic_split(self, chunk: ContentChunk) -> list[ContentChunk]:
        """使用SemanticChunker进行语义分块"""
        sub_texts = self.semantic_chunker.split_text(chunk.content)
        
        result = []
        for sub_text in sub_texts:
            result.append(ContentChunk(
                content=sub_text,
                chunk_type=chunk.chunk_type,
                doc_category=chunk.doc_category,
                source_doc_id=chunk.source_doc_id,
                source_block_ids=chunk.source_block_ids,
                token_count=self.semantic_chunker._count_tokens(sub_text),
                metadata={**chunk.metadata, "split_by": "semantic"},
            ))
        
        return result
```

## 完整的改进架构

### 新的Chunker设计

```python
# 1. 移除BaseChunker（或改为工具类）
# src/indexing/chunkers/base_chunker.py
class ChunkerUtils:
    """Chunker工具类 - 提供通用方法"""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """计算token数"""
        ...
    
    @staticmethod
    def should_split(chunk: ContentChunk, threshold: int = 512) -> bool:
        """判断是否需要分块"""
        return chunk.token_count > threshold
    
    @staticmethod
    def create_sub_chunk(
        parent_chunk: ContentChunk,
        content: str,
        split_by: str = "semantic"
    ) -> ContentChunk:
        """创建子chunk"""
        return ContentChunk(
            content=content,
            chunk_type=parent_chunk.chunk_type,
            doc_category=parent_chunk.doc_category,
            source_doc_id=parent_chunk.source_doc_id,
            source_block_ids=parent_chunk.source_block_ids,
            token_count=ChunkerUtils.count_tokens(content),
            metadata={**parent_chunk.metadata, "split_by": split_by},
        )


# 2. 各个Chunker内部集成SemanticChunker
# src/indexing/chunkers/recipe_chunker.py
class RecipeChunker:
    def __init__(self):
        self.semantic_chunker = SemanticChunkerV2()
        self.utils = ChunkerUtils()
    
    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        # 提取菜谱结构
        chunks = self._extract_recipe_structure(document)
        
        # 对每个chunk进行处理
        return self._process_chunks(chunks)
    
    def _process_chunks(self, chunks: list[ContentChunk]) -> list[ContentChunk]:
        """处理chunks：评估质量，决定是否分块"""
        result = []
        
        for chunk in chunks:
            if self._should_preserve(chunk):
                # 保持原样：表格、配料列表等
                result.append(chunk)
            elif self.utils.should_split(chunk):
                # 需要分块：长文本
                sub_chunks = self._semantic_split(chunk)
                result.extend(sub_chunks)
            else:
                # 质量可以，保持原样
                result.append(chunk)
        
        return result
    
    def _should_preserve(self, chunk: ContentChunk) -> bool:
        """判断是否应该保持原样（不分块）"""
        preserve_types = [
            "recipe_ingredients",
            "recipe_basic",
            "table",
            "list",
        ]
        return chunk.chunk_type in preserve_types
    
    def _semantic_split(self, chunk: ContentChunk) -> list[ContentChunk]:
        """使用SemanticChunker进行语义分块"""
        sub_texts = self.semantic_chunker.split_text(chunk.content)
        
        return [
            self.utils.create_sub_chunk(chunk, sub_text)
            for sub_text in sub_texts
        ]


# 3. 其他Chunker类似改进
# src/indexing/chunkers/nutrition_chunker.py
class NutritionChunker:
    def __init__(self):
        self.semantic_chunker = SemanticChunkerV2()
        self.utils = ChunkerUtils()
    
    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        chunks = self._extract_nutrition_structure(document)
        return self._process_chunks(chunks)
    
    def _process_chunks(self, chunks: list[ContentChunk]) -> list[ContentChunk]:
        result = []
        
        for chunk in chunks:
            if chunk.chunk_type == "table":
                # 表格保持原样
                result.append(chunk)
            elif chunk.chunk_type == "nutrition_text" and self.utils.should_split(chunk):
                # 长文本按营养素类别分块
                sub_chunks = self._split_by_nutrient(chunk)
                result.extend(sub_chunks)
            else:
                result.append(chunk)
        
        return result
    
    def _split_by_nutrient(self, chunk: ContentChunk) -> list[ContentChunk]:
        """按营养素类别分块"""
        # 先按营养素关键词分块
        nutrient_parts = self._split_by_keywords(chunk.content)
        
        # 再对长部分进行语义分块
        result = []
        for part in nutrient_parts:
            if len(part) > 512:
                sub_chunks = self.semantic_chunker.split_text(part)
                for sub_text in sub_chunks:
                    result.append(self.utils.create_sub_chunk(chunk, sub_text))
            else:
                result.append(self.utils.create_sub_chunk(chunk, part))
        
        return result
```

## ChunkRouter的简化

```python
# 改进后的ChunkRouter - 只负责路由，不负责二次处理
class ChunkRouter:
    def route(
        self,
        document: UnifiedDocument,
        purpose: ChunkPurpose = ChunkPurpose.RETRIEVAL,
    ) -> list[ContentChunk]:
        """智能路由"""
        
        # 第一步：文档级路由 - 选择合适的Chunker
        chunker = self._get_chunker(document.doc_category)
        
        # 第二步：调用Chunker（Chunker内部已处理二次分块）
        chunks = chunker.chunk(document)
        
        # 第三步：根据用途优化（可选）
        final_chunks = self._optimize_for_purpose(chunks, purpose)
        
        return final_chunks
    
    def _get_chunker(self, category: DocCategory):
        """获取对应的Chunker"""
        chunker_map = {
            DocCategory.PERSONAL: PersonalChunker,
            DocCategory.DAILY: DailyRecordChunker,
            DocCategory.NUTRITION: NutritionChunker,
            DocCategory.RECIPE: RecipeChunker,
            DocCategory.MEDICAL: MedicalChunker,
            DocCategory.UNKNOWN: SemanticChunkerV2,
        }
        
        chunker_class = chunker_map.get(category, SemanticChunkerV2)
        return chunker_class()
    
    def _optimize_for_purpose(
        self,
        chunks: list[ContentChunk],
        purpose: ChunkPurpose,
    ) -> list[ContentChunk]:
        """根据用途优化（简化版）"""
        if purpose == ChunkPurpose.RETRIEVAL:
            # 过滤掉太短的chunk
            return [c for c in chunks if c.token_count >= 50]
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
├── base_chunker.py          # 改为ChunkerUtils工具类
├── semantic_chunker_v2.py   # 通用语义分块（被各Chunker调用）
├── recipe_chunker.py        # 菜谱分块（内部调用SemanticChunker）
├── nutrition_chunker.py     # 营养分块（内部调用SemanticChunker）
├── personal_chunker.py      # 个人数据分块（内部调用SemanticChunker）
├── daily_record_chunker.py  # 每日记录分块（内部调用SemanticChunker）
└── medical_chunker.py       # 医学分块（内部调用SemanticChunker）

chunk_router.py             # 只负责路由和用途优化
```

## 优势对比

| 方面 | 改进前 | 改进后 |
|------|--------|--------|
| **职责清晰** | ChunkRouter混合了路由和处理 | 每个Chunker自己处理，Router只路由 |
| **代码复用** | SemanticChunker在Router中被调用 | 各Chunker可独立使用SemanticChunker |
| **灵活性** | 固定的处理流程 | 每个Chunker可自定义处理策略 |
| **可维护性** | 修改处理逻辑需要改Router | 修改逻辑只需改对应Chunker |
| **可测试性** | 难以单独测试Chunker | 可独立测试每个Chunker |
| **扩展性** | 添加新Chunker需要改Router | 新Chunker自己处理，无需改Router |

## 实现步骤

### Step 1: 改进ChunkerUtils

```python
# src/indexing/chunkers/base_chunker.py
class ChunkerUtils:
    """Chunker工具类"""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """计算token数"""
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
        """创建子chunk"""
        from src.indexing.models import ContentChunk
        
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

### Step 2: 改进各个Chunker

以RecipeChunker为例：

```python
# src/indexing/chunkers/recipe_chunker.py
from src.indexing.chunkers.base_chunker import ChunkerUtils
from src.indexing.chunkers.semantic_chunker_v2 import SemanticChunker

class RecipeChunker:
    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.utils = ChunkerUtils()
    
    def chunk(self, document: UnifiedDocument) -> list[ContentChunk]:
        """将菜谱解耦为多个关联chunk"""
        # 提取菜谱结构
        chunks = self._extract_recipe_structure(document)
        
        # 处理chunks（内部处理二次分块）
        return self._process_chunks(chunks)
    
    def _process_chunks(self, chunks: list[ContentChunk]) -> list[ContentChunk]:
        """处理chunks：评估质量，决定是否分块"""
        result = []
        
        for chunk in chunks:
            # 保留类型：不分块
            if chunk.chunk_type in ["recipe_ingredients", "recipe_basic", "table"]:
                result.append(chunk)
            # 长文本：分块
            elif self.utils.should_split(chunk):
                sub_chunks = self.semantic_chunker.split_text(chunk.content)
                for sub_text in sub_chunks:
                    result.append(self.utils.create_sub_chunk(chunk, sub_text))
            # 其他：保持原样
            else:
                result.append(chunk)
        
        return result
    
    # ... 其他方法保持不变
```

### Step 3: 简化ChunkRouter

```python
# src/indexing/chunk_router.py
class ChunkRouter:
    def route(
        self,
        document: UnifiedDocument,
        purpose: ChunkPurpose = ChunkPurpose.RETRIEVAL,
    ) -> list[ContentChunk]:
        """智能路由"""
        
        # 获取对应的Chunker
        chunker = self._get_chunker(document.doc_category)
        
        # 调用Chunker（Chunker内部已处理二次分块）
        chunks = chunker.chunk(document)
        
        # 根据用途优化
        final_chunks = self._optimize_for_purpose(chunks, purpose)
        
        return final_chunks
    
    def _get_chunker(self, category: DocCategory):
        """获取对应的Chunker"""
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
        """根据用途优化"""
        if purpose == ChunkPurpose.RETRIEVAL:
            # 检索用：过滤掉太短的chunk
            return [c for c in chunks if c.token_count >= 50]
        elif purpose == ChunkPurpose.STORAGE:
            # 存储用：添加元数据
            for chunk in chunks:
                chunk.metadata["optimized_for"] = "storage"
            return chunks
        else:
            return chunks
```

## 总结

你的建议非常正确：

1. **BaseChunker可以删除或改为工具类** ✅
   - 改为ChunkerUtils，提供通用方法
   - 各Chunker可以直接使用

2. **在各个Chunker内部调用SemanticChunker** ✅
   - 每个Chunker自己决定何时分块
   - 更灵活，更易维护

3. **ChunkRouter只负责路由** ✅
   - 不再处理二次分块
   - 职责清晰，代码简洁

这样的架构更符合**单一职责原则**和**开闭原则**。
