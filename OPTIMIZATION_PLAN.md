# 营养RAG系统 - 架构优化方案

## 问题1：图片处理策略分化

### 当前问题
- 所有图片都用Docling处理，无法区分文档型和自然场景

### 优化方案
```
图片输入
  ├─ 文档型（排版清晰、结构化）
  │  ├─ 营养成分表截图
  │  ├─ 菜谱图片（排版清晰）
  │  ├─ 医学报告扫描件
  │  └─ PDF转出来的图片页
  │  └─→ Docling处理 → 结构化提取
  │
  └─ 自然场景（非结构化）
     ├─ 一盘饭拍照
     ├─ 外卖图片
     ├─ 冰箱食材照片
     └─ 食物识别等
     └─→ DeepSeek Vision处理 → 自然语言描述 → 转为UnifiedDocument

### 实现
- 新增 `image_classifier.py`：用DeepSeek判断图片类型
- 新增 `deepseek_vision_parser.py`：处理自然场景图片
- 修改 `document_router.py`：根据图片类型分发
```

## 问题2：语义分块的相似度断点识别

### 当前问题
- 现有SemanticChunker只是按token数硬切，没有考虑语义连贯性
- 相邻句子可能语义差异大，但被强行合并

### 优化方案
```
输入文本 → 句子分割 → 生成句子向量 → 计算相邻相似度 → 识别断点 → 合并chunks

关键参数：
- similarity_threshold: 0.5（相似度低于此值认为是断点）
- min_chunk_size: 100 tokens（最小chunk大小）
- max_chunk_size: 512 tokens（最大chunk大小）

算法：
1. 对每个句子生成embedding
2. 计算相邻句子的cosine相似度
3. 相似度 < threshold → 标记为断点
4. 在断点处切分，同时保持overlap
```

## 问题3：KV提取的两阶段策略

### 当前问题
- 直接用LLM提取所有KV，效率低且质量不稳定
- 没有区分哪些实体值得被结构化

### 优化方案
```
第一阶段：识别值得被结构化的实体
- 输入：chunk文本
- LLM提示词：识别出"食材"、"营养素"、"症状"等核心实体
- 输出：实体列表 + 实体类型

第二阶段：生成EAV Schema
- 输入：实体 + 上下文
- LLM提示词：为每个实体生成标准化的属性-值对
- 输出：EAV三元组（Entity-Attribute-Value）

EAV格式（同时支持PostgreSQL和Neo4j）：
{
  "entity_id": "鸡蛋_001",
  "entity_name": "鸡蛋",
  "entity_type": "ingredient",
  "attributes": [
    {"attr": "分类", "value": "蛋类", "confidence": 0.95},
    {"attr": "热量", "value": "144kcal/100g", "confidence": 0.92},
    {"attr": "蛋白质", "value": "13.3g/100g", "confidence": 0.93},
    {"attr": "禁忌", "value": ["与豆浆同食"], "confidence": 0.85},
    {"attr": "适宜人群", "value": ["一般人群", "儿童"], "confidence": 0.90}
  ]
}

PostgreSQL存储：
- eav_entities 表：entity_id, entity_name, entity_type
- eav_attributes 表：entity_id, attr_name, attr_value, confidence, source_chunk_id

Neo4j存储：
- (Entity {name, type}) -[HAS_ATTRIBUTE {attr, value, confidence}]-> (Attribute)
```

## 问题4：中文BM25分词

### 当前问题
- 现有实现用正则简单分词，中文处理不当
- 无法正确识别中文词边界

### 优化方案
```
使用jieba分词库：
- 精确模式：用于BM25索引
- 支持自定义词典：添加营养学、烹饪领域词汇
- 停用词过滤：去除"的"、"了"等无意义词

实现：
from jieba import cut
from jieba.analyse import extract_tags

def tokenize_chinese(text):
    # 精确模式分词
    tokens = list(cut(text, cut_all=False))
    # 过滤停用词
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens
```

## 问题5：Hybrid Score Fusion

### 当前问题
- 现有实现分别返回dense和sparse结果，没有融合策略

### 优化方案
```
两种融合方案对比：

方案A：RRF (Reciprocal Rank Fusion)
- 公式：score = 1/(k + rank_dense) + 1/(k + rank_sparse)
- 优点：不需要调参，对rank分布鲁棒
- 缺点：无法体现不同模态的重要性

方案B：Weighted Fusion（推荐）
- 公式：score = w_dense * norm(score_dense) + w_sparse * norm(score_sparse)
- 参数：w_dense=0.7, w_sparse=0.3（可调）
- 优点：灵活，可根据场景调整权重
- 缺点：需要调参

建议：
- 对于精准查询（如"鸡蛋热量"）：w_dense=0.5, w_sparse=0.5
- 对于语义查询（如"清淡饮食"）：w_dense=0.7, w_sparse=0.3
- 对于关键词查询（如"海鲜"）：w_dense=0.3, w_sparse=0.7
```

## 问题6：BM25中的DF全库统计

### 当前问题
- 现有实现在每个batch内计算DF，导致DF值不准确
- 应该用全库的文档频率统计

### 优化方案
```
两阶段方案：

第一阶段：离线构建全库统计
- 在索引时，维护全局的词频统计表
- 表结构：token_id, token_text, df (document_frequency), idf

第二阶段：查询时使用全库统计
- 查询时从统计表读取DF值
- 计算BM25分数时使用全库IDF

实现：
class GlobalCorpusStats:
    def __init__(self, db_connection):
        self.db = db_connection
        self.cache = {}  # 本地缓存
    
    def get_df(self, token: str) -> int:
        if token in self.cache:
            return self.cache[token]
        df = self.db.query("SELECT df FROM token_stats WHERE token = ?", token)
        self.cache[token] = df
        return df
    
    def update_stats(self, tokens: list[str], total_docs: int):
        # 更新全库统计
        for token in set(tokens):
            self.db.execute(
                "UPDATE token_stats SET df = df + 1 WHERE token = ?",
                token
            )
```

## 完整改进清单

| # | 模块 | 改进内容 | 优先级 |
|---|------|--------|--------|
| 1 | image_classifier.py | 图片类型分类（文档型/自然场景） | P0 |
| 2 | deepseek_vision_parser.py | DeepSeek Vision处理自然场景图片 | P0 |
| 3 | document_router.py | 根据图片类型分发到不同解析器 | P0 |
| 4 | semantic_chunker.py | 改进：用相似度识别断点 | P1 |
| 5 | kv_extractor.py | 改进：两阶段提取 + EAV格式 | P1 |
| 6 | embedding_service.py | 改进：jieba分词 + 全库DF统计 | P1 |
| 7 | embedding_service.py | 新增：Hybrid score fusion | P1 |
| 8 | llm_client.py | 改进：统一用DeepSeek | P0 |
| 9 | config/settings.py | 新增：DeepSeek配置、融合权重配置 | P0 |

## 实现顺序

**第一批（P0 - 基础改进）**：
1. 修改 `llm_client.py`：统一用DeepSeek
2. 修改 `config/settings.py`：添加DeepSeek配置
3. 新增 `image_classifier.py`：图片类型分类
4. 新增 `deepseek_vision_parser.py`：Vision处理
5. 修改 `document_router.py`：图片分发逻辑

**第二批（P1 - 核心优化）**：
6. 修改 `semantic_chunker.py`：相似度断点识别
7. 修改 `kv_extractor.py`：两阶段EAV提取
8. 修改 `embedding_service.py`：jieba + 全库DF + fusion
9. 新增 `corpus_stats.py`：全库统计管理

## 关键配置参数

```python
# config/settings.py

# LLM统一用DeepSeek
LLM_PROVIDER = "deepseek"
LLM_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 语义分块参数
SEMANTIC_SIMILARITY_THRESHOLD = 0.5
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 512

# Hybrid融合权重
HYBRID_FUSION_MODE = "weighted"  # "rrf" or "weighted"
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3

# 图片处理
IMAGE_CLASSIFICATION_THRESHOLD = 0.6  # 文档型置信度阈值
```
