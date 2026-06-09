# 营养RAG系统 - 完整架构设计文档

## 系统概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户输入层                                 │
│  PDF | 图片 | 文本 | 对话 | 表单                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    第一阶段：解析与清洗                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ 文档路由器    │→ │ 多解析器     │→ │ 数据清洗     │           │
│  │ (问题1改进)  │  │ (Docling/    │  │              │           │
│  │              │  │  DeepSeek)   │  │              │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                   │
│  输出：UnifiedDocument (统一中间格式)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    第二阶段：分类与分块                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ 内容分类器    │→ │ 专业化分块器  │→ │ 语义分块器   │           │
│  │              │  │ (5种)        │  │ (问题2改进)  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                   │
│  输出：ContentChunk列表                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    第三阶段：知识提取                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ KV提取器     │  │ 图谱构建器    │  │ 向量化服务   │           │
│  │ (问题3改进)  │  │              │  │ (问题4/5/6)  │           │
│  │ EAV格式      │  │ 三元组        │  │ jieba+fusion │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                   │
│  输出：KVPair | GraphTriple | 向量表示                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    第四阶段：存储与索引                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ PostgreSQL   │  │ Milvus       │  │ Neo4j        │           │
│  │ (EAV数据)    │  │ (向量索引)   │  │ (图谱)       │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 问题1：图片处理策略分化

### 核心改进
**区分文档型和自然场景图片，采用不同的处理策略**

```
图片输入
  ↓
ImageClassifier.classify()
  ├─ 文档型（置信度 > 0.6）
  │  ├─ nutrition_table: 营养成分表截图
  │  ├─ recipe: 菜谱图片（排版清晰）
  │  ├─ medical_report: 医学报告扫描件
  │  └─ pdf_page: PDF转出来的图片页
  │  └─→ Docling处理 → 结构化提取
  │
  └─ 自然场景（置信度 ≤ 0.6）
     ├─ food_photo: 一盘饭拍照
     ├─ meal: 饭菜、餐食
     ├─ fridge: 冰箱食材照片
     └─ takeout: 外卖图片
     └─→ DeepSeek Vision处理 → 自然语言描述 → UnifiedDocument
```

### 实现文件
- **新增**：`image_classifier.py` - 图片类型分类
- **新增**：`deepseek_vision_parser.py` - DeepSeek Vision处理
- **修改**：`document_router.py` - 图片分发逻辑

### 配置参数
```python
IMAGE_CLASSIFICATION_THRESHOLD = 0.6
DOCUMENT_IMAGE_TYPES = ["nutrition_table", "recipe", "medical_report", "pdf_page"]
NATURAL_IMAGE_TYPES = ["food_photo", "meal", "fridge", "takeout"]
```

---

## 问题2：语义分块的相似度断点识别

### 核心改进
**不再用token数硬切，而是用相邻句子的相似度识别断点**

```
文本输入
  ↓
按句子分割（。！？；\n）
  ↓
生成句子embedding（SentenceTransformer）
  ↓
计算相邻句子的cosine相似度
  ↓
识别断点（相似度 < threshold）
  ↓
在断点处切分 + 添加overlap
  ↓
输出chunks
```

### 相似度计算
```
相邻句子相似度 = cosine(embedding_i, embedding_i+1)

如果相似度 < 0.5 → 标记为断点
如果相似度 ≥ 0.5 → 继续合并
```

### 实现文件
- **新增**：`chunkers/semantic_chunker_v2.py` - 改进的语义分块器

### 配置参数
```python
SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # 相似度阈值
MIN_CHUNK_SIZE = 100  # 最小chunk大小（tokens）
MAX_CHUNK_SIZE = 512  # 最大chunk大小（tokens）
CHUNK_OVERLAP = 50    # overlap大小
```

---

## 问题3：KV提取的两阶段策略

### 核心改进
**先识别值得被结构化的实体，再用LLM生成标准化的EAV三元组**

```
Chunk输入
  ↓
第一阶段：实体识别
  LLM提示词：识别食材、营养素、症状等核心实体
  输出：实体列表 + 实体类型 + 上下文
  ↓
第二阶段：EAV生成
  LLM提示词：为每个实体生成标准化的属性-值对
  输出：EAV三元组
  ↓
合并同名实体（按置信度）
  ↓
输出KVPair列表
```

### EAV格式（Entity-Attribute-Value）
```json
{
  "entity_id": "鸡蛋_001",
  "entity_name": "鸡蛋",
  "entity_type": "ingredient",
  "attributes": [
    {
      "attr": "分类",
      "value": "蛋类",
      "confidence": 0.95,
      "source": "文本依据"
    },
    {
      "attr": "热量",
      "value": "144kcal/100g",
      "confidence": 0.92,
      "source": "营养表"
    },
    {
      "attr": "蛋白质",
      "value": "13.3g/100g",
      "confidence": 0.93,
      "source": "营养表"
    },
    {
      "attr": "禁忌",
      "value": ["与豆浆同食"],
      "confidence": 0.85,
      "source": "医学建议"
    },
    {
      "attr": "适宜人群",
      "value": ["一般人群", "儿童"],
      "confidence": 0.90,
      "source": "营养指南"
    }
  ]
}
```

### 存储方案

**PostgreSQL**：
```sql
-- 实体表
CREATE TABLE eav_entities (
  entity_id TEXT PRIMARY KEY,
  entity_name TEXT,
  entity_type TEXT,
  created_at TIMESTAMP
);

-- 属性表
CREATE TABLE eav_attributes (
  id SERIAL PRIMARY KEY,
  entity_id TEXT REFERENCES eav_entities(entity_id),
  attr_name TEXT,
  attr_value TEXT,
  confidence FLOAT,
  source_chunk_id TEXT,
  created_at TIMESTAMP
);
```

**Neo4j**：
```cypher
(Entity {id, name, type}) 
  -[HAS_ATTRIBUTE {attr, value, confidence}]-> 
(Attribute {name})
```

### 实现文件
- **新增**：`kv_extractor_v2.py` - 两阶段EAV提取

---

## 问题4：中文BM25分词

### 核心改进
**用jieba进行精确模式分词，支持自定义词典和停用词过滤**

```
文本输入
  ↓
jieba.cut(text, cut_all=False)  # 精确模式
  ↓
过滤停用词（的、了、和、是等）
  ↓
过滤短词（长度 ≤ 1）
  ↓
输出token列表
```

### 分词示例
```
输入：鸡蛋含有丰富的蛋白质，每100克含13.3克蛋白质。
输出：['鸡蛋', '含有', '丰富', '蛋白质', '100', '克', '13.3', '克', '蛋白质']
```

### 实现文件
- **修改**：`embedding_service_v2.py` - 集成jieba分词

### 配置参数
```python
USE_JIEBA_TOKENIZER = True
JIEBA_USER_DICT_PATH = "data/jieba_dict.txt"  # 自定义词典
STOPWORDS_PATH = "data/stopwords.txt"
```

### 自定义词典示例
```
# data/jieba_dict.txt
鸡蛋 3 n
番茄 3 n
蛋白质 3 n
热量 3 n
营养素 3 n
```

---

## 问题5：Hybrid Score Fusion

### 核心改进
**支持RRF和Weighted两种融合方式，灵活组合dense和sparse向量**

### 两种融合方案

**方案A：RRF (Reciprocal Rank Fusion)**
```
score = 1/(k + rank_dense) + 1/(k + rank_sparse)
k = 60（常用值）

优点：
- 不需要调参
- 对rank分布鲁棒
- 理论基础扎实

缺点：
- 无法体现不同模态的重要性
- 需要在检索时计算rank
```

**方案B：Weighted Fusion（推荐）**
```
score = w_dense * norm(score_dense) + w_sparse * norm(score_sparse)
w_dense = 0.7, w_sparse = 0.3（可调）

优点：
- 灵活，可根据场景调整权重
- 计算简单，实时性好
- 易于理解和调试

缺点：
- 需要调参
- 权重选择影响效果
```

### 场景化权重建议
```python
# 精准查询（如"鸡蛋热量"）
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5

# 语义查询（如"清淡饮食"）
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3

# 关键词查询（如"海鲜"）
DENSE_WEIGHT = 0.3
SPARSE_WEIGHT = 0.7
```

### 实现文件
- **修改**：`embedding_service_v2.py` - 集成fusion逻辑

### 配置参数
```python
HYBRID_FUSION_MODE = "weighted"  # "rrf" or "weighted"
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
```

---

## 问题6：BM25中的DF全库统计

### 核心改进
**DF值从全库统计获取，而不是在每个batch内计算**

```
索引时：
  ↓
维护全库token统计表
  ↓
INSERT/UPDATE token_stats
  token_id | token | df | idf | updated_at

查询时：
  ↓
GlobalCorpusStats.get_df(token)
  ├─ 本地缓存命中 → 返回
  └─ 缓存未命中 → 查询DB → 缓存 → 返回
  ↓
用全库DF计算BM25分数
```

### BM25公式
```
BM25(q, d) = Σ IDF(qi) * (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * |d| / avgdl))

其中：
- IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
- N = 全库文档总数
- df(qi) = 包含qi的文档数（从全库统计获取）
- f(qi, d) = qi在文档d中的频率
- |d| = 文档d的长度
- avgdl = 平均文档长度
- k1 = 1.2, b = 0.75（常用参数）
```

### 数据库表结构
```sql
CREATE TABLE token_stats (
  token_id INT PRIMARY KEY,
  token TEXT UNIQUE NOT NULL,
  df INT NOT NULL,  -- document frequency
  idf FLOAT,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_token ON token_stats(token);
```

### 实现文件
- **新增**：`embedding_service_v2.py` 中的 `GlobalCorpusStats` 类

### 配置参数
```python
CORPUS_STATS_CACHE_SIZE = 10000  # 本地缓存大小
CORPUS_STATS_UPDATE_BATCH = 1000  # 批量更新阈值
```

---

## 完整文件清单

### 新增文件（6个）
| 文件 | 作用 | 问题 | 优先级 |
|------|------|------|--------|
| `image_classifier.py` | 图片类型分类 | 问题1 | P0 |
| `deepseek_vision_parser.py` | DeepSeek Vision处理 | 问题1 | P0 |
| `chunkers/semantic_chunker_v2.py` | 相似度断点识别 | 问题2 | P1 |
| `kv_extractor_v2.py` | 两阶段EAV提取 | 问题3 | P1 |
| `embedding_service_v2.py` | jieba + fusion + 全库DF | 问题4/5/6 | P1 |
| `corpus_stats.py` | 全库统计管理（可选） | 问题6 | P2 |

### 修改文件（3个）
| 文件 | 改进内容 | 问题 | 优先级 |
|------|--------|------|--------|
| `config/settings.py` | 新增配置参数 | 全部 | P0 |
| `document_router.py` | 图片分类分发逻辑 | 问题1 | P0 |
| `llm_client.py` | 统一用DeepSeek | 全部 | P0 |

### 保留文件（不变）
- `models.py` - 数据模型
- `docling_parser.py` - PDF解析
- `text_parser.py` - 文本解析
- `data_cleaner.py` - 数据清洗
- `content_classifier.py` - 内容分类
- `chunkers/*.py` - 专业化分块器（除semantic_chunker_v2）
- `graph_builder.py` - 图谱构建
- `index_pipeline.py` - 主管道

---

## 关键配置参数总结

```python
# config/settings.py

# ===== LLM配置（统一用DeepSeek）=====
LLM_PROVIDER = "deepseek"
LLM_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ===== 问题2：语义分块参数 =====
SEMANTIC_SIMILARITY_THRESHOLD = 0.5
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ===== 问题5：Hybrid融合参数 =====
HYBRID_FUSION_MODE = "weighted"  # "rrf" or "weighted"
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3

# ===== 问题1：图片处理参数 =====
IMAGE_CLASSIFICATION_THRESHOLD = 0.6
DOCUMENT_IMAGE_TYPES = ["nutrition_table", "recipe", "medical_report", "pdf_page"]
NATURAL_IMAGE_TYPES = ["food_photo", "meal", "fridge", "takeout"]

# ===== 问题4：中文分词参数 =====
USE_JIEBA_TOKENIZER = True
JIEBA_USER_DICT_PATH = PROJECT_ROOT / "data" / "jieba_dict.txt"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"

# ===== 问题6：全库统计参数 =====
CORPUS_STATS_CACHE_SIZE = 10000
CORPUS_STATS_UPDATE_BATCH = 1000
```

---

## 实现路线图

### 第一周：P0基础改进
- [ ] Day 1: 修改 `config/settings.py` + `llm_client.py`
- [ ] Day 2: 实现 `image_classifier.py` + `deepseek_vision_parser.py`
- [ ] Day 3: 修改 `document_router.py` 集成图片分发
- [ ] Day 4: 测试问题1的完整流程

### 第二周：P1核心优化
- [ ] Day 5: 实现 `chunkers/semantic_chunker_v2.py`
- [ ] Day 6: 实现 `kv_extractor_v2.py`
- [ ] Day 7: 实现 `embedding_service_v2.py`
- [ ] Day 8: 测试问题2-6的完整流程

### 第三周：P2集成与优化
- [ ] Day 9: 修改 `index_pipeline.py` 集成v2版本
- [ ] Day 10: 端到端测试
- [ ] Day 11: 性能优化和调参
- [ ] Day 12: 文档和示例

---

## 验证清单

- [ ] 图片能正确分类为文档型/自然场景
- [ ] 文档型图片用Docling处理，自然场景用DeepSeek Vision处理
- [ ] 语义分块能识别相似度断点，chunk质量提升
- [ ] KV提取能生成标准化的EAV三元组
- [ ] jieba分词能正确处理中文，BM25效果提升
- [ ] Hybrid融合能正确计算融合分数
- [ ] 全库DF统计能正确维护和查询，BM25准确性提升
- [ ] 端到端管道能正常运行
- [ ] 性能指标达到预期（延迟、准确率等）
