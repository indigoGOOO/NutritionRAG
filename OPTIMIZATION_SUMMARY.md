# 营养RAG系统 - 6个问题的完整优化方案

## 问题1：图片处理策略分化 ✅

### 改进内容
- **新增文件**：`image_classifier.py` - 图片类型分类器
  - 区分文档型（营养表、菜谱、医学报告、PDF页面）
  - 区分自然场景（拍照、外卖、冰箱、食物识别）
  - 支持LLM Vision分类和规则fallback

- **新增文件**：`deepseek_vision_parser.py` - DeepSeek Vision解析器
  - 处理自然场景图片
  - 生成结构化的自然语言描述
  - 转为UnifiedDocument

- **修改文件**：`document_router.py`
  - 添加图片分类逻辑
  - 根据分类结果分发到Docling或DeepSeek Vision

### 执行流程
```
图片输入
  ↓
ImageClassifier.classify()
  ├─ 文档型 → Docling处理 → 结构化提取
  └─ 自然场景 → DeepSeek Vision → 自然语言描述 → UnifiedDocument
```

---

## 问题2：语义分块的相似度断点识别 ✅

### 改进内容
- **新增文件**：`chunkers/semantic_chunker_v2.py` - 改进的语义分块器
  - 用embedding生成句子向量
  - 计算相邻句子的cosine相似度
  - 相似度 < threshold → 标记为断点
  - 在断点处切分，保持overlap

### 关键参数
```python
SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # 相似度阈值
MIN_CHUNK_SIZE = 100  # 最小chunk大小
MAX_CHUNK_SIZE = 512  # 最大chunk大小
CHUNK_OVERLAP = 50    # overlap大小
```

### 执行流程
```
文本输入
  ↓
按句子分割
  ↓
生成句子embedding
  ↓
计算相邻相似度
  ↓
识别断点（相似度 < 0.5）
  ↓
在断点处切分 + 添加overlap
  ↓
输出chunks
```

---

## 问题3：KV提取的两阶段策略 ✅

### 改进内容
- **新增文件**：`kv_extractor_v2.py` - 改进的KV提取器
  - 第一阶段：LLM识别值得被结构化的实体
  - 第二阶段：LLM为每个实体生成标准化的EAV三元组

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
    }
  ]
}
```

### 存储方案
- **PostgreSQL**：
  - `eav_entities` 表：entity_id, entity_name, entity_type
  - `eav_attributes` 表：entity_id, attr_name, attr_value, confidence

- **Neo4j**：
  - `(Entity {name, type}) -[HAS_ATTRIBUTE {attr, value, confidence}]-> (Attribute)`

### 执行流程
```
Chunk输入
  ↓
第一阶段：识别实体
  LLM提示词：识别食材、营养素、症状等
  输出：实体列表 + 类型
  ↓
第二阶段：生成EAV
  LLM提示词：为每个实体生成属性-值对
  输出：EAV三元组
  ↓
合并同名实体
  ↓
输出KVPair列表
```

---

## 问题4：中文BM25分词 ✅

### 改进内容
- **修改文件**：`embedding_service_v2.py`
  - 使用jieba进行精确模式分词
  - 支持自定义词典（营养学、烹饪领域词汇）
  - 停用词过滤

### 配置
```python
USE_JIEBA_TOKENIZER = True
JIEBA_USER_DICT_PATH = "data/jieba_dict.txt"  # 自定义词典
STOPWORDS_PATH = "data/stopwords.txt"
```

### 执行流程
```
文本输入
  ↓
jieba.cut(text, cut_all=False)  # 精确模式
  ↓
过滤停用词和短词
  ↓
输出token列表
```

---

## 问题5：Hybrid Score Fusion ✅

### 改进内容
- **修改文件**：`embedding_service_v2.py`
  - 支持RRF (Reciprocal Rank Fusion)
  - 支持Weighted Fusion（推荐）

### 两种融合方案

**方案A：RRF**
```
score = 1/(k + rank_dense) + 1/(k + rank_sparse)
k = 60（常用值）
优点：不需要调参，对rank分布鲁棒
缺点：无法体现不同模态的重要性
```

**方案B：Weighted Fusion（推荐）**
```
score = w_dense * norm(score_dense) + w_sparse * norm(score_sparse)
w_dense = 0.7, w_sparse = 0.3（可调）
优点：灵活，可根据场景调整权重
缺点：需要调参
```

### 配置
```python
HYBRID_FUSION_MODE = "weighted"  # "rrf" or "weighted"
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
```

### 执行流程
```
Dense向量 + Sparse向量
  ↓
选择融合方式
  ├─ RRF：计算倒数排名和
  └─ Weighted：加权组合
  ↓
输出融合向量
```

---

## 问题6：BM25中的DF全库统计 ✅

### 改进内容
- **新增文件**：`embedding_service_v2.py` 中的 `GlobalCorpusStats` 类
  - 维护全局的词频统计表
  - 支持本地缓存和数据库持久化
  - 查询时使用全库DF值

### 数据库表结构
```sql
CREATE TABLE token_stats (
  token_id INT PRIMARY KEY,
  token TEXT UNIQUE,
  df INT,  -- document frequency
  idf FLOAT,
  updated_at TIMESTAMP
);
```

### 执行流程
```
索引时：
  ↓
维护全库token统计
  ↓
INSERT/UPDATE token_stats

查询时：
  ↓
GlobalCorpusStats.get_df(token)
  ├─ 本地缓存命中 → 返回
  └─ 缓存未命中 → 查询DB → 缓存 → 返回
  ↓
用全库DF计算BM25分数
```

---

## 完整文件清单

### 新增文件（6个）
| 文件 | 作用 | 问题 |
|------|------|------|
| `image_classifier.py` | 图片类型分类 | 问题1 |
| `deepseek_vision_parser.py` | DeepSeek Vision处理 | 问题1 |
| `chunkers/semantic_chunker_v2.py` | 相似度断点识别 | 问题2 |
| `kv_extractor_v2.py` | 两阶段EAV提取 | 问题3 |
| `embedding_service_v2.py` | jieba + 全库DF + fusion | 问题4/5/6 |
| `corpus_stats.py` | 全库统计管理（可选） | 问题6 |

### 修改文件（3个）
| 文件 | 改进内容 | 问题 |
|------|--------|------|
| `config/settings.py` | 新增配置参数 | 全部 |
| `document_router.py` | 图片分类分发逻辑 | 问题1 |
| `llm_client.py` | 统一用DeepSeek | 全部 |

### 保留文件（不变）
- `models.py` - 数据模型
- `docling_parser.py` - PDF解析
- `text_parser.py` - 文本解析
- `data_cleaner.py` - 数据清洗
- `content_classifier.py` - 内容分类
- `chunkers/*.py` - 专业化分块器
- `graph_builder.py` - 图谱构建
- `index_pipeline.py` - 主管道

---

## 实现优先级

### P0（立即实现）
1. 修改 `config/settings.py` - 添加所有新配置
2. 修改 `llm_client.py` - 统一用DeepSeek
3. 新增 `image_classifier.py` + `deepseek_vision_parser.py`
4. 修改 `document_router.py` - 图片分发逻辑

### P1（核心优化）
5. 新增 `chunkers/semantic_chunker_v2.py` - 相似度分块
6. 新增 `kv_extractor_v2.py` - 两阶段EAV提取
7. 新增 `embedding_service_v2.py` - jieba + fusion + 全库DF

### P2（可选）
8. 新增 `corpus_stats.py` - 独立的全库统计管理
9. 修改 `index_pipeline.py` - 集成新的v2版本

---

## 关键配置参数总结

```python
# config/settings.py

# LLM统一用DeepSeek
LLM_PROVIDER = "deepseek"
LLM_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 问题2：语义分块参数
SEMANTIC_SIMILARITY_THRESHOLD = 0.5
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# 问题5：Hybrid融合参数
HYBRID_FUSION_MODE = "weighted"  # "rrf" or "weighted"
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3

# 问题1：图片处理参数
IMAGE_CLASSIFICATION_THRESHOLD = 0.6
DOCUMENT_IMAGE_TYPES = ["nutrition_table", "recipe", "medical_report", "pdf_page"]
NATURAL_IMAGE_TYPES = ["food_photo", "meal", "fridge", "takeout"]

# 问题4：中文分词参数
USE_JIEBA_TOKENIZER = True
JIEBA_USER_DICT_PATH = "data/jieba_dict.txt"
STOPWORDS_PATH = "data/stopwords.txt"

# 问题6：全库统计参数
CORPUS_STATS_CACHE_SIZE = 10000
CORPUS_STATS_UPDATE_BATCH = 1000
```

---

## 下一步行动

1. **立即**：更新 `.env.example` 添加 `DEEPSEEK_API_KEY`
2. **立即**：修改 `config/settings.py` 添加所有新配置
3. **立即**：修改 `llm_client.py` 添加DeepSeek支持
4. **今天**：实现问题1-3的新增文件
5. **明天**：实现问题4-6的改进
6. **后天**：集成所有改进到 `index_pipeline.py`
7. **测试**：端到端测试所有改进

---

## 验证清单

- [ ] 图片能正确分类为文档型/自然场景
- [ ] 文档型图片用Docling处理，自然场景用DeepSeek Vision处理
- [ ] 语义分块能识别相似度断点
- [ ] KV提取能生成标准化的EAV三元组
- [ ] jieba分词能正确处理中文
- [ ] Hybrid融合能正确计算融合分数
- [ ] 全库DF统计能正确维护和查询
- [ ] 端到端管道能正常运行
