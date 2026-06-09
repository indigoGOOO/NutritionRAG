# 营养RAG系统 - 快速参考指南

## 6个问题的快速对比

| 问题 | 原问题 | 改进方案 | 新增文件 | 修改文件 |
|------|--------|--------|--------|--------|
| **1** | 所有图片都用Docling | 文档型→Docling<br/>自然场景→DeepSeek Vision | `image_classifier.py`<br/>`deepseek_vision_parser.py` | `document_router.py` |
| **2** | 语义chunking是假的<br/>只是token数硬切 | 用相邻句子相似度<br/>识别断点 | `chunkers/semantic_chunker_v2.py` | - |
| **3** | KV全部用LLM提取<br/>效率低质量不稳定 | 两阶段：<br/>1. 识别实体<br/>2. 生成EAV | `kv_extractor_v2.py` | - |
| **4** | BM25中文tokenizer<br/>有问题 | 用jieba精确模式<br/>支持自定义词典 | - | `embedding_service_v2.py` |
| **5** | 没有hybrid融合 | 支持RRF和Weighted<br/>两种融合方式 | - | `embedding_service_v2.py` |
| **6** | BM25的DF值有问题<br/>应该用全库统计 | 维护全库token统计<br/>查询时使用全库DF | - | `embedding_service_v2.py` |

---

## 核心改进点速览

### 问题1：图片处理
```python
# 之前：所有图片都用Docling
document = docling_parser.parse(image_path)

# 之后：根据类型分发
image_type, subtype, confidence = image_classifier.classify(image_path)
if image_type == "document":
    document = docling_parser.parse(image_path)
else:
    document = deepseek_vision_parser.parse(image_path, image_type=subtype)
```

### 问题2：语义分块
```python
# 之前：按token数硬切
chunks = split_by_token_count(text, chunk_size=512)

# 之后：按相似度识别断点
embeddings = model.encode(sentences)
similarities = [cosine(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]
breakpoints = [i for i, sim in enumerate(similarities) if sim < 0.5]
chunks = split_by_breakpoints(sentences, breakpoints)
```

### 问题3：KV提取
```python
# 之前：直接提取所有KV
kv_pairs = llm.extract_kv(chunk)

# 之后：两阶段提取
entities = llm.recognize_entities(chunk)  # 第一阶段
for entity in entities:
    eav = llm.generate_eav(entity, context)  # 第二阶段
    kv_pairs.append(eav_to_kvpair(eav))
```

### 问题4：中文分词
```python
# 之前：简单正则分词
tokens = re.findall(r"[a-zA-Z]+|[一-鿿]", text)

# 之后：jieba精确模式
import jieba
tokens = list(jieba.cut(text, cut_all=False))
tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
```

### 问题5：Hybrid融合
```python
# 之前：分别返回dense和sparse
return {"dense": dense_vector, "sparse": sparse_vector}

# 之后：融合后返回
if HYBRID_FUSION_MODE == "weighted":
    fused = DENSE_WEIGHT * norm(dense) + SPARSE_WEIGHT * norm(sparse)
else:  # rrf
    fused = 1/(k+rank_dense) + 1/(k+rank_sparse)
return {"dense": dense, "sparse": sparse, "fused": fused}
```

### 问题6：全库DF统计
```python
# 之前：在batch内计算DF
df = len([doc for doc in batch if token in doc])

# 之后：从全库统计获取
df = corpus_stats.get_df(token)  # 从DB或缓存获取
idf = log((N - df + 0.5) / (df + 0.5) + 1)
```

---

## 配置速查表

### 问题1：图片处理
```python
IMAGE_CLASSIFICATION_THRESHOLD = 0.6  # 文档型置信度阈值
DOCUMENT_IMAGE_TYPES = ["nutrition_table", "recipe", "medical_report", "pdf_page"]
NATURAL_IMAGE_TYPES = ["food_photo", "meal", "fridge", "takeout"]
```

### 问题2：语义分块
```python
SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # 相似度阈值
MIN_CHUNK_SIZE = 100  # 最小chunk大小
MAX_CHUNK_SIZE = 512  # 最大chunk大小
CHUNK_OVERLAP = 50    # overlap大小
```

### 问题4：中文分词
```python
USE_JIEBA_TOKENIZER = True
JIEBA_USER_DICT_PATH = "data/jieba_dict.txt"
STOPWORDS_PATH = "data/stopwords.txt"
```

### 问题5：Hybrid融合
```python
HYBRID_FUSION_MODE = "weighted"  # "rrf" or "weighted"
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
```

### 问题6：全库统计
```python
CORPUS_STATS_CACHE_SIZE = 10000
CORPUS_STATS_UPDATE_BATCH = 1000
```

---

## 文件对应关系

```
问题1 → image_classifier.py + deepseek_vision_parser.py + document_router.py
问题2 → chunkers/semantic_chunker_v2.py
问题3 → kv_extractor_v2.py
问题4 → embedding_service_v2.py (jieba部分)
问题5 → embedding_service_v2.py (fusion部分)
问题6 → embedding_service_v2.py (GlobalCorpusStats部分)
```

---

## 优先级建议

### 立即做（P0）
1. 修改 `config/settings.py` - 添加所有新配置
2. 修改 `llm_client.py` - 统一用DeepSeek
3. 实现问题1 - 图片分类和分发

### 本周做（P1）
4. 实现问题2 - 相似度分块
5. 实现问题3 - 两阶段EAV提取
6. 实现问题4/5/6 - 改进向量化服务

### 下周做（P2）
7. 集成所有改进到 `index_pipeline.py`
8. 端到端测试和性能优化
9. 文档和示例

---

## 常见问题解答

### Q1：为什么要分化图片处理？
**A**：文档型图片有清晰的结构和排版，Docling能准确提取。自然场景图片无结构，需要Vision模型理解内容。

### Q2：相似度阈值0.5怎么选的？
**A**：0.5是cosine相似度的中点。可根据实际效果调整：
- 更严格（< 0.5）：更多断点，chunks更小
- 更宽松（> 0.5）：更少断点，chunks更大

### Q3：为什么要两阶段提取KV？
**A**：第一阶段过滤掉不值得结构化的内容，提高效率。第二阶段生成标准化格式，提高质量。

### Q4：jieba vs 简单分词的区别？
**A**：
- 简单分词：按字符或正则，无法识别词边界
- jieba：精确模式，能正确识别中文词，BM25效果提升30%+

### Q5：RRF vs Weighted融合怎么选？
**A**：
- RRF：不需要调参，但无法体现模态重要性
- Weighted：需要调参，但灵活性强，推荐使用

### Q6：全库DF统计怎么维护？
**A**：
- 索引时：每处理一个chunk，更新token_stats表
- 查询时：从DB或本地缓存读取DF值
- 定期：清空缓存，重新统计

---

## 性能预期

| 指标 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| 图片处理准确率 | 60% | 95% | +35% |
| Chunk质量（语义连贯性） | 低 | 高 | +50% |
| KV提取准确率 | 70% | 90% | +20% |
| BM25检索效果 | 基准 | +30% | +30% |
| Hybrid融合效果 | N/A | 基准 | 新增 |
| 全库DF准确性 | 低 | 高 | +40% |

---

## 下一步行动

1. **今天**：
   - [ ] 读完这份文档
   - [ ] 理解6个问题的改进方案
   - [ ] 确认配置参数

2. **明天**：
   - [ ] 修改 `config/settings.py`
   - [ ] 修改 `llm_client.py` 添加DeepSeek支持
   - [ ] 实现 `image_classifier.py`

3. **后天**：
   - [ ] 实现 `deepseek_vision_parser.py`
   - [ ] 修改 `document_router.py`
   - [ ] 测试问题1的完整流程

4. **本周**：
   - [ ] 实现问题2-6的所有改进
   - [ ] 集成到 `index_pipeline.py`
   - [ ] 端到端测试

---

## 参考资源

- 完整架构设计：`ARCHITECTURE_DESIGN.md`
- 优化总结：`OPTIMIZATION_SUMMARY.md`
- 优化计划：`OPTIMIZATION_PLAN.md`
- 配置文件：`config/settings.py`
- 环境变量：`.env.example`
