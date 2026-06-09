"""全局配置管理"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ===== 项目路径 =====
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ===== 索引管线配置 =====
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_CHUNKS_PER_DOC = 200

# ===== LLM 配置（统一用豆包）=====
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "doubao")
LLM_MODEL = os.getenv("LLM_MODEL", "doubao-pro-32k")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ===== Embedding 配置 =====
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# ===== PostgreSQL =====
PG_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", "5432")),
    "database": os.getenv("PG_DATABASE", "nutrition_rag"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "postgres"),
}

# ===== Milvus =====
MILVUS_CONFIG = {
    "host": os.getenv("MILVUS_HOST", "localhost"),
    "port": int(os.getenv("MILVUS_PORT", "19530")),
    "collection": os.getenv("MILVUS_COLLECTION", "nutrition_chunks"),
}

# ===== Neo4j =====
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "neo4j123"),
}

# ===== 文档类型映射 =====
DOC_TYPE_KEYWORDS = {
    "personal": ["个人信息", "用户", "偏好", "过敏", "体重", "身高", "年龄"],
    "daily": ["每日", "今日", "记录", "早餐", "午餐", "晚餐", "加餐", "日期"],
    "nutrition": ["热量", "蛋白质", "脂肪", "碳水", "维生素", "矿物质", "营养成分", "含量"],
    "recipe": ["做法", "步骤", "配料", "食材", "烹饪", "菜谱", "用料", "调料"],
    "medical": ["建议", "禁忌", "适宜", "不宜", "患者", "症状", "疾病", "医嘱"],
}

# ===== 支持的文件类型 =====
SUPPORTED_EXTENSIONS = {
    "pdf": [".pdf"],
    "image": [".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
    "text": [".txt", ".md", ".markdown"],
}

# ===== 语义分块参数（问题2：相似度断点识别）=====
SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # 相似度低于此值认为是断点
MIN_CHUNK_SIZE = 100  # 最小chunk大小（tokens）
MAX_CHUNK_SIZE = 512  # 最大chunk大小（tokens）
CHUNK_OVERLAP = 50

# ===== Hybrid融合参数（问题5：RRF/Weighted融合）=====
HYBRID_FUSION_MODE = "weighted"  # "rrf" or "weighted"
DENSE_WEIGHT = 0.7  # dense向量权重
SPARSE_WEIGHT = 0.3  # sparse向量权重

# ===== 图片处理参数（问题1：图片类型分化）=====
IMAGE_CLASSIFICATION_THRESHOLD = 0.6  # 文档型置信度阈值
DOCUMENT_IMAGE_TYPES = ["nutrition_table", "recipe", "medical_report", "pdf_page"]
NATURAL_IMAGE_TYPES = ["food_photo", "meal", "fridge", "takeout"]

# ===== 中文分词参数（问题4：jieba分词）=====
USE_JIEBA_TOKENIZER = True
JIEBA_USER_DICT_PATH = PROJECT_ROOT / "data" / "jieba_dict.txt"  # 自定义词典
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"

# ===== 全库统计参数（问题6：全库DF统计）=====
CORPUS_STATS_CACHE_SIZE = 10000  # 本地缓存大小
CORPUS_STATS_UPDATE_BATCH = 1000  # 批量更新阈值

# ===== 长期问答记忆参数 =====
KNOWLEDGE_CONTEXT_THRESHOLD = float(os.getenv("KNOWLEDGE_CONTEXT_THRESHOLD", "0.82"))
KNOWLEDGE_DEDUPE_THRESHOLD = float(os.getenv("KNOWLEDGE_DEDUPE_THRESHOLD", "0.92"))
KNOWLEDGE_REUSE_THRESHOLD = float(os.getenv("KNOWLEDGE_REUSE_THRESHOLD", "0.94"))
KNOWLEDGE_MIN_QUALITY_SCORE = float(os.getenv("KNOWLEDGE_MIN_QUALITY_SCORE", "0.65"))
KNOWLEDGE_MIN_ANSWER_CHARS = int(os.getenv("KNOWLEDGE_MIN_ANSWER_CHARS", "80"))
KNOWLEDGE_SAFE_REUSE_INTENTS = {
    "nutrition_info",
    "ingredient_knowledge",
    "recipe_recommend",
    "general",
}
KNOWLEDGE_UNSAFE_REUSE_INTENTS = {
    "disease_diet",
    "diet_advice",
}
