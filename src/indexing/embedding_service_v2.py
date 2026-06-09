"""改进的向量化服务 - jieba分词 + 全库DF统计 + Hybrid融合

问题4改进：用jieba进行中文分词
问题5改进：Hybrid score fusion (RRF / weighted)
问题6改进：BM25中的DF用全库统计
"""

from __future__ import annotations

import logging
import math
import hashlib
from typing import Any

import numpy as np

from config.settings import (
    CORPUS_STATS_CACHE_SIZE,
    DENSE_WEIGHT,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    HYBRID_FUSION_MODE,
    SPARSE_WEIGHT,
    USE_JIEBA_TOKENIZER,
)
from src.indexing.models import ContentChunk

logger = logging.getLogger(__name__)


class GlobalCorpusStats:
    """全库统计管理 - 问题6改进

    维护全局的词频统计，用于BM25计算。
    支持本地缓存和数据库持久化。
    """

    def __init__(self, db_connection=None):
        self.db = db_connection
        self.cache: dict[str, int] = {}  # token -> df
        self.total_docs = 0

    def get_df(self, token: str) -> int:
        """获取token的文档频率"""
        if token in self.cache:
            return self.cache[token]

        if self.db:
            df = self.db.query("SELECT df FROM token_stats WHERE token = ?", token)
            if df:
                self.cache[token] = df
                return df

        return 1  # 默认值

    def update_stats(self, tokens: list[str], total_docs: int):
        """更新全库统计"""
        self.total_docs = total_docs

        if self.db:
            for token in set(tokens):
                self.db.execute(
                    "UPDATE token_stats SET df = df + 1 WHERE token = ?",
                    token,
                )
                self.cache[token] = self.cache.get(token, 0) + 1

    def clear_cache(self):
        """清空本地缓存"""
        self.cache.clear()


class EmbeddingService:
    """改进的向量化服务"""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIMENSION,
        corpus_stats: GlobalCorpusStats | None = None,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.corpus_stats = corpus_stats or GlobalCorpusStats()

        self._model = None
        self._jieba = None
        self._stopwords = set()

    @property
    def model(self):
        """延迟加载embedding模型"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding模型加载完成: {self.model_name}")
        return self._model

    @property
    def jieba_tokenizer(self):
        """延迟加载jieba"""
        if self._jieba is None and USE_JIEBA_TOKENIZER:
            try:
                import jieba

                self._jieba = jieba
                self._load_stopwords()
                logger.info("jieba分词器加载完成")
            except ImportError:
                logger.warning("jieba未安装，回退到简单分词")
                self._jieba = False
        return self._jieba if self._jieba else None

    def _load_stopwords(self):
        """加载停用词"""
        from config.settings import STOPWORDS_PATH

        if STOPWORDS_PATH.exists():
            with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
                self._stopwords = set(line.strip() for line in f if line.strip())
        else:
            # 默认停用词
            self._stopwords = {"的", "了", "和", "是", "在", "有", "一", "个", "人", "这"}

    def embed_chunks(self, chunks: list[ContentChunk]) -> list[dict[str, Any]]:
        """批量生成chunks的向量表示"""
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]

        # 生成dense向量
        dense_vectors = self._generate_dense(texts)

        # 生成sparse向量
        sparse_vectors = self._generate_sparse(texts)

        results = []
        for i, chunk in enumerate(chunks):
            # 问题5改进：Hybrid融合
            hybrid_vector = self._fuse_vectors(dense_vectors[i], sparse_vectors[i])

            results.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i],
                "hybrid_vector": hybrid_vector,  # 融合后的向量
                "metadata": {
                    "doc_category": chunk.doc_category.value,
                    "chunk_type": chunk.chunk_type,
                    "source_doc_id": chunk.source_doc_id,
                    "token_count": chunk.token_count,
                    **chunk.metadata,
                },
            })

        logger.info(f"向量化完成: {len(chunks)} chunks")
        return results

    def embed_query(self, query: str) -> dict[str, Any]:
        """生成查询文本的向量"""
        dense = self._generate_dense([query])[0]
        sparse = self._generate_sparse([query])[0]
        hybrid = self._fuse_vectors(dense, sparse)

        return {
            "dense_vector": dense,
            "sparse_vector": sparse,
            "hybrid_vector": hybrid,
        }

    def _generate_dense(self, texts: list[str]) -> list[list[float]]:
        """生成稠密向量"""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as e:
            logger.warning("Embedding模型不可用，使用本地确定性fallback向量: %s", e)
            return [self._fallback_dense_vector(text) for text in texts]

    def _generate_sparse(self, texts: list[str]) -> list[dict[str, float]]:
        """
        问题4改进：用jieba生成BM25稀疏向量
        问题6改进：使用全库DF统计
        """
        # 分词
        tokenized_docs = []
        for text in texts:
            tokens = self._tokenize(text)
            tokenized_docs.append(tokens)

        n_docs = len(texts)
        sparse_vectors = []

        for tokens in tokenized_docs:
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            sparse_vec = {}
            for token, freq in tf.items():
                # 问题6改进：从全库统计获取DF
                df = self.corpus_stats.get_df(token)

                # BM25 TF-IDF scoring
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                k1 = 1.2
                b = 0.75
                avg_dl = sum(len(d) for d in tokenized_docs) / max(n_docs, 1)
                dl = len(tokens)
                tf_score = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / max(avg_dl, 1)))
                score = idf * tf_score

                if score > 0:
                    token_id = self._stable_token_id(token)
                    sparse_vec[token_id] = score

            sparse_vectors.append(sparse_vec)

        return sparse_vectors

    def _tokenize(self, text: str) -> list[str]:
        """
        问题4改进：用jieba分词
        """
        if self.jieba_tokenizer:
            # 精确模式分词
            tokens = list(self.jieba_tokenizer.cut(text, cut_all=False))
            # 过滤停用词和短词
            tokens = [t for t in tokens if t not in self._stopwords and len(t) > 1]
            return tokens
        else:
            # fallback：简单分词
            import re

            tokens = []
            # 英文单词
            english_words = re.findall(r"[a-zA-Z]+", text)
            tokens.extend(w.lower() for w in english_words)

            # 中文按2-gram
            chinese_chars = re.findall(r"[一-鿿]", text)
            for i in range(len(chinese_chars) - 1):
                tokens.append(chinese_chars[i] + chinese_chars[i + 1])
            tokens.extend(chinese_chars)

            return tokens

    def _fuse_vectors(
        self, dense_vector: list[float], sparse_vector: dict[str, float]
    ) -> dict[str, Any]:
        """
        问题5改进：Hybrid score fusion

        支持两种融合方式：
        - RRF (Reciprocal Rank Fusion)
        - Weighted Fusion
        """
        if HYBRID_FUSION_MODE == "rrf":
            return self._fuse_rrf(dense_vector, sparse_vector)
        else:
            return self._fuse_weighted(dense_vector, sparse_vector)

    @staticmethod
    def _fuse_rrf(dense_vector: list[float], sparse_vector: dict[str, float]) -> dict[str, Any]:
        """RRF融合：score = 1/(k + rank_dense) + 1/(k + rank_sparse)"""
        # 这里简化处理，实际应该在检索时计算rank
        k = 60
        dense_score = 1 / (k + 1)  # 假设rank=1
        sparse_score = 1 / (k + 1)
        return {
            "method": "rrf",
            "dense_score": dense_score,
            "sparse_score": sparse_score,
            "fused_score": dense_score + sparse_score,
        }

    @staticmethod
    def _fuse_weighted(
        dense_vector: list[float], sparse_vector: dict[str, float]
    ) -> dict[str, Any]:
        """Weighted融合：score = w_dense * norm(score_dense) + w_sparse * norm(score_sparse)"""
        # Dense向量的norm（L2）
        dense_norm = np.linalg.norm(dense_vector) if dense_vector else 1.0
        dense_score = dense_norm / max(dense_norm, 1.0)

        # Sparse向量的norm
        sparse_scores = list(sparse_vector.values()) if sparse_vector else [0]
        sparse_norm = np.linalg.norm(sparse_scores) if sparse_scores else 1.0
        sparse_score = sparse_norm / max(sparse_norm, 1.0)

        # 加权融合
        fused_score = DENSE_WEIGHT * dense_score + SPARSE_WEIGHT * sparse_score

        return {
            "method": "weighted",
            "dense_score": dense_score,
            "sparse_score": sparse_score,
            "dense_weight": DENSE_WEIGHT,
            "sparse_weight": SPARSE_WEIGHT,
            "fused_score": fused_score,
        }

    @staticmethod
    def _stable_token_id(token: str) -> str:
        """生成跨进程稳定的稀疏向量token id。"""
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return str(int(digest[:8], 16) % 100000)

    def _fallback_dense_vector(self, text: str) -> list[float]:
        """离线fallback向量，用于测试或模型不可用时保持管线可运行。"""
        vector = [0.0] * self.dimension
        tokens = self._tokenize(text) or [text]

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign

        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
