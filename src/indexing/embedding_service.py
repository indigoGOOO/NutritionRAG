"""向量化服务

生成dense向量（语义检索）和sparse向量（BM25关键词检索），
用于写入Milvus的hybrid索引。
"""

from __future__ import annotations

import logging
import math
import hashlib
from collections import Counter
from typing import Any

from config.settings import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from src.indexing.models import ContentChunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """向量化服务 - 生成dense和sparse向量"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, dimension: int = EMBEDDING_DIMENSION):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None

    @property
    def model(self):
        """延迟加载embedding模型"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding模型加载完成: {self.model_name}")
        return self._model

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
            results.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i],
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
        """生成查询文本的向量（用于检索时）"""
        dense = self._generate_dense([query])[0]
        sparse = self._generate_sparse([query])[0]
        return {"dense_vector": dense, "sparse_vector": sparse}

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
        """生成BM25稀疏向量"""
        # 构建文档频率
        doc_freq: Counter[str] = Counter()
        tokenized_docs = []

        for text in texts:
            tokens = self._tokenize(text)
            tokenized_docs.append(tokens)
            doc_freq.update(set(tokens))

        n_docs = len(texts)
        sparse_vectors = []

        for tokens in tokenized_docs:
            tf = Counter(tokens)
            sparse_vec = {}

            for token, freq in tf.items():
                # BM25 TF-IDF scoring
                df = doc_freq.get(token, 0)
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

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词（中文按字，英文按词）"""
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
