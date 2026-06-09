"""Semantic retrieval node: Milvus hybrid search."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from sentence_transformers import SentenceTransformer

from config.settings import DENSE_WEIGHT, EMBEDDING_MODEL, SPARSE_WEIGHT
from src.agent.state import AgentState
from src.storage.milvus_client import MilvusClient

logger = logging.getLogger(__name__)

DISCOVERABLE_ENTITIES: dict[str, str] = {
    "番茄": "ingredient",
    "西红柿": "ingredient",
    "鸡蛋": "ingredient",
    "牛奶": "ingredient",
    "花生": "ingredient",
    "虾": "ingredient",
    "鱼": "ingredient",
    "鸡肉": "ingredient",
    "牛肉": "ingredient",
    "豆腐": "ingredient",
    "菠菜": "ingredient",
    "香蕉": "ingredient",
    "苹果": "ingredient",
    "米饭": "ingredient",
    "燕麦": "ingredient",
    "蛋白质": "nutrient",
    "脂肪": "nutrient",
    "碳水": "nutrient",
    "碳水化合物": "nutrient",
    "热量": "nutrient",
    "钠": "nutrient",
    "糖": "nutrient",
    "嘌呤": "nutrient",
    "膳食纤维": "nutrient",
    "痛风": "symptom",
    "糖尿病": "symptom",
    "高血压": "symptom",
    "高血脂": "symptom",
}


def semantic_retrieval_node(state: AgentState, milvus: MilvusClient) -> dict:
    """Run Milvus dense+sparse hybrid search and discover extra entities."""
    route_context = state.get("route_context", {}).get("semantic", {})
    query = route_context.get("query") or state["query"]
    search_query = query
    if state.get("clarification_answer"):
        search_query = f"{query} {state['clarification_answer']}"

    logger.info("[SemanticRetrieval] hybrid search: %s...", search_query[:50])

    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        query_vec = embedder.encode(search_query, normalize_embeddings=True).tolist()
        query_terms = _build_query_terms(search_query)

        results = milvus.hybrid_search(
            query_vector=query_vec,
            query_terms=query_terms,
            top_k=15,
            dense_weight=DENSE_WEIGHT,
            sparse_weight=SPARSE_WEIGHT,
        )

        evidence = [
            {
                "content": r.get("content", ""),
                "chunk_id": r.get("chunk_id"),
                "chunk_type": r.get("chunk_type", ""),
                "doc_category": r.get("doc_category", ""),
                "source_doc_id": r.get("source_doc_id"),
                "score": r.get("final_score", r.get("score", 0)),
                "source": "milvus",
            }
            for r in results
        ]
        discovered = _discover_entities_from_evidence(evidence, state.get("entities", []))
        logger.info(
            "[SemanticRetrieval] results=%s discovered_entities=%s",
            len(evidence),
            [e["name"] for e in discovered],
        )
        return {
            "evidence": _with_evidence(state, "semantic", evidence),
            "semantic_discovered_entities": discovered,
            "route_status": _with_route_status(
                state,
                "semantic",
                "success" if evidence else "empty",
                len(evidence),
            ),
        }

    except Exception as exc:
        logger.error("[SemanticRetrieval] failed: %s", exc)
        return {
            "evidence": _with_evidence(state, "semantic", []),
            "semantic_discovered_entities": [],
            "route_status": _with_route_status(state, "semantic", "error", 0, str(exc)),
        }


def _build_query_terms(query: str) -> dict[str, float] | None:
    try:
        import jieba

        terms = jieba.lcut(query)
    except ImportError:
        terms = [c for c in query if c.strip()]

    if not terms:
        return None

    term_counts = Counter(terms)
    max_count = max(term_counts.values()) or 1
    return {term: count / max_count for term, count in term_counts.items() if term.strip()}


def _discover_entities_from_evidence(
    evidence: list[dict[str, Any]],
    existing_entities: list[dict[str, str]],
    top_n: int = 5,
) -> list[dict[str, str]]:
    existing = {e.get("name", "") for e in existing_entities}
    discovered = []
    seen = set(existing)
    text = "\n".join(str(item.get("content", "")) for item in evidence[:top_n])
    for name, entity_type in DISCOVERABLE_ENTITIES.items():
        if name in seen:
            continue
        if name in text:
            discovered.append({"name": name, "type": entity_type})
            seen.add(name)
    return discovered[:8]


def _with_evidence(state: AgentState, route: str, evidence: list[dict]) -> dict:
    merged = {**state.get("evidence", {})}
    merged[route] = evidence
    return merged


def _with_route_status(
    state: AgentState,
    route: str,
    status: str,
    count: int,
    error: str = "",
) -> dict:
    merged = {**state.get("route_status", {})}
    merged[route] = {"status": status, "count": count, "error": error}
    return merged
