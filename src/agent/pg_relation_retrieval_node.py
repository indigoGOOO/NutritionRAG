"""PG relation retrieval node.

Replaces the old Neo4j GraphRAG path with structured relation retrieval from
PostgreSQL triples and kv_pairs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.state import AgentState
from src.storage.pg_client import PostgreSQLClient

logger = logging.getLogger(__name__)

RELATION_LABELS: dict[str, str] = {
    "contraindicated_for": "禁忌",
    "suitable_for": "适宜人群",
    "pairs_with": "相宜搭配",
    "conflicts_with": "相克",
    "can_replace": "替代",
    "recommends": "推荐食材",
    "contains": "包含",
    "belongs_to": "类别",
    "source_of": "来源",
    "cooked_by": "烹饪方法",
    "related_to": "关联",
}

HIGH_PRIORITY_RELATIONS = {"contraindicated_for", "conflicts_with", "pairs_with"}
EXPAND_PREDICATES = {
    "contraindicated_for",
    "conflicts_with",
    "pairs_with",
    "can_replace",
    "recommends",
    "suitable_for",
}


def pg_relation_retrieval_node(state: AgentState, pg: PostgreSQLClient) -> dict:
    """Query relation triples, entity KV attributes, and one-hop neighbors."""
    route_context = state.get("route_context", {}).get("relation", {})
    entities = route_context.get("entities") or state.get("entities", [])
    if not entities:
        logger.info("[PRelation] no entities to retrieve")
        return _route_result(state, [], "empty")

    entity_names = [e.get("name", "") for e in entities if e.get("name")]
    if not entity_names:
        return _route_result(state, [], "empty")

    logger.info("[PRelation] entities=%s", entity_names)

    try:
        evidence = _retrieve_relations(entity_names, pg)
        logger.info("[PRelation] evidence=%s", len(evidence))
        return _route_result(state, evidence, "success" if evidence else "empty")
    except Exception as exc:
        logger.error("[PRelation] failed: %s", exc)
        return _route_result(state, [], "error", str(exc))


def _retrieve_relations(entity_names: list[str], pg: PostgreSQLClient) -> list[dict[str, Any]]:
    all_evidence: list[dict[str, Any]] = []
    seen_relations: set[tuple[str, str, str]] = set()

    raw_relations = pg.query_triples_by_entities(entity_names, limit=80)
    for rel in raw_relations:
        key = (rel["subject"], rel["predicate"], rel["object"])
        if key in seen_relations:
            continue
        seen_relations.add(key)
        all_evidence.append(_format_relation(rel))

    all_evidence.extend(_entity_info_evidence(entity_names, pg))

    discovered_entities = _collect_expand_entities(all_evidence, entity_names)
    if discovered_entities:
        expand_list = list(discovered_entities)[:5]
        logger.info("[PRelation] one-hop expand=%s", expand_list)
        expanded = pg.query_triples_by_entities(expand_list, limit=30)
        for rel in expanded:
            key = (rel["subject"], rel["predicate"], rel["object"])
            if key in seen_relations:
                continue
            seen_relations.add(key)
            all_evidence.append({**_format_relation(rel), "hop": 1})
        all_evidence.extend(_entity_info_evidence(expand_list, pg, hop=1))

    return all_evidence


def _entity_info_evidence(entity_names: list[str], pg: PostgreSQLClient, hop: int | None = None) -> list[dict]:
    kv_rows = pg.query_kv_by_entity_batch(entity_names)
    kv_map: dict[str, list[dict]] = {}
    for kv in kv_rows:
        kv_map.setdefault(kv["entity_id"], []).append(kv)

    evidence = []
    for entity_name in entity_names:
        entity_kvs = kv_map.get(entity_name, [])
        attrs = []
        entity_type = "unknown"
        for kv in entity_kvs:
            val = kv.get("value", {})
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    val = {}
            if kv.get("attribute") == "_entity_type" and isinstance(val, dict):
                entity_type = val.get("type", "unknown")
            if isinstance(val, dict) and "attributes" in val:
                raw_attrs = val["attributes"]
                if isinstance(raw_attrs, list):
                    attrs.extend(raw_attrs)
                elif isinstance(raw_attrs, dict):
                    attrs.append(raw_attrs)
        if attrs:
            item = {
                "subject": entity_name,
                "entity_type": entity_type,
                "attributes": attrs,
                "evidence_type": "entity_info",
                "source": "pg",
            }
            if hop is not None:
                item["hop"] = hop
            evidence.append(item)
    return evidence


def _collect_expand_entities(evidence: list[dict], original_entities: list[str]) -> set[str]:
    discovered = set()
    original = set(original_entities)
    for item in evidence:
        predicate = item.get("predicate")
        if predicate not in EXPAND_PREDICATES:
            continue
        for candidate in (item.get("subject", ""), item.get("object", "")):
            if candidate and candidate not in original:
                discovered.add(candidate)
    return discovered


def _format_relation(rel: dict) -> dict:
    metadata = rel.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    predicate = rel["predicate"]
    return {
        "subject": rel["subject"],
        "predicate": predicate,
        "object": rel["object"],
        "subject_type": metadata.get("subject_type", "unknown"),
        "object_type": metadata.get("object_type", "unknown"),
        "label": RELATION_LABELS.get(predicate, predicate),
        "priority": "high" if predicate in HIGH_PRIORITY_RELATIONS else "normal",
        "evidence_type": "relation",
        "source": "pg",
        "confidence": rel.get("confidence", 0.8),
        "metadata": metadata,
    }


def _route_result(state: AgentState, evidence: list[dict], status: str, error: str = "") -> dict:
    return {
        "evidence": _with_evidence(state, evidence),
        "route_status": _with_route_status(state, "relation", status, len(evidence), error),
    }


def _with_evidence(state: AgentState, evidence: list[dict]) -> dict:
    merged = {**state.get("evidence", {})}
    merged["relation"] = evidence
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
