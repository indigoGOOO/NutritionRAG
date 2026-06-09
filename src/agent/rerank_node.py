"""Rerank node.

Merge semantic, relation, and text2sql evidence, remove duplicates, and
rerank with optional Cross-Encoder. User-saved private content is isolated by
user_id and can be promoted with weak or strong personalization policies.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_EVIDENCE_PER_SOURCE = {
    "semantic": 10,
    "relation": 15,
    "text2sql": 10,
}

SOURCE_WEIGHT = {
    "text2sql": 1.15,
    "relation": 1.05,
    "semantic": 0.95,
}

USER_CONTENT_SOURCE_PREFIX = "user_content:"

USER_CONTENT_WEIGHT_BY_INTENT = {
    "diet_advice": {
        "meal_plan": 1.25,
        "food_log": 1.2,
        "body_metrics": 1.2,
        "workout_plan": 1.15,
        "lab_report": 1.25,
    },
    "recipe_recommend": {
        "recipe": 1.2,
        "meal_plan": 1.15,
        "food_log": 1.1,
    },
    "recipe_instruction": {
        "recipe": 1.2,
    },
    "meal_record_analysis": {
        "food_log": 1.3,
        "meal_plan": 1.15,
        "body_metrics": 1.1,
    },
    "disease_diet": {
        "lab_report": 1.3,
        "body_metrics": 1.2,
        "meal_plan": 1.1,
        "food_log": 1.1,
    },
    "safety_check": {
        "lab_report": 1.3,
        "body_metrics": 1.2,
    },
    "profile_management": {
        "body_metrics": 1.15,
        "lab_report": 1.15,
    },
}

PERSONALIZED_QUERY_TERMS = (
    "我",
    "我的",
    "本人",
    "适合我",
    "按我的",
    "根据我的",
    "my",
)

STRONG_PERSONAL_DATA_TERMS = (
    "我保存",
    "我存的",
    "我记录",
    "我记的",
    "我的饮食计划",
    "我的菜单",
    "我的菜谱",
    "我的食谱",
    "我的饮食记录",
    "我的体检",
    "我的体检报告",
    "我的化验",
    "我的化验单",
    "我的训练计划",
    "昨天记录",
    "上次记录",
    "之前保存",
    "保存的饮食计划",
    "保存的菜单",
    "保存的菜谱",
    "保存的体检报告",
    "保存的化验单",
    "saved",
)

PRIVATE_CONTENT_TYPE_KEYWORDS = {
    "recipe": ("菜谱", "食谱", "菜单", "做法"),
    "meal_plan": ("饮食计划", "餐单", "膳食计划", "减脂计划"),
    "workout_plan": ("训练计划", "健身计划", "运动计划"),
    "food_log": ("饮食记录", "吃了什么", "昨天吃", "今天吃", "记录的饮食"),
    "body_metrics": ("体重", "体脂", "身高", "腰围", "身体指标", "体测"),
    "lab_report": ("化验单", "体检报告", "检查报告", "血糖", "血脂", "尿酸"),
}


def rerank_node(state: AgentState, cross_encoder: Any = None) -> dict:
    """Merge and rerank evidence from all retrieval routes."""
    evidence = state.get("evidence", {})
    query = state["query"]
    personalization_policy = _build_personalization_policy(state)

    logger.info(
        "[ReRank] merge evidence: semantic=%s relation=%s text2sql=%s policy=%s",
        len(evidence.get("semantic", [])),
        len(evidence.get("relation", [])),
        len(evidence.get("text2sql", [])),
        personalization_policy.get("mode"),
    )

    all_evidence: list[dict[str, Any]] = []
    for source_type in ("semantic", "relation", "text2sql"):
        items = evidence.get(source_type, [])[: MAX_EVIDENCE_PER_SOURCE.get(source_type, 10)]
        for item in items:
            item["source_type"] = source_type
            all_evidence.append(item)

    all_evidence = _filter_and_annotate_user_content(
        all_evidence,
        state,
        personalization_policy,
    )

    if not all_evidence:
        logger.info("[ReRank] no evidence to merge")
        return _rerank_result([], personalization_policy)

    unique_evidence: list[dict[str, Any]] = []
    for item in all_evidence:
        if not _is_duplicate(item, unique_evidence):
            unique_evidence.append(item)

    if cross_encoder is not None:
        try:
            texts = [_get_rerank_text(item) for item in unique_evidence]
            scores = cross_encoder.predict([(query, text) for text in texts])
            for i, item in enumerate(unique_evidence):
                raw_score = float(scores[i])
                item["rerank_score"] = _weighted_score(item, raw_score)
            unique_evidence.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        except Exception as exc:
            logger.warning("[ReRank] Cross-Encoder failed, fallback to score sort: %s", exc)
            _sort_by_score(unique_evidence)
    else:
        _sort_by_score(unique_evidence)

    if personalization_policy.get("private_content_required"):
        unique_evidence.sort(
            key=lambda item: (
                1 if item.get("user_content_context") else 0,
                item.get("rerank_score", 0),
            ),
            reverse=True,
        )

    logger.info("[ReRank] output evidence=%s", len(unique_evidence))
    return _rerank_result(unique_evidence, personalization_policy)


def _rerank_result(evidence: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    private_content_found = any(item.get("user_content_context") for item in evidence)
    return {
        "reranked_evidence": evidence,
        "personalization_policy": {
            **policy,
            "private_content_found": private_content_found,
            "status": _personalization_status(policy, private_content_found),
        },
    }


def _build_personalization_policy(state: AgentState) -> dict[str, Any]:
    query = str(state.get("query") or "")
    requested_types = _requested_private_content_types(query)
    strong = bool(requested_types) or any(
        term.lower() in query.lower() for term in STRONG_PERSONAL_DATA_TERMS
    )
    weak = any(term.lower() in query.lower() for term in PERSONALIZED_QUERY_TERMS)

    if strong:
        return {
            "mode": "strong",
            "private_content_required": True,
            "requested_content_types": requested_types,
            "reason": "explicit_user_saved_content_reference",
        }
    if weak:
        return {
            "mode": "weak",
            "private_content_required": False,
            "requested_content_types": [],
            "reason": "personalized_query_signal",
        }
    return {
        "mode": "none",
        "private_content_required": False,
        "requested_content_types": [],
        "reason": "",
    }


def _requested_private_content_types(query: str) -> list[str]:
    owner_terms = ("我的", "我保存", "我记录", "保存的", "记录的", "之前", "上次", "昨天", "今天")
    if not any(term in query for term in owner_terms):
        return []

    result = []
    for content_type, keywords in PRIVATE_CONTENT_TYPE_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            result.append(content_type)
    return result


def _personalization_status(policy: dict[str, Any], private_content_found: bool) -> str:
    if not policy.get("private_content_required"):
        return "not_required"
    return "private_content_found" if private_content_found else "private_content_missing"


def _filter_and_annotate_user_content(
    items: list[dict[str, Any]],
    state: AgentState,
    personalization_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter private content owned by other users and attach weights."""
    current_user_id = str(state.get("user_id") or "default")
    filtered: list[dict[str, Any]] = []

    for item in items:
        info = _extract_user_content_info(item)
        if info and info.get("user_id") and info["user_id"] != current_user_id:
            logger.info(
                "[ReRank] filter private user content: owner=%s current=%s",
                info["user_id"],
                current_user_id,
            )
            continue

        weight = _user_content_weight(info, state, personalization_policy)
        item["personalization_weight"] = weight
        if info:
            item["user_content_context"] = {
                "matched": True,
                "owner_user_id": info.get("user_id"),
                "content_type": info.get("content_type"),
                "source_doc_id": info.get("source_doc_id"),
                "weight": weight,
            }
        filtered.append(item)

    return filtered


def _extract_user_content_info(item: dict[str, Any]) -> dict[str, str] | None:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    properties = item.get("properties")
    if isinstance(properties, dict):
        nested_metadata = properties.get("metadata")
        if isinstance(nested_metadata, dict):
            metadata = {**nested_metadata, **metadata}

    source_doc_id = (
        item.get("source_doc_id")
        or metadata.get("source_doc_id")
        or item.get("source_doc")
        or ""
    )
    if isinstance(source_doc_id, str) and source_doc_id.startswith(USER_CONTENT_SOURCE_PREFIX):
        parts = source_doc_id.split(":")
        if len(parts) >= 4:
            return {
                "user_id": parts[1],
                "content_type": parts[2],
                "source_doc_id": source_doc_id,
            }

    if metadata.get("source_type") == "user_saved_content" or metadata.get("user_content_type"):
        return {
            "user_id": str(metadata.get("user_id") or ""),
            "content_type": str(metadata.get("user_content_type") or ""),
            "source_doc_id": str(source_doc_id or metadata.get("source_doc_id") or ""),
        }

    return None


def _user_content_weight(
    info: dict[str, str] | None,
    state: AgentState,
    personalization_policy: dict[str, Any],
) -> float:
    if not info:
        return 1.0

    content_type = info.get("content_type") or ""
    requested_types = personalization_policy.get("requested_content_types") or []
    if personalization_policy.get("private_content_required"):
        if not requested_types or content_type in requested_types:
            return 10.0
        return 2.0

    intent = state.get("intent") or ""
    intent_weights = USER_CONTENT_WEIGHT_BY_INTENT.get(intent, {})
    if content_type in intent_weights:
        return intent_weights[content_type]

    query = str(state.get("query") or "").lower()
    if any(term.lower() in query for term in PERSONALIZED_QUERY_TERMS):
        return 1.1
    return 1.0


def _weighted_score(item: dict[str, Any], base_score: float) -> float:
    source_weight = SOURCE_WEIGHT.get(item.get("source_type", ""), 1.0)
    personalization_weight = item.get("personalization_weight", 1.0) or 1.0
    return base_score * source_weight * personalization_weight


def _is_duplicate(
    candidate: dict,
    existing: list[dict],
    ngram_n: int = 2,
    threshold: float = 0.85,
) -> bool:
    """Return True when candidate duplicates existing evidence."""
    candidate_pred = candidate.get("predicate")
    if candidate_pred:
        key = f"{candidate.get('subject', '')}|{candidate_pred}|{candidate.get('object', '')}"
        for item in existing:
            if item.get("predicate"):
                other = f"{item.get('subject', '')}|{item.get('predicate', '')}|{item.get('object', '')}"
                if key == other:
                    return True
        return False

    content = candidate.get("content", "")
    if not content:
        return False
    cand_ngrams = _ngram_set(content, ngram_n)
    if not cand_ngrams:
        return False
    for item in existing:
        other = item.get("content", "")
        if not other:
            continue
        other_ngrams = _ngram_set(other, ngram_n)
        if not other_ngrams:
            continue
        if _jaccard(cand_ngrams, other_ngrams) >= threshold:
            return True
    return False


def _ngram_set(text: str, n: int = 2) -> set[str]:
    cleaned = text.strip()
    return {cleaned[i:i + n] for i in range(len(cleaned) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _get_rerank_text(item: dict) -> str:
    content = item.get("content", "")
    if content:
        return content

    subject = item.get("subject", "")
    predicate = item.get("predicate", "")
    obj = item.get("object", "")
    if subject and predicate:
        return _triple_to_sentence(subject, predicate, obj)
    return str(item.get("value", ""))


_TRIPLE_TEMPLATES: dict[str, tuple[str, str]] = {
    "相宜": ("{sub}和{obj}相宜", "{sub}与{obj}搭配有益"),
    "相克": ("{sub}和{obj}相克", "{sub}不宜与{obj}同食"),
    "禁忌": ("{sub}对{obj}有禁忌", "{sub}不适合{obj}"),
    "适宜": ("{sub}适宜{obj}", "{sub}适合{obj}"),
    "不宜": ("{sub}不宜{obj}", "{sub}不适合{obj}"),
    "含有": ("{sub}含有{obj}", "{sub}中包含{obj}"),
    "属于": ("{sub}属于{obj}", ""),
    "别名": ("{sub}的别名是{obj}", "{sub}也称{obj}"),
    "类别": ("{sub}属于{obj}类别", ""),
    "富含": ("{sub}富含{obj}", "{sub}含有丰富的{obj}"),
    "含量": ("{sub}的{obj}", ""),
    "作用": ("{sub}有{obj}的作用", "{sub}可以{obj}"),
    "推荐摄入": ("{sub}推荐摄入{obj}", ""),
}


def _triple_to_sentence(subject: str, predicate: str, obj: str) -> str:
    templates = _TRIPLE_TEMPLATES.get(predicate)
    if templates:
        for template in templates:
            if template:
                return template.format(sub=subject, obj=obj)
    if predicate.endswith("的"):
        return f"{subject}{predicate}{obj}"
    return f"{subject}{predicate}{obj}"


def _sort_by_score(items: list[dict]):
    """Sort by rerank_score/score after route and personalization weights."""
    for item in items:
        base_score = item.get("rerank_score", item.get("score", 0)) or 0
        item["rerank_score"] = _weighted_score(item, float(base_score))
    items.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
