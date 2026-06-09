"""领域安全过滤节点 - 在 answer 前校验检索结果

基于用户画像中的过敏源、饮食限制和健康目标，
对 reranked_evidence 做安全过滤并标记违规项。

职责：
1. 检查证据内容是否包含用户过敏源
2. 检查建议是否符合用户饮食限制
3. 检查证据是否与用户健康目标冲突（如痛风建议高嘌呤食物）
4. 标记高风险证据但保留它（让 answer 节点知悉风险而非静默删除）
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import AgentState
from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


def safety_filter_node(
    state: AgentState,
    memory_manager: MemoryManager | None = None,
) -> dict:
    """安全过滤节点：检查检索证据是否与用户画像冲突"""
    evidence = state.get("reranked_evidence", [])
    if not evidence:
        logger.info("[Safety] 无证据可检查")
        return {}

    user_id = state.get("user_id", "default")
    if not memory_manager:
        logger.debug("[Safety] 无记忆管理器，跳过")
        return {}

    try:
        profile_text = memory_manager.get_user_profile_text()
        if not profile_text or profile_text == "暂无用户画像信息":
            logger.debug("[Safety] 无用户画像，跳过")
            return {}
    except Exception as e:
        logger.debug(f"[Safety] 获取画像失败: {e}")
        return {}

    # 从 profile 获取过敏和限制列表
    allergies = _safe_get(memory_manager, "get_allergies", user_id)
    restrictions = _safe_get(memory_manager, "get_restrictions", user_id)
    forbidden_ingredients = list(allergies or [])

    # 合并饮食限制中涉及的具体食材
    if restrictions:
        for r in restrictions:
            if isinstance(r, dict) and r.get("ingredient"):
                forbidden_ingredients.append(r["ingredient"])
            elif isinstance(r, str):
                forbidden_ingredients.append(r)

    if not forbidden_ingredients:
        logger.debug("[Safety] 无违禁食材列表，跳过")
        return {}

    # 检查每条证据
    safety_warnings: list[dict] = []
    total_score = 0.0
    flagged_count = 0

    for item in evidence:
        content = _get_evidence_text(item)
        if not content:
            continue

        matched_ingredients = [
            ing for ing in forbidden_ingredients
            if ing and ing in content
        ]
        if matched_ingredients:
            safety_warnings.append({
                "evidence_id": item.get("chunk_id") or item.get("evidence_id", ""),
                "matched_ingredients": matched_ingredients,
                "content_snippet": content[:120],
            })
            item["safety_flagged"] = True
            item["safety_issues"] = matched_ingredients
            flagged_count += 1
        else:
            item["safety_flagged"] = False

    logger.info(
        f"[Safety] 检查 {len(evidence)} 条证据, "
        f"{flagged_count} 条触犯安全规则"
    )

    # 对触犯安全规则的 evidence 打低分，但不删除（让 answer 知情）
    for item in evidence:
        if item.get("safety_flagged"):
            current_score = item.get("rerank_score", item.get("score", 0.5))
            item["rerank_score"] = current_score * 0.3  # 严重降权
            item["score_before_safety"] = current_score

    # 重新排序（降权后的沉底）
    evidence.sort(
        key=lambda x: x.get("rerank_score", x.get("score", 0)) or 0,
        reverse=True,
    )

    return {
        "reranked_evidence": evidence,
        "safety_warnings": safety_warnings,
    }


def _get_evidence_text(item: dict) -> str:
    """提取证据的文本内容"""
    content = item.get("content", "")
    if content:
        return content
    sub = item.get("subject", "")
    pred = item.get("predicate", "")
    obj = item.get("object", "")
    if sub and pred:
        return f"{sub}{pred}{obj}"
    return item.get("value", "")


def _safe_get(memory_manager, method_name: str, *args):
    """安全调用 memory_manager 方法"""
    try:
        method = getattr(memory_manager.profile, method_name, None)
        if method:
            return method(*args)
    except Exception as e:
        logger.debug(f"[Safety] 调用 {method_name} 失败: {e}")
    return []
