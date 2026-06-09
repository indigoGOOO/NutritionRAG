"""Answer node: generate the final response from retrieved evidence."""

from __future__ import annotations

import logging
import re

from src.agent.state import AgentState
from src.indexing.llm_client import BaseLLMClient
from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个专业的智能膳食营养助手。
回答要求：
1. 优先基于检索证据回答，不确定时明确说明。
2. 引用检索证据时使用 [来源n] 标注。
3. 用户画像和最近对话只用于个性化与上下文承接，不作为事实来源引用。
4. 中文回答，结构清晰、具体、实用。
"""

ANSWER_PROMPT = """基于以下检索到的信息回答用户问题。

## 检索证据
{evidence_text}

{memory_text}

## 用户问题
{query}

## 回答要求
1. 优先使用检索到的证据回答，并标注 [来源n]。
2. 综合多源信息，包括文档、图谱关系、数据库结果。
3. “可强参考的历史问答”可以作为主要参考，但必须优先核对当前检索证据；如果当前证据不足而使用它，必须明确写“根据历史问答参考”。
4. “仅供理解意图的历史问答”只能帮助理解用户意图，不可作为事实依据，不允许作为来源引用。
5. 医疗、疾病、个性化饮食问题不要直接照搬历史回答。
6. “用户画像”用于个性化建议，但不要直接复述给用户。
7. “最近对话历史”用于保持对话连贯性，避免重复已给过的信息。
8. 信息不足时明确告知。
9. 回答要具体、准确、实用。

回答："""

FALLBACK_PROMPT = """{personal_context}

## 用户问题
{query}

当前没有可用检索证据。请根据你的营养学知识回答：
1. 如果使用了用户画像或最近对话，只用于个性化和承接上下文，不要把它当作事实来源引用。
2. 不确定的信息要明确说明，不要编造。
3. 医疗、疾病、个性化饮食问题要谨慎，必要时建议咨询专业人士。
4. 回答要具体、准确、实用。

回答："""


def answer_node(
    state: AgentState,
    llm: BaseLLMClient,
    memory_manager: MemoryManager | None = None,
) -> dict:
    """Generate the final answer."""
    query = state["query"]
    session_id = state.get("session_id", "default")
    evidence = state.get("reranked_evidence", [])
    personalization_policy = state.get("personalization_policy", {})
    logger.info(f"[Answer] generating from {len(evidence)} evidence items")

    if _missing_required_private_content(personalization_policy):
        return {
            "answer": _private_content_missing_answer(personalization_policy),
            "citations": [],
        }

    memory_text = ""
    if memory_manager:
        try:
            memory_text = memory_manager.format_memory_context(
                query=query,
                session_id=session_id,
                intent=state.get("intent", ""),
            )
        except Exception as e:
            logger.debug(f"[Answer] failed to get memory context: {e}")

    if not evidence:
        logger.info("[Answer] no retrieved evidence, using fallback with personal memory")
        personal_context = ""
        if memory_manager:
            try:
                personal_context = memory_manager.format_personal_memory_context(
                    session_id=session_id,
                )
            except Exception as e:
                logger.debug(f"[Answer] failed to get personal memory context: {e}")

        prompt = FALLBACK_PROMPT.format(
            personal_context=personal_context,
            query=query,
        )
        fallback_answer = llm.generate(prompt=prompt, system=SYSTEM_PROMPT)
        return {"answer": fallback_answer, "citations": []}

    evidence_text = _format_evidence(evidence)
    if personalization_policy.get("mode") == "strong":
        evidence_text = (
            "注意：用户明确要求基于其已保存的个人数据回答。"
            "带有 user_content_context 的证据是主依据；公共证据只能用于解释和安全补充，"
            "不能压过用户已保存数据。\n\n"
            + evidence_text
        )

    try:
        prompt = ANSWER_PROMPT.format(
            evidence_text=evidence_text,
            memory_text=memory_text,
            query=query,
        )
        answer = llm.generate(prompt=prompt, system=SYSTEM_PROMPT)
    except Exception as e:
        logger.error(f"[Answer] generation failed: {e}")
        answer = f"抱歉，回答生成遇到问题：{e}"

    citations = _extract_citations(answer, evidence)

    if memory_manager and citations:
        try:
            entities = [e.get("name", "") for e in state.get("entities", [])]
            memory_manager.store_qa(
                question=query,
                answer=answer,
                entities=entities,
                tags=[state.get("intent", "general")],
                citations=citations,
                evidence_count=len(evidence),
            )
            logger.info("[Answer] stored answer into knowledge memory")
        except Exception as e:
            logger.warning(f"[Answer] failed to store knowledge memory: {e}")

    return {"answer": answer, "citations": citations}


def _missing_required_private_content(policy: dict) -> bool:
    return (
        bool(policy.get("private_content_required"))
        and policy.get("status") == "private_content_missing"
    )


def _private_content_missing_answer(policy: dict) -> str:
    requested = policy.get("requested_content_types") or []
    type_names = {
        "recipe": "菜谱",
        "meal_plan": "饮食计划",
        "workout_plan": "健身训练计划",
        "food_log": "饮食记录",
        "body_metrics": "身体指标记录",
        "lab_report": "化验单/体检报告",
    }
    if requested:
        target = "、".join(type_names.get(item, item) for item in requested)
        return f"我没有找到你已保存的{target}，所以不能假装基于这些个人数据来回答。你可以先把相关内容发给我并说明“保存”，我保存后再基于它分析。"
    return "我没有找到你明确提到的已保存个人数据，所以不能假装基于历史记录来回答。你可以先把相关内容发给我并说明“保存”，我保存后再基于它分析。"


def _format_evidence(evidence: list[dict]) -> str:
    """Format evidence as numbered source blocks."""
    parts = []
    for i, item in enumerate(evidence, 1):
        source_type = item.get("source_type", "unknown")
        lines = [f"[来源{i}] (类型: {source_type})"]

        content = item.get("content")
        if content:
            lines.append(content[:600])
        else:
            subject = item.get("subject", "")
            predicate = item.get("predicate", "")
            obj = item.get("object", "")
            if subject and predicate:
                lines.append(f"{subject} --({predicate})--> {obj}")
            elif item.get("entity"):
                attr = item.get("attribute", "")
                val = item.get("value", "")
                lines.append(f"{item['entity']}: {attr} = {val}")

        score = item.get("rerank_score", item.get("score"))
        if score is not None:
            lines.append(f"(相关度: {score:.3f})")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _extract_citations(answer: str, evidence: list[dict]) -> list[dict]:
    """Extract cited evidence ids from answer text."""
    seen_ids = set()
    citations = []
    for match in re.finditer(r"\[(?:来源|证据)(\d+)\]", answer):
        idx = int(match.group(1))
        if idx in seen_ids or idx < 1 or idx > len(evidence):
            continue
        seen_ids.add(idx)
        item = evidence[idx - 1]
        citations.append({
            "source_id": idx,
            "source_type": item.get("source_type", "unknown"),
            "content": item.get("content", "")[:200]
            or f"{item.get('subject', '')} {item.get('predicate', '')} {item.get('object', '')}",
            "score": item.get("rerank_score", item.get("score")),
        })
    return citations
