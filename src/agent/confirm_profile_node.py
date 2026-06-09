"""画像确认节点 - 回答生成后，让用户确认是否保存画像信息

当 planner 标记了 has_profile_signal=true 时触发：
1. 用 LLM 从用户查询中提取结构化画像信息
2. interrupt 询问用户是否保存
3. 确认后写入 ProfileMemory
"""

from __future__ import annotations

import json
import logging

from langgraph.types import interrupt

from src.agent.state import AgentState
from src.indexing.llm_client import BaseLLMClient
from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """从用户的问题中提取与个人饮食相关的画像信息。

用户问题：{query}

提取规则：
- 只提取用户明确表达的个人信息，不要推测
- 如果是通用知识查询（如"糖尿病患者吃什么好"），不要提取
- "我XX"才可能是画像，"糖尿病的人"不是

可提取的类型和判断标准：
1. allergy: 明确说"我对XX过敏""我不能吃XX"等
2. dietary_restriction: "我是素食者""我在控糖"等
3. health_goal: "我想减肥""我的目标是增肌"等
4. favorite_ingredient: "我喜欢吃XX""我偏爱XX"等
5. disliked_ingredient: "我不喜欢XX""我讨厌XX"等
6. temporary_preference: "今天想吃清淡的""这顿不要辣"等临时需求

输出 JSON 格式（没有候选时返回空列表）：
{{
  "candidates": [
    {{
      "type": "allergy",
      "value": "花生",
      "evidence": "我对花生过敏"
    }}
  ]
}}"""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "allergy", "dietary_restriction", "health_goal",
                            "favorite_ingredient", "disliked_ingredient",
                            "temporary_preference",
                        ],
                    },
                    "value": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["type", "value", "evidence"],
            },
        },
    },
    "required": ["candidates"],
}


def confirm_profile_node(
    state: AgentState,
    llm: BaseLLMClient,
    memory_manager: MemoryManager | None = None,
) -> dict:
    """画像确认节点：提取画像 → 用户确认 → 写入"""
    if not state.get("has_profile_signal") or not memory_manager:
        return {"has_profile_signal": False}

    query = state["query"]
    logger.info("[ConfirmProfile] 检测到画像信号，开始提取")

    # Step 1: 提取画像候选
    candidates = _extract_candidates(query, llm)
    if not candidates:
        logger.info("[ConfirmProfile] 未提取到有效画像信息")
        return {"has_profile_signal": False}

    # Step 2: 过滤临时偏好（不需要用户确认，直接丢弃）
    permanent = [c for c in candidates if c["type"] != "temporary_preference"]
    if not permanent:
        logger.info("[ConfirmProfile] 仅临时偏好，跳过确认")
        return {"has_profile_signal": False}

    # Step 3: 构造确认问题并 interrupt
    question = _build_confirm_question(permanent)
    logger.info(f"[ConfirmProfile] 请求确认: {question}")

    user_response = interrupt({
        "type": "profile_confirmation",
        "question": question,
        "candidates": permanent,
    })

    reply = str(user_response).strip()
    logger.info(f"[ConfirmProfile] 用户回应: {reply}")

    # Step 4: 判断用户是否确认
    if _is_affirmative(reply):
        _write_candidates(memory_manager, permanent)
        logger.info("[ConfirmProfile] 用户已确认，写入画像")
    else:
        logger.info("[ConfirmProfile] 用户未确认，丢弃画像候选")

    return {"has_profile_signal": False}


def _extract_candidates(query: str, llm: BaseLLMClient) -> list[dict]:
    """用 LLM 从查询中提取画像信息"""
    try:
        prompt = EXTRACT_PROMPT.format(query=query)
        result = llm.extract_structured(prompt=prompt, schema=EXTRACT_SCHEMA)
        candidates = result.get("candidates", [])
        return [
            c for c in candidates
            if c.get("type") and c.get("value")
        ]
    except Exception as e:
        logger.warning(f"[ConfirmProfile] 提取画像失败: {e}")
        return []


def _build_confirm_question(candidates: list[dict]) -> str:
    """构造确认问题"""
    lines = []
    for c in candidates:
        type_labels = {
            "allergy": "过敏源",
            "dietary_restriction": "饮食限制",
            "health_goal": "健康目标",
            "favorite_ingredient": "偏好食材",
            "disliked_ingredient": "不喜欢的食材",
        }
        label = type_labels.get(c["type"], c["type"])
        evidence = c.get("evidence", "")
        if evidence:
            lines.append(f"- {label}：{c['value']}（根据：{evidence}）")
        else:
            lines.append(f"- {label}：{c['value']}")
    return (
        f"我注意到以下可能和您个人相关的信息：\n"
        f"{chr(10).join(lines)}\n\n"
        f"是否将这些信息保存到您的个人档案？(是/否)"
    )


def _is_affirmative(reply: str) -> bool:
    """判断用户是否确认"""
    positive = {"是", "对", "嗯", "可以", "好", "保存", "确认", "是的", "对的", "好的", "可以呢", "嗯嗯"}
    return reply.strip().lower() in positive or reply.strip() in positive or any(
        reply.strip().startswith(w) for w in positive
    )


def _write_candidates(memory_manager: MemoryManager, candidates: list[dict]):
    """写入确认后的画像信息"""
    for c in candidates:
        try:
            ctype = c["type"]
            value = c["value"]
            if ctype == "allergy":
                memory_manager.profile.add_allergy(memory_manager.user_id, value)
            elif ctype == "dietary_restriction":
                memory_manager.profile.add_dietary_restriction(memory_manager.user_id, value)
            elif ctype == "health_goal":
                memory_manager.profile.add_health_goal(memory_manager.user_id, value)
            elif ctype == "favorite_ingredient":
                memory_manager.profile.add_favorite_ingredient(memory_manager.user_id, value)
            elif ctype == "disliked_ingredient":
                memory_manager.profile.add_disliked_ingredient(memory_manager.user_id, value)
        except Exception as e:
            logger.warning(f"[ConfirmProfile] 写入失败 {c.get('type')}/{c.get('value')}: {e}")