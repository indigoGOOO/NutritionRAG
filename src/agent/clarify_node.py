"""追问节点 - Human-in-the-Loop意图澄清

当Planner判断查询意图不明确时触发：
1. 生成澄清问题
2. 中断执行等待用户回答（interrupt）
3. 将用户的回答写回state.clarification_answer
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt

from src.agent.state import AgentState
from src.indexing.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

CLARIFY_PROMPT = """用户的问题缺少必要信息，无法准确回答。

用户查询：{query}

请生成一个简洁的追问，引导用户补充关键信息。
例如：'您是想了解番茄的营养成分，还是它的饮食禁忌？'
'您指的是哪种食材？请提供具体名称。'
'您想针对哪类人群（如儿童、孕妇、老年人）？'

只输出追问本身，不要其他内容。"""


def clarify_node(state: AgentState, llm: BaseLLMClient) -> dict:
    """追问节点：Human-in-the-Loop意图澄清"""
    query = state["query"]
    logger.info(f"[Clarify] 生成追问: {query[:50]}...")

    # 生成澄清问题
    try:
        prompt = CLARIFY_PROMPT.format(query=query)
        question = llm.generate(prompt=prompt)
        question = question.strip().split("\n")[0][:100]
    except Exception as e:
        logger.warning(f"[Clarify] 生成失败，使用默认: {e}")
        question = "您能更具体地描述一下想了解什么吗？"

    logger.info(f"[Clarify] 追问: {question}")

    # interrupt：暂停图执行，等待用户输入
    user_response = interrupt({
        "type": "clarification",
        "question": question,
    })

    clarification = str(user_response).strip()
    logger.info(f"[Clarify] 用户回应: {clarification}")

    return {
        "clarification_needed": False,
        "clarification_question": question,
        "clarification_answer": clarification,
    }
