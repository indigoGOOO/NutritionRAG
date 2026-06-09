"""Planner node: intent detection, route planning, and guardrail checks."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agent.state import AgentState
from src.indexing.llm_client import BaseLLMClient
from src.memory.memory_manager import MemoryManager
from src.user_content.classifier import UserContentClassifier

logger = logging.getLogger(__name__)

INTENTS = [
    "nutrition_fact",
    "nutrition_compare",
    "nutrition_filter",
    "ingredient_knowledge",
    "recipe_recommend",
    "recipe_instruction",
    "disease_diet",
    "profile_management",
    "meal_record_analysis",
    "food_image_analysis",
    "content_save",
    "safety_check",
    "diet_advice",
    "general",
]

ROUTES = ["semantic", "relation", "text2sql"]
LOW_CONFIDENCE_CLARIFY_THRESHOLD = 0.65

SYSTEM_PROMPT = """你是膳食营养 RAG Agent 的 Planner。你的任务是理解用户问题，输出意图、实体、检索路线、是否需要澄清，以及是否存在用户画像信号。

意图 taxonomy：
- nutrition_fact: 查询营养事实，例如“番茄有什么营养”
- nutrition_compare: 比较食材/营养，例如“鸡蛋和牛奶哪个蛋白质高”
- nutrition_filter: 按数值条件筛选，例如“蛋白质大于10g的食物”
- ingredient_knowledge: 食材知识、相宜相克、特点
- recipe_recommend: 推荐菜谱/餐食
- recipe_instruction: 询问做法、步骤、烹饪方法
- disease_diet: 疾病/症状相关饮食建议，例如痛风、糖尿病、高血压
- profile_management: 用户明确要求保存/修改画像，例如“帮我记住我花生过敏”
- meal_record_analysis: 分析用户已经吃了什么、饮食记录
- food_image_analysis: 分析食物图片/餐盘图片
- safety_check: 问某食材“我能不能吃/适不适合我”
- diet_advice: 个性化饮食建议，但不明显属于疾病饮食或安全检查
- general: 其它

路线说明：
- semantic: 检索文档中的通用知识、建议、菜谱、解释
- relation: 查询实体关系、禁忌、疾病-食材关系、相宜相克
- text2sql: 查询明确数值或结构化营养数据

注意：
- intent 是用户想做什么，route 是系统需要查什么，二者不要混淆。
- has_profile_signal 是附加信号，不替代 intent。
- 只要用户明确表达过敏、忌口、长期目标、稳定偏好，就标记 has_profile_signal=true。
- 临时偏好如“今天想清淡点”不算长期画像信号。"""

PLANNER_PROMPT = """分析用户查询并返回 JSON。

用户查询：{query}

Few-shot 示例：
Q: 番茄有什么营养？
{{
  "entities": [{{"name": "番茄", "type": "ingredient"}}],
  "intent": "nutrition_fact",
  "intent_confidence": 0.95,
  "intent_reason": "用户询问番茄的营养事实",
  "planned_routes": ["semantic"],
  "route_reason": {{"semantic": "需要检索营养知识文本"}},
  "clarification_needed": false,
  "clarification_question": "",
  "has_profile_signal": false
}}

Q: 蛋白质大于10g的食物有哪些？
{{
  "entities": [{{"name": "蛋白质", "type": "nutrient"}}],
  "intent": "nutrition_filter",
  "intent_confidence": 0.96,
  "intent_reason": "用户提出明确数值筛选条件",
  "planned_routes": ["text2sql"],
  "route_reason": {{"text2sql": "需要按营养数值条件查询数据库"}},
  "clarification_needed": false,
  "clarification_question": "",
  "has_profile_signal": false
}}

Q: 痛风可以吃虾吗？
{{
  "entities": [
    {{"name": "痛风", "type": "symptom"}},
    {{"name": "虾", "type": "ingredient"}}
  ],
  "intent": "disease_diet",
  "intent_confidence": 0.94,
  "intent_reason": "用户询问疾病状态下某食材是否适合",
  "planned_routes": ["semantic", "relation"],
  "route_reason": {{
    "semantic": "需要检索疾病饮食建议",
    "relation": "需要查询痛风与虾、嘌呤相关关系"
  }},
  "clarification_needed": false,
  "clarification_question": "",
  "has_profile_signal": false
}}

Q: 我花生过敏，帮我记一下
{{
  "entities": [{{"name": "花生", "type": "ingredient"}}],
  "intent": "profile_management",
  "intent_confidence": 0.98,
  "intent_reason": "用户明确要求保存个人过敏信息",
  "planned_routes": [],
  "route_reason": {{}},
  "clarification_needed": false,
  "clarification_question": "",
  "has_profile_signal": true
}}

Q: 我今天早餐吃了鸡蛋和牛奶，怎么样？
{{
  "entities": [
    {{"name": "鸡蛋", "type": "ingredient"}},
    {{"name": "牛奶", "type": "ingredient"}}
  ],
  "intent": "meal_record_analysis",
  "intent_confidence": 0.9,
  "intent_reason": "用户要求分析已摄入的饮食记录",
  "planned_routes": ["semantic", "text2sql"],
  "route_reason": {{
    "semantic": "需要检索饮食建议",
    "text2sql": "需要查询食材营养数据"
  }},
  "clarification_needed": false,
  "clarification_question": "",
  "has_profile_signal": false
}}

请严格返回 JSON，字段包括：
- entities: [{{"name": "...", "type": "ingredient|dish|nutrient|symptom|person|cooking_method"}}]
- intent: {intents}
- intent_confidence: 0.0-1.0
- intent_reason: 简短中文理由
- planned_routes: 从 semantic/relation/text2sql 中选择
- route_reason: 每条 route 的中文理由
- clarification_needed: boolean
- clarification_question: string
- has_profile_signal: boolean
"""

PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "ingredient",
                            "dish",
                            "nutrient",
                            "symptom",
                            "person",
                            "cooking_method",
                        ],
                    },
                },
                "required": ["name", "type"],
            },
        },
        "intent": {"type": "string", "enum": INTENTS},
        "intent_confidence": {"type": "number"},
        "intent_reason": {"type": "string"},
        "planned_routes": {
            "type": "array",
            "items": {"type": "string", "enum": ROUTES},
        },
        "route_reason": {"type": "object"},
        "clarification_needed": {"type": "boolean"},
        "clarification_question": {"type": "string"},
        "has_profile_signal": {"type": "boolean"},
    },
    "required": [
        "entities",
        "intent",
        "intent_confidence",
        "intent_reason",
        "planned_routes",
        "route_reason",
        "clarification_needed",
        "clarification_question",
        "has_profile_signal",
    ],
}


def planner_node(
    state: AgentState,
    llm: BaseLLMClient,
    memory_manager: MemoryManager | None = None,
) -> dict:
    """Analyze query intent, plan retrieval routes, and apply deterministic guardrails."""
    query = state["query"]
    logger.info(f"[Planner] analyze query: {query[:60]}...")

    full_query = query
    if state.get("clarification_answer"):
        full_query = f"{query}（补充信息：{state['clarification_answer']}）"

    direct_result = _strong_rule_short_circuit(full_query)
    if direct_result:
        logger.info(
            f"[Planner] strong rule hit: intent={direct_result['intent']} "
            f"routes={direct_result['planned_routes']}"
        )
        return direct_result

    planner_prompt = PLANNER_PROMPT.format(
        query=full_query,
        intents=" | ".join(INTENTS),
    )

    try:
        result = llm.extract_structured(
            prompt=planner_prompt,
            schema=PLANNER_SCHEMA,
            system=SYSTEM_PROMPT,
        )
        entities = _normalize_entities(result.get("entities", []))
        intent = str(result.get("intent", "general") or "general")
        intent_confidence = _clamp_float(result.get("intent_confidence", 0.0))
        intent_reason = str(result.get("intent_reason", "") or "")
        planned_routes = _normalize_routes(result.get("planned_routes", []))
        route_reason = dict(result.get("route_reason", {}) or {})
        clarification_needed = bool(result.get("clarification_needed", False))
        clarification_question = str(result.get("clarification_question", "") or "")
        has_profile_signal = bool(result.get("has_profile_signal", False))
    except Exception as exc:
        logger.warning(f"[Planner] LLM analysis failed, using fallback: {exc}")
        entities = _keyword_fallback(full_query)
        intent, planned_routes, intent_reason = _fallback_intent_and_routes(full_query)
        intent_confidence = 0.55
        route_reason = {route: "规则兜底选择" for route in planned_routes}
        clarification_needed = False
        clarification_question = ""
        has_profile_signal = _has_profile_signal(full_query)

    checked = _post_check(
        query=full_query,
        entities=entities,
        intent=intent,
        planned_routes=planned_routes,
        clarification_needed=clarification_needed,
        clarification_question=clarification_question,
        has_profile_signal=has_profile_signal,
        route_reason=route_reason,
        intent_confidence=intent_confidence,
    )

    logger.info(
        f"[Planner] intent={checked['intent']} conf={intent_confidence:.2f}, "
        f"entities={[e.get('name', '') for e in checked['entities']]}, "
        f"routes={checked['planned_routes']}, clarify={checked['clarification_needed']}"
    )

    return _build_result(
        intent=checked["intent"],
        intent_confidence=intent_confidence,
        intent_reason=intent_reason or checked["intent_reason"],
        route_reason=checked["route_reason"],
        entities=checked["entities"],
        has_profile_signal=checked["has_profile_signal"],
        planned_routes=checked["planned_routes"],
        clarification_needed=checked["clarification_needed"],
        clarification_question=checked["clarification_question"],
    )


def _strong_rule_short_circuit(query: str) -> dict[str, Any] | None:
    """Return a planner result only for high-precision keyword patterns."""
    entities = _keyword_fallback(query)
    content_classifier = UserContentClassifier()
    if content_classifier.is_explicit_save_request(query):
        content_result = content_classifier.classify(query)
        if content_result.content_type is not None:
            return _build_result(
                intent="content_save",
                intent_confidence=content_result.confidence,
                intent_reason=f"用户明确要求保存{content_result.content_type.value}内容",
                route_reason={},
                entities=entities,
                has_profile_signal=False,
                planned_routes=[],
            )

    if _looks_like_profile_management(query):
        return _build_result(
            intent="profile_management",
            intent_confidence=0.99,
            intent_reason="命中强规则：用户明确要求保存或修改长期画像",
            route_reason={},
            entities=entities,
            has_profile_signal=True,
            planned_routes=[],
        )

    if _is_clear_numeric_filter(query):
        return _build_result(
            intent="nutrition_filter",
            intent_confidence=0.98,
            intent_reason="命中强规则：包含营养指标和明确数值筛选条件",
            route_reason={"text2sql": "按结构化营养数值筛选"},
            entities=entities,
            has_profile_signal=_has_profile_signal(query),
            planned_routes=["text2sql"],
        )

    if _has_disease_term(query) and (_looks_like_safety_check(query) or _has_food_entity(entities)):
        return _build_result(
            intent="disease_diet",
            intent_confidence=0.97,
            intent_reason="命中强规则：疾病/症状场景下询问食物是否适合",
            route_reason={
                "semantic": "检索疾病饮食建议",
                "relation": "查询疾病、症状与食材关系",
            },
            entities=entities,
            has_profile_signal=_has_profile_signal(query),
            planned_routes=["semantic", "relation"],
        )

    if _looks_like_recipe_instruction(query) and _has_food_entity(entities):
        return _build_result(
            intent="recipe_instruction",
            intent_confidence=0.96,
            intent_reason="命中强规则：询问明确菜品或食材的做法",
            route_reason={"semantic": "检索菜谱步骤或烹饪方法"},
            entities=entities,
            has_profile_signal=_has_profile_signal(query),
            planned_routes=["semantic"],
        )

    if _looks_like_safety_check(query) and _has_food_entity(entities):
        return _build_result(
            intent="safety_check",
            intent_confidence=0.95,
            intent_reason="命中强规则：询问某食物是否适合自己或特定人群",
            route_reason={
                "semantic": "检索食材安全和饮食建议",
                "relation": "查询人群、限制与食材关系",
            },
            entities=entities,
            has_profile_signal=_has_profile_signal(query),
            planned_routes=["semantic", "relation"],
        )

    return None


def _build_result(
    *,
    intent: str,
    intent_confidence: float,
    intent_reason: str,
    route_reason: dict[str, Any],
    entities: list[dict[str, str]],
    has_profile_signal: bool,
    planned_routes: list[str],
    clarification_needed: bool = False,
    clarification_question: str = "",
) -> dict[str, Any]:
    return {
        "intent": intent,
        "intent_confidence": _clamp_float(intent_confidence),
        "intent_reason": intent_reason,
        "route_reason": route_reason,
        "entities": entities,
        "has_profile_signal": has_profile_signal,
        "planned_routes": planned_routes,
        "original_planned_routes": list(planned_routes),
        "executed_routes": [],
        "clarification_needed": clarification_needed,
        "clarification_question": clarification_question,
    }


def _post_check(
    query: str,
    entities: list[dict[str, str]],
    intent: str,
    planned_routes: list[str],
    clarification_needed: bool,
    clarification_question: str,
    has_profile_signal: bool,
    route_reason: dict[str, Any],
    intent_confidence: float,
) -> dict[str, Any]:
    """Keep LLM planner output inside deterministic safety boundaries."""
    if intent not in INTENTS:
        intent = "general"

    if _has_disease_term(query):
        intent = "disease_diet" if intent in {"general", "diet_advice", "nutrition_fact"} else intent
        planned_routes = _ensure_routes(planned_routes, ["semantic", "relation"])
    if _has_numeric_filter(query):
        intent = "nutrition_filter" if intent in {"general", "nutrition_fact"} else intent
        planned_routes = _ensure_routes(planned_routes, ["text2sql"])
    if _looks_like_recipe_instruction(query):
        intent = "recipe_instruction" if intent in {"general", "recipe_recommend"} else intent
        planned_routes = _ensure_routes(planned_routes, ["semantic"])
    if _has_profile_signal(query):
        has_profile_signal = True
        if _looks_like_profile_management(query):
            intent = "profile_management"
            planned_routes = []
    if _looks_like_safety_check(query) and intent not in {"disease_diet", "profile_management"}:
        intent = "safety_check"
        planned_routes = _ensure_routes(planned_routes, ["semantic", "relation"])

    if not planned_routes and intent not in {"profile_management", "content_save", "general"}:
        planned_routes = ["semantic"]

    if _needs_clarification(intent, entities, query):
        clarification_needed = True
        clarification_question = clarification_question or _clarification_question(intent)
    elif (
        intent_confidence < LOW_CONFIDENCE_CLARIFY_THRESHOLD
        and not _strong_rule_signal(query)
        and intent not in {"profile_management", "content_save"}
    ):
        clarification_needed = True
        clarification_question = (
            "我不太确定你的具体意图。你是想查营养信息、推荐饮食、比较食材，"
            "还是确认某种食材是否适合你？"
        )

    for route in planned_routes:
        route_reason.setdefault(route, "规则后处理保留或补充该检索路线")

    return {
        "intent": intent,
        "intent_reason": _fallback_reason(intent),
        "entities": entities,
        "planned_routes": planned_routes,
        "clarification_needed": clarification_needed,
        "clarification_question": clarification_question,
        "has_profile_signal": has_profile_signal,
        "route_reason": route_reason,
    }


def _fallback_intent_and_routes(query: str) -> tuple[str, list[str], str]:
    if _looks_like_profile_management(query):
        return "profile_management", [], "规则识别到用户要求保存或修改画像"
    if _has_disease_term(query):
        return "disease_diet", ["semantic", "relation"], "规则识别到疾病/症状饮食问题"
    if _has_numeric_filter(query):
        return "nutrition_filter", ["text2sql"], "规则识别到数值筛选问题"
    if _looks_like_recipe_instruction(query):
        return "recipe_instruction", ["semantic"], "规则识别到做法/步骤问题"
    if _looks_like_safety_check(query):
        return "safety_check", ["semantic", "relation"], "规则识别到食材安全检查问题"
    return "general", ["semantic"], "规则兜底为通用查询"


def _strong_rule_signal(query: str) -> bool:
    return any(
        [
            _has_disease_term(query),
            _is_clear_numeric_filter(query),
            _looks_like_profile_management(query),
            _looks_like_recipe_instruction(query) and bool(_keyword_fallback(query)),
            _looks_like_safety_check(query) and bool(_keyword_fallback(query)),
        ]
    )


def _normalize_entities(raw_entities: Any) -> list[dict[str, str]]:
    if not isinstance(raw_entities, list):
        return []
    allowed = {"ingredient", "dish", "nutrient", "symptom", "person", "cooking_method"}
    result = []
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        entity_type = str(item.get("type", "")).strip()
        if name and entity_type in allowed:
            result.append({"name": name, "type": entity_type})
    return result


def _normalize_routes(raw_routes: Any) -> list[str]:
    if not isinstance(raw_routes, list):
        return []
    result = []
    for route in raw_routes:
        route = str(route)
        if route in ROUTES and route not in result:
            result.append(route)
    return result


def _ensure_routes(routes: list[str], required: list[str]) -> list[str]:
    result = [route for route in routes if route in ROUTES]
    for route in required:
        if route not in result:
            result.append(route)
    return result


def _needs_clarification(intent: str, entities: list[dict[str, str]], query: str) -> bool:
    if any(entity.get("name") for entity in entities):
        return False
    if intent in {"nutrition_fact", "nutrition_compare", "nutrition_filter", "safety_check"}:
        return True
    if re.search(r"(这个|它|那个).*(能不能|可以|热量|营养|怎么吃)", query):
        return True
    return False


def _clarification_question(intent: str) -> str:
    if intent == "safety_check":
        return "你想确认哪种食材或菜品是否适合你？"
    if intent == "nutrition_filter":
        return "你想按哪个营养指标和数值范围筛选？"
    return "你想查询的具体食材、菜品或目标是什么？"


def _has_disease_term(query: str) -> bool:
    disease_terms = [
        "痛风",
        "糖尿病",
        "高血压",
        "高血脂",
        "脂肪肝",
        "贫血",
        "肾病",
        "胃病",
        "胃炎",
        "肠胃炎",
        "乳糖不耐",
        "胆固醇",
        "尿酸",
        "胰岛素",
    ]
    return any(term in query for term in disease_terms)


def _has_numeric_filter(query: str) -> bool:
    return bool(
        re.search(
            r"(大于|小于|超过|低于|不少于|不超过|高于|少于|>=|<=|>|<|\d+\s*(g|克|毫克|mg|kcal|千卡|大卡))",
            query,
            flags=re.IGNORECASE,
        )
    )


def _is_clear_numeric_filter(query: str) -> bool:
    nutrient_terms = [
        "蛋白质",
        "脂肪",
        "碳水",
        "碳水化合物",
        "热量",
        "能量",
        "卡路里",
        "钠",
        "盐",
        "糖",
        "膳食纤维",
        "钙",
        "铁",
        "锌",
        "钾",
        "嘌呤",
    ]
    filter_terms = ["有哪些", "筛选", "推荐", "找", "列出", "食物", "食材"]
    return _has_numeric_filter(query) and (
        any(term in query for term in nutrient_terms)
        or any(term in query for term in filter_terms)
    )


def _has_profile_signal(query: str) -> bool:
    patterns = [
        r"我.*(过敏|不能吃|不吃|忌口)",
        r"(我是|我属于).*(素食|乳糖不耐|孕妇|健身|减脂|增肌)",
        r"(我的目标|我想|计划).*(减脂|减肥|增肌|控糖|控盐|控制体重)",
        r"(我喜欢|我不喜欢|我讨厌|我偏好)",
    ]
    return any(re.search(pattern, query) for pattern in patterns)


def _looks_like_profile_management(query: str) -> bool:
    management_terms = ["记住", "记录", "保存", "帮我记", "以后", "更新", "改成", "删除"]
    return _has_profile_signal(query) and any(term in query for term in management_terms)


def _looks_like_recipe_instruction(query: str) -> bool:
    return any(term in query for term in ["怎么做", "做法", "步骤", "烹饪", "怎么炒", "怎么煮", "教程"])


def _looks_like_safety_check(query: str) -> bool:
    return bool(
        re.search(
            r"(我|本人|孩子|老人|孕妇|糖尿病|痛风|高血压|高血脂)?.*(能不能吃|可以吃|适合吃|能吃吗|安全吗|可不可以吃)",
            query,
        )
    )


def _has_food_entity(entities: list[dict[str, str]]) -> bool:
    return any(entity.get("type") in {"ingredient", "dish"} for entity in entities)


def _fallback_reason(intent: str) -> str:
    return {
        "nutrition_fact": "查询营养事实",
        "nutrition_compare": "比较营养或食材",
        "nutrition_filter": "按数值条件筛选营养数据",
        "ingredient_knowledge": "查询食材知识",
        "recipe_recommend": "推荐菜谱或餐食",
        "recipe_instruction": "询问烹饪做法",
        "disease_diet": "疾病或症状相关饮食问题",
        "profile_management": "用户画像管理",
        "meal_record_analysis": "饮食记录分析",
        "food_image_analysis": "食物图片分析",
        "safety_check": "个性化食材安全检查",
        "diet_advice": "个性化饮食建议",
        "general": "通用问题",
    }.get(intent, "通用问题")


def _clamp_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _keyword_fallback(query: str) -> list[dict[str, str]]:
    known: dict[str, list[str]] = {
        "ingredient": [
            "番茄",
            "鸡蛋",
            "黄瓜",
            "菠菜",
            "豆腐",
            "牛肉",
            "猪肉",
            "鸡肉",
            "鱼",
            "虾",
            "白菜",
            "土豆",
            "胡萝卜",
            "洋葱",
            "大蒜",
            "姜",
            "牛奶",
            "酸奶",
            "苹果",
            "香蕉",
            "米饭",
            "面条",
            "燕麦",
            "花生",
            "坚果",
        ],
        "dish": [
            "番茄炒蛋",
            "西红柿炒鸡蛋",
            "宫保鸡丁",
            "鱼香肉丝",
            "麻婆豆腐",
            "沙拉",
            "粥",
        ],
        "nutrient": [
            "蛋白质",
            "脂肪",
            "碳水",
            "碳水化合物",
            "热量",
            "能量",
            "维生素",
            "钙",
            "铁",
            "锌",
            "钠",
            "钾",
            "膳食纤维",
            "糖",
            "嘌呤",
        ],
        "symptom": [
            "糖尿病",
            "高血压",
            "高血脂",
            "痛风",
            "贫血",
            "脂肪肝",
            "肾病",
            "乳糖不耐",
            "尿酸",
        ],
        "person": ["孕妇", "儿童", "老人", "老年人", "青少年", "运动员", "素食者"],
    }
    found = []
    for entity_type, keywords in known.items():
        for keyword in keywords:
            if keyword in query and not any(item["name"] == keyword for item in found):
                found.append({"name": keyword, "type": entity_type})
    return found
