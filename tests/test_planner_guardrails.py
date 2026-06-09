import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.planner_node import planner_node
from src.agent.state import create_initial_state


class WrongLLM:
    def extract_structured(self, prompt, schema, system=""):
        return {
            "entities": [],
            "intent": "general",
            "intent_confidence": 0.2,
            "intent_reason": "wrong",
            "planned_routes": [],
            "route_reason": {},
            "clarification_needed": False,
            "clarification_question": "",
            "has_profile_signal": False,
        }


class LowConfidenceLLM:
    def extract_structured(self, prompt, schema, system=""):
        return {
            "entities": [],
            "intent": "general",
            "intent_confidence": 0.31,
            "intent_reason": "用户表达不明确",
            "planned_routes": ["semantic"],
            "route_reason": {"semantic": "兜底检索"},
            "clarification_needed": False,
            "clarification_question": "",
            "has_profile_signal": False,
        }


class FailingLLM:
    def extract_structured(self, prompt, schema, system=""):
        raise RuntimeError("planner failed")


class ShouldNotCallLLM:
    def extract_structured(self, prompt, schema, system=""):
        raise AssertionError("strong rule should skip llm")


def test_disease_query_strong_rule_skips_llm_and_adds_relation_route():
    result = planner_node(create_initial_state("痛风可以吃虾吗？"), ShouldNotCallLLM())

    assert result["intent"] == "disease_diet"
    assert result["intent_confidence"] == 0.97
    assert result["planned_routes"] == ["semantic", "relation"]


def test_numeric_filter_strong_rule_skips_llm_and_adds_text2sql():
    result = planner_node(create_initial_state("蛋白质大于10g的食物有哪些？"), ShouldNotCallLLM())

    assert result["intent"] == "nutrition_filter"
    assert result["intent_confidence"] == 0.98
    assert result["planned_routes"] == ["text2sql"]


def test_profile_management_strong_rule_sets_profile_signal_and_no_routes():
    result = planner_node(create_initial_state("帮我记住，我花生过敏"), ShouldNotCallLLM())

    assert result["intent"] == "profile_management"
    assert result["intent_confidence"] == 0.99
    assert result["has_profile_signal"] is True
    assert result["planned_routes"] == []


def test_content_save_strong_rule_sets_no_routes():
    result = planner_node(create_initial_state("保存这个菜谱：食材鸡蛋，步骤1 打散"), ShouldNotCallLLM())

    assert result["intent"] == "content_save"
    assert result["planned_routes"] == []
    assert result["has_profile_signal"] is False


def test_recipe_instruction_strong_rule_skips_llm():
    result = planner_node(create_initial_state("番茄炒蛋怎么做？"), ShouldNotCallLLM())

    assert result["intent"] == "recipe_instruction"
    assert result["planned_routes"] == ["semantic"]
    assert result["intent_confidence"] == 0.96


def test_fallback_uses_rule_classifier_when_llm_fails_without_strong_rule():
    result = planner_node(create_initial_state("营养搭配有什么建议？"), FailingLLM())

    assert result["intent"] == "general"
    assert result["planned_routes"] == ["semantic"]
    assert result["intent_confidence"] == 0.55


def test_llm_output_still_gets_post_checked_when_not_short_circuited():
    result = planner_node(create_initial_state("痛风饮食注意什么？"), WrongLLM())

    assert result["intent"] == "disease_diet"
    assert "semantic" in result["planned_routes"]
    assert "relation" in result["planned_routes"]


def test_low_confidence_without_strong_rule_signal_triggers_clarification():
    result = planner_node(create_initial_state("这个怎么样？"), LowConfidenceLLM())

    assert result["clarification_needed"] is True
    assert "不太确定" in result["clarification_question"]
