import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.rerank_node import rerank_node
from src.agent.state import create_initial_state


def test_strong_personalized_query_pins_current_user_content_and_filters_others():
    state = create_initial_state("按我的饮食计划，晚餐怎么吃？", user_id="u1")
    state["intent"] = "diet_advice"
    state["evidence"]["semantic"] = [
        {
            "content": "通用晚餐建议：多吃蔬菜，控制油盐。",
            "score": 0.9,
            "source_doc_id": "public:guideline:001",
        },
        {
            "content": "用户保存的饮食计划：晚餐控制主食，增加优质蛋白。",
            "score": 0.8,
            "source_doc_id": "user_content:u1:meal_plan:abc123",
        },
        {
            "content": "其他用户保存的饮食计划：晚餐高碳水。",
            "score": 0.99,
            "source_doc_id": "user_content:u2:meal_plan:def456",
        },
    ]

    result = rerank_node(state)
    evidence = result["reranked_evidence"]

    assert len(evidence) == 2
    assert result["personalization_policy"]["mode"] == "strong"
    assert result["personalization_policy"]["status"] == "private_content_found"
    assert all(item.get("source_doc_id") != "user_content:u2:meal_plan:def456" for item in evidence)
    assert evidence[0]["source_doc_id"] == "user_content:u1:meal_plan:abc123"
    assert evidence[0]["personalization_weight"] == 10.0
    assert evidence[0]["user_content_context"]["content_type"] == "meal_plan"


def test_weak_personalized_query_only_soft_boosts_user_content():
    state = create_initial_state("我晚餐怎么吃比较好？", user_id="u1")
    state["intent"] = "diet_advice"
    state["evidence"]["semantic"] = [
        {
            "content": "用户保存的饮食计划：晚餐控制主食。",
            "score": 0.8,
            "source_doc_id": "user_content:u1:meal_plan:abc123",
        }
    ]

    result = rerank_node(state)

    assert result["personalization_policy"]["mode"] == "weak"
    assert result["personalization_policy"]["status"] == "not_required"
    assert result["reranked_evidence"][0]["personalization_weight"] == 1.25


def test_strong_personalized_query_records_missing_private_content():
    state = create_initial_state("根据我的体检报告给点建议", user_id="u1")
    state["intent"] = "diet_advice"
    state["evidence"]["semantic"] = [
        {
            "content": "公共指南：均衡饮食，控制油盐。",
            "score": 0.9,
            "source_doc_id": "public:guideline:001",
        }
    ]

    result = rerank_node(state)

    assert result["personalization_policy"]["mode"] == "strong"
    assert result["personalization_policy"]["requested_content_types"] == ["lab_report"]
    assert result["personalization_policy"]["status"] == "private_content_missing"


def test_rerank_uses_metadata_for_user_saved_content():
    state = create_initial_state("根据我的体重记录给点建议", user_id="u1")
    state["intent"] = "diet_advice"
    state["evidence"]["relation"] = [
        {
            "subject": "体重记录",
            "predicate": "属于",
            "object": "用户体测数据",
            "confidence": 0.7,
            "metadata": {
                "source_type": "user_saved_content",
                "user_id": "u1",
                "user_content_type": "body_metrics",
                "source_doc_id": "user_content:u1:body_metrics:abc123",
            },
        }
    ]

    result = rerank_node(state)
    evidence = result["reranked_evidence"]

    assert result["personalization_policy"]["mode"] == "strong"
    assert evidence[0]["user_content_context"]["content_type"] == "body_metrics"
    assert evidence[0]["personalization_weight"] == 10.0
