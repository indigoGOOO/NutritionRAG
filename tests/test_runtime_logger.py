import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.observability.runtime_logger import build_runtime_log, summarize_evidence


def test_runtime_log_compacts_personalization_and_evidence_summary():
    result = {
        "answer": "建议参考你保存的饮食计划。",
        "citations": [{"source_id": 1}],
        "intent": "diet_advice",
        "intent_confidence": 0.9,
        "planned_routes": ["semantic"],
        "executed_routes": ["semantic"],
        "route_status": {"semantic": {"status": "success", "count": 1}},
        "personalization_policy": {
            "mode": "strong",
            "private_content_required": True,
            "private_content_found": True,
        },
        "reranked_evidence": [
            {
                "content": "用户保存的饮食计划：" + "晚餐控制主食。" * 30,
                "source_type": "semantic",
                "chunk_id": 12,
                "source_doc_id": "user_content:u1:meal_plan:abc123",
                "rerank_score": 7.6,
                "personalization_weight": 10.0,
                "user_content_context": {"content_type": "meal_plan"},
            }
        ],
    }

    log = build_runtime_log(
        request_id="req1",
        query="按我的饮食计划，晚餐怎么吃？",
        session_id="s1",
        user_id="u1",
        result=result,
        latency_ms=123,
    )

    assert log["personalization_mode"] == "strong"
    assert log["private_content_required"] is True
    assert log["private_content_found"] is True
    assert log["evidence_summary"][0]["source_doc_id"] == "user_content:u1:meal_plan:abc123"
    assert log["evidence_summary"][0]["user_content_type"] == "meal_plan"
    assert len(log["evidence_summary"][0]["content_preview"]) <= 160


def test_summarize_evidence_limits_item_count():
    evidence = [{"content": str(i), "score": i} for i in range(20)]

    assert len(summarize_evidence(evidence)) == 8
