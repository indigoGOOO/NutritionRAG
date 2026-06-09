import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.answer_node import answer_node
from src.agent.state import create_initial_state


class FailingLLM:
    def generate(self, *args, **kwargs):
        raise AssertionError("LLM should not be called when required private content is missing")


def test_answer_does_not_hard_answer_when_required_private_content_missing():
    state = create_initial_state("根据我的体检报告给点建议", user_id="u1")
    state["personalization_policy"] = {
        "mode": "strong",
        "private_content_required": True,
        "requested_content_types": ["lab_report"],
        "private_content_found": False,
        "status": "private_content_missing",
    }
    state["reranked_evidence"] = []

    result = answer_node(state, FailingLLM())

    assert "没有找到" in result["answer"]
    assert "体检报告" in result["answer"]
    assert result["citations"] == []
