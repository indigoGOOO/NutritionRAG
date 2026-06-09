import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.planner_node import planner_node
from src.agent.state import create_initial_state


class FakeLLM:
    def __init__(self):
        self.prompt = ""

    def extract_structured(self, prompt, schema, system=""):
        self.prompt = prompt
        return {
            "entities": [{"name": "花生", "type": "ingredient"}],
            "intent": "diet_advice",
            "planned_routes": ["semantic"],
            "clarification_needed": False,
            "clarification_question": "",
            "has_profile_signal": True,
        }


class FakeMemoryManager:
    def format_memory_context(self, query):
        return "SHOULD_NOT_BE_IN_PLANNER_PROMPT"


def test_planner_flags_profile_signal_without_injecting_memory_context():
    llm = FakeLLM()
    state = create_initial_state("我对花生过敏")

    result = planner_node(state, llm, FakeMemoryManager())

    assert result["has_profile_signal"] is True
    assert result["intent"] == "diet_advice"
    assert "SHOULD_NOT_BE_IN_PLANNER_PROMPT" not in llm.prompt
