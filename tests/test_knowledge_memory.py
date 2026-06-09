"""长期问答记忆策略测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.knowledge_memory import KnowledgeMemory
from src.memory.memory_manager import MemoryManager
from src.memory.base import MemoryItem


def _memory_without_backends() -> KnowledgeMemory:
    return KnowledgeMemory.__new__(KnowledgeMemory)


class TestKnowledgeMemoryPolicy:
    def test_quality_rejects_short_uncited_answer(self):
        memory = _memory_without_backends()
        quality = memory.evaluate_quality(
            answer="可以。",
            citations=[],
            tags=["nutrition_info"],
            evidence_count=0,
        )

        assert quality["score"] < 0.65
        assert "no_citations" in quality["reasons"]
        assert "too_short" in quality["reasons"]

    def test_quality_accepts_cited_evidence_backed_answer(self):
        memory = _memory_without_backends()
        answer = (
            "鸡蛋含有较丰富的蛋白质，适合作为日常优质蛋白来源。一般成年人可以适量食用，"
            "通常建议结合蔬菜、全谷物一起搭配，避免单一摄入。如果对鸡蛋过敏，或医生明确要求限制胆固醇摄入，"
            "则应根据个人情况调整。"
        )
        quality = memory.evaluate_quality(
            answer=answer,
            citations=[{"source_id": 1}, {"source_id": 2}],
            tags=["nutrition_info"],
            evidence_count=3,
        )

        assert quality["score"] >= 0.65
        assert "has_citations" in quality["reasons"]
        assert "has_evidence" in quality["reasons"]

    def test_reuse_policy_blocks_medical_direct_reuse(self):
        assert KnowledgeMemory.get_reuse_policy(0.98, ["disease_diet"]) == "context_only"
        assert not KnowledgeMemory.is_safe_reuse_intent("disease_diet")

    def test_reuse_policy_allows_safe_high_similarity_intent(self):
        assert KnowledgeMemory.get_reuse_policy(0.98, ["recipe_recommend"]) == "direct_reuse"
        assert KnowledgeMemory.is_safe_reuse_intent("recipe_recommend")


class FakeKnowledge:
    @staticmethod
    def is_safe_reuse_intent(intent: str) -> bool:
        return intent == "recipe_recommend"

    def search(self, query: str, limit: int = 3):
        long_context_answer = "这是一段只能作为意图参考的历史答案。" * 20
        return [
            MemoryItem(
                id="km_1",
                content="完整做法：先炒鸡蛋，再炒番茄，最后合炒并调味。",
                metadata={"question": "番茄炒蛋怎么做？", "reuse_policy": "direct_reuse"},
                score=0.98,
            ),
            MemoryItem(
                id="km_2",
                content=long_context_answer,
                metadata={"question": "晚餐吃什么？", "reuse_policy": "context_only"},
                score=0.86,
            ),
        ]


class TestAnswerMemoryContext:
    def test_direct_reuse_and_context_only_are_formatted_differently(self):
        manager = MemoryManager.__new__(MemoryManager)
        manager.knowledge = FakeKnowledge()

        text = manager.format_answer_memory_context("番茄炒蛋", intent="recipe_recommend")

        assert text.index("## 可强参考的历史问答") < text.index("## 仅供理解意图的历史问答")
        assert "完整做法：先炒鸡蛋，再炒番茄，最后合炒并调味。" in text
        assert "不可作为事实依据，不允许作为来源引用" in text
        assert "这是一段只能作为意图参考的历史答案。" in text
        assert "..." in text

    def test_unsafe_intent_demotes_direct_reuse_to_context_only(self):
        manager = MemoryManager.__new__(MemoryManager)
        manager.knowledge = FakeKnowledge()

        text = manager.format_answer_memory_context("痛风晚餐怎么吃", intent="disease_diet")

        assert "## 可强参考的历史问答" not in text
        assert "## 仅供理解意图的历史问答" in text
