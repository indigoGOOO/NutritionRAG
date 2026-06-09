import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.semantic_retrieval_node import semantic_retrieval_node
from src.agent.state import create_initial_state


class FakeEmbedding:
    def encode(self, text, normalize_embeddings=True):
        class Vector:
            def tolist(self):
                return [0.1, 0.2, 0.3]

        return Vector()


class FakeMilvus:
    def hybrid_search(self, **kwargs):
        return [
            {
                "content": "番茄相宜鸡蛋，常用于番茄炒蛋。",
                "chunk_id": "c1",
                "final_score": 0.91,
            }
        ]


def test_semantic_retrieval_discovers_entities_from_top_results():
    state = create_initial_state("番茄有什么营养？")
    state["entities"] = [{"name": "番茄", "type": "ingredient"}]

    with patch("src.agent.semantic_retrieval_node.SentenceTransformer", return_value=FakeEmbedding()):
        result = semantic_retrieval_node(state, FakeMilvus())

    assert result["route_status"]["semantic"]["status"] == "success"
    assert result["route_status"]["semantic"]["count"] == 1
    assert result["semantic_discovered_entities"] == [{"name": "鸡蛋", "type": "ingredient"}]
