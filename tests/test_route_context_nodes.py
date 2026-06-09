import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.pg_relation_retrieval_node import pg_relation_retrieval_node
from src.agent.semantic_retrieval_node import semantic_retrieval_node
from src.agent.state import create_initial_state
from src.agent import text2sql_node as text2sql_module


class RecordingEmbedding:
    text = ""

    def encode(self, text, normalize_embeddings=True):
        RecordingEmbedding.text = text

        class Vector:
            def tolist(self):
                return [0.1, 0.2, 0.3]

        return Vector()


class EmptyMilvus:
    def hybrid_search(self, **kwargs):
        return []


class RecordingPG:
    entities = []

    def query_triples_by_entities(self, entity_names, limit=80):
        RecordingPG.entities = entity_names
        return []

    def query_kv_by_entity_batch(self, entity_names):
        return []


class DummyLLM:
    pass


def test_semantic_node_uses_route_context_query():
    state = create_initial_state("original query")
    state["route_context"] = {
        "semantic": {"query": "expanded semantic query", "reason": "test"}
    }

    with patch("src.agent.semantic_retrieval_node.SentenceTransformer", return_value=RecordingEmbedding()):
        semantic_retrieval_node(state, EmptyMilvus())

    assert RecordingEmbedding.text == "expanded semantic query"


def test_relation_node_uses_route_context_entities():
    state = create_initial_state("original query")
    state["entities"] = [{"name": "old", "type": "ingredient"}]
    state["route_context"] = {
        "relation": {
            "entities": [{"name": "expanded", "type": "ingredient"}],
            "reason": "test",
        }
    }

    pg_relation_retrieval_node(state, RecordingPG())

    assert RecordingPG.entities == ["expanded"]


def test_text2sql_node_uses_route_context_query(monkeypatch):
    state = create_initial_state("original query")
    state["route_context"] = {
        "text2sql": {"query": "expanded sql query", "reason": "test"}
    }
    seen = {}

    def fake_generate_sql(query, llm):
        seen["generate_query"] = query
        return "SELECT 1"

    def fake_execute_sql(sql, pg):
        return [{"value": 1}]

    def fake_structure_results(query, sql, raw_results, llm):
        seen["structure_query"] = query
        return [{"content": "ok"}]

    monkeypatch.setattr(text2sql_module, "_generate_sql", fake_generate_sql)
    monkeypatch.setattr(text2sql_module, "_execute_sql", fake_execute_sql)
    monkeypatch.setattr(text2sql_module, "_structure_results", fake_structure_results)

    text2sql_module.text2sql_node(state, DummyLLM(), object())

    assert seen == {
        "generate_query": "expanded sql query",
        "structure_query": "expanded sql query",
    }

