import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import text2sql_node as text2sql_module
from src.agent.state import create_initial_state


class DummyLLM:
    pass


def test_text2sql_reflects_and_retries_execution(monkeypatch):
    state = create_initial_state("protein foods")
    calls = {"execute": 0, "reflect": 0}

    monkeypatch.setattr(text2sql_module, "_generate_sql", lambda query, llm: "SELECT bad")

    def fake_execute(sql, pg):
        calls["execute"] += 1
        if calls["execute"] == 1:
            raise Exception("syntax error")
        return [{"entity_id": "egg", "protein": 13}]

    def fake_reflect(query, sql, error, llm):
        calls["reflect"] += 1
        return "SELECT good"

    monkeypatch.setattr(text2sql_module, "_execute_sql", fake_execute)
    monkeypatch.setattr(text2sql_module, "_reflect_sql", fake_reflect)
    monkeypatch.setattr(
        text2sql_module,
        "_structure_results",
        lambda query, sql, rows, llm: [{"entity": rows[0]["entity_id"], "value": rows[0]["protein"]}],
    )

    result = text2sql_module.text2sql_node(state, DummyLLM(), object())

    assert calls == {"execute": 2, "reflect": 1}
    assert result["route_status"]["text2sql"]["status"] == "success"
    assert result["route_status"]["text2sql"]["reason"] == "success"
    assert result["evidence"]["text2sql"] == [{"entity": "egg", "value": 13}]


def test_text2sql_fuzzy_retry_when_exact_query_empty(monkeypatch):
    state = create_initial_state("chicken protein")
    executed_sql = []

    monkeypatch.setattr(
        text2sql_module,
        "_generate_sql",
        lambda query, llm: "SELECT * FROM kv_pairs WHERE entity_id = 'chicken'",
    )

    def fake_execute(sql, pg):
        executed_sql.append(sql)
        if "ILIKE" in sql:
            return [{"entity_id": "chicken breast", "protein": 31}]
        return []

    monkeypatch.setattr(text2sql_module, "_execute_sql", fake_execute)
    monkeypatch.setattr(
        text2sql_module,
        "_structure_results",
        lambda query, sql, rows, llm: [{"entity": rows[0]["entity_id"], "value": rows[0]["protein"]}],
    )

    result = text2sql_module.text2sql_node(state, DummyLLM(), object())

    assert executed_sql == [
        "SELECT * FROM kv_pairs WHERE entity_id = 'chicken'",
        "SELECT * FROM kv_pairs WHERE entity_id ILIKE '%chicken%'",
    ]
    assert result["route_status"]["text2sql"]["status"] == "success"
    assert result["route_status"]["text2sql"]["retry_strategy"] == "fuzzy_ilike"
    assert result["evidence"]["text2sql"] == [{"entity": "chicken breast", "value": 31}]


def test_text2sql_marks_empty_after_fuzzy_retry(monkeypatch):
    state = create_initial_state("unknown food")

    monkeypatch.setattr(
        text2sql_module,
        "_generate_sql",
        lambda query, llm: "SELECT * FROM kv_pairs WHERE entity_id = 'unknown'",
    )
    monkeypatch.setattr(text2sql_module, "_execute_sql", lambda sql, pg: [])

    result = text2sql_module.text2sql_node(state, DummyLLM(), object())
    status = result["route_status"]["text2sql"]

    assert status["status"] == "empty"
    assert status["reason"] == "empty_after_fuzzy_retry"
    assert status["retry_strategy"] == "fuzzy_ilike"
    assert result["evidence"]["text2sql"] == []

