"""Text2SQL node with reflection retry and fuzzy empty-result retry."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.agent.state import AgentState
from src.indexing.llm_client import BaseLLMClient
from src.storage.pg_client import PostgreSQLClient

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

SQL_GENERATION_PROMPT = """Generate a PostgreSQL SELECT query for the user's nutrition or diet question.

Database schema:

CREATE TABLE chunks (
    chunk_id SERIAL PRIMARY KEY,
    content TEXT,
    chunk_type VARCHAR(50),
    doc_category VARCHAR(50),
    source_doc_id VARCHAR(255),
    token_count INTEGER,
    metadata JSONB
);

CREATE TABLE kv_pairs (
    kv_id SERIAL PRIMARY KEY,
    entity_id VARCHAR(255),
    entity_type VARCHAR(50),
    attribute VARCHAR(255),
    value JSONB,
    source_chunk_id INTEGER REFERENCES chunks,
    confidence FLOAT
);

CREATE TABLE triples (
    triple_id SERIAL PRIMARY KEY,
    subject VARCHAR(255),
    predicate VARCHAR(255),
    object VARCHAR(255),
    source_chunk_id INTEGER REFERENCES chunks,
    confidence FLOAT
);

User query: {query}

Rules:
1. Output only one PostgreSQL SELECT statement.
2. Use JSONB operators such as value->>'attr' when querying kv_pairs.value.
3. Cast numeric JSONB text values with CAST(value->>'attr' AS numeric) when needed.
4. Add a reasonable LIMIT.
5. Do not output markdown fences or explanations.
"""

SQL_REFLECTION_PROMPT = """The previous PostgreSQL query failed.

User query: {query}
Previous SQL: {sql}
Error: {error}

Analyze the error and generate a corrected PostgreSQL SELECT statement.
Output only one SQL statement. Do not output markdown fences or explanations.
"""

SQL_RESULT_PROMPT = """Convert these SQL query results into JSON evidence.

User query: {query}
SQL: {sql}
Results: {results}

Return JSON:
{{
  "summary": "brief result summary",
  "rows": [
    {{"entity": "entity name", "attribute": "attribute", "value": "value", "detail": "brief detail"}}
  ]
}}

If results are empty, return rows=[].
"""


def text2sql_node(state: AgentState, llm: BaseLLMClient, pg: PostgreSQLClient) -> dict:
    """Generate SQL, execute it, reflect on failures, and fuzzy-retry empty results."""
    route_context = state.get("route_context", {}).get("text2sql", {})
    query = route_context.get("query") or state["query"]
    logger.info("[Text2SQL] NL to SQL: %s...", query[:60])

    sql = _generate_sql(query, llm)
    if not sql:
        logger.warning("[Text2SQL] SQL generation failed")
        return _route_result(state, [], "error", "sql_generation_failed", reason="sql_generation_failed")

    raw_results: list[dict[str, Any]] = []
    for attempt in range(MAX_RETRIES):
        try:
            raw_results = _execute_sql(sql, pg)
            break
        except Exception as exc:
            logger.warning("[Text2SQL] execution failed (attempt %s): %s", attempt + 1, exc)
            if attempt < MAX_RETRIES - 1:
                sql = _reflect_sql(query, sql, str(exc), llm)
                if not sql:
                    return _route_result(
                        state,
                        [],
                        "error",
                        "sql_reflection_failed",
                        reason="sql_reflection_failed",
                    )
                continue
            return _route_result(
                state,
                [],
                "error",
                str(exc),
                reason=_classify_sql_error(str(exc)),
            )

    retry_strategy = ""
    if not raw_results:
        fuzzy_sql = _build_fuzzy_sql(sql)
        if not fuzzy_sql or fuzzy_sql == sql:
            return _route_result(state, [], "empty", "", reason="empty_exact_match")

        retry_strategy = "fuzzy_ilike"
        logger.info("[Text2SQL] empty result, retrying with fuzzy ILIKE")
        try:
            raw_results = _execute_sql(fuzzy_sql, pg)
            sql = fuzzy_sql
        except Exception as exc:
            logger.warning("[Text2SQL] fuzzy retry failed: %s", exc)
            return _route_result(
                state,
                [],
                "error",
                str(exc),
                reason=_classify_sql_error(str(exc)),
                retry_strategy=retry_strategy,
            )

    if not raw_results:
        return _route_result(
            state,
            [],
            "empty",
            "",
            reason="empty_after_fuzzy_retry",
            retry_strategy=retry_strategy or "fuzzy_ilike",
        )

    evidence = _structure_results(query, sql, raw_results, llm)
    logger.info("[Text2SQL] structured evidence count=%s", len(evidence))
    if not evidence:
        return _route_result(
            state,
            evidence,
            "empty",
            "",
            reason="empty_after_result_structuring",
            retry_strategy=retry_strategy,
        )

    return _route_result(
        state,
        evidence,
        "success",
        "",
        reason="success",
        retry_strategy=retry_strategy,
    )


def _route_result(
    state: AgentState,
    evidence: list[dict],
    status: str,
    error: str = "",
    reason: str = "",
    retry_strategy: str = "",
) -> dict:
    return {
        "evidence": _with_text2sql_evidence(state, evidence),
        "route_status": _with_route_status(
            state,
            "text2sql",
            status,
            len(evidence),
            error,
            reason,
            retry_strategy,
        ),
    }


def _with_text2sql_evidence(state: AgentState, evidence: list[dict]) -> dict:
    merged_evidence = {**state.get("evidence", {})}
    merged_evidence["text2sql"] = evidence
    return merged_evidence


def _with_route_status(
    state: AgentState,
    route: str,
    status: str,
    count: int,
    error: str = "",
    reason: str = "",
    retry_strategy: str = "",
) -> dict:
    merged = {**state.get("route_status", {})}
    merged[route] = {
        "status": status,
        "count": count,
        "error": error,
        "reason": reason or status,
        "retry_strategy": retry_strategy,
    }
    return merged


def _generate_sql(query: str, llm: BaseLLMClient) -> str:
    try:
        prompt = SQL_GENERATION_PROMPT.format(query=query)
        sql = llm.generate(prompt=prompt).strip()
        return _sanitize_select_sql(_strip_markdown_fence(sql))
    except Exception as exc:
        logger.error("[Text2SQL] generation failed: %s", exc)
        return ""


def _execute_sql(sql: str, pg: PostgreSQLClient) -> list[dict]:
    with pg.conn.cursor() as cur:
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = []
        for row in cur.fetchmany(50):
            processed = {}
            for idx, col in enumerate(columns):
                val = row[idx]
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
                processed[col] = val
            rows.append(processed)
        return rows


def _reflect_sql(query: str, prev_sql: str, error: str, llm: BaseLLMClient) -> str:
    try:
        prompt = SQL_REFLECTION_PROMPT.format(query=query, sql=prev_sql, error=error)
        sql = llm.generate(prompt=prompt).strip()
        return _sanitize_select_sql(_strip_markdown_fence(sql))
    except Exception as exc:
        logger.error("[Text2SQL] reflection failed: %s", exc)
        return ""


def _build_fuzzy_sql(sql: str) -> str:
    """Rewrite exact string equality predicates to ILIKE fuzzy predicates."""
    fuzzy = re.sub(
        r"(?P<lhs>(?:[\w.]+|[\w.]+->>'[^']+'))\s*=\s*'(?P<value>[^']+)'",
        lambda m: f"{m.group('lhs')} ILIKE '%{_escape_like_value(m.group('value'))}%'",
        sql,
        flags=re.IGNORECASE,
    )
    fuzzy = re.sub(
        r"(?P<lhs>(?:[\w.]+|[\w.]+->>'[^']+'))\s+IN\s*\(\s*'(?P<value>[^']+)'\s*\)",
        lambda m: f"{m.group('lhs')} ILIKE '%{_escape_like_value(m.group('value'))}%'",
        fuzzy,
        flags=re.IGNORECASE,
    )
    return _sanitize_select_sql(fuzzy)


def _escape_like_value(value: str) -> str:
    return value.replace("%", r"\%").replace("_", r"\_")


def _classify_sql_error(error: str) -> str:
    lowered = error.lower()
    if any(token in lowered for token in ("does not exist", "undefinedcolumn", "undefinedtable")):
        return "schema_mismatch"
    if "syntax" in lowered or "invalid" in lowered:
        return "invalid_sql"
    return "sql_execution_failed"


def _strip_markdown_fence(sql: str) -> str:
    if not sql.startswith("```"):
        return sql
    lines = [line for line in sql.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def _sanitize_select_sql(sql: str) -> str:
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()

    upper = sql.upper()
    if not upper.startswith("SELECT"):
        logger.warning("[Text2SQL] rejected non-SELECT SQL: %s", sql[:50])
        return ""

    if ";" in sql or "--" in sql or "/*" in sql or "*/" in sql:
        logger.warning("[Text2SQL] rejected suspicious SQL")
        return ""

    forbidden = (" INSERT ", " UPDATE ", " DELETE ", " DROP ", " ALTER ", " CREATE ", " TRUNCATE ")
    padded = f" {upper} "
    if any(token in padded for token in forbidden):
        logger.warning("[Text2SQL] rejected write SQL")
        return ""

    return sql


def _structure_results(
    query: str,
    sql: str,
    results: list[dict],
    llm: BaseLLMClient,
) -> list[dict]:
    if not results:
        return []

    try:
        result_str = json.dumps(results, ensure_ascii=False, default=str)
        prompt = SQL_RESULT_PROMPT.format(query=query, sql=sql, results=result_str[:3000])
        structured = llm.extract_structured(
            prompt=prompt,
            schema={"summary": "str", "rows": []},
        )
        rows = structured.get("rows", [])
        for row in rows:
            row["source"] = "postgresql"
            row["evidence_type"] = "sql_result"
        return rows
    except Exception as exc:
        logger.debug("[Text2SQL] structuring failed, returning raw results: %s", exc)
        return [
            {
                "entity": str(row.get("entity_id", row.get("subject", ""))),
                "attribute": str(row.get("attribute", row.get("predicate", ""))),
                "value": str(row.get("value", row.get("object", ""))),
                "source": "postgresql",
                "evidence_type": "sql_raw",
            }
            for row in results[:20]
        ]
