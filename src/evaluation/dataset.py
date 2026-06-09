"""Dataset models and IO helpers for RAG evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationExample:
    """One evaluated RAG answer.

    Required fields are intentionally close to RAGAS names:
    - question: user query
    - answer: generated answer
    - contexts: retrieved evidence text chunks used by the answer
    - ground_truth: reference answer, if available
    """

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    ground_truth: str = ""
    intent: str = ""
    expected_intent: str = ""
    expected_routes: list[str] = field(default_factory=list)
    expected_route_order: list[str] = field(default_factory=list)
    actual_routes: list[str] = field(default_factory=list)
    expected_context_ids: list[str] = field(default_factory=list)
    retrieved_context_ids: list[str] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    user_profile: dict[str, Any] = field(default_factory=dict)
    forbidden_ingredients: list[str] = field(default_factory=list)
    memory_policies: list[str] = field(default_factory=list)
    # Safety filter evaluation
    expected_safety_warnings: list[str] = field(default_factory=list)
    actual_safety_warnings: list[str] = field(default_factory=list)
    # Router dynamic discovery evaluation
    expected_dynamic_entities: list[str] = field(default_factory=list)
    actual_dynamic_entities: list[str] = field(default_factory=list)
    expected_dynamic_routes: list[str] = field(default_factory=list)
    actual_dynamic_routes: list[str] = field(default_factory=list)
    # Router behavior
    executed_routes: list[str] = field(default_factory=list)
    fallback_routes: list[str] = field(default_factory=list)
    route_status: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    # Multi-turn
    multi_turn_group: str = ""
    turn_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationExample":
        question = str(data.get("question", data.get("query", ""))).strip()
        answer = str(data.get("answer", "")).strip()
        ground_truth = str(
            data.get("ground_truth", data.get("reference", data.get("expected_answer", "")))
        ).strip()
        contexts = _coerce_str_list(data.get("contexts", data.get("retrieved_contexts", [])))

        if not question:
            raise ValueError("evaluation example missing required field: question")
        if not answer:
            raise ValueError("evaluation example missing required field: answer")

        return cls(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            intent=str(data.get("intent", "")).strip(),
            expected_intent=str(data.get("expected_intent", "")).strip(),
            expected_routes=_coerce_str_list(data.get("expected_routes", [])),
            expected_route_order=_coerce_str_list(data.get("expected_route_order", [])),
            actual_routes=_coerce_str_list(data.get("actual_routes", data.get("planned_routes", []))),
            expected_context_ids=_coerce_str_list(data.get("expected_context_ids", [])),
            retrieved_context_ids=_coerce_str_list(data.get("retrieved_context_ids", [])),
            citations=_coerce_int_list(data.get("citations", [])),
            user_profile=dict(data.get("user_profile", {}) or {}),
            forbidden_ingredients=_coerce_str_list(data.get("forbidden_ingredients", [])),
            memory_policies=_coerce_str_list(data.get("memory_policies", [])),
            expected_safety_warnings=_coerce_str_list(data.get("expected_safety_warnings", [])),
            actual_safety_warnings=_coerce_str_list(data.get("actual_safety_warnings", [])),
            expected_dynamic_entities=_coerce_str_list(data.get("expected_dynamic_entities", [])),
            actual_dynamic_entities=_coerce_str_list(data.get("actual_dynamic_entities", [])),
            expected_dynamic_routes=_coerce_str_list(data.get("expected_dynamic_routes", [])),
            actual_dynamic_routes=_coerce_str_list(data.get("actual_dynamic_routes", [])),
            executed_routes=_coerce_str_list(data.get("executed_routes", [])),
            fallback_routes=_coerce_str_list(data.get("fallback_routes", [])),
            route_status=dict(data.get("route_status", {}) or {}),
            trace=dict(data.get("trace", {}) or {}),
            multi_turn_group=str(data.get("multi_turn_group", "")),
            turn_index=int(data.get("turn_index", 0)),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def to_ragas_dict(self) -> dict[str, Any]:
        """Return a row compatible with common RAGAS dataset schemas."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "reference": self.ground_truth,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Evaluation scores for a dataset."""

    examples: list[dict[str, Any]]
    aggregate: dict[str, float]
    ragas: dict[str, Any] = field(default_factory=dict)
    diff: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_jsonl(path: str | Path) -> list[EvaluationExample]:
    """Load evaluation examples from a JSONL file."""
    examples = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            examples.append(EvaluationExample.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid evaluation JSONL at line {line_no}: {exc}") from exc
    return examples


def save_json(path: str | Path, result: EvaluationResult):
    """Save evaluation result as UTF-8 JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(value, list):
        return [int(item) for item in value]
    return []
