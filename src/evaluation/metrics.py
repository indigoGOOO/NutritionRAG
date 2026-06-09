"""Deterministic multi-layer metrics for RAG evaluation."""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean
from typing import Any

from src.evaluation.dataset import EvaluationExample, EvaluationResult


class DeterministicMetricSuite:
    """Fast metrics that do not require LLM judges.

    These complement RAGAS and cover system-layer behavior:
    - retrieval id precision/recall when golden context ids are provided
    - planner intent and route accuracy when golden labels are provided
    - citation validity and coverage
    - answer/reference lexical overlap
    - context utilization by answer lexical overlap
    - nutrition safety and profile violation checks
    - memory policy compliance checks
    """

    MEDICAL_INTENTS = {"disease_diet", "medical", "diet_advice"}
    SAFETY_TERMS = ("医生", "医师", "营养师", "专业人士", "就医", "咨询")

    def score_dataset(self, examples: list[EvaluationExample]) -> EvaluationResult:
        rows = [self.score_example(example) for example in examples]
        aggregate = self._aggregate(rows)
        return EvaluationResult(examples=rows, aggregate=aggregate)

    def score_example(self, example: EvaluationExample) -> dict[str, Any]:
        retrieval = self._retrieval_scores(
            example.expected_context_ids,
            example.retrieved_context_ids,
        )
        planner = self._planner_scores(
            example.expected_intent,
            example.intent,
            example.expected_routes,
            example.actual_routes,
        )
        citation = self._citation_scores(example.citations, len(example.contexts))
        answer_overlap = lexical_f1(example.answer, example.ground_truth)
        context_utilization = max(
            (lexical_f1(example.answer, context) for context in example.contexts),
            default=0.0,
        )
        safety = self._safety_score(example.intent, example.answer)
        profile_safety = self._profile_safety_scores(
            example.answer,
            example.user_profile,
            example.forbidden_ingredients,
        )
        memory = self._memory_policy_scores(example.answer, example.memory_policies)
        route_order = self._route_order_scores(
            example.expected_route_order,
            example.executed_routes or example.actual_routes,
        )
        dynamic = self._dynamic_discovery_scores(
            example.expected_dynamic_entities,
            example.actual_dynamic_entities,
            example.expected_dynamic_routes,
            example.actual_dynamic_routes,
        )
        safety_filter = self._safety_filter_scores(
            example.expected_safety_warnings,
            example.actual_safety_warnings,
        )

        scores = {
            **retrieval,
            **planner,
            **citation,
            "answer_reference_f1": answer_overlap,
            "context_utilization": context_utilization,
            "safety_notice": safety,
            **profile_safety,
            **memory,
            **route_order,
            **dynamic,
            **safety_filter,
        }
        return {
            "question": example.question,
            "intent": example.intent,
            "scores": scores,
        }

    @staticmethod
    def _retrieval_scores(expected_ids: list[str], retrieved_ids: list[str]) -> dict[str, float]:
        if not expected_ids:
            return {
                "retrieval_precision": 0.0,
                "retrieval_recall": 0.0,
                "retrieval_f1": 0.0,
            }
        expected = set(expected_ids)
        retrieved = set(retrieved_ids)
        hits = len(expected & retrieved)
        precision = hits / len(retrieved) if retrieved else 0.0
        recall = hits / len(expected) if expected else 0.0
        return {
            "retrieval_precision": precision,
            "retrieval_recall": recall,
            "retrieval_f1": _f1(precision, recall),
        }

    @staticmethod
    def _planner_scores(
        expected_intent: str,
        actual_intent: str,
        expected_routes: list[str],
        actual_routes: list[str],
    ) -> dict[str, float]:
        scores = {}
        if expected_intent:
            scores["intent_accuracy"] = 1.0 if expected_intent == actual_intent else 0.0

        if expected_routes:
            expected = set(expected_routes)
            actual = set(actual_routes)
            hits = len(expected & actual)
            precision = hits / len(actual) if actual else 0.0
            recall = hits / len(expected)
            scores.update({
                "route_precision": precision,
                "route_recall": recall,
                "route_f1": _f1(precision, recall),
            })
        return scores

    @staticmethod
    def _citation_scores(citations: list[int], context_count: int) -> dict[str, float]:
        if not citations:
            return {
                "citation_validity": 0.0,
                "citation_coverage": 0.0,
            }
        valid = [idx for idx in citations if 1 <= idx <= context_count]
        return {
            "citation_validity": len(valid) / len(citations),
            "citation_coverage": len(set(valid)) / context_count if context_count else 0.0,
        }

    def _safety_score(self, intent: str, answer: str) -> float:
        if intent not in self.MEDICAL_INTENTS:
            return 1.0
        return 1.0 if any(term in answer for term in self.SAFETY_TERMS) else 0.0

    @staticmethod
    def _profile_safety_scores(
        answer: str,
        user_profile: dict[str, Any],
        forbidden_ingredients: list[str],
    ) -> dict[str, float]:
        allergies = _as_str_list(user_profile.get("allergies", []))
        restrictions = _as_str_list(user_profile.get("dietary_restrictions", []))
        forbidden = [*forbidden_ingredients, *allergies]

        mentioned_forbidden = [item for item in forbidden if item and item in answer]
        return {
            "allergy_violation": 1.0 if any(item in answer for item in allergies) else 0.0,
            "restriction_violation": 1.0 if any(item in answer for item in restrictions) else 0.0,
            "forbidden_ingredient_violation": 1.0 if mentioned_forbidden else 0.0,
        }

    @staticmethod
    def _memory_policy_scores(answer: str, memory_policies: list[str]) -> dict[str, float]:
        context_only_expected = "context_only" in memory_policies
        has_history_reference = "历史问答参考" in answer or "根据历史问答" in answer
        return {
            "context_only_citation_violation": (
                1.0 if context_only_expected and has_history_reference else 0.0
            ),
            "memory_policy_compliance": (
                0.0 if context_only_expected and has_history_reference else 1.0
            ),
        }

    @staticmethod
    def _route_order_scores(
        expected_order: list[str],
        actual_routes: list[str],
    ) -> dict[str, float]:
        """Score how well the actual execution order matches the expected order.

        Uses normalized edit distance (Levenshtein) over the route sequence.
        1.0 = perfect order, 0.0 = completely wrong order.
        """
        if not expected_order:
            return {"route_order_accuracy": 1.0}
        # Only consider the subset of routes that overlap between expected and actual
        seq = [r for r in actual_routes if r in expected_order]
        if not seq:
            return {"route_order_accuracy": 0.0}
        distance = _levenshtein(expected_order, seq)
        max_dist = max(len(expected_order), len(seq))
        return {"route_order_accuracy": 1.0 - (distance / max_dist)}

    @staticmethod
    def _dynamic_discovery_scores(
        expected_dynamic_entities: list[str],
        actual_dynamic_entities: list[str],
        expected_dynamic_routes: list[str],
        actual_dynamic_routes: list[str],
    ) -> dict[str, float]:
        """Evaluate router's dynamic entity discovery and route injection."""
        scores = {}
        # Entity discovery recall
        if expected_dynamic_entities:
            expected_set = set(expected_dynamic_entities)
            actual_set = set(actual_dynamic_entities)
            hits = len(expected_set & actual_set)
            scores["dynamic_entity_recall"] = hits / len(expected_set)
        else:
            scores["dynamic_entity_recall"] = 1.0

        # Entity discovery precision (no spurious entities)
        scores["dynamic_entity_precision"] = (
            1.0 if not actual_dynamic_entities else
            hits / len(actual_dynamic_entities)
        ) if expected_dynamic_entities else (0.0 if actual_dynamic_entities else 1.0)

        # Route injection accuracy
        if expected_dynamic_routes:
            expected_route_set = set(expected_dynamic_routes)
            actual_route_set = set(actual_dynamic_routes)
            route_hits = len(expected_route_set & actual_route_set)
            scores["dynamic_route_recall"] = route_hits / len(expected_route_set)
            scores["dynamic_route_precision"] = (
                route_hits / len(actual_route_set) if actual_route_set else 0.0
            )
        else:
            scores["dynamic_route_recall"] = 1.0
            scores["dynamic_route_precision"] = 0.0 if actual_dynamic_routes else 1.0

        return scores

    @staticmethod
    def _safety_filter_scores(
        expected_warnings: list[str],
        actual_warnings: list[str],
    ) -> dict[str, float]:
        """Evaluate safety filter's recall and precision on flagging forbidden ingredients."""
        if not expected_warnings:
            return {
                "safety_filter_recall": 1.0,
                "safety_filter_precision": 0.0 if actual_warnings else 1.0,
                "safety_filter_false_positive": 1.0 if actual_warnings else 0.0,
            }
        expected_set = set(expected_warnings)
        actual_set = set(actual_warnings)
        hits = len(expected_set & actual_set)
        return {
            "safety_filter_recall": hits / len(expected_set),
            "safety_filter_precision": hits / len(actual_set) if actual_set else 0.0,
            "safety_filter_false_positive": 1.0 if actual_warnings and not hits else 0.0,
        }

    @staticmethod
    def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {}
        metric_names = sorted({name for row in rows for name in row["scores"]})
        result = {}
        for name in metric_names:
            values = [row["scores"][name] for row in rows if name in row["scores"]]
            result[name] = mean(values) if values else 0.0
        result["example_count"] = float(len(rows))
        return result


def lexical_f1(answer: str, reference: str) -> float:
    """Simple token-overlap F1 for deterministic smoke evaluation."""
    if not answer or not reference:
        return 0.0
    answer_tokens = tokenize(answer)
    ref_tokens = tokenize(reference)
    if not answer_tokens or not ref_tokens:
        return 0.0
    answer_counts = Counter(answer_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((answer_counts & ref_counts).values())
    precision = overlap / len(answer_tokens)
    recall = overlap / len(ref_tokens)
    return _f1(precision, recall)


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese and alphanumeric text without external dependencies."""
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Compute Levenshtein (edit) distance between two lists of strings.

    The distance is the minimum number of insertions / deletions / substitutions
    needed to turn sequence *a* into sequence *b*.
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]
