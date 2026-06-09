"""Multi-turn conversation evaluation helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.evaluation.agent_runner import AgentLike
from src.evaluation.dataset import EvaluationExample
from src.evaluation.metrics import tokenize


def run_multi_turn_session(
    agent: AgentLike,
    group_id: str,
    turns: list[EvaluationExample],
) -> list[EvaluationExample]:
    """Run a sequence of turns as a multi-turn session on a single agent.

    Turns are executed in turn_index order.  Each turn sees the full history
    of preceding (question, answer) pairs, so the agent can rely on its
    conversation memory and context.
    """
    sorted_turns = sorted(turns, key=lambda t: t.turn_index)
    results: list[EvaluationExample] = []
    history: list[dict[str, str]] = []

    for turn in sorted_turns:
        response = agent.run(turn.question, session_history=list(history))
        result = _fill_from_response(turn, response)
        result.multi_turn_group = group_id
        result.turn_index = turn.turn_index
        results.append(result)
        history.append({"role": "user", "content": turn.question})
        history.append({"role": "assistant", "content": result.answer})

    return results


def _fill_from_response(
    gold: EvaluationExample,
    response: dict[str, Any],
) -> EvaluationExample:
    """Fill runtime fields from an agent response, preserving gold labels."""
    return EvaluationExample(
        question=gold.question,
        answer=str(response.get("answer", "")),
        contexts=gold.contexts,
        ground_truth=gold.ground_truth,
        intent=str(response.get("intent", gold.intent)),
        expected_intent=gold.expected_intent,
        expected_routes=gold.expected_routes,
        expected_route_order=gold.expected_route_order,
        actual_routes=response.get("planned_routes", gold.actual_routes),
        expected_context_ids=gold.expected_context_ids,
        retrieved_context_ids=gold.retrieved_context_ids,
        citations=gold.citations,
        user_profile=gold.user_profile,
        forbidden_ingredients=gold.forbidden_ingredients,
        memory_policies=gold.memory_policies,
        expected_safety_warnings=gold.expected_safety_warnings,
        actual_safety_warnings=gold.actual_safety_warnings,
        expected_dynamic_entities=gold.expected_dynamic_entities,
        actual_dynamic_entities=gold.actual_dynamic_entities,
        expected_dynamic_routes=gold.expected_dynamic_routes,
        actual_dynamic_routes=gold.actual_dynamic_routes,
        executed_routes=response.get("executed_routes", []),
        fallback_routes=response.get("fallback_routes", []),
        multi_turn_group=gold.multi_turn_group,
        turn_index=gold.turn_index,
    )


def score_multi_turn_consistency(session_results: list[EvaluationExample]) -> dict[str, float]:
    """Score the consistency of a multi-turn session.

    Detects:
    - pronoun_resolution: whether follow-up turns mention entities from prior turns
    - factual_consistency: no obvious contradictions between turns
    - info_repetition: whether the assistant repeats the same fact verbatim
    """
    if not session_results:
        return {
            "mt_pronoun_resolution": 0.0,
            "mt_factual_consistency": 0.0,
            "mt_info_repetition": 1.0,
        }

    answers = [r.answer for r in session_results]
    questions = [r.question for r in session_results]

    pronoun = _score_pronoun_resolution(answers, questions)
    factual = _score_factual_consistency(answers)
    repetition = _score_info_repetition(answers)

    return {
        "mt_pronoun_resolution": pronoun,
        "mt_factual_consistency": factual,
        "mt_info_repetition": repetition,
    }


def _score_pronoun_resolution(answers: list[str], questions: list[str]) -> float:
    """Check whether follow-up turns (turn >= 1) mention entities from prior turns.

    A crude but effective heuristic: if turn N has a pronoun (那, 这个, 它)
    and turn N-1's answer contains named entities, at least one entity should
    also appear in turn N's answer.
    """
    if len(answers) < 2:
        return 1.0

    pronoun_questions_found = 0
    pronoun_questions_resolved = 0

    for i in range(1, len(questions)):
        # Does this question contain a pronoun / anaphora?
        if not re.search(r"[那这它]", questions[i]):
            continue
        pronoun_questions_found += 1

        # Extract token-level entities from the previous answer
        prev_tokens = {t for t in tokenize(answers[i - 1]) if len(t) >= 2}
        curr_tokens = {t for t in tokenize(answers[i]) if len(t) >= 2}

        # If any token from the previous answer reappears in the current answer,
        # we consider the entity resolved (conservative).
        overlap = prev_tokens & curr_tokens
        if len(overlap) >= 1:
            pronoun_questions_resolved += 1

    if pronoun_questions_found == 0:
        return 1.0
    return pronoun_questions_resolved / pronoun_questions_found


def _score_factual_consistency(answers: list[str]) -> float:
    """Detect obvious contradictions between turns.

    Looks for cases where an assertion like "热量 100 kcal" in one turn
    is directly contradicted by a later "热量 200 kcal".
    """
    if len(answers) < 2:
        return 1.0

    # Extract numeric nutrient claims across all turns
    # The attribute must be >=2 Chinese chars AND not be a common verb/filler
    _SKIP_ATTRS = {"约含", "含有", "大约", "约莫", "约为", "包括", "包含", "分为", "其中", "分为"}
    pattern = re.compile(
        r"([一-鿿]{2,6})\s*"
        r"(?:是|为|含|有|约|大约|约莫|达到|超过|低于)?\s*"
        r"(\d+[\d.]*)\s*(k?cal|g|mg|μg|千卡|大卡|克|毫克|微克)?"
    )

    per_turn: list[dict[str, list[tuple[str, str]]]] = []
    for answer in answers:
        claims: dict[str, list[tuple[str, str]]] = {}
        for match in pattern.finditer(answer):
            attr = match.group(1).strip()
            if attr in _SKIP_ATTRS:
                continue
            value = match.group(2)
            unit = match.group(3) or ""
            claims.setdefault(attr, []).append((value, unit))
        per_turn.append(claims)

    conflicts = 0
    checks = 0
    for i in range(1, len(per_turn)):
        for attr, prev_vals in per_turn[i - 1].items():
            if attr not in per_turn[i]:
                continue
            curr_vals = per_turn[i][attr]
            checks += 1
            # If values for the same nutrient are different, it's a contradiction
            # We only flag if units also match (same measurement scale)
            for pv, pu in prev_vals:
                for cv, cu in curr_vals:
                    if pu == cu and pv != cv:
                        conflicts += 1
                        break

    if checks == 0:
        return 1.0
    return 1.0 - (conflicts / checks)


def _score_info_repetition(answers: list[str]) -> float:
    """Detect whether the assistant repeats the same information verbatim.

    High repetition = low score.  We treat consecutive-turns n-gram overlap
    as a signal of unnecessary repetition.
    """
    if len(answers) < 2:
        return 1.0

    scores = []
    for i in range(1, len(answers)):
        prev_ngrams = _char_ngrams(answers[i - 1], 10)
        curr_ngrams = _char_ngrams(answers[i], 10)
        if not prev_ngrams or not curr_ngrams:
            scores.append(1.0)
            continue
        overlap = len(prev_ngrams & curr_ngrams)
        union = len(prev_ngrams | curr_ngrams)
        jaccard = overlap / union if union else 1.0
        # Jaccard > 0.4 means >40% character reuse → penalise
        scores.append(1.0 - jaccard)

    return sum(scores) / len(scores) if scores else 1.0


def _char_ngrams(text: str, n: int = 10) -> set[str]:
    cleaned = re.sub(r"\s+", "", text)
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}
