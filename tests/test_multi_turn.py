"""Multi-turn conversation stability tests.

These tests validate the multi-turn evaluation helpers with mock data.
They do NOT require a running agent or database — they test the scoring
logic that you would later use in an end-to-end multi-turn session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.dataset import EvaluationExample
from src.evaluation.multi_turn import (
    _fill_from_response,
    _score_pronoun_resolution,
    _score_factual_consistency,
    _score_info_repetition,
    run_multi_turn_session,
    score_multi_turn_consistency,
)


# ==============================================================
#  Scenario 1 — follow-up with pronoun reference
# ==============================================================


def test_pronoun_resolution_returns_1_when_follow_up_mentions_prior_entity():
    """Turn 2 says '那和橙子比呢' — Turn 2 answer should contain '番茄' from T1."""
    t1 = EvaluationExample(
        question="番茄的维C含量是多少？",
        answer="番茄每100g约含20mg维生素C。",
        turn_index=0,
    )
    t2 = EvaluationExample(
        question="那和橙子比呢？",
        answer="橙子每100g约含53mg维C，比番茄的20mg更高。",
        turn_index=1,
    )
    scores = score_multi_turn_consistency([t1, t2])
    assert scores["mt_pronoun_resolution"] == 1.0
    assert scores["mt_factual_consistency"] == 1.0


def test_pronoun_resolution_penalises_when_entity_is_dropped():
    """Follow-up contains a pronoun but answer forgets the prior entity entirely."""
    t1 = EvaluationExample(
        question="番茄的维C含量是多少？",
        answer="番茄每100g约含20mg维生素C。",
        turn_index=0,
    )
    t2 = EvaluationExample(
        question="那和橙子比呢？",
        answer="苹果也富含膳食纤维。",  #  Wrong — "那" refers to 番茄, not mentioned
        turn_index=1,
    )
    scores = score_multi_turn_consistency([t1, t2])
    # No token overlap between "番茄每100g约含20mg维生素C" and
    # "苹果也富含膳食纤维" → fails resolution
    assert scores["mt_pronoun_resolution"] == 0.0


# ==============================================================
#  Scenario 2 — factual consistency (no contradictions)
# ==============================================================


def test_factual_consistency_penalises_numeric_contradictions():
    """Same nutrient, same unit, different value across turns = contradiction."""
    t1 = EvaluationExample(
        question="番茄的热量是多少？",
        answer="番茄每100g热量为18kcal。",
        turn_index=0,
    )
    t2 = EvaluationExample(
        question="确认一下，番茄热量到底多少？",
        answer="番茄每100g热量为25kcal。",  # Contradicts T1
        turn_index=1,
    )
    scores = score_multi_turn_consistency([t1, t2])
    assert scores["mt_factual_consistency"] < 1.0


def test_factual_consistency_allows_different_units():
    """Same nutrient but different units is not a contradiction (may be per 100g vs per serving)."""
    t1 = EvaluationExample(
        question="番茄的热量是多少？",
        answer="番茄每100g热量为18kcal。",
        turn_index=0,
    )
    t2 = EvaluationExample(
        question="那一整个番茄呢？",
        answer="一个中等番茄约含25kcal热量。",  # Different unit (per-fruit vs per-100g)
        turn_index=1,
    )
    scores = score_multi_turn_consistency([t1, t2])
    # No contradiction if values differ (can't compare kcal vs kcal directly without knowing
    # the weight of a fruit) — but our heuristic only flags when units match
    assert scores["mt_factual_consistency"] == 1.0


# ==============================================================
#  Scenario 3 — info repetition
# ==============================================================


def test_info_repetition_penalises_verbatim_restatement():
    """If two consecutive turns share >40% char n-grams, that's repetition."""
    t1 = EvaluationExample(
        question="番茄有什么营养？",
        answer="番茄富含维生素C和膳食纤维，每100g约含20mg维C。",
        turn_index=0,
    )
    t2 = EvaluationExample(
        question="那番茄还有什么营养？",
        answer="番茄富含维生素C和膳食纤维，每100g约含20mg维C。还有钾元素。",
        turn_index=1,
    )
    scores = score_multi_turn_consistency([t1, t2])
    # t2 starts with a near-verbatim copy of t1 → high overlap → low score
    assert scores["mt_info_repetition"] < 1.0


def test_info_repetition_allows_fresh_info():
    """If consecutive turns cover different info, the score stays high."""
    t1 = EvaluationExample(
        question="番茄有什么营养？",
        answer="番茄富含维生素C，每100g约含20mg。",
        turn_index=0,
    )
    t2 = EvaluationExample(
        question="番茄适合和什么搭配？",
        answer="番茄和鸡蛋相宜，可以炒食或做汤。",
        turn_index=1,
    )
    scores = score_multi_turn_consistency([t1, t2])
    assert scores["mt_info_repetition"] >= 0.8


# ==============================================================
#  Scenario 4 — profile memory across turns
# ==============================================================


def test_profile_memory_should_avoid_allergen_in_follow_up():
    """Turn 1 records an allergy, Turn 2 should not recommend that ingredient.

    This test validates that the evaluation framework CAN check profile memory
    cross-turn — but it relies on the agent actually loading profile context.
    Here we only verify the scoring function with mock data.
    """
    t1 = EvaluationExample(
        question="我花生过敏，帮我记一下",
        answer="好的，已记录花生过敏。",
        turn_index=0,
        user_profile={"allergies": ["花生"]},
        forbidden_ingredients=["花生"],
    )
    t2 = EvaluationExample(
        question="早餐吃什么好？",
        answer="可以吃花生燕麦粥。",  #  Violation — answer includes 花生
        turn_index=1,
        user_profile={"allergies": ["花生"]},
        forbidden_ingredients=["花生"],
    )

    # The metric suite already has allergy_violation for single examples.
    # Here we additionally verify that the same check works across turns
    # by running the profile_safety check on t2.
    from src.evaluation.metrics import DeterministicMetricSuite

    suite = DeterministicMetricSuite()
    scores = suite.score_example(t2)["scores"]
    assert scores["allergy_violation"] == 1.0


# ==============================================================
#  Scenario 5 — _fill_from_response preserves gold labels
# ==============================================================


def test_fill_from_response_preserves_gold():
    gold = EvaluationExample(
        question="番茄的热量？",
        answer="",
        ground_truth="每100g约18kcal。",
        expected_intent="nutrition_fact",
        turn_index=0,
        multi_turn_group="test-group",
    )
    response = {
        "answer": "每100g番茄约18kcal。[来源1]",
        "intent": "nutrition_fact",
        "planned_routes": ["text2sql"],
        "executed_routes": ["text2sql"],
    }
    result = _fill_from_response(gold, response)
    assert result.question == "番茄的热量？"
    assert result.ground_truth == gold.ground_truth
    assert result.expected_intent == "nutrition_fact"
    assert result.turn_index == 0
    assert result.multi_turn_group == "test-group"
    assert "18kcal" in result.answer
    assert result.executed_routes == ["text2sql"]
