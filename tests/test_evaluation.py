import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.dataset import EvaluationExample, load_jsonl
from src.evaluation.agent_runner import example_from_agent_response
from src.evaluation.metrics import DeterministicMetricSuite, lexical_f1
from src.evaluation.runner import EvaluationRunner


def test_evaluation_example_accepts_common_aliases():
    example = EvaluationExample.from_dict({
        "query": "番茄有什么营养？",
        "answer": "番茄含有维生素C。",
        "reference": "番茄富含维生素C。",
        "retrieved_contexts": ["番茄含维生素C。"],
        "citations": "1",
        "expected_intent": "nutrition_info",
        "expected_routes": ["semantic"],
        "planned_routes": ["semantic", "graphrag"],
        "forbidden_ingredients": ["花生"],
        "memory_policies": ["context_only"],
    })

    assert example.question == "番茄有什么营养？"
    assert example.ground_truth == "番茄富含维生素C。"
    assert example.contexts == ["番茄含维生素C。"]
    assert example.citations == [1]
    assert example.expected_intent == "nutrition_info"
    assert example.expected_routes == ["semantic"]
    assert example.actual_routes == ["semantic", "graphrag"]
    assert example.forbidden_ingredients == ["花生"]
    assert example.memory_policies == ["context_only"]
    assert example.to_ragas_dict()["reference"] == "番茄富含维生素C。"


def test_load_jsonl_reports_valid_examples(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({
            "question": "低盐饮食怎么吃？",
            "answer": "少用盐，多用天然香辛料。",
            "ground_truth": "低盐饮食应减少食盐。",
        }, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    examples = load_jsonl(path)

    assert len(examples) == 1
    assert examples[0].question == "低盐饮食怎么吃？"


def test_deterministic_metrics_cover_retrieval_citation_and_safety():
    example = EvaluationExample(
        question="痛风怎么吃？",
        answer="痛风饮食要控制嘌呤，建议咨询医生或营养师。",
        contexts=["痛风患者应限制高嘌呤食物。"],
        ground_truth="痛风患者应限制高嘌呤食物。",
        intent="disease_diet",
        expected_intent="disease_diet",
        expected_routes=["semantic", "graphrag"],
        actual_routes=["semantic"],
        expected_context_ids=["c1", "c2"],
        retrieved_context_ids=["c1", "c3"],
        citations=[1, 3],
        user_profile={"allergies": ["花生"]},
        forbidden_ingredients=["花生"],
        memory_policies=["context_only"],
    )

    result = DeterministicMetricSuite().score_example(example)
    scores = result["scores"]

    assert scores["retrieval_precision"] == 0.5
    assert scores["retrieval_recall"] == 0.5
    assert scores["citation_validity"] == 0.5
    assert scores["citation_coverage"] == 1.0
    assert scores["safety_notice"] == 1.0
    assert scores["intent_accuracy"] == 1.0
    assert scores["route_recall"] == 0.5
    assert scores["allergy_violation"] == 0.0
    assert scores["memory_policy_compliance"] == 1.0
    assert scores["answer_reference_f1"] > 0


def test_runner_aggregates_deterministic_scores():
    examples = [
        EvaluationExample(
            question="番茄营养",
            answer="番茄含维生素C。",
            contexts=["番茄含维生素C。"],
            ground_truth="番茄含维生素C。",
            expected_context_ids=["c1"],
            retrieved_context_ids=["c1"],
            citations=[1],
        )
    ]

    result = EvaluationRunner().evaluate(examples)

    assert result.aggregate["example_count"] == 1.0
    assert result.aggregate["retrieval_recall"] == 1.0
    assert result.ragas == {}


def test_lexical_f1_handles_empty_inputs():
    assert lexical_f1("", "参考") == 0.0
    assert lexical_f1("番茄", "番茄") == 1.0


def test_profile_and_memory_policy_violations_are_flagged():
    example = EvaluationExample(
        question="我花生过敏，早餐吃什么？",
        answer="可以吃花生燕麦粥。根据历史问答参考。",
        contexts=["早餐可以选择燕麦。"],
        user_profile={"allergies": ["花生"]},
        forbidden_ingredients=["花生"],
        memory_policies=["context_only"],
    )

    scores = DeterministicMetricSuite().score_example(example)["scores"]

    assert scores["allergy_violation"] == 1.0
    assert scores["forbidden_ingredient_violation"] == 1.0
    assert scores["context_only_citation_violation"] == 1.0
    assert scores["memory_policy_compliance"] == 0.0


def test_agent_response_can_fill_runtime_evaluation_fields():
    gold = EvaluationExample(
        question="番茄有什么营养？",
        answer="",
        ground_truth="番茄含维生素C。",
        expected_intent="nutrition_info",
    )
    response = {
        "answer": "番茄含维生素C。[来源1]",
        "intent": "nutrition_info",
        "planned_routes": ["semantic"],
        "citations": [{"source_id": 1, "content": "番茄含维生素C。"}],
        "entities": [{"name": "番茄"}],
    }

    example = example_from_agent_response(gold, response)

    assert example.answer.startswith("番茄")
    assert example.intent == "nutrition_info"
    assert example.actual_routes == ["semantic"]
    assert example.contexts == ["番茄含维生素C。"]
    assert example.citations == [1]
