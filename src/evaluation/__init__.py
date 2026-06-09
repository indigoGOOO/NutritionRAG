"""Evaluation layer for the nutrition RAG system."""

from src.evaluation.dataset import EvaluationExample, EvaluationResult, load_jsonl
from src.evaluation.metrics import DeterministicMetricSuite
from src.evaluation.agent_runner import example_from_agent_response, run_agent_examples
from src.evaluation.runner import EvaluationDiff, EvaluationRunner

__all__ = [
    "DeterministicMetricSuite",
    "EvaluationDiff",
    "EvaluationExample",
    "EvaluationResult",
    "EvaluationRunner",
    "example_from_agent_response",
    "load_jsonl",
    "run_agent_examples",
]
