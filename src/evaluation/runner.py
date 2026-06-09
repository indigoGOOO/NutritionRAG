"""Evaluation runner combining deterministic metrics, optional RAGAS, and A/B comparison."""

from __future__ import annotations

import copy
from statistics import mean
from typing import Any

from src.evaluation.dataset import EvaluationExample, EvaluationResult
from src.evaluation.metrics import DeterministicMetricSuite
from src.evaluation.ragas_adapter import evaluate_with_ragas


class EvaluationRunner:
    """Run the multi-layer evaluation suite."""

    def __init__(self, deterministic_suite: DeterministicMetricSuite | None = None):
        self.deterministic_suite = deterministic_suite or DeterministicMetricSuite()

    def evaluate(
        self,
        examples: list[EvaluationExample],
        include_ragas: bool = False,
        ragas_metrics: list[str] | None = None,
        ragas_llm=None,
        ragas_embeddings=None,
    ) -> EvaluationResult:
        result = self.deterministic_suite.score_dataset(examples)
        if include_ragas:
            result.ragas = evaluate_with_ragas(
                examples,
                ragas_metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )
        return result

    def compare(
        self,
        baseline: EvaluationResult,
        experiment: EvaluationResult,
    ) -> EvaluationDiff:
        """Compare two evaluation results and produce a diff."""
        return EvaluationDiff.from_results(baseline, experiment)


class EvaluationDiff:
    """A/B comparison between two evaluation runs."""

    def __init__(
        self,
        baseline_aggregate: dict[str, float],
        experiment_aggregate: dict[str, float],
        diff: dict[str, float],
        metric_names: list[str],
    ):
        self.baseline_aggregate = baseline_aggregate
        self.experiment_aggregate = experiment_aggregate
        self.diff = diff
        self.metric_names = metric_names

    @classmethod
    def from_results(cls, baseline: EvaluationResult, experiment: EvaluationResult) -> "EvaluationDiff":
        baseline_agg = baseline.aggregate
        experiment_agg = experiment.aggregate
        all_metric_names = sorted(
            set(baseline_agg.keys()) | set(experiment_agg.keys())
        )
        all_metric_names = [m for m in all_metric_names if m != "example_count"]

        diff = {}
        for name in all_metric_names:
            b = baseline_agg.get(name, 0.0)
            e = experiment_agg.get(name, 0.0)
            diff[name] = round(e - b, 4)

        return cls(
            baseline_aggregate=baseline_agg,
            experiment_aggregate=experiment_agg,
            diff=diff,
            metric_names=all_metric_names,
        )

    def improved(self) -> dict[str, float]:
        """Metrics that improved (positive diff)."""
        return {k: v for k, v in self.diff.items() if v > 0}

    def regressed(self) -> dict[str, float]:
        """Metrics that regressed (negative diff)."""
        return {k: v for k, v in self.diff.items() if v < 0}

    def unchanged(self) -> dict[str, float]:
        """Metrics that stayed the same."""
        return {k: v for k, v in self.diff.items() if v == 0}

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_aggregate": self.baseline_aggregate,
            "experiment_aggregate": self.experiment_aggregate,
            "diff": self.diff,
            "metric_names": self.metric_names,
        }
