"""RAGAS integration with lazy imports."""

from __future__ import annotations

from typing import Any

from src.evaluation.dataset import EvaluationExample


DEFAULT_RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


class RagasUnavailableError(RuntimeError):
    """Raised when RAGAS dependencies are not installed."""


def evaluate_with_ragas(
    examples: list[EvaluationExample],
    metric_names: list[str] | None = None,
    llm: Any | None = None,
    embeddings: Any | None = None,
) -> dict[str, Any]:
    """Run RAGAS metrics for prepared examples.

    The project keeps this as a lazy adapter because local development and unit
    tests should not require RAGAS, datasets, or LLM judge credentials.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        import ragas.metrics as ragas_metrics
    except Exception as exc:
        raise RagasUnavailableError(
            "RAGAS evaluation requires installed packages: ragas and datasets. "
            "Install project dependencies before running --ragas."
        ) from exc

    metric_names = metric_names or list(DEFAULT_RAGAS_METRICS)
    metrics = []
    for name in metric_names:
        metric = getattr(ragas_metrics, name, None)
        if metric is None:
            raise ValueError(f"Unsupported RAGAS metric for installed version: {name}")
        metrics.append(metric)

    dataset = Dataset.from_list([example.to_ragas_dict() for example in examples])
    kwargs: dict[str, Any] = {"dataset": dataset, "metrics": metrics}
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings
    result = evaluate(**kwargs)
    return _ragas_result_to_dict(result)


def _ragas_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        return {
            "rows": frame.to_dict(orient="records"),
            "aggregate": {
                key: float(value)
                for key, value in frame.mean(numeric_only=True).to_dict().items()
            },
        }
    if isinstance(result, dict):
        return result
    return {"raw": str(result)}
