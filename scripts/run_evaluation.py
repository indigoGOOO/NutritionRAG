"""Run offline RAG evaluation from a JSONL dataset.

Examples:
    python scripts/run_evaluation.py --input data/eval/sample.jsonl
    python scripts/run_evaluation.py --input data/eval/sample.jsonl --ragas
    python scripts/run_evaluation.py --compare baseline.json experiment.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.dataset import load_jsonl, save_json
from src.evaluation.ragas_adapter import RagasUnavailableError
from src.evaluation.runner import EvaluationRunner


def main():
    parser = argparse.ArgumentParser(description="Run nutrition RAG evaluation.")
    parser.add_argument("--input", default="", help="Evaluation JSONL path.")
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    parser.add_argument("--ragas", action="store_true", help="Run RAGAS metrics as well.")
    parser.add_argument(
        "--ragas-metrics",
        default="",
        help="Comma-separated RAGAS metric names. Defaults to the adapter list.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "EXPERIMENT"),
        default=None,
        help="Compare two existing evaluation result JSON files and print diff.",
    )
    args = parser.parse_args()

    if args.compare:
        _do_compare(args.compare[0], args.compare[1])
        return

    if not args.input:
        parser.error("Either --input or --compare is required.")

    examples = load_jsonl(args.input)
    metrics = [m.strip() for m in args.ragas_metrics.split(",") if m.strip()] or None
    runner = EvaluationRunner()

    try:
        result = runner.evaluate(examples, include_ragas=args.ragas, ragas_metrics=metrics)
    except RagasUnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    output = result.to_dict()

    if args.output:
        save_json(args.output, result)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _do_compare(baseline_path: str, experiment_path: str):
    """Load two result JSONs and print A/B diff."""
    from src.evaluation.runner import EvaluationDiff

    def load_result(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    baseline_raw = load_result(baseline_path)
    experiment_raw = load_result(experiment_path)

    # Build EvaluationDiff from the aggregate fields
    diff = EvaluationDiff(
        baseline_aggregate=baseline_raw.get("aggregate", {}),
        experiment_aggregate=experiment_raw.get("aggregate", {}),
        diff={},
        metric_names=[],
    )
    all_metric_names = sorted(
        set(baseline_raw.get("aggregate", {}).keys())
        | set(experiment_raw.get("aggregate", {}).keys())
    )
    all_metric_names = [m for m in all_metric_names if m != "example_count"]

    diff_lines = []
    for name in all_metric_names:
        b = baseline_raw.get("aggregate", {}).get(name, 0.0)
        e = experiment_raw.get("aggregate", {}).get(name, 0.0)
        delta = e - b
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        diff_lines.append(f"  {arrow} {name}: {b:.3f} → {e:.3f} ({delta:+.3f})")

    improved = [l for l in diff_lines if "▲" in l]
    regressed = [l for l in diff_lines if "▼" in l]
    unchanged = [l for l in diff_lines if "─" in l]

    print("=" * 60)
    print("A/B comparison: baseline vs experiment")
    print(f"  baseline:   {baseline_path}")
    print(f"  experiment: {experiment_path}")
    print("=" * 60)
    if improved:
        print(f"\n  ▲ Improved ({len(improved)}):")
        for l in improved:
            print(l)
    if regressed:
        print(f"\n  ▼ Regressed ({len(regressed)}):")
        for l in regressed:
            print(l)
    if unchanged:
        print(f"\n  ─ Unchanged ({len(unchanged)}):")
        for l in unchanged:
            print(l)

    # Also include RAGAS comparison if available
    baseline_ragas = baseline_raw.get("ragas", {}).get("aggregate", {})
    experiment_ragas = experiment_raw.get("ragas", {}).get("aggregate", {})
    if baseline_ragas or experiment_ragas:
        all_ragas = sorted(set(baseline_ragas.keys()) | set(experiment_ragas.keys()))
        print(f"\n  RAGAS metrics:")
        for name in all_ragas:
            b = baseline_ragas.get(name, 0.0)
            e = experiment_ragas.get(name, 0.0)
            delta = e - b
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
            print(f"  {arrow} {name}: {b:.3f} → {e:.3f} ({delta:+.3f})")

    print("=" * 60)


if __name__ == "__main__":
    main()
