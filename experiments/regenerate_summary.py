#!/usr/bin/env python3
"""
Regenerate summary.json from existing per-task JSON result files.

Usage:
    python experiments/regenerate_summary.py \
        --results-dir experiments/results \
        --tasks "dashdish-custom-1 dashdish-custom-2 dashdish-custom-3"
    
    # Or auto-detect from existing JSON files:
    python experiments/regenerate_summary.py --results-dir experiments/results
"""

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import click


@dataclass
class CodeCheckResult:
    """Minimal codecheck result for summary computation."""

    overall_pass: bool
    failure_types: List[str]
    overall_pass_fraction: float = 0.0


@dataclass
class CodegenResultData:
    """Minimal codegen result for summary computation."""

    model: str
    with_protocol: bool
    generation_latency: float
    estimated_cost: float
    codecheck: CodeCheckResult
    actual_latency: Optional[float] = None
    generation_failed: bool = False


def load_results_from_json(json_path: Path) -> List[CodegenResultData]:
    """Load results from a per-task JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    results = []
    for item in data:
        config = item.get("config", {})
        codecheck_data = item.get("codecheck", {})

        codecheck = CodeCheckResult(
            overall_pass=codecheck_data.get("overall_pass", False),
            failure_types=codecheck_data.get("failure_types", []) or [],
            overall_pass_fraction=codecheck_data.get("overall_pass_fraction", 0.0),
        )

        result = CodegenResultData(
            model=config.get("model", "unknown"),
            with_protocol=config.get("with_protocol", True),
            generation_latency=item.get("generation_latency", 0.0),
            estimated_cost=item.get("estimated_cost", 0.0),
            codecheck=codecheck,
            actual_latency=item.get("actual_latency"),
            generation_failed=item.get("generation_failed", False),
        )
        results.append(result)

    return results


def compute_summary_statistics(
    all_results: Dict[str, List[CodegenResultData]],
) -> Dict[str, Any]:
    """
    Compute averaged metrics for each unique config.

    Mirrors the logic in evaluate_codegen.py's compute_summary_statistics.
    """
    # Group results by config
    config_groups = defaultdict(list)

    for task_id, codegen_results in all_results.items():
        for result in codegen_results:
            config_key = (result.model, result.with_protocol)
            config_groups[config_key].append(result)

    # Compute averages
    summary = {}

    for config_key, results in config_groups.items():
        model, with_protocol = config_key

        # Calculate averages
        avg_gen_latency = statistics.mean(r.generation_latency for r in results)
        avg_est_cost = statistics.mean(r.estimated_cost for r in results)

        # Variance metrics
        var_gen_latency = (
            statistics.variance(r.generation_latency for r in results)
            if len(results) > 1
            else 0
        )
        var_est_cost = (
            statistics.variance(r.estimated_cost for r in results)
            if len(results) > 1
            else 0
        )

        # Pass/failure rates
        num_runs_cfg = len(results)
        overall_pass_rate = (
            sum(1 for r in results if r.codecheck.overall_pass) / num_runs_cfg
        )

        failure_counts: Dict[str, int] = {}
        for r in results:
            for ft in r.codecheck.failure_types or []:
                failure_counts[ft] = failure_counts.get(ft, 0) + 1

        failure_rates = {k: v / num_runs_cfg for k, v in failure_counts.items()}
        num_failed_runs = sum(1 for r in results if not r.codecheck.overall_pass)
        failure_rates_among_failures = {
            k: (v / num_failed_runs) if num_failed_runs else 0.0
            for k, v in failure_counts.items()
        }

        # Metrics among passing candidates only
        passing_results = [r for r in results if r.codecheck.overall_pass]
        avg_est_cost_passing = (
            statistics.mean(r.estimated_cost for r in passing_results)
            if passing_results
            else None
        )
        min_gen_latency_passing = min(
            (r.generation_latency for r in passing_results), default=None
        )
        min_est_cost_passing = min(
            (r.estimated_cost for r in passing_results), default=None
        )
        max_est_cost_passing = max(
            (r.estimated_cost for r in passing_results), default=None
        )

        # Actual latency (only for runs that measured it)
        actual_latencies = [
            r.actual_latency for r in results if r.actual_latency is not None
        ]
        avg_actual_latency = (
            statistics.mean(actual_latencies) if actual_latencies else None
        )

        generation_failed_count = sum(
            1
            for r in results
            if r.generation_failed or ("generation" in r.codecheck.failure_types)
        )

        summary[f"{model}|with_protocol={with_protocol}"] = {
            "model": model,
            "with_protocol": with_protocol,
            "num_runs": len(results),
            "num_generation_failures": generation_failed_count,
            "avg_generation_latency": avg_gen_latency,
            "var_generation_latency": var_gen_latency,
            "avg_estimated_cost": avg_est_cost,
            "var_estimated_cost": var_est_cost,
            "overall_pass_rate": overall_pass_rate,
            "failure_rates": failure_rates,
            "failure_rates_among_failures": failure_rates_among_failures,
            "avg_actual_latency": avg_actual_latency,
            "num_passing": len(passing_results),
            "avg_est_cost_passing": avg_est_cost_passing,
            "min_gen_latency_passing": min_gen_latency_passing,
            "min_est_cost_passing": min_est_cost_passing,
            "max_est_cost_passing": max_est_cost_passing,
        }

    return summary


@click.command()
@click.option(
    "--results-dir",
    type=click.Path(exists=True),
    default="experiments/results",
    help="Directory containing per-task JSON files",
)
@click.option(
    "--tasks",
    type=str,
    default=None,
    help="Space-separated task IDs (e.g., 'dashdish-custom-1 dashdish-custom-2'). If not provided, auto-detects from JSON files.",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help="Output file path (default: {results-dir}/summary.json)",
)
def main(results_dir: str, tasks: Optional[str], output: Optional[str]):
    """Regenerate summary.json from existing per-task JSON result files."""
    results_path = Path(results_dir)

    # Determine task IDs
    if tasks:
        task_ids = tasks.split()
    else:
        # Auto-detect from JSON files (exclude summary*.json and timestamped files)
        json_files = list(results_path.glob("*.json"))
        task_ids = []
        for f in json_files:
            name = f.stem
            # Skip summary files, timestamped files, and e2e_detailed files
            if name.startswith("summary") or "_2026" in name or "_e2e_" in name:
                continue
            task_ids.append(name)
        task_ids = sorted(set(task_ids))

    print(f"Found {len(task_ids)} tasks: {task_ids}")

    # Load all results
    all_results: Dict[str, List[CodegenResultData]] = {}

    for task_id in task_ids:
        json_file = results_path / f"{task_id}.json"
        if not json_file.exists():
            print(f"Warning: {json_file} not found, skipping")
            continue

        print(f"Loading: {json_file}")
        results = load_results_from_json(json_file)
        all_results[task_id] = results
        print(f"  - {len(results)} results")

    if not all_results:
        print("Error: No results loaded")
        return

    # Compute summary
    print("\nComputing summary statistics...")
    summary = compute_summary_statistics(all_results)

    # Save summary
    output_file = Path(output) if output else results_path / "summary.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {output_file}")

    # Print summary table
    print("\nSummary:")
    print("-" * 80)
    for config_name, metrics in sorted(summary.items()):
        model = metrics["model"]
        with_proto = "w/proto" if metrics["with_protocol"] else "no-proto"
        pass_rate = metrics["overall_pass_rate"]
        num_runs = metrics["num_runs"]
        failure_rates = metrics.get("failure_rates", {})

        failure_str = (
            ", ".join(f"{k}:{v:.1%}" for k, v in failure_rates.items())
            if failure_rates
            else "none"
        )

        print(
            f"{model:25s} {with_proto:8s} | pass: {pass_rate:6.1%} | runs: {num_runs:4d} | failures: {failure_str}"
        )


if __name__ == "__main__":
    main()
