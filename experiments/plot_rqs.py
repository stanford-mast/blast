"""
Generate RQ-style plots from evaluate_codegen results.

Inputs:
- experiments/results/<task_id>.json (from evaluate_codegen.py)
- experiments/results/<task_id>.md (for context; not strictly needed here)

Outputs (saved to experiments/results/):
- <task_id>_summary_bars.png
    Single figure with 3 subplots:
        * Min successful generation latency (s)
        * Avg estimated cost (s)
        * Pass rate (%)
    For each metric, 4 bars in the order: [gpt-4.1 with, gpt-4.1 without, gpt-oss-20b with, gpt-oss-20b without].
    No legend; x-axis uses two-level labeling: within each model, bars labeled 'with'/'without'; model name centered below the pair.

Usage:
  python experiments/plot_rqs.py --results-dir experiments/results --id dashdish-deepresearch1 --trials 200
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rich_click as click
import matplotlib
matplotlib.use("Agg")  # Render to files (non-interactive)
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Run:
    model: str
    with_protocol: bool
    generation_latency: float
    estimated_cost: float
    overall_pass: bool


def load_runs(json_path: Path) -> List[Run]:
    data = json.loads(json_path.read_text())
    runs: List[Run] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if "config" not in entry or "generation_latency" not in entry:
            continue
        cfg = entry.get("config", {})
        codecheck = entry.get("codecheck", {})
        runs.append(
            Run(
                model=str(cfg.get("model", "unknown")),
                with_protocol=bool(cfg.get("with_protocol", False)),
                generation_latency=float(entry.get("generation_latency", math.nan)),
                estimated_cost=float(entry.get("estimated_cost", math.nan)),
                overall_pass=bool(codecheck.get("overall_pass", False)),
            )
        )
    return runs


def group_by_config(runs: List[Run]) -> Dict[Tuple[str, bool], List[Run]]:
    from collections import defaultdict
    groups: Dict[Tuple[str, bool], List[Run]] = defaultdict(list)
    for r in runs:
        groups[(r.model, r.with_protocol)].append(r)
    return groups


def compute_config_metrics(runs: List[Run]) -> Dict[str, Optional[float]]:
    # min successful generation latency, avg estimated cost, pass rate
    if not runs:
        return {"min_success_latency": None, "avg_estimated_cost": None, "pass_rate": None}
    success_lats = [r.generation_latency for r in runs if r.overall_pass and not math.isnan(r.generation_latency)]
    min_success = float(min(success_lats)) if success_lats else None
    avg_est_cost = float(np.mean([r.estimated_cost for r in runs if not math.isnan(r.estimated_cost)])) if runs else None
    pass_rate = (sum(1 for r in runs if r.overall_pass) / len(runs)) if runs else None
    return {"min_success_latency": min_success, "avg_estimated_cost": avg_est_cost, "pass_rate": pass_rate}


def is_big_model(model: str) -> bool:
    # Heuristic mapping: Treat GPT-4.1 as big; gpt-oss-20b as small.
    model_l = model.lower()
    if "gpt-4.1" in model_l:
        return True
    return False


def summarize_metrics(runs: List[Run]) -> Dict[str, Optional[float]]:
    # Compute metrics for a set of runs (one config)
    if not runs:
        return {k: None for k in [
            "min_success_latency",
            "avg_latency",
            "accuracy",
            "avg_estimated_cost",
            "est_cost_of_fastest_success",
        ]}

    latencies = [r.generation_latency for r in runs if not math.isnan(r.generation_latency)]
    avg_latency = float(np.mean(latencies)) if latencies else None

    success_runs = [r for r in runs if r.overall_pass and not math.isnan(r.generation_latency)]
    min_success_latency = float(min(r.generation_latency for r in success_runs)) if success_runs else None
    accuracy = (sum(1 for r in runs if r.overall_pass) / len(runs)) if runs else None
    avg_estimated_cost = float(np.mean([r.estimated_cost for r in runs if not math.isnan(r.estimated_cost)])) if runs else None

    if success_runs:
        fastest = min(success_runs, key=lambda r: r.generation_latency)
        est_cost_fastest = fastest.estimated_cost if not math.isnan(fastest.estimated_cost) else None
    else:
        est_cost_fastest = None

    return {
        "min_success_latency": min_success_latency,
        "avg_latency": avg_latency,
        "accuracy": accuracy,
        "avg_estimated_cost": avg_estimated_cost,
        "est_cost_of_fastest_success": est_cost_fastest,
    }


@click.command()
@click.option('--results-dir', type=click.Path(exists=True), default='experiments/results', help='Directory with <task_id>.json and .md')
@click.option('--id', 'task_id', type=str, required=True, help='Task id (e.g., dashdish-deepresearch1)')
@click.option('--show/--no-show', default=False, help='Also display plots (if environment allows)')
def main(results_dir: str, task_id: str, show: bool):
    results_dir_p = Path(results_dir)
    json_path = results_dir_p / f"{task_id}.json"
    md_path = results_dir_p / f"{task_id}.md"  # not required but checked for existence

    if not json_path.exists():
        raise click.ClickException(f"Missing results JSON: {json_path}")
    if not md_path.exists():
        click.echo(f"[warn] Markdown file missing: {md_path} (continuing)")

    runs = load_runs(json_path)
    groups = group_by_config(runs)

    # ===== Single figure: three subplots, four bars each (gpt-4.1 with/without, gpt-oss-20b with/without) =====
    # Prepare metrics per model/protocol
    order = [
        ("gpt-4.1", True), ("gpt-4.1", False),
        ("openai/gpt-oss-20b", True), ("openai/gpt-oss-20b", False),
    ]

    # Aggregate runs by config string equality (exact match)
    by_cfg = group_by_config(runs)
    metrics = {cfg: compute_config_metrics(rs) for cfg, rs in by_cfg.items()}

    # Build arrays in required order
    def val_or_nan(model: str, with_p: bool, key: str):
        m = metrics.get((model, with_p))
        v = m.get(key) if m else None
        return v if v is not None else np.nan

    mins = [val_or_nan(m, w, "min_success_latency") for m, w in order]
    avg_costs = [val_or_nan(m, w, "avg_estimated_cost") for m, w in order]
    pass_rates = [val_or_nan(m, w, "pass_rate") * 100.0 if not math.isnan(val_or_nan(m, w, "pass_rate")) else np.nan for m, w in order]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ["Min successful latency (s)", "Avg estimated cost (s)", "Pass rate (%)"]
    datasets = [mins, avg_costs, pass_rates]

    x = np.arange(4)
    width = 0.6

    for ax, title, data in zip(axes, titles, datasets):
        bars = ax.bar(x, data, width, color=["#4C78A8", "#9ECae9", "#F58518", "#FFBF79"])  # alternating colors
        # Mark missing with X
        for xi, yi in zip(x, data):
            if np.isnan(yi):
                ax.text(xi, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, 'X', ha='center', va='bottom', color='red', fontsize=12)
        ax.set_title(title)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.set_xticks(x)
        # Two-level labels: first line with/without, second line model name
        lvl1 = ["protocol ✓", "protocol ✗", "protocol ✓", "protocol ✗"]
        lvl2 = ["gpt-4.1", "gpt-4.1", "gpt-oss-20b", "gpt-oss-20b"]
        labels = [f"{a}\n{b}" for a, b in zip(lvl1, lvl2)]
        ax.set_xticklabels(labels, rotation=0)

    fig.suptitle(f"Summary metrics (task={task_id})")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out = results_dir_p / f"{task_id}_summary_bars.png"
    fig.savefig(out, dpi=150)

    if show:
        plt.show()

    click.echo(f"Saved: {out}")


if __name__ == '__main__':
    main()
