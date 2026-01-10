#!/usr/bin/env python3
"""
Merge multiple evaluation result files into a single file.

Usage:
    python experiments/merge_results.py \
        experiments/results/dashdish-custom1_20260108_*.json \
        --output experiments/results/dashdish-custom1.json
"""

import json
import sys
from glob import glob
from pathlib import Path
from typing import List

import click


def merge_json_results(input_files: List[str], output_file: str):
    """Merge multiple JSON result files into one."""
    merged_results = []
    seen_configs = set()  # Track (model, run_id) to avoid duplicates

    for pattern in input_files:
        # Expand glob patterns
        files = glob(pattern) if "*" in pattern else [pattern]

        for file_path in files:
            path = Path(file_path)
            if not path.exists():
                print(f"Warning: File not found: {file_path}")
                continue

            print(f"Reading: {file_path}")
            with open(path) as f:
                results = json.load(f)

            for result in results:
                # Create a unique key for deduplication
                model = result.get("config", {}).get("model", "unknown")
                run_key = f"{model}_{result.get('generation_latency', 0):.6f}"

                if run_key not in seen_configs:
                    seen_configs.add(run_key)
                    merged_results.append(result)
                else:
                    print(f"  Skipping duplicate: {model}")

    # Sort by model name for consistent ordering
    merged_results.sort(key=lambda x: x.get("config", {}).get("model", ""))

    # Save merged results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(merged_results, f, indent=2)

    print(f"\nMerged {len(merged_results)} results into: {output_file}")

    # Print summary by model
    model_counts = {}
    for result in merged_results:
        model = result.get("config", {}).get("model", "unknown")
        model_counts[model] = model_counts.get(model, 0) + 1

    print("\nResults per model:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}")


def merge_markdown_results(input_files: List[str], output_file: str):
    """Merge multiple Markdown result files into one, renumbering runs sequentially."""
    import re
    
    header = None
    all_runs = []  # List of (original_run_num, content_between_runs)
    current_run_counter = 0

    for pattern in input_files:
        files = glob(pattern) if "*" in pattern else [pattern]

        for file_path in sorted(files):
            path = Path(file_path)
            if not path.exists():
                continue

            print(f"Reading: {file_path}")
            with open(path) as f:
                content = f.read()

            # Extract header from first file
            if header is None:
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("## Run"):
                        header = "\n".join(lines[:i])
                        content = "\n".join(lines[i:])
                        break
            else:
                # Skip header for subsequent files
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("## Run"):
                        content = "\n".join(lines[i:])
                        break

            # Split content by run headers and renumber
            run_pattern = re.compile(r"^## Run\s+(\d+)\s*$", re.MULTILINE)
            matches = list(run_pattern.finditer(content))
            
            for i, match in enumerate(matches):
                current_run_counter += 1
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
                
                # Get the run content and replace the run number
                run_content = content[start:end]
                run_content = re.sub(
                    r"^## Run\s+\d+\s*$", 
                    f"## Run {current_run_counter}", 
                    run_content, 
                    count=1, 
                    flags=re.MULTILINE
                )
                all_runs.append(run_content)

    # Save merged markdown
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        if header:
            f.write(header)
        f.write("".join(all_runs))

    print(f"Merged {current_run_counter} runs into: {output_file}")


def merge_summary_results(input_files: List[str], output_file: str):
    """Merge multiple summary JSON files into one, aggregating statistics properly."""
    from collections import defaultdict
    
    # Accumulate data per config key
    # We need to track totals to recompute weighted averages
    merged_data = defaultdict(lambda: {
        "num_runs": 0,
        "num_generation_failures": 0,
        "num_passing": 0,
        # For weighted averages, track sum and count
        "sum_generation_latency": 0.0,
        "sum_estimated_cost": 0.0,
        # For variance, we'd need more complex merging - just take weighted avg for now
        "sum_var_generation_latency": 0.0,
        "sum_var_estimated_cost": 0.0,
        # Failure counts by type
        "failure_counts": defaultdict(int),
        # Passing stats
        "sum_est_cost_passing": 0.0,
        "min_gen_latency_passing": float("inf"),
        "min_est_cost_passing": float("inf"),
        "max_est_cost_passing": float("-inf"),
        # Metadata
        "model": None,
        "with_protocol": None,
    })

    for pattern in input_files:
        files = glob(pattern) if "*" in pattern else [pattern]

        for file_path in sorted(files):
            path = Path(file_path)
            if not path.exists():
                print(f"Warning: File not found: {file_path}")
                continue

            print(f"Reading: {file_path}")
            with open(path) as f:
                summary = json.load(f)

            for config_key, stats in summary.items():
                m = merged_data[config_key]
                n = stats.get("num_runs", 0)
                
                if n == 0:
                    continue
                
                # Store metadata
                m["model"] = stats.get("model")
                m["with_protocol"] = stats.get("with_protocol")
                
                # Accumulate counts
                m["num_runs"] += n
                m["num_generation_failures"] += stats.get("num_generation_failures", 0)
                m["num_passing"] += stats.get("num_passing", 0)
                
                # Accumulate for weighted averages
                m["sum_generation_latency"] += stats.get("avg_generation_latency", 0) * n
                m["sum_estimated_cost"] += stats.get("avg_estimated_cost", 0) * n
                m["sum_var_generation_latency"] += stats.get("var_generation_latency", 0) * n
                m["sum_var_estimated_cost"] += stats.get("var_estimated_cost", 0) * n
                
                # Accumulate failure counts from rates
                for failure_type, rate in stats.get("failure_rates", {}).items():
                    m["failure_counts"][failure_type] += int(rate * n)
                
                # Accumulate passing stats
                num_passing = stats.get("num_passing", 0)
                if num_passing > 0:
                    m["sum_est_cost_passing"] += stats.get("avg_est_cost_passing", 0) * num_passing
                    
                    min_lat = stats.get("min_gen_latency_passing")
                    if min_lat is not None:
                        m["min_gen_latency_passing"] = min(m["min_gen_latency_passing"], min_lat)
                    
                    min_cost = stats.get("min_est_cost_passing")
                    if min_cost is not None:
                        m["min_est_cost_passing"] = min(m["min_est_cost_passing"], min_cost)
                    
                    max_cost = stats.get("max_est_cost_passing")
                    if max_cost is not None:
                        m["max_est_cost_passing"] = max(m["max_est_cost_passing"], max_cost)

    # Convert accumulated data to final format
    final_summary = {}
    
    for config_key, m in merged_data.items():
        n = m["num_runs"]
        num_passing = m["num_passing"]
        
        if n == 0:
            continue
        
        # Compute failure rates
        failure_rates = {}
        failure_rates_among_failures = {}
        total_failures = n - num_passing
        
        for failure_type, count in m["failure_counts"].items():
            failure_rates[failure_type] = count / n
            if total_failures > 0:
                failure_rates_among_failures[failure_type] = count / total_failures
        
        final_summary[config_key] = {
            "model": m["model"],
            "with_protocol": m["with_protocol"],
            "num_runs": n,
            "num_generation_failures": m["num_generation_failures"],
            "avg_generation_latency": m["sum_generation_latency"] / n,
            "var_generation_latency": m["sum_var_generation_latency"] / n,  # Approximate
            "avg_estimated_cost": m["sum_estimated_cost"] / n,
            "var_estimated_cost": m["sum_var_estimated_cost"] / n,  # Approximate
            "overall_pass_rate": num_passing / n,
            "failure_rates": failure_rates,
            "failure_rates_among_failures": failure_rates_among_failures,
            "avg_actual_latency": None,
            "num_passing": num_passing,
            "avg_est_cost_passing": m["sum_est_cost_passing"] / num_passing if num_passing > 0 else None,
            "min_gen_latency_passing": m["min_gen_latency_passing"] if m["min_gen_latency_passing"] != float("inf") else None,
            "min_est_cost_passing": m["min_est_cost_passing"] if m["min_est_cost_passing"] != float("inf") else None,
            "max_est_cost_passing": m["max_est_cost_passing"] if m["max_est_cost_passing"] != float("-inf") else None,
        }

    # Save merged summary
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(final_summary, f, indent=2)

    print(f"\nMerged {len(final_summary)} configurations into: {output_file}")
    
    # Print summary
    print("\nMerged statistics per configuration:")
    for config_key, stats in sorted(final_summary.items()):
        print(f"  {config_key}: {stats['num_runs']} runs, {stats['overall_pass_rate']:.1%} pass rate")


@click.command()
@click.argument("input_files", nargs=-1, required=True)
@click.option("--output", "-o", required=True, help="Output file path")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "md", "summary", "all"]),
    default="json",
    help="Output format: json (results), md (markdown), summary (stats), all (default: json)",
)
def main(input_files: tuple, output: str, format: str):
    """
    Merge multiple evaluation result files.

    Examples:
        # Merge JSON result files
        python merge_results.py results/dashdish-custom1_*.json -o results/dashdish-custom1.json

        # Merge markdown files
        python merge_results.py results/dashdish-custom1_*.md -o results/dashdish-custom1.md -f md

        # Merge summary files
        python merge_results.py results/summary_*.json -o results/summary.json -f summary

        # Merge all file types for a task
        python merge_results.py "results/dashdish-custom1_*" -o results/dashdish-custom1 -f all
    """
    input_list = list(input_files)

    if format in ("json", "all"):
        # Filter to result JSON files (exclude summary files)
        json_inputs = [
            f for f in input_list 
            if (f.endswith(".json") or "*" in f) and "summary" not in f
        ]
        if json_inputs:
            merge_json_results(
                json_inputs, output if output.endswith(".json") else f"{output}.json"
            )

    if format in ("md", "all"):
        md_inputs = [f for f in input_list if f.endswith(".md") or "*" in f]
        if md_inputs:
            merge_markdown_results(
                md_inputs, output if output.endswith(".md") else f"{output}.md"
            )

    if format in ("summary", "all"):
        # For summary format, look for summary_*.json files
        if format == "summary":
            summary_inputs = input_list
        else:
            # For "all", derive summary pattern from the input pattern
            summary_inputs = []
            for pattern in input_list:
                if "*" in pattern:
                    # Convert "results/dashdish-custom1_*" to "results/summary_*.json"
                    base_dir = str(Path(pattern).parent)
                    # Extract timestamp pattern if present
                    import re
                    match = re.search(r"_(\d{8}_\d+)", pattern)
                    if match:
                        summary_inputs.append(f"{base_dir}/summary_{match.group(1)}*.json")
                    else:
                        summary_inputs.append(f"{base_dir}/summary_*.json")
        
        if summary_inputs:
            merge_summary_results(
                summary_inputs, 
                output if output.endswith(".json") else f"{output}_summary.json" if format == "all" else f"{output}.json"
            )


if __name__ == "__main__":
    main()
