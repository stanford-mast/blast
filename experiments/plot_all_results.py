#!/usr/bin/env python3
"""
Unified plotting script for all sites (dashdish, gomail, etc.).
Generates all figures with consistent styling.
Merges regenerate_figs.py and regenerate_fig5.py with updated layouts.
"""

import argparse
import json
import shutil
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description="Generate all figures for ICML paper")
parser.add_argument("--site", type=str, default="dashdish",
                   help="Site to process (dashdish, gomail, or 'all')")
parser.add_argument("--copy-to", type=str, default=None,
                   help="Optional directory to copy figures to (e.g., paper figures directory)")
args = parser.parse_args()

SITE = args.site
COPY_TO_DIR = Path(args.copy_to) if args.copy_to else None

# ============================================================================
# DATA LOADING
# ============================================================================

results_dir = Path("experiments/results")

# New structure: results/planner/_DATA/{site}/
data_dir = results_dir / "planner" / "_DATA"

# Sites to process
if SITE == "all":
    sites = ["dashdish", "gomail"]
elif SITE in ["dashdish", "gomail"]:
    sites = [SITE]
else:
    print(f"Unknown site: {SITE}. Using dashdish.")
    sites = ["dashdish"]

print(f"Processing sites: {sites}")

# Load main evaluation data from new structure:
# planner/_DATA/{site}/{model}/{site}-custom-*_{model}.json files
planner_files = []
for site in sites:
    site_dir = data_dir / site
    if not site_dir.exists():
        print(f"Warning: {site_dir} does not exist, skipping {site}")
        continue

    # Look in model subdirectories
    for model_dir in site_dir.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith("_"):
            # Pattern: {site}-custom-*_{model}.json
            for f in model_dir.glob(f"{site}-custom-*.json"):
                if "_e2e_detailed" not in f.name:
                    planner_files.append((site, f))

planner_files = sorted(planner_files, key=lambda x: x[1])

planner_data = []
for site, pf in planner_files:
    # Extract task_id from filename
    stem = pf.stem  # e.g., "dashdish-custom-1_gpt-4.1"
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[0].startswith(f"{site}-custom-"):
        task_id = parts[0]  # e.g., "dashdish-custom-1"
    else:
        task_id = stem

    with open(pf) as f:
        task_results = json.load(f)

    # Tag each result with site and task_id
    for r in task_results:
        r["site"] = site
        r["task_id"] = task_id
    planner_data.extend(task_results)

print(f"\nLoaded {len(planner_data)} planner results from {len(planner_files)} task files")

# Load retry data for Figure 4 (retry comparison)
# Try to load from OLD deepresearch benchmark (the only one with retry data)
retry_data = []
old_planner_file = results_dir / "dashdish-deepresearch1_20251229_180533.json"
old_retry_file = results_dir / "dashdish-deepresearch1_20251229_234920.json"

# Check if we have planner data for retry figure
if old_planner_file.exists() and old_retry_file.exists():
    old_planner_data = json.load(open(old_planner_file))
    retry_data = json.load(open(old_retry_file))
    print(f"Loaded OLD deepresearch data for retry comparison: {len(old_planner_data)} planner + {len(retry_data)} retry results")
else:
    old_planner_data = []
    print("Warning: No retry data found (will skip Figure 4)")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Models to include
MODELS = [
    "openai/gpt-oss-120b",
    "gpt-4.1",
    "gpt-5-mini",
    "gpt-5",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# For Figure 3 (Pass@k and Pass@t) - 3 models only
MODELS_FIG3 = ["gpt-4.1", "gpt-5", "gemini-2.5-flash"]

# Color scheme
COLORS = {
    # GPT models - Blues
    "openai/gpt-oss-120b": "#AED6F1",
    "gpt-4.1-mini": "#B0D8F3",
    "gpt-4.1": "#5499C7",
    "gpt-5-mini": "#2874A6",
    "gpt-5": "#154360",
    # Gemini models - Greens
    "gemini-2.0-flash-lite": "#ABEBC6",
    "gemini-2.0-flash": "#82E0AA",
    "gemini-2.5-flash-lite": "#58D68D",
    "gemini-2.5-flash": "#28B463",
    "gemini-2.5-pro": "#196F3D",
}

# Colors for Figure 3 (3 models)
COLORS_FIG3 = {
    "gpt-4.1": "#5499C7",      # GPT light blue
    "gpt-5": "#154360",         # GPT dark blue
    "gemini-2.5-flash": "#28B463"  # Gemini green
}

# Protocol styles
PROTOCOL_STYLE_WITH = '-'              # Solid line - with protocol
PROTOCOL_STYLE_WITHOUT = ':'           # Dotted - without protocol

LINE_STYLES = {
    True: PROTOCOL_STYLE_WITH,
    False: PROTOCOL_STYLE_WITHOUT,
}

# Model display names
MODEL_NAMES = {
    "openai/gpt-oss-120b": "GPT-OSS-120B",
    "gpt-4.1-mini": "GPT-4.1-mini",
    "gpt-4.1": "GPT-4.1",
    "gpt-5-mini": "GPT-5-mini",
    "gpt-5": "GPT-5",
    "gemini-2.0-flash-lite": "Gemini-2.0-Flash-Lite",
    "gemini-2.0-flash": "Gemini-2.0-Flash",
    "gemini-2.5-flash-lite": "Gemini-2.5-Flash-Lite",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "gemini-2.5-pro": "Gemini-2.5-Pro",
}

#============================================================================
# FIGURE 3: PASS@K AND PASS@T (1x2 LAYOUT, 3 MODELS)
# From regenerate_figs.py - Updated layout
# ============================================================================

print("\n=== Generating Figure 3: Pass@k and Pass@t (1x2 layout, 3 models) ===")

# Use OLD planner data for Figure 3 if available, otherwise use new data
fig3_data = old_planner_data if old_planner_data else planner_data

# Create 2x1 grid: Pass@k (top), Pass@t (bottom), all 3 models together
fig, (ax_k, ax_t) = plt.subplots(2, 1, figsize=(5, 7))
fig.patch.set_facecolor('white')

# Plot each model with both protocol settings
for model in MODELS_FIG3:
    model_data = [r for r in fig3_data if r['config']['model'] == model]

    for with_protocol in [True, False]:
        protocol_data = [r for r in model_data if r['config']['with_protocol'] == with_protocol]

        if not protocol_data:
            continue

        n = len(protocol_data)
        c = sum(1 for r in protocol_data if r['codecheck']['overall_pass'])
        p = c / n if n > 0 else 0

        color = COLORS_FIG3[model]
        linestyle = LINE_STYLES[with_protocol]

        # --- Pass@k (left) ---
        k_values = range(1, 33)
        pass_at_k = []
        for k in k_values:
            if c == 0:
                pass_at_k.append(0.0)
            elif n - c < k:
                pass_at_k.append(1.0)
            else:
                pass_at_k.append(1 - comb(n - c, k) / comb(n, k))

        ax_k.plot(k_values, pass_at_k, color=color, linestyle=linestyle,
                  linewidth=2, alpha=0.85)

        # --- Pass@t (right) ---
        latencies = [r['generation_latency'] for r in protocol_data if r['generation_latency'] > 0]
        if latencies:
            t_max = max(latencies)
            t_values = np.linspace(0, min(t_max, 100), 200)

            pass_at_t = []
            for t in t_values:
                F_t = sum(1 for lat in latencies if lat <= t) / len(latencies)
                n_parallel = 64
                pass_at_t.append(1 - (1 - F_t * p) ** n_parallel)

            ax_t.plot(t_values, pass_at_t, color=color, linestyle=linestyle,
                      linewidth=2, alpha=0.85)

# Configure axes - vertical layout with smaller fonts
ax_k.set_xlabel("k", fontsize=8, fontweight="bold")
ax_k.set_ylabel("Pass@k", fontsize=8, fontweight="bold")
ax_k.set_xlim(1, 32)
ax_k.set_ylim(0, 1.05)
ax_k.tick_params(axis='both', labelsize=7)
ax_k.grid(True, alpha=0.3, linewidth=0.5)

ax_t.set_xlabel("t (seconds)", fontsize=8, fontweight="bold")
ax_t.set_ylabel("Pass@t", fontsize=8, fontweight="bold")
ax_t.set_ylim(0, 1.05)
ax_t.tick_params(axis='both', labelsize=7)
ax_t.grid(True, alpha=0.3, linewidth=0.5)

# Create legend with 2 rows x 3 columns (COLUMN-MAJOR order for matplotlib)
# Display: Row 1: GPT-4.1, Gemini-2.5-Flash, With Protocol
#          Row 2: GPT-5, (empty), Without Protocol
# Column-major means items fill top-to-bottom per column, so:
#   Col0: items 0,1  Col1: items 2,3  Col2: items 4,5
from matplotlib.lines import Line2D

legend_elements = [
    # Column 0 (GPT models)
    Line2D([0], [0], color=COLORS_FIG3["gpt-4.1"], linewidth=2.5),
    Line2D([0], [0], color=COLORS_FIG3["gpt-5"], linewidth=2.5),
    # Column 1 (Gemini models)
    Line2D([0], [0], color=COLORS_FIG3["gemini-2.5-flash"], linewidth=2.5),
    Line2D([0], [0], linestyle='None'),  # empty spacer
    # Column 2 (Protocol lines)
    Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5),
    Line2D([0], [0], color='gray', linestyle=':', linewidth=2.5),
]
legend_labels = [
    # Column 0
    "GPT-4.1", "GPT-5",
    # Column 1
    "Gemini-2.5-Flash", " ",
    # Column 2
    "With Protocol", "Without Protocol"
]

fig.legend(legend_elements, legend_labels, loc='lower center',
           bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7,
           frameon=True, framealpha=0.95, edgecolor='black',
           columnspacing=1.0, handletextpad=0.3)

plt.tight_layout(rect=[0, 0.06, 1, 1])

fig3_png = results_dir / "fig3_candidate_efficiency.png"
fig3_pdf = results_dir / "fig3_candidate_efficiency.pdf"
plt.savefig(fig3_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig3_pdf, bbox_inches='tight', facecolor='white')
print(f"Created: {fig3_png}")
print(f"Created: {fig3_pdf}")
plt.close()

# ============================================================================
# FIGURE 4: SERIAL RETRY VS PARALLEL HEDGING
# From regenerate_figs.py - Uses OLD retry data
# ============================================================================

if old_planner_data and retry_data:
    print("\n=== Generating Figure 4: Serial Retry vs Parallel Hedging ===")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.patch.set_facecolor('white')

    comparison_model = 'gemini-2.5-flash'

    # Colors
    color_retry = '#E74C3C'    # Red
    color_parallel = '#27AE60' # Green

    all_curves = []
    t_max_global = 0

    for with_protocol in [True, False]:
        # Get single attempt data from OLD planner data
        single_data = [r for r in old_planner_data
                       if r['config']['model'] == comparison_model
                       and r['config']['with_protocol'] == with_protocol]

        # Get retry data
        retry_data_filtered = [r for r in retry_data
                              if r['config']['model'] == comparison_model
                              and r['config']['with_protocol'] == with_protocol]

        if not single_data or not retry_data_filtered:
            continue

        single_passed = sum(1 for r in single_data if r['codecheck']['overall_pass'])
        single_success_rate = single_passed / len(single_data)
        single_latencies = [r['generation_latency'] for r in single_data if r['generation_latency'] > 0]

        retry_latencies = [r['generation_latency'] for r in retry_data_filtered if r['generation_latency'] > 0]

        t_max = max(max(single_latencies), max(retry_latencies))
        t_max_global = max(t_max_global, t_max)

        all_curves.append({
            'with_protocol': with_protocol,
            'single_latencies': single_latencies,
            'single_success_rate': single_success_rate,
            'retry_data_filtered': retry_data_filtered,
        })

    t_values = np.linspace(0, t_max_global, 200)

    for curve_data in all_curves:
        with_protocol = curve_data['with_protocol']
        single_latencies = curve_data['single_latencies']
        single_success_rate = curve_data['single_success_rate']
        retry_results = curve_data['retry_data_filtered']

        linestyle = LINE_STYLES[True] if with_protocol else LINE_STYLES[False]
        protocol_label = 'w/ protocol' if with_protocol else 'w/o protocol'

        # Parallel hedging (k=8)
        pass_at_t_parallel = []
        for t in t_values:
            F_t = sum(1 for lat in single_latencies if lat <= t) / len(single_latencies)
            k = 8
            pass_at_t_parallel.append(1 - (1 - F_t * single_success_rate) ** k)

        # Serial retry - empirical CDF
        pass_at_t_retry = []
        for t in t_values:
            successes_by_t = sum(1 for r in retry_results
                               if r['codecheck']['overall_pass'] and r['generation_latency'] <= t)
            pass_at_t_retry.append(successes_by_t / len(retry_results))

        ax.plot(t_values, pass_at_t_parallel, color=color_parallel, linewidth=2.5,
               linestyle=linestyle, label=f'Parallel k=8 ({protocol_label})', alpha=0.85)
        ax.plot(t_values, pass_at_t_retry, color=color_retry, linewidth=2.5,
               linestyle=linestyle, label=f'Serial retry ({protocol_label})', alpha=0.85)

    ax.set_xlabel("t (seconds)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Pass@k", fontsize=12, fontweight='bold')
    ax.set_xlim(0, t_max_global)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
             ncol=2, fontsize=10, frameon=True, framealpha=0.95,
             edgecolor='black', columnspacing=1.5)

    plt.tight_layout()
    fig4_png = results_dir / "fig4_retry_comparison.png"
    fig4_pdf = results_dir / "fig4_retry_comparison.pdf"
    plt.savefig(fig4_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(fig4_pdf, bbox_inches='tight', facecolor='white')
    print(f"Created: {fig4_png}")
    print(f"Created: {fig4_pdf}")
    plt.close()

# ============================================================================
# FIGURE 5: LATENCY BREAKDOWN (GROUPED BAR CHART)
# From regenerate_fig5.py - Updated naming
# ============================================================================

print("\n=== Generating Figure 5: Latency Breakdown ===")

# Load e2e_detailed.json files from all sites
e2e_files = []
for site in sites:
    site_dir = data_dir / site
    if site_dir.exists():
        e2e_files.extend(list(site_dir.glob(f"{site}-custom-*_e2e_detailed.json")))

print(f"Found {len(e2e_files)} E2E result files")

if e2e_files:
    # Aggregate results
    data_by_model = defaultdict(dict)

    for e2e_file in e2e_files:
        with open(e2e_file) as f:
            e2e_data = json.load(f)

        for result in e2e_data.get('results', []):
            name = result.get('name', '')
            avg_timing = result.get('avg_timing', {})

            # Parse model and suffix from name (e.g., "gpt-4.1-best")
            model_raw = None
            suffix = None
            for s in ["loop-tools-baseline", "loop-baseline", "best", "worst"]:
                if name.endswith(f"-{s}"):
                    suffix = s
                    model_raw = name[:-len(f"-{s}")]
                    break

            if not model_raw or not suffix:
                continue

            # Initialize if needed
            if model_raw not in data_by_model:
                data_by_model[model_raw] = {}

            # Aggregate timing
            if suffix not in data_by_model[model_raw]:
                data_by_model[model_raw][suffix] = {
                    "planning": [],
                    "llm": [],
                    "action": [],
                    "correctness": []
                }

            planning = avg_timing.get('planning_seconds', 0)
            llm_total = avg_timing.get('llm_total_seconds', 0)
            execution = avg_timing.get('execution_seconds', 0)
            action = max(0, execution - llm_total)
            correctness = result.get('avg_correctness_pct', 0)

            data_by_model[model_raw][suffix]["planning"].append(planning)
            data_by_model[model_raw][suffix]["llm"].append(llm_total)
            data_by_model[model_raw][suffix]["action"].append(action)
            data_by_model[model_raw][suffix]["correctness"].append(correctness)

    # Average aggregated data
    for model in data_by_model:
        for suffix in data_by_model[model]:
            data_by_model[model][suffix] = {
                "planning": np.mean(data_by_model[model][suffix]["planning"]),
                "llm": np.mean(data_by_model[model][suffix]["llm"]),
                "action": np.mean(data_by_model[model][suffix]["action"]),
                "correctness": np.mean(data_by_model[model][suffix]["correctness"])
            }

    # Models and strategies for figure - grouped by METHOD, bars for MODELS
    models = ["gpt-4.1", "gemini-2.5-flash", "gemini-2.5-pro"]
    suffixes = ["loop-baseline", "loop-tools-baseline", "best"]  # Order: Browser-Use, +cache, JIT

    # Create figure with single subplot - VERTICAL bars showing absolute latency (TALLER)
    fig, ax = plt.subplots(1, 1, figsize=(7, 8))  # Taller figure
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Strategy display names
    strategy_names = {
        "loop-baseline": "Browser-Use",
        "loop-tools-baseline": "Browser-Use\n+cache",
        "best": "JIT-Planner"
    }

    # Model display names
    model_names = {
        "gpt-4.1": "GPT-4.1",
        "gemini-2.5-flash": "Gemini-2.5-Flash",
        "gemini-2.5-pro": "Gemini-2.5-Pro"
    }

    # Hatching patterns for models (no text labels)
    model_hatches = {
        "gpt-4.1": None,        # No hatch (solid)
        "gemini-2.5-flash": '///',  # Forward slash hatch
        "gemini-2.5-pro": '\\\\\\'     # Backward slash hatch
    }

    # Colors for stacked components
    color_planning = "#FF6B6B"   # Red
    color_llm = "#4ECDC4"        # Teal
    color_action = "#FFA726"     # Orange

    # Build VERTICAL grouped bars
    bar_width = 0.22
    group_spacing = 0.8  # Space between method groups

    # Calculate x positions for each method group
    x_group_centers = []
    x_group_labels = []

    for i, suffix in enumerate(suffixes):
        # Center position for this method group
        center = i * (len(models) * bar_width + group_spacing)
        x_group_centers.append(center + len(models) * bar_width / 2)
        x_group_labels.append(strategy_names[suffix])

        # Plot bars for each model within this group (NO spacing between bars)
        for j, model in enumerate(models):
            if model not in data_by_model or suffix not in data_by_model[model]:
                continue

            data = data_by_model[model][suffix]
            planning = data["planning"]
            llm = data["llm"]
            action = data["action"]

            # X position for this bar
            x = center + j * bar_width

            # Get hatch pattern for this model
            hatch = model_hatches[model]

            # Stack the components vertically (ABSOLUTE values) with hatching
            ax.bar(x, planning, bar_width, color=color_planning, hatch=hatch,
                   edgecolor='white', linewidth=0.5, label='Planning' if i == 0 and j == 0 else '')
            ax.bar(x, llm, bar_width, bottom=planning, color=color_llm, hatch=hatch,
                   edgecolor='white', linewidth=0.5, label='Inference' if i == 0 and j == 0 else '')
            ax.bar(x, action, bar_width, bottom=planning + llm, color=color_action, hatch=hatch,
                   edgecolor='white', linewidth=0.5, label='Tool Execution' if i == 0 and j == 0 else '')

    # Configure axes
    ax.set_ylabel("Latency (seconds)", fontsize=12, fontweight="bold")
    ax.set_xticks(x_group_centers)
    ax.set_xticklabels(x_group_labels, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')  # Only horizontal gridlines

    # Create legend with 2 rows: Row 1 = components, Row 2 = models
    # IMPORTANT: matplotlib ncol=3 fills COLUMN-MAJOR (top-to-bottom per column)
    # For 2 rows × 3 cols: Col0=[0,1], Col1=[2,3], Col2=[4,5]
    # Display: Row0=[0,2,4], Row1=[1,3,5]
    from matplotlib.patches import Patch

    # Build in COLUMN-MAJOR order for proper display:
    # Col0: Planning, GPT-4.1
    # Col1: Inference, Gemini-2.5-Flash
    # Col2: Tool Execution, Gemini-2.5-Pro
    all_handles = [
        # Column 0
        Patch(facecolor=color_planning, edgecolor='white'),  # Planning
        Patch(facecolor='gray', hatch=model_hatches["gpt-4.1"], edgecolor='none'),  # GPT-4.1
        # Column 1
        Patch(facecolor=color_llm, edgecolor='white'),  # Inference
        Patch(facecolor='gray', hatch=model_hatches["gemini-2.5-flash"], edgecolor='none'),  # Gemini-Flash
        # Column 2
        Patch(facecolor=color_action, edgecolor='white'),  # Tool Execution
        Patch(facecolor='gray', hatch=model_hatches["gemini-2.5-pro"], edgecolor='none'),  # Gemini-Pro
    ]
    all_labels = [
        # Column 0
        'Planning', model_names["gpt-4.1"],
        # Column 1
        'Inference', model_names["gemini-2.5-flash"],
        # Column 2
        'Tool Execution', model_names["gemini-2.5-pro"]
    ]

    fig.legend(all_handles, all_labels, loc='lower center',
              bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10,
              frameon=True, framealpha=0.95, edgecolor='black')

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    fig5_png = results_dir / "fig5_latency_breakdown.png"
    fig5_pdf = results_dir / "fig5_latency_breakdown.pdf"
    plt.savefig(fig5_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(fig5_pdf, bbox_inches='tight', facecolor='white')
    print(f"Created: {fig5_png}")
    print(f"Created: {fig5_pdf}")

    # Export CSV for Figure 5
    fig5_csv = results_dir / "fig5_latency_breakdown.csv"
    with open(fig5_csv, 'w') as f:
        f.write("Method,Model,Planning,Inference,ToolExecution,Total\n")
        for suffix in suffixes:
            for model in models:
                if model in data_by_model and suffix in data_by_model[model]:
                    data = data_by_model[model][suffix]
                    total = data["planning"] + data["llm"] + data["action"]
                    f.write(f"{strategy_names[suffix].replace(chr(10), ' ')},{model_names[model]},"
                           f"{data['planning']:.2f},{data['llm']:.2f},{data['action']:.2f},{total:.2f}\n")
    print(f"Created: {fig5_csv}")

    plt.close()

# ============================================================================
# TABLE 1: E2E RESULTS AGGREGATED BY STRATEGY
# ============================================================================

print("\n=== Generating Table 1: E2E Results ===")

if e2e_files:
    # Task cardinality mapping (based on task number)
    TASK_CARDINALITY = {}
    for site in sites:
        for i in range(1, 10):
            task_id = f"{site}-custom-{i}"
            if i <= 3:
                TASK_CARDINALITY[task_id] = "C-Low"
            elif i <= 6:
                TASK_CARDINALITY[task_id] = "C-Medium"
            else:
                TASK_CARDINALITY[task_id] = "C-High"

    # Strategy name mapping
    STRATEGY_MAPPING = {
        "best": "JIT-Planner",
        "loop-tools-baseline": "Browser-Use +cache",
        "loop-baseline": "Browser-Use",
    }

    # First pass: compute Browser-Use execution times for each task to create execution length bins
    task_execution_times = {}
    for e2e_file in e2e_files:
        with open(e2e_file) as f:
            data = json.load(f)
        task_id = data["task_id"]

        # Find Browser-Use (loop-baseline) results for this task
        for result in data["results"]:
            name = result["name"]
            if name.endswith("-loop-baseline"):
                exec_time = result["avg_timing"]["execution_seconds"]
                task_execution_times[task_id] = exec_time
                break

    # Create execution length bins based on Browser-Use execution times
    if task_execution_times:
        exec_times_sorted = sorted(task_execution_times.values())
        # Use 33rd and 67th percentile as thresholds
        short_threshold = exec_times_sorted[len(exec_times_sorted) // 3]
        long_threshold = exec_times_sorted[(2 * len(exec_times_sorted)) // 3]

        TASK_EXEC_LENGTH = {}
        for task_id, exec_time in task_execution_times.items():
            if exec_time <= short_threshold:
                TASK_EXEC_LENGTH[task_id] = "Short"
            elif exec_time <= long_threshold:
                TASK_EXEC_LENGTH[task_id] = "Medium"
            else:
                TASK_EXEC_LENGTH[task_id] = "Long"
    else:
        TASK_EXEC_LENGTH = {}

    # Aggregate results
    aggregated_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"latency": [], "accuracy": []})))

    for e2e_file in e2e_files:
        with open(e2e_file) as f:
            data = json.load(f)

        task_id = data["task_id"]
        cardinality = TASK_CARDINALITY.get(task_id, "Unknown")
        exec_length = TASK_EXEC_LENGTH.get(task_id, "Unknown")

        # Extract site/application from task_id (e.g., "dashdish-custom-1" -> "dashdish")
        site = task_id.split("-custom-")[0] if "-custom-" in task_id else "Unknown"
        # Capitalize for display
        site_display = site.capitalize() if site != "Unknown" else site

        for result in data["results"]:
            name = result["name"]

            # Parse strategy and model from name
            strategy_raw = None
            model_raw = None

            for strat in ["loop-tools-baseline", "loop-baseline", "best", "worst"]:
                if name.endswith(f"-{strat}"):
                    strategy_raw = strat
                    model_raw = name[:-len(f"-{strat}")]
                    break

            if not strategy_raw or not model_raw:
                continue

            strategy = STRATEGY_MAPPING.get(strategy_raw, strategy_raw)
            model = MODEL_NAMES.get(model_raw, model_raw)

            latency = result["avg_timing"]["total_seconds"]
            accuracy = result["avg_correctness_pct"]

            # Aggregate by model, cardinality, execution length, and site/application
            aggregated_table[strategy]["model"][model]["latency"].append(latency)
            aggregated_table[strategy]["model"][model]["accuracy"].append(accuracy)
            aggregated_table[strategy]["cardinality"][cardinality]["latency"].append(latency)
            aggregated_table[strategy]["cardinality"][cardinality]["accuracy"].append(accuracy)
            aggregated_table[strategy]["exec_length"][exec_length]["latency"].append(latency)
            aggregated_table[strategy]["exec_length"][exec_length]["accuracy"].append(accuracy)
            aggregated_table[strategy]["site"][site_display]["latency"].append(latency)
            aggregated_table[strategy]["site"][site_display]["accuracy"].append(accuracy)

    # Generate LaTeX table - Latency/Accuracy/Range as COLUMNS
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{l l c c c c c c c}")
    latex.append(r"\toprule")

    # Header rows
    latex.append(r"& & \multicolumn{2}{c}{\textbf{Browser-Use}} & \multicolumn{2}{c}{\textbf{Browser-Use +cache}} & \multicolumn{3}{c}{\textbf{JIT-Planner}} \\")
    latex.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-9}")
    latex.append(r"& & Latency & Accuracy & Latency & Accuracy & Latency (s) & Range (s) & Accuracy \\")
    latex.append(r"\midrule")

    models = ["GPT-4.1", "Gemini-2.5-Flash", "Gemini-2.5-Pro"]
    cardinalities = ["C-Low", "C-Medium", "C-High"]
    exec_lengths = ["Short", "Medium", "Long"]
    strategy_order = ["Browser-Use", "Browser-Use +cache", "JIT-Planner"]

    # Helper to format row with speedup/improvement in parentheses (no bold)
    def format_row(row_data):
        # row_data is dict: strategy -> {"lat": val, "acc": val, "lat_range": (min, max)}
        lats = [row_data[s]["lat"] for s in strategy_order if s in row_data]
        accs = [row_data[s]["acc"] for s in strategy_order if s in row_data]

        if not lats:
            return ["--"]*8

        # Format cells
        cells = []
        for i, strategy in enumerate(strategy_order):
            if strategy not in row_data:
                if i == 2:  # JIT-Planner has 3 columns
                    cells.extend(["--", "--", "--"])
                else:
                    cells.extend(["--", "--"])
                continue

            lat = row_data[strategy]["lat"]
            acc = row_data[strategy]["acc"]

            # Format latency (no bold)
            lat_str = f"{lat:.1f}"

            # Add speedup in parentheses for JIT-Planner
            if i == 2 and lats[0] > 0:  # JIT-Planner column
                speedup = lats[0] / lat
                lat_str += f" ({speedup:.1f}x)"

            cells.append(lat_str)

            # Add Range column for JIT-Planner
            if i == 2:  # JIT-Planner column
                lat_min, lat_max = row_data[strategy].get("lat_range", (0, 0))
                range_val = lat_max - lat_min
                range_str = f"{range_val:.1f}"
                cells.append(range_str)

            # Format accuracy (no bold)
            acc_str = f"{acc*100:.0f}\\%"

            # Add improvement in parentheses for JIT-Planner
            if i == 2:  # JIT-Planner column
                acc_improvement = (acc - accs[0]) * 100
                acc_str += f" ({acc_improvement:+.0f}\\%)"

            cells.append(acc_str)

        return cells

    # BY MODEL
    for i, model in enumerate(models):
        model_data = {}
        for strategy in strategy_order:
            if strategy in aggregated_table:
                data = aggregated_table[strategy]["model"][model]
                lat = np.mean(data["latency"]) if data["latency"] else 0.0
                acc = np.mean(data["accuracy"]) if data["accuracy"] else 0.0
                lat_range = (min(data["latency"]), max(data["latency"])) if data["latency"] else (0.0, 0.0)
                model_data[strategy] = {"lat": lat, "acc": acc, "lat_range": lat_range}

        cells = format_row(model_data)
        if i == 0:
            latex.append(f"\\multirow{{3}}{{*}}{{Model}} & {model} & {' & '.join(cells)} \\\\")
        else:
            latex.append(f"& {model} & {' & '.join(cells)} \\\\")

    latex.append(r"\midrule")

    # BY TASK COMPLEXITY
    for i, card in enumerate(cardinalities):
        card_data = {}
        for strategy in strategy_order:
            if strategy in aggregated_table:
                data = aggregated_table[strategy]["cardinality"][card]
                lat = np.mean(data["latency"]) if data["latency"] else 0.0
                acc = np.mean(data["accuracy"]) if data["accuracy"] else 0.0
                lat_range = (min(data["latency"]), max(data["latency"])) if data["latency"] else (0.0, 0.0)
                card_data[strategy] = {"lat": lat, "acc": acc, "lat_range": lat_range}

        cells = format_row(card_data)
        if i == 0:
            latex.append(f"\\multirow{{3}}{{*}}{{\\shortstack[l]{{Task\\\\Cardinality}}}} & {card} & {' & '.join(cells)} \\\\")
        else:
            latex.append(f"& {card} & {' & '.join(cells)} \\\\")

    latex.append(r"\midrule")

    # BY TASK LENGTH (with T- prefix)
    length_display = {"Short": "T-Short", "Medium": "T-Medium", "Long": "T-Long"}
    for i, length in enumerate(exec_lengths):
        length_data = {}
        for strategy in strategy_order:
            if strategy in aggregated_table:
                data = aggregated_table[strategy]["exec_length"][length]
                lat = np.mean(data["latency"]) if data["latency"] else 0.0
                acc = np.mean(data["accuracy"]) if data["accuracy"] else 0.0
                lat_range = (min(data["latency"]), max(data["latency"])) if data["latency"] else (0.0, 0.0)
                length_data[strategy] = {"lat": lat, "acc": acc, "lat_range": lat_range}

        cells = format_row(length_data)
        length_label = length_display.get(length, length)
        if i == 0:
            latex.append(f"\\multirow{{3}}{{*}}{{\\shortstack[l]{{Task\\\\Length}}}} & {length_label} & {' & '.join(cells)} \\\\")
        else:
            latex.append(f"& {length_label} & {' & '.join(cells)} \\\\")

    latex.append(r"\midrule")

    # BY APPLICATION
    # Get unique sites from aggregated data
    all_sites = set()
    for strategy in aggregated_table.values():
        if "site" in strategy:
            all_sites.update(strategy["site"].keys())

    sites_sorted = sorted(all_sites)

    for i, site in enumerate(sites_sorted):
        site_data = {}
        for strategy in strategy_order:
            if strategy in aggregated_table and site in aggregated_table[strategy]["site"]:
                data = aggregated_table[strategy]["site"][site]
                lat = np.mean(data["latency"]) if data["latency"] else 0.0
                acc = np.mean(data["accuracy"]) if data["accuracy"] else 0.0
                lat_range = (min(data["latency"]), max(data["latency"])) if data["latency"] else (0.0, 0.0)
                site_data[strategy] = {"lat": lat, "acc": acc, "lat_range": lat_range}

        cells = format_row(site_data)
        if i == 0:
            latex.append(f"\\multirow{{{len(sites_sorted)}}}{{*}}{{Application}} & {site} & {' & '.join(cells)} \\\\")
        else:
            latex.append(f"& {site} & {' & '.join(cells)} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{\textbf{End-to-end results across methods.} Latency (seconds), Range (seconds), and Accuracy (\%) aggregated by model, task complexity, task length, and application. Speedup and improvement shown in parentheses for JIT-Planner.}")
    latex.append(r"\label{tab:e2e_results}")
    latex.append(r"\end{table*}")

    # Write LaTeX to file
    table_output = Path("papers/icml26/6945e24f11603a0b74fd3186/table1_results.tex")
    table_output.parent.mkdir(parents=True, exist_ok=True)
    with open(table_output, "w") as f:
        f.write("\n".join(latex))
    print(f"Created: {table_output}")

    # Export CSV for Table 1
    table1_csv = results_dir / "table1_e2e_results.csv"
    with open(table1_csv, 'w') as f:
        f.write("Category,Subcategory,BrowserUse_Latency,BrowserUse_Accuracy,BrowserUse+cache_Latency,BrowserUse+cache_Accuracy,JIT-Planner_Latency,JIT-Planner_Range,JIT-Planner_Accuracy,JIT-Planner_Speedup,JIT-Planner_Improvement\n")

        # By Model
        for model in models:
            model_data = {}
            for strategy in strategy_order:
                if strategy in aggregated_table:
                    data = aggregated_table[strategy]["model"][model]
                    lat = np.mean(data["latency"]) if data["latency"] else 0.0
                    acc = np.mean(data["accuracy"]) if data["accuracy"] else 0.0
                    lat_range = (min(data["latency"]), max(data["latency"])) if data["latency"] else (0.0, 0.0)
                    model_data[strategy] = {"lat": lat, "acc": acc, "lat_range": lat_range}

            bu_lat = model_data.get("Browser-Use", {}).get("lat", 0)
            bu_acc = model_data.get("Browser-Use", {}).get("acc", 0)
            buc_lat = model_data.get("Browser-Use +cache", {}).get("lat", 0)
            buc_acc = model_data.get("Browser-Use +cache", {}).get("acc", 0)
            jit_lat = model_data.get("JIT-Planner", {}).get("lat", 0)
            jit_range = model_data.get("JIT-Planner", {}).get("lat_range", (0, 0))
            jit_acc = model_data.get("JIT-Planner", {}).get("acc", 0)
            speedup = bu_lat / jit_lat if jit_lat > 0 else 0
            improvement = (jit_acc - bu_acc) * 100

            f.write(f"Model,{model},{bu_lat:.1f},{bu_acc:.2f},{buc_lat:.1f},{buc_acc:.2f},"
                   f"{jit_lat:.1f},{jit_range[1]-jit_range[0]:.1f},{jit_acc:.2f},{speedup:.1f},{improvement:.1f}\n")

        # By Task Complexity
        for card in cardinalities:
            card_data = {}
            for strategy in strategy_order:
                if strategy in aggregated_table:
                    data = aggregated_table[strategy]["cardinality"][card]
                    lat = np.mean(data["latency"]) if data["latency"] else 0.0
                    acc = np.mean(data["accuracy"]) if data["accuracy"] else 0.0
                    lat_range = (min(data["latency"]), max(data["latency"])) if data["latency"] else (0.0, 0.0)
                    card_data[strategy] = {"lat": lat, "acc": acc, "lat_range": lat_range}

            bu_lat = card_data.get("Browser-Use", {}).get("lat", 0)
            bu_acc = card_data.get("Browser-Use", {}).get("acc", 0)
            buc_lat = card_data.get("Browser-Use +cache", {}).get("lat", 0)
            buc_acc = card_data.get("Browser-Use +cache", {}).get("acc", 0)
            jit_lat = card_data.get("JIT-Planner", {}).get("lat", 0)
            jit_range = card_data.get("JIT-Planner", {}).get("lat_range", (0, 0))
            jit_acc = card_data.get("JIT-Planner", {}).get("acc", 0)
            speedup = bu_lat / jit_lat if jit_lat > 0 else 0
            improvement = (jit_acc - bu_acc) * 100

            f.write(f"Task Complexity,{card},{bu_lat:.1f},{bu_acc:.2f},{buc_lat:.1f},{buc_acc:.2f},"
                   f"{jit_lat:.1f},{jit_range[1]-jit_range[0]:.1f},{jit_acc:.2f},{speedup:.1f},{improvement:.1f}\n")

        # By Task Length
        for length in exec_lengths:
            length_data = {}
            for strategy in strategy_order:
                if strategy in aggregated_table:
                    data = aggregated_table[strategy]["exec_length"][length]
                    lat = np.mean(data["latency"]) if data["latency"] else 0.0
                    acc = np.mean(data["accuracy"]) if data["accuracy"] else 0.0
                    lat_range = (min(data["latency"]), max(data["latency"])) if data["latency"] else (0.0, 0.0)
                    length_data[strategy] = {"lat": lat, "acc": acc, "lat_range": lat_range}

            bu_lat = length_data.get("Browser-Use", {}).get("lat", 0)
            bu_acc = length_data.get("Browser-Use", {}).get("acc", 0)
            buc_lat = length_data.get("Browser-Use +cache", {}).get("lat", 0)
            buc_acc = length_data.get("Browser-Use +cache", {}).get("acc", 0)
            jit_lat = length_data.get("JIT-Planner", {}).get("lat", 0)
            jit_range = length_data.get("JIT-Planner", {}).get("lat_range", (0, 0))
            jit_acc = length_data.get("JIT-Planner", {}).get("acc", 0)
            speedup = bu_lat / jit_lat if jit_lat > 0 else 0
            improvement = (jit_acc - bu_acc) * 100

            f.write(f"Task Length,{length},{bu_lat:.1f},{bu_acc:.2f},{buc_lat:.1f},{buc_acc:.2f},"
                   f"{jit_lat:.1f},{jit_range[1]-jit_range[0]:.1f},{jit_acc:.2f},{speedup:.1f},{improvement:.1f}\n")

        # By Application
        for site in sites_sorted:
            site_data = {}
            for strategy in strategy_order:
                if strategy in aggregated_table and site in aggregated_table[strategy]["site"]:
                    data = aggregated_table[strategy]["site"][site]
                    lat = np.mean(data["latency"]) if data["latency"] else 0.0
                    acc = np.mean(data["accuracy"]) if data["accuracy"] else 0.0
                    lat_range = (min(data["latency"]), max(data["latency"])) if data["latency"] else (0.0, 0.0)
                    site_data[strategy] = {"lat": lat, "acc": acc, "lat_range": lat_range}

            bu_lat = site_data.get("Browser-Use", {}).get("lat", 0)
            bu_acc = site_data.get("Browser-Use", {}).get("acc", 0)
            buc_lat = site_data.get("Browser-Use +cache", {}).get("lat", 0)
            buc_acc = site_data.get("Browser-Use +cache", {}).get("acc", 0)
            jit_lat = site_data.get("JIT-Planner", {}).get("lat", 0)
            jit_range = site_data.get("JIT-Planner", {}).get("lat_range", (0, 0))
            jit_acc = site_data.get("JIT-Planner", {}).get("acc", 0)
            speedup = bu_lat / jit_lat if jit_lat > 0 else 0
            improvement = (jit_acc - bu_acc) * 100

            f.write(f"Application,{site},{bu_lat:.1f},{bu_acc:.2f},{buc_lat:.1f},{buc_acc:.2f},"
                   f"{jit_lat:.1f},{jit_range[1]-jit_range[0]:.1f},{jit_acc:.2f},{speedup:.1f},{improvement:.1f}\n")

    print(f"Created: {table1_csv}")


# ============================================================================
# FIGURE 1: ACCURACY VS LATENCY (PARETO FRONTIER)
# ============================================================================

print("\n=== Generating Figure 1: Accuracy vs Latency (Pareto Frontier) ===")

# Only plot if we have planner data
if planner_data:
    # Stack subplots vertically for single-column figure, shared x-axis
    fig, (ax_gpt, ax_gemini) = plt.subplots(2, 1, figsize=(5, 7), sharex=True)
    fig.patch.set_facecolor('white')

    # Organize data by model and protocol - USE ALL MODELS
    for model in MODELS:  # Use all models, not just fig3 subset
        model_data = [r for r in planner_data if r["config"]["model"] == model]

        for with_protocol in [True, False]:
            protocol_data = [r for r in model_data if r["config"]["with_protocol"] == with_protocol]

            if not protocol_data:
                continue

            # Calculate metrics
            passed = sum(1 for r in protocol_data if r["codecheck"]["overall_pass"])
            total = len(protocol_data)
            accuracy = passed / total if total > 0 else 0
            avg_latency = np.mean([r["generation_latency"] for r in protocol_data])

            # Determine subplot - handle openai prefix
            is_gpt = model.startswith("gpt") or model.startswith("openai")
            ax = ax_gpt if is_gpt else ax_gemini

            # Plot point - circles ONLY (no squares), smaller dots
            color = COLORS[model]  # Use COLORS not COLORS_FIG3
            ax.scatter(
                avg_latency, accuracy,
                color=color, s=80, marker='o',  # Smaller dots
                alpha=0.85, edgecolors='black', linewidth=1.0, zorder=5
            )

    # Connect points for each model with protocol-specific lines
    for ax, model_prefix in [(ax_gpt, ("gpt", "openai")), (ax_gemini, "gemini")]:
        # Collect all points for this subplot grouped by protocol
        for with_protocol in [True, False]:
            points = []
            for model in MODELS:  # Use all models
                # Check if model matches prefix (handle openai specially for GPT subplot)
                if isinstance(model_prefix, tuple):
                    matches = any(model.startswith(p) for p in model_prefix)
                else:
                    matches = model.startswith(model_prefix)

                if matches:
                    protocol_data = [
                        r for r in planner_data
                        if r["config"]["model"] == model and r["config"]["with_protocol"] == with_protocol
                    ]
                    if protocol_data:
                        passed = sum(1 for r in protocol_data if r["codecheck"]["overall_pass"])
                        accuracy = passed / len(protocol_data)
                        avg_latency = np.mean([r["generation_latency"] for r in protocol_data])
                        points.append((avg_latency, accuracy, model))

            # Draw one line connecting all points for this protocol
            if len(points) >= 2:
                points.sort(key=lambda x: x[0])  # Sort by latency
                # Use solid '-' for with protocol, dotted ':' for without
                style = '-' if with_protocol else ':'
                ax.plot(
                    [p[0] for p in points], [p[1] for p in points],
                    color='#555555', linestyle=style, linewidth=2.5, alpha=0.6, zorder=3
                )

        # Vertical layout with shared x-axis: only bottom gets xlabel
        if ax == ax_gemini:
            ax.set_xlabel('Planning Latency (s)', fontsize=8, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=8, fontweight='bold')
        ax.tick_params(axis='both', labelsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    # Create legend with 3 columns: Col1=OpenAI, Col2=Gemini, Col3=Protocol
    # IMPORTANT: matplotlib's ncol fills COLUMN-MAJOR (top-to-bottom per column, then next column)
    # So for ncol=3 with N items total, items fill:
    #   Col0: items 0,1,2,...  Col1: items n,n+1,...  Col2: items 2n,2n+1,...
    from matplotlib.lines import Line2D

    gpt_models = [m for m in MODELS if m.startswith("gpt") or m.startswith("openai")]
    gemini_models = [m for m in MODELS if m.startswith("gemini")]

    # Build in COLUMN-MAJOR order for ncol=3
    # Column 1 (all GPT models), Column 2 (all Gemini models), Column 3 (protocols)
    max_rows = max(len(gpt_models), len(gemini_models), 2)

    legend_elements = []
    legend_labels = []

    # Column 1: All GPT/OpenAI models
    for i in range(max_rows):
        if i < len(gpt_models):
            m = gpt_models[i]
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=COLORS[m], markersize=6,
                                          markeredgecolor='black', markeredgewidth=0.8))
            legend_labels.append(MODEL_NAMES[m])
        else:
            legend_elements.append(Line2D([0], [0], linestyle='None'))
            legend_labels.append(' ')

    # Column 2: All Gemini models
    for i in range(max_rows):
        if i < len(gemini_models):
            m = gemini_models[i]
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=COLORS[m], markersize=6,
                                          markeredgecolor='black', markeredgewidth=0.8))
            legend_labels.append(MODEL_NAMES[m])
        else:
            legend_elements.append(Line2D([0], [0], linestyle='None'))
            legend_labels.append(' ')

    # Column 3: Protocol lines
    legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5))
    legend_labels.append('With Protocol')
    legend_elements.append(Line2D([0], [0], color='gray', linestyle=':', linewidth=2.5))
    legend_labels.append('Without Protocol')
    for _ in range(max_rows - 2):
        legend_elements.append(Line2D([0], [0], linestyle='None'))
        legend_labels.append(' ')

    print(f"  Fig1 legend (col-major): {len(legend_elements)} items, labels: {legend_labels}")

    fig.legend(legend_elements, legend_labels, loc='lower center',
              bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=6,
              frameon=True, framealpha=0.95, edgecolor='black',
              columnspacing=0.8, handletextpad=0.3)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig1_png = results_dir / "fig1_accuracy_vs_latency.png"
    fig1_pdf = results_dir / "fig1_accuracy_vs_latency.pdf"
    plt.savefig(fig1_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(fig1_pdf, bbox_inches='tight', facecolor='white')
    print(f"Created: {fig1_png}")
    print(f"Created: {fig1_pdf}")
    plt.close()


# ============================================================================
# FIGURE 2: FAILURE BREAKDOWN
# ============================================================================

print("\n=== Generating Figure 2: Failure Breakdown ===")

if planner_data:
    # Failure type mapping
    failure_type_map = {
        'syntax': 'Syntax Error',
        'types': 'Type Error',
        'ordering': 'Tool Ordering',
        'pre-tools': 'Tool Ordering',
        'state-access': 'State Mutation',
        'generation': 'Timeout',
    }

    failure_colors = {
        'Syntax Error': '#E74C3C',
        'Type Error': '#F39C12',
        'State Mutation': '#3498DB',
        'Tool Ordering': '#9B59B6',
        'Timeout': '#95A5A6',
    }

    # Collect failure data aggregated across all models
    failure_data_agg = {True: defaultdict(int), False: defaultdict(int)}
    total_runs_agg = {True: 0, False: 0}
    total_failures_agg = {True: 0, False: 0}

    for result in planner_data:
        model = result["config"]["model"]
        if model not in MODELS:  # Use all models
            continue

        with_protocol = result["config"]["with_protocol"]
        total_runs_agg[with_protocol] += 1

        if not result["codecheck"]["overall_pass"]:
            total_failures_agg[with_protocol] += 1
            if 'failure_types' in result["codecheck"] and result["codecheck"]["failure_types"]:
                for failure_type in result["codecheck"]["failure_types"]:
                    failure_display = failure_type_map.get(failure_type, 'Timeout')
                    failure_data_agg[with_protocol][failure_display] += 1
            else:
                failure_data_agg[with_protocol]['Timeout'] += 1

    # Create single horizontal stacked bar chart with 2 bars
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('white')

    y_positions = [0, 1]
    y_labels = ['With Protocol', 'Without Protocol']
    bar_height = 0.6

    # Order failure types
    failure_types_ordered = ['Tool Ordering', 'Timeout', 'Syntax Error', 'Type Error', 'State Mutation']

    # Print failure breakdown percentages
    for with_protocol in [True, False]:
        total_failure_types = sum(failure_data_agg[with_protocol].values())
        protocol_label = "With Protocol" if with_protocol else "Without Protocol"
        print(f"  {protocol_label} failure breakdown:")
        for failure_type in failure_types_ordered:
            count = failure_data_agg[with_protocol].get(failure_type, 0)
            pct = (count / total_failure_types * 100) if total_failure_types > 0 else 0
            print(f"    {failure_type}: {pct:.1f}%")

    for i, with_protocol in enumerate([True, False]):
        total_failure_types = sum(failure_data_agg[with_protocol].values())
        if total_failure_types == 0:
            continue

        left = 0
        for failure_type in failure_types_ordered:
            count = failure_data_agg[with_protocol].get(failure_type, 0)
            proportion = count / total_failure_types

            if proportion > 0:
                ax.barh(i, proportion, left=left, height=bar_height,
                       color=failure_colors.get(failure_type, '#95A5A6'),
                       edgecolor='white', linewidth=1.5)
                left += proportion

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Failure Rate', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)
    ax.invert_yaxis()

    # Create legend
    unique_labels = failure_types_ordered
    unique_handles = [plt.Rectangle((0, 0), 1, 1,
                                    facecolor=failure_colors.get(label, '#95A5A6'),
                                    edgecolor='white', linewidth=1)
                     for label in unique_labels]

    ax.legend(unique_handles, unique_labels, loc='upper center',
             bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=10,
             frameon=True, framealpha=0.95, edgecolor='black')

    plt.tight_layout()
    fig2_png = results_dir / "fig2_failure_breakdown.png"
    fig2_pdf = results_dir / "fig2_failure_breakdown.pdf"
    plt.savefig(fig2_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(fig2_pdf, bbox_inches='tight', facecolor='white')
    print(f"Created: {fig2_png}")
    print(f"Created: {fig2_pdf}")
    plt.close()


# ============================================================================
# FIGURE 6: SCHEDULER ACCURACY VS LATENCY (ALL EXPERIMENTS)
# ============================================================================

print("\n=== Generating Figure 6: Accuracy vs Latency (Scheduler) ===")

# Check if scheduler data exists
import csv
all_results_csv = results_dir / "all_results.csv"
all_tasks_xlsx = results_dir / "all_tasks.xlsx"

if all_results_csv.exists():
    # Load scheduler data
    with open(all_results_csv) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Load optimal strategy mapping from all_tasks.xlsx
    optimal_strategy_map = {}
    if all_tasks_xlsx.exists():
        import openpyxl
        wb = openpyxl.load_workbook(all_tasks_xlsx)
        ws = wb.active
        # Headers: ['Source', 'Id', 'Task', 'Supported by Tools?', 'Can Parallelize?', 'Optimal', ...]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[1] and row[5]:  # Id and Optimal columns
                task_id = row[1]
                optimal = row[5]
                # Map optimal strategy name to stage_name
                if optimal == 'Sequential':
                    optimal_strategy_map[task_id] = 'baseline'
                elif optimal == 'Parallel':
                    optimal_strategy_map[task_id] = 'task_parallelism_only'
                elif optimal == 'Hedged':
                    optimal_strategy_map[task_id] = 'first_of_n_only'

    # More contrasted blues for GPT models as requested
    model_info_fig6 = {
        'blast-gpt4.1': ('GPT-4.1', '#2E86DE'),  # Brighter blue
        'blast-gpt4.1-mini': ('GPT-4.1-mini', '#82CCDD'),  # Light blue
        'blast-gemini': ('Gemini-2.5-Pro', '#10AC84'),  # Green
        'anthropic-cua': ('Anthropic CUA', '#9B59B6'),  # Purple
        'openai-cua': ('OpenAI CUA', '#E67E22'),  # Orange
    }

    cua_models = {'anthropic-cua', 'openai-cua'}

    # Map stage_name to strategy name and marker
    strategy_map = {
        'baseline': ('Serial', '^'),
        'task_parallelism_only': ('Parallel', 's'),
        'first_of_n_only': ('Hedge', 'o'),
    }

    # Group data by (model, strategy, task)
    task_stage_data = defaultdict(lambda: {'latencies': [], 'successes': []})

    for row in all_rows:
        try:
            model_source = row['model_source']
            stage = row['stage_name']
            task_id = row['task_id']
            latency = float(row['total_time_seconds'])
            success = row['evaluated_success'].lower() == 'true'

            if model_source not in model_info_fig6 or stage not in strategy_map:
                continue

            key = (model_source, stage, task_id)
            task_stage_data[key]['latencies'].append(latency)
            task_stage_data[key]['successes'].append(1 if success else 0)
        except (ValueError, KeyError):
            continue

    # Compute aggregate statistics for each (model, strategy) pair
    strategy_stats = defaultdict(lambda: {'latencies': [], 'successes': []})

    for (model_source, stage, task_id), data in task_stage_data.items():
        if data['latencies']:
            avg_latency = sum(data['latencies']) / len(data['latencies'])
            accuracy = sum(data['successes']) / len(data['successes'])
            strategy_stats[(model_source, stage)]['latencies'].append(avg_latency)
            strategy_stats[(model_source, stage)]['successes'].append(accuracy)

    # Prepare plot data
    plot_data = []
    for (model_source, stage), data in sorted(strategy_stats.items()):
        if not data['latencies']:
            continue

        model_name, color = model_info_fig6[model_source]
        strategy_name, marker = strategy_map[stage]

        avg_latency = sum(data['latencies']) / len(data['latencies'])
        avg_accuracy = sum(data['successes']) / len(data['successes'])

        plot_data.append({
            'model': model_name,
            'model_source': model_source,
            'strategy': strategy_name,
            'stage': stage,
            'color': color,
            'marker': marker,
            'latency': avg_latency,
            'accuracy': avg_accuracy,
        })

    # Compute JIT-Scheduler (uses optimal strategy per task as specified in all_tasks.xlsx)
    jit_scheduler = {}
    if optimal_strategy_map:
        for model_source, (model_name, color) in model_info_fig6.items():
            if model_source in cua_models:
                continue

            tasks = set(task_id for (m, _, task_id) in task_stage_data.keys() if m == model_source)
            per_task_latencies = []
            per_task_successes = []

            for task_id in tasks:
                # Use the optimal strategy for this task
                if task_id in optimal_strategy_map:
                    optimal_stage = optimal_strategy_map[task_id]
                    key = (model_source, optimal_stage, task_id)
                    if key in task_stage_data and task_stage_data[key]['latencies']:
                        avg_lat = sum(task_stage_data[key]['latencies']) / len(task_stage_data[key]['latencies'])
                        avg_suc = sum(task_stage_data[key]['successes']) / len(task_stage_data[key]['successes'])
                        per_task_latencies.append(avg_lat)
                        per_task_successes.append(avg_suc)

            if per_task_latencies:
                jit_scheduler[model_name] = {
                    'latency': sum(per_task_latencies) / len(per_task_latencies),
                    'accuracy': sum(per_task_successes) / len(per_task_successes),
                    'color': color,
                }

    # Compute oracle-scheduler (minimum latency per task, then average)
    oracle_scheduler = {}
    for model_source, (model_name, color) in model_info_fig6.items():
        if model_source in cua_models:
            continue

        tasks = set(task_id for (m, _, task_id) in task_stage_data.keys() if m == model_source)
        per_task_min_latencies = []
        per_task_successes = []

        for task_id in tasks:
            min_latency = float('inf')
            min_success = 0

            for stage in strategy_map.keys():
                key = (model_source, stage, task_id)
                if key in task_stage_data and task_stage_data[key]['latencies']:
                    avg_lat = sum(task_stage_data[key]['latencies']) / len(task_stage_data[key]['latencies'])
                    avg_suc = sum(task_stage_data[key]['successes']) / len(task_stage_data[key]['successes'])

                    if avg_lat < min_latency:
                        min_latency = avg_lat
                        min_success = avg_suc

            if min_latency < float('inf'):
                per_task_min_latencies.append(min_latency)
                per_task_successes.append(min_success)

        if per_task_min_latencies:
            oracle_scheduler[model_name] = {
                'latency': sum(per_task_min_latencies) / len(per_task_min_latencies),
                'accuracy': sum(per_task_successes) / len(per_task_successes),
                'color': color,
            }

    # Create figure with broken x-axis: main plot (60-160s) and outlier region (280-320s)
    # Wider subplots as requested
    fig6, (ax_main, ax_break) = plt.subplots(1, 2, figsize=(6, 4.5), sharey=True,
                                              gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.05})
    fig6.patch.set_facecolor('white')

    # Plot all strategy points on BOTH axes
    for point in plot_data:
        for ax in [ax_main, ax_break]:
            ax.scatter(
                point['latency'], point['accuracy'],
                color=point['color'], marker=point['marker'], s=100,
                alpha=0.85, edgecolors='black', linewidth=1.0
            )

    # Plot JIT-Scheduler points (diamond marker)
    jit_scheduler_points_by_model = {}
    for model, point in jit_scheduler.items():
        for ax in [ax_main, ax_break]:
            ax.scatter(
                point['latency'], point['accuracy'],
                color=point['color'], marker='D', s=100,
                edgecolors='black', linewidth=1.0, alpha=0.85, zorder=9
            )
        jit_scheduler_points_by_model[model] = point

    # Plot oracle-scheduler points (star marker)
    oracle_scheduler_points_by_model = {}
    for model, point in oracle_scheduler.items():
        for ax in [ax_main, ax_break]:
            ax.scatter(
                point['latency'], point['accuracy'],
                color=point['color'], marker='*', s=140,
                edgecolors='black', linewidth=1.0, alpha=0.85, zorder=10
            )
        oracle_scheduler_points_by_model[model] = point

    # Plot lines through sorted dots for each model
    for model_source, (model_name, color) in model_info_fig6.items():
        if model_source in cua_models:
            continue

        model_points = [p for p in plot_data if p['model_source'] == model_source]

        if model_name in jit_scheduler_points_by_model:
            jit_pt = jit_scheduler_points_by_model[model_name]
            model_points.append({'latency': jit_pt['latency'], 'accuracy': jit_pt['accuracy']})

        if model_name in oracle_scheduler_points_by_model:
            oracle_pt = oracle_scheduler_points_by_model[model_name]
            model_points.append({'latency': oracle_pt['latency'], 'accuracy': oracle_pt['accuracy']})

        if len(model_points) >= 2:
            model_points_sorted = sorted(model_points, key=lambda p: p['latency'])
            latencies = [p['latency'] for p in model_points_sorted]
            accuracies = [p['accuracy'] for p in model_points_sorted]
            for ax in [ax_main, ax_break]:
                ax.plot(latencies, accuracies, color=color, linestyle='-', linewidth=1.5, alpha=0.4, zorder=2)

    # Configure axes
    ax_main.set_xlim(60, 160)
    ax_break.set_xlim(280, 320)
    for ax in [ax_main, ax_break]:
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=7)

    # Set custom x ticks to avoid the break area (drop ticks near edges)
    ax_main.set_xticks([80, 100, 120, 140])  # Skip 60 and 160 near edges
    ax_break.set_xticks([300])  # Single tick in the middle of outlier region

    ax_main.set_ylabel('Accuracy', fontsize=8, fontweight='bold')
    ax_main.set_xlabel('Latency (s)', fontsize=8, fontweight='bold')
    ax_break.set_xlabel('')

    # Hide spines between the two plots to show break
    ax_main.spines['right'].set_visible(False)
    ax_break.spines['left'].set_visible(False)
    ax_break.tick_params(left=False)

    # Add break indicators (diagonal lines)
    d = 0.015  # size of diagonal lines
    kwargs = dict(transform=ax_main.transAxes, color='k', clip_on=False, linewidth=1)
    ax_main.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax_main.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    kwargs.update(transform=ax_break.transAxes)
    ax_break.plot((-d*4, +d*4), (-d, +d), **kwargs)
    ax_break.plot((-d*4, +d*4), (1-d, 1+d), **kwargs)

    # Create custom legend with models and strategies
    legend_elements = []
    legend_labels = []

    # Column 1: Models
    for model_source, (model_name, color) in model_info_fig6.items():
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='none'))
        legend_labels.append(model_name)

    # Column 2: Strategies (full names as requested)
    strategy_markers = [
        ('^', 'Serial'),
        ('s', 'Parallel'),
        ('o', 'Hedge'),
        ('D', 'JIT-Scheduler'),
        ('*', 'Oracle-Scheduler'),
    ]

    for marker, label in strategy_markers:
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w',
                                         markerfacecolor='none', markeredgecolor='black',
                                         markersize=6, markeredgewidth=1.0))
        legend_labels.append(label)

    # Move legend down and center it (shift right to 0.5)
    fig6.legend(legend_elements, legend_labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=6,
               frameon=True, framealpha=0.95, edgecolor='black',
               columnspacing=1.0, handletextpad=0.3)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    fig6_png = results_dir / "fig6_accuracy_vs_latency_all.png"
    fig6_pdf = results_dir / "fig6_accuracy_vs_latency_all.pdf"
    plt.savefig(fig6_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(fig6_pdf, bbox_inches='tight', facecolor='white')
    print(f"Created: {fig6_png}")
    print(f"Created: {fig6_pdf}")

    # Export CSV for Figure 6
    fig6_csv = results_dir / "fig6_accuracy_vs_latency.csv"
    with open(fig6_csv, 'w') as f:
        f.write("Model,Strategy,Latency,Accuracy\n")
        # Write all strategy points
        for point in plot_data:
            f.write(f"{point['model']},{point['strategy']},{point['latency']:.2f},{point['accuracy']:.3f}\n")
        # Write JIT-Scheduler points
        for model, point in jit_scheduler.items():
            f.write(f"{model},JIT-Scheduler,{point['latency']:.2f},{point['accuracy']:.3f}\n")
        # Write Oracle-Scheduler points
        for model, point in oracle_scheduler.items():
            f.write(f"{model},Oracle-Scheduler,{point['latency']:.2f},{point['accuracy']:.3f}\n")
    print(f"Created: {fig6_csv}")

    plt.close()
else:
    print("  Skipping: all_results.csv not found")


# ============================================================================
# COPY FIGURES AND TABLES TO OUTPUT DIRECTORY
# ============================================================================

if COPY_TO_DIR:
    COPY_TO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Copying figures and tables to {COPY_TO_DIR} ===")

    outputs = [
        "fig1_accuracy_vs_latency.png",
        "fig1_accuracy_vs_latency.pdf",
        "fig2_failure_breakdown.png",
        "fig2_failure_breakdown.pdf",
        "fig3_candidate_efficiency.png",
        "fig3_candidate_efficiency.pdf",
        "fig4_retry_comparison.png",
        "fig4_retry_comparison.pdf",
        "fig5_latency_breakdown.png",
        "fig5_latency_breakdown.pdf",
        "fig6_accuracy_vs_latency_all.png",
        "fig6_accuracy_vs_latency_all.pdf",
    ]

    for output in outputs:
        src = results_dir / output
        if src.exists():
            dst = COPY_TO_DIR / output
            shutil.copy2(src, dst)
            print(f"  Copied: {output}")

    # Copy tables
    table1_src = Path("papers/icml26/6945e24f11603a0b74fd3186/table1_results.tex")
    if table1_src.exists():
        table1_dst = COPY_TO_DIR / "table1_results.tex"
        shutil.copy2(table1_src, table1_dst)
        print(f"  Copied: table1_results.tex")


print("\n=== All Figures and Tables Generated Successfully ===")
