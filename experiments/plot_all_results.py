#!/usr/bin/env python3
"""
Unified plotting script for dashdish-deepresearch1 results.
Generates all figures with consistent styling.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from math import comb

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# ============================================================================
# DATA LOADING
# ============================================================================

results_dir = Path("experiments/results")

# Load main evaluation data (64 runs per config, no retries)
# planner_data = json.load(open(results_dir / "dashdish-deepresearch1_20251222_004633.json"))
# planner_data = json.load(open(results_dir / "dashdish-deepresearch1_20251224_123503.json"))  # Old: has negative latencies
planner_data = json.load(open(results_dir / "dashdish-deepresearch1_20251229_180533.json"))  # NEW: all positive latencies

# Load retry comparison data (gemini-2.5-flash, max_iterations=3, newly generated Dec 29)
retry_data = json.load(open(results_dir / "dashdish-deepresearch1_20251229_234920.json"))

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Models to include
MODELS = [
    'openai/gpt-oss-120b',
    # 'gpt-4.1-mini',
    'gpt-4.1',
    'gpt-5-mini',
    'gpt-5',
    # 'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.5-flash',
    # 'gemini-2.5-flash-lite',
    'gemini-2.5-pro',
]

# Color scheme - more dramatic shading differences
COLORS = {
    # GPT models (5) - Blues with progressive shading
    'openai/gpt-oss-120b': '#AED6F1',   # Lightest blue
    'gpt-4.1-mini': '#B0D8F3',          # Light blue
    'gpt-4.1': '#5499C7',               # Light-medium blue
    'gpt-5-mini': '#2874A6',            # Medium blue
    'gpt-5': '#154360',                 # Dark blue
    
    # Gemini models (5) - Greens with progressive shading
    'gemini-2.0-flash-lite': '#ABEBC6',  # Lightest green
    'gemini-2.0-flash': '#82E0AA',       # Light green
    'gemini-2.5-flash-lite': '#58D68D',  # Light-medium green
    'gemini-2.5-flash': '#28B463',       # Medium green
    'gemini-2.5-pro': '#196F3D',         # Dark green
}

# Protocol styles for lines and legends (global - used throughout all figures)
PROTOCOL_STYLE_WITH = (0, (1, 1))      # Nearly solid (short dash) - with protocol
PROTOCOL_STYLE_WITHOUT = (0, (2, 2))   # Longer dash - without protocol

# Line styles for plots
LINE_STYLES = {
    True: PROTOCOL_STYLE_WITH,
    False: PROTOCOL_STYLE_WITHOUT,
}

# Line styles for legend display (same as plots)
LEGEND_LINE_STYLES = {
    True: PROTOCOL_STYLE_WITH,
    False: PROTOCOL_STYLE_WITHOUT,
}

# Legend positioning (consistent across all figures)
LEGEND_Y_ANCHOR = -0.12
LEGEND_BOTTOM_ADJUST = 0.18

# Model display names
MODEL_NAMES = {
    'openai/gpt-oss-120b': 'GPT-OSS-120B',
    'gpt-4.1-mini': 'GPT-4.1-mini',
    'gpt-4.1': 'GPT-4.1',
    'gpt-5-mini': 'GPT-5-mini',
    'gpt-5': 'GPT-5',
    'gemini-2.0-flash-lite': 'Gemini-2.0-Flash-Lite',
    'gemini-2.0-flash': 'Gemini-2.0-Flash',
    'gemini-2.5-flash-lite': 'Gemini-2.5-Flash-Lite',
    'gemini-2.5-flash': 'Gemini-2.5-Flash',
    'gemini-2.5-pro': 'Gemini-2.5-Pro',
}

# ============================================================================
# FIGURE 1: ACCURACY VS LATENCY
# ============================================================================

print("\n=== Generating Figure 1: Accuracy vs Latency ===")

fig, (ax_gpt, ax_gemini) = plt.subplots(1, 2, figsize=(12, 6))
fig.patch.set_facecolor('white')

# Organize data
for model in MODELS:
    model_data = [r for r in planner_data if r['config']['model'] == model]
    
    for with_protocol in [True, False]:
        protocol_data = [r for r in model_data if r['config']['with_protocol'] == with_protocol]
        
        if not protocol_data:
            continue
        
        # Calculate metrics
        passed = sum(1 for r in protocol_data if r['codecheck']['overall_pass'])
        total = len(protocol_data)
        accuracy = passed / total if total > 0 else 0
        avg_latency = np.mean([r['generation_latency'] for r in protocol_data])
        
        # Determine subplot
        is_gpt = model.startswith('gpt') or model.startswith('openai')
        ax = ax_gpt if is_gpt else ax_gemini
        
        # Plot point
        color = COLORS[model]
        marker = 'o' if with_protocol else 's'
        ax.scatter(avg_latency, accuracy, color=color, s=120, marker=marker,
                  alpha=0.85, edgecolors='black', linewidth=1.2, zorder=5)

# Connect points for each model - draw 2 lines per subplot (with/without protocol)
for ax, model_prefix in [(ax_gpt, ('gpt', 'openai')), (ax_gemini, 'gemini')]:
    # Collect all points for this subplot grouped by protocol
    for with_protocol in [True, False]:
        points = []
        for model in MODELS:
            if model.startswith(model_prefix):
                protocol_data = [r for r in planner_data 
                               if r['config']['model'] == model 
                               and r['config']['with_protocol'] == with_protocol]
                if protocol_data:
                    passed = sum(1 for r in protocol_data if r['codecheck']['overall_pass'])
                    accuracy = passed / len(protocol_data)
                    avg_latency = np.mean([r['generation_latency'] for r in protocol_data])
                    points.append((avg_latency, accuracy, model))
        
        # Draw one line connecting all points for this protocol
        if len(points) >= 2:
            points.sort(key=lambda x: x[0])  # Sort by latency
            style = LINE_STYLES[with_protocol]
            ax.plot([p[0] for p in points], [p[1] for p in points],
                   color='#555555', linestyle=style,
                   linewidth=2.5, alpha=0.6, zorder=3)
    
    ax.set_xlabel('Planning Latency (seconds)', fontsize=12, fontweight='bold')
    if ax == ax_gpt:
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel('')  # No label for right subplot
        ax.set_yticklabels([])  # No tick labels for right subplot
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)

# Create custom legend with 3 columns
# Column 1: GPT models, Column 2: Gemini models, Column 3: Protocol styles
legend_elements = []
legend_labels = []

# Column 1: GPT models
gpt_models = [m for m in MODELS if m.startswith('gpt') or m.startswith('openai')]
for model in gpt_models:
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=COLORS[model], markersize=8,
                                     markeredgecolor='black', markeredgewidth=1))
    legend_labels.append(MODEL_NAMES[model])

# Column 2: Gemini models at top
gemini_models = [m for m in MODELS if m.startswith('gemini')]
for model in gemini_models:
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=COLORS[model], markersize=8,
                                     markeredgecolor='black', markeredgewidth=1))
    legend_labels.append(MODEL_NAMES[model])

# Column 3: Protocol line styles at top (without spacing before them)
legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=LINE_STYLES[True],
                                 linewidth=2.5))
legend_labels.append('With Protocol')

legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=LINE_STYLES[False],
                                 linewidth=2.5))
legend_labels.append('Without Protocol')

# Add spacing in column 2 to align with column 1
legend_elements.append(plt.Line2D([0], [0], color='none'))
legend_labels.append('')

fig.legend(legend_elements, legend_labels, loc='lower center',
           bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=10,
           frameon=True, framealpha=0.95, edgecolor='black',
           columnspacing=2.0, handletextpad=0.5)

plt.tight_layout(rect=[0, 0.13, 1, 1])
fig1_png = results_dir / 'fig1_accuracy_vs_latency.png'
fig1_pdf = results_dir / 'fig1_accuracy_vs_latency.pdf'
plt.savefig(fig1_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig1_pdf, bbox_inches='tight', facecolor='white')
print(f"Created/Updated:\n  {fig1_png}\n  {fig1_pdf}")
plt.close()

# ============================================================================
# FIGURE 2: FAILURE BREAKDOWN (SINGLE HORIZONTAL STACKED BARS, RATES)
# ============================================================================

print("\n=== Generating Figure 2: Failure Breakdown ===")

# Failure type mapping (no 'Other')
failure_type_map = {
    'syntax': 'Syntax Error',
    'types': 'Type Error',
    'ordering': 'Tool Ordering',
    'state-access': 'State Mutation',
    'generation': 'Timeout',
}

failure_colors = {
    'Syntax Error': '#E74C3C',
    'Type Error': '#F39C12',
    'State Mutation': '#3498DB',
    'Tool Ordering': '#9B59B6',
    'Timeout': '#95A5A6'
}

# Collect failure data aggregated across all models
failure_data_agg = {True: defaultdict(int), False: defaultdict(int)}
total_runs_agg = {True: 0, False: 0}
total_failures_agg = {True: 0, False: 0}

for result in planner_data:
    model = result['config']['model']
    if model not in MODELS:
        continue
    
    with_protocol = result['config']['with_protocol']
    total_runs_agg[with_protocol] += 1
    
    if not result['codecheck']['overall_pass']:
        total_failures_agg[with_protocol] += 1
        # Count all failure types in the list
        if 'failure_types' in result['codecheck'] and result['codecheck']['failure_types']:
            for failure_type in result['codecheck']['failure_types']:
                failure_display = failure_type_map.get(failure_type, 'Timeout')
                failure_data_agg[with_protocol][failure_display] += 1
        else:
            # Default to timeout if no failure types
            failure_data_agg[with_protocol]['Timeout'] += 1

# Create single horizontal stacked bar chart with 2 bars
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('white')

y_positions = [0, 1]
y_labels = ['With Protocol', 'Without Protocol']
bar_height = 0.6

# Order failure types by total frequency (most to least)
# Based on data analysis: Tool Ordering (395), Timeout (48), Syntax Error (15), Type Error (9)
failure_types_ordered = ['Tool Ordering', 'Timeout', 'Syntax Error', 'Type Error', 'State Mutation']

for i, with_protocol in enumerate([True, False]):
    # Calculate total failure type counts (sum of all failure types)
    total_failure_types = sum(failure_data_agg[with_protocol].values())
    if total_failure_types == 0:
        continue
    
    # Stack failures from left to right as proportions of all failure type occurrences
    left = 0
    for failure_type in failure_types_ordered:
        count = failure_data_agg[with_protocol].get(failure_type, 0)
        proportion = count / total_failure_types  # Proportion of this failure type
        
        if proportion > 0:
            ax.barh(i, proportion, left=left, height=bar_height,
                   color=failure_colors.get(failure_type, '#95A5A6'),
                   edgecolor='white', linewidth=1.5)
            left += proportion

# Configure axes
ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels)
ax.set_xlabel('Failure Rate', fontsize=11, fontweight='bold')
ax.set_xlim(0, 1.0)
ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)
ax.invert_yaxis()

# Create legend - centered below plot with 3 columns
unique_labels = failure_types_ordered
unique_handles = [plt.Rectangle((0, 0), 1, 1, 
                                facecolor=failure_colors.get(label, '#95A5A6'),
                                edgecolor='white', linewidth=1)
                 for label in unique_labels]

ax.legend(unique_handles, unique_labels, loc='upper center',
         bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=10,
         frameon=True, framealpha=0.95, edgecolor='black',
         columnspacing=1.5, handletextpad=0.5)

plt.tight_layout(rect=[0, 0.14, 1, 1])

plt.tight_layout()
fig2_png = results_dir / 'fig2_failure_breakdown.png'
fig2_pdf = results_dir / 'fig2_failure_breakdown.pdf'
plt.savefig(fig2_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig2_pdf, bbox_inches='tight', facecolor='white')
print(f"Created/Updated:\n  {fig2_png}\n  {fig2_pdf}")
plt.close()

# ============================================================================
# FIGURE 3: PASS@K AND PASS@T (Candidate Efficiency Analysis)
# ============================================================================

print("\n=== Generating Figure 3: Pass@k and Pass@t (2x2 layout: GPT top, Gemini bottom) ===")

# Create 2x2 grid: top row for GPT models, bottom row for Gemini models
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.patch.set_facecolor('white')

# Axes mapping
ax_passk_gpt, ax_passt_gpt = axs[0, 0], axs[0, 1]
ax_passk_gem, ax_passt_gem = axs[1, 0], axs[1, 1]

# Collect data for each model configuration and plot on appropriate subplot
for model in MODELS:
    model_data = [r for r in planner_data if r['config']['model'] == model]
    
    for with_protocol in [True, False]:
        protocol_data = [r for r in model_data if r['config']['with_protocol'] == with_protocol]
        
        if not protocol_data:
            continue
        
        n = len(protocol_data)
        c = sum(1 for r in protocol_data if r['codecheck']['overall_pass'])
        p = c / n if n > 0 else 0
        
        # Choose subplot based on model family
        is_gpt = model.startswith('gpt') or model.startswith('openai')
        ax_k = ax_passk_gpt if is_gpt else ax_passk_gem
        ax_t = ax_passt_gpt if is_gpt else ax_passt_gem

        # --- Pass@k (left column) ---
        k_values = range(1, 33)  # Focus on 1-32 candidates
        pass_at_k = []
        for k in k_values:
            if c == 0:
                pass_at_k.append(0.0)
            elif n - c < k:
                pass_at_k.append(1.0)
            else:
                pass_at_k.append(1 - comb(n - c, k) / comb(n, k))

        color = COLORS[model]
        linestyle = LINE_STYLES[with_protocol]
        label = f"{MODEL_NAMES.get(model, model)}"

        ax_k.plot(k_values, pass_at_k, color=color, linestyle=linestyle,
                  linewidth=2, alpha=0.85, label=label)

        # --- Pass@t (right column) ---
        latencies = [r['generation_latency'] for r in protocol_data if r['generation_latency'] > 0]
        if latencies:
            t_max = max(latencies)
            t_values = np.linspace(0, min(t_max, 100), 200)  # Cap at 100s for visibility

            pass_at_t = []
            for t in t_values:
                F_t = sum(1 for lat in latencies if lat <= t) / len(latencies)
                n_parallel = 64
                pass_at_t.append(1 - (1 - F_t * p) ** n_parallel)

            ax_t.plot(t_values, pass_at_t, color=color, linestyle=linestyle,
                      linewidth=2, alpha=0.85, label=label)

# Configure axes - left column (Pass@k)
for ax in [ax_passk_gpt, ax_passk_gem]:
    ax.set_xlabel('k', fontsize=11, fontweight='bold')
    ax.set_ylabel('Pass@k', fontsize=11, fontweight='bold')
    ax.set_xlim(1, 32)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)

# Configure axes - right column (Pass@t)
for ax in [ax_passt_gpt, ax_passt_gem]:
    ax.set_xlabel('t (seconds)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Pass@t', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    # Keep x-limits automatic; no axis break applied here for clarity

# Remove the 95% horizontal dashed reference line (not needed)

# Create combined legend (reuse approach from Figure 1)
legend_elements = []
legend_labels = []

# Column 1: GPT models
gpt_models = [m for m in MODELS if m.startswith('gpt') or m.startswith('openai')]
for model in gpt_models:
    legend_elements.append(plt.Line2D([0], [0], color=COLORS[model], linewidth=2.5))
    legend_labels.append(MODEL_NAMES.get(model, model))

# Column 2: Gemini models
gemini_models = [m for m in MODELS if m.startswith('gemini')]
for model in gemini_models:
    legend_elements.append(plt.Line2D([0], [0], color=COLORS[model], linewidth=2.5))
    legend_labels.append(MODEL_NAMES.get(model, model))

# Protocol styles at top of column 3
legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=LEGEND_LINE_STYLES[True], linewidth=2.5))
legend_labels.append('With Protocol')
legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=LEGEND_LINE_STYLES[False], linewidth=2.5))
legend_labels.append('Without Protocol')

# Spacer
legend_elements.append(plt.Line2D([0], [0], color='none'))
legend_labels.append('')

fig.legend(legend_elements, legend_labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10,
           frameon=True, framealpha=0.95, edgecolor='black',
           columnspacing=2.0, handletextpad=0.5)

plt.tight_layout(rect=[0, 0.12, 1, 1])
fig3_png = results_dir / 'fig3_candidate_efficiency.png'
fig3_pdf = results_dir / 'fig3_candidate_efficiency.pdf'
plt.savefig(fig3_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig3_pdf, bbox_inches='tight', facecolor='white')
print(f"Created/Updated:\n  {fig3_png}\n  {fig3_pdf}")
plt.close()

# ============================================================================
# FIGURE 4: SERIAL RETRY VS PARALLEL HEDGING (Empirical Comparison)
# ============================================================================

print("\n=== Generating Figure 4: Serial Retry vs Parallel Hedging ===")

# EMPIRICAL COMPARISON using gemini-2.5-flash-lite data:
# - Serial retry: Up to 3 sequential attempts, each run has total latency
# - Parallel hedging: k parallel attempts, we compute Pass@t from single attempts
# All curves show: at time t, what's the probability of having at least one success?
# Show all 4 curves on same plot: serial/parallel × with/without protocol

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
fig.patch.set_facecolor('white')

# Use gemini-2.5-flash for comparison (now we have both single and retry data for this model)
comparison_model = 'gemini-2.5-flash'

# Colors and styles for 4 curves
color_retry = '#E74C3C'    # Red - Serial retry
color_parallel = '#27AE60' # Green - Parallel hedging

# Collect all curves
all_curves = []
t_max_global = 0

for with_protocol in [True, False]:
    # Get single attempt data
    single_data = [r for r in planner_data 
                   if r['config']['model'] == comparison_model
                   and r['config']['with_protocol'] == with_protocol]
    
    # Get retry data
    retry_data_filtered = [r for r in retry_data
                          if r['config']['model'] == comparison_model
                          and r['config']['with_protocol'] == with_protocol]
    
    if not single_data or not retry_data_filtered:
        continue
    
    # Single attempt stats
    single_passed = sum(1 for r in single_data if r['codecheck']['overall_pass'])
    single_success_rate = single_passed / len(single_data)
    single_latencies = [r['generation_latency'] for r in single_data if r['generation_latency'] > 0]
    
    # Retry stats - EMPIRICAL distribution
    retry_passed = sum(1 for r in retry_data_filtered if r['codecheck']['overall_pass'])
    retry_success_rate = retry_passed / len(retry_data_filtered)
    retry_latencies = [r['generation_latency'] for r in retry_data_filtered if r['generation_latency'] > 0]
    
    # Track max time
    t_max = max(max(single_latencies), max(retry_latencies))
    t_max_global = max(t_max_global, t_max)
    
    # Store curve data
    all_curves.append({
        'with_protocol': with_protocol,
        'single_latencies': single_latencies,
        'single_success_rate': single_success_rate,
        'retry_latencies': retry_latencies,
        'retry_success_rate': retry_success_rate,
    })

# Compute and plot all curves
t_values = np.linspace(0, t_max_global, 200)

for curve_data in all_curves:
    with_protocol = curve_data['with_protocol']
    single_latencies = curve_data['single_latencies']
    single_success_rate = curve_data['single_success_rate']
    retry_latencies = curve_data['retry_latencies']
    
    # Line style: solid for with protocol, dashed for without
    linestyle = LINE_STYLES[True] if with_protocol else LINE_STYLES[False]
    protocol_label = 'w/ protocol' if with_protocol else 'w/o protocol'
    
    # Pass@t for parallel hedging (k=8, using single attempt data)
    pass_at_t_parallel = []
    for t in t_values:
        F_t = sum(1 for lat in single_latencies if lat <= t) / len(single_latencies)
        k = 8
        pass_at_t_parallel.append(1 - (1 - F_t * single_success_rate) ** k)
    
    # Pass@t for serial retry - EMPIRICAL CDF
    pass_at_t_retry = []
    for t in t_values:
        # Fraction of retry runs that succeeded within time t
        successes_by_t = sum(1 for i, lat in enumerate(retry_latencies)
                           if lat <= t)
        # Need to check if they actually passed
        retry_results = [r for r in retry_data
                        if r['config']['model'] == comparison_model
                        and r['config']['with_protocol'] == with_protocol
                        and r['generation_latency'] > 0]
        successes_by_t = sum(1 for r in retry_results
                           if r['codecheck']['overall_pass'] and r['generation_latency'] <= t)
        pass_at_t_retry.append(successes_by_t / len(retry_results))
    
    # Plot curves
    ax.plot(t_values, pass_at_t_parallel, color=color_parallel, linewidth=2.5,
           linestyle=linestyle, label=f'Parallel k=8 ({protocol_label})', alpha=0.85)
    ax.plot(t_values, pass_at_t_retry, color=color_retry, linewidth=2.5,
           linestyle=linestyle, label=f'Serial retry ({protocol_label})', alpha=0.85)

# Formatting
ax.set_xlabel('Latency Budget (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
ax.set_xlim(0, t_max_global)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, linewidth=0.5)

# Legend below the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
         ncol=2, fontsize=10, framealpha=0.95, columnspacing=1.5)

plt.tight_layout()
fig4_png = results_dir / 'fig4_retry_comparison.png'
fig4_pdf = results_dir / 'fig4_retry_comparison.pdf'
plt.savefig(fig4_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig4_pdf, bbox_inches='tight', facecolor='white')
print(f"Created/Updated:\n  {fig4_png}\n  {fig4_pdf}")
plt.close()

# ============================================================================
# FIGURE 5: LATENCY BREAKDOWN (STACKED HORIZONTAL BARS)
# ============================================================================
# Compare latency components (planning, LLM inference, action execution)
# across different execution strategies:
# - Browser-Use (loop-baseline): Loop mode without tools
# - Browser-Use + Tools (loop-tools-baseline): Loop mode with tools
# - Cost-Optimal (gemini-2.5-flash-best): Pre-generated code, best cost
# - Cost-Worst (gemini-2.5-flash-worst): Pre-generated code, worst cost
# - Cost-Unaware (gemini-2.5-flash-serial-retry): Code generation with retries
# 
# NOTE: This figure requires e2e_detailed.json which contains timing breakdowns.
# The data is collected via measure_e2e_detailed.py.

# Try to load e2e detailed latency data from multiple tasks and aggregate
# Load all *_e2e_detailed.json files in results directory
e2e_files = list(results_dir.glob("*_e2e_detailed.json"))

if e2e_files:
    print(f"\nGenerating Figure 5: Latency Breakdown (E2E Measurements)...")
    print(f"Found {len(e2e_files)} E2E result files: {[f.name for f in e2e_files]}")

    # Aggregate results across all tasks
    # Structure: {config_name: [result1, result2, ...]} for all matching configs
    aggregated_results = defaultdict(list)

    for e2e_file in e2e_files:
        with open(e2e_file) as f:
            e2e_data = json.load(f)

        task_id = e2e_data.get('task_id', e2e_file.stem.replace('_e2e_detailed', ''))
        e2e_results = e2e_data.get('results', [])

        # Collect results for each configuration
        for r in e2e_results:
            config_name = r.get('name', '')
            if config_name:
                aggregated_results[config_name].append(r)

    # Target configurations we want to plot
    # Order: Cost-Optimal, Cost-Worst, Browser-Use (+ tool synthesis), Browser-Use
    target_names = [
        'gemini-2.5-flash-best',
        'gemini-2.5-flash-worst',
        'loop-tools-baseline',
        'loop-baseline',
    ]

    # Average timing across tasks and trials for each configuration
    selected_results = []
    config_names = []

    for target in target_names:
        if target in aggregated_results and aggregated_results[target]:
            # Average across all tasks
            all_task_results = aggregated_results[target]

            # Aggregate avg_timing fields
            planning_seconds = []
            execution_seconds = []
            llm_total_seconds = []
            llm_prefill_seconds = []
            llm_decode_seconds = []
            correctness_pcts = []

            for r in all_task_results:
                avg_t = r.get('avg_timing', {})
                planning_seconds.append(avg_t.get('planning_seconds', 0))
                execution_seconds.append(avg_t.get('execution_seconds', 0))
                llm_total_seconds.append(avg_t.get('llm_total_seconds', 0))
                llm_prefill_seconds.append(avg_t.get('llm_prefill_seconds', 0))
                llm_decode_seconds.append(avg_t.get('llm_decode_seconds', 0))
                correctness_pcts.append(r.get('avg_correctness_pct', 0))

            # Compute averages
            avg_result = {
                'name': target,
                'avg_timing': {
                    'planning_seconds': sum(planning_seconds) / len(planning_seconds) if planning_seconds else 0,
                    'execution_seconds': sum(execution_seconds) / len(execution_seconds) if execution_seconds else 0,
                    'llm_total_seconds': sum(llm_total_seconds) / len(llm_total_seconds) if llm_total_seconds else 0,
                    'llm_prefill_seconds': sum(llm_prefill_seconds) / len(llm_prefill_seconds) if llm_prefill_seconds else 0,
                    'llm_decode_seconds': sum(llm_decode_seconds) / len(llm_decode_seconds) if llm_decode_seconds else 0,
                },
                'avg_correctness_pct': sum(correctness_pcts) / len(correctness_pcts) if correctness_pcts else 0,
                'num_tasks': len(all_task_results),
            }

            selected_results.append(avg_result)
            config_names.append(target)
    
    if selected_results:
        # Create 2-subplot layout: absolute latency + percentage breakdown
        fig5 = plt.figure(figsize=(14, 5))
        fig5.patch.set_facecolor('white')
        
        # Create grid with custom width ratios
        gs = fig5.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
        ax1 = fig5.add_subplot(gs[0, 0])
        ax2 = fig5.add_subplot(gs[0, 1])
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('white')
        
        # Prepare data for stacked bar chart (reversed order)
        display_names = [
            'Cost-Optimal',
            'Cost-Worst',
            'Browser-Use\n(+ tool synthesis)',
            'Browser-Use',
        ]
        
        planning_times = []
        llm_times = []
        action_times = []
        correctness_pcts = []
        
        for result in selected_results:
            avg_timing = result.get('avg_timing', {})
            
            planning = avg_timing.get('planning_seconds', 0)
            llm_total = avg_timing.get('llm_total_seconds', 0)
            execution = avg_timing.get('execution_seconds', 0)
            action = max(0, execution - llm_total)  # Action time = execution minus LLM time
            
            planning_times.append(planning)
            llm_times.append(llm_total)
            action_times.append(action)
            correctness_pcts.append(result.get('avg_correctness_pct', 0))
        
        # Convert to numpy arrays
        planning_times = np.array(planning_times)
        llm_times = np.array(llm_times)
        action_times = np.array(action_times)
        correctness_pcts = np.array(correctness_pcts)
        
        # Calculate totals and percentages
        total_times = planning_times + llm_times + action_times
        planning_pcts = (planning_times / total_times) * 100
        llm_pcts = (llm_times / total_times) * 100
        action_pcts = (action_times / total_times) * 100
        
        y_pos = np.arange(len(display_names))
        
        # Colors for each component
        color_planning = '#FF6B6B'       # Red
        color_llm = '#4ECDC4'            # Teal
        color_action = '#95E1D3'         # Light teal
        
        # SUBPLOT 1: Latency in seconds
        ax1.barh(y_pos, planning_times, label='Planning', color=color_planning, height=0.65)
        ax1.barh(y_pos, llm_times, left=planning_times, color=color_llm, height=0.65)
        ax1.barh(y_pos, action_times, left=planning_times + llm_times, color=color_action, height=0.65)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(display_names, fontsize=11)
        ax1.set_xlabel('Latency (seconds)', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, np.max(total_times) * 1.1)
        ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # SUBPLOT 2: Percentage of total execution time (all bars same width)
        ax2.barh(y_pos, planning_pcts, label='Planning', color=color_planning, height=0.65)
        ax2.barh(y_pos, llm_pcts, left=planning_pcts, color=color_llm, height=0.65)
        ax2.barh(y_pos, action_pcts, left=planning_pcts + llm_pcts, color=color_action, height=0.65)
        
        ax2.set_yticks([])  # Drop y-axis labels on second subplot
        ax2.set_xlabel('Percentage of Execution Time', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Create legend below the plots with more space
        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=color_planning),
            plt.Rectangle((0, 0), 1, 1, fc=color_llm),
            plt.Rectangle((0, 0), 1, 1, fc=color_action),
        ]
        fig5.legend(handles, ['Planning', 'Inference', 'Action'], 
                   loc='lower center', ncol=3, fontsize=11, 
                   bbox_to_anchor=(0.5, -0.12), frameon=True, framealpha=0.95)
        
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        
        fig5_png = results_dir / 'fig5_latency_breakdown.png'
        fig5_pdf = results_dir / 'fig5_latency_breakdown.pdf'
        plt.savefig(fig5_png, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(fig5_pdf, bbox_inches='tight', facecolor='white')
        
        # Generate LaTeX table for correctness data
        latex_table = r"""\begin{table}[h]
\centering
\begin{tabular}{lc}
\toprule
\textbf{Strategy} & \textbf{Correctness} \\
\midrule
"""
        strategy_names = [
            'Cost-Optimal',
            'Cost-Worst',
            'Browser-Use + tool synthesis',
            'Browser-Use',
        ]
        
        for name, acc in zip(strategy_names, correctness_pcts):
            latex_table += f"{name} & {acc*100:.0f}\\% \\\\\n"
        
        num_tasks = selected_results[0].get('num_tasks', 1) if selected_results else 1

        latex_table += r"""\bottomrule
\end{tabular}
\caption{Correctness rates for each execution strategy, averaged across """ + f"{num_tasks} task{'s' if num_tasks != 1 else ''} and multiple trials" + r""".}
\label{table:fig5_correctness}
\end{table}
"""
        
        # Save LaTeX table
        table_path = results_dir / 'fig5_correctness_table.tex'
        with open(table_path, 'w') as f:
            f.write(latex_table)
        
        print(f"Created/Updated:\n  {fig5_png}\n  {fig5_pdf}\n  {table_path}")
        plt.close()
    else:
        print("No results found matching the expected e2e configurations")
        if aggregated_results:
            print(f"Available configurations: {list(aggregated_results.keys())}")
else:
    print(f"Note: E2E detailed latency data not found in {results_dir}")
    print("      Run: python experiments/measure_e2e_detailed.py --ids '<task-ids>' ... to generate Figure 5 data")

# ============================================================================
# FIGURE 6: ACCURACY VS LATENCY (ALL EXPERIMENTS - SCATTER PLOT)
# ============================================================================

print("\n=== Generating Figure 6: Accuracy vs Latency (All Experiments) ===")

import csv
import openpyxl
from collections import defaultdict

# Load all_results.csv
all_results_path = results_dir / "all_results.csv"

if all_results_path.exists():
    # Parse CSV
    with open(all_results_path) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    
    # Load scheduler's cost-optimal choices from all_tasks.xlsx
    xlsx_path = results_dir / "all_tasks.xlsx"
    task_to_scheduled_strategy = {}
    
    if xlsx_path.exists():
        try:
            wb = openpyxl.load_workbook(xlsx_path)
            ws = wb['all-dashdish']
            
            for row_idx in range(2, ws.max_row + 1):
                task_id = ws.cell(row_idx, 2).value  # Id column
                optimal = ws.cell(row_idx, 5).value  # Optimal column (scheduler's choice)
                
                if task_id and optimal:
                    # Map to stage names
                    optimal_lower = optimal.lower()
                    if optimal_lower == 'sequential':
                        task_to_scheduled_strategy[task_id] = 'baseline'
                    elif optimal_lower == 'parallel':
                        task_to_scheduled_strategy[task_id] = 'task_parallelism_only'
                    elif optimal_lower == 'hedged':
                        task_to_scheduled_strategy[task_id] = 'first_of_n_only'
        except Exception as e:
            print(f"  Warning: Could not load all_tasks.xlsx: {e}")
    
    # Map model identifiers to display names and colors (order matters for legend)
    from collections import OrderedDict
    model_info = OrderedDict([
        ('blast-gpt4.1', ('GPT-4.1', COLORS.get('gpt-4.1', '#5499C7'))),
        ('blast-gpt4.1-mini', ('GPT-4.1-mini', COLORS.get('gpt-4.1-mini', '#FFFFFF'))),
        ('blast-gemini', ('Gemini-2.5-Pro', COLORS.get('gemini-2.5-pro', '#196F3D'))),
        ('anthropic-cua', ('Anthropic CUA', '#9467bd')),
        ('openai-cua', ('OpenAI CUA', '#ff7f0e')),
    ])

    # CUA models (exclude from Oracle/Cost-Optimal calculations)
    cua_models = {'anthropic-cua', 'openai-cua'}

    # Map stage_name to strategy name and marker
    # Markers: Serial=triangle(^), Parallel=square(s), Hedge=circle(o)
    strategy_map = {
        'baseline': ('Serial', '^'),
        'task_parallelism_only': ('Parallel', 's'),
        'first_of_n_only': ('Hedge', 'o'),
    }
    
    # Group data by (model_source, stage, task_id) to track task-level details
    task_stage_data = defaultdict(lambda: {'latencies': [], 'successes': []})
    
    for row in all_rows:
        try:
            model_source = row['model_source']
            stage = row['stage_name']
            task_id = row['task_id']
            latency = float(row['total_time_seconds'])
            success = row['evaluated_success'].lower() == 'true'
            
            if model_source not in model_info or stage not in strategy_map:
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
    
    # Prepare plot data for the three strategies per model
    plot_data = []
    
    for (model_source, stage), data in sorted(strategy_stats.items()):
        if not data['latencies']:
            continue
        
        model_name, color = model_info[model_source]
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
            'label': f"{model_name} - {strategy_name}",
        })
    
    # Compute oracle-optimal per model (only for non-CUA models)
    # Oracle = for each task, pick the strategy with minimum latency, then average across tasks
    oracle_optimal = {}

    for model_source, (model_name, color) in model_info.items():
        if model_source in cua_models:
            continue  # Skip CUA models

        # Get all tasks for this model
        tasks = set(task_id for (m, _, task_id) in task_stage_data.keys() if m == model_source)

        per_task_min_latencies = []
        per_task_successes = []

        for task_id in tasks:
            # Find minimum latency strategy for this task
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
            oracle_optimal[model_name] = {
                'model': model_name,
                'latency': sum(per_task_min_latencies) / len(per_task_min_latencies),
                'accuracy': sum(per_task_successes) / len(per_task_successes),
                'color': color,
            }

    # Compute cost-optimal per model (only for non-CUA models): what the scheduler CHOSE
    cost_optimal = {}

    for model_source, (model_name, color) in model_info.items():
        if model_source in cua_models:
            continue  # Skip CUA models

        # Find which strategy the scheduler chose for each task
        task_latencies = []
        task_successes = []

        for (m, stage, task_id), data in task_stage_data.items():
            if m != model_source:
                continue
            if task_id not in task_to_scheduled_strategy:
                continue

            scheduled_stage = task_to_scheduled_strategy[task_id]
            if scheduled_stage == stage and data['latencies']:
                avg_latency = sum(data['latencies']) / len(data['latencies'])
                avg_accuracy = sum(data['successes']) / len(data['successes']) if data['successes'] else 0
                task_latencies.append(avg_latency)
                task_successes.append(avg_accuracy)

        if task_latencies:
            cost_optimal[model_name] = {
                'model': model_name,
                'latency': sum(task_latencies) / len(task_latencies),
                'accuracy': sum(task_successes) / len(task_successes),
                'color': color,
            }
    
    # Create figure
    fig6, ax6 = plt.subplots(figsize=(12, 7))
    fig6.patch.set_facecolor('white')
    ax6.set_facecolor('white')
    
    # Plot all strategy points (3 per model) - no labels, we'll create custom legend
    for point in plot_data:
        ax6.scatter(point['latency'], point['accuracy'],
                   color=point['color'], marker=point['marker'], s=120,
                   alpha=0.85, edgecolors='black', linewidth=1.2)

    # Plot oracle-optimal points (star marker, minimum latency for each model)
    for model, point in oracle_optimal.items():
        ax6.scatter(point['latency'], point['accuracy'],
                   color=point['color'], marker='*', s=120,
                   edgecolors='black', linewidth=1.2, alpha=0.85, zorder=10)

    # Plot cost-optimal points (diamond marker, scheduler's choice)
    for model, point in cost_optimal.items():
        ax6.scatter(point['latency'], point['accuracy'],
                   color=point['color'], marker='D', s=120,
                   edgecolors='black', linewidth=1.2, alpha=0.85, zorder=9)
    
    # Configure plot
    ax6.set_xlabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax6.set_ylim(-0.05, 1.05)
    ax6.grid(True, alpha=0.3, linestyle='--')

    # Create custom legend: models in one column, strategies in another
    legend_elements = []
    legend_labels = []

    # Column 1: Models (colored rectangles, no outline) - in order from model_info
    for model_source, (model_name, color) in model_info.items():
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='none'))
        legend_labels.append(model_name)

    # Column 2: Strategies (outlined shapes, no fill) - only show Cost/Oracle for non-CUA
    strategy_markers = [
        ('^', 'Serial'),
        ('s', 'Parallel'),
        ('o', 'Hedge'),
        ('D', 'Cost-Optimal'),
        ('*', 'Oracle-Optimal'),
    ]

    for marker, label in strategy_markers:
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w',
                                         markerfacecolor='none', markeredgecolor='black',
                                         markersize=8, markeredgewidth=1.2))
        legend_labels.append(label)

    fig6.legend(legend_elements, legend_labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=10,
               frameon=True, framealpha=0.95, edgecolor='black',
               columnspacing=2.0, handletextpad=0.5)

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    fig6_png = results_dir / 'fig6_accuracy_vs_latency_all.png'
    fig6_pdf = results_dir / 'fig6_accuracy_vs_latency_all.pdf'
    plt.savefig(fig6_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(fig6_pdf, bbox_inches='tight', facecolor='white')
    print(f"Created/Updated:\n  {fig6_png}\n  {fig6_pdf}")
    
    # Print analysis (only for non-CUA models)
    print("\n=== Strategy Analysis (Non-CUA Models Only) ===")
    print("\nOracle-Optimal (Per-Task Minimum Latency, Averaged):")
    print(f"  {'Model':<20s} {'Latency':<12s} {'Accuracy':<12s}")
    print("  " + "="*50)
    for model in sorted(oracle_optimal.keys()):
        opt = oracle_optimal[model]
        print(f"  {model:<20s} {opt['latency']:>7.1f}s {'':>3s} {opt['accuracy']:>5.1%}")

    print("\nCost-Optimal (Scheduler's Choice from all_tasks.xlsx):")
    print(f"  {'Model':<20s} {'Latency':<12s} {'Accuracy':<12s}")
    print("  " + "="*50)
    for model in sorted(cost_optimal.keys()):
        opt = cost_optimal[model]
        print(f"  {model:<20s} {opt['latency']:>7.1f}s {'':>3s} {opt['accuracy']:>5.1%}")

    print("\n=== Comparison: Oracle vs Cost Latency ===")
    print(f"  {'Model':<20s} {'Oracle (s)':<12s} {'Cost (s)':<12s} {'Difference':<20s}")
    print("  " + "="*70)
    for model in sorted(oracle_optimal.keys()):
        oracle_lat = oracle_optimal[model]['latency']
        cost_lat = cost_optimal[model]['latency'] if model in cost_optimal else float('nan')
        diff = cost_lat - oracle_lat if model in cost_optimal else float('nan')
        status = "✓ Cost ≥ Oracle" if diff >= 0 else "✗ Cost < Oracle (ERROR!)"
        print(f"  {model:<20s} {oracle_lat:>7.1f}s {'':>3s} {cost_lat:>7.1f}s {'':>3s} {status:<20s}")
    
    plt.close()
else:
    print(f"Note: All results data not found at {all_results_path}")

# ============================================================================
# GENERATE CSV TABLES FOR PAPER
# ============================================================================

print("\n=== Generating CSV Tables for Paper ===")

csv_dir = results_dir / "csv_tables"
csv_dir.mkdir(exist_ok=True)

# TABLE 1: Accuracy vs Latency by Model and Protocol (Figure 1 data)
print("\nGenerating Table 1: Accuracy vs Latency by Model and Protocol")
table1_data = []
for model in MODELS:
    model_data = [r for r in planner_data if r['config']['model'] == model]

    for with_protocol in [True, False]:
        protocol_data = [r for r in model_data if r['config']['with_protocol'] == with_protocol]

        if not protocol_data:
            continue

        passed = sum(1 for r in protocol_data if r['codecheck']['overall_pass'])
        total = len(protocol_data)
        accuracy = passed / total if total > 0 else 0
        avg_latency = np.mean([r['generation_latency'] for r in protocol_data])

        table1_data.append({
            'model': MODEL_NAMES.get(model, model),
            'with_protocol': 'Yes' if with_protocol else 'No',
            'accuracy': f"{accuracy:.2%}",
            'accuracy_raw': accuracy,
            'latency_seconds': f"{avg_latency:.1f}",
            'latency_raw': avg_latency,
            'total_runs': total,
            'passed': passed,
        })

import csv as csv_module
table1_path = csv_dir / "table1_accuracy_latency.csv"
with open(table1_path, 'w', newline='') as f:
    writer = csv_module.DictWriter(f, fieldnames=['model', 'with_protocol', 'accuracy', 'latency_seconds', 'total_runs', 'passed'])
    writer.writeheader()
    for row in table1_data:
        writer.writerow({k: v for k, v in row.items() if k in ['model', 'with_protocol', 'accuracy', 'latency_seconds', 'total_runs', 'passed']})

print(f"  Created: {table1_path}")

# TABLE 2: Failure Breakdown by Type and Protocol (Figure 2 data)
print("\nGenerating Table 2: Failure Breakdown by Type")
table2_data = []
for with_protocol in [True, False]:
    total_failure_types = sum(failure_data_agg[with_protocol].values())
    total_runs = total_runs_agg[with_protocol]

    row = {
        'protocol': 'Yes' if with_protocol else 'No',
        'total_runs': total_runs,
        'total_failures': total_failures_agg[with_protocol],
    }

    for failure_type in failure_types_ordered:
        count = failure_data_agg[with_protocol].get(failure_type, 0)
        proportion = count / total_failure_types if total_failure_types > 0 else 0
        row[f'{failure_type}_count'] = count
        row[f'{failure_type}_rate'] = f"{proportion:.2%}"

    table2_data.append(row)

table2_path = csv_dir / "table2_failure_breakdown.csv"
with open(table2_path, 'w', newline='') as f:
    fieldnames = ['protocol', 'total_runs', 'total_failures'] + [f'{ft}_count' for ft in failure_types_ordered] + [f'{ft}_rate' for ft in failure_types_ordered]
    writer = csv_module.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(table2_data)

print(f"  Created: {table2_path}")

# TABLE 3: Pass@k and Pass@t for selected k and t values (Figure 3 data)
print("\nGenerating Table 3: Pass@k and Pass@t")
table3_data = []
k_values_selected = [1, 2, 3, 5, 8, 16, 32]
t_values_selected = [5, 10, 15, 20, 30]

for model in MODELS:
    model_data = [r for r in planner_data if r['config']['model'] == model]

    for with_protocol in [True, False]:
        protocol_data = [r for r in model_data if r['config']['with_protocol'] == with_protocol]

        if not protocol_data:
            continue

        n = len(protocol_data)
        c = sum(1 for r in protocol_data if r['codecheck']['overall_pass'])
        p = c / n if n > 0 else 0

        row = {
            'model': MODEL_NAMES.get(model, model),
            'with_protocol': 'Yes' if with_protocol else 'No',
            'success_rate': f"{p:.2%}",
        }

        # Pass@k
        for k in k_values_selected:
            if c == 0:
                pass_at_k = 0.0
            elif n - c < k:
                pass_at_k = 1.0
            else:
                pass_at_k = 1 - comb(n - c, k) / comb(n, k)
            row[f'pass@{k}'] = f"{pass_at_k:.2%}"

        # Pass@t
        latencies = [r['generation_latency'] for r in protocol_data if r['generation_latency'] > 0]
        if latencies:
            for t in t_values_selected:
                F_t = sum(1 for lat in latencies if lat <= t) / len(latencies)
                n_parallel = 64
                pass_at_t = 1 - (1 - F_t * p) ** n_parallel
                row[f'pass@{t}s'] = f"{pass_at_t:.2%}"

        table3_data.append(row)

table3_path = csv_dir / "table3_pass_at_k_t.csv"
with open(table3_path, 'w', newline='') as f:
    fieldnames = ['model', 'with_protocol', 'success_rate'] + [f'pass@{k}' for k in k_values_selected] + [f'pass@{t}s' for t in t_values_selected]
    writer = csv_module.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(table3_data)

print(f"  Created: {table3_path}")

# TABLE 4: E2E Latency Breakdown (Figure 5 data)
if e2e_detailed_path.exists():
    print("\nGenerating Table 4: E2E Latency Breakdown")
    table4_data = []

    for result in selected_results:
        avg_timing = result.get('avg_timing', {})

        planning = avg_timing.get('planning_seconds', 0)
        llm_total = avg_timing.get('llm_total_seconds', 0)
        execution = avg_timing.get('execution_seconds', 0)
        action = max(0, execution - llm_total)
        total = planning + llm_total + action

        table4_data.append({
            'method': result.get('name', ''),
            'planning_seconds': f"{planning:.1f}",
            'inference_seconds': f"{llm_total:.1f}",
            'action_seconds': f"{action:.1f}",
            'total_seconds': f"{total:.1f}",
            'planning_pct': f"{(planning/total)*100:.1f}%",
            'inference_pct': f"{(llm_total/total)*100:.1f}%",
            'action_pct': f"{(action/total)*100:.1f}%",
            'correctness': f"{result.get('avg_correctness_pct', 0)*100:.0f}%",
        })

    table4_path = csv_dir / "table4_e2e_latency.csv"
    with open(table4_path, 'w', newline='') as f:
        fieldnames = ['method', 'planning_seconds', 'inference_seconds', 'action_seconds', 'total_seconds',
                     'planning_pct', 'inference_pct', 'action_pct', 'correctness']
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table4_data)

    print(f"  Created: {table4_path}")

# TABLE 5: Scheduler Performance (Figure 6 data - from all_results.csv)
if all_results_path.exists():
    print("\nGenerating Table 5: Scheduler Strategy Comparison")
    table5_data = []

    # Add oracle, cost-optimal, and individual strategies
    for model_source, (model_name, color) in model_info.items():
        row_oracle = None
        row_cost = None

        if model_source not in cua_models and model_name in oracle_optimal:
            row_oracle = {
                'model': model_name,
                'strategy': 'Oracle-Optimal',
                'latency_seconds': f"{oracle_optimal[model_name]['latency']:.1f}",
                'accuracy': f"{oracle_optimal[model_name]['accuracy']:.2%}",
                'latency_raw': oracle_optimal[model_name]['latency'],
                'accuracy_raw': oracle_optimal[model_name]['accuracy'],
            }

        if model_source not in cua_models and model_name in cost_optimal:
            row_cost = {
                'model': model_name,
                'strategy': 'Cost-Optimal',
                'latency_seconds': f"{cost_optimal[model_name]['latency']:.1f}",
                'accuracy': f"{cost_optimal[model_name]['accuracy']:.2%}",
                'latency_raw': cost_optimal[model_name]['latency'],
                'accuracy_raw': cost_optimal[model_name]['accuracy'],
            }

        # Add individual strategies
        for point in plot_data:
            if point['model'] == model_name:
                table5_data.append({
                    'model': model_name,
                    'strategy': point['strategy'],
                    'latency_seconds': f"{point['latency']:.1f}",
                    'accuracy': f"{point['accuracy']:.2%}",
                    'latency_raw': point['latency'],
                    'accuracy_raw': point['accuracy'],
                })

        if row_oracle:
            table5_data.append(row_oracle)
        if row_cost:
            table5_data.append(row_cost)

    table5_path = csv_dir / "table5_scheduler_strategies.csv"
    with open(table5_path, 'w', newline='') as f:
        fieldnames = ['model', 'strategy', 'latency_seconds', 'accuracy']
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table5_data:
            writer.writerow({k: v for k, v in row.items() if k in fieldnames})

    print(f"  Created: {table5_path}")

    # TABLE 6: Latency Variance Analysis (Cost-Worst vs Cost-Optimal)
    print("\nGenerating Table 6: Latency Variance Analysis")
    table6_data = []

    # Find cost-optimal and cost-worst from e2e data
    if e2e_detailed_path.exists():
        best_result = None
        worst_result = None

        for r in e2e_results:
            if r.get('name') == 'gemini-2.5-flash-best':
                best_result = r
            elif r.get('name') == 'gemini-2.5-flash-worst':
                worst_result = r

        if best_result and worst_result:
            best_total = sum([
                best_result['avg_timing'].get('planning_seconds', 0),
                best_result['avg_timing'].get('llm_total_seconds', 0),
                best_result['avg_timing'].get('execution_seconds', 0) - best_result['avg_timing'].get('llm_total_seconds', 0)
            ])

            worst_total = sum([
                worst_result['avg_timing'].get('planning_seconds', 0),
                worst_result['avg_timing'].get('llm_total_seconds', 0),
                worst_result['avg_timing'].get('execution_seconds', 0) - worst_result['avg_timing'].get('llm_total_seconds', 0)
            ])

            latency_ratio = worst_total / best_total if best_total > 0 else 0

            table6_data.append({
                'comparison': 'Cost-Optimal vs Cost-Worst',
                'cost_optimal_latency': f"{best_total:.1f}s",
                'cost_worst_latency': f"{worst_total:.1f}s",
                'latency_ratio': f"{latency_ratio:.2f}×",
                'cost_optimal_accuracy': f"{best_result.get('avg_correctness_pct', 0)*100:.0f}%",
                'cost_worst_accuracy': f"{worst_result.get('avg_correctness_pct', 0)*100:.0f}%",
            })

    table6_path = csv_dir / "table6_latency_variance.csv"
    with open(table6_path, 'w', newline='') as f:
        fieldnames = ['comparison', 'cost_optimal_latency', 'cost_worst_latency', 'latency_ratio',
                     'cost_optimal_accuracy', 'cost_worst_accuracy']
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table6_data)

    print(f"  Created: {table6_path}")

print("\n=== All CSV Tables Generated ===")
print(f"Tables saved to: {csv_dir}/")

print("\n=== All Figures Generated Successfully ===")

print("Files created/updated:")
print(f"  {fig1_png}")
print(f"  {fig1_pdf}")
print(f"  {fig2_png}")
print(f"  {fig2_pdf}")
print(f"  {fig3_png}")
print(f"  {fig3_pdf}")
print(f"  {fig4_png}")
print(f"  {fig4_pdf}")
if e2e_detailed_path.exists():
    print(f"  {fig5_png}")
    print(f"  {fig5_pdf}")
    print(f"  {table_path}")
