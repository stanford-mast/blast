# BLAST Experiments Framework

This directory contains the evaluation framework for BLAST, including tools for:
- Tool synthesis via web crawling
- Code generation evaluation
- End-to-end latency measurement
- Result visualization and analysis

## Table of Contents

1. [Overview](#overview)
2. [Task Definition](#task-definition)
3. [Task Validators](#task-validators)
4. [Tool Synthesis](#tool-synthesis)
5. [Code Generation Evaluation](#code-generation-evaluation)
6. [End-to-End Latency Measurement](#end-to-end-latency-measurement)
7. [Plotting Results](#plotting-results)
8. [Complete Workflow Example](#complete-workflow-example)

---

## Overview

The BLAST evaluation pipeline consists of four main stages:

```
1. Tool Synthesis → 2. Code Generation → 3. E2E Measurement → 4. Visualization
   (blastai crawl)    (evaluate_codegen)   (measure_e2e)        (plot_all_results)
```

Each stage produces outputs used by subsequent stages, enabling comprehensive evaluation of BLAST's planning and execution capabilities.

---

## Task Definition

### Task YAML Files

Tasks are defined in YAML files (e.g., `tasks/agisdk/agisdk.yaml`). Each task specifies:

```yaml
- id: dashdish-deepresearch1              # Unique identifier
  initial_url: https://example.com        # Starting URL
  goal: "Task description..."             # Natural language goal
  smcp_registry: smcp-registries/...json  # (Optional) Path to SMCP tools
  user_id: user123                        # (Optional) Browser profile ID
```

**Field Descriptions:**

- **id**: Unique task identifier (lowercase, hyphen-separated)
- **initial_url**: URL where the browser starts
- **goal**: Natural language description of what to accomplish
- **smcp_registry**: Path to SMCP tool registry (generated via `blastai crawl`)
- **user_id**: Browser profile for persistent cookies/sessions

### Imported Tasks

Pre-imported tasks from [agisdk-REAL](https://github.com/agi-inc/agisdk/tree/main/src/agisdk/REAL/browsergym/webclones/tasks) are available in `tasks/agisdk/agisdk.yaml`.

To re-import:
```bash
# Clone agisdk repo first
python tasks/agisdk/import_real.py
```

---

## Task Validators

Each task must have a corresponding validator implementation to measure correctness.

### Creating a Task Validator

1. **Create a Python file** in `experiments/tasks/` named `{task_id}.py` (replace hyphens with underscores)
   - Example: `dashdish-deepresearch1` → `dashdish_deepresearch1.py`

2. **Define the output schema** using Pydantic:
   ```python
   from pydantic import BaseModel, Field

   class MyTaskOutput(BaseModel):
       """Expected output structure for my-task"""
       restaurant_name: str
       num_reviews: int
       items: list[str]
   ```

3. **Implement the validator** by subclassing `TaskValidator`:
   ```python
   from experiments.tasks.base import TaskValidator

   class MyTaskValidator(TaskValidator):
       @property
       def output_schema(self) -> type[BaseModel]:
           """Return the Pydantic model for expected output."""
           return MyTaskOutput

       def check_correctness(self, parsed_output: BaseModel) -> bool:
           """Check if output is 100% correct."""
           # Compare parsed_output against ground truth
           return parsed_output.restaurant_name == "Expected Name"

       def check_correctness_pct(self, parsed_output: BaseModel) -> float:
           """(Optional) Return partial correctness (0.0-1.0)."""
           # Count correct fields and return percentage
           correct_fields = 0
           total_fields = 3

           if parsed_output.restaurant_name == "Expected":
               correct_fields += 1
           # ... check other fields ...

           return correct_fields / total_fields
   ```

4. **Export a singleton instance**:
   ```python
   # At the end of the file
   validator = MyTaskValidator()
   ```

### Validator API

The `TaskValidator` base class provides:

- **`output_schema`** (property): Returns Pydantic model defining expected output structure
- **`check_correctness(parsed_output)`**: Returns `True` if output is 100% correct
- **`check_correctness_pct(parsed_output)`**: Returns percentage correct (0.0-1.0)
- **`validate(output, return_pct=False)`**: Parses text output using LLM and checks correctness

### Automatic Registration

Task validators are automatically discovered and registered by filename:
- `dashdish_deepresearch1.py` → task ID `dashdish-deepresearch1`
- Must export a `validator` instance

---

## Tool Synthesis

Before evaluating code generation, you must synthesize SMCP tools for tasks that require them.

### Command: `blastai crawl`

```bash
blastai crawl \
  --url <starting-url> \
  --smcp-registry <output-path.json> \
  --prompt "<tool generation instructions>" \
  --user-id <optional-browser-profile>
```

**Example:**
```bash
blastai crawl \
  --url https://dashdish.example.com \
  --smcp-registry smcp-registries/dashdish.json \
  --prompt "Create tools to browse restaurants, view menus, and extract pricing" \
  --user-id dashdish-user
```

### What It Does

1. Launches browser at `--url`
2. Uses synthesis agent (with `update_smcp_tool`, `remove_smcp_tool`, `ask_html`) to:
   - Explore the website structure
   - Generate SMCP tools for common operations
   - Validate generated tools
3. Saves SMCP tools to `--smcp-registry`

### Iterative Refinement

Tool synthesis may require multiple runs to generate valid tools:

```bash
# First attempt - generates initial tools
blastai crawl --url ... --smcp-registry tools.json --prompt "..."

# Check tools.json - may have syntax/validation errors

# Second attempt - refines existing tools
blastai crawl --url ... --smcp-registry tools.json --prompt "Fix validation errors in existing tools"

# Continue until tools pass validation
```

**Tips:**
- Start with simple operations (navigation, observation)
- Add complex operations (form filling, filtering) in later iterations
- Use `--user-id` to maintain login sessions across runs

---

## Code Generation Evaluation

Evaluates code generation across different models and configurations.

### Command: `evaluate_codegen.py`

```bash
python experiments/evaluate_codegen.py \
  --tasks <tasks.yaml> \
  --ids "<space-separated task IDs>" \
  --results-dir <output-directory> \
  [options]
```

**Example:**
```bash
python experiments/evaluate_codegen.py \
  --tasks experiments/tasks/agisdk/agisdk.yaml \
  --ids "dashdish-deepresearch1 gomail-10" \
  --results-dir experiments/results \
  --models "gpt-5.1 gemini-2.5-flash" \
  --runs 64 \
  --parallel 4
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tasks` | (required) | Path to tasks YAML file |
| `--ids` | (required) | Space-separated task IDs to evaluate |
| `--results-dir` | `experiments/results` | Output directory |
| `--models` | (all defaults) | Models to test (space-separated) |
| `--runs` | 32 | Number of runs per (model, protocol) config |
| `--parallel` | 1 | Number of parallel evaluations (max 8) |
| `--max-iterations` | 1 | Retry iterations for code generation |
| `--actual-latency` | False | Measure actual execution time |
| `--page-load` | False | Measure page load latency |
| `--no-overwrite` | False | Add timestamp to filenames |

### Output Files

Per task:
- `{task_id}.json`: Raw evaluation results (all runs)
- `{task_id}.md`: Generated code with metrics
- `{task_id}_{timestamp}.json/md`: Timestamped versions (if `--no-overwrite`)

Summary:
- `summary.json`: Aggregated metrics across all tasks/configs

### Metrics Collected

For each run:
- **Generation latency**: Time to generate code (seconds)
- **Estimated cost**: Estimated execution time from static analysis
- **Validation results**: Syntax, type, ordering, state-access checks
- **Generated code**: Full Python code
- **Actual latency** (optional): Real execution time for best/worst cost candidates

Summary statistics per (model, protocol) configuration:
- Average generation latency (mean, variance)
- Average estimated cost (mean, variance)
- Pass rate (percentage of runs passing all checks)
- Failure breakdown by type (syntax, types, ordering, etc.)

---

## End-to-End Latency Measurement

Measures actual execution latency with detailed timing breakdowns.

### Command: `measure_e2e_detailed.py`

```bash
python experiments/measure_e2e_detailed.py \
  --tasks <tasks.yaml> \
  --ids "<space-separated task IDs>" \
  --results-dir <results-directory> \
  --models <models> \
  --num-trials <trials> \
  [options]
```

**Example:**
```bash
python experiments/measure_e2e_detailed.py \
  --tasks experiments/tasks/agisdk/agisdk.yaml \
  --ids "dashdish-deepresearch1 gomail-10" \
  --results-dir experiments/results \
  --models "gemini-2.5-flash,gemini-2.5-pro" \
  --num-trials 3
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tasks` | (required) | Path to tasks YAML file |
| `--ids` | (required) | Space-separated task IDs to measure |
| `--results-dir` | `experiments/results` | Directory containing evaluation results |
| `--models` | `gemini-2.5-flash,gemini-2.5-pro` | Comma-separated models to test |
| `--num-trials` | 1 | Number of trials per configuration |
| `--test-best/--no-test-best` | True | Test best-cost candidate |
| `--test-worst/--no-test-worst` | True | Test worst-cost candidate |
| `--test-loop/--no-test-loop` | True | Test baseline (no tools) |
| `--test-loop-tools` | True | Test loop mode with tools |
| `--test-retry` | False | Test serial retry mode |

### Output Files

- `{task_id}_e2e_detailed.json`: Timing breakdowns for each task
- Files are auto-loaded from `--results-dir` based on task IDs

### Timing Breakdown

For each configuration:
- **Planning time**: Code generation latency
- **Execution time**: Total time running code/loop
- **LLM inference breakdown**:
  - Prefill time (prompt encoding)
  - Decode time (token generation)
- **Action time**: Execution minus LLM time (browser actions, page loads)
- **Correctness percentage**: Task-specific validation score

### Configurations Measured

1. **Cost-Optimal**: Pre-generated code with lowest estimated cost (best planning latency tiebreaker)
2. **Cost-Worst**: Pre-generated code with highest estimated cost
3. **Serial Retry** (optional): Code generation with up to 3 retry iterations
4. **Loop Baseline**: Browser-Use loop mode without SMCP tools
5. **Loop + Tools**: Browser-Use loop mode with SMCP tools

---

## Plotting Results

Generates publication-ready figures from evaluation results.

### Command: `plot_all_results.py`

```bash
python experiments/plot_all_results.py
```

**Note**: Modify the script to point to your result files (hardcoded paths at top of file).

### Generated Figures

#### Figure 1: Accuracy vs Latency
- **File**: `fig1_accuracy_vs_latency.png/pdf`
- **X-axis**: Planning latency (seconds)
- **Y-axis**: Accuracy (pass rate)
- **Points**: Each (model, protocol) configuration
- **Aggregation**:
  - Averaged across all runs for each (model, protocol, task) combination
  - Single task only (currently hardcoded to `dashdish-deepresearch1`)
- **Shows**: Trade-off between speed and correctness

#### Figure 2: Failure Breakdown
- **File**: `fig2_failure_breakdown.png/pdf`
- **Bars**: Horizontal stacked bars (with/without protocol)
- **Segments**: Failure types (syntax, types, ordering, timeout, state mutation)
- **Aggregation**:
  - Counts aggregated across all models and runs
  - Proportions show percentage of each failure type among all failures
- **Shows**: Most common failure modes

#### Figure 3: Pass@k and Pass@t (Candidate Efficiency)
- **File**: `fig3_candidate_efficiency.png/pdf`
- **Left subplots**: Pass@k (probability of success with k parallel candidates)
- **Right subplots**: Pass@t (probability of success within time budget t)
- **Top row**: GPT models
- **Bottom row**: Gemini models
- **Aggregation**:
  - Based on per-run pass/fail and latencies
  - Uses combinatorial formula for Pass@k
  - Uses empirical CDF for Pass@t
- **Shows**: Benefit of parallel hedging strategies

#### Figure 4: Serial Retry vs Parallel Hedging
- **File**: `fig4_retry_comparison.png/pdf`
- **X-axis**: Latency budget (seconds)
- **Y-axis**: Success rate
- **Curves**: Serial retry (max 3 iterations) vs Parallel hedging (k=8)
- **Aggregation**:
  - Serial: Empirical CDF from retry evaluation runs
  - Parallel: Analytical from single-attempt distributions
- **Shows**: Latency-accuracy trade-offs for different retry strategies

#### Figure 5: Latency Breakdown
- **File**: `fig5_latency_breakdown.png/pdf`
- **Bars**: Horizontal stacked bars
- **Segments**: Planning (red), LLM Inference (teal), Action (light teal)
- **Left subplot**: Absolute latency (seconds)
- **Right subplot**: Percentage of total execution time
- **Aggregation**:
  - Loads all `*_e2e_detailed.json` files from results directory
  - Averages across all tasks and trials for each configuration
  - Includes correctness percentage (in separate LaTeX table)
- **Shows**: Where time is spent in different execution strategies
- **Requires**: Output from `measure_e2e_detailed.py` for one or more tasks

#### Figure 6: Accuracy vs Latency (All Experiments)
- **File**: `fig6_accuracy_vs_latency_all.png/pdf`
- **Points**: Each (model, strategy, task) combination
- **Markers**: Serial (triangle), Parallel (square), Hedge (circle)
- **Special points**:
  - **Diamond**: Cost-Optimal (scheduler's choice from `all_tasks.xlsx`)
  - **Star**: Oracle-Optimal (per-task minimum latency, averaged)
- **Aggregation**:
  - Per-task average across multiple runs
  - Strategies: baseline, task_parallelism_only, first_of_n_only
  - Loaded from `all_results.csv`
- **Shows**: Comparison across execution strategies and scheduler performance
- **Requires**: `all_results.csv` and `all_tasks.xlsx` from experiment runs

### CSV Tables

The plotting script also generates CSV tables in `experiments/results/csv_tables/`:

- `table1_accuracy_latency.csv`: Per-model accuracy and latency
- `table2_failure_breakdown.csv`: Failure counts and rates
- `table3_pass_at_k_t.csv`: Pass@k and Pass@t values
- `table4_e2e_latency.csv`: E2E timing breakdown
- `table5_scheduler_strategies.csv`: Scheduler strategy comparison
- `table6_latency_variance.csv`: Cost-optimal vs cost-worst comparison

---

## Complete Workflow Example

Here's a complete workflow for evaluating a new task:

### 1. Define the Task

Create `tasks/my_experiment/tasks.yaml`:
```yaml
- id: restaurant-search
  initial_url: https://example.com/restaurants
  goal: "Find top 5 restaurants and list their ratings"
  smcp_registry: smcp-registries/restaurant.json
  user_id: restaurant-user
```

### 2. Create Task Validator

Create `experiments/tasks/restaurant_search.py`:
```python
from pydantic import BaseModel
from experiments.tasks.base import TaskValidator

class RestaurantOutput(BaseModel):
    restaurants: list[dict]  # [{"name": str, "rating": float}, ...]

GROUND_TRUTH = RestaurantOutput(
    restaurants=[
        {"name": "Restaurant A", "rating": 4.5},
        # ... 4 more
    ]
)

class RestaurantSearchValidator(TaskValidator):
    @property
    def output_schema(self):
        return RestaurantOutput

    def check_correctness(self, parsed):
        return parsed.restaurants == GROUND_TRUTH.restaurants

validator = RestaurantSearchValidator()
```

### 3. Synthesize Tools

```bash
blastai crawl \
  --url https://example.com/restaurants \
  --smcp-registry smcp-registries/restaurant.json \
  --prompt "Create tools to search restaurants, view details, extract ratings" \
  --user-id restaurant-user
```

Check `smcp-registries/restaurant.json` and iterate if needed.

### 4. Evaluate Code Generation

```bash
python experiments/evaluate_codegen.py \
  --tasks tasks/my_experiment/tasks.yaml \
  --ids "restaurant-search" \
  --results-dir experiments/results/my_experiment \
  --models "gpt-5.1 gemini-2.5-flash gemini-2.5-pro" \
  --runs 64 \
  --parallel 4 \
  --max-iterations 1
```

Results saved to:
- `experiments/results/my_experiment/restaurant-search.json`
- `experiments/results/my_experiment/restaurant-search.md`
- `experiments/results/my_experiment/summary.json`

### 5. Measure E2E Latency

```bash
python experiments/measure_e2e_detailed.py \
  --tasks tasks/my_experiment/tasks.yaml \
  --ids "restaurant-search" \
  --results-dir experiments/results/my_experiment \
  --models "gemini-2.5-flash,gemini-2.5-pro" \
  --num-trials 3
```

Results saved to:
- `experiments/results/my_experiment/restaurant-search_e2e_detailed.json`

### 6. Plot Results

The plotting script automatically discovers result files in the results directory.

For single-task analysis, edit `experiments/plot_all_results.py` to point to your result directory:
```python
# Line ~26: Update results_dir
results_dir = Path("experiments/results/my_experiment")
```

Then run:
```bash
python experiments/plot_all_results.py
```

Figures saved to `experiments/results/my_experiment/`:
- `fig1_accuracy_vs_latency.png/pdf`
- `fig2_failure_breakdown.png/pdf`
- `fig3_candidate_efficiency.png/pdf`
- `fig4_retry_comparison.png/pdf`
- `fig5_latency_breakdown.png/pdf` (if E2E data available)

---

## Multi-Task Evaluation

```bash
# 1. Define tasks (tasks.yaml with 3 tasks)
# 2. Create validators for each task

# 3. Generate code for all tasks
python experiments/evaluate_codegen.py \
  --tasks tasks.yaml \
  --ids "task-1 task-2 task-3" \
  --results-dir results \
  --runs 64

# 4. Measure E2E for all tasks in one command
python experiments/measure_e2e_detailed.py \
  --tasks tasks.yaml \
  --ids "task-1 task-2 task-3" \
  --results-dir results \
  --num-trials 3

# 5. Plot (Figure 5 automatically aggregates across all 3 tasks)
python experiments/plot_all_results.py
```

To evaluate across multiple tasks:

### 1. Define Multiple Tasks

```yaml
- id: task-1
  initial_url: https://example1.com
  goal: "Task 1 description"
  smcp_registry: smcp-registries/task1.json

- id: task-2
  initial_url: https://example2.com
  goal: "Task 2 description"
  smcp_registry: smcp-registries/task2.json
```

### 2. Create Validators for Each Task

- `experiments/tasks/task_1.py` → `validator`
- `experiments/tasks/task_2.py` → `validator`

### 3. Run Code Generation for All Tasks

```bash
python experiments/evaluate_codegen.py \
  --tasks tasks/multi_task.yaml \
  --ids "task-1 task-2" \
  --results-dir experiments/results/multi_task \
  --runs 64
```

This creates per-task results:
- `task-1.json`, `task-1.md`
- `task-2.json`, `task-2.md`
- `summary.json` (aggregated across both tasks)

### 4. Run E2E Measurement for All Tasks

```bash
# Measure E2E for both tasks in one command
python experiments/measure_e2e_detailed.py \
  --tasks tasks/multi_task.yaml \
  --ids "task-1 task-2" \
  --results-dir experiments/results/multi_task \
  --num-trials 3
```

This creates E2E results for each task:
- `task-1_e2e_detailed.json`
- `task-2_e2e_detailed.json`

### 5. Plot Results

The plotting script automatically aggregates across all E2E files found in the results directory.

Edit `experiments/plot_all_results.py`:
```python
# Line ~26: Update results_dir
results_dir = Path("experiments/results/multi_task")
```

Then run:
```bash
python experiments/plot_all_results.py
```

**Multi-Task Aggregation:**
- **Figure 5** (Latency Breakdown): Automatically loads all `*_e2e_detailed.json` files and averages timing across tasks and trials
- Other figures: Currently designed for single-task analysis (load from specific task files)

For full multi-task plots across all figures, you would need to modify the script to load and aggregate data from multiple task JSON/MD files.

---

## How To Extend

### Custom Evaluation Metrics

Implement custom metrics by:
1. Extending `TaskValidator.check_correctness_pct()`
2. Adding fields to output schemas
3. Modifying plotting scripts to visualize new metrics

### Custom Models

Add new models by:
1. Updating `LLMFactory` in `blastai/agents/llm_factory.py`
2. Adding to default configs in `evaluate_codegen.py` (line ~990)
3. Adding color schemes in `plot_all_results.py` (line ~54)

---

## File Structure

```
experiments/
├── README.md                       # This file
├── evaluate_codegen.py             # Code generation evaluation
├── measure_e2e_detailed.py         # E2E latency measurement
├── plot_all_results.py             # Visualization
├── tasks/
│   ├── base.py                     # TaskValidator base class
│   ├── registry.py                 # Validator auto-discovery
│   ├── dashdish_deepresearch1.py   # Example validator
│   └── agisdk/
│       └── agisdk.yaml             # Pre-imported tasks
└── results/                        # Evaluation outputs
    ├── *.json                      # Raw results
    ├── *.md                        # Generated code
    ├── *_e2e_detailed.json         # E2E timing
    ├── fig*.png/pdf                # Figures
    └── csv_tables/                 # Data tables
```
