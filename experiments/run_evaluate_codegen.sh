#!/bin/bash
# Example script to run code generation evaluation

# Quick test with 1 task, 4 runs
echo "Running quick test..."
python experiments/evaluate_codegen.py \
    --tasks experiments/tasks/agisdk/agisdk.yaml \
    --ids "dashdish-1" \
    --runs 4 \
    --results experiments/results/test_results.json

# Full evaluation with multiple tasks, 32 runs per config
echo "Running full evaluation (this will take a while)..."
python experiments/evaluate_codegen.py \
    --tasks experiments/tasks/agisdk/agisdk.yaml \
    --ids "dashdish-1 dashdish-2 gomail-3 gocalendar-11" \
    --runs 32 \
    --results experiments/results/rq_results.json

# With actual latency measurement (slower but more complete)
echo "Running evaluation with actual latency..."
python experiments/evaluate_codegen.py \
    --tasks experiments/tasks/agisdk/agisdk.yaml \
    --ids "dashdish-1 gomail-3" \
    --runs 32 \
    --actual-latency \
    --results experiments/results/rq_results_with_latency.json
