#!/bin/bash
# Quick evaluation with 8 runs, 8 parallel, and code printing enabled

cd /home/calebwin/blast

echo "=== Running evaluation with 8 runs, 8 parallel ==="
blast-venv/bin/python experiments/evaluate_codegen.py \
  --tasks experiments/tasks/agisdk/agisdk.yaml \
  --ids dashdish-deepresearch1 \
  --runs 8 \
  --parallel 8 \
  --print-code

echo ""
echo "=== Results saved to experiments/results/ ==="
echo ""
echo "Summary:"
cat experiments/results/rq_results.json | python3 -m json.tool
