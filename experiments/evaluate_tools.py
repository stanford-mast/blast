"""
Evaluate task performance with and without generated SMCP tools.

Usage:
    python experiments/evaluate_tools.py <task_id> [--tools <tools_path>] [--runs <num_runs>]

Example:
    python experiments/evaluate_tools.py dashdish-1 --tools experiments/tools/dashdish-1.json --runs 3
"""

import asyncio
import argparse
import yaml
import json
import time
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

# ⚠️ IMPORTANT: Enable standalone mode FIRST, before importing blastai
# This prevents blastai from capturing/filtering logging output
# Pass 'DEBUG' to see all browser-use logs (or set BLASTAI_LOG_LEVEL=DEBUG)
from blastai.logging_setup import enable_standalone_mode
enable_standalone_mode(browser_use_log_level="INFO")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from blastai.agents import Agent, AgentExecutor


@dataclass
class EvaluationResult:
    """Result of a single evaluation run."""
    task_id: str
    with_tools: bool
    success: bool
    latency_seconds: float
    num_actions: int
    error: str = ""
    result: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Summary of evaluation across multiple runs."""
    task_id: str
    task_goal: str
    
    # Without tools
    baseline_runs: List[EvaluationResult]
    baseline_avg_latency: float
    baseline_success_rate: float
    baseline_avg_actions: float
    
    # With tools (loop mode)
    loop_tools_runs: List[EvaluationResult]
    loop_tools_avg_latency: float
    loop_tools_success_rate: float
    loop_tools_avg_actions: float
    
    # With tools (code mode) - optional
    code_tools_runs: List[EvaluationResult]
    code_tools_avg_latency: float
    code_tools_success_rate: float
    code_tools_avg_actions: float
    
    # Comparisons
    loop_latency_improvement: float  # Percentage
    loop_success_improvement: float  # Percentage
    loop_actions_reduction: float  # Percentage
    
    code_latency_improvement: float  # Percentage
    code_success_improvement: float  # Percentage
    code_actions_reduction: float  # Percentage
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['baseline_runs'] = [r.to_dict() for r in self.baseline_runs]
        result['loop_tools_runs'] = [r.to_dict() for r in self.loop_tools_runs]
        result['code_tools_runs'] = [r.to_dict() for r in self.code_tools_runs]
        return result
    
    def print_summary(self):
        """Print human-readable summary."""
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY: {self.task_id}")
        print(f"{'='*80}")
        print(f"Goal: {self.task_goal}")
        print()
        
        print(f"BASELINE (no tools):")
        print(f"  Success Rate: {self.baseline_success_rate:.1%}")
        print(f"  Avg Latency: {self.baseline_avg_latency:.2f}s")
        print(f"  Avg Actions: {self.baseline_avg_actions:.1f}")
        print()
        
        print(f"LOOP MODE WITH TOOLS:")
        print(f"  Success Rate: {self.loop_tools_success_rate:.1%}")
        print(f"  Avg Latency: {self.loop_tools_avg_latency:.2f}s")
        print(f"  Avg Actions: {self.loop_tools_avg_actions:.1f}")
        print(f"  Latency Improvement: {self.loop_latency_improvement:+.1%}")
        print(f"  Success Improvement: {self.loop_success_improvement:+.1%}")
        print(f"  Actions Reduction: {self.loop_actions_reduction:+.1%}")
        print()
        
        if self.code_tools_runs:
            print(f"CODE MODE WITH TOOLS:")
            print(f"  Success Rate: {self.code_tools_success_rate:.1%}")
            print(f"  Avg Latency: {self.code_tools_avg_latency:.2f}s")
            print(f"  Avg Actions: {self.code_tools_avg_actions:.1f}")
            print(f"  Latency Improvement: {self.code_latency_improvement:+.1%}")
            print(f"  Success Improvement: {self.code_success_improvement:+.1%}")
            print(f"  Actions Reduction: {self.code_actions_reduction:+.1%}")
            print()
        
        print(f"{'='*80}\n")


async def run_single_evaluation(
    task_id: str,
    task_data: Dict[str, Any],
    agent: Agent,
    run_number: int,
    with_tools: bool,
    mode: str = "loop"
) -> EvaluationResult:
    """
    Run a single evaluation.
    
    Args:
        task_id: Task identifier
        task_data: Task data with initial_url and goal
        agent: Agent to use
        run_number: Run number for logging
        with_tools: Whether using generated tools
        mode: "loop" or "code" execution mode
        
    Returns:
        EvaluationResult with metrics
    """
    mode_label = "with tools" if with_tools else "baseline"
    print(f"\n[Run {run_number}] Evaluating {task_id} ({mode_label}, {mode} mode)...")
    
    # Show agent info BEFORE creating executor
    smcp_count = sum(1 for t in agent.tools if hasattr(t, 'tool_executor_type') and t.tool_executor_type.value == 'smcp')
    print(f"  Agent has {len(agent.tools)} tools ({smcp_count} SMCP)")
    
    # Debug: Print each tool's type
    for tool in agent.tools:
        if hasattr(tool, 'tool_executor_type'):
            print(f"    - {tool.name}: {tool.tool_executor_type} (type attr: {type(tool.tool_executor_type)})")
        else:
            print(f"    - {tool.name}: NO tool_executor_type attribute!")
    
    executor = AgentExecutor(agent)
    
    # Task is just the goal - executor handles initial_url navigation separately
    task = task_data.get('goal')
    initial_url = task_data.get('initial_url')
    
    start_time = time.time()
    num_actions = 0
    success = False
    error = ""
    result = ""
    
    try:
        # Run agent
        result = await executor.run(
            task, 
            mode=mode,
            initial_url=initial_url
        )
        
        # TODO: Extract num_actions from browser_use agent history
        # For now, estimate based on result
        num_actions = len(str(result).split('\n'))
        
        success = True
        print(f"  ✓ Success in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        error = str(e)
        print(f"  ✗ Failed: {error}")
        
    finally:
        await executor.cleanup()
    
    latency = time.time() - start_time
    
    return EvaluationResult(
        task_id=task_id,
        with_tools=with_tools,
        success=success,
        latency_seconds=latency,
        num_actions=num_actions,
        error=error,
        result=str(result)[:500]  # Truncate
    )


async def evaluate_task(
    task_id: str,
    task_data: Dict[str, Any],
    tools_path: str,
    num_runs: int,
    test_code_mode: bool = False,
    skip_baseline: bool = False
) -> EvaluationSummary:
    """
    Evaluate a task with and without tools, in loop mode (and optionally code mode).
    
    Args:
        task_id: Task identifier
        task_data: Task data with initial_url and goal
        tools_path: Path to generated tools JSON
        num_runs: Number of runs for each configuration
        test_code_mode: Whether to also test code mode (default False)
        
    Returns:
        EvaluationSummary with comparison
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {task_id}")
    print(f"Goal: {task_data.get('goal')}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Test code mode: {test_code_mode}")
    print(f"{'='*80}")
    
    # Load tools if provided
    agent_with_tools = None
    if tools_path and Path(tools_path).exists():
        print(f"\nLoading tools from: {tools_path}")
        agent_with_tools = Agent.from_json(tools_path)
        print(f"Loaded {len(agent_with_tools.tools)} tools")
        
        # Print tool details
        for tool in agent_with_tools.tools:
            tool_type = tool.tool_executor_type.value if hasattr(tool, 'tool_executor_type') else 'unknown'
            print(f"  - {tool.name} ({tool_type}): {tool.description[:80]}...")
    else:
        print(f"\nWarning: Tools file not found: {tools_path}")
        print("Will only run baseline evaluation")
    
    # Create baseline agent
    baseline_agent = Agent(description="", tools=[])
    
    # Run baseline evaluations
    baseline_runs = []
    if not skip_baseline:
        print(f"\n{'='*60}")
        print("BASELINE EVALUATION (no tools)")
        print(f"{'='*60}")
        for i in range(num_runs):
            result = await run_single_evaluation(
                task_id, task_data, baseline_agent, i + 1, with_tools=False, mode="loop"
            )
            baseline_runs.append(result)
    else:
        print(f"\n{'='*60}")
        print("SKIPPING BASELINE EVALUATION (--skip-baseline)")
        print(f"{'='*60}")
    
    # Run loop mode with-tools evaluations
    loop_tools_runs = []
    if agent_with_tools:
        print(f"\n{'='*60}")
        print("LOOP MODE WITH TOOLS EVALUATION")
        print(f"{'='*60}")
        for i in range(num_runs):
            result = await run_single_evaluation(
                task_id, task_data, agent_with_tools, i + 1, with_tools=True, mode="loop"
            )
            loop_tools_runs.append(result)
    
    # Run code mode with-tools evaluations (optional)
    code_tools_runs = []
    if agent_with_tools and test_code_mode:
        print(f"\n{'='*60}")
        print("CODE MODE WITH TOOLS EVALUATION")
        print(f"{'='*60}")
        for i in range(num_runs):
            result = await run_single_evaluation(
                task_id, task_data, agent_with_tools, i + 1, with_tools=True, mode="code"
            )
            code_tools_runs.append(result)
    
    # Calculate statistics
    def calc_stats(runs: List[EvaluationResult]):
        if not runs:
            return 0.0, 0.0, 0.0
        
        success_rate = sum(1 for r in runs if r.success) / len(runs)
        avg_latency = sum(r.latency_seconds for r in runs) / len(runs)
        avg_actions = sum(r.num_actions for r in runs) / len(runs)
        return success_rate, avg_latency, avg_actions
    
    baseline_success, baseline_latency, baseline_actions = calc_stats(baseline_runs)
    loop_success, loop_latency, loop_actions = calc_stats(loop_tools_runs)
    code_success, code_latency, code_actions = calc_stats(code_tools_runs)
    
    # Calculate improvements for loop mode
    loop_latency_improvement = ((baseline_latency - loop_latency) / baseline_latency * 100) if baseline_latency > 0 else 0
    loop_success_improvement = ((loop_success - baseline_success) / baseline_success * 100) if baseline_success > 0 else 0
    loop_actions_reduction = ((baseline_actions - loop_actions) / baseline_actions * 100) if baseline_actions > 0 else 0
    
    # Calculate improvements for code mode
    code_latency_improvement = ((baseline_latency - code_latency) / baseline_latency * 100) if baseline_latency > 0 and code_latency > 0 else 0
    code_success_improvement = ((code_success - baseline_success) / baseline_success * 100) if baseline_success > 0 and code_success > 0 else 0
    code_actions_reduction = ((baseline_actions - code_actions) / baseline_actions * 100) if baseline_actions > 0 and code_actions > 0 else 0
    
    summary = EvaluationSummary(
        task_id=task_id,
        task_goal=task_data.get('goal', ''),
        baseline_runs=baseline_runs,
        baseline_avg_latency=baseline_latency,
        baseline_success_rate=baseline_success,
        baseline_avg_actions=baseline_actions,
        loop_tools_runs=loop_tools_runs,
        loop_tools_avg_latency=loop_latency,
        loop_tools_success_rate=loop_success,
        loop_tools_avg_actions=loop_actions,
        code_tools_runs=code_tools_runs,
        code_tools_avg_latency=code_latency,
        code_tools_success_rate=code_success,
        code_tools_avg_actions=code_actions,
        loop_latency_improvement=loop_latency_improvement,
        loop_success_improvement=loop_success_improvement,
        loop_actions_reduction=loop_actions_reduction,
        code_latency_improvement=code_latency_improvement,
        code_success_improvement=code_success_improvement,
        code_actions_reduction=code_actions_reduction
    )
    
    return summary


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate task performance with and without generated tools"
    )
    parser.add_argument("task_id", help="Task ID to evaluate (e.g., dashdish-1)")
    parser.add_argument(
        "--tools",
        help="Path to generated tools JSON",
        default=None
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per configuration (default: 3)"
    )
    parser.add_argument(
        "--code-mode",
        action="store_true",
        help="Also test code mode in addition to loop mode"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation (only run with tools)"
    )
    parser.add_argument(
        "--tasks-file",
        help="Path to tasks YAML file",
        default="experiments/tasks/agisdk/agisdk.yaml"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Accumulate results in existing output file instead of overwriting"
    )
    parser.add_argument(
        "--output",
        help="Output path for evaluation results JSON",
        default=None
    )
    
    args = parser.parse_args()
    
    # Load tasks
    tasks_file = Path(args.tasks_file)
    if not tasks_file.exists():
        print(f"Error: Tasks file not found: {tasks_file}")
        sys.exit(1)
    
    with open(tasks_file, 'r') as f:
        tasks = yaml.safe_load(f)
    
    # Find task
    task_data = None
    for task in tasks:
        if task.get('id') == args.task_id:
            task_data = task
            break
    
    if task_data is None:
        print(f"Error: Task ID '{args.task_id}' not found in {tasks_file}")
        sys.exit(1)
    
    # Determine tools path
    tools_path = args.tools
    if not tools_path:
        # Default to experiments/tools/<task_id>.json
        tools_path = f"experiments/tools/{args.task_id}.json"
    
    # Run evaluation
    summary = await evaluate_task(
        args.task_id,
        task_data,
        tools_path,
        args.runs,
        test_code_mode=args.code_mode,
        skip_baseline=args.skip_baseline
    )
    
    # Print summary
    summary.print_summary()
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_dir = Path("experiments/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{args.task_id}_evaluation.json")
    
    if args.incremental and Path(output_path).exists():
        # Load existing results and merge
        with open(output_path, 'r') as f:
            existing = json.load(f)
        
        # Append new runs
        existing['baseline_runs'].extend([r.to_dict() for r in summary.baseline_runs])
        existing['loop_tools_runs'].extend([r.to_dict() for r in summary.loop_tools_runs])
        existing['code_tools_runs'].extend([r.to_dict() for r in summary.code_tools_runs])
        
        # Recalculate stats
        def recalc_stats(runs_key: str):
            runs = [EvaluationResult(**r) for r in existing[runs_key]]
            if not runs:
                prefix = runs_key.split('_')[0]
                existing[f'{prefix}_avg_latency'] = 0.0
                existing[f'{prefix}_success_rate'] = 0.0
                existing[f'{prefix}_avg_actions'] = 0.0
                return
            success_rate = sum(1 for r in runs if r.success) / len(runs)
            avg_latency = sum(r.latency_seconds for r in runs) / len(runs)
            avg_actions = sum(r.num_actions for r in runs) / len(runs)
            prefix = runs_key.split('_')[0]
            existing[f'{prefix}_avg_latency'] = avg_latency
            existing[f'{prefix}_success_rate'] = success_rate
            existing[f'{prefix}_avg_actions'] = avg_actions
        
        recalc_stats('baseline_runs')
        recalc_stats('loop_tools_runs')
        recalc_stats('code_tools_runs')
        
        # Recalculate improvements
        baseline_latency = existing['baseline_avg_latency']
        baseline_success = existing['baseline_success_rate']
        baseline_actions = existing['baseline_avg_actions']
        loop_latency = existing['loop_tools_avg_latency']
        loop_success = existing['loop_tools_success_rate']
        loop_actions = existing['loop_tools_avg_actions']
        code_latency = existing['code_tools_avg_latency']
        code_success = existing['code_tools_success_rate']
        code_actions = existing['code_tools_avg_actions']
        
        existing['loop_latency_improvement'] = ((baseline_latency - loop_latency) / baseline_latency * 100) if baseline_latency > 0 else 0
        existing['loop_success_improvement'] = ((loop_success - baseline_success) / baseline_success * 100) if baseline_success > 0 else 0
        existing['loop_actions_reduction'] = ((baseline_actions - loop_actions) / baseline_actions * 100) if baseline_actions > 0 else 0
        existing['code_latency_improvement'] = ((baseline_latency - code_latency) / baseline_latency * 100) if baseline_latency > 0 and code_latency > 0 else 0
        existing['code_success_improvement'] = ((code_success - baseline_success) / baseline_success * 100) if baseline_success > 0 and code_success > 0 else 0
        existing['code_actions_reduction'] = ((baseline_actions - code_actions) / baseline_actions * 100) if baseline_actions > 0 and code_actions > 0 else 0
        
        data = existing
    else:
        data = summary.to_dict()
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved detailed results to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
