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
    config: str  # 'loop', 'loop-smcp', 'code-smcp', etc.
    run_number: int
    success: bool
    latency_seconds: float
    num_actions: int
    error: str = ""
    result: str = ""
    # Code mode metrics
    codegen_overhead_seconds: float = 0.0  # Total time spent on LLM code generation
    time_to_first_token_seconds: float = 0.0  # Average TTFT across LLM calls
    num_llm_calls: int = 0  # Number of LLM calls made
    # TODO: Track LLM inference time vs execution time vs browser state retrieval time
    # TODO: Track cost (tokens * rate) and accuracy metrics
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Summary of evaluation across multiple runs and configs."""
    task_ids: List[str]  # All tasks evaluated
    configs: List[str]  # All configs tested
    
    # Raw results
    results: List[EvaluationResult]
    
    # Aggregated stats per config (averaged across all runs and tasks for that config)
    config_stats: Dict[str, Dict[str, float]]  # config -> {success_rate, avg_latency, avg_actions}
    
    # Per-task per-config stats (averaged across runs)
    task_config_stats: Dict[str, Dict[str, Dict[str, float]]]  # task_id -> config -> stats
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_ids': self.task_ids,
            'configs': self.configs,
            'results': [r.to_dict() for r in self.results],
            'config_stats': self.config_stats,
            'task_config_stats': self.task_config_stats
        }
    
    def print_summary(self):
        """Print human-readable summary."""
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Tasks: {', '.join(self.task_ids)}")
        print(f"Configs: {', '.join(self.configs)}")
        print(f"Total runs: {len(self.results)}")
        print()
        
        # Print per-config aggregated stats
        for config in self.configs:
            stats = self.config_stats.get(config, {})
            print(f"{config.upper()}:")
            print(f"  Success Rate: {stats.get('success_rate', 0):.1%}")
            print(f"  Avg Latency: {stats.get('avg_latency', 0):.2f}s")
            print(f"  Avg Actions: {stats.get('avg_actions', 0):.1f}")
            
            # Code mode metrics
            if 'code' in config and stats.get('avg_codegen_overhead', 0) > 0:
                print(f"  Avg Codegen Overhead: {stats.get('avg_codegen_overhead', 0):.2f}s")
                print(f"  Avg TTFT: {stats.get('avg_ttft', 0):.3f}s")
                print(f"  Avg LLM Calls: {stats.get('avg_llm_calls', 0):.1f}")
            print()
        
        # Print per-task breakdown if multiple tasks
        if len(self.task_ids) > 1:
            print(f"{'='*80}")
            print("PER-TASK BREAKDOWN:")
            print(f"{'='*80}")
            for task_id in self.task_ids:
                print(f"\nTask: {task_id}")
                task_stats = self.task_config_stats.get(task_id, {})
                for config in self.configs:
                    stats = task_stats.get(config, {})
                    if stats:
                        print(f"  {config}: Success={stats.get('success_rate', 0):.1%}, "
                              f"Latency={stats.get('avg_latency', 0):.1f}s, "
                              f"Actions={stats.get('avg_actions', 0):.1f}")
        
        print(f"{'='*80}\n")


async def run_single_evaluation(
    task_id: str,
    task_data: Dict[str, Any],
    agent: Agent,
    run_number: int,
    config: str
) -> EvaluationResult:
    """
    Run a single evaluation.
    
    Args:
        task_id: Task identifier
        task_data: Task data with initial_url and goal
        agent: Agent to use
        run_number: Run number for logging
        config: Config name (e.g., "loop", "loop-smcp", "code-smcp")
        
    Returns:
        EvaluationResult with metrics
    """
    # Parse config to determine mode and whether using tools
    mode = "code" if config.startswith("code") else "loop"
    with_tools = "smcp" in config
    
    print(f"\n[Run {run_number}] Evaluating {task_id} ({config})...")
    
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
    result_str = ""  # Initialize to avoid UnboundLocalError
    
    # Code mode metrics (TODO: implement collection)
    codegen_overhead = 0.0
    ttft = 0.0
    num_llm_calls = 0
    
    try:
        # Run agent
        result = await executor.run(
            task, 
            mode=mode,
            initial_url=initial_url
        )
        
        # Extract actual metrics from AgentHistoryList
        if result and hasattr(result, 'number_of_steps'):
            num_actions = result.number_of_steps()
        else:
            num_actions = 0
        
        # Get final result string (not truncated)
        if result and hasattr(result, 'final_result'):
            final_result = result.final_result()
            result_str = final_result if final_result else str(result)
        else:
            result_str = str(result)
        
        # TODO: Extract code mode metrics from executor
        # codegen_overhead = executor.get_codegen_overhead()
        # ttft = executor.get_time_to_first_token()
        # num_llm_calls = executor.get_num_llm_calls()
        
        success = True
        print(f"  ✓ Success in {time.time() - start_time:.2f}s ({num_actions} steps)")
        
    except Exception as e:
        error = str(e)
        print(f"  ✗ Failed: {error}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        
    finally:
        await executor.cleanup()
    
    latency = time.time() - start_time
    
    return EvaluationResult(
        task_id=task_id,
        config=config,
        run_number=run_number,
        success=success,
        latency_seconds=latency,
        num_actions=num_actions,
        error=error,
        result=result_str,  # Full result, not truncated
        codegen_overhead_seconds=codegen_overhead,
        time_to_first_token_seconds=ttft,
        num_llm_calls=num_llm_calls
    )


async def evaluate_task(
    task_id: str,
    task_data: Dict[str, Any],
    tools_path: str,
    num_runs: int,
    configs: List[str]
) -> EvaluationSummary:
    """
    Evaluate a task across multiple configurations.
    
    Args:
        task_id: Task identifier
        task_data: Task data with initial_url and goal
        tools_path: Path to generated tools JSON
        num_runs: Number of runs for each configuration
        configs: List of config names to test (e.g., ["loop", "loop-smcp", "code-smcp"])
        
    Returns:
        EvaluationSummary with comparison across configs
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {task_id}")
    print(f"Goal: {task_data.get('goal')}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Configs: {', '.join(configs)}")
    print(f"{'='*80}")
    
    # Load tools if provided
    agent_with_tools = None
    if tools_path and Path(tools_path).exists():
        print(f"\nLoading tools from: {tools_path}")
        agent_with_tools = Agent.from_smcp_registry(tools_path)
        print(f"Loaded {len(agent_with_tools.tools)} SMCP tools")
        
        # Print tool details
        for tool in agent_with_tools.tools:
            tool_type = tool.tool_executor_type.value if hasattr(tool, 'tool_executor_type') else 'unknown'
            print(f"  - {tool.name} ({tool_type}): {tool.description[:80]}...")
    else:
        print(f"\nWarning: Tools file not found: {tools_path}")
        print("Will only run configs without tools")
    
    # Create baseline agent (no tools)
    baseline_agent = Agent(description="", tools=[])
    
    # Collect all results
    all_results: List[EvaluationResult] = []
    
    # Run each config
    for config in configs:
        print(f"\n{'='*60}")
        print(f"RUNNING CONFIG: {config}")
        print(f"{'='*60}")
        
        # Determine which agent to use
        needs_tools = "smcp" in config
        if needs_tools and not agent_with_tools:
            print(f"Skipping {config} - no tools loaded")
            continue
        
        agent = agent_with_tools if needs_tools else baseline_agent
        
        # Run all iterations for this config
        for i in range(num_runs):
            result = await run_single_evaluation(
                task_id, task_data, agent, i + 1, config
            )
            all_results.append(result)
    
    # Calculate aggregated statistics
    def calc_config_stats(results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate stats for a set of results."""
        if not results:
            return {}
        
        success_rate = sum(1 for r in results if r.success) / len(results)
        avg_latency = sum(r.latency_seconds for r in results) / len(results)
        avg_actions = sum(r.num_actions for r in results) / len(results)
        
        stats = {
            'success_rate': success_rate,
            'avg_latency': avg_latency,
            'avg_actions': avg_actions
        }
        
        # Add code mode metrics if applicable
        code_results = [r for r in results if r.num_llm_calls > 0]
        if code_results:
            stats['avg_codegen_overhead'] = sum(r.codegen_overhead_seconds for r in code_results) / len(code_results)
            stats['avg_ttft'] = sum(r.time_to_first_token_seconds for r in code_results) / len(code_results)
            stats['avg_llm_calls'] = sum(r.num_llm_calls for r in code_results) / len(code_results)
        
        return stats
    
    # Aggregate by config
    config_stats = {}
    for config in configs:
        config_results = [r for r in all_results if r.config == config]
        if config_results:
            config_stats[config] = calc_config_stats(config_results)
    
    # Aggregate by task and config (useful when evaluating multiple tasks)
    task_config_stats = {
        task_id: config_stats  # For now, just one task per call
    }
    
    summary = EvaluationSummary(
        task_ids=[task_id],
        configs=configs,
        results=all_results,
        config_stats=config_stats,
        task_config_stats=task_config_stats
    )
    
    return summary


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate task performance across different configurations"
    )
    parser.add_argument(
        "task_prefix", 
        nargs='?',
        default="",
        help="Task ID prefix to evaluate (e.g., 'dashdish' or 'dashdish-deepresearch1'). Empty = all tasks."
    )
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
        "--configs",
        nargs='+',
        default=["loop-smcp"],
        help="Configs to test (e.g., 'loop', 'loop-smcp', 'code-smcp'). Default: loop-smcp"
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
        all_tasks = yaml.safe_load(f)
    
    # Filter tasks by prefix
    if args.task_prefix:
        matching_tasks = [t for t in all_tasks if t.get('id', '').startswith(args.task_prefix)]
        if not matching_tasks:
            print(f"Error: No tasks matching prefix '{args.task_prefix}' found in {tasks_file}")
            sys.exit(1)
        print(f"Found {len(matching_tasks)} task(s) matching prefix '{args.task_prefix}'")
    else:
        matching_tasks = all_tasks
        print(f"Evaluating all {len(matching_tasks)} tasks")
    
    # For now, handle single task (can extend to multiple later)
    if len(matching_tasks) > 1:
        print(f"Note: Multi-task evaluation not yet supported, using first match only")
    
    task_data = matching_tasks[0]
    task_id = task_data.get('id')
    
    # Determine tools path
    tools_path = args.tools
    if not tools_path:
        # Default to experiments/tools/<task_id>.json
        tools_path = f"experiments/tools/{task_id}.json"
    
    # Run evaluation
    summary = await evaluate_task(
        task_id,
        task_data,
        tools_path,
        args.runs,
        configs=args.configs
    )
    
    # Print summary
    summary.print_summary()
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_dir = Path("experiments/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{task_id}_evaluation.json")
    
    if args.incremental and Path(output_path).exists():
        # Load existing results and merge
        with open(output_path, 'r') as f:
            existing = json.load(f)
        
        # Merge results
        existing_results = [EvaluationResult(**r) for r in existing.get('results', [])]
        all_results = existing_results + summary.results
        
        # Rebuild summary with all results
        all_task_ids = list(set([r.task_id for r in all_results]))
        all_configs = list(set([r.config for r in all_results]))
        
        # Recalculate stats
        def calc_config_stats(results: List[EvaluationResult]) -> Dict[str, float]:
            if not results:
                return {}
            success_rate = sum(1 for r in results if r.success) / len(results)
            avg_latency = sum(r.latency_seconds for r in results) / len(results)
            avg_actions = sum(r.num_actions for r in results) / len(results)
            stats = {
                'success_rate': success_rate,
                'avg_latency': avg_latency,
                'avg_actions': avg_actions
            }
            code_results = [r for r in results if r.num_llm_calls > 0]
            if code_results:
                stats['avg_codegen_overhead'] = sum(r.codegen_overhead_seconds for r in code_results) / len(code_results)
                stats['avg_ttft'] = sum(r.time_to_first_token_seconds for r in code_results) / len(code_results)
                stats['avg_llm_calls'] = sum(r.num_llm_calls for r in code_results) / len(code_results)
            return stats
        
        config_stats = {}
        for config in all_configs:
            config_results = [r for r in all_results if r.config == config]
            if config_results:
                config_stats[config] = calc_config_stats(config_results)
        
        task_config_stats = {}
        for task in all_task_ids:
            task_config_stats[task] = {}
            for config in all_configs:
                task_config_results = [r for r in all_results if r.task_id == task and r.config == config]
                if task_config_results:
                    task_config_stats[task][config] = calc_config_stats(task_config_results)
        
        merged_summary = EvaluationSummary(
            task_ids=all_task_ids,
            configs=all_configs,
            results=all_results,
            config_stats=config_stats,
            task_config_stats=task_config_stats
        )
        
        data = merged_summary.to_dict()
    else:
        data = summary.to_dict()
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved detailed results to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
