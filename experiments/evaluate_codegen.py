"""
Evaluate code generation variance, latency, and validation across multiple candidates.

This script analyzes the performance of the code generator by generating multiple
candidates in parallel and collecting detailed metrics on:
- Latency range (time to generate each candidate)
- Cost range (estimated execution cost of each candidate)
- Validation success (how many iterations needed to get valid code)
- Code quality (display best/worst candidates)

Usage:
    python experiments/evaluate_codegen.py <task_id> --candidates <num> [--runs <num_runs>]

Example:
    python experiments/evaluate_codegen.py dashdish-deepresearch1 --candidates 10 --runs 3
"""

import asyncio
import argparse
import yaml
import json
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Enable standalone mode for proper logging
from blastai.logging_setup import enable_standalone_mode
enable_standalone_mode(browser_use_log_level="INFO")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from blastai.agents import Agent
from blastai.agents.codegen import CodeGenerator, CodeCandidate
from browser_use.llm.base import BaseChatModel
from browser_use.llm.openai.chat import ChatOpenAI


logger = logging.getLogger(__name__)


@dataclass
class CandidateMetrics:
    """Metrics for a single code generation candidate."""
    candidate_id: int
    success: bool
    code: str
    cost: float
    latency_seconds: float
    iterations_to_valid: int  # 0 if never became valid
    validation_error: Optional[str] = None
    
    # Timing breakdown
    llm_time: float = 0.0
    validation_time: float = 0.0
    fix_time: float = 0.0
    llm_time_percentage: float = 0.0
    validation_time_percentage: float = 0.0
    fix_time_percentage: float = 0.0
    
    # LLM streaming metrics (averaged across all iterations)
    avg_time_to_first_token: Optional[float] = None
    avg_tokens_per_second: Optional[float] = None
    
    # Iteration failure tracking
    total_iterations: int = 0
    failed_iterations: int = 0
    iteration_failure_rate: float = 0.0  # Percentage of iterations that failed
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CodegenEvaluationResult:
    """Result of a single codegen evaluation run."""
    task_id: str
    run_number: int
    num_candidates: int
    
    # Overall metrics
    total_latency_seconds: float
    num_valid: int
    num_invalid: int
    
    # Cost statistics
    cost_range_min: float
    cost_range_max: float
    cost_mean: float
    
    # Latency statistics
    latency_range_min: float
    latency_range_max: float
    latency_mean: float
    
    # Best/worst candidates
    lowest_cost_candidate: CandidateMetrics
    highest_cost_candidate: CandidateMetrics
    fastest_candidate: CandidateMetrics
    
    # Validation statistics
    valid_on_iteration: Dict[int, int]  # iteration -> count
    never_valid_count: int
    
    # Iteration failure statistics (across all candidates)
    total_iterations_all_candidates: int  # Sum of all iteration attempts
    failed_iterations_all_candidates: int  # Sum of all failed iterations
    overall_iteration_failure_rate: float  # Percentage of all iterations that failed
    
    # All candidates
    all_candidates: List[CandidateMetrics]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'run_number': self.run_number,
            'num_candidates': self.num_candidates,
            'total_latency_seconds': self.total_latency_seconds,
            'num_valid': self.num_valid,
            'num_invalid': self.num_invalid,
            'cost_range_min': self.cost_range_min,
            'cost_range_max': self.cost_range_max,
            'cost_mean': self.cost_mean,
            'latency_range_min': self.latency_range_min,
            'latency_range_max': self.latency_range_max,
            'latency_mean': self.latency_mean,
            'lowest_cost_candidate': self.lowest_cost_candidate.to_dict(),
            'highest_cost_candidate': self.highest_cost_candidate.to_dict(),
            'fastest_candidate': self.fastest_candidate.to_dict(),
            'valid_on_iteration': self.valid_on_iteration,
            'never_valid_count': self.never_valid_count,
            'all_candidates': [c.to_dict() for c in self.all_candidates]
        }
    
    def print_summary(self):
        """Print human-readable summary."""
        print(f"\n{'='*80}")
        print(f"CODEGEN EVALUATION: {self.task_id} (Run {self.run_number})")
        print(f"{'='*80}")
        print(f"Candidates: {self.num_candidates}")
        print(f"Valid: {self.num_valid} ({self.num_valid/self.num_candidates*100:.1f}%)")
        print(f"Invalid: {self.num_invalid} ({self.num_invalid/self.num_candidates*100:.1f}%)")
        print(f"Total Time: {self.total_latency_seconds:.2f}s")
        print()
        
        print("COST STATISTICS:")
        print(f"  Range: {self.cost_range_min:.2f}s - {self.cost_range_max:.2f}s")
        print(f"  Mean: {self.cost_mean:.2f}s")
        print()
        
        print("LATENCY STATISTICS:")
        print(f"  Range: {self.latency_range_min:.2f}s - {self.latency_range_max:.2f}s")
        print(f"  Mean: {self.latency_mean:.2f}s")
        print()
        
        # Calculate average timing breakdown across all candidates
        valid_candidates = [c for c in self.all_candidates if c.success]
        if valid_candidates:
            avg_llm_pct = sum(c.llm_time_percentage for c in valid_candidates) / len(valid_candidates)
            avg_validation_pct = sum(c.validation_time_percentage for c in valid_candidates) / len(valid_candidates)
            avg_fix_pct = sum(c.fix_time_percentage for c in valid_candidates) / len(valid_candidates)
            print("TIMING BREAKDOWN (Valid Candidates):")
            print(f"  LLM: {avg_llm_pct:.1f}%")
            print(f"  Validation: {avg_validation_pct:.1f}%")
            print(f"  Fixes: {avg_fix_pct:.1f}%")
            print()
            
            # Calculate average streaming metrics
            candidates_with_ttft = [c for c in valid_candidates if c.avg_time_to_first_token is not None]
            candidates_with_speed = [c for c in valid_candidates if c.avg_tokens_per_second is not None]
            if candidates_with_ttft or candidates_with_speed:
                print("LLM STREAMING METRICS (Valid Candidates):")
                if candidates_with_ttft:
                    avg_ttft = sum(c.avg_time_to_first_token for c in candidates_with_ttft) / len(candidates_with_ttft)
                    print(f"  Avg Time to First Token: {avg_ttft:.2f}s")
                if candidates_with_speed:
                    avg_speed = sum(c.avg_tokens_per_second for c in candidates_with_speed) / len(candidates_with_speed)
                    print(f"  Avg Token Generation Speed: {avg_speed:.1f} tokens/sec")
                print()
        
        print("VALIDATION STATISTICS:")
        for iteration in sorted(self.valid_on_iteration.keys()):
            count = self.valid_on_iteration[iteration]
            print(f"  Valid on iteration {iteration}: {count} ({count/self.num_candidates*100:.1f}%)")
        print(f"  Never valid: {self.never_valid_count} ({self.never_valid_count/self.num_candidates*100:.1f}%)")
        print()
        
        print("ITERATION FAILURE STATISTICS:")
        print(f"  Total iterations (all candidates): {self.total_iterations_all_candidates}")
        print(f"  Failed iterations: {self.failed_iterations_all_candidates}")
        print(f"  Overall failure rate: {self.overall_iteration_failure_rate:.1f}%")
        print(f"  (This shows how often the LLM generates invalid code)")
        print()
        
        print(f"{'='*80}")
        print("LOWEST COST CANDIDATE:")
        print(f"Cost: {self.lowest_cost_candidate.cost:.2f}s, Latency: {self.lowest_cost_candidate.latency_seconds:.2f}s")
        print(f"Timing: LLM={self.lowest_cost_candidate.llm_time:.2f}s ({self.lowest_cost_candidate.llm_time_percentage:.1f}%), Val={self.lowest_cost_candidate.validation_time:.2f}s ({self.lowest_cost_candidate.validation_time_percentage:.1f}%), Fix={self.lowest_cost_candidate.fix_time:.2f}s")
        print(f"```python\n{self.lowest_cost_candidate.code}\n```")
        print()
        
        print(f"{'='*80}")
        print("HIGHEST COST CANDIDATE:")
        print(f"Cost: {self.highest_cost_candidate.cost:.2f}s, Latency: {self.highest_cost_candidate.latency_seconds:.2f}s")
        print(f"Timing: LLM={self.highest_cost_candidate.llm_time:.2f}s ({self.highest_cost_candidate.llm_time_percentage:.1f}%), Val={self.highest_cost_candidate.validation_time:.2f}s ({self.highest_cost_candidate.validation_time_percentage:.1f}%), Fix={self.highest_cost_candidate.fix_time:.2f}s")
        print(f"```python\n{self.highest_cost_candidate.code}\n```")
        print()
        
        print(f"{'='*80}")
        print("FASTEST CANDIDATE:")
        print(f"Cost: {self.fastest_candidate.cost:.2f}s, Latency: {self.fastest_candidate.latency_seconds:.2f}s")
        print(f"Timing: LLM={self.fastest_candidate.llm_time:.2f}s ({self.fastest_candidate.llm_time_percentage:.1f}%), Val={self.fastest_candidate.validation_time:.2f}s ({self.fastest_candidate.validation_time_percentage:.1f}%), Fix={self.fastest_candidate.fix_time:.2f}s")
        print(f"```python\n{self.fastest_candidate.code}\n```")
        print()
        
        if self.never_valid_count > 0:
            print(f"{'='*80}")
            print("FAILED CANDIDATES:")
            for candidate in self.all_candidates:
                if not candidate.success:
                    print(f"\nCandidate {candidate.candidate_id}:")
                    print(f"  Error: {candidate.validation_error}")
                    print(f"  Timing: LLM={candidate.llm_time:.2f}s, Val={candidate.validation_time:.2f}s, Fix={candidate.fix_time:.2f}s")
                    print(f"  Code:\n```python\n{candidate.code}\n```")
        
        print(f"{'='*80}\n")


@dataclass
class CodegenEvaluationSummary:
    """Summary across multiple runs."""
    task_ids: List[str]
    results: List[CodegenEvaluationResult]
    
    # Aggregated stats across all runs
    avg_valid_rate: float
    avg_cost_mean: float
    avg_latency_mean: float
    avg_valid_on_iteration: Dict[int, float]  # iteration -> avg percentage
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_ids': self.task_ids,
            'results': [r.to_dict() for r in self.results],
            'avg_valid_rate': self.avg_valid_rate,
            'avg_cost_mean': self.avg_cost_mean,
            'avg_latency_mean': self.avg_latency_mean,
            'avg_valid_on_iteration': self.avg_valid_on_iteration
        }
    
    def print_summary(self):
        """Print aggregated summary."""
        print(f"\n{'='*80}")
        print(f"CODEGEN EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Tasks: {', '.join(self.task_ids)}")
        print(f"Total Runs: {len(self.results)}")
        print()
        
        print("AGGREGATED STATISTICS:")
        print(f"  Avg Valid Rate: {self.avg_valid_rate*100:.1f}%")
        print(f"  Avg Cost: {self.avg_cost_mean:.2f}s")
        print(f"  Avg Latency: {self.avg_latency_mean:.2f}s")
        print()
        
        print("VALIDATION SUCCESS BY ITERATION:")
        for iteration in sorted(self.avg_valid_on_iteration.keys()):
            pct = self.avg_valid_on_iteration[iteration]
            print(f"  Iteration {iteration}: {pct*100:.1f}%")
        
        print(f"{'='*80}\n")


async def generate_candidate_with_metrics(
    generator: CodeGenerator,
    task: str,
    candidate_id: int,
    initial_state: Optional[Dict[str, Any]] = None,
    current_url: Optional[str] = None
) -> CandidateMetrics:
    """
    Generate a single candidate and collect detailed metrics.
    
    This wraps _generate_candidate to measure total latency from start to finish.
    The latency includes all LLM calls across all iterations, validation time, etc.
    
    NOTE: Latency measures wall-clock time for the entire candidate generation,
    NOT individual LLM call latency. When running in parallel, candidates will
    have similar total latency even if their individual iterations vary in speed.
    
    Args:
        generator: CodeGenerator instance
        task: Task description
        candidate_id: Candidate identifier
        initial_state: Optional initial STATE values
        current_url: Optional current URL
        
    Returns:
        CandidateMetrics with detailed information
    """
    start_time = time.time()
    iterations_to_valid = 0
    final_code = ""
    final_cost = 0.0
    success = False
    validation_error = None
    llm_time = 0.0
    validation_time = 0.0
    fix_time = 0.0
    
    # Call the actual _generate_candidate method
    candidate = await generator._generate_candidate(
        task=task,
        history=[],
        initial_error=None,
        initial_state=initial_state,
        current_url=current_url
    )
    
    latency = time.time() - start_time
    
    # Extract iteration stats and streaming metrics
    total_iters = 0
    failed_iters = 0
    avg_ttft = None
    avg_tok_per_sec = None
    
    if candidate and candidate.is_valid:
        success = True
        final_code = candidate.code
        final_cost = candidate.rank
        iterations_to_valid = candidate.iterations_used
        llm_time = candidate.llm_time
        validation_time = candidate.validation_time
        fix_time = candidate.fix_time
        total_iters = candidate.total_iterations
        failed_iters = candidate.failed_iterations
        
        # Calculate average streaming metrics across all iterations
        if candidate.llm_timings:
            ttfts = [t.time_to_first_token for t in candidate.llm_timings if t.time_to_first_token is not None]
            tok_speeds = [t.tokens_per_second for t in candidate.llm_timings if t.tokens_per_second is not None]
            avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
            avg_tok_per_sec = sum(tok_speeds) / len(tok_speeds) if tok_speeds else None
            
    elif candidate:
        final_code = candidate.code
        validation_error = candidate.validation_error
        iterations_to_valid = 0  # Never became valid
        llm_time = candidate.llm_time
        validation_time = candidate.validation_time
        fix_time = candidate.fix_time
        total_iters = candidate.total_iterations
        failed_iters = candidate.failed_iterations
        
        # Calculate streaming metrics even for failed candidates
        if candidate.llm_timings:
            ttfts = [t.time_to_first_token for t in candidate.llm_timings if t.time_to_first_token is not None]
            tok_speeds = [t.tokens_per_second for t in candidate.llm_timings if t.tokens_per_second is not None]
            avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
            avg_tok_per_sec = sum(tok_speeds) / len(tok_speeds) if tok_speeds else None
    else:
        final_code = "<no code generated>"
        validation_error = "Generator returned None"
        iterations_to_valid = 0
    
    # Calculate percentages
    llm_pct = (llm_time / latency * 100) if latency > 0 else 0
    validation_pct = (validation_time / latency * 100) if latency > 0 else 0
    fix_pct = (fix_time / latency * 100) if latency > 0 else 0
    iter_failure_rate = (failed_iters / total_iters * 100) if total_iters > 0 else 0
    
    return CandidateMetrics(
        candidate_id=candidate_id,
        success=success,
        code=final_code,
        cost=final_cost,
        latency_seconds=latency,
        iterations_to_valid=iterations_to_valid,
        validation_error=validation_error,
        llm_time=llm_time,
        validation_time=validation_time,
        fix_time=fix_time,
        llm_time_percentage=llm_pct,
        validation_time_percentage=validation_pct,
        fix_time_percentage=fix_pct,
        avg_time_to_first_token=avg_ttft,
        avg_tokens_per_second=avg_tok_per_sec,
        total_iterations=total_iters,
        failed_iterations=failed_iters,
        iteration_failure_rate=iter_failure_rate
    )


async def evaluate_codegen(
    task_id: str,
    task_data: Dict[str, Any],
    agent: Agent,
    llm: BaseChatModel,
    num_candidates: int,
    run_number: int
) -> CodegenEvaluationResult:
    """
    Evaluate code generation with multiple candidates.
    
    Args:
        task_id: Task identifier
        task_data: Task data with goal
        agent: Agent with tools
        llm: LLM for code generation
        num_candidates: Number of candidates to generate
        run_number: Run number
        
    Returns:
        CodegenEvaluationResult with detailed metrics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING CODEGEN: {task_id} (Run {run_number})")
    print(f"Generating {num_candidates} candidates in parallel...")
    print(f"{'='*80}")
    
    # Create code generator
    generator = CodeGenerator(
        agent=agent,
        llm=llm,
        state_aware=False,  # Keep simple for now
        num_candidates=1,  # We'll manage parallelism ourselves
        max_iterations=3
    )
    
    task = task_data.get('goal')
    
    # Generate candidates in parallel
    overall_start = time.time()
    
    tasks = [
        generate_candidate_with_metrics(generator, task, i + 1)
        for i in range(num_candidates)
    ]
    
    all_metrics = await asyncio.gather(*tasks)
    
    total_latency = time.time() - overall_start
    
    # Calculate statistics
    valid_candidates = [m for m in all_metrics if m.success]
    invalid_candidates = [m for m in all_metrics if not m.success]
    
    num_valid = len(valid_candidates)
    num_invalid = len(invalid_candidates)
    
    # Cost statistics (only from valid candidates)
    if valid_candidates:
        costs = [m.cost for m in valid_candidates]
        cost_min = min(costs)
        cost_max = max(costs)
        cost_mean = sum(costs) / len(costs)
        
        lowest_cost = min(valid_candidates, key=lambda m: m.cost)
        highest_cost = max(valid_candidates, key=lambda m: m.cost)
    else:
        # No valid candidates - use dummy values
        cost_min = cost_max = cost_mean = 0.0
        lowest_cost = all_metrics[0]  # Use first invalid as placeholder
        highest_cost = all_metrics[0]
    
    # Latency statistics (all candidates)
    latencies = [m.latency_seconds for m in all_metrics]
    latency_min = min(latencies)
    latency_max = max(latencies)
    latency_mean = sum(latencies) / len(latencies)
    
    fastest = min(all_metrics, key=lambda m: m.latency_seconds)
    
    # Validation statistics
    valid_on_iteration = {}
    for m in all_metrics:
        if m.success and m.iterations_to_valid > 0:
            valid_on_iteration[m.iterations_to_valid] = valid_on_iteration.get(m.iterations_to_valid, 0) + 1
    
    never_valid_count = sum(1 for m in all_metrics if not m.success)
    
    # Iteration failure statistics (across all candidates)
    total_iterations_all = sum(m.total_iterations for m in all_metrics)
    failed_iterations_all = sum(m.failed_iterations for m in all_metrics)
    overall_failure_rate = (failed_iterations_all / total_iterations_all * 100) if total_iterations_all > 0 else 0
    
    result = CodegenEvaluationResult(
        task_id=task_id,
        run_number=run_number,
        num_candidates=num_candidates,
        total_latency_seconds=total_latency,
        num_valid=num_valid,
        num_invalid=num_invalid,
        cost_range_min=cost_min,
        cost_range_max=cost_max,
        cost_mean=cost_mean,
        latency_range_min=latency_min,
        latency_range_max=latency_max,
        latency_mean=latency_mean,
        lowest_cost_candidate=lowest_cost,
        highest_cost_candidate=highest_cost,
        fastest_candidate=fastest,
        valid_on_iteration=valid_on_iteration,
        never_valid_count=never_valid_count,
        total_iterations_all_candidates=total_iterations_all,
        failed_iterations_all_candidates=failed_iterations_all,
        overall_iteration_failure_rate=overall_failure_rate,
        all_candidates=all_metrics
    )
    
    return result


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate code generation variance and performance"
    )
    parser.add_argument(
        "task_prefix",
        help="Task ID prefix to evaluate (e.g., 'dashdish-deepresearch1')"
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=10,
        help="Number of candidates to generate per run (default: 10)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation runs (default: 1)"
    )
    parser.add_argument(
        "--tools",
        help="Path to generated tools JSON",
        default=None
    )
    parser.add_argument(
        "--tasks-file",
        help="Path to tasks YAML file",
        default="experiments/tasks/agisdk/agisdk.yaml"
    )
    parser.add_argument(
        "--output",
        help="Output path for evaluation results JSON",
        default=None
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use for code generation (default: gpt-4o)"
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
    matching_tasks = [t for t in all_tasks if t.get('id', '').startswith(args.task_prefix)]
    if not matching_tasks:
        print(f"Error: No tasks matching prefix '{args.task_prefix}' found")
        sys.exit(1)
    
    task_data = matching_tasks[0]
    task_id = task_data.get('id')
    
    # Determine tools path - try multiple fallbacks
    if args.tools:
        tools_path = args.tools
    else:
        # Try exact task ID first
        tools_path = f"experiments/tools/{task_id}.json"
        if not Path(tools_path).exists():
            # Try task prefix (e.g., "dashdish" from "dashdish-deepresearch1")
            base_name = args.task_prefix.split('-')[0]
            tools_path = f"experiments/tools/{base_name}.json"
    
    # Load agent with tools
    if Path(tools_path).exists():
        print(f"Loading tools from: {tools_path}")
        agent = Agent.from_json(tools_path)
        print(f"Loaded {len(agent.tools)} tools")
    else:
        print(f"Warning: Tools file not found: {tools_path}")
        print("Using agent without tools")
        agent = Agent(description="", tools=[])
    
    # Create LLM with temperature for variation between candidates
    # Using temperature=1.0 (default) to get diverse candidates
    llm = ChatOpenAI(model=args.model, temperature=1.0)
    
    # Run evaluations
    all_results = []
    
    for run_num in range(args.runs):
        result = await evaluate_codegen(
            task_id=task_id,
            task_data=task_data,
            agent=agent,
            llm=llm,
            num_candidates=args.candidates,
            run_number=run_num + 1
        )
        
        result.print_summary()
        all_results.append(result)
    
    # Calculate aggregated statistics
    if len(all_results) > 1:
        avg_valid_rate = sum(r.num_valid / r.num_candidates for r in all_results) / len(all_results)
        avg_cost_mean = sum(r.cost_mean for r in all_results) / len(all_results)
        avg_latency_mean = sum(r.latency_mean for r in all_results) / len(all_results)
        
        # Aggregate validation statistics
        all_iterations = set()
        for r in all_results:
            all_iterations.update(r.valid_on_iteration.keys())
        
        avg_valid_on_iteration = {}
        for iteration in all_iterations:
            total_pct = sum(
                r.valid_on_iteration.get(iteration, 0) / r.num_candidates
                for r in all_results
            )
            avg_valid_on_iteration[iteration] = total_pct / len(all_results)
        
        summary = CodegenEvaluationSummary(
            task_ids=[task_id],
            results=all_results,
            avg_valid_rate=avg_valid_rate,
            avg_cost_mean=avg_cost_mean,
            avg_latency_mean=avg_latency_mean,
            avg_valid_on_iteration=avg_valid_on_iteration
        )
        
        summary.print_summary()
    else:
        summary = CodegenEvaluationSummary(
            task_ids=[task_id],
            results=all_results,
            avg_valid_rate=all_results[0].num_valid / all_results[0].num_candidates,
            avg_cost_mean=all_results[0].cost_mean,
            avg_latency_mean=all_results[0].latency_mean,
            avg_valid_on_iteration={k: v/all_results[0].num_candidates for k, v in all_results[0].valid_on_iteration.items()}
        )
    
    # Save to file (append to existing results or create new)
    if args.output:
        output_path = args.output
    else:
        output_dir = Path("experiments/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{task_id}_evaluation.json")
    
    # Load existing data if present
    if Path(output_path).exists():
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    
    # Add codegen section
    existing_data['codegen'] = summary.to_dict()
    
    # Save updated data
    with open(output_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"Saved codegen evaluation results to: {output_path}")
    print(f"Results saved under 'codegen' section")


if __name__ == "__main__":
    asyncio.run(main())
