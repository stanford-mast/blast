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
from blastai.agents.llm_factory import LLMFactory
from browser_use.llm.base import BaseChatModel


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
class ExecutionResult:
    """Result of executing a generated code candidate."""
    candidate_id: int
    candidate_type: str  # 'lowest_cost', 'highest_cost'
    success: bool
    execution_latency_seconds: float
    num_actions: int
    error: str = ""
    result: str = ""


@dataclass
class CodegenEvaluationResult:
    """Result of a single codegen evaluation run."""
    task_id: str
    model_name: str
    run_number: int
    num_candidates: int
    
    # Overall metrics
    total_latency_seconds: float
    num_valid: int
    num_invalid: int
    
    # Codegen timing statistics
    fastest_candidate_time: float  # Time to generate fastest candidate
    slowest_candidate_time: float  # Time to generate slowest candidate
    
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
    
    # All candidates (required field)
    all_candidates: List[CandidateMetrics]
    
    # Execution results (if --execute was enabled) - optional fields must come last
    lowest_cost_execution: Optional[ExecutionResult] = None
    highest_cost_execution: Optional[ExecutionResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            'task_id': self.task_id,
            'model_name': self.model_name,
            'run_number': self.run_number,
            'num_candidates': self.num_candidates,
            'total_latency_seconds': self.total_latency_seconds,
            'num_valid': self.num_valid,
            'num_invalid': self.num_invalid,
            'fastest_candidate_time': self.fastest_candidate_time,
            'slowest_candidate_time': self.slowest_candidate_time,
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
            'total_iterations_all_candidates': self.total_iterations_all_candidates,
            'failed_iterations_all_candidates': self.failed_iterations_all_candidates,
            'overall_iteration_failure_rate': self.overall_iteration_failure_rate,
            'all_candidates': [c.to_dict() for c in self.all_candidates]
        }
        
        if self.lowest_cost_execution:
            data['lowest_cost_execution'] = {
                'candidate_id': self.lowest_cost_execution.candidate_id,
                'candidate_type': self.lowest_cost_execution.candidate_type,
                'success': self.lowest_cost_execution.success,
                'execution_latency_seconds': self.lowest_cost_execution.execution_latency_seconds,
                'num_actions': self.lowest_cost_execution.num_actions,
                'error': self.lowest_cost_execution.error,
                'result': self.lowest_cost_execution.result
            }
        
        if self.highest_cost_execution:
            data['highest_cost_execution'] = {
                'candidate_id': self.highest_cost_execution.candidate_id,
                'candidate_type': self.highest_cost_execution.candidate_type,
                'success': self.highest_cost_execution.success,
                'execution_latency_seconds': self.highest_cost_execution.execution_latency_seconds,
                'num_actions': self.highest_cost_execution.num_actions,
                'error': self.highest_cost_execution.error,
                'result': self.highest_cost_execution.result
            }
        
        return data
    
    def print_summary(self):
        """Print human-readable summary."""
        print(f"\n{'='*80}")
        print(f"CODEGEN EVALUATION: {self.task_id} (Run {self.run_number}, Model: {self.model_name})")
        print(f"{'='*80}")
        print(f"Candidates: {self.num_candidates}")
        print(f"Valid: {self.num_valid} ({self.num_valid/self.num_candidates*100:.1f}%)")
        print(f"Invalid: {self.num_invalid} ({self.num_invalid/self.num_candidates*100:.1f}%)")
        print(f"Total Time: {self.total_latency_seconds:.2f}s")
        print()
        
        print("GENERATION TIMING:")
        print(f"  Fastest candidate: {self.fastest_candidate_time:.2f}s")
        print(f"  Slowest candidate: {self.slowest_candidate_time:.2f}s")
        print(f"  Range: {self.slowest_candidate_time - self.fastest_candidate_time:.2f}s")
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
        
        # Print execution results if available
        if self.lowest_cost_execution or self.highest_cost_execution:
            print(f"{'='*80}")
            print("EXECUTION RESULTS:")
            print(f"{'='*80}")
            
            if self.lowest_cost_execution:
                exec_result = self.lowest_cost_execution
                status = "✓ Success" if exec_result.success else "✗ Failed"
                print(f"\nLOWEST COST CANDIDATE (Cost: {self.lowest_cost_candidate.cost:.2f}s):")
                print(f"  {status}")
                print(f"  Execution Time: {exec_result.execution_latency_seconds:.2f}s")
                print(f"  Actions: {exec_result.num_actions}")
                if exec_result.error:
                    print(f"  Error: {exec_result.error}")
                if exec_result.result:
                    print(f"  Result: {exec_result.result[:200]}...")
            
            if self.highest_cost_execution:
                exec_result = self.highest_cost_execution
                status = "✓ Success" if exec_result.success else "✗ Failed"
                print(f"\nHIGHEST COST CANDIDATE (Cost: {self.highest_cost_candidate.cost:.2f}s):")
                print(f"  {status}")
                print(f"  Execution Time: {exec_result.execution_latency_seconds:.2f}s")
                print(f"  Actions: {exec_result.num_actions}")
                if exec_result.error:
                    print(f"  Error: {exec_result.error}")
                if exec_result.result:
                    print(f"  Result: {exec_result.result[:200]}...")
            
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
    """Summary across multiple runs and models."""
    task_ids: List[str]
    models: List[str]
    results: List[CodegenEvaluationResult]
    
    # Per-model aggregated stats
    model_stats: Dict[str, Dict[str, Any]]  # model -> stats
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_ids': self.task_ids,
            'models': self.models,
            'results': [r.to_dict() for r in self.results],
            'model_stats': self.model_stats
        }
    
    def print_summary(self):
        """Print aggregated summary."""
        print(f"\n{'='*80}")
        print(f"CODEGEN EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Tasks: {', '.join(self.task_ids)}")
        print(f"Models: {', '.join(self.models)}")
        print(f"Total Runs: {len(self.results)}")
        print()
        
        # Print per-model statistics
        for model in self.models:
            stats = self.model_stats.get(model, {})
            print(f"{'='*80}")
            print(f"MODEL: {model}")
            print(f"{'='*80}")
            print(f"  Overall Iteration Failure Rate: {stats.get('avg_iteration_failure_rate', 0):.1f}%")
            print(f"  Time to Fastest Candidate: {stats.get('avg_fastest_time', 0):.2f}s")
            print(f"  Time to Slowest Candidate: {stats.get('avg_slowest_time', 0):.2f}s")
            print(f"  Avg Valid Rate: {stats.get('avg_valid_rate', 0)*100:.1f}%")
            print(f"  Avg Cost Mean: {stats.get('avg_cost_mean', 0):.2f}s")
            
            # Execution results if available
            if stats.get('has_execution_results'):
                print()
                print("  EXECUTION RESULTS:")
                lowest = stats.get('avg_lowest_cost_execution_time', 0)
                highest = stats.get('avg_highest_cost_execution_time', 0)
                print(f"    Lowest Cost Execution: {lowest:.2f}s")
                print(f"    Highest Cost Execution: {highest:.2f}s")
                print(f"    Delta: {highest - lowest:.2f}s ({(highest/lowest - 1)*100:.1f}% slower)")
            
            print()
        
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


async def execute_candidate(
    candidate: CandidateMetrics,
    candidate_type: str,
    task_data: Dict[str, Any],
    agent: Agent,
    llm: BaseChatModel
) -> ExecutionResult:
    """
    Execute a code candidate and measure its performance.
    
    Args:
        candidate: The candidate to execute
        candidate_type: 'lowest_cost' or 'highest_cost'
        task_data: Task data with initial_url and goal
        agent: Agent with tools
        llm: LLM for code generation (not used, but needed for executor)
        
    Returns:
        ExecutionResult with execution metrics
    """
    from blastai.agents import AgentExecutor
    from blastai.agents.codegen import CodeGenerator, CodeCandidate
    
    print(f"\n{'='*60}")
    print(f"EXECUTING {candidate_type.upper()} CANDIDATE")
    print(f"{'='*60}")
    print(f"Estimated Cost: {candidate.cost:.2f}s")
    print(f"Code:\n```python\n{candidate.code}\n```")
    print()
    
    # Create a custom executor with a mocked code generator that returns our pre-generated code
    executor = AgentExecutor(agent, llm=llm)
    
    # Create a mock CodeGenerator that returns our pre-selected candidate
    class MockCodeGenerator:
        async def generate_code(self, task, history, error=None, initial_state=None, current_url=None):
            # Return our pre-generated code wrapped in a CodeCandidate
            return CodeCandidate(
                code=candidate.code,
                rank=candidate.cost,
                is_valid=True,
                iterations_used=0,
                validation_error=None,
                llm_time=0,
                validation_time=0,
                fix_time=0,
                total_iterations=0,
                failed_iterations=0,
                llm_timings=[]
            )
    
    # Inject the mock generator
    executor.code_generator = MockCodeGenerator()
    
    initial_url = task_data.get('initial_url')
    task = task_data.get('goal')
    
    start_time = time.time()
    num_actions = 0
    success = False
    error = ""
    result_str = ""
    
    try:
        # Execute in code mode - will use our pre-generated code
        result = await executor.run(
            task,
            mode='code',
            initial_url=initial_url
        )
        
        # Extract metrics
        if result and hasattr(result, 'number_of_steps'):
            num_actions = result.number_of_steps()
        else:
            num_actions = 0
        
        if result and hasattr(result, 'final_result'):
            final_result = result.final_result()
            result_str = final_result if final_result else str(result)
        else:
            result_str = str(result)
        
        success = True
        print(f"  ✓ Success in {time.time() - start_time:.2f}s ({num_actions} steps)")
        
    except Exception as e:
        error = str(e)
        print(f"  ✗ Failed: {error}")
        import traceback
        traceback.print_exc()
        
    finally:
        await executor.cleanup()
    
    execution_time = time.time() - start_time
    
    return ExecutionResult(
        candidate_id=candidate.candidate_id,
        candidate_type=candidate_type,
        success=success,
        execution_latency_seconds=execution_time,
        num_actions=num_actions,
        error=error,
        result=result_str
    )


async def evaluate_codegen(
    task_id: str,
    task_data: Dict[str, Any],
    agent: Agent,
    llm: BaseChatModel,
    num_candidates: int,
    run_number: int,
    model_name: str,
    execute: bool = False
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
        model_name: Name of the model being evaluated
        execute: Whether to execute the best/worst candidates
        
    Returns:
        CodegenEvaluationResult with detailed metrics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING CODEGEN: {task_id} (Run {run_number}, Model: {model_name})")
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
    
    # Generation timing statistics
    fastest_time = min(latencies)
    slowest_time = max(latencies)
    
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
    
    # Execute candidates if requested
    lowest_cost_exec = None
    highest_cost_exec = None
    
    if execute and valid_candidates:
        print(f"\n{'='*80}")
        print("EXECUTING CANDIDATES")
        print(f"{'='*80}")
        
        # Execute lowest cost candidate
        lowest_cost_exec = await execute_candidate(
            lowest_cost,
            'lowest_cost',
            task_data,
            agent,
            llm
        )
        
        # Execute highest cost candidate
        highest_cost_exec = await execute_candidate(
            highest_cost,
            'highest_cost',
            task_data,
            agent,
            llm
        )
    
    result = CodegenEvaluationResult(
        task_id=task_id,
        model_name=model_name,
        run_number=run_number,
        num_candidates=num_candidates,
        total_latency_seconds=total_latency,
        num_valid=num_valid,
        num_invalid=num_invalid,
        fastest_candidate_time=fastest_time,
        slowest_candidate_time=slowest_time,
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
        lowest_cost_execution=lowest_cost_exec,
        highest_cost_execution=highest_cost_exec,
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
        help="Number of evaluation runs per model (default: 1)"
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
        "--models",
        nargs='+',
        default=["gpt-4o"],
        help="LLM models to use for code generation (default: gpt-4o). Can specify multiple."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the lowest and highest cost candidates"
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
    
    # Load agent with SMCP tools from registry
    if Path(tools_path).exists():
        print(f"Loading SMCP tools from: {tools_path}")
        agent = Agent.from_smcp_registry(tools_path)
        print(f"Loaded {len(agent.tools)} SMCP tools")
    else:
        print(f"Warning: Tools file not found: {tools_path}")
        print("Using agent without tools")
        agent = Agent(description="", tools=[])
    
    # Run evaluations for each model
    all_results = []
    
    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'='*80}")
        
        # Create LLM with temperature for variation between candidates
        # Using temperature=1.0 (default) to get diverse candidates
        # LLMFactory auto-detects provider from model name
        llm = LLMFactory.create_llm(model_name, temperature=1.0)
        
        for run_num in range(args.runs):
            result = await evaluate_codegen(
                task_id=task_id,
                task_data=task_data,
                agent=agent,
                llm=llm,
                num_candidates=args.candidates,
                run_number=run_num + 1,
                model_name=model_name,
                execute=args.execute
            )
            
            result.print_summary()
            all_results.append(result)
    
    # Calculate aggregated statistics per model
    if len(all_results) > 1:
        # Group results by model
        models = list(set([r.model_name for r in all_results]))
        model_stats = {}
        
        for model in models:
            model_results = [r for r in all_results if r.model_name == model]
            
            # Calculate statistics for this model
            avg_valid_rate = sum(r.num_valid / r.num_candidates for r in model_results) / len(model_results)
            avg_cost_mean = sum(r.cost_mean for r in model_results) / len(model_results)
            avg_iteration_failure_rate = sum(r.overall_iteration_failure_rate for r in model_results) / len(model_results)
            avg_fastest_time = sum(r.fastest_candidate_time for r in model_results) / len(model_results)
            avg_slowest_time = sum(r.slowest_candidate_time for r in model_results) / len(model_results)
            
            stats = {
                'avg_valid_rate': avg_valid_rate,
                'avg_cost_mean': avg_cost_mean,
                'avg_iteration_failure_rate': avg_iteration_failure_rate,
                'avg_fastest_time': avg_fastest_time,
                'avg_slowest_time': avg_slowest_time
            }
            
            # Add execution results if available
            exec_results = [r for r in model_results if r.lowest_cost_execution is not None]
            if exec_results:
                avg_lowest_exec = sum(r.lowest_cost_execution.execution_latency_seconds for r in exec_results) / len(exec_results)
                avg_highest_exec = sum(r.highest_cost_execution.execution_latency_seconds for r in exec_results) / len(exec_results)
                stats['has_execution_results'] = True
                stats['avg_lowest_cost_execution_time'] = avg_lowest_exec
                stats['avg_highest_cost_execution_time'] = avg_highest_exec
            else:
                stats['has_execution_results'] = False
            
            model_stats[model] = stats
        
        summary = CodegenEvaluationSummary(
            task_ids=[task_id],
            models=models,
            results=all_results,
            model_stats=model_stats
        )
        
        summary.print_summary()
    else:
        # Single result
        model = all_results[0].model_name
        stats = {
            'avg_valid_rate': all_results[0].num_valid / all_results[0].num_candidates,
            'avg_cost_mean': all_results[0].cost_mean,
            'avg_iteration_failure_rate': all_results[0].overall_iteration_failure_rate,
            'avg_fastest_time': all_results[0].fastest_candidate_time,
            'avg_slowest_time': all_results[0].slowest_candidate_time
        }
        
        if all_results[0].lowest_cost_execution:
            stats['has_execution_results'] = True
            stats['avg_lowest_cost_execution_time'] = all_results[0].lowest_cost_execution.execution_latency_seconds
            stats['avg_highest_cost_execution_time'] = all_results[0].highest_cost_execution.execution_latency_seconds
        else:
            stats['has_execution_results'] = False
        
        summary = CodegenEvaluationSummary(
            task_ids=[task_id],
            models=[model],
            results=all_results,
            model_stats={model: stats}
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
