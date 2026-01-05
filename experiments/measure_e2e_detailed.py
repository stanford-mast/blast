"""
Detailed E2E latency measurement with timing breakdowns.

Measures:
1. Best/worst cost candidates per model
2. Loop mode baselines (with/without tools)
3. Serial retry mode (random selection with max_iterations=3)

Captures detailed timing:
- Planning time (code generation)
- Execution time (running code/loop)
- LLM inference breakdown (prefill vs decode)
- Correctness percentage

Output format: JSON with normalized timing (sum to 1.0 for stacked bars)
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# CRITICAL: Set BEFORE importing anything
# 1. Enable browser-use logging BEFORE importing browser-use modules
os.environ['BROWSER_USE_SETUP_LOGGING'] = 'true'
# 2. Set log level (default to DEBUG for full visibility, respects BLASTAI_LOG_LEVEL env var)
os.environ.setdefault('BLASTAI_LOG_LEVEL', 'INFO')

import rich_click as click
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from blastai.agents.models import Agent
from blastai.agents.executor import AgentExecutor
from blastai.agents.timing_tracker import TimingTracker
from experiments.tasks.dashdish_deepresearch1 import validator

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Configuration for a test run."""
    name: str
    mode: str  # "code" or "loop"
    model: Optional[str] = None
    code: Optional[str] = None
    use_tools: bool = True
    max_iterations: int = 1
    run_id: Optional[int] = None  # For tracking back to original evaluation
    planning_time: float = 0.0  # Pre-loaded from evaluation data


def load_task_def(tasks_file: Path, task_id: str) -> Dict[str, Any]:
    import yaml
    data = yaml.safe_load(tasks_file.read_text())
    tasks = data if isinstance(data, list) else data.get('tasks', [])
    for t in tasks:
        if t.get("id") == task_id:
            return t
    raise ValueError(f"Task id '{task_id}' not found in {tasks_file}")


def parse_markdown_runs(md_path: Path) -> Dict[int, Tuple[str, str, bool, float, float]]:
    """Parse markdown to extract: run_id -> (model, code, passed, estimated_cost, planning_latency)."""
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    
    pattern = re.compile(r"^## Run\s+(\d+)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    
    runs = {}
    for i, m in enumerate(matches):
        run_id = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end]
        
        model_match = re.search(r'- Model: `([^`]+)`', section)
        model = model_match.group(1) if model_match else "unknown"
        
        pass_match = re.search(r'- Overall Pass: ([✓✗])', section)
        passed = pass_match.group(1) == '✓' if pass_match else False
        
        cost_match = re.search(r'- Estimated Cost: (-?[\d.]+)s', section)
        estimated_cost = float(cost_match.group(1)) if cost_match else 0.0
        # Clamp to 0 (negative costs are data errors)
        estimated_cost = max(0.0, estimated_cost)
        
        # Extract planning/generation latency (may be negative)
        planning_match = re.search(r'- Generation Latency: (-?[\d.]+)s', section)
        planning_latency = float(planning_match.group(1)) if planning_match else 0.0
        # Clamp to 0 (negative latencies are data errors)
        planning_latency = max(0.0, planning_latency)
        
        fence = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        code = fence.group(1).strip() if fence else ""
        
        runs[run_id] = (model, code, passed, estimated_cost, planning_latency)
    
    return runs


def load_planning_times(json_path: Path) -> Dict[int, float]:
    """Load generation latencies from JSON evaluation file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        # Handle both list format (new) and dict format (old)
        if isinstance(data, list):
            results = data
        else:
            results = data.get('results', [])
        
        planning_times = {}
        for idx, result in enumerate(results):
            # Use index + 1 as run_id if not explicitly provided
            run_id = result.get('run_id', idx + 1)
            gen_latency = result.get('generation_latency', 0.0)
            planning_times[run_id] = max(0.0, gen_latency)  # Avoid negative
        
        return planning_times
    except Exception as e:
        logger.warning(f"Could not load planning times from {json_path}: {e}")
        return {}


async def execute_with_timing(
    config: TestConfig,
    agent: Agent,
    task_goal: str,
    initial_url: str,
    user_id: str,
    timing_tracker: TimingTracker
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Execute a test config and track timing.
    
    Returns: (result, error, correctness_pct)
    """
    os.environ.setdefault("BROWSER_DISABLE_GPU", "true")
    
    timing_tracker.reset()
    
    # NOTE: For pre-generated code mode, planning already happened during evaluation.
    # We set planning_seconds from the config's planning_time (from markdown evaluation).
    # For generation mode, planning happens during executor.run() and is captured automatically.
    if config.planning_time is not None and config.planning_time > 0:
        timing_tracker._timing.planning_seconds = config.planning_time
        logger.info(f"Set planning_seconds = {config.planning_time:.3f}s from config")
    else:
        logger.info(f"No planning time: planning_time={config.planning_time}")
    
    if config.mode == "code" and config.code:
        # Code mode with pre-generated code (no planning needed)
        executor = AgentExecutor(
            agent=agent,
            user_id=user_id,
            timezone=os.getenv("BLASTAI_TIMEZONE", "America/Los_Angeles"),
            stop_if_codegen_fails=True,
            disable_ai_exec_fallback=True,
            timing_tracker=timing_tracker,  # Pass tracker
        )
        
        try:
            # Try to start browser with timeout
            try:
                await asyncio.wait_for(executor._ensure_browser_started(), timeout=30.0)
            except asyncio.TimeoutError:
                return None, "Browser startup timeout (30s)", 0.0
            if executor.python_executor is None:
                executor.python_executor = executor._create_python_executor()
            
            if initial_url:
                try:
                    await executor.python_executor.state["goto"](initial_url)
                except Exception as e:
                    logger.warning(f"goto({initial_url}) failed: {e}")
            
            # Execute code with timing
            from blastai.agents.timing_tracker import set_current_tracker
            set_current_tracker(timing_tracker)  # Ensure tracker is available to ai_eval calls
            try:
                timing_tracker.start_execution()
                try:
                    result = await executor.python_executor(config.code)
                    timing_tracker.end_execution()
                    
                    if result.error:
                        return None, result.error, 0.0
                    
                    # Validate with percentage
                    if result.output:
                        validation = await validator.validate(str(result.output), return_pct=True)
                        correctness_pct = validation.get('correctness_pct', 0.0)
                        return str(result.output), None, correctness_pct
                    else:
                        return None, "No output", 0.0
                
                except Exception as e:
                    timing_tracker.end_execution()
                    return None, f"Execution error: {str(e)}", 0.0
            finally:
                set_current_tracker(None)  # Clear tracker
        
        finally:
            await executor.cleanup()
    
    elif config.mode == "loop":
        # Loop mode
        executor = AgentExecutor(
            agent=agent,
            user_id=user_id,
            timezone=os.getenv("BLASTAI_TIMEZONE", "America/Los_Angeles"),
            timing_tracker=timing_tracker,  # Pass tracker
        )
        
        try:
            # No planning in loop mode, goes straight to execution
            logger.info(f"Starting loop mode execution for task: {task_goal[:100]}")
            timing_tracker.start_execution()
            try:
                result = await executor.run(task_goal, mode="loop", initial_url=initial_url)
                timing_tracker.end_execution()
                
                # Extract text from result
                if result is None:
                    logger.error("Loop mode returned None")
                    return None, "Loop mode returned None", 0.0
                
                result_text = str(result)
                logger.info(f"Loop mode completed, result length: {len(result_text)}")
                
                # Validate
                validation = await validator.validate(result_text, return_pct=True)
                correctness_pct = validation.get('correctness_pct', 0.0)
                is_correct = validation.get('correct', False)
                
                logger.info(f"Validation complete: correctness={correctness_pct:.0%}, correct={is_correct}")
                
                return result_text, None, correctness_pct
            
            except Exception as e:
                timing_tracker.end_execution()
                logger.exception(f"Loop mode execution failed")
                return None, str(e), 0.0
        
        finally:
            await executor.cleanup()
    
    elif config.mode == "code" and not config.code:
        # Code mode with generation (planning + execution)
        from blastai.agents.llm_factory import LLMFactory

        executor = AgentExecutor(
            agent=agent,
            user_id=user_id,
            timezone=os.getenv("BLASTAI_TIMEZONE", "America/Los_Angeles"),
            stop_if_codegen_fails=True,
            disable_ai_exec_fallback=True,
            timing_tracker=timing_tracker,  # Pass tracker
            codegen_llm=LLMFactory.create_llm(config.model, temperature=0.0),
        )

        # Inject a CodeGenerator configured to use the target model and the
        # requested max_iterations. This ensures the executor uses our
        # desired max_iterations instead of a hardcoded default.
        try:
            from blastai.agents.codegen import CodeGenerator
            # Create a small ensemble of identical LLM instances for codegen
            llms_for_codegen = [LLMFactory.create_llm(config.model, temperature=0.0) for _ in range(4)]
            executor.code_generator = CodeGenerator(
                agent=agent,
                llms=llms_for_codegen,
                state_aware=True,
                num_candidates=len(llms_for_codegen),
                max_iterations=getattr(config, 'max_iterations', 3),
                timezone=os.getenv('BLASTAI_TIMEZONE', 'UTC'),
            )
            logger.info(f"Injected CodeGenerator for model={config.model} with max_iterations={executor.code_generator.max_iterations}")
        except Exception as e:
            logger.warning(f"Failed to inject custom CodeGenerator: {e}")
        
        try:
            # Code generation mode: executor.run() handles both planning and execution
            # with timing via the passed timing_tracker (via set_current_tracker call)
            result = await executor.run(task_goal, mode="code", initial_url=initial_url)
            
            validation = await validator.validate(str(result), return_pct=True)
            correctness_pct = validation.get('correctness_pct', 0.0)
            
            return str(result), None, correctness_pct
        
        except Exception as e:
            return None, str(e), 0.0
        
        finally:
            await executor.cleanup()
    
    return None, "Invalid config", 0.0


async def run_test(
    config: TestConfig,
    agent: Agent,
    task_goal: str,
    initial_url: str,
    user_id_base: str,
    num_trials: int = 1
) -> Dict[str, Any]:
    """Run a test configuration multiple times and collect results."""
    
    console.print(f"\n[cyan]Testing: {config.name}[/]")
    
    results = []
    for trial in range(num_trials):
        console.print(f"  Trial {trial+1}/{num_trials}...", end=" ")
        
        timing_tracker = TimingTracker()
        result, error, correctness_pct = await execute_with_timing(
            config,
            agent,
            task_goal,
            initial_url,
            f"{user_id_base}-{config.name}-{trial}",
            timing_tracker
        )
        
        timing = timing_tracker.get_timing()
        
        if error:
            console.print(f"[red]ERROR[/red] ({timing.total_seconds:.1f}s)")
            console.print(f"    {error[:100]}")
        else:
            console.print(f"[green]{correctness_pct*100:.0f}%[/green] ({timing.total_seconds:.1f}s)")
        
        results.append({
            'trial': trial,
            'timing': timing.to_dict(),
            'correctness_pct': correctness_pct,
            'error': error,
        })
    
    # Compute averages
    avg_timing = {
        'planning_seconds': sum(r['timing']['planning_seconds'] for r in results) / num_trials,
        'execution_seconds': sum(r['timing']['execution_seconds'] for r in results) / num_trials,
        'total_seconds': sum(r['timing']['total_seconds'] for r in results) / num_trials,
        'llm_total_seconds': sum(r['timing']['llm_total_seconds'] for r in results) / num_trials,
        'llm_prefill_seconds': sum(r['timing']['llm_prefill_seconds'] for r in results) / num_trials,
        'llm_decode_seconds': sum(r['timing']['llm_decode_seconds'] for r in results) / num_trials,
    }
    
    avg_correctness = sum(r['correctness_pct'] for r in results) / num_trials
    
    # Normalize timing to percentages (for stacked bar chart)
    # Breakdown: Planning (LLM planning), LLM (actual LLM API calls), Action (execution - LLM)
    total = avg_timing['total_seconds']
    if total > 0:
        # Action time = execution_seconds minus the time spent in LLM API calls
        action_time = max(0, avg_timing['execution_seconds'] - avg_timing['llm_total_seconds'])
        normalized = {
            'planning_pct': avg_timing['planning_seconds'] / total,
            'llm_pct': avg_timing['llm_total_seconds'] / total,
            'action_pct': action_time / total,
        }
    else:
        normalized = {'planning_pct': 0, 'llm_pct': 0, 'action_pct': 0}
    
    return {
        'name': config.name,
        'mode': config.mode,
        'model': config.model,
        'num_trials': num_trials,
        'results': results,
        'avg_timing': avg_timing,
        'avg_correctness_pct': avg_correctness,
        'normalized_timing': normalized,  # Format: planning_pct, llm_pct, action_pct
    }


@click.command()
@click.option('--tasks', type=click.Path(exists=True), required=True)
@click.option('--id', 'task_id', type=str, required=True)
@click.option('--results-dir', type=click.Path(), default='experiments/results')
@click.option('--md-file', type=str, required=True)
@click.option('--json-file', type=str, default=None, help='JSON file with generation latencies')
@click.option('--models', type=str, default='gemini-2.5-flash,gemini-2.5-pro')
@click.option('--num-trials', type=int, default=1)
@click.option('--test-best/--no-test-best', default=True)
@click.option('--test-worst/--no-test-worst', default=True)
@click.option('--test-loop/--no-test-loop', default=True)
@click.option('--test-loop-tools/--no-test-loop-tools', default=True)
@click.option('--test-retry/--no-test-retry', default=False)
def main(tasks: str, task_id: str, results_dir: str, md_file: str, json_file: Optional[str],
         models: str, num_trials: int, test_best: bool, test_worst: bool, test_loop: bool,
         test_loop_tools: bool, test_retry: bool):
    """Run detailed E2E evaluation with timing breakdowns."""
    
    # Setup logging EARLY with enable_standalone_mode to ensure browser-use logs are visible
    # This respects BLASTAI_LOG_LEVEL env var, defaults to DEBUG for full visibility
    from blastai.logging_setup import enable_standalone_mode
    log_level = os.getenv('BLASTAI_LOG_LEVEL', 'INFO')
    enable_standalone_mode(browser_use_log_level=log_level)
    
    logger.info(f"Logging configured with BLASTAI_LOG_LEVEL={log_level}")
    logger.info(f"BROWSER_USE_SETUP_LOGGING={os.environ.get('BROWSER_USE_SETUP_LOGGING')}")
    
    os.environ['HEADLESS'] = 'false'
    
    # Load task
    task_def = load_task_def(Path(tasks), task_id)
    initial_url = task_def.get('initial_url', '')
    user_id_base = task_def.get('user_id', f"e2e-{task_id}")
    task_goal = task_def.get('goal', '')
    smcp_registry = task_def.get('smcp_registry')
    
    # Load agent with tools
    if smcp_registry and Path(smcp_registry).exists():
        agent_with_tools = Agent.from_smcp_registry(smcp_registry)
        console.print(f"[green]Loaded {len(agent_with_tools.tools)} SMCP tools[/]")
    else:
        agent_with_tools = Agent(description='', tools=[])
    
    agent_no_tools = Agent(description='', tools=[])
    
    # Parse markdown for candidates
    md_path = Path(results_dir) / md_file
    runs = parse_markdown_runs(md_path)
    console.print(f"[green]Parsed {len(runs)} runs[/]")
    
    # Load planning times if JSON provided
    planning_times = {}
    if json_file:
        json_path = Path(results_dir) / json_file
        if json_path.exists():
            planning_times = load_planning_times(json_path)
            console.print(f"[green]Loaded planning times for {len(planning_times)} runs[/]")
        else:
            console.print(f"[yellow]Warning: JSON file {json_path} not found[/]")
    
    # Build test configs
    model_list = [m.strip() for m in models.split(',')]
    configs = []
    
    for model in model_list:
        # Find best and worst by cost (passing runs only)
        # When costs are tied, use planning latency as tiebreaker
        best_run, worst_run = None, None
        best_cost, worst_cost = float('inf'), 0
        best_planning = 0.0
        worst_planning = float('inf')
        
        for run_id, (run_model, code, passed, cost, planning_latency) in runs.items():
            if run_model == model and passed:
                # Best: lowest cost, with planning latency as tiebreaker
                if cost < best_cost or (cost == best_cost and planning_latency < best_planning):
                    best_cost = cost
                    best_planning = planning_latency
                    best_run = (run_id, code)
                # Worst: highest cost, with planning latency as tiebreaker (prefer higher planning if tied)
                if cost > worst_cost or (cost == worst_cost and planning_latency < worst_planning):
                    worst_cost = cost
                    worst_planning = planning_latency
                    worst_run = (run_id, code)
        
        if test_best and best_run:
            # Look up planning time from planning_times dict if available
            best_planning_from_json = planning_times.get(best_run[0], best_planning)
            config = TestConfig(
                name=f"{model.replace('/', '-')}-best",
                mode="code",
                model=model,
                code=best_run[1],
                run_id=best_run[0],
                planning_time=best_planning_from_json,
            )
            logger.info(f"Best config for {model}: run_id={best_run[0]}, cost={best_cost:.2f}s, planning_time={best_planning_from_json:.3f}s")
            console.print(f"[dim]Best config for {model}: run_id={best_run[0]}, cost={best_cost:.2f}s, planning_time={best_planning_from_json:.3f}s[/]")
            configs.append(config)
        
        if test_worst and worst_run:
            # Look up planning time from planning_times dict if available
            worst_planning_from_json = planning_times.get(worst_run[0], worst_planning)
            config = TestConfig(
                name=f"{model.replace('/', '-')}-worst",
                mode="code",
                model=model,
                code=worst_run[1],
                run_id=worst_run[0],
                planning_time=worst_planning_from_json,
            )
            logger.info(f"Worst config for {model}: run_id={worst_run[0]}, cost={worst_cost:.2f}s, planning_time={worst_planning_from_json:.3f}s")
            configs.append(config)
        # Retry serial mode: run code-generation with up to 3 iterations and treat
        # as a distinct configuration that exercises the serial retry behaviour.
        if test_retry:
            configs.append(TestConfig(
                name=f"{model.replace('/', '-')}-serial-retry",
                mode="code",
                model=model,
                code=None,
                use_tools=True,
                max_iterations=3,
            ))
    
    # Baselines
    if test_loop:
        configs.append(TestConfig(
            name="loop-baseline",
            mode="loop",
            use_tools=False,
        ))
    
    if test_loop_tools:
        configs.append(TestConfig(
            name="loop-tools-baseline",
            mode="loop",
            use_tools=True,
        ))
    
    console.print(f"\n[bold]Running {len(configs)} configurations, {num_trials} trials each[/]")
    
    # Run all tests with error handling
    all_results = []
    for config in configs:
        agent = agent_with_tools if config.use_tools else agent_no_tools
        try:
            result = asyncio.run(run_test(
                config, agent, task_goal, initial_url, user_id_base, num_trials
            ))
            all_results.append(result)
        except Exception as e:
            console.print(f"[red]✗ {config.name}: FAILED[/]")
            console.print(f"  [red]{str(e)[:100]}[/]")
            # Still add a result record with error
            all_results.append({
                'name': config.name,
                'mode': config.mode,
                'model': config.model,
                'num_trials': num_trials,
                'results': [],
                'avg_timing': {},
                'avg_correctness_pct': 0.0,
                'normalized_timing': {},
                'error': str(e)[:200],
            })
    
    # Save results
    output = {
        'task_id': task_id,
        'num_trials': num_trials,
        'results': all_results,
    }
    
    out_path = Path(results_dir) / f"{task_id}_e2e_detailed.json"
    out_path.write_text(json.dumps(output, indent=2))
    console.print(f"\n[green]✓ Saved results to {out_path}[/]")
    
    # Print summary
    console.print("\n[bold]Summary (Normalized Timing):[/]")
    for r in all_results:
        norm = r['normalized_timing']
        avg_t = r['avg_timing']
        console.print(f"\n  {r['name']}:")
        console.print(f"    Total: {avg_t['total_seconds']:.1f}s")
        console.print(f"    Planning: {norm['planning_pct']*100:.1f}% ({avg_t['planning_seconds']:.1f}s)")
        console.print(f"    LLM: {norm['llm_pct']*100:.1f}% ({avg_t['llm_total_seconds']:.1f}s)")
        console.print(f"    Action: {norm['action_pct']*100:.1f}% ({avg_t['execution_seconds'] - avg_t['llm_total_seconds']:.1f}s)")
        console.print(f"    Correctness: {r['avg_correctness_pct']*100:.0f}%")


if __name__ == '__main__':
    main()
