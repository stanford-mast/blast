"""
Measure latencies using existing results and baseline loop modes (headful only).

- Reads experiments/results/<task_id>.json and <task_id>.md
- Selects a run (min or max estimated cost, or by index)
- Executes the captured Python code exactly once (no regeneration, no retries)
- Optional baselines:
    - loop: run task in loop mode without SMCP tools
    - loop-tools: run task in loop mode with SMCP tools from the task registry
- Always runs headful (never headless).
- Writes results to experiments/results/<task_id>_latencies.json
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rich_click as click
from rich.console import Console
from rich.panel import Panel

# Make project root importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from blastai.agents.models import Agent, ToolExecutorType, SMCPToolType
from blastai.agents.executor import AgentExecutor
from blastai.agents.coderun import create_python_executor, find_and_call_observe

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class JsonRun:
    index: int
    model: str
    with_protocol: bool
    generation_latency: float
    generation_cost: float
    estimated_cost: float
    overall_pass: bool
    failure_types: List[str]
    actual_latency: Optional[float]


def load_task_def(tasks_file: Path, task_id: str) -> Dict[str, Any]:
    import yaml
    tasks = yaml.safe_load(tasks_file.read_text())
    for t in tasks:
        if t.get("id") == task_id:
            return t
    raise ValueError(f"Task id '{task_id}' not found in {tasks_file}")


def load_runs_from_json(json_path: Path) -> List[JsonRun]:
    data = json.loads(json_path.read_text())
    runs: List[JsonRun] = []
    idx = 0
    for entry in data:
        if isinstance(entry, dict) and "config" in entry and "generation_latency" in entry:
            cfg = entry.get("config", {})
            codecheck = entry.get("codecheck", {})
            runs.append(
                JsonRun(
                    index=idx,
                    model=cfg.get("model", "unknown"),
                    with_protocol=bool(cfg.get("with_protocol", False)),
                    generation_latency=float(entry.get("generation_latency", 0.0)),
                    generation_cost=float(entry.get("generation_cost", 0.0)),
                    estimated_cost=float(entry.get("estimated_cost", 0.0)),
                    overall_pass=bool(codecheck.get("overall_pass", False)),
                    failure_types=list(codecheck.get("failure_types", []) or []),
                    actual_latency=entry.get("actual_latency") if isinstance(entry.get("actual_latency"), (int, float)) else None,
                )
            )
            idx += 1
    return runs


def parse_markdown_code_blocks(md_path: Path) -> Dict[int, str]:
    """Parse the markdown file into a mapping of run_index -> code string.
    Assumes sections are written in order with headings '## Run {i}' and a fenced python block.
    """
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    run_blocks: Dict[int, str] = {}

    # Split by Run sections
    # Use a regex that finds '## Run X' and captures subsequent fenced block
    pattern = re.compile(r"^## Run\s+(\d+)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, m in enumerate(matches):
        run_no = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end]
        # Find first python code fence inside section
        fence = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        code = fence.group(1).strip() if fence else ""
        run_blocks[run_no - 1] = code  # zero-based index

    return run_blocks


def select_run(runs: List[JsonRun], which: str, index: Optional[int]) -> JsonRun:
    if which == "min":
        return min(runs, key=lambda r: r.estimated_cost)
    if which == "max":
        return max(runs, key=lambda r: r.estimated_cost)
    if which == "index":
        if index is None:
            raise click.UsageError("--index must be provided when --which=index")
        try:
            return next(r for r in runs if r.index == index)
        except StopIteration:
            raise click.UsageError(f"Run index {index} not found; available 0..{len(runs)-1}")
    raise click.UsageError(f"Unknown which={which}")


async def execute_once(
    code: str,
    agent: Agent,
    initial_url: str,
    user_id: Optional[str]
) -> Tuple[float, Dict[str, Any]]:
    """Start browser, navigate to initial_url (goto), then execute code once.
    Returns (latency_seconds, details dict with logs and maybe output type).
    """
    # Enable verbose logging for this measurement
    os.environ["BROWSER_USE_LOGGING_LEVEL"] = os.getenv("BROWSER_USE_LOGGING_LEVEL", "info")
    os.environ["BROWSER_USE_DISABLE_LOGGING"] = "false"
    os.environ["ANONYMIZED_TELEMETRY"] = os.getenv("ANONYMIZED_TELEMETRY", "false")
    # Keep GPU disabled for stability unless explicitly overridden
    os.environ.setdefault("BROWSER_DISABLE_GPU", "true")

    executor = AgentExecutor(
        agent=agent,
        user_id=user_id,
        timezone=os.getenv("BLASTAI_TIMEZONE", "America/Los_Angeles"),
        stop_if_codegen_fails=True,
    )

    try:
        # Ensure browser up and python executor created
        await executor._ensure_browser_started()
        if executor.python_executor is None:
            executor.python_executor = executor._create_python_executor()

        # Navigate via goto tool (ensures observe + STATE population)
        if initial_url:
            try:
                await executor.python_executor.state["goto"](initial_url)
            except Exception as e:
                logger.warning(f"goto({initial_url}) failed: {e}")

        # Execute exactly once
        import time
        start = time.time()
        result = await executor.python_executor(code)
        latency = time.time() - start

        details = {
            "error": result.error,
            "logs": result.logs,
            "is_final_answer": result.is_final_answer,
            "output_type": type(result.output).__name__ if result.output is not None else None,
        }
        return latency, details
    finally:
        await executor.cleanup()


async def measure_loop(
    agent: Agent,
    goal: str,
    initial_url: str,
    user_id: Optional[str]
) -> Tuple[float, Dict[str, Any]]:
    """Run the task in loop mode once and measure wall time (headful, logs enabled)."""
    # Ensure logging visible
    os.environ["BROWSER_USE_LOGGING_LEVEL"] = os.getenv("BROWSER_USE_LOGGING_LEVEL", "info")
    os.environ["BROWSER_USE_DISABLE_LOGGING"] = "false"
    os.environ.setdefault("BROWSER_DISABLE_GPU", "true")

    executor = AgentExecutor(
        agent=agent,
        user_id=user_id,
        timezone=os.getenv("BLASTAI_TIMEZONE", "America/Los_Angeles"),
    )
    import time
    try:
        start = time.time()
        result = await executor.run(goal, mode="loop", initial_url=initial_url)
        latency = time.time() - start
        # Best-effort representation (result type may vary)
        desc = getattr(result, "__class__", type(result)).__name__
        return latency, {"result_type": desc}
    except Exception as e:
        return -1.0, {"error": str(e)}
    finally:
        await executor.cleanup()


@click.command()
@click.option('--tasks', type=click.Path(exists=True), required=True,
              help='Path to tasks YAML (same used for generation).')
@click.option('--id', 'task_id', type=str, required=True,
              help='Task id whose results to measure (e.g., dashdish-1).')
@click.option('--results-dir', type=click.Path(), default='experiments/results',
              help='Directory containing <task_id>.json and <task_id>.md')
@click.option('--which', type=click.Choice(['min', 'max', 'index']), default='min',
              help='Which code run to execute: min cost, max cost, or by index.')
@click.option('--index', type=int, default=None,
              help='Run index to execute when --which=index (zero-based).')
@click.option('--print-code/--no-print-code', default=False,
              help='Print the code block before execution.')
@click.option('--model', 'filter_model', default=None,
              help='Optional filter: only consider runs with this model name.')
@click.option('--with-protocol', 'filter_with_protocol', type=bool, default=None,
              help='Optional filter: only consider runs where with_protocol matches.')
@click.option('--baselines', type=click.Choice(['loop', 'loop-tools']), multiple=True, default=(),
              help='Additionally measure loop baselines: loop (no tools), loop-tools (with SMCP tools).')
def main(tasks: str, task_id: str, results_dir: str, which: str, index: Optional[int], print_code: bool,
         filter_model: Optional[str], filter_with_protocol: Optional[bool], baselines: Tuple[str, ...]):
    # Configure logging verbose for measurement
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('browser_use').setLevel(logging.INFO)

    # Always headful
    os.environ['HEADLESS'] = 'false'

    # Resolve files
    results_dir = Path(results_dir)
    json_path = results_dir / f"{task_id}.json"
    md_path = results_dir / f"{task_id}.md"

    if not json_path.exists() or not md_path.exists():
        raise click.ClickException(f"Missing results for {task_id}: expected {json_path} and {md_path}")

    # Load task def and build agent
    task_def = load_task_def(Path(tasks), task_id)
    initial_url = task_def.get('initial_url', '')
    user_id = task_def.get('user_id', f"measure-{task_id}")
    goal = task_def.get('goal', '')

    smcp_registry = task_def.get('smcp_registry')
    if smcp_registry and Path(smcp_registry).exists():
        agent = Agent.from_smcp_registry(smcp_registry)
        console.print(f"[green]Loaded {len(agent.tools)} SMCP tools from[/] {smcp_registry}")
    else:
        agent = Agent(description='', tools=[])
        console.print("[yellow]No SMCP registry; running without tools[/]")

    # Get runs and code blocks
    runs = load_runs_from_json(json_path)
    if filter_model is not None:
        runs = [r for r in runs if r.model == filter_model]
    if filter_with_protocol is not None:
        runs = [r for r in runs if r.with_protocol == filter_with_protocol]
    if not runs:
        raise click.ClickException("No matching runs in JSON after filters")

    code_blocks = parse_markdown_code_blocks(md_path)

    # Choose run
    chosen = select_run(runs, which=which, index=index)
    code = code_blocks.get(chosen.index, "")
    if not code.strip():
        raise click.ClickException(f"Selected run index {chosen.index} has empty or missing code block in markdown")

    console.print(Panel(
        f"[bold]Executing existing code[/]\n\n"
        f"Task: {task_id}\n"
        f"Model: {chosen.model}\n"
        f"With Protocol: {chosen.with_protocol}\n"
        f"Estimated Cost: {chosen.estimated_cost:.2f}s\n"
        f"Initial URL: {initial_url}",
        title="Measurement",
        border_style="blue"
    ))

    if print_code:
        console.print("\n[dim]Code:[/]\n")
        console.print(f"""```python\n{code}\n```""")

    # Run code once
    code_latency, code_details = asyncio.run(execute_once(code, agent, initial_url, user_id))

    # Baselines
    baseline_results: Dict[str, Any] = {}
    if baselines:
        # loop baseline: no tools
        if 'loop' in baselines:
            agent_no_tools = Agent(description='', tools=[])
            loop_latency, loop_details = asyncio.run(measure_loop(agent_no_tools, goal, initial_url, user_id))
            baseline_results['loop'] = {
                'latency_seconds': loop_latency,
                'details': loop_details,
            }
        # loop-tools baseline: with SMCP tools from registry
        if 'loop-tools' in baselines:
            if smcp_registry and Path(smcp_registry).exists():
                agent_with_tools = Agent.from_smcp_registry(smcp_registry)
            else:
                agent_with_tools = Agent(description='', tools=[])
            lt_latency, lt_details = asyncio.run(measure_loop(agent_with_tools, goal, initial_url, user_id))
            baseline_results['loop-tools'] = {
                'latency_seconds': lt_latency,
                'details': lt_details,
            }

    # Prepare and save combined results (fixed filename)
    out_path = Path(results_dir) / f"{task_id}_latencies.json"
    out = {
        'task_id': task_id,
        'selected': asdict(chosen),
        'code': {
            'latency_seconds': code_latency,
            'details': code_details,
            'model': chosen.model,
            'with_protocol': chosen.with_protocol,
        },
        'baselines': baseline_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    console.print(f"\n[green]✓ Saved latencies to[/] {out_path}")


if __name__ == '__main__':
    main()
