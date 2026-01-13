"""
Evaluation script for codegen performance and quality metrics.

This script evaluates code generation across different configurations:
- Different LLM models (gpt-4.1, meta-llama/llama-4-maverick-17b-128e-instruct)
- With/without protocol assertions (state, preconditions, postconditions)

For each task in agisdk.yaml, generates code and measures:
- Generation latency (time to generate code)
- Generation cost (actual LLM cost from token usage)
- Estimated cost (from codecost module)
- Code validation (passes codecheck criteria)
- Actual execution latency (optional, for highest/lowest cost configs)
- Page load latency (time to load initial_url)

Results are saved to:
- JSON file: raw data for all runs with configs and metrics
- Markdown file: generated code for each run
- Summary JSON: averaged metrics per unique config
"""

import ast
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from browser_use import BrowserProfile, BrowserSession
from browser_use.llm.base import BaseChatModel

from blastai.agents import Agent, AgentExecutor, check_code_candidate, compute_code_cost
from blastai.agents.codegen import CodeGenerator
from blastai.agents.llm_factory import LLMFactory
from blastai.agents.models import CoreTool, SMCPToolType, ToolExecutorType

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class CodegenConfig:
    """Configuration for a single codegen run."""

    model: str
    with_protocol: bool
    max_iterations: int = 1  # Number of retry iterations (1 = no retries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "with_protocol": self.with_protocol,
            "max_iterations": self.max_iterations,
        }


@dataclass
class CodecheckResult:
    """Validation result capturing potentially multiple failure types even with single generation iteration.

    Fields:
      overall_pass: True if all checks (syntax, types, ordering) passed.
      failure_types: List of failure type strings (empty if overall_pass=True).
      failure_details: List of dicts with 'type' and 'message' for each failure.
      error_message: First failure message (legacy convenience field).
    """

    overall_pass: bool
    failure_types: List[str] = field(default_factory=list)
    failure_details: List[Dict[str, str]] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class CodegenResult:
    """Results from a single codegen run."""

    config: CodegenConfig
    generation_latency: float  # Time to generate code in seconds
    estimated_cost: float  # Estimated cost from codecost module in seconds
    codecheck: CodecheckResult
    actual_latency: Optional[float] = None  # Actual execution time if run
    generated_code: str = ""  # The generated code
    generation_failed: bool = False  # True if we failed to produce any code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "generation_latency": self.generation_latency,
            "estimated_cost": self.estimated_cost,
            "codecheck": {
                "overall_pass": self.codecheck.overall_pass,
                "failure_types": self.codecheck.failure_types,
                "failure_details": self.codecheck.failure_details,
                "error_message": self.codecheck.error_message,
            },
            "actual_latency": self.actual_latency,
            "generation_failed": self.generation_failed,
        }


@dataclass
class PageLoadResult:
    """Results from page load measurement."""

    initial_url: str
    load_latency: float  # Time to load page in seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {"initial_url": self.initial_url},
            "load_latency": self.load_latency,
        }


async def load_task_definitions(tasks_file: Path) -> List[Dict[str, Any]]:
    """Load task definitions from YAML file."""
    with open(tasks_file) as f:
        data = yaml.safe_load(f)
    # Handle both formats: direct list or {'tasks': [...]}
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    return data


async def create_agent_for_task(task: Dict[str, Any]) -> Agent:
    """Create an Agent with SMCP tools from registry if available."""
    smcp_registry = task.get("smcp_registry")

    if smcp_registry and Path(smcp_registry).exists():
        logger.info(f"Loading SMCP tools from {smcp_registry}")
        agent = Agent.from_smcp_registry(smcp_registry)
        logger.info(f"Loaded {len(agent.tools)} SMCP tools")
    else:
        logger.info("No SMCP registry found, creating agent with no tools")
        agent = Agent(description="", tools=[])

    return agent


async def generate_code_with_timing(
    agent: Agent,
    task: str,
    config: CodegenConfig,
    codegen_llm: BaseChatModel,
    timezone: str = "America/Los_Angeles",
    debug_print_prompt: bool = False,
    task_def: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, float, object, object]:
    """
    Generate code for a task and measure timing.

    Returns:
        Tuple of (generated_code, generation_time_seconds, unused_cost, candidate, code_generator)
    """
    from browser_use.llm.messages import UserMessage

    from blastai.agents.codegen import CodeGenerator

    # Create code generator
    code_generator = CodeGenerator(
        agent=agent,
        llm=codegen_llm,
        num_candidates=1,  # Generate single candidate
        state_aware=config.with_protocol,
        max_iterations=config.max_iterations,  # Use max_iterations from config
        timezone=timezone,
        debug_print_prompt=debug_print_prompt,
    )

    # Measure generation time
    start_time = time.time()

    # Generate code using the generate_code method
    # This requires history (list of BaseMessage) and initial state
    # NOTE: For benchmark evaluation, we know the initial state from the task definition:
    # - The browser starts at initial_url (home page)
    # - STATE starts with page="home" (or whatever the initial page is)
    # - This allows proper static validation of tool preconditions
    # TODO: Remove these deterministic assumptions for production where initial state is unknown
    initial_state = {
        "page_type": "home"
    }  # Initial state for evaluation - starts on home page
    current_url = task_def.get("initial_url", "")  # Get initial URL from task config
    history = []  # Empty history for initial generation

    # Call private _generate_candidate to retain iteration metadata
    candidate = await code_generator._generate_candidate(
        task=task,
        history=history,
        initial_error=None,
        initial_state=initial_state,
        current_url=current_url,
    )
    # Include last attempt's code even if it failed validation (useful for debugging)
    generated_code = candidate.code if candidate else ""

    generation_time = time.time() - start_time

    # Safety check: clamp negative times to 0.0 (should never happen, but guards against edge cases)
    # Negative times would indicate clock skew or timing bugs in dependencies
    if generation_time < 0:
        logger.warning(
            f"Negative generation time detected: {generation_time:.4f}s - clamping to 0.0"
        )
        generation_time = 0.0

    # Return generated code, timing, and metadata
    return generated_code or "", generation_time, 0.0, candidate, code_generator


async def validate_code(
    code: str, agent: Agent, code_generator=None, candidate: Optional[object] = None
) -> CodecheckResult:
    """Run syntax, type, and ordering checks collecting all applicable failures (no early return)."""
    if not code:
        # Distinguish between true generation failure vs extraction failure
        reason = "No code generated"
        ftype = "generation"
        if candidate is not None and getattr(candidate, "validation_error", ""):
            ve = candidate.validation_error or ""
            if "no code block" in ve.lower():
                reason = ve
                ftype = "extraction"
        return CodecheckResult(
            overall_pass=False,
            failure_types=[ftype],
            failure_details=[{"type": ftype, "message": reason}],
            error_message=reason,
        )

    # Apply code fixes: strip tool redefinitions that shadow SMCP tools
    # LLMs sometimes redefine tools with their own ai_exec implementations
    from blastai.agents.codefix import strip_tool_redefinitions

    tool_names = {tool.name for tool in getattr(agent, "tools", [])}
    if tool_names:
        code, was_modified = strip_tool_redefinitions(code, tool_names)
        if was_modified:
            logger.info(f"Stripped tool redefinitions from generated code")

    failure_types: List[str] = []
    failure_details: List[Dict[str, str]] = []
    syntax_ok = True
    types_ok = True
    ordering_ok = True
    # 1. Syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        syntax_ok = False
        msg = f"Syntax error at line {e.lineno}: {e.msg}"
        failure_types.append("syntax")
        failure_details.append({"type": "syntax", "message": msg})
    # 2. Types (only if syntax passed)
    if syntax_ok:
        definition_code = (
            code_generator._build_definition_code()
            if code_generator is not None
            else None
        )
        if definition_code is not None:
            wrapped = (
                """\nasync def _user_generated_code() -> Any:\n"""
                + "\n".join("    " + ln for ln in code.split("\n"))
                + "\n"
            )
            full_code = definition_code + "\n\n" + wrapped
            try:
                # Use same function used in check_code_candidate indirectly
                from blastai.agents.codecheck import check_ty_types

                ty_valid, ty_err = check_ty_types(full_code)
                if not ty_valid:
                    types_ok = False
                    failure_types.append("types")
                    failure_details.append(
                        {"type": "types", "message": ty_err or "Type checking failed"}
                    )
            except Exception as e:
                logger.debug(f"Type checking exception ignored: {e}")
    # 3. Check for illegal STATE/get_url access (only if syntax passed)
    # TODO: Remove this check for production use where initial state is unknown
    # For benchmark evaluation, we forbid direct STATE access/modification since:
    # - Initial state is deterministically known (page="home", url=initial_url)
    # - Tools should be used instead of manipulating STATE
    # - get_url() shouldn't be called (initial URL is known via assertion in prompt)
    if syntax_ok:
        try:
            tree = ast.parse(code)
            # Check for STATE access/modification
            for node in ast.walk(tree):
                # Check for STATE dictionary access: STATE["key"], STATE.get(...), STATE.update(...)
                if isinstance(node, ast.Subscript):
                    if isinstance(node.value, ast.Name) and node.value.id == "STATE":
                        msg = "Code illegally accesses STATE directly. Use tools instead of manipulating STATE."
                        failure_types.append("state-access")
                        failure_details.append({"type": "state-access", "message": msg})
                        break
                # Check for STATE method calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if (
                            isinstance(node.func.value, ast.Name)
                            and node.func.value.id == "STATE"
                        ):
                            msg = "Code illegally modifies STATE directly. Use tools instead of manipulating STATE."
                            failure_types.append("state-access")
                            failure_details.append(
                                {"type": "state-access", "message": msg}
                            )
                            break
                    # Check for get_url() calls
                    if isinstance(node.func, ast.Name) and node.func.id == "get_url":
                        msg = "Code illegally calls get_url(). Initial URL is known via assertion in prompt."
                        failure_types.append("state-access")
                        failure_details.append({"type": "state-access", "message": msg})
                        break
        except Exception as e:
            logger.debug(f"STATE access checking exception ignored: {e}")

    # 4. Ordering (only if syntax passed and no state-access violations)
    state_access_ok = "state-access" not in failure_types
    if syntax_ok and state_access_ok:
        try:
            from blastai.agents.codecheck import CFGBuilder, check_tool_ordering

            tree = ast.parse(code)
            builder = CFGBuilder()
            start_block, blocks = builder.build(tree)
            tools_by_name = {}
            for tool in getattr(agent, "tools", []):
                info = {
                    "pre": getattr(tool, "pre", {}),
                    "post": getattr(tool, "post", {}),
                    "param_names": [],
                    "param_patterns": {},
                    "pre_tools": getattr(tool, "pre_tools", {}),
                }
                if hasattr(tool, "input_schema") and tool.input_schema:
                    props = tool.input_schema.get("properties", {})
                    info["param_names"] = list(props.keys())
                    for p_name, p_schema in props.items():
                        if isinstance(p_schema, dict) and "pattern" in p_schema:
                            info["param_patterns"][p_name] = p_schema["pattern"]
                tools_by_name[tool.name] = info
            # Pass initial_state for proper precondition checking
            # TODO: Remove for production - use {} for unknown initial state
            initial_state_for_validation = {"page_type": "home"}
            valid_ordering, ordering_err = check_tool_ordering(
                blocks, start_block, initial_state_for_validation, tools_by_name
            )
            if not valid_ordering:
                ordering_ok = False
                otype = classify_failure(ordering_err)
                if otype == "unknown":
                    otype = "ordering"
                failure_types.append(otype)
                failure_details.append({"type": otype, "message": ordering_err})
        except Exception as e:
            logger.debug(f"Ordering validation exception ignored: {e}")
    overall_pass = syntax_ok and types_ok and state_access_ok and ordering_ok
    return CodecheckResult(
        overall_pass=overall_pass,
        failure_types=failure_types,
        failure_details=failure_details,
        error_message=None
        if overall_pass
        else (
            failure_details[0]["message"] if failure_details else "Validation failed"
        ),
    )


def classify_failure(msg: Optional[str]) -> str:
    if not msg:
        return "unknown"
    l = msg.lower()
    if "no code block" in l:
        return "extraction"
    if "syntax" in l or "invalid syntax" in l:
        return "syntax"
    if "line " in l and ":" in msg:
        return "types"
    if "pre-tools" in l:
        return "pre-tools"
    if "precondition" in l or "tool ordering" in l or "ordering" in l:
        return "ordering"
    if "state access" in l or "get_url" in l:
        return "state-access"
    return "unknown"


async def execute_code_with_timing(
    code: str, agent: Agent, task: str, initial_url: str, user_id: str = None
) -> float:
    """
    Execute generated code and measure actual latency.

    Returns execution time in seconds.
    """
    # Normalize environment for browser-use to match CLI behavior and reduce flakiness
    os.environ.setdefault("BROWSER_USE_LOGGING_LEVEL", "error")
    os.environ.setdefault("BROWSER_USE_DISABLE_LOGGING", "true")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    # Ensure GPU disabled consistently
    os.environ.setdefault("BROWSER_DISABLE_GPU", "true")
    # Reasonable startup timeout for browser
    os.environ.setdefault("BROWSER_START_TIMEOUT", "20")

    # Create executor for running the code (prevent fallback to loop mode on codegen failure)
    executor = AgentExecutor(
        agent=agent,
        user_id=user_id,
        timezone="America/Los_Angeles",
        stop_if_codegen_fails=True,
    )

    try:
        # Measure execution time
        start_time = time.time()

        # Run in code mode (this will execute the generated code; retries occur within code mode only)
        await executor.run(task, mode="code", initial_url=initial_url)

        execution_time = time.time() - start_time

        return execution_time
    except Exception as e:
        logger.error(f"Error during code execution: {e}")
        return -1.0  # Indicate failure
    finally:
        await executor.cleanup()


async def measure_page_load_latency(
    initial_url: str, headful: bool = True, agent: Optional[Agent] = None
) -> float:
    """
    Measure time to load a page until fully loaded.

    Uses the same approach as blastai code mode - calls observe tool to wait for page readiness.

    Args:
        initial_url: URL to load
        headful: If True, show browser window (default: True)

    Returns load time in seconds.
    """
    from pathlib import Path

    from browser_use import BrowserProfile, BrowserSession

    from blastai.agents.coderun import find_and_call_observe

    # Get viewport/window size from environment (matching AgentExecutor)
    width = int(os.getenv("BROWSER_WIDTH", "1280"))
    height = int(os.getenv("BROWSER_HEIGHT", "720"))

    # Build Chrome args for WSL GPU fix (matching AgentExecutor)
    # ALWAYS disable GPU for WSL to avoid transparency/black bar issues
    args = ["--disable-gpu", "--disable-gpu-sandbox"]

    # Create profile with proper viewport and window settings (matching AgentExecutor)
    profile = BrowserProfile(
        headless=not headful,  # headless=False when headful=True
        viewport={"width": width, "height": height},
        window_size={"width": width, "height": height},
        args=args,
        keep_alive=False,  # Don't keep browser alive after we're done
        wait_for_network_idle_page_load_time=2.0,  # Wait 2 seconds for network idle
    )

    # Create browser session
    browser = BrowserSession(browser_profile=profile)

    try:
        # Start timing before launching browser (measure from process start)
        start_time = time.time()

        # Start browser (this launches Chrome/Chromium with all proper args)
        await browser.start()

        # Get current page and navigate
        page = await browser.get_current_page()
        await page.goto(initial_url)

        # If an agent with SMCP observe tools is provided, call the observe tool
        # to wait for page readiness (this is the canonical blastai approach).
        if agent is not None:
            # Check if any observe tools exist
            try:
                has_observe = any(
                    getattr(t, "tool_executor_type", None) == ToolExecutorType.SMCP
                    and getattr(t, "type", None) == SMCPToolType.OBSERVE
                    for t in getattr(agent, "tools", [])
                )
            except Exception:
                has_observe = False

            if has_observe:
                # Create an AgentExecutor that reuses our BrowserSession so SMCP tools run
                agent_executor = AgentExecutor(agent=agent, browser=browser)

                # Create the python executor which initializes STATE used by observe
                from blastai.agents.coderun import (
                    create_python_executor,
                    find_and_call_observe,
                )

                try:
                    python_executor = create_python_executor(
                        agent, browser, agent_executor.llm, agent_executor
                    )
                    STATE = python_executor.state.get("STATE", {})
                    # Call observe for requested URL (matches pre_path patterns)
                    await find_and_call_observe(
                        initial_url, agent.tools, STATE, agent_executor, " (page load)"
                    )
                except Exception as e:
                    logger.debug(f"Observe tool call failed or not available: {e}")
        else:
            has_observe = False

        # Fallback: wait for profile-configured minimal + network idle time if no observe ran
        if not (agent is not None and has_observe):
            try:
                wait_time = (profile.minimum_wait_page_load_time or 0.25) + (
                    profile.wait_for_network_idle_page_load_time or 0.5
                )
                await asyncio.sleep(wait_time)
            except Exception:
                # Last-resort small sleep
                await asyncio.sleep(0.5)

        # Page has loaded successfully (or we waited heuristically)
        load_time = time.time() - start_time

        logger.info(f"Page loaded in {load_time:.2f}s")
        return load_time

    except Exception as e:
        logger.error(f"Error loading page {initial_url}: {e}")
        import traceback

        traceback.print_exc()
        return -1.0
    finally:
        # Kill browser session properly - this dispatches BrowserStopEvent and cleans up EventBus
        try:
            logger.debug("Killing browser session...")
            await browser.kill()
            logger.debug("Browser session killed successfully")
        except Exception as e:
            logger.warning(f"Error killing browser: {e}")


async def run_evaluation_for_task(
    task_id: str,
    task_def: Dict[str, Any],
    configs: List[CodegenConfig],
    num_runs: int,
    measure_actual_latency: bool,
    measure_page_load: bool,
    results_dir: Path,
    parallel: int = 1,
    print_code: bool = False,
    print_prompt: bool = False,
) -> Tuple[List[CodegenResult], List[PageLoadResult]]:
    """
    Run evaluation for a single task across all configs.

    Args:
        parallel: Number of runs to execute in parallel (default: 1)
        print_code: Whether to print generated code for each run (default: False)
        print_prompt: Whether to print codegen prompt for first run (default: False)

    Returns:
        Tuple of (codegen_results, page_load_results)
    """
    console.print(f"\n[blue]Evaluating task:[/] {task_id}")
    console.print(f"[dim]Goal:[/] {task_def.get('goal', 'N/A')}")
    console.print(f"[dim]URL:[/] {task_def.get('initial_url', 'N/A')}")

    # Load agent for this task
    agent = await create_agent_for_task(task_def)

    task_goal = task_def.get("goal", "")
    initial_url = task_def.get("initial_url", "")
    user_id = task_def.get("user_id", f"eval-{task_id}")

    codegen_results: List[CodegenResult] = []
    page_load_results: List[PageLoadResult] = []

    # Measure page load latency (8 times) - optional
    if measure_page_load:
        console.print(
            f"[yellow]Measuring page load latency (8 runs with headful browser)...[/]"
        )
        for i in range(8):
            console.print(f"  Page load run {i + 1}/8...")
            load_latency = await measure_page_load_latency(
                initial_url, headful=True, agent=agent
            )
            page_load_results.append(
                PageLoadResult(initial_url=initial_url, load_latency=load_latency)
            )
            console.print(f"    Load time: {load_latency:.2f}s")
    else:
        console.print(f"[dim]Skipping page load latency measurement[/]")

    # Run codegen evaluation for each config
    for config in configs:
        console.print(
            f"\n[cyan]Config:[/] model={config.model}, with_protocol={config.with_protocol}"
        )

        # Track best and worst cost for actual latency measurement
        config_results = []

        # Create a semaphore to limit parallelism
        semaphore = asyncio.Semaphore(parallel)

        async def run_single_evaluation(run_num: int):
            """Run a single evaluation with semaphore for concurrency control."""
            async with semaphore:
                try:
                    # Create LLM for this config
                    codegen_llm = LLMFactory.create_llm(
                        model_name=config.model, temperature=0.5
                    )

                    # Generate code
                    # Print prompt only for first run (run_num == 0) and first config
                    debug_prompt = print_prompt and run_num == 0
                    (
                        generated_code,
                        gen_time,
                        _,
                        candidate,
                        code_generator,
                    ) = await generate_code_with_timing(
                        agent=agent,
                        task=task_goal,
                        config=config,
                        codegen_llm=codegen_llm,
                        debug_print_prompt=debug_prompt,
                        task_def=task_def,
                    )

                    # Handle failed generation
                    if not generated_code:
                        # Create failed CodegenResult so summary stats include this failure
                        failed_codecheck = await validate_code(
                            code="",  # triggers failure path
                            agent=agent,
                            code_generator=None,
                            candidate=None,
                        )
                        result = CodegenResult(
                            config=config,
                            generation_latency=gen_time,
                            estimated_cost=0.0,
                            codecheck=failed_codecheck,
                            generated_code="",
                            generation_failed=True,
                        )
                        console.print(
                            f"  Run {run_num + 1}/{num_runs}... ✗ (generation failed)"
                        )
                        return result

                    # Strip tool redefinitions before cost estimation
                    # LLMs sometimes redefine tools with their own ai_exec implementations
                    from blastai.agents.codefix import strip_tool_redefinitions

                    tool_names = {tool.name for tool in agent.tools}
                    code_for_analysis = generated_code
                    if tool_names:
                        code_for_analysis, _ = strip_tool_redefinitions(
                            generated_code, tool_names
                        )

                    # Estimate cost using codecost module (without stripping user-defined functions)
                    # The CFG builder properly handles user-defined functions via two-pass analysis
                    try:
                        estimated_cost = compute_code_cost(
                            code_for_analysis, agent.tools
                        )
                    except Exception as e:
                        logger.warning(f"Failed to compute code cost: {e}")
                        estimated_cost = 0.0

                    # Validate code
                    codecheck_result = await validate_code(
                        code=generated_code,
                        agent=agent,
                        code_generator=code_generator,
                        candidate=candidate,
                    )

                    # Create result
                    result = CodegenResult(
                        config=config,
                        generation_latency=gen_time,
                        estimated_cost=estimated_cost,
                        codecheck=codecheck_result,
                        generated_code=generated_code,
                    )

                    status = "✓" if codecheck_result.overall_pass else "✗"
                    console.print(
                        f"  Run {run_num + 1}/{num_runs}... {status} (gen: {gen_time:.2f}s, est_cost: {estimated_cost:.2f}s)"
                    )

                    # Print generated code if requested
                    if print_code:
                        console.print(
                            f"\n[dim]Generated code for run {run_num + 1}:[/]"
                        )
                        console.print(
                            Panel(
                                generated_code,
                                title=f"Run {run_num + 1} - {config.model}",
                                border_style="cyan",
                            )
                        )

                    return result

                except Exception as e:
                    console.print(f"  Run {run_num + 1}/{num_runs}... ✗ (error: {e})")
                    logger.error(f"Error in run {run_num + 1}: {e}")
                    import traceback

                    traceback.print_exc()
                    return None

        # Run all evaluations for this config (potentially in parallel)
        tasks_to_run = [run_single_evaluation(run_num) for run_num in range(num_runs)]
        results = await asyncio.gather(*tasks_to_run)

        # Filter out None results (failed runs)
        config_results = [r for r in results if r is not None]
        codegen_results.extend(config_results)

        # Measure actual latency for best and worst cost only if >1 successful runs
        if measure_actual_latency and config_results:
            successful_only = [
                r
                for r in config_results
                if not r.generation_failed and r.generated_code
            ]
            if len(successful_only) > 1:
                # Sort successes by estimated cost
                sorted_by_cost = sorted(successful_only, key=lambda r: r.estimated_cost)
                lowest_cost = sorted_by_cost[0]
                highest_cost = sorted_by_cost[-1]
                # Skip if they refer to the same run (shouldn't happen with >1 success, but guard anyway)
                if lowest_cost is highest_cost:
                    console.print(
                        "  [dim]Skipping actual latency (only one unique successful run)"
                    )
                else:
                    console.print(
                        f"  [yellow]Measuring actual latency for lowest cost...[/]"
                    )
                    lowest_cost.actual_latency = await execute_code_with_timing(
                        code=lowest_cost.generated_code,
                        agent=agent,
                        task=task_goal,
                        initial_url=initial_url,
                        user_id=user_id,
                    )
                    console.print(
                        f"  [yellow]Measuring actual latency for highest cost...[/]"
                    )
                    highest_cost.actual_latency = await execute_code_with_timing(
                        code=highest_cost.generated_code,
                        agent=agent,
                        task=task_goal,
                        initial_url=initial_url,
                        user_id=user_id,
                    )
            else:
                console.print(
                    "  [dim]Skipping actual latency (need >1 successful runs for config)"
                )

    return codegen_results, page_load_results


def save_results_to_files(
    task_id: str,
    codegen_results: List[CodegenResult],
    page_load_results: List[PageLoadResult],
    results_dir: Path,
    suffix: str = "",
):
    """
    Save results to JSON and Markdown files, grouped by model.

    Files are named {task_id}_{model}.json/.md
    If a file already exists, a timestamp suffix is added.

    Args:
        suffix: Deprecated - timestamp is only added when file exists
    """
    from collections import defaultdict
    from datetime import datetime

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # Group results by model
    results_by_model: Dict[str, List[CodegenResult]] = defaultdict(list)
    for result in codegen_results:
        model_safe = result.config.model.replace("/", "-")
        results_by_model[model_safe].append(result)

    saved_files = []

    # Save each model's results to separate files
    for model_safe, model_results in results_by_model.items():
        # Build base filename with model name
        base_name = f"{task_id}_{model_safe}"

        # Check if file exists, add timestamp if so
        json_file = results_dir / f"{base_name}.json"
        md_file = results_dir / f"{base_name}.md"

        if json_file.exists() or md_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{task_id}_{model_safe}_{timestamp}"
            json_file = results_dir / f"{base_name}.json"
            md_file = results_dir / f"{base_name}.md"

        # Save JSON file
        all_results = []
        for result in model_results:
            all_results.append(result.to_dict())

        # Add page load results (only to first model's file to avoid duplication)
        if model_safe == list(results_by_model.keys())[0]:
            for result in page_load_results:
                all_results.append(result.to_dict())

        with open(json_file, "w") as f:
            json.dump(all_results, f, indent=2)

        # Save Markdown file with generated code
        with open(md_file, "w") as f:
            f.write(f"# Generated Code for Task: {task_id}\n\n")

            for i, result in enumerate(model_results):
                f.write(f"## Run {i + 1}\n\n")
                f.write(f"**Config:**\n")
                f.write(f"- Model: `{result.config.model}`\n")
                f.write(f"- With Protocol: `{result.config.with_protocol}`\n\n")
                f.write(f"**Metrics:**\n")
                f.write(f"- Generation Latency: {result.generation_latency:.3f}s\n")
                f.write(f"- Estimated Cost: {result.estimated_cost:.2f}s\n")
                f.write(
                    f"- Overall Pass: {'✓' if result.codecheck.overall_pass else '✗'}\n"
                )
                if result.codecheck.failure_types:
                    f.write(
                        f"- Failure Types: {', '.join(result.codecheck.failure_types)}\n"
                    )
                if result.actual_latency is not None:
                    f.write(f"- Actual Latency: {result.actual_latency:.2f}s\n")
                f.write(f"\n**Generated Code:**\n\n")
                if result.generated_code.strip():
                    f.write(f"```python\n{result.generated_code}\n```\n\n")
                else:
                    # Provide a readable placeholder for empty code results
                    placeholder = (
                        "# (no code generated or code block missing in model output)"
                    )
                    f.write(f"```python\n{placeholder}\n```\n\n")
                f.write("---\n\n")

        saved_files.append((json_file, md_file))

    console.print(f"[green]Saved results to:[/]")
    for json_file, md_file in saved_files:
        console.print(f"  - {json_file}")
        console.print(f"  - {md_file}")


def compute_summary_statistics(
    all_results: Dict[str, Tuple[List[CodegenResult], List[PageLoadResult]]],
) -> Dict[str, Any]:
    """
    Compute averaged metrics for each unique config.

    Returns summary dict with averaged metrics per config.
    """
    import statistics
    from collections import defaultdict

    # Group results by config
    config_groups = defaultdict(list)

    for task_id, (codegen_results, page_load_results) in all_results.items():
        for result in codegen_results:
            config_key = (result.config.model, result.config.with_protocol)
            config_groups[config_key].append(result)

    # Compute averages
    summary = {}

    for config_key, results in config_groups.items():
        model, with_protocol = config_key

        # Calculate averages
        avg_gen_latency = statistics.mean(r.generation_latency for r in results)
        # generation_cost removed - use estimated_cost instead
        avg_est_cost = statistics.mean(r.estimated_cost for r in results)

        # Variance metrics
        var_gen_latency = (
            statistics.variance(r.generation_latency for r in results)
            if len(results) > 1
            else 0
        )
        var_est_cost = (
            statistics.variance(r.estimated_cost for r in results)
            if len(results) > 1
            else 0
        )

        # Pass/failure rates (single attempt per run, multi failure types counted)
        num_runs_cfg = len(results)
        overall_pass_rate = (
            sum(1 for r in results if r.codecheck.overall_pass) / num_runs_cfg
        )
        failure_counts: Dict[str, int] = {}
        for r in results:
            for ft in getattr(r.codecheck, "failure_types", []) or []:
                failure_counts[ft] = failure_counts.get(ft, 0) + 1
        failure_rates = {k: v / num_runs_cfg for k, v in failure_counts.items()}
        num_failed_runs = sum(1 for r in results if not r.codecheck.overall_pass)
        failure_rates_among_failures = {
            k: (v / num_failed_runs) if num_failed_runs else 0.0
            for k, v in failure_counts.items()
        }

        # Metrics among passing candidates only
        passing_results = [r for r in results if r.codecheck.overall_pass]
        avg_est_cost_passing = (
            statistics.mean(r.estimated_cost for r in passing_results)
            if passing_results
            else None
        )
        min_gen_latency_passing = min(
            (r.generation_latency for r in passing_results), default=None
        )
        min_est_cost_passing = min(
            (r.estimated_cost for r in passing_results), default=None
        )
        max_est_cost_passing = max(
            (r.estimated_cost for r in passing_results), default=None
        )

        # Actual latency (only for runs that measured it)
        actual_latencies = [
            r.actual_latency for r in results if r.actual_latency is not None
        ]
        avg_actual_latency = (
            statistics.mean(actual_latencies) if actual_latencies else None
        )

        generation_failed_count = sum(
            1
            for r in results
            if getattr(r, "generation_failed", False)
            or ("generation" in getattr(r.codecheck, "failure_types", []))
        )
        # Average fractional pass metrics if present
        overall_pass_rate_fraction = statistics.mean(
            getattr(r.codecheck, "overall_pass_fraction", 0.0) for r in results
        )
        summary[f"{model}|with_protocol={with_protocol}"] = {
            "model": model,
            "with_protocol": with_protocol,
            "num_runs": len(results),
            "num_generation_failures": generation_failed_count,
            "avg_generation_latency": avg_gen_latency,
            "var_generation_latency": var_gen_latency,
            "avg_estimated_cost": avg_est_cost,
            "var_estimated_cost": var_est_cost,
            "overall_pass_rate": overall_pass_rate,
            "failure_rates": failure_rates,
            "failure_rates_among_failures": failure_rates_among_failures,
            "avg_actual_latency": avg_actual_latency,
            # Metrics among passing candidates only
            "num_passing": len(passing_results),
            "avg_est_cost_passing": avg_est_cost_passing,
            "min_gen_latency_passing": min_gen_latency_passing,
            "min_est_cost_passing": min_est_cost_passing,
            "max_est_cost_passing": max_est_cost_passing,
        }

    # Add page load statistics
    all_page_loads = []
    for task_id, (codegen_results, page_load_results) in all_results.items():
        all_page_loads.extend(page_load_results)

    if all_page_loads:
        avg_page_load = statistics.mean(
            r.load_latency for r in all_page_loads if r.load_latency > 0
        )
        var_page_load = (
            statistics.variance(
                r.load_latency for r in all_page_loads if r.load_latency > 0
            )
            if len(all_page_loads) > 1
            else 0
        )

        summary["page_load"] = {
            "avg_load_latency": avg_page_load,
            "var_load_latency": var_page_load,
            "num_measurements": len(all_page_loads),
        }

    return summary


@click.command()
@click.option(
    "--tasks",
    type=click.Path(exists=True),
    required=True,
    help="Path to tasks YAML file (e.g., experiments/tasks/agisdk/agisdk.yaml)",
)
@click.option(
    "--ids",
    type=str,
    required=True,
    help='Space-separated task IDs to evaluate (e.g., "dashdish-deepresearch1 gomail-3")',
)
@click.option(
    "--results-dir",
    type=click.Path(),
    default="experiments/results",
    help="Directory to save results (default: experiments/results)",
)
@click.option(
    "--models",
    type=str,
    default=None,
    help='Space-separated list of models to test (e.g., "gpt-5.1 gemini-2.0-flash-lite"). If not provided, tests all default models.',
)
@click.option(
    "--actual-latency/--no-actual-latency",
    default=False,
    help="Measure actual execution latency (for highest/lowest cost)",
)
@click.option(
    "--page-load/--no-page-load",
    default=False,
    help="Measure page load latency (default: False)",
)
@click.option(
    "--runs", type=int, default=32, help="Number of runs per config (default: 32)"
)
@click.option(
    "--parallel",
    type=int,
    default=1,
    help="Number of parallel runs to execute simultaneously (default: 1, max: 32)",
)
@click.option(
    "--print-code/--no-print-code",
    default=False,
    help="Print generated code for each run (default: False)",
)
@click.option(
    "--print-prompt/--no-print-prompt",
    default=False,
    help="Print codegen prompt for first run (default: False)",
)
@click.option(
    "--max-iterations",
    type=int,
    default=1,
    help="Maximum retry iterations for code generation (default: 1, no retries)",
)
def main(
    tasks: str,
    ids: str,
    results_dir: str,
    models: Optional[str],
    actual_latency: bool,
    page_load: bool,
    runs: int,
    parallel: int,
    print_code: bool,
    print_prompt: bool,
    max_iterations: int,
):
    """
    Evaluate code generation performance across different configurations.

    Example:
        python experiments/evaluate_codegen.py \\
            --tasks experiments/tasks/agisdk/agisdk.yaml \\
            --ids "dashdish-deepresearch1 gomail-3" \\
            --results-dir experiments/results \\
            --actual-latency \\
            --page-load
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress browser-use logging
    logging.getLogger("browser_use").setLevel(logging.ERROR)

    # Parse task IDs
    task_ids = ids.split()

    # Define configs to test
    # If models are specified, use only those models
    # Otherwise, use the default set
    if models:
        model_list = models.split()
        configs = []
        for model in model_list:
            configs.append(
                CodegenConfig(
                    model=model, with_protocol=True, max_iterations=max_iterations
                )
            )
            configs.append(
                CodegenConfig(
                    model=model, with_protocol=False, max_iterations=max_iterations
                )
            )
    else:
        # Default configs - Testing various OpenAI models with/without protocol
        # gpt-5.1: $1.25 input, $10.00 output
        # gpt-5-mini: $0.25 input, $2.00 output
        # openai/gpt-oss-120b: Open-source on Groq ($0.30/M tokens)
        # gemini-2.5-pro: Google's best model
        # gemini-2.0-flash-lite: Google's fast/cheap model
        configs = [
            CodegenConfig(
                model="gpt-5.1", with_protocol=True, max_iterations=max_iterations
            ),
            CodegenConfig(
                model="gpt-5.1", with_protocol=False, max_iterations=max_iterations
            ),
            CodegenConfig(
                model="gpt-5-mini", with_protocol=True, max_iterations=max_iterations
            ),
            CodegenConfig(
                model="gpt-5-mini", with_protocol=False, max_iterations=max_iterations
            ),
            CodegenConfig(
                model="openai/gpt-oss-120b",
                with_protocol=True,
                max_iterations=max_iterations,
            ),
            CodegenConfig(
                model="openai/gpt-oss-120b",
                with_protocol=False,
                max_iterations=max_iterations,
            ),
            CodegenConfig(
                model="gemini-2.5-pro",
                with_protocol=True,
                max_iterations=max_iterations,
            ),
            CodegenConfig(
                model="gemini-2.5-pro",
                with_protocol=False,
                max_iterations=max_iterations,
            ),
            CodegenConfig(
                model="gemini-2.0-flash-lite",
                with_protocol=True,
                max_iterations=max_iterations,
            ),
            CodegenConfig(
                model="gemini-2.0-flash-lite",
                with_protocol=False,
                max_iterations=max_iterations,
            ),
        ]

    # Clamp parallel to reasonable range
    parallel = max(1, min(parallel, 32))

    console.print(
        Panel(
            f"[bold]Code Generation Evaluation[/]\n\n"
            f"Tasks file: {tasks}\n"
            f"Task IDs: {', '.join(task_ids)}\n"
            f"Configs: {len(configs)}\n"
            f"Runs per config: {runs}\n"
            f"Parallel runs: {parallel}\n"
            f"Max iterations: {max_iterations}\n"
            f"Measure actual latency: {actual_latency}\n"
            f"Measure page load: {page_load}",
            title="Configuration",
            border_style="blue",
        )
    )

    async def run_all_evaluations():
        from datetime import datetime

        # Load tasks
        tasks_file = Path(tasks)
        all_tasks = await load_task_definitions(tasks_file)

        # Filter to requested IDs
        tasks_dict = {task["id"]: task for task in all_tasks if task["id"] in task_ids}

        if len(tasks_dict) != len(task_ids):
            found_ids = set(tasks_dict.keys())
            missing_ids = set(task_ids) - found_ids
            console.print(f"[red]Warning: Missing task IDs: {missing_ids}[/]")

        # Results directory
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # Store all results for summary
        all_results = {}

        # Evaluate each task
        for task_id in task_ids:
            if task_id not in tasks_dict:
                console.print(f"[red]Skipping missing task: {task_id}[/]")
                continue

            task_def = tasks_dict[task_id]

            codegen_results, page_load_results = await run_evaluation_for_task(
                task_id=task_id,
                task_def=task_def,
                configs=configs,
                num_runs=runs,
                measure_actual_latency=actual_latency,
                measure_page_load=page_load,
                results_dir=results_path,
                parallel=parallel,
                print_code=print_code,
                print_prompt=print_prompt,
            )

            # Save results (per-model files, timestamp added if file exists)
            save_results_to_files(
                task_id,
                codegen_results,
                page_load_results,
                results_path,
            )

            # Store for summary
            all_results[task_id] = (codegen_results, page_load_results)

        # Compute and save summary statistics
        console.print("\n[blue]Computing summary statistics...[/]")
        summary = compute_summary_statistics(all_results)

        # Save summary to results directory (add timestamp if file exists)
        summary_file = results_path / "summary.json"
        if summary_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = results_path / f"summary_{timestamp}.json"

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        console.print(f"\n[green]✓ Summary saved to:[/] {summary_file}")

        # Print summary table
        console.print("\n[bold]Summary Results:[/]\n")
        for config_name, metrics in summary.items():
            if config_name == "page_load":
                continue
            console.print(f"[cyan]{config_name}[/]")
            console.print(
                f"  Avg Generation Latency: {metrics['avg_generation_latency']:.3f}s (var: {metrics['var_generation_latency']:.6f})"
            )
            console.print(
                f"  Avg Estimated Cost: {metrics['avg_estimated_cost']:.2f}s (var: {metrics['var_estimated_cost']:.6f})"
            )
            console.print(f"  Overall Pass Rate: {metrics['overall_pass_rate']:.1%}")
            # Show metrics among passing candidates
            num_passing = metrics.get("num_passing", 0)
            if num_passing > 0:
                console.print(f"  [green]Among {num_passing} Passing:[/]")
                if metrics.get("avg_est_cost_passing") is not None:
                    console.print(
                        f"    Avg Estimated Cost: {metrics['avg_est_cost_passing']:.2f}s"
                    )
                if metrics.get("min_gen_latency_passing") is not None:
                    console.print(
                        f"    Min Generation Latency: {metrics['min_gen_latency_passing']:.3f}s"
                    )
                if (
                    metrics.get("min_est_cost_passing") is not None
                    and metrics.get("max_est_cost_passing") is not None
                ):
                    console.print(
                        f"    Est Cost Range: {metrics['min_est_cost_passing']:.2f}s - {metrics['max_est_cost_passing']:.2f}s"
                    )
            if "failure_rates" in metrics and metrics["failure_rates"]:
                fr_str = ", ".join(
                    f"{k}:{v:.1%}" for k, v in metrics["failure_rates"].items()
                )
                console.print(f"  Failure Rates (per run): {fr_str}")
            if (
                "failure_rates_among_failures" in metrics
                and metrics["failure_rates_among_failures"]
            ):
                fr2_str = ", ".join(
                    f"{k}:{v:.1%}"
                    for k, v in metrics["failure_rates_among_failures"].items()
                )
                console.print(f"  Failure Breakdown (among failures): {fr2_str}")
            if (
                "num_generation_failures" in metrics
                and metrics.get("num_generation_failures", 0) > 0
            ):
                total = metrics.get("num_runs", 0)
                fails = metrics["num_generation_failures"]
                succ_rate = (total - fails) / total if total else 0.0
                console.print(
                    f"  Generation Success Rate: {succ_rate:.1%} ({total - fails}/{total})"
                )
            if metrics["avg_actual_latency"] is not None:
                console.print(
                    f"  Avg Actual Latency: {metrics['avg_actual_latency']:.2f}s"
                )
            console.print()

        if "page_load" in summary:
            pl = summary["page_load"]
            console.print(f"[cyan]Page Load Statistics[/]")
            console.print(
                f"  Avg Load Latency: {pl['avg_load_latency']:.2f}s (var: {pl['var_load_latency']:.6f})"
            )
            console.print(f"  Measurements: {pl['num_measurements']}")

    # Run async evaluation with explicit cleanup
    async def run_with_cleanup():
        try:
            await run_all_evaluations()
        finally:
            # Cancel any pending tasks to prevent hanging
            pending = [
                t for t in asyncio.all_tasks() if t is not asyncio.current_task()
            ]
            if pending:
                for task in pending:
                    task.cancel()
                # Wait for cancellations with timeout
                await asyncio.wait(pending, timeout=5.0)

    asyncio.run(run_with_cleanup())

    console.print("\n[green]✓ Evaluation complete![/]\n")


if __name__ == "__main__":
    main()
