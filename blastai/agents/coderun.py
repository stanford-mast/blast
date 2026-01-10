"""
Python code executor creation with tool wrapping and STATE management.

Provides function to create LocalPythonExecutor instances with wrapped SMCP tools,
precondition/postcondition checking, and automatic observe tool calls.
"""

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, create_model

from .execution_hooks import ExecutionHooks, StopExecutionError
from .local_python_executor import LocalPythonExecutor
from .models import Agent, SMCPTool, SMCPToolType, ToolExecutorType
from .schema_utils import json_schema_to_pydantic
from .tools_smcp import execute_smcp_tool

if TYPE_CHECKING:
    from browser_use.browser import BrowserSession
    from browser_use.llm.base import BaseChatModel
    from browser_use.llm.messages import UserMessage

logger = logging.getLogger(__name__)


async def find_and_call_observe(
    url: str, tools: List, state: Dict[str, Any], agent_executor, context: str = ""
) -> Optional[Dict[str, Any]]:
    """
    Find an observe tool whose pre_path matches the given URL and call it.
    Updates STATE with the result if successful.

    Args:
        url: The URL to match against observe tool pre_path patterns
        tools: List of all tools (will be filtered to SMCP observe tools)
        state: STATE dict to update with observe results
        agent_executor: AgentExecutor instance for executing the tool
        context: Description of when/why this is being called (for logging)

    Returns:
        The observe result dict if successful, None otherwise
    """
    if not tools or not url:
        return None

    # Filter to SMCP observe tools
    observe_tools = [
        t
        for t in tools
        if t.tool_executor_type == ToolExecutorType.SMCP
        and hasattr(t, "type")
        and t.type == SMCPToolType.OBSERVE
    ]

    if not observe_tools:
        return None

    for obs_tool in observe_tools:
        obs_pattern = getattr(obs_tool, "pre_path", None)
        if obs_pattern:
            obs_regex = obs_pattern.replace("*", ".*")
            if re.match(obs_regex, url):
                try:
                    observe_result = await execute_smcp_tool(
                        agent_executor, obs_tool, {}
                    )
                    if isinstance(observe_result, dict):
                        state.update(observe_result)
                        logger.debug(
                            f"Updated STATE from {obs_tool.name}{context}: {observe_result}"
                        )
                        return observe_result
                except Exception as e:
                    logger.warning(f"Failed to call {obs_tool.name}{context}: {e}")
                break
    return None


def create_python_executor(
    agent: Agent,
    browser: "BrowserSession",
    llm: "BaseChatModel",
    agent_executor,
    no_state_checking: bool = False,
) -> LocalPythonExecutor:
    """
    Create Python executor for code mode with wrapped tools and STATE management.

    This function:
    - Creates LocalPythonExecutor with basic imports
    - Initializes STATE dictionary from tool contracts (excluding current_url)
    - Creates Pydantic models for tool input validation
    - Wraps SMCP tools with precondition/postcondition checking
    - Registers observe tools that update STATE with page state
    - Provides utility functions (get_url, goto, ai_exec, ai_eval)

    Args:
        agent: Agent with tools to wrap
        browser: BrowserSession for accessing current page/URL
        llm: LLM for ai_eval utility
        agent_executor: AgentExecutor instance (for creating sub-executors in ai_exec)
        no_state_checking: If True, skip automatic observe calls in tool wrappers.
                          Observe is still called in goto() and ai_exec() regardless.
                          Default False for backward compatibility.

    Returns:
        LocalPythonExecutor with all tools and utilities registered
    """
    # Create execution hooks for code mode (handles stop checking and AgentThought messages)
    hooks = ExecutionHooks(
        agent_executor=agent_executor,
        session_id=agent_executor.session_id,
        cycle_id=agent_executor.cycle_id,
    )
    tool_hook = hooks.create_code_mode_decorator()

    # Create executor first so we can register wrapped tool functions that
    # assert preconditions against STATE and update STATE with postconditions.
    import_modules = {
        "json": __import__("json"),
        "re": __import__("re"),
        "asyncio": __import__("asyncio"),
        "math": __import__("math"),
        "statistics": __import__("statistics"),
        "typing": __import__("typing"),
    }

    # Add comprehensive typing imports for code execution
    from typing import (
        Any,
        Callable,
        Dict,
        Generic,
        Iterable,
        List,
        Literal,
        Mapping,
        Optional,
        Sequence,
        Set,
        Tuple,
        TypeVar,
        Union,
    )

    import_modules.update(
        {
            "Optional": Optional,
            "List": List,
            "Dict": Dict,
            "Tuple": Tuple,
            "Set": Set,
            "Union": Union,
            "Any": Any,
            "Callable": Callable,
            "Iterable": Iterable,
            "Sequence": Sequence,
            "Mapping": Mapping,
            "TypeVar": TypeVar,
            "Generic": Generic,
            "Literal": Literal,
        }
    )

    # Add pydantic imports if available
    try:
        from pydantic import BaseModel, Field, root_validator, validator

        import_modules.update(
            {
                "BaseModel": BaseModel,
                "Field": Field,
                "validator": validator,
                "root_validator": root_validator,
            }
        )
    except ImportError:
        pass

    executor = LocalPythonExecutor(
        additional_functions={}, additional_imports=import_modules
    )

    # Initialize STATE dictionary in executor.state with keys from tool contracts
    # Collect all state keys referenced in pre/post (but NOT current_url - use get_url() instead)
    state_keys = set()
    for t in agent.tools:
        if t.tool_executor_type == ToolExecutorType.SMCP:
            if getattr(t, "pre", None) and isinstance(t.pre, dict):
                state_keys.update(t.pre.keys())
            if getattr(t, "post", None) and isinstance(t.post, dict):
                state_keys.update(t.post.keys())

    # Initialize STATE as a dict with all keys set to None
    STATE = {key: None for key in sorted(state_keys)} if state_keys else {"page": None}
    executor.state["STATE"] = STATE

    # Helper for runtime pattern checks
    def _check_pattern(state_val, pattern, param_vals: Dict[str, Any]):
        # None: skip if None means no constraint
        if pattern is None:
            return True
        # Non-string exact match
        if not isinstance(pattern, str):
            return state_val == pattern
        if pattern == "*":
            return state_val is not None
        if pattern == "":
            return not state_val
        if "|" in pattern:
            return state_val in pattern.split("|")
        if pattern.startswith("$"):
            ref = pattern[1:]
            # Compare to provided param value if present
            return param_vals.get(ref) == state_val
        return state_val == pattern

    first_smcp_after_fallback = {"value": False}

    # Build wrappers for SMCP tools and register them into executor.state
    # Also create Pydantic models and register them for use in generated code
    wrapped_tools: Dict[str, Callable] = {}
    pydantic_models: Dict[str, type] = {}

    for t in agent.tools:
        if t.tool_executor_type != ToolExecutorType.SMCP:
            continue

        # Create Pydantic model for input validation if tool has parameters
        input_model = None
        param_types = {}  # For building typed signature
        if t.input_schema and t.input_schema.get("properties"):
            # Use shared helper to convert JSON schema to Pydantic model
            model_name = f"{t.name.title().replace('_', '')}Input"
            input_model = json_schema_to_pydantic(t.input_schema, model_name)

            # Also build param_types for wrapper signature
            properties = t.input_schema["properties"]
            required = t.input_schema.get("required", [])

            for param_name, param_schema in properties.items():
                param_type_str = param_schema.get("type", "string")
                is_required = param_name in required

                # Map JSON schema types to Python types
                type_map = {
                    "string": str,
                    "number": float,
                    "integer": int,
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                }
                python_type = type_map.get(param_type_str, str)
                param_types[param_name] = (python_type, is_required)

        # Create Pydantic model for output validation if tool has output schema
        output_model = None
        if t.output_schema and t.output_schema.get("properties"):
            # Use shared helper to convert JSON schema to Pydantic model
            output_model_name = f"{t.name.title().replace('_', '')}Output"
            output_model = json_schema_to_pydantic(t.output_schema, output_model_name)

        def make_tool_wrapper(
            tool: SMCPTool, input_pydantic_model, output_pydantic_model, param_type_info
        ):
            # Build typed signature for wrapper
            if param_type_info:
                # Create wrapper with typed parameters
                # We'll use exec to create a function with the right signature
                param_parts = []
                # Partition into required and optional while preserving original property order
                required_names = [
                    name for name, (_, req) in param_type_info.items() if req
                ]
                optional_names = [
                    name for name, (_, req) in param_type_info.items() if not req
                ]

                # Add required params first (in properties order)
                for param_name in required_names:
                    param_type = param_type_info[param_name][0]
                    type_name = param_type.__name__
                    param_parts.append(f"{param_name}: {type_name}")

                # Then add optional params (with defaults) in properties order
                for param_name in optional_names:
                    param_type = param_type_info[param_name][0]
                    type_name = param_type.__name__
                    param_parts.append(f"{param_name}: Optional[{type_name}] = None")

                sig_str = ", ".join(param_parts)

                # Build the wrapper function code
                wrapper_code = f"""
async def {tool.name}({sig_str}):
    '''Wrapper that validates inputs, calls observe, checks preconditions, executes tool, and updates STATE.'''
    # Build kwargs from parameters
    kwargs = {{{", ".join([f'"{p}": {p}' for p in param_type_info.keys()])}}}
    # Remove None values
    kwargs = {{k: v for k, v in kwargs.items() if v is not None}}
    return await _execute_{tool.name}(kwargs)
"""
            else:
                # No parameters
                wrapper_code = f"""
async def {tool.name}():
    '''Wrapper that calls observe, checks preconditions, executes tool, and updates STATE.'''
    return await _execute_{tool.name}({{}})
"""

            # Create the actual execution function
            @tool_hook
            async def execute_impl(kwargs):
                """Implementation that handles tool execution with STATE management:

                CRITICAL FLOW FOR HANDLING LOGIN/REDIRECT ISSUES:
                1. Call observe BEFORE execution (based on current URL) to update STATE
                2. Check preconditions against STATE (including URL pattern)
                3. Execute the tool
                4. For OBSERVE tools: Update STATE with result (no URL check needed - it's the observer!)
                5. For NON-OBSERVE tools: Call observe AFTER execution to refresh STATE

                FALLBACK TO AI_EXEC:
                If any step fails (observe not found, preconditions not met, execution error),
                we fall back to ai_exec with the tool's output_schema as structured output.
                This makes ai_exec a true universal fallback that can handle any situation.

                This ensures:
                - If initial URL redirects to login, observe sets STATE={page:"login"}
                - Codegen sees login state and generates ai_exec/ask_human to handle it
                - After any tool execution, STATE is refreshed with current page state
                - If observe pre_path doesn't match, precondition fails → ai_exec fallback
                - If tool execution fails for any reason → ai_exec fallback with output schema

                The key insight: We match observe tools to current URL for BEFORE/AFTER calls,
                but the tool's pre_path precondition ensures we're on the right page to BEGIN with.
                If precondition fails (wrong page), ai_exec fallback handles navigation/login.
                """
                # Access STATE from executor.state
                STATE = executor.state["STATE"]

                # Helper to create ai_exec fallback call
                async def fallback_to_ai_exec(
                    reason: str, error: Optional[Exception] = None
                ):
                    """
                    Fallback to ai_exec when tool cannot execute.
                    Uses the tool's output_schema as structured output for ai_exec.
                    """
                    # Build detailed error message
                    error_details = f"{reason}"
                    if error:
                        error_details += f": {str(error)}"

                    # Check if ai_exec fallback is disabled (for evaluation)
                    if agent_executor.disable_ai_exec_fallback:
                        error_msg = (
                            f"Tool {tool.name} failed to execute: {error_details}"
                        )
                        logger.error(f"{error_msg} (ai_exec fallback disabled)")
                        if error:
                            raise RuntimeError(error_msg) from error
                        else:
                            raise RuntimeError(error_msg)

                    logger.warning(
                        f"Tool {tool.name} cannot execute ({error_details}), falling back to ai_exec"
                    )

                    # Construct structured task for ai_exec
                    task_lines = [f"Task: {tool.description}"]
                    if kwargs:
                        task_lines.append("Parameters:")
                        for k, v in kwargs.items():
                            task_lines.append(f"- {k}: {v}")

                    # Add motivation if there was an error
                    if error and str(error).strip():
                        task_lines.append(
                            f"Motivation: We just attempted this task but ran into the following error: {str(error)}"
                        )

                    fallback_task = "\n".join(task_lines)

                    # Call ai_exec with output schema if available
                    if "ai_exec" in executor.state:
                        # Pass output_schema to ai_exec for structured output
                        return await executor.state["ai_exec"](
                            fallback_task,
                            output_schema=tool.output_schema
                            if hasattr(tool, "output_schema")
                            else None,
                        )
                    else:
                        # Fallback not available, raise detailed error
                        error_msg = (
                            f"Tool {tool.name} failed to execute: {error_details}"
                        )
                        if error:
                            raise RuntimeError(error_msg) from error
                        else:
                            raise RuntimeError(error_msg)

                # Validate input parameters with Pydantic if model exists
                if input_pydantic_model is not None:
                    try:
                        validated = input_pydantic_model(**kwargs)
                        kwargs = validated.model_dump(exclude_none=True)
                    except Exception as e:
                        # Parameter validation failed - fallback to ai_exec
                        return await fallback_to_ai_exec("invalid parameters", e)

                # Check if this is the first SMCP tool call after ai_exec/ask_human
                # If so, use fast timeout (1s) to fail quickly on precondition violations
                # This handles cases where login redirects to different page (e.g., directory instead of dashboard)
                use_fast_timeout = first_smcp_after_fallback["value"]
                if use_fast_timeout:
                    logger.info(
                        f"First SMCP tool {tool.name} after fallback - using fast timeout (1s)"
                    )
                    first_smcp_after_fallback["value"] = (
                        False  # Reset for subsequent tools
                    )

                # BEFORE execution: Call matching observe tool to update STATE
                # This is critical for detecting login pages, wrong pages, etc.
                # Skip if no_state_checking=True (observe already called at code mode start)
                observe_found = False
                if not no_state_checking and getattr(tool, "pre_path", None):
                    current_url = await executor.state[
                        "get_url"
                    ]()  # Use get_url() function
                    if current_url:
                        observe_result = await find_and_call_observe(
                            current_url,
                            agent.tools,
                            STATE,
                            agent_executor,
                            f" (before {tool.name})",
                        )
                        observe_found = observe_result is not None
                    else:
                        # No current URL but tool requires pre_path
                        logger.warning(
                            f"Tool {tool.name} requires pre_path but current URL is empty"
                        )

                # Check URL pre_path precondition
                if getattr(tool, "pre_path", None):
                    pattern = tool.pre_path.replace("*", ".*")
                    current_url = await executor.state["get_url"]()
                    if not current_url or not re.match(pattern, current_url):
                        # URL precondition failed - fallback to ai_exec
                        error_msg = (
                            f"Expected URL matching {tool.pre_path}, got {current_url}"
                        )
                        return await fallback_to_ai_exec(
                            f"URL precondition failed: {error_msg}"
                        )

                # Check state preconditions
                if getattr(tool, "pre", None) and isinstance(tool.pre, dict):
                    for key, pat in tool.pre.items():
                        if pat is None:
                            continue
                        ok = _check_pattern(STATE.get(key), pat, kwargs)
                        if not ok:
                            # State precondition failed - fallback to ai_exec
                            error_msg = f"Precondition failed for {key}: expected {pat}, got {STATE.get(key)}"
                            return await fallback_to_ai_exec(
                                f"state precondition failed: {error_msg}"
                            )

                # Try to call the actual SMCP tool implementation
                # If it fails, fall back to ai_exec
                try:
                    result = await execute_smcp_tool(
                        agent_executor, tool, kwargs, use_fast_timeout=use_fast_timeout
                    )
                except Exception as e:
                    # Execution failed - fallback to ai_exec
                    # This will trigger regeneration with new code in next iteration
                    return await fallback_to_ai_exec("execution error", e)

                # For OBSERVE tools: Always update STATE with result
                # (Observe tools ARE the source of truth for page state)
                if hasattr(tool, "type") and tool.type == SMCPToolType.OBSERVE:
                    if isinstance(result, dict):
                        # Update STATE with raw dict values
                        STATE.update(result)
                        logger.debug(
                            f"Updated STATE from observe tool {tool.name}: {result}"
                        )
                else:
                    # For NON-OBSERVE tools:
                    # Call matching observe tool AFTER execution to refresh STATE
                    # (The tool may have navigated to a new page, so we need fresh STATE)

                    # Call observe tool AFTER execution to refresh STATE
                    # The postcondition check below will generally ensure we reached the right page,
                    # so get_url() will return the correct URL to match against observe tools
                    current_url = await executor.state["get_url"]()
                    if current_url:
                        await find_and_call_observe(
                            current_url,
                            agent.tools,
                            STATE,
                            agent_executor,
                            f" (after {tool.name})",
                        )

                # Ensure postconditions reflected in STATE (handle $param references)
                if getattr(tool, "post", None) and isinstance(tool.post, dict):
                    for key, pat in tool.post.items():
                        if pat is None:
                            continue
                        if isinstance(pat, str) and pat.startswith("$"):
                            ref = pat[1:]
                            if ref in kwargs:
                                STATE[key] = kwargs.get(ref)
                            else:
                                # fallback to result value
                                STATE[key] = (result or {}).get(ref, STATE.get(key))

                # Validate postconditions after observe
                if getattr(tool, "post", None) and isinstance(tool.post, dict):
                    for key, pat in tool.post.items():
                        if pat is None:
                            continue
                        # Skip $param references - they were already set above
                        if isinstance(pat, str) and pat.startswith("$"):
                            continue
                        # Validate the pattern
                        ok = _check_pattern(STATE.get(key), pat, kwargs)
                        if not ok:
                            logger.warning(
                                f"Postcondition validation failed for {key}: expected {pat}, got {STATE.get(key)}"
                            )

                # Wrap result in Pydantic output model if one exists
                # This allows LLM-generated code to work with pure Pydantic types
                if output_pydantic_model and isinstance(result, dict):
                    try:
                        result = output_pydantic_model(**result)
                    except Exception as e:
                        logger.warning(
                            f"Failed to wrap result in {output_pydantic_model.__name__}: {e}"
                        )
                        # Return raw dict if wrapping fails

                return result

            # Execute the wrapper code to create the typed wrapper function
            local_scope = {
                "Optional": Optional,
                f"_execute_{tool.name}": execute_impl,
                **{
                    param_type.__name__: param_type
                    for param_type, _ in param_type_info.values()
                    if param_type_info
                },
            }
            exec(wrapper_code, local_scope)
            wrapper = local_scope[tool.name]

            return wrapper

        wrapped_tools[t.name] = make_tool_wrapper(
            t, input_model, output_model, param_types
        )

    # Create _call_tool_impl that maps to the wrapped tools
    async def _call_tool_impl(tool_name: str, params: Dict[str, Any]):
        """Internal implementation that calls the appropriate wrapped tool."""
        if tool_name in wrapped_tools:
            return await wrapped_tools[tool_name](**params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    # Register wrapped SMCP tools into executor state
    executor.send_tools(wrapped_tools)

    # Register Pydantic models so they're available in generated code
    executor.state.update(pydantic_models)

    # Register _call_tool_impl
    executor.state["_call_tool_impl"] = _call_tool_impl

    # Add utility functions (these are available inside generated code)

    @tool_hook
    async def get_url() -> str:
        """
        Get the current URL from the browser.
        This is the single source of truth for URL - don't use STATE["current_url"].

        CRITICAL: Uses multiple fallbacks with recovery logic to ensure correct URL:
        1. browser.get_current_page_url() - most reliable when agent_focus is set
        2. browser.get_current_page().url - direct page access as fallback
        3. If agent_focus is None, get first available page from getTargets()
        4. Return empty string if all fail (better than about:blank)

        This recovery is needed because agent_focus can become None between iterations
        (especially after ai_exec sub-executor finishes), causing get_current_page_url()
        to return about:blank incorrectly.
        """
        try:
            # First attempt: Use browser-use's built-in method
            url = await browser.get_current_page_url()
            if url and url != "about:blank":
                logger.debug(f"get_url() via get_current_page_url: {url}")
                return url
            elif url == "about:blank":
                logger.warning(
                    "get_current_page_url returned about:blank, attempting recovery"
                )
        except Exception as e:
            logger.debug(f"Failed to get current URL via get_current_page_url: {e}")

        # Fallback 1: Get page and extract URL directly
        try:
            page = await browser.get_current_page()
            if page:
                url = getattr(page, "url", "")
                if url and url != "about:blank":
                    logger.info(f"Recovered URL via page.url: {url}")
                    return url
                elif url == "about:blank":
                    logger.warning(
                        "page.url is also about:blank - browser genuinely on blank page"
                    )
                    return ""  # Return empty instead of about:blank
        except Exception as e2:
            logger.warning(f"Failed to get current URL via page.url: {e2}")

        # Fallback 2: Try getting from browser session's CDP target info
        try:
            if hasattr(browser, "agent_focus") and browser.agent_focus:
                target_info = await browser.agent_focus.get_target_info()
                if target_info and target_info.get("url"):
                    url = target_info["url"]
                    if url and url != "about:blank":
                        logger.info(f"Recovered URL via CDP target_info: {url}")
                        return url
        except Exception as e3:
            logger.debug(f"Failed to get URL via agent_focus target_info: {e3}")

        # Fallback 3: When agent_focus is None/lost, get first available page from CDP targets
        try:
            if hasattr(browser, "cdp_client"):
                targets_result = await browser.cdp_client.send.Target.getTargets()
                targetInfos = targets_result.get("targetInfos", [])

                # Filter for main page (type='page', not 'iframe' or 'other')
                page_targets = [t for t in targetInfos if t.get("type") == "page"]

                if page_targets:
                    # Use first page target
                    main_target = page_targets[0]
                    url = main_target.get("url", "")
                    if url and url != "about:blank":
                        logger.info(f"Recovered URL via CDP first page target: {url}")
                        # Restore agent_focus to this target for future calls
                        try:
                            if hasattr(browser, "agent_focus") and hasattr(
                                browser.agent_focus, "target_id"
                            ):
                                browser.agent_focus.target_id = main_target["targetId"]
                                logger.info(f"Restored agent_focus to main page target")
                        except Exception as focus_err:
                            logger.debug(f"Could not restore agent_focus: {focus_err}")
                        return url
        except Exception as e4:
            logger.debug(f"Failed to get URL via CDP getTargets fallback: {e4}")

        # Final fallback: return empty string instead of "about:blank"
        logger.error("get_url() failed all methods, returning empty string")
        return ""

    @tool_hook
    async def goto(url: str):
        """
        Navigate to a given URL and update STATE by calling matching observe tool.

        CRITICAL: This function handles the login/redirect problem by:
        1. Navigating to the URL
        2. Finding observe tool matching the REQUESTED URL (not final URL after redirects)
        3. Running observe to update STATE with actual page state (e.g. page="login")
        4. This allows codegen to see if login is needed before proceeding

        Example: goto("https://app.com/dashboard") may redirect to login page.
        The observe tool for "/dashboard" will run and return {page: "login"},
        allowing generated code to handle login before proceeding.
        """
        page = await browser.get_current_page()
        if not page:
            return {"success": False, "error": "No active page"}

        # Navigate to URL
        await page.goto(url)

        # Find and call observe tool matching the REQUESTED URL (not current URL after redirects)
        # This is KEY for handling login redirects - we match on intent, not result
        STATE = executor.state["STATE"]
        await find_and_call_observe(
            url, agent.tools, STATE, agent_executor, f" (after goto {url})"
        )

        return {"success": True, "url": url}

    @tool_hook
    async def ai_exec(
        subtask: str, output_schema: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute an AI agent to complete a given subtask using browser-use loop mode.

        CRITICAL: Reuses the existing browser session (shares the same BrowserSession instance)
        to avoid "BrowserStateRequestEvent has no handlers" errors.

        STAYS ON CURRENT TAB: Does NOT navigate to new URLs or switch tabs. The agent works
        with the current page state to complete the subtask.

        Args:
            subtask: Description of the task to perform
            output_schema: Optional JSON schema for structured output

        Returns:
            Result from the AI agent execution (unwrapped from AgentHistoryList if needed)
        """
        # Mark that next SMCP tool should use fast timeout (ai_exec may have changed page state)
        first_smcp_after_fallback["value"] = True

        # Avoid circular import
        from typing import get_type_hints

        from pydantic import Field, create_model

        from .executor import AgentExecutor

        # Convert JSON schema to Pydantic model if output_schema provided
        output_model = None
        if output_schema and isinstance(output_schema, dict):
            try:
                # Use shared helper to convert JSON schema to Pydantic model
                output_model = json_schema_to_pydantic(output_schema, "AIExecOutput")
                if output_model:
                    logger.debug(
                        f"Created output model for ai_exec: {output_model.model_json_schema()}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to create Pydantic model from output_schema: {e}"
                )
                # Continue without structured output

        # Create new agent with same tools but new task
        # EXCLUDE SMCP tools - ai_exec should only use browser-use vision/interaction tools
        # SMCP tools require specific page state and pre/post conditions that ai_exec can't manage
        # TODO: Track tools that have failed and exclude them from retries to avoid repeated failures
        non_smcp_tools = [
            t for t in agent.tools if t.tool_executor_type != ToolExecutorType.SMCP
        ]

        new_agent = Agent(
            description=agent.description,
            tools=non_smcp_tools,  # Only include non-SMCP tools
            is_ready_timeout_ms=agent.is_ready_timeout_ms,
        )

        # CRITICAL: Reuse the existing browser session to avoid creating new EventBus
        # and causing "BrowserStateRequestEvent has no handlers" errors
        # DO NOT pass initial_url - we want to stay on the current page/tab
        # DO NOT allow navigation - ai_exec should work with current page state

        # Preserve session context so sub-executor can access ask_human and stop callbacks
        # CRITICAL: Do NOT pass send_message_callback to sub_executor!
        # ai_exec is a sub-task and should NOT send ResponseToHuman messages.
        # Only the top-level executor should send ResponseToHuman when the entire cycle completes.
        executor_kwargs = {
            "llm": llm,
            "browser": browser,
            # "send_message_callback": NOT PASSED - sub-executor should not send ResponseToHuman
            "check_stop_callback": getattr(agent_executor, "check_stop_callback", None),
            "ask_human_callback": getattr(
                agent_executor, "ask_human_callback", None
            ),  # Pass ask_human_callback so sub-tasks can ask human
            "session_id": getattr(agent_executor, "session_id", None),
            "cycle_id": getattr(agent_executor, "cycle_id", None),
            "user_id": getattr(agent_executor, "user_id", None),
            "allowed_domains": getattr(agent_executor, "allowed_domains", None),
            "timezone": getattr(agent_executor, "timezone", None),
        }
        if output_model:
            executor_kwargs["output_model_schema"] = output_model

        # Remove None-valued kwargs so defaults on AgentExecutor still apply cleanly
        executor_kwargs = {k: v for k, v in executor_kwargs.items() if v is not None}

        sub_executor = AgentExecutor(
            new_agent, **executor_kwargs
        )  # REUSE existing browser session

        # Run in loop mode (browser-use agent with visual interaction)
        # CRITICAL: Do NOT pass initial_url - stay on current page
        # browser session already has keep_alive=True set in executor._create_browser()
        result = await sub_executor.run(subtask, mode="loop", initial_url=None)

        # After ai_exec completes, call matching observe tool to update STATE
        # BUT: Skip this if stop was requested (get_url() will raise StopExecutionError)
        STATE = executor.state["STATE"]
        try:
            current_url = await get_url()
            if current_url:
                await find_and_call_observe(
                    current_url, agent.tools, STATE, agent_executor, " (after ai_exec)"
                )
        except Exception as e:
            # If stop was requested, get_url() will raise StopExecutionError
            # That's expected behavior - don't try to update STATE after stop
            if "StopExecutionError" in str(type(e).__name__):
                logger.debug(f"Skipping post-ai_exec observe due to stop request")
                # Re-raise to maintain stop semantics
                raise
            else:
                # Other errors should be logged but not block ai_exec result
                logger.warning(f"Failed to get URL after ai_exec: {e}")

        # CRITICAL: Unwrap AgentHistoryList to get the actual result
        # browser-use Agent.run() returns AgentHistoryList[AgentStructuredOutput] which has:
        # - .history (list of AgentHistory items with model_output)
        # - .structured_output property (parses final_result as JSON using output_schema)
        # - .final_result() method returns extracted_content (string)
        #
        # If output_schema was provided, it's stored in _output_model_schema and we can
        # access the parsed structured output via .structured_output property
        from browser_use.agent.views import AgentHistoryList
        from pydantic import BaseModel

        if isinstance(result, AgentHistoryList):
            logger.debug(
                f"Unwrapping AgentHistoryList (history length: {len(result.history)})"
            )

            # Empty history means the loop-mode agent never produced a step (e.g., ask_human aborted)
            if len(result.history) == 0:
                logger.warning(
                    "AgentHistoryList returned with no history; ai_exec produced no actions"
                )
                # Return a dict with items property so generated code can safely check .items
                return {
                    "status": "no_actions",
                    "message": "ai_exec completed without executing any steps",
                    "items": [],  # Empty list so code like 'for item in result.items' works
                }

            # If we have output_schema, try to get structured output
            if result._output_model_schema is not None:
                try:
                    structured = result.structured_output
                    if structured is not None:
                        logger.debug(f"Found structured_output: {type(structured)}")
                        # Return Pydantic model directly so .items works as attribute
                        # Don't convert to dict - that makes .items the dict method!
                        return structured
                except Exception as e:
                    logger.warning(f"Failed to parse structured_output: {e}")

            # Fallback to final_result (extracted_content from last action as string)
            final = result.final_result()
            if final:
                logger.debug(f"Using final_result: {final}")
                # Wrap string result in dict with items property for compatibility
                if isinstance(final, str):
                    return {"result": final, "items": []}
                return final

            # If still nothing, return dict with empty items list
            logger.warning(
                f"No structured_output or final_result found in AgentHistoryList"
            )
            return {"items": []}

        # If result is a list, get the last element (usually the final result)
        if isinstance(result, list) and len(result) > 0:
            last_item = result[-1]
            if isinstance(last_item, AgentHistoryList):
                return await ai_exec(subtask, output_schema)  # Recursively unwrap
            return last_item

        # Convert Pydantic models to dict for compatibility with generated code
        if isinstance(result, BaseModel):
            logger.debug(f"Converting Pydantic model to dict: {type(result)}")
            return result.model_dump()

        return result

    @tool_hook
    async def ai_eval(expr: str, **kwargs):
        """
        Evaluate an expression by asking the LLM.

        Supports two formats:
        1. Simple string: ai_eval("Name closest to 'Amber'")
        2. Template with variables: ai_eval("Name in {A} closest to 'Amber'", A=options.employeeOptions)

        When kwargs are provided, the prompt is formatted to show variable values:
        A=<value>
        B=<value>

        Respond with ONLY exactly what is requested: <expr with variables>

        Args:
            expr: Expression to evaluate (may contain {var} placeholders)
            **kwargs: Variable values to substitute into the expression

        Returns:
            LLM's response (just the requested value)
        """
        import json

        from browser_use.llm.messages import SystemMessage, UserMessage

        def to_serializable(value):
            """Recursively convert Pydantic models to JSON-serializable dicts."""
            if value is None:
                return None

            # Handle Pydantic models (v2)
            if hasattr(value, "model_dump"):
                return value.model_dump()

            # Handle Pydantic models (v1)
            if hasattr(value, "dict") and callable(value.dict):
                return value.dict()

            # Handle lists - recursively convert each item
            if isinstance(value, list):
                return [to_serializable(item) for item in value]

            # Handle dicts - recursively convert values
            if isinstance(value, dict):
                return {k: to_serializable(v) for k, v in value.items()}

            # Primitives pass through
            return value

        def serialize_value(value):
            """Serialize any value (Pydantic, dict, list, etc.) to string."""
            # Handle None
            if value is None:
                return "null"

            # Convert to JSON-serializable structure first
            serializable = to_serializable(value)

            # Handle dict, list, primitives - use JSON for structure
            if isinstance(serializable, (dict, list, int, float, bool)):
                return json.dumps(serializable, indent=2)

            # Handle string (could be JSON or plain text)
            if isinstance(value, str):
                # Try to parse and re-format as JSON
                try:
                    parsed = json.loads(value)
                    return json.dumps(parsed, indent=2)
                except:
                    return value

            # Fallback: use str() representation
            return str(value)

        # Build prompt with variable context if kwargs provided
        if kwargs:
            # Format variables section
            var_lines = []
            for var_name, var_value in kwargs.items():
                # Convert value to string representation (no truncation - send full data)
                var_str = serialize_value(var_value)
                var_lines.append(f"{var_name}={var_str}")

            # Build full prompt: query first, then variable mappings
            prompt = f"{expr}\n\n" + "\n".join(var_lines) + "\n"
        else:
            # Simple format
            prompt = expr

        # Log input for debugging
        logger.info(f"ai_eval INPUT:\n{prompt}\n{'=' * 80}")

        messages = [
            SystemMessage(
                content="You are a helpful assistant. Do NOT generate code. Do NOT generate JSON unless explicitly requested. Prefer standard Markdown/text and using full sentences unless otherwise instructed. Try understanding the context, e.g., 1. if the request is for picking a single item, then only respond a single item. 2. If the request is for matching from a list, then return the best match. 3. If the input is a statement presenting data (e.g., 'Here are the results: {data}'), format and present ALL the data clearly - do not summarize or pick just one item. Respond with ONLY what is requested, or format all provided data if no specific request is given."
            ),
            UserMessage(content=prompt),
        ]

        # Use stream_llm_call to record timing
        from blastai.agents.llm_streaming import stream_llm_call

        try:
            completion, timing = await stream_llm_call(llm, messages)
            result = completion.strip()
            logger.info(f"ai_eval OUTPUT: {result}")
            logger.info(
                f"ai_eval TIMING: {timing.total_seconds:.3f}s (ttft: {timing.time_to_first_token}, gen: {timing.generation_seconds})"
            )
        except Exception as e:
            logger.error(
                f"ai_eval stream_llm_call failed: {e}, falling back to direct ainvoke"
            )
            # Fallback to direct call if stream_llm_call fails
            response = await llm.ainvoke(messages)
            result = response.completion.strip()
            logger.info(f"ai_eval OUTPUT (fallback): {result}")

        # Log output for debugging
        logger.info(f"ai_eval OUTPUT: {result}")

        # Try to parse as JSON if output looks like a list or dict
        # This handles cases where LLM returns "['item1', 'item2']" as a string
        if result and (result.startswith("[") or result.startswith("{")):
            try:
                parsed = json.loads(result)
                logger.info(f"ai_eval PARSED: {type(parsed).__name__}")
                return parsed
            except json.JSONDecodeError:
                # Try Python literal eval for single-quoted strings
                try:
                    import ast

                    parsed = ast.literal_eval(result)
                    logger.info(f"ai_eval PARSED (ast): {type(parsed).__name__}")
                    return parsed
                except (ValueError, SyntaxError):
                    pass  # Return as string if parsing fails

        return result

    additional_functions = {}
    additional_functions["get_url"] = get_url
    additional_functions["goto"] = goto
    additional_functions["ai_exec"] = ai_exec
    additional_functions["ai_eval"] = ai_eval

    # Add CORE tools (ask_human, etc.) from agent.tools
    # CoreTool model doesn't have a 'function' field - it's just metadata
    # We need to create the actual implementation based on the tool name
    for tool in agent.tools:
        if tool.tool_executor_type == ToolExecutorType.CORE:
            if tool.name in ("ask_human", "ask_human_cli"):
                # Create ask_human function for code mode
                # Check if DBOS context is available (ask_human_callback)
                if (
                    hasattr(agent_executor, "ask_human_callback")
                    and agent_executor.ask_human_callback
                ):
                    # DBOS workflow context - use create_ask_human_tool factory with callback
                    # CRITICAL: Don't pass cycle_id/user_email here - the callback already has them from workflow context
                    from .tools_hitl import create_ask_human_tool

                    ask_human_func = create_ask_human_tool(
                        agent_executor.ask_human_callback
                    )
                    additional_functions[tool.name] = ask_human_func
                    logger.info(f"Registered DBOS-backed {tool.name} for code mode")
                else:
                    # CLI context - use ask_human_cli
                    from .tools_hitl import ask_human_cli

                    additional_functions[tool.name] = ask_human_cli
                    logger.info(f"Registered CLI-backed {tool.name} for code mode")

    # Register utilities into executor
    executor.send_tools(additional_functions)

    # Return the prepared executor
    return executor


__all__ = ["create_python_executor"]
