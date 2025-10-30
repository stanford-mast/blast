"""
Python code executor creation with tool wrapping and STATE management.

Provides function to create LocalPythonExecutor instances with wrapped SMCP tools,
precondition/postcondition checking, and automatic observe tool calls.
"""

import asyncio
import logging
import re
from typing import Dict, Any, Optional, Callable, List, TYPE_CHECKING

from pydantic import BaseModel, Field, create_model

from .local_python_executor import LocalPythonExecutor
from .models import Agent, SMCPTool, ToolExecutorType, SMCPToolType
from .tools_smcp import execute_smcp_tool
from .schema_utils import json_schema_to_pydantic

if TYPE_CHECKING:
    from browser_use.llm.base import BaseChatModel
    from browser_use.browser import BrowserSession
    from browser_use.llm.messages import UserMessage

logger = logging.getLogger(__name__)


async def find_and_call_observe(
    url: str,
    tools: List,
    state: Dict[str, Any],
    agent_executor,
    context: str = ""
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
    observe_tools = [t for t in tools 
                    if t.tool_executor_type == ToolExecutorType.SMCP 
                    and hasattr(t, 'type') and t.type == SMCPToolType.OBSERVE]
    
    if not observe_tools:
        return None
    
    for obs_tool in observe_tools:
        obs_pattern = getattr(obs_tool, "pre_path", None)
        if obs_pattern:
            obs_regex = obs_pattern.replace("*", ".*")
            if re.match(obs_regex, url):
                try:
                    observe_result = await execute_smcp_tool(agent_executor, obs_tool, {})
                    if isinstance(observe_result, dict):
                        state.update(observe_result)
                        logger.debug(f"Updated STATE from {obs_tool.name}{context}: {observe_result}")
                        return observe_result
                except Exception as e:
                    logger.warning(f"Failed to call {obs_tool.name}{context}: {e}")
                break
    return None


def create_python_executor(
    agent: Agent,
    browser: 'BrowserSession',
    llm: 'BaseChatModel',
    agent_executor
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
    
    Returns:
        LocalPythonExecutor with all tools and utilities registered
    """
    # Create executor first so we can register wrapped tool functions that
    # assert preconditions against STATE and update STATE with postconditions.
    import_modules = {
        "json": __import__("json"),
        "re": __import__("re"),
        "asyncio": __import__("asyncio"),
    }

    executor = LocalPythonExecutor(
        additional_functions={},
        additional_imports=import_modules
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
                    "object": dict
                }
                python_type = type_map.get(param_type_str, str)
                param_types[param_name] = (python_type, is_required)
        
        # Create Pydantic model for output validation if tool has output schema
        output_model = None
        if t.output_schema and t.output_schema.get("properties"):
            # Use shared helper to convert JSON schema to Pydantic model
            output_model_name = f"{t.name.title().replace('_', '')}Output"
            output_model = json_schema_to_pydantic(t.output_schema, output_model_name)

        def make_tool_wrapper(tool: SMCPTool, input_pydantic_model, output_pydantic_model, param_type_info):
            # Build typed signature for wrapper
            if param_type_info:
                # Create wrapper with typed parameters
                # We'll use exec to create a function with the right signature
                param_parts = []
                for param_name, (param_type, is_required) in param_type_info.items():
                    type_name = param_type.__name__
                    if is_required:
                        param_parts.append(f"{param_name}: {type_name}")
                    else:
                        param_parts.append(f"{param_name}: Optional[{type_name}] = None")
                
                sig_str = ", ".join(param_parts)
                
                # Build the wrapper function code
                wrapper_code = f"""
async def {tool.name}({sig_str}):
    '''Wrapper that validates inputs, calls observe, checks preconditions, executes tool, and updates STATE.'''
    # Build kwargs from parameters
    kwargs = {{{', '.join([f'"{p}": {p}' for p in param_type_info.keys()])}}}
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
                async def fallback_to_ai_exec(reason: str, error: Optional[Exception] = None):
                    """
                    Fallback to ai_exec when tool cannot execute.
                    Uses the tool's output_schema as structured output for ai_exec.
                    """
                    logger.warning(f"Tool {tool.name} cannot execute ({reason}), falling back to ai_exec")
                    if error:
                        logger.debug(f"Error details: {error}")
                    
                    # Construct task for ai_exec
                    params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                    fallback_task = f"{tool.description}"
                    if params_str:
                        fallback_task += f" with parameters: {params_str}"
                    
                    # Call ai_exec with output schema if available
                    if 'ai_exec' in executor.state:
                        # Pass output_schema to ai_exec for structured output
                        return await executor.state['ai_exec'](
                            fallback_task, 
                            output_schema=tool.output_schema if hasattr(tool, 'output_schema') else None
                        )
                    else:
                        # Fallback not available, re-raise original error if exists
                        if error:
                            raise error
                        else:
                            raise RuntimeError(f"Tool {tool.name} preconditions not met and ai_exec not available")
                
                # Validate input parameters with Pydantic if model exists
                if input_pydantic_model is not None:
                    try:
                        validated = input_pydantic_model(**kwargs)
                        kwargs = validated.model_dump(exclude_none=True)
                    except Exception as e:
                        # Parameter validation failed - fallback to ai_exec
                        return await fallback_to_ai_exec("invalid parameters", e)
                
                # BEFORE execution: Call matching observe tool to update STATE
                # This is critical for detecting login pages, wrong pages, etc.
                observe_found = False
                if getattr(tool, "pre_path", None):
                    current_url = await executor.state["get_url"]()  # Use get_url() function
                    if current_url:
                        observe_result = await find_and_call_observe(
                            current_url, agent.tools, STATE, agent_executor, f" (before {tool.name})"
                        )
                        observe_found = observe_result is not None
                    else:
                        # No current URL but tool requires pre_path
                        logger.warning(f"Tool {tool.name} requires pre_path but current URL is empty")
                
                # Check URL pre_path precondition
                if getattr(tool, "pre_path", None):
                    pattern = tool.pre_path.replace("*", ".*")
                    current_url = await executor.state["get_url"]()
                    if not current_url or not re.match(pattern, current_url):
                        # URL precondition failed - fallback to ai_exec
                        error_msg = f"Expected URL matching {tool.pre_path}, got {current_url}"
                        return await fallback_to_ai_exec(f"URL precondition failed: {error_msg}")

                # Check state preconditions
                if getattr(tool, "pre", None) and isinstance(tool.pre, dict):
                    for key, pat in tool.pre.items():
                        if pat is None:
                            continue
                        ok = _check_pattern(STATE.get(key), pat, kwargs)
                        if not ok:
                            # State precondition failed - fallback to ai_exec
                            error_msg = f"Precondition failed for {key}: expected {pat}, got {STATE.get(key)}"
                            return await fallback_to_ai_exec(f"state precondition failed: {error_msg}")

                # Try to call the actual SMCP tool implementation
                # If it fails, fall back to ai_exec
                try:
                    result = await execute_smcp_tool(agent_executor, tool, kwargs)
                except Exception as e:
                    # Execution failed - fallback to ai_exec
                    return await fallback_to_ai_exec("execution error", e)

                # For OBSERVE tools: Always update STATE with result
                # (Observe tools ARE the source of truth for page state)
                if hasattr(tool, 'type') and tool.type == SMCPToolType.OBSERVE:
                    if isinstance(result, dict):
                        STATE.update(result)
                        logger.debug(f"Updated STATE from observe tool {tool.name}: {result}")
                else:
                    # For NON-OBSERVE tools:
                    # 1. Update STATE with any result data
                    # 2. Call matching observe tool AFTER execution to refresh STATE
                    #    (The tool may have navigated to a new page, so we need fresh STATE)
                    
                    # Update STATE with result dict contents
                    try:
                        if isinstance(result, dict):
                            STATE.update(result)
                    except Exception:
                        # Best-effort update - don't fail on update
                        pass
                    
                    # Call observe tool AFTER execution to refresh STATE
                    # The postcondition check below will generally ensure we reached the right page,
                    # so get_url() will return the correct URL to match against observe tools
                    current_url = await executor.state["get_url"]()
                    if current_url:
                        await find_and_call_observe(
                            current_url, agent.tools, STATE, agent_executor, f" (after {tool.name})"
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
                            logger.warning(f"Postcondition validation failed for {key}: expected {pat}, got {STATE.get(key)}")

                # Wrap result in Pydantic model if output schema exists
                if output_pydantic_model is not None and isinstance(result, dict):
                    try:
                        result = output_pydantic_model(**result)
                    except Exception as e:
                        logger.warning(f"Failed to validate output with Pydantic model for {tool.name}: {e}")
                        # Continue with dict result if validation fails
                
                return result
            
            # Execute the wrapper code to create the typed wrapper function
            local_scope = {
                'Optional': Optional,
                f'_execute_{tool.name}': execute_impl,
                **{param_type.__name__: param_type for param_type, _ in param_type_info.values() if param_type_info}
            }
            exec(wrapper_code, local_scope)
            wrapper = local_scope[tool.name]
            
            return wrapper

        wrapped_tools[t.name] = make_tool_wrapper(t, input_model, output_model, param_types)

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
    
    async def get_url() -> str:
        """
        Get the current URL from the browser.
        This is the single source of truth for URL - don't use STATE["current_url"].
        """
        try:
            page = await browser.get_current_page()
            if page:
                return getattr(page, "url", "")
        except Exception as e:
            logger.warning(f"Failed to get current URL: {e}")
        return ""
    
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
        await find_and_call_observe(url, agent.tools, STATE, agent_executor, f" (after goto {url})")
        
        return {"success": True, "url": url}
    
    async def ai_exec(subtask: str, output_schema: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute an AI agent to complete a given subtask.
        
        Args:
            subtask: Description of the task to perform
            output_schema: Optional JSON schema for structured output
        
        Returns:
            Result from the AI agent execution
        """
        # Avoid circular import
        from .executor import AgentExecutor
        from pydantic import create_model, Field
        from typing import get_type_hints
        
        # Convert JSON schema to Pydantic model if output_schema provided
        output_model = None
        if output_schema and isinstance(output_schema, dict):
            try:
                # Use shared helper to convert JSON schema to Pydantic model
                output_model = json_schema_to_pydantic(output_schema, "AIExecOutput")
                if output_model:
                    logger.debug(f"Created output model for ai_exec: {output_model.model_json_schema()}")
            except Exception as e:
                logger.warning(f"Failed to create Pydantic model from output_schema: {e}")
                # Continue without structured output
        
        # Create new agent with same tools but new task
        new_agent = Agent(
            description=agent.description,
            tools=agent.tools.copy(),
            is_ready_timeout_ms=agent.is_ready_timeout_ms
        )
        
        # Create executor with shared browser and optional output_model_schema
        # Get current URL to pass as initial_url to sub-agent
        current_url = await get_url()
        
        if output_model:
            sub_executor = AgentExecutor(
                new_agent, 
                llm=llm, 
                browser=browser,
                output_model_schema=output_model  # Pass structured output to browser-use Agent
            )
        else:
            sub_executor = AgentExecutor(new_agent, llm=llm, browser=browser)
        
        result = await sub_executor.run(subtask, mode="loop", initial_url=current_url if current_url else None)
        
        # After ai_exec completes, call matching observe tool to update STATE
        STATE = executor.state["STATE"]
        current_url = await get_url()
        if current_url:
            await find_and_call_observe(current_url, agent.tools, STATE, agent_executor, " (after ai_exec)")
        
        return result

    async def ai_eval(expr: str):
        """
        Evaluate an expression by asking the LLM.
        """
        from browser_use.llm.messages import UserMessage
        messages = [UserMessage(content=expr + "\nRespond with ONLY exactly what is requested.")]
        response = await llm.ainvoke(messages)
        return response.completion

    additional_functions = {}
    additional_functions["get_url"] = get_url
    additional_functions["goto"] = goto
    additional_functions["ai_exec"] = ai_exec
    additional_functions["ai_eval"] = ai_eval

    # Register utilities into executor
    executor.send_tools(additional_functions)

    # Return the prepared executor
    return executor


__all__ = ["create_python_executor"]
