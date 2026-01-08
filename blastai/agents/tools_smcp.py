import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from browser_use import ActionResult, Tools
from pydantic import BaseModel, Field, create_model

from .models import SMCPTool

logger = logging.getLogger(__name__)


@dataclass
class DescriptionOptions:
    """Configuration for what to include in tool descriptions sent to LLM."""

    include_pre: bool = True  # Include preconditions (e.g., page_type: store)
    include_pre_tools: bool = True  # Include pre_tools requirements
    include_post: bool = True  # Include postconditions


# Default options - can be overridden per agent_executor
DEFAULT_DESCRIPTION_OPTIONS = DescriptionOptions()


def build_enhanced_description(
    tool: SMCPTool, options: Optional[DescriptionOptions] = None
) -> str:
    """
    Build an enhanced description for an SMCP tool that includes pre/post conditions.

    Similar to _build_definition_code in codegen.py, this communicates tool requirements
    to the LLM so it knows what state/tools are needed before calling.

    Args:
        tool: The SMCP tool
        options: Configuration for what to include (defaults to DEFAULT_DESCRIPTION_OPTIONS)

    Returns:
        Enhanced description string
    """
    if options is None:
        options = DEFAULT_DESCRIPTION_OPTIONS

    parts = [tool.description]

    # Add preconditions (e.g., "Preconditions: page_type=store")
    if options.include_pre and tool.pre:
        pre_items = []
        for key, value in tool.pre.items():
            if value is not None and value != "":
                pre_items.append(f"{key}={value}")
        if pre_items:
            parts.append(f"Preconditions: {', '.join(pre_items)}")

    # Add pre_tools requirements (e.g., "Requires: Call goto_store() first")
    if options.include_pre_tools and tool.pre_tools:
        required_tools = set()
        for param_name, tool_names in tool.pre_tools.items():
            required_tools.update(tool_names)
        if required_tools:
            for req_tool in sorted(required_tools):
                parts.append(f"Requires: Call {req_tool}() first.")

    # Add postconditions (e.g., "Postconditions: page_type=checkout")
    if options.include_post and tool.post:
        post_items = []
        for key, value in tool.post.items():
            if value is not None and value != "":
                post_items.append(f"{key}={value}")
        if post_items:
            parts.append(f"Postconditions: {', '.join(post_items)}")

    return " ".join(parts)


def add_smcp_tool(
    agent_executor,
    tool: SMCPTool,
    description_options: Optional[DescriptionOptions] = None,
):
    """Add an SMCP tool to browser-use with unique registration.

    Args:
        agent_executor: The AgentExecutor instance
        tool: The SMCP tool to add
        description_options: Options for what to include in tool description
            (pre-conditions, pre-tools, post-conditions). If None, uses defaults.
    """

    logger.info(f"Adding SMCP tool: {tool.name}")

    # Build enhanced description with pre/post conditions
    description = build_enhanced_description(tool, description_options)

    # Create the tool function that takes inputs dict and returns ActionResult
    async def tool_func(inputs: Dict[str, Any]) -> ActionResult:
        """
        Execute SMCP tool and return ActionResult.
        """
        output = await execute_smcp_tool(agent_executor, tool, inputs)
        # Convert dict output to ActionResult
        return ActionResult(extracted_content=json.dumps(output))

    # Set unique function name to avoid collisions in registry
    tool_func.__name__ = tool.name

    # Register with browser-use using the tool's unique name
    # The registry uses function names as identifiers
    # NOTE: Don't set allowed_domains - we want SMCP tools always available in system prompt
    # The pre_path is used for validation INSIDE the tool execution, not for filtering
    allowed_domains = None

    # Create Pydantic model from tool's input_schema if it has properties
    # This avoids OpenAI structured output validation errors
    param_model = None
    has_inputs = tool.input_schema and bool(tool.input_schema.get("properties"))

    if has_inputs:
        # Create a Pydantic model from the JSON schema
        # All fields will be required (no defaults) to satisfy OpenAI strict mode
        fields = {}
        properties = tool.input_schema.get("properties", {})
        required = tool.input_schema.get("required", [])

        for prop_name, prop_schema in properties.items():
            # Get type from schema
            prop_type = Any  # Default to Any
            if "type" in prop_schema:
                type_str = prop_schema["type"]
                if type_str == "string":
                    prop_type = str
                elif type_str == "integer":
                    prop_type = int
                elif type_str == "number":
                    prop_type = float
                elif type_str == "boolean":
                    prop_type = bool
                elif type_str == "object":
                    prop_type = Dict[str, Any]
                elif type_str == "array":
                    prop_type = list

            # Add field (no default if required)
            if prop_name in required:
                fields[prop_name] = (prop_type, ...)
            else:
                fields[prop_name] = (Optional[prop_type], None)

        # Create dynamic Pydantic model
        if fields:
            param_model = create_model(f"{tool.name}_Params", **fields)

    # Register action with browser-use
    # CRITICAL: Must set __name__ BEFORE decorator runs, as registry uses func.__name__ for registration
    # If we have a param_model, the function will receive params as a BaseModel instance
    if param_model:

        async def dynamic_tool(params):
            """Execute SMCP tool with inputs."""
            # Convert Pydantic model to dict for SMCP execution
            inputs_dict = (
                params.model_dump() if hasattr(params, "model_dump") else params.dict()
            )
            return await tool_func(inputs_dict)

        # Set name BEFORE decorator
        dynamic_tool.__name__ = tool.name

        # Now apply decorator - it will use tool.name as the action name
        dynamic_tool = agent_executor.tools.action(
            description=description,
            allowed_domains=allowed_domains,
            param_model=param_model,
        )(dynamic_tool)
    else:
        # No parameters needed
        async def dynamic_tool():
            """Execute SMCP tool."""
            return await tool_func({})

        # Set name BEFORE decorator
        dynamic_tool.__name__ = tool.name

        # Now apply decorator - it will use tool.name as the action name
        dynamic_tool = agent_executor.tools.action(
            description=description, allowed_domains=allowed_domains
        )(dynamic_tool)

    # Store reference
    agent_executor._dynamic_tools[tool.name] = tool_func
    agent_executor._registered_tool_names.add(tool.name)

    logger.info(f"Successfully registered SMCP tool: {tool.name}")


async def execute_smcp_tool(
    agent_executor,
    tool: SMCPTool,
    inputs: Dict[str, Any],
    use_fast_timeout: bool = False,
) -> Dict[str, Any]:
    """
    Execute an SMCP tool with is_ready -> execute -> is_completed phases.

    EFFICIENT: Makes only 3 CDP calls total by wrapping loops in JavaScript.

    Args:
        agent_executor: The AgentExecutor instance
        tool: The SMCP tool to execute
        inputs: Input parameters matching the tool's input schema
        use_fast_timeout: If True, use 1s timeout for is_ready (for fast precondition failure)

    Returns:
        Output matching the tool's output schema
    """
    from browser_use.browser.session import BrowserSession

    # agent_executor.browser IS a BrowserSession instance
    browser_session: BrowserSession = agent_executor.browser
    cdp_session = await browser_session.get_or_create_cdp_session()

    # Helper to evaluate JavaScript
    async def evaluate_js(code: str) -> Any:
        """Evaluate JavaScript code in the browser."""
        try:
            result = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={
                    "expression": code,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
                session_id=cdp_session.session_id,
            )

            if result.get("exceptionDetails"):
                exception = result["exceptionDetails"]
                # Get detailed error information
                error_text = exception.get("text", "Unknown error")
                line_number = exception.get("lineNumber", "unknown")
                column_number = exception.get("columnNumber", "unknown")

                # Try to get exception details from the exception object
                exc_obj = exception.get("exception", {})
                description = exc_obj.get("description", "")

                error_msg = f"JavaScript error: {error_text}"
                if description:
                    error_msg += f" - {description}"
                error_msg += f" at line {line_number}:{column_number}"

                logger.error(error_msg)
                logger.error(
                    f"Full exception details: {json.dumps(exception, indent=2)}"
                )
                raise RuntimeError(error_msg)

            return result.get("result", {}).get("value")
        except Exception as e:
            logger.error(f"Failed to evaluate JS: {e}")
            raise

    # Phase 1: Is_ready (ONE CDP call with while loop in JS)
    if tool.is_ready:
        logger.info(f"Running is_ready phase for {tool.name}")
        inputs_json = json.dumps(inputs)

        # Adjust timeout and attempts based on execution mode or fast timeout flag
        # Loop mode OR first tool after fallback: fast failure (1s timeout, 3 attempts)
        # Code mode: full timeout (30s or configured, 30 attempts)
        if use_fast_timeout or (
            hasattr(agent_executor, "current_mode")
            and agent_executor.current_mode == "loop"
        ):
            max_attempts = 3
            total_timeout_ms = 1000  # 1 second for fast failure
            if use_fast_timeout:
                logger.info(
                    f"Using fast timeout (1s) for is_ready on {tool.name} (first tool after fallback)"
                )
        else:
            max_attempts = 30
            total_timeout_ms = agent_executor.agent.is_ready_timeout_ms

        delay_ms = total_timeout_ms // max_attempts

        is_ready_code = f"""
(async function() {{
    const inputs = {inputs_json};
    const maxAttempts = {max_attempts};
    const delayMs = {delay_ms};
    let attempts = 0;
    let lastReason = null;
    
    while (attempts < maxAttempts) {{
        try {{
            const result = (function(inputs) {{ {tool.is_ready} }})(inputs);
            // Handle both true/false and [false, "reason"] formats
            if (result === true) {{
                return {{ success: true, attempts: attempts + 1 }};
            }} else if (Array.isArray(result) && result[0] === false) {{
                lastReason = result[1] || "Unknown reason";
            }} else if (result === false) {{
                lastReason = "Not yet ready for execution according to is_ready";
            }}
        }} catch (e) {{
            lastReason = e.message || e.toString();
        }}
        attempts++;
        
        // Async delay - yields to event loop
        await new Promise(resolve => setTimeout(resolve, delayMs));
    }}
    
    return {{ success: false, attempts: attempts, error: "is_ready timeout", reason: lastReason }};
}})()
"""
        result = await evaluate_js(is_ready_code)
        if not result.get("success"):
            reason = result.get("reason", "Unknown reason")

            # Context-specific suggestions based on tool and error
            suggestions = []
            if (
                "get_restaurant_details" in tool.name.lower()
                and "not found" in reason.lower()
            ):
                suggestions = [
                    "You must navigate to a restaurant's detail page first using goto_restaurant_detail(restaurantName='...')",
                    "Then you can call get_restaurant_details() to extract the data",
                    "Or use click actions to manually navigate to the restaurant page",
                ]
            elif (
                "goto_restaurant_detail" in tool.name.lower()
                and "not found" in reason.lower()
            ):
                suggestions = [
                    "Make sure you're on the restaurants list page (use goto_restaurants_list() first)",
                    "Or try clicking on the restaurant name directly from the list",
                    "The restaurant name must match exactly what appears in the list",
                ]
            else:
                suggestions = [
                    "click on interactive elements to navigate",
                    "scroll to find relevant content",
                    "extract data from the current page",
                    "navigate back or to a different page",
                    "use input/search to find what you need",
                ]

            suggestion_text = "\\n".join(f"  - {s}" for s in suggestions)
            error_msg = (
                f"Tool '{tool.name}' is_ready failed after {result.get('attempts', 0)} attempts over {total_timeout_ms}ms. "
                f"Last reason: {reason}\\n"
                f"Suggestion: Try alternative approaches:\\n"
                f"{suggestion_text}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        logger.info(f"is_ready passed after {result.get('attempts')} attempts")

    # Phase 2: Execute (ONE CDP call)
    logger.info(f"Executing {tool.name} with inputs: {inputs}")

    # Inject inputs into JavaScript context
    inputs_json = json.dumps(inputs)

    # Check if execute is a full function expression (starts with 'function' or 'async function')
    execute_stripped = tool.execute.strip()
    is_function_expr = execute_stripped.startswith(
        "function"
    ) or execute_stripped.startswith("async function")

    # Check if execute code uses await (needs async wrapper)
    uses_await = "await " in tool.execute

    if is_function_expr:
        # Full function expression - call it directly and await if async
        execute_code = f"""
(async function() {{
    try {{
        const inputs = {inputs_json};
        return await ({tool.execute})(inputs);
    }} catch (e) {{
        return {{ error: e.message }};
    }}
}})()
"""
    else:
        # Function body - wrap in async function if it uses await
        if uses_await:
            execute_code = f"""
(async function() {{
    try {{
        const inputs = {inputs_json};
        return await (async function(inputs) {{ {tool.execute} }})(inputs);
    }} catch (e) {{
        return {{ error: e.message }};
    }}
}})()
"""
        else:
            # Synchronous code - use regular function wrapper
            execute_code = f"""
(function() {{
    try {{
        const inputs = {inputs_json};
        return (function(inputs) {{ {tool.execute} }})(inputs);
    }} catch (e) {{
        return {{ error: e.message }};
    }}
}})()
"""

    output = await evaluate_js(execute_code)
    if isinstance(output, dict) and "error" in output:
        raise RuntimeError(f"Execute error: {output['error']}")

    logger.info(f"Execute output: {output}")

    # Phase 3: Is_completed (ONE CDP call with while loop in JS)
    if tool.is_completed:
        logger.info(f"Running is_completed phase for {tool.name}")
        output_json = json.dumps(output)
        inputs_json = json.dumps(inputs)

        # Adjust timeout and attempts based on execution mode
        # Loop mode: fast failure (1.5s total, 3 attempts)
        # Code mode: full timeout (15s total, 30 attempts)
        if (
            hasattr(agent_executor, "current_mode")
            and agent_executor.current_mode == "loop"
        ):
            max_attempts = 3
            delay_ms = 500  # 3 * 500ms = 1.5s total
        else:
            max_attempts = 30
            delay_ms = 500  # 30 * 500ms = 15s total

        is_completed_code = f"""
(async function() {{
    const inputs = {inputs_json};
    const output = {output_json};
    const maxAttempts = {max_attempts};
    const delayMs = {delay_ms};
    let attempts = 0;
    let lastReason = null;
    
    while (attempts < maxAttempts) {{
        try {{
            const result = (function(inputs, output) {{ {tool.is_completed} }})(inputs, output);
            // Handle both true/false and [false, "reason"] formats
            if (result === true) {{
                return {{ success: true, attempts: attempts + 1 }};
            }} else if (Array.isArray(result) && result[0] === false) {{
                lastReason = result[1] || "Unknown reason";
            }} else if (result === false) {{
                lastReason = "is_completed returned false";
            }}
        }} catch (e) {{
            lastReason = e.message || e.toString();
        }}
        attempts++;
        
        // Async delay - yields to event loop
        await new Promise(resolve => setTimeout(resolve, delayMs));
    }}
    
    return {{ success: false, attempts: attempts, error: "is_completed timeout", reason: lastReason }};
}})()
"""

        try:
            result = await evaluate_js(is_completed_code)
            if not result.get("success"):
                reason = result.get("reason", "Unknown reason")
                error_msg = f"is_completed failed after {result.get('attempts', 0)} attempts. Last reason: {reason}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            logger.info(f"is_completed passed after {result.get('attempts')} attempts")
        except Exception as e:
            # If execution context was destroyed during is_completed check,
            # it usually means the page navigated/reloaded, which often indicates success
            error_str = str(e)
            if (
                "Execution context was destroyed" in error_str
                or "Cannot find context" in error_str
            ):
                logger.warning(
                    f"is_completed check failed due to context destruction (page likely navigated) - assuming success: {e}"
                )
                # Don't raise - treat as successful completion since page navigated
            else:
                # Other errors should still fail
                raise

    return output
