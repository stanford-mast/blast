"""
Agent executor that integrates browser-use with SMCP and core tools.
"""

import asyncio
import os
from typing import Dict, Any, Optional, Callable, List, TYPE_CHECKING
from pydantic import BaseModel, Field, create_model
import json
import logging

from browser_use import Browser, Agent as BrowserUseAgent, Tools, ActionResult
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage
from openai import OpenAI

from smolagents.local_python_executor import LocalPythonExecutor

from .models import Agent, Tool, SMCPTool, CoreTool, ToolExecutorType, SMCPToolType

if TYPE_CHECKING:
    from browser_use.browser import BrowserSession

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Executes an Agent by integrating with browser-use.
    
    Handles:
    - Converting SMCP tools to browser-use actions
    - Managing core tools (update/remove SMCP tools)
    - Running in loop mode (browser-use) or code mode (LocalPythonExecutor)
    """
    
    def __init__(
        self, 
        agent: Agent, 
        llm: Optional[BaseChatModel] = None,
        browser: Optional['BrowserSession'] = None,
        state_aware: bool = True
    ):
        """
        Initialize AgentExecutor.
        
        Args:
            agent: The Agent to execute
            llm: Optional LLM model. If None, will create from environment.
            browser: Optional BrowserSession instance. If None, creates new one. 
                     Multiple Agents can share a BrowserSession instance.
            state_aware: Include preconditions/postconditions in generated code (default True)
        """
        self.agent = agent
        self.llm = llm or self._create_llm_from_env()
        self.browser = browser or self._create_browser()
        self.state_aware = state_aware
        self.tools = Tools()
        
        # Storage for dynamically created tools
        self._dynamic_tools: Dict[str, Callable] = {}
        
        # Track which tool names are registered to avoid conflicts
        self._registered_tool_names: set[str] = set()
        
        # Browser-use agent will be created per run (can't be reused across tasks)
        self.browser_use_agent: Optional[BrowserUseAgent] = None
        
        # Set up tools
        self._setup_tools()
        
        # Set up Python executor for code mode
        self.python_executor = None  # Created lazily
    
    def _create_llm_from_env(self) -> BaseChatModel:
        """Create LLM from environment variables."""
        # Import here to avoid circular dependencies
        from browser_use.llm.openai.chat import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4")
        
        return ChatOpenAI(
            model=model,
            api_key=api_key
        )
    
    def _create_browser(self):
        """Create browser-use BrowserSession instance."""
        from browser_use import BrowserSession, BrowserProfile
        
        # Get viewport/window size from environment
        width = int(os.getenv("BROWSER_WIDTH", "1280"))
        height = int(os.getenv("BROWSER_HEIGHT", "720"))
        headless = os.getenv("HEADLESS", "false").lower() == "true"
        
        # Build Chrome args for WSL GPU fix
        args = []
        
        # Disable GPU if BROWSER_DISABLE_GPU=1 (for WSL transparency issues)
        if os.getenv("BROWSER_DISABLE_GPU", "").lower() in ("1", "true", "yes"):
            args.extend(["--disable-gpu", "--disable-gpu-sandbox"])
        
        # Create profile with proper viewport and window settings
        profile = BrowserProfile(
            headless=headless,
            # Viewport is what the browser *sees*
            viewport={"width": width, "height": height},
            # Window size is the OS window size (used when not headless)
            window_size={"width": width, "height": height},
            # Chrome command-line arguments
            args=args,
        )
        
        return BrowserSession(browser_profile=profile)
    
    def _setup_tools(self):
        """Set up all tools from the agent."""
        for tool in self.agent.tools:
            if tool.tool_executor_type == ToolExecutorType.SMCP:
                self._add_smcp_tool(tool)
            elif tool.tool_executor_type == ToolExecutorType.CORE:
                self._add_core_tool(tool)
    
    def _add_smcp_tool(self, tool: SMCPTool):
        """Add an SMCP tool to browser-use with unique registration."""
        
        # Create the tool function that takes inputs dict and returns ActionResult
        async def tool_func(inputs: Dict[str, Any]) -> ActionResult:
            """
            Execute SMCP tool and return ActionResult.
            """
            output = await self._execute_smcp_tool(tool, inputs)
            # Convert dict output to ActionResult
            import json
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
                param_model = create_model(
                    f'{tool.name}_Params',
                    **fields
                )
        
        # Register action with browser-use
        # CRITICAL: Must set __name__ BEFORE decorator runs, as registry uses func.__name__ for registration
        # If we have a param_model, the function will receive params as a BaseModel instance
        if param_model:
            async def dynamic_tool(params):
                """Execute SMCP tool with inputs."""
                # Convert Pydantic model to dict for SMCP execution
                inputs_dict = params.model_dump() if hasattr(params, 'model_dump') else params.dict()
                return await tool_func(inputs_dict)
            
            # Set name BEFORE decorator
            dynamic_tool.__name__ = tool.name
            
            # Now apply decorator - it will use tool.name as the action name
            dynamic_tool = self.tools.action(
                description=tool.description,
                allowed_domains=allowed_domains,
                param_model=param_model
            )(dynamic_tool)
        else:
            # No parameters needed
            async def dynamic_tool():
                """Execute SMCP tool."""
                return await tool_func({})
            
            # Set name BEFORE decorator
            dynamic_tool.__name__ = tool.name
            
            # Now apply decorator - it will use tool.name as the action name
            dynamic_tool = self.tools.action(
                description=tool.description,
                allowed_domains=allowed_domains
            )(dynamic_tool)
        
        # Store reference
        self._dynamic_tools[tool.name] = tool_func
        self._registered_tool_names.add(tool.name)
    
    async def _execute_smcp_tool(self, tool: SMCPTool, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an SMCP tool with is_ready -> execute -> check phases.
        
        EFFICIENT: Makes only 3 CDP calls total by wrapping loops in JavaScript.
        
        Args:
            tool: The SMCP tool to execute
            inputs: Input parameters matching the tool's input schema
            
        Returns:
            Output matching the tool's output schema
        """
        from browser_use.browser.session import BrowserSession
        
        # self.browser IS a BrowserSession instance
        browser_session: BrowserSession = self.browser
        cdp_session = await browser_session.get_or_create_cdp_session()
        
        # Helper to evaluate JavaScript
        async def evaluate_js(code: str) -> Any:
            """Evaluate JavaScript code in the browser."""
            try:
                result = await cdp_session.cdp_client.send.Runtime.evaluate(
                    params={
                        'expression': code,
                        'returnByValue': True,
                        'awaitPromise': True
                    },
                    session_id=cdp_session.session_id
                )
                
                if result.get('exceptionDetails'):
                    exception = result['exceptionDetails']
                    error_msg = f"JavaScript error: {exception.get('text', 'Unknown error')}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                return result.get('result', {}).get('value')
            except Exception as e:
                logger.error(f"Failed to evaluate JS: {e}")
                raise
        
        # Phase 1: Is_ready (ONE CDP call with while loop in JS)
        if tool.is_ready:
            logger.info(f"Running is_ready phase for {tool.name}")
            inputs_json = json.dumps(inputs)
            
            # Calculate delay between attempts based on agent's timeout
            # 30 attempts spread over is_ready_timeout_ms
            max_attempts = 30
            total_timeout_ms = self.agent.is_ready_timeout_ms
            delay_ms = total_timeout_ms // max_attempts
            
            stabilize_code = f"""
(function() {{
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
        
        // Synchronous delay (blocking but simple)
        const start = Date.now();
        while (Date.now() - start < delayMs) {{}}
    }}
    
    return {{ success: false, attempts: attempts, error: "Stabilization timeout", reason: lastReason }};
}})()
"""
            result = await evaluate_js(stabilize_code)
            if not result.get('success'):
                reason = result.get('reason', 'Unknown reason')
                error_msg = f"Failed to is_ready after {result.get('attempts', 0)} attempts over {total_timeout_ms}ms. Last reason: {reason}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            logger.info(f"Stabilized after {result.get('attempts')} attempts")
        
        # Phase 2: Execute (ONE CDP call)
        logger.info(f"Executing {tool.name} with inputs: {inputs}")
        
        # Inject inputs into JavaScript context
        inputs_json = json.dumps(inputs)
        
        # Check if execute is a full function expression (starts with 'function' or 'async function')
        execute_stripped = tool.execute.strip()
        is_function_expr = execute_stripped.startswith('function') or execute_stripped.startswith('async function')
        
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
            # Function body - wrap like is_ready/is_correct for consistency
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
        if isinstance(output, dict) and 'error' in output:
            raise RuntimeError(f"Execute error: {output['error']}")
        
        logger.info(f"Execute output: {output}")
        
        # Phase 3: Is_correct (ONE CDP call with while loop in JS)
        if tool.is_correct:
            logger.info(f"Running check phase for {tool.name}")
            output_json = json.dumps(output)
            check_code = f"""
(function() {{
    const output = {output_json};
    const maxAttempts = 30;
    const delayMs = 500;
    let attempts = 0;
    let lastReason = null;
    
    while (attempts < maxAttempts) {{
        try {{
            const result = (function(output) {{ {tool.is_correct} }})(output);
            // Handle both true/false and [false, "reason"] formats
            if (result === true) {{
                return {{ success: true, attempts: attempts + 1 }};
            }} else if (Array.isArray(result) && result[0] === false) {{
                lastReason = result[1] || "Unknown reason";
            }} else if (result === false) {{
                lastReason = "Check returned false";
            }}
        }} catch (e) {{
            lastReason = e.message || e.toString();
        }}
        attempts++;
        
        // Synchronous delay
        const start = Date.now();
        while (Date.now() - start < delayMs) {{}}
    }}
    
    return {{ success: false, attempts: attempts, error: "Check timeout", reason: lastReason }};
}})()
"""
            
            result = await evaluate_js(check_code)
            if not result.get('success'):
                reason = result.get('reason', 'Unknown reason')
                error_msg = f"Check failed after {result.get('attempts', 0)} attempts. Last reason: {reason}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            logger.info(f"Check passed after {result.get('attempts')} attempts")
        
        return output
    
    def _add_core_tool(self, tool: CoreTool):
        """Add a core tool for managing SMCP tools."""
        
        if tool.name == "update_smcp_tool":
            # Create proper Pydantic model for parameters (inspired by MCP client)
            # ALL fields are required to satisfy OpenAI strict mode
            # Use List for pre/post to avoid Dict[str, Any] schema issues
            from pydantic import ConfigDict
            
            class ConfiguredBaseModel(BaseModel):
                model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
            
            # Define a model for key-value pairs in pre/post conditions
            StateEntry = create_model(
                'StateEntry',
                __base__=ConfiguredBaseModel,
                key=(str, Field(..., description="State variable name")),
                value=(str, Field(..., description="State variable value"))
            )
            
            UpdateSMCPToolParams = create_model(
                'UpdateSMCPToolParams',
                __base__=ConfiguredBaseModel,
                name=(str, Field(..., description="Tool identifier (e.g. 'list_restaurants')")),
                type=(str, Field(..., description="One of: observe, listItems, getFields, setFilter, setFields, gotoItem, gotoField")),
                execute=(str, Field(..., description="JavaScript BODY (has access to 'inputs' object)")),
                is_ready=(str, Field(..., description="JavaScript BODY returning true when ready, or [false, 'reason'] when not ready (empty string if not needed)")),
                is_correct=(str, Field(..., description="JavaScript BODY returning true when valid, or [false, 'reason'] when invalid (empty string if not needed)")),
                preconditions=(List[StateEntry], Field(..., description="State before running as list of {key, value} pairs (empty list [] if none)")),
                postconditions=(List[StateEntry], Field(..., description="State after running as list of {key, value} pairs (empty list [] if none)")),
                input_parameters=(List[str], Field(..., description="Array of param names (empty list [] if none)"))
            )
            
            # Register with proper param_model
            @self.tools.action(
                description="Create or update an SMCP tool with auto-generated schemas",
                param_model=UpdateSMCPToolParams
            )
            async def update_smcp_tool(params: BaseModel) -> ActionResult:
                """
                Create or update an SMCP tool with auto-generated schemas and metadata.
                """
                try:
                    # Extract parameters from Pydantic model
                    name = params.name  # type: ignore
                    type_str = params.type  # type: ignore
                    execute = params.execute  # type: ignore
                    is_ready = params.is_ready  # type: ignore
                    check = params.is_correct  # type: ignore
                    preconditions_list = params.preconditions  # type: ignore - List[StateEntry]
                    postconditions_list = params.postconditions  # type: ignore - List[StateEntry]
                    input_parameters = params.input_parameters  # type: ignore
                    
                    # Convert list of {key, value} to dict for AbstractState
                    preconditions = {entry.key: entry.value for entry in preconditions_list}  # type: ignore
                    postconditions = {entry.key: entry.value for entry in postconditions_list}  # type: ignore
                    
                    # AUTO-GENERATE: lang (always js)
                    lang = "js"
                    
                    # AUTO-GENERATE: title from name (convert snake_case/camelCase to words)
                    def name_to_title(name: str) -> str:
                        # Handle snake_case
                        name = name.replace('_', ' ')
                        # Handle camelCase by inserting spaces before capitals
                        import re
                        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
                        # Capitalize each word
                        return name.title()
                    
                    title = name_to_title(name)
                    
                    # AUTO-GENERATE: pre_path from current browser URL
                    pre_path = ""
                    try:
                        current_url = await self.browser.get_current_page_url()
                        if current_url and current_url != 'about:blank':
                            # Extract origin + /* pattern
                            from urllib.parse import urlparse
                            parsed = urlparse(current_url)
                            pre_path = f"{parsed.scheme}://{parsed.netloc}/*"
                            logger.info(f"Auto-generated pre_path: {pre_path} from current URL: {current_url}")
                    except Exception as e:
                        logger.warning(f"Could not auto-generate pre_path from browser URL: {e}")
                        pre_path = "*"  # Fallback to match any URL
                    
                    # AUTO-GENERATE: input_schema, output_schema, description using LLM
                    logger.info(f"Generating schemas and description for {name} using LLM...")
                    
                    # Retry loop for LLM generation
                    max_attempts = 3
                    schema_data = None
                    last_error = None
                    
                    for attempt in range(max_attempts):
                        try:
                            schema_prompt = f"""Generate JSON with description, input_schema, and output_schema for this SMCP tool.

Tool: {name}
Type: {type_str}
Input Parameters: {input_parameters}
Execute Script:
```javascript
{execute}
```

Respond with valid JSON only:
{{
  "description": "What this tool does (one sentence)",
  "input_schema": {{
    "type": "object",
    "properties": {{}},
    "required": []
  }},
  "output_schema": {{
    "type": "object",
    "properties": {{}},
    "required": []
  }}
}}

For input_schema: Use parameters {input_parameters}. Each needs type (string/number/boolean/array/object) and description.
For output_schema: Analyze the execute script's return value. Common patterns:
- listItems type: {{"items": {{"type": "array"}}}}
- observe type: {{"page": {{"type": "string"}}, ...}}
- getFields type: field objects
- setFields/setFilter/gotoItem/gotoField: {{"success": {{"type": "boolean"}}}}"""
                            
                            # Call LLM
                            from browser_use.llm.messages import SystemMessage, UserMessage
                            messages = [
                                SystemMessage(content="You are a JSON generator. Output ONLY valid JSON, no markdown, no explanation."),
                                UserMessage(content=schema_prompt)
                            ]
                            
                            response = await self.llm.ainvoke(messages)
                            response_text = response.completion.strip()
                            
                            # Extract JSON from markdown if needed
                            import re
                            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
                            if json_match:
                                response_text = json_match.group(1)
                            
                            # Parse JSON
                            schema_data = json.loads(response_text)
                            
                            # Validate structure
                            if "description" not in schema_data:
                                raise ValueError("Missing 'description' field")
                            if "input_schema" not in schema_data:
                                raise ValueError("Missing 'input_schema' field")
                            if "output_schema" not in schema_data:
                                raise ValueError("Missing 'output_schema' field")
                            
                            # Validate schemas have required structure
                            for schema_name in ["input_schema", "output_schema"]:
                                schema = schema_data[schema_name]
                                if not isinstance(schema, dict):
                                    raise ValueError(f"{schema_name} must be object")
                                if schema.get("type") != "object":
                                    raise ValueError(f"{schema_name} must have type='object'")
                                if "properties" not in schema:
                                    raise ValueError(f"{schema_name} must have 'properties'")
                                if "required" not in schema:
                                    schema["required"] = []  # Auto-add if missing
                            
                            # Success!
                            logger.info(f"LLM generation succeeded on attempt {attempt + 1}")
                            break
                            
                        except Exception as e:
                            last_error = str(e)
                            logger.warning(f"LLM generation attempt {attempt + 1}/{max_attempts} failed: {e}")
                            if attempt < max_attempts - 1:
                                continue
                    
                    if schema_data is None:
                        # All attempts failed
                        return ActionResult(
                            error=f"Failed to generate schemas after {max_attempts} attempts. Last error: {last_error}"
                        )
                    
                    description = schema_data["description"]
                    input_schema = schema_data["input_schema"]
                    output_schema = schema_data["output_schema"]
                    
                    logger.info(f"Generated - description: {description}")
                    logger.info(f"Generated - input_schema: {input_schema}")
                    logger.info(f"Generated - output_schema: {output_schema}")

                    # Create SMCPTool
                    smcp_tool = SMCPTool(
                        name=name,
                        title=title,
                        description=description,
                        input_schema=input_schema,
                        output_schema=output_schema,
                        tool_executor_type=ToolExecutorType.SMCP,
                        lang=lang,
                        is_ready=is_ready,
                        is_correct=check,
                        execute=execute,
                        pre_path=pre_path,
                        pre=preconditions,
                        post=postconditions,
                        type=SMCPToolType(type_str)
                    )
                    
                    # Remove existing tool with same name if it exists
                    existing = self.agent.get_tool(smcp_tool.name)
                    if existing:
                        self.agent.remove_tool(smcp_tool.name)
                        if smcp_tool.name in self._dynamic_tools:
                            del self._dynamic_tools[smcp_tool.name]
                        if smcp_tool.name in self._registered_tool_names:
                            self._registered_tool_names.remove(smcp_tool.name)
                        logger.info(f"Overriding existing tool: {smcp_tool.name}")
                    
                    # Add new tool to agent
                    self.agent.add_tool(smcp_tool)
                    
                    # Register with browser-use
                    self._add_smcp_tool(smcp_tool)
                    
                    # Build invocation example
                    if input_schema and input_schema.get("properties"):
                        props = input_schema["properties"]
                        example_params = {prop: f"<{prop}>" for prop in props.keys()}
                        invocation = f'{{{json.dumps(smcp_tool.name)}: {json.dumps(example_params)}}}'
                    else:
                        invocation = f'{{{json.dumps(smcp_tool.name)}: {{}}}}'
                    
                    success_msg = f"✓ Tool '{smcp_tool.name}' {'updated' if existing else 'created'}.\n\n{description}\n\nCall it: {invocation}"
                    return ActionResult(extracted_content=success_msg)
                    
                except Exception as e:
                    logger.error(f"Failed to update SMCP tool: {e}")
                    import traceback
                    traceback.print_exc()
                    return ActionResult(error=f"Failed to create tool '{name}': {str(e)}")
            
            # Set unique name
            update_smcp_tool.__name__ = "update_smcp_tool"
        
        elif tool.name == "remove_smcp_tool":
            @self.tools.action(description=tool.description)
            async def remove_smcp_tool(name: str) -> ActionResult:
                """
                Remove an SMCP tool by name.
                """
                success = self.agent.remove_tool(name)
                if success:
                    if name in self._dynamic_tools:
                        del self._dynamic_tools[name]
                    if name in self._registered_tool_names:
                        self._registered_tool_names.remove(name)
                    return ActionResult(
                        extracted_content=f"Tool {name} removed successfully"
                    )
                else:
                    return ActionResult(error=f"Tool {name} not found")
            
            remove_smcp_tool.__name__ = "remove_smcp_tool"
        
        elif tool.name == "list_smcp_tools":
            def format_tool_info(t: SMCPTool, include_code: bool = False) -> List[str]:
                """Helper to format tool information with optional JavaScript code."""
                tool_info = [f"\n{t.name} (callable action):"]
                tool_info.append(f"  Description: {t.description}")
                tool_info.append(f"  Type: {t.type.value}")
                
                if not include_code:
                    tool_info.append("")
                
                # Show input schema
                if t.input_schema and t.input_schema.get("properties"):
                    props = t.input_schema["properties"]
                    required = t.input_schema.get("required", [])
                    params = []
                    for prop_name, prop_schema in props.items():
                        prop_type = prop_schema.get("type", "any")
                        is_required = "required" if prop_name in required else "optional"
                        params.append(f"{prop_name} ({prop_type}, {is_required})")
                    tool_info.append(f"  Parameters: {', '.join(params)}")
                    
                    # Show example invocation
                    example_params = {prop: f"<{prop}>" for prop in props.keys()}
                    tool_info.append(f"  Call as action: {{{json.dumps(t.name)}: {json.dumps(example_params)}}}")
                else:
                    tool_info.append(f"  Parameters: none")
                    tool_info.append(f"  Call as action: {{{json.dumps(t.name)}: {{}}}}")
                
                # Show output schema
                if t.output_schema and t.output_schema.get("properties"):
                    props = t.output_schema["properties"]
                    outputs = [f"{prop_name} ({prop_schema.get('type', 'any')})" 
                              for prop_name, prop_schema in props.items()]
                    tool_info.append(f"  Returns: {', '.join(outputs)}")
                else:
                    tool_info.append(f"  Returns: (schema not defined)")
                
                # Show pre/post state mappings
                if not include_code:
                    if t.pre and any(t.pre.values()):
                        tool_info.append(f"  Preconditions: {json.dumps(t.pre)}")
                    else:
                        tool_info.append(f"  Preconditions: (not defined)")
                    
                    if t.post and any(t.post.values()):
                        tool_info.append(f"  Postconditions: {json.dumps(t.post)}")
                    else:
                        tool_info.append(f"  Postconditions: (not defined)")
                    
                    if t.pre_path:
                        tool_info.append(f"  URL Pattern: {t.pre_path}")
                    else:
                        tool_info.append(f"  URL Pattern: (not defined)")
                else:
                    # Include code view - show pre/post and code
                    tool_info.append("")
                    tool_info.append(f"  Preconditions: {json.dumps(t.pre) if t.pre and any(t.pre.values()) else '{}'}")
                    tool_info.append(f"  Postconditions: {json.dumps(t.post) if t.post and any(t.post.values()) else '{}'}")
                    tool_info.append(f"  URL Pattern: {t.pre_path if t.pre_path else '(not defined)'}")
                    tool_info.append("")
                    
                    # Show JavaScript code for each phase
                    tool_info.append("  === JavaScript Code ===")
                    tool_info.append("")
                    
                    if t.is_ready:
                        tool_info.append("  IS_READY (checks page ready; returns true or [false, reason]):")
                        tool_info.append("  ```javascript")
                        tool_info.append(f"  {t.is_ready}")
                        tool_info.append("  ```")
                        tool_info.append("")
                    else:
                        tool_info.append("  IS_READY: (not defined)")
                        tool_info.append("")
                    
                    tool_info.append("  EXECUTE (main action):")
                    tool_info.append("  ```javascript")
                    tool_info.append(f"  {t.execute}")
                    tool_info.append("  ```")
                    tool_info.append("")
                    
                    if t.is_correct:
                        tool_info.append("  IS_CORRECT (validates output; returns true or [false, reason]):")
                        tool_info.append("  ```javascript")
                        tool_info.append(f"  {t.is_correct}")
                        tool_info.append("  ```")
                        tool_info.append("")
                    else:
                        tool_info.append("  IS_CORRECT: (not defined)")
                        tool_info.append("")
                
                return tool_info
            
            @self.tools.action(description=tool.description)
            async def list_smcp_tools(get_code_for: str = "") -> ActionResult:
                """
                List all SMCP tools with their signatures and example usage.
                
                Args:
                    get_code_for: Optional tool name to get detailed code for (includes is_ready/execute/is_correct scripts)
                """
                smcp_tools_data = [
                    t for t in self.agent.tools
                    if t.tool_executor_type == ToolExecutorType.SMCP
                ]
                
                if not smcp_tools_data:
                    return ActionResult(extracted_content="No SMCP tools available")
                
                # If get_code_for specified, filter to just that tool and show full code
                if get_code_for:
                    filtered = [t for t in smcp_tools_data if t.name == get_code_for]
                    if not filtered:
                        return ActionResult(error=f"Tool '{get_code_for}' not found")
                    
                    t = filtered[0]
                    assert isinstance(t, SMCPTool)
                    
                    result = "\n".join(format_tool_info(t, include_code=True))
                    return ActionResult(extracted_content=result)
                
                # Default: List all tools with basic info (no code)
                tools_info = [format_tool_info(t, include_code=False) for t in smcp_tools_data]
                result = "Available SMCP actions (use them in your action list):\n" + "\n".join("\n".join(info) for info in tools_info)
                result += "\n\nTip: Use get_code_for parameter to see detailed JavaScript code for a specific tool"
                return ActionResult(extracted_content=result)
            
            list_smcp_tools.__name__ = "list_smcp_tools"
        
        elif tool.name == "ask_html":
            @self.tools.action(description="Ask a question about the page HTML to get guidance on selectors, structure, or data attributes for creating SMCP tools")
            async def ask_html(query: str, max_length: int = None, print_html: bool = False) -> ActionResult:
                """
                Query the current page HTML to get targeted guidance for writing JavaScript.
                
                This uses an LLM to analyze the page HTML and answer specific questions about:
                - CSS selectors for finding elements
                - Data attributes available
                - Page structure patterns
                - How to extract specific data
                
                Much more efficient than returning full HTML - only returns what you need.
                
                Args:
                    query: Question about the page (e.g. "What selector finds restaurant cards?")
                    max_length: Deprecated - no longer used. HTML is not truncated initially.
                    print_html: If True, print the filtered HTML sent to LLM for debugging (default False)
                """
                try:
                    # Import utilities
                    from browser_use.llm.messages import SystemMessage, UserMessage
                    import random
                    
                    logger.info(f"Analyzing HTML to answer: {query}")
                    
                    # Get current page HTML directly via CDP (includes ALL attributes)
                    page = await self.browser.get_current_page()
                    cdp_session = await self.browser.get_or_create_cdp_session()
                    
                    # Get the outer HTML of the body element (excludes head/scripts/styles)
                    result = await cdp_session.cdp_client.send.Runtime.evaluate(
                        params={
                            'expression': 'document.body ? document.body.outerHTML : document.documentElement.outerHTML',
                            'returnByValue': True
                        },
                        session_id=cdp_session.session_id
                    )
                    
                    page_html = result.get('result', {}).get('value', '')
                    
                    # Get current URL for context
                    current_url = await self.browser.get_current_page_url()
                    
                    original_length = len(page_html)
                    logger.info(f"Extracted {original_length:,} chars of raw HTML from {current_url}")
                    
                    # Basic cleaning only (remove head/script/style, NO truncation)
                    page_html = self._clean_html_for_analysis(page_html, max_length=None)
                    
                    cleaned_length = len(page_html)
                    logger.info(f"Cleaned HTML: {original_length:,} -> {cleaned_length:,} chars (no truncation)")
                    
                    # Progressive truncation on LLM failure
                    max_attempts = 10
                    current_html = page_html
                    
                    for attempt in range(max_attempts):
                        # Print HTML if requested (for debugging)
                        if print_html and attempt == 0:
                            print("\n" + "="*60)
                            print("FILTERED HTML (sent to LLM):")
                            print("="*60)
                            print(current_html)
                            print("="*60 + "\n")
                        
                        # Create LLM prompt
                        system_prompt = """You are an expert at analyzing HTML and providing JavaScript guidance.

<input>
You will receive a question about a webpage's HTML structure and the HTML content.
</input>

<instructions>
- Analyze the HTML to answer the question precisely
- Provide concrete CSS selectors, data attributes, or code patterns
- Be specific - include actual class names, IDs, and attributes from the HTML
- If multiple approaches exist, recommend the most reliable one
- Focus on what's actually in the HTML, don't guess
- Keep answers concise and actionable
</instructions>

<output_format>
Provide a direct, actionable answer with:
1. The specific selector or approach
2. An example of how to use it in JavaScript
3. Any caveats or edge cases to consider
</output_format>"""
                        
                        user_prompt = f"""URL: {current_url}

Question: {query}

HTML:
{current_html}"""
                        
                        # Ask the LLM - create a fresh instance to avoid state issues
                        from browser_use.llm.openai.chat import ChatOpenAI
                        
                        api_key = os.getenv("OPENAI_API_KEY")
                        model = os.getenv("OPENAI_MODEL", "gpt-4o")
                        
                        # Create a dedicated LLM instance for this ask_html call
                        ask_html_llm = ChatOpenAI(
                            model=model,
                            api_key=api_key,
                            temperature=0  # Deterministic for consistency
                        )
                        
                        messages = [
                            SystemMessage(content=system_prompt),
                            UserMessage(content=user_prompt)
                        ]
                        
                        try:
                            response = await ask_html_llm.ainvoke(messages)
                            guidance = response.completion.strip()
                            
                            logger.info(f"HTML guidance generated ({len(guidance)} chars)")
                            logger.info(f"ask_html output:\n{guidance}")
                            
                            # Success! Format result
                            result_text = f"Question: {query}\n\nGuidance:\n{guidance}"
                            
                            # For long guidance, use include_extracted_content_only_once
                            # to add it to context once without bloating long-term memory
                            if len(result_text) > 1000:
                                memory = f"Received HTML guidance for: {query[:100]}..."
                                include_once = True
                            else:
                                memory = result_text  # Short enough to include in long-term memory
                                include_once = False
                            
                            return ActionResult(
                                extracted_content=result_text,
                                long_term_memory=memory,
                                include_extracted_content_only_once=include_once
                            )
                            
                        except Exception as llm_error:
                            # LLM call failed - probably context length exceeded
                            logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_attempts}): {llm_error}")
                            
                            if attempt < max_attempts - 1:
                                # Randomly remove HTML elements to reduce size by ~50%
                                current_html = self._reduce_html_randomly(current_html, target_ratio=0.5)
                                new_length = len(current_html)
                                logger.info(f"Reduced HTML to {new_length:,} chars (~50% reduction)")
                            else:
                                # All attempts failed
                                raise Exception(f"Failed after {max_attempts} attempts with progressive truncation") from llm_error
                    
                except Exception as e:
                    logger.error(f"Failed to ask HTML: {e}")
                    import traceback
                    traceback.print_exc()
                    return ActionResult(error=f"Failed to analyze HTML: {str(e)}")
            
            ask_html.__name__ = "ask_html"
    
    def _clean_html_for_analysis(self, html: str, max_length: int = None) -> str:
        """
        Clean HTML for LLM analysis while preserving selector-relevant information.
        
        Strategy:
        1. Remove <head>, <script>, <style>, <svg> tags entirely
        2. Remove URLs longer than 128 characters
        3. Truncate extremely long text content or attribute values with ellipsis
        4. Compress repeated similar sibling elements
        5. Keep ALL attributes including data-* (crucial for selectors!)
        6. Only apply final truncation if max_length is specified
        
        Args:
            html: Raw HTML string
            max_length: Target maximum length (None = no truncation)
            
        Returns:
            Cleaned HTML string
        """
        import re
        
        # Step 1: Remove head, script, style, svg tags
        html = re.sub(r'<head[^>]*>.*?</head>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<svg[^>]*>.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Step 2: Remove long URLs (>128 chars) from attributes like href, src, etc.
        def filter_long_urls(match):
            attr_name = match.group(1)
            attr_value = match.group(2)
            if len(attr_value) > 128:
                return f'{attr_name}="[long URL omitted]"'
            return match.group(0)
        
        # Match URL attributes: href, src, action, data, etc.
        html = re.sub(r'((?:href|src|action|data|srcset|poster|cite|formaction))\s*=\s*"([^"]{129,})"', 
                     filter_long_urls, html, flags=re.IGNORECASE)
        
        # Step 3: Truncate very long text content (>1000 chars) with ellipsis in middle
        def truncate_long_text(match):
            text = match.group(0)
            if len(text) > 1000:
                # Keep first 400 and last 400 chars with ellipsis in middle
                return text[:400] + '...[text truncated]...' + text[-400:]
            return text
        
        # Match text content between tags (not attributes)
        html = re.sub(r'(?<=>)[^<]{1000,}(?=<)', truncate_long_text, html)
        
        # Step 4: Truncate very long attribute values (>500 chars) that aren't URLs
        def truncate_long_attr(match):
            attr_name = match.group(1)
            # Skip if it's a URL attribute (already handled above)
            if attr_name.lower() in ['href', 'src', 'action', 'data', 'srcset', 'poster', 'cite', 'formaction']:
                return match.group(0)
            attr_value = match.group(2)
            if len(attr_value) > 500:
                # Keep first 200 and last 200 chars
                truncated = attr_value[:200] + '...' + attr_value[-200:]
                return f'{attr_name}="{truncated}"'
            return match.group(0)
        
        html = re.sub(r'(\w+)="([^"]{500,})"', truncate_long_attr, html)
        
        # Step 5: Compress repeated similar sibling elements
        # Find sequences of 10+ similar consecutive tags with same attributes
        def compress_repeated_elements(html_text):
            lines = html_text.split('\n')
            result = []
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for opening tag
                tag_match = re.match(r'<(\w+)([^>]*)>', line)
                if tag_match:
                    tag_name = tag_match.group(1)
                    attrs = tag_match.group(2).strip()
                    
                    # Count consecutive similar siblings
                    similar_count = 1
                    j = i + 1
                    
                    while j < len(lines) and similar_count < 100:  # Cap at 100 to avoid infinite loops
                        next_line = lines[j].strip()
                        next_match = re.match(r'<(\w+)([^>]*)>', next_line)
                        
                        if next_match and next_match.group(1) == tag_name:
                            # Same tag - check if attributes are similar (allowing values to differ)
                            next_attrs = next_match.group(2).strip()
                            # Extract attribute names only (not values)
                            attr_names = set(re.findall(r'(\w+)=', attrs))
                            next_attr_names = set(re.findall(r'(\w+)=', next_attrs))
                            
                            if attr_names == next_attr_names:
                                similar_count += 1
                                j += 1
                            else:
                                break
                        else:
                            break
                    
                    # If we found 10+ similar elements, compress the middle
                    if similar_count >= 10:
                        # Keep first 3
                        for k in range(i, min(i + 3, i + similar_count)):
                            result.append(lines[k])
                        
                        # Add compression comment
                        result.append(f'<!-- ... omitted {similar_count - 6} similar <{tag_name}> elements ... -->')
                        
                        # Keep last 3
                        for k in range(max(i + similar_count - 3, i + 3), i + similar_count):
                            result.append(lines[k])
                        
                        i += similar_count
                        continue
                
                result.append(lines[i])
                i += 1
            
            return '\n'.join(result)
        
        html = compress_repeated_elements(html)
        
        # Step 6: Final truncation only if max_length is specified
        if max_length is not None and len(html) > max_length:
            # Try to truncate at a tag boundary
            last_close_tag = html.rfind('>', max_length - 1000, max_length)
            if last_close_tag > 0:
                html = html[:last_close_tag + 1] + '\n<!-- HTML truncated at max_length -->'
            else:
                html = html[:max_length] + '\n<!-- HTML truncated at max_length -->'
        
        return html
    
    def _reduce_html_randomly(self, html: str, target_ratio: float = 0.5) -> str:
        """
        Randomly remove leaf elements (elements with no children) to reduce size by target_ratio.
        
        This is used when LLM context length is exceeded - we progressively
        remove random leaf elements until we reach the target size.
        
        Strategy:
        1. Find all leaf elements (elements with no child elements, only text)
        2. Iteratively remove random leaf elements until size is reduced by ~50%
        3. Preserves structure by only removing complete leaf elements
        
        Args:
            html: HTML string to reduce
            target_ratio: Target size ratio (0.5 = reduce to 50% of current size)
            
        Returns:
            Reduced HTML string
        """
        import re
        import random
        
        target_length = int(len(html) * target_ratio)
        current_html = html
        
        # Keep removing random leaf elements until we reach target size
        max_iterations = 1000  # Safety limit
        iteration = 0
        
        while len(current_html) > target_length and iteration < max_iterations:
            iteration += 1
            
            # Find all leaf elements - elements that contain only text (no child tags)
            # This pattern matches: opening tag + content without any tags + closing tag
            # Use negative lookahead to ensure content has no opening tags
            leaf_pattern = r'<(\w+)([^>]*)>(?:(?!<\w+).)*?</\1>'
            
            matches = list(re.finditer(leaf_pattern, current_html, re.DOTALL))
            
            if not matches:
                # No leaf elements found - may have self-closing tags or unusual structure
                # Try to find any small element (< 200 chars)
                small_element_pattern = r'<(\w+)([^>]*)>(.{0,200}?)</\1>'
                matches = list(re.finditer(small_element_pattern, current_html, re.DOTALL))
                
                if not matches:
                    # Can't find anything to remove safely, stop
                    break
            
            # Randomly select a leaf element to remove
            match = random.choice(matches)
            tag_name = match.group(1)
            
            # Replace the element with a smaller comment
            replacement = f'<!-- X -->'
            current_html = current_html[:match.start()] + replacement + current_html[match.end():]
        
        return current_html
    
    def _create_python_executor(self) -> LocalPythonExecutor:
        """Create Python executor for code mode."""
        
        # Create function stubs for all tools
        additional_functions = {}
        
        # Add SMCP tool functions
        for tool in self.agent.tools:
            if tool.tool_executor_type == ToolExecutorType.SMCP:
                # Create async wrapper that can be called from Python code
                def make_tool_func(t: SMCPTool):
                    async def func(**kwargs):
                        """
                        Tool implementation omitted.
                        Actual execution handled by AgentExecutor.
                        """
                        return await self._execute_smcp_tool(t, kwargs)
                    func.__name__ = t.name
                    return func
                
                additional_functions[tool.name] = make_tool_func(tool)
        
        # Add core tool functions
        for tool in self.agent.tools:
            if tool.tool_executor_type == ToolExecutorType.CORE:
                # These are already available through the tools registry
                pass
        
        # Add utility functions
        async def run(task: str, initial_url: Optional[str] = None):
            """
            Run browser-use agent with the given task.
            Creates a new Agent instance for the task.
            """
            return await self._run_loop_mode(task, initial_url)
        
        async def ask(prompt: str):
            """
            Ask the LLM a question.
            """
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.ainvoke(messages)
            return response.completion
        
        additional_functions["run"] = run
        additional_functions["ask"] = ask
        
        # Create executor
        return LocalPythonExecutor(
            additional_authorized_imports=["json", "asyncio", "re"],
            additional_functions=additional_functions
        )
    
    async def run(self, task: str, mode: str = "loop", initial_url: Optional[str] = None) -> Any:
        """
        Run the agent with the given task.
        
        Args:
            task: The task description
            mode: "loop" for browser-use Agent.run, "code" for Python code generation
            initial_url: Optional initial URL to navigate to
            
        Returns:
            The result of execution
        """
        # Prepend agent description to task
        full_task = f"{self.agent.description}{task}" if self.agent.description else task
        
        if mode == "loop":
            return await self._run_loop_mode(full_task, initial_url)
        elif mode == "code":
            return await self._run_code_mode(full_task, initial_url)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'loop' or 'code'.")
    
    async def _run_loop_mode(self, task: str, initial_url: Optional[str] = None) -> Any:
        """
        Run in loop mode using browser-use Agent.run.
        
        Creates a new browser-use Agent for each run to ensure clean state.
        Multiple runs can share the same BrowserSession instance.
        
        If initial_url is provided, it's passed as initial_actions to the Agent,
        which automatically navigates before starting the main task.
        """
        try:
            # Build initial actions if URL provided
            initial_actions = None
            if initial_url:
                logger.info(f"Setting initial URL: {initial_url}")
                initial_actions = [{'navigate': {'url': initial_url, 'new_tab': False}}]
            
            # Create a fresh browser-use agent for this run
            # Note: We can reuse the same BrowserSession instance across multiple agents
            self.browser_use_agent = BrowserUseAgent(
                task=task,
                llm=self.llm,
                browser=self.browser,
                tools=self.tools,
                initial_actions=initial_actions  # Agent will execute navigation before task
            )
            
            result = await self.browser_use_agent.run()
            return result
        finally:
            # Clean up agent reference (but keep browser)
            self.browser_use_agent = None
    
    async def _run_code_mode(self, task: str, initial_url: Optional[str] = None) -> Any:
        """
        Run in code mode using LLM code generation + LocalPythonExecutor.
        
        This mode:
        1. If initial_url provided, navigates there first
        2. Generates Python code using the LLM
        3. Code begins with assertion validating current URL matches initial_url
        4. Code includes assertions for tool preconditions (pre_path) and postconditions
        5. Executes code with LocalPythonExecutor
        6. If there's an error, sends it back to LLM to fix
        7. Loops until success
        """
        # First, navigate to initial URL if provided (before generating code)
        if initial_url:
            logger.info(f"Navigating to initial URL: {initial_url}")
            try:
                page = await self.browser.get_current_page()
                if page:
                    await page.goto(initial_url)
                    logger.debug(f"Navigation complete: {initial_url}")
                else:
                    logger.warning("Could not navigate: no active page in browser session")
            except Exception as e:
                logger.warning(f"Error during initial navigation: {e}")
        
        # Create python executor lazily (only when code mode is used)
        if self.python_executor is None:
            self.python_executor = self._create_python_executor()
        
        # Build available functions documentation with precondition/postcondition info
        functions_doc = self._build_functions_doc()
        
        # Build code template with imports and initial URL assertion
        code_lines = []
        code_lines.append("import re")
        code_lines.append("import asyncio")
        code_lines.append("")
        
        if initial_url:
            code_lines.append("# Verify browser is on the correct page")
            code_lines.append(f'assert current_url == "{initial_url}", \\')
            code_lines.append(f'    f"Expected URL {initial_url}, got {{current_url}}"')
            code_lines.append("")
        
        code_template = "\n".join(code_lines)
        
        prompt = f"""{task}

You have access to the following Python functions (they include preconditions and postconditions with assertions):

```python
{code_template}
{functions_doc}
```

Generate Python code to complete the task using the provided functions. 
The code is already started with imports and a URL assertion (if applicable).
Include assertions to validate tool preconditions before calling each function.
Return ONLY the Python code in a markdown code block, no explanations.
"""
        
        messages = [{"role": "user", "content": prompt}]
        max_iterations = 10
        
        for iteration in range(max_iterations):
            logger.info(f"Code mode iteration {iteration + 1}/{max_iterations}")
            
            # Get code from LLM
            response = await self.llm.ainvoke(messages)
            code = self._extract_code_from_response(response.completion)
            
            if not code:
                logger.error("No code found in LLM response")
                messages.append({"role": "assistant", "content": response.completion})
                messages.append({
                    "role": "user",
                    "content": "Please provide Python code in a markdown code block."
                })
                continue
            
            logger.info(f"Generated code:\n{code}")
            
            # Execute code
            try:
                result = self.python_executor(code)
                
                if result.is_final_answer:
                    logger.info(f"Code execution completed: {result.output}")
                    return result.output
                
                # Not final answer yet, continue
                logger.info(f"Intermediate result: {result.output}")
                logger.info(f"Logs: {result.logs}")
                
                # Add to conversation
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                messages.append({
                    "role": "user",
                    "content": f"Execution result:\nOutput: {result.output}\nLogs: {result.logs}\n\nContinue or provide final answer."
                })
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Code execution error: {error_msg}")
                
                # Send error back to LLM
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                messages.append({
                    "role": "user",
                    "content": f"Error executing code:\n{error_msg}\n\nPlease fix the code."
                })
        
        raise RuntimeError(f"Failed to complete task after {max_iterations} iterations")
    
    def _build_functions_doc(self) -> str:
        """
        Build documentation for available functions with precondition/postcondition assertions.
        
        Generates function signatures with assert statements sandwiched between
        preconditions and postconditions, all visible in the generated code block.
        """
        lines = []
        
        # Add SMCP tool functions with full precondition/implementation/postcondition structure
        for tool in self.agent.tools:
            if tool.tool_executor_type == ToolExecutorType.SMCP:
                smcp_tool = tool  # Type hint
                assert isinstance(smcp_tool, SMCPTool)
                
                lines.append(f"async def {smcp_tool.name}(**kwargs):")
                lines.append(f'    """')
                lines.append(f'    {smcp_tool.description}')
                lines.append(f'    """')
                
                # PRECONDITIONS (assert statements)
                if self.state_aware:
                    # URL pattern precondition
                    if smcp_tool.pre_path and smcp_tool.pre_path != "":
                        url_pattern = smcp_tool.pre_path.replace("*", ".*")
                        lines.append(f'    # Precondition: URL must match "{smcp_tool.pre_path}"')
                        lines.append(f'    assert re.match(r"{url_pattern}", current_url), \\')
                        lines.append(f'        f"Expected URL matching {smcp_tool.pre_path}, got {{current_url}}"')
                    
                    # State preconditions
                    if smcp_tool.pre and isinstance(smcp_tool.pre, dict):
                        for key, value in smcp_tool.pre.items():
                            lines.append(f'    # Precondition: {key} == {json.dumps(value)}')
                            lines.append(f'    assert {key} == {json.dumps(value)}, \\')
                            lines.append(f'        f"{key} must be {json.dumps(value)}"')
                
                # IMPLEMENTATION (omitted marker)
                lines.append(f'    ')
                lines.append(f'    # ... implementation omitted ...')
                lines.append(f'    result = await _call_tool_impl("{smcp_tool.name}", kwargs)')
                lines.append(f'    ')
                
                # POSTCONDITIONS (assert statements after result)
                if self.state_aware and smcp_tool.post and isinstance(smcp_tool.post, dict):
                    for key, value in smcp_tool.post.items():
                        lines.append(f'    # Postcondition: result[{json.dumps(key)}] == {json.dumps(value)}')
                        lines.append(f'    assert result.get({json.dumps(key)}) == {json.dumps(value)}, \\')
                        lines.append(f'        f"Result {json.dumps(key)} must be {json.dumps(value)}"')
                
                lines.append(f'    return result')
                lines.append('')
        
        # Add core tool functions
        for tool in self.agent.tools:
            if tool.tool_executor_type == ToolExecutorType.CORE:
                if tool.name == "update_smcp_tool":
                    lines.append('async def update_smcp_tool(name: str, title: str, description: str, execute: str, type: str, **kwargs):')
                elif tool.name == "remove_smcp_tool":
                    lines.append('async def remove_smcp_tool(name: str):')
                elif tool.name == "list_smcp_tools":
                    lines.append('async def list_smcp_tools():')
                else:
                    lines.append(f'async def {tool.name}(**kwargs):')
                
                lines.append(f'    """')
                lines.append(f'    {tool.description}')
                lines.append(f'    """')
                lines.append(f'    # ... implementation omitted ...')
                lines.append(f'    pass')
                lines.append('')
        
        # Add utility functions
        lines.append('async def run(task: str, initial_url: str = None):')
        lines.append('    """')
        lines.append('    Run browser-use Agent with the given task.')
        lines.append('    Creates a new Agent instance for this task.')
        lines.append('    Multiple runs can share the same Browser.')
        lines.append('    """')
        lines.append('    # ... implementation omitted ...')
        lines.append('    pass')
        lines.append('')
        lines.append('async def ask(prompt: str):')
        lines.append('    """Ask the LLM a question."""')
        lines.append('    # ... implementation omitted ...')
        lines.append('    pass')
        
        return '\n'.join(lines)
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from markdown code blocks."""
        import re
        
        # Look for ```python ... ``` or ``` ... ```
        patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        return None
    
    async def cleanup(self):
        """Clean up resources."""
        # Try to close the Browser if it has a close() coroutine
        if self.browser:
            try:
                close_fn = getattr(self.browser, "close", None)
                if close_fn and asyncio.iscoroutinefunction(close_fn):
                    await close_fn()
                elif close_fn and callable(close_fn):
                    # synchronous close
                    close_fn()
            except AttributeError:
                # Fallback: try to close browser-use agent if available
                pass

        # If a browser-use Agent was created, attempt to close its session
        if self.browser_use_agent:
            try:
                await self.browser_use_agent.close()
            except Exception:
                # Best-effort cleanup; don't raise
                pass
