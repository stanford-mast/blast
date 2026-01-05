

from typing import Dict, Any, Optional, List
import json
import logging
import re
import random
import os
from pydantic import BaseModel, Field, create_model
from browser_use import Tools, ActionResult

from .models import CoreTool, SMCPTool, ToolExecutorType, SMCPToolType
from .tools_smcp import add_smcp_tool

logger = logging.getLogger(__name__)


def add_core_tool(agent_executor, tool: CoreTool):
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
            is_completed=(str, Field(..., description="JavaScript BODY returning true when valid, or [false, 'reason'] when invalid (empty string if not needed)")),
            preconditions=(List[StateEntry], Field(..., description="State before running as list of {key, value} pairs (empty list [] if none)")),
            postconditions=(List[StateEntry], Field(..., description="State after running as list of {key, value} pairs (empty list [] if none)")),
            input_parameters=(List[str], Field(..., description="Array of param names (empty list [] if none)"))
        )
        
        # Register with proper param_model
        @agent_executor.tools.action(
            description="Create or update an SMCP tool with auto-generated schemas",
            param_model=UpdateSMCPToolParams
        )
        async def update_smcp_tool(params: BaseModel) -> ActionResult:
            """
            Create or update an SMCP tool with auto-generated schemas and metadata.
            """
            # TODO: Refactor much of this out
            try:
                # Extract parameters from Pydantic model
                name = params.name  # type: ignore
                type_str = params.type  # type: ignore
                execute = params.execute  # type: ignore
                is_ready = params.is_ready  # type: ignore
                check = params.is_completed  # type: ignore
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
                    current_url = await agent_executor.browser.get_current_page_url()
                    if current_url and current_url != 'about:blank':
                        # Extract domain and create wildcard pattern
                        from urllib.parse import urlparse
                        parsed = urlparse(current_url)
                        hostname = parsed.netloc
                        
                        # Get base domain (e.g., "sage.hr" from "acme.sage.hr")
                        # Strategy: If domain has 3+ parts (e.g., acme.sage.hr), use last 2 parts
                        # If domain has 2 parts (e.g., sage.hr), use as-is
                        # If domain has 1 part (e.g., localhost), use as-is
                        parts = hostname.split('.')
                        if len(parts) >= 3:
                            # acme.sage.hr -> sage.hr
                            # www.example.co.uk -> example.co.uk (handles this case too)
                            base_domain = '.'.join(parts[-2:])
                        else:
                            # sage.hr or localhost -> use as-is
                            base_domain = hostname
                        
                        # Create wildcard pattern: *sage.hr* (matches both sage.hr and acme.sage.hr)
                        pre_path = f"*{base_domain}*"
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

CRITICAL: For array types, ALWAYS specify the items schema with full detail!

For input_schema: Use parameters {input_parameters}. Each needs type (string/number/boolean/array/object) and description.

For output_schema: Analyze the execute script's return value. Common patterns:
- listItems type: {{"items": {{"type": "array", "items": {{"type": "object", "properties": {{...}}, "required": [...]}}}}}}
- observe type: {{"page": {{"type": "string", "enum": ["a", "b", "other"]}}, ...}}
- getFields type: {{"items": {{"type": "array", "items": {{"type": "object", "properties": {{...}}, "required": [...]}}}}}}
- setFields/setFilter/gotoItem/gotoField: {{"success": {{"type": "boolean"}}}}

Examples of GOOD output schemas:
1. For listItems returning [{{"name": "X", "price": "Y"}}]:
   {{"items": {{"type": "array", "items": {{"type": "object", "properties": {{"name": {{"type": "string"}}, "price": {{"type": "string"}}}}, "required": ["name", "price"]}}}}}}

2. For getFields returning [{{"title": "X", "rating": 4.5}}]:
   {{"items": {{"type": "array", "items": {{"type": "object", "properties": {{"title": {{"type": "string"}}, "rating": {{"type": "number"}}}}, "required": ["title", "rating"]}}}}}}

BAD: {{"items": {{"type": "array"}}}} (no items schema!)
BAD: {{"items": {{"type": "array", "description": "..."}}}} (description without items schema!)

If the execute script returns an array, ALWAYS analyze what properties each array element has and specify them fully."""
                        
                        # Call LLM
                        from browser_use.llm.messages import SystemMessage, UserMessage
                        messages = [
                            SystemMessage(content="You are a JSON generator. Output ONLY valid JSON, no markdown, no explanation."),
                            UserMessage(content=schema_prompt)
                        ]
                        
                        response = await agent_executor.llm.ainvoke(messages)
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
                
                # AUTO-GENERATE: pre_tools dependencies (concurrent with schema generation)
                logger.info(f"Analyzing pre_tools dependencies for {name}...")
                pre_tools = {}
                
                try:
                    # Get all other tools with their output schemas
                    other_tools = [
                        t for t in agent_executor.agent.tools 
                        if t.tool_executor_type == ToolExecutorType.SMCP and t.name != name
                    ]
                    
                    if other_tools and input_schema.get("properties"):
                        # Build prompt with other tools' output schemas
                        tools_context = []
                        for other_tool in other_tools:
                            if isinstance(other_tool, SMCPTool) and other_tool.output_schema:
                                output_props = other_tool.output_schema.get("properties", {})
                                if output_props:
                                    # Format output schema with descriptions
                                    prop_lines = []
                                    for prop_name, prop_info in output_props.items():
                                        prop_type = prop_info.get("type", "any")
                                        prop_desc = prop_info.get("description", "")
                                        if prop_desc:
                                            prop_lines.append(f"  - {prop_name} ({prop_type}): {prop_desc}")
                                        else:
                                            prop_lines.append(f"  - {prop_name} ({prop_type})")
                                    
                                    tools_context.append(f"{other_tool.name}:")
                                    tools_context.append(f"  Description: {other_tool.description}")
                                    tools_context.append(f"  Returns:")
                                    tools_context.extend(prop_lines)
                        
                        if tools_context:
                            pretools_prompt = f"""Analyze if this tool needs to call other tools first to get valid input parameters.

Tool being analyzed: {name}
Input parameters:
{json.dumps(input_schema.get("properties", {}), indent=2)}

Available tools that could provide data:
{chr(10).join(tools_context)}

For each input parameter, determine if any available tool's output would be needed to select a valid value.
Example: If parameter is "employee" (employee name), and get_employee_options returns list of valid employee names, then employee needs get_employee_options.

Respond with valid JSON only:
{{
  "pre_tools": {{
    "param_name": ["tool1", "tool2"],
    "another_param": ["tool3"]
  }}
}}

If no dependencies needed, respond with: {{"pre_tools": {{}}}}

ONLY include dependencies where the parameter value MUST be chosen from output of another tool.
Do NOT include dependencies for simple strings, numbers, or booleans that user provides directly."""

                            # Call LLM for pre_tools analysis
                            from browser_use.llm.messages import SystemMessage, UserMessage
                            pretools_messages = [
                                SystemMessage(content="You are a JSON generator analyzing tool dependencies. Output ONLY valid JSON."),
                                UserMessage(content=pretools_prompt)
                            ]
                            
                            pretools_response = await agent_executor.llm.ainvoke(pretools_messages)
                            pretools_text = pretools_response.completion.strip()
                            
                            # Extract JSON from markdown if needed
                            import re
                            pretools_json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', pretools_text, re.DOTALL)
                            if pretools_json_match:
                                pretools_text = pretools_json_match.group(1)
                            
                            # Parse pre_tools JSON
                            pretools_data = json.loads(pretools_text)
                            pre_tools = pretools_data.get("pre_tools", {})
                            
                            if pre_tools:
                                logger.info(f"Generated pre_tools: {pre_tools}")
                            else:
                                logger.info(f"No pre_tools dependencies identified")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate pre_tools (will use empty): {e}")
                    pre_tools = {}

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
                    is_completed=check,
                    execute=execute,
                    pre_path=pre_path,
                    pre=preconditions,
                    post=postconditions,
                    type=SMCPToolType(type_str),
                    pre_tools=pre_tools
                )
                
                # Remove existing tool with same name if it exists
                existing = agent_executor.agent.get_tool(smcp_tool.name)
                if existing:
                    agent_executor.agent.remove_tool(smcp_tool.name)
                    if smcp_tool.name in agent_executor._dynamic_tools:
                        del agent_executor._dynamic_tools[smcp_tool.name]
                    if smcp_tool.name in agent_executor._registered_tool_names:
                        agent_executor._registered_tool_names.remove(smcp_tool.name)
                    logger.info(f"Overriding existing tool: {smcp_tool.name}")
                
                # Add new tool to agent
                agent_executor.agent.add_tool(smcp_tool)
                
                # Register with browser-use
                add_smcp_tool(agent_executor, smcp_tool)
                
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
        @agent_executor.tools.action(description=tool.description)
        async def remove_smcp_tool(name: str) -> ActionResult:
            """
            Remove an SMCP tool by name.
            """
            success = agent_executor.agent.remove_tool(name)
            if success:
                if name in agent_executor._dynamic_tools:
                    del agent_executor._dynamic_tools[name]
                if name in agent_executor._registered_tool_names:
                    agent_executor._registered_tool_names.remove(name)
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
                
                if t.is_completed:
                    tool_info.append("  IS_COMPLETED (validates output; returns true or [false, reason]):")
                    tool_info.append("  ```javascript")
                    tool_info.append(f"  {t.is_completed}")
                    tool_info.append("  ```")
                    tool_info.append("")
                else:
                    tool_info.append("  IS_COMPLETED: (not defined)")
                    tool_info.append("")
            
            return tool_info
        
        @agent_executor.tools.action(description=tool.description)
        async def list_smcp_tools(get_code_for: str = "") -> ActionResult:
            """
            List SMCP tools that match the current page URL.
            
            Only shows tools whose pre_path glob pattern matches the current URL.
            This ensures you only see tools relevant to the current page.
            
            Args:
                get_code_for: Optional tool name to get detailed code for (includes is_ready/execute/is_completed scripts)
            """
            import fnmatch
            
            # Get current URL
            try:
                current_url = await agent_executor.browser.get_current_page_url()
            except Exception as e:
                logger.warning(f"Could not get current URL: {e}")
                current_url = ""
            
            # Get all SMCP tools
            smcp_tools_data = [
                t for t in agent_executor.agent.tools
                if t.tool_executor_type == ToolExecutorType.SMCP
                and fnmatch.fnmatch(current_url, t.pre_path)
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
    
    elif tool.name in ("ask_human_cli", "ask_human"):
        # Prefer DBOS-backed human-in-the-loop when available (AgentExecutor provides ask_human_callback)
        try:
            if getattr(agent_executor, 'ask_human_callback', None):
                from .tools_hitl import create_ask_human_tool

                # Create DBOS-backed ask_human callable using the callback
                # The callback has cycle_id and user_email in its closure from agent_workflow
                dbos_ask = create_ask_human_tool(
                    agent_executor.ask_human_callback
                )

                @agent_executor.tools.action(description="Ask for human assistance")
                async def ask_human_action(prompt: str) -> ActionResult:
                    try:
                        response = await dbos_ask(prompt)
                        return ActionResult(
                            is_done=False,
                            extracted_content=f"Human responded: {response}",
                            include_in_memory=True
                        )
                    except Exception as e:
                        logger.error(f"DBOS ask_human failed: {e}")
                        import traceback
                        traceback.print_exc()
                        return ActionResult(error=f"Failed to get human input: {str(e)}")

                ask_human_action.__name__ = tool.name
            else:
                # Fallback to CLI-based ask_human (for CLI-only context without DBOS)
                logger.warning(f"No ask_human_callback available - falling back to CLI-based ask_human_cli")
                @agent_executor.tools.action(description="Ask for human assistance when stuck, unauthenticated, or task is ambiguous")
                async def ask_human_action(prompt: str) -> ActionResult:
                    try:
                        # Import the actual implementation
                        from .tools_hitl import ask_human_cli
                        response = await ask_human_cli(prompt)
                        return ActionResult(
                            is_done=False,
                            extracted_content=f"Human responded: {response}",
                            include_in_memory=True
                        )
                    except Exception as e:
                        logger.error(f"ask_human failed: {e}")
                        import traceback
                        traceback.print_exc()
                        return ActionResult(error=f"Failed to get human input: {str(e)}")

                ask_human_action.__name__ = tool.name
        except Exception as e:
            logger.error(f"Failed to register {tool.name} tool: {e}")
            # As a last resort, register the CLI variant to avoid missing tool
            @agent_executor.tools.action(description="Ask for human assistance (fallback CLI)")
            async def ask_human_action(prompt: str) -> ActionResult:
                try:
                    from .tools_hitl import ask_human_cli
                    response = await ask_human_cli(prompt)
                    return ActionResult(is_done=False, extracted_content=f"Human responded: {response}", include_in_memory=True)
                except Exception as e2:
                    logger.error(f"Fallback {tool.name} failed: {e2}")
                    return ActionResult(error=f"Failed to get human input: {str(e2)}")

            ask_human_action.__name__ = tool.name
    
    elif tool.name == "ask_html":
        @agent_executor.tools.action(description="Ask a question about the page HTML to get guidance on selectors, structure, or data attributes for creating SMCP tools")
        async def ask_html(query: str, print_html: bool = False) -> ActionResult:
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
                page = await agent_executor.browser.get_current_page()
                cdp_session = await agent_executor.browser.get_or_create_cdp_session()
                
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
                current_url = await agent_executor.browser.get_current_page_url()
                
                original_length = len(page_html)
                logger.info(f"Extracted {original_length:,} chars of raw HTML from {current_url}")
                
                # Basic cleaning only (remove head/script/style, NO truncation)
                page_html = _clean_html_for_analysis(page_html, max_length=None)
                
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
- ONLY return selectors that exist on the current page and URL. If asked but not present, say so.
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
                    from .llm_factory import LLMFactory
                    
                    model = os.getenv("BLASTAI_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1")
                    provider = os.getenv("BLASTAI_PROVIDER")
                    
                    # Create a dedicated LLM instance for this ask_html call
                    ask_html_llm = LLMFactory.create_llm(
                        model_name=model,
                        provider=provider,
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
                            current_html = _reduce_html_randomly(current_html, target_ratio=0.5)
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

def _clean_html_for_analysis(html: str, max_length: int = None) -> str:
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

def _reduce_html_randomly(html: str, target_ratio: float = 0.5) -> str:
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