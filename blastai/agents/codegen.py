"""
Code generation module for agent code execution mode.

Handles code generation with validation and definition code inclusion.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import SystemMessage, AssistantMessage, UserMessage, BaseMessage

from .models import ToolExecutorType, SMCPTool, SMCPToolType
from .codecheck import check_code_candidate, enhance_validation_error
from .codecost import compute_code_cost
from .schema_utils import snake_to_pascal, json_type_to_python, generate_nested_pydantic_classes
from .codefix import apply_code_fixes
from .llm_streaming import stream_llm_call, StreamingTiming

logger = logging.getLogger(__name__)


@dataclass
class LLMTiming:
    """Detailed timing information for a single LLM call."""
    total_seconds: float
    time_to_first_token: Optional[float] = None  # Time until first token arrives
    tokens_per_second: Optional[float] = None     # Average token generation speed
    total_tokens: Optional[int] = None            # Total tokens generated


# Example code showing realistic patterns
# TODO: add example of logging in via ai_exec when on login page
EXAMPLE_LABEL_WITH_STATE = "E4: ordering tool calls based on current STATE and tool preconditions/postconditions, passing previous tools results into next tool call either by direct access or ai_eval, using ai_eval to match a user-provided term to an option in a list"
EXAMPLE_CODE = """
E1: immediate return
```python
result = "The response to the user's task can be immediately returned"
```

E2: call single tool with handling of empty tool result
```python
x = await tool_a(42)
result = f"It is {x.items[0].field1} and {x.items[0].field2}" if x.items else "<user-friendly message about no results>"
```

E3: result with ai_eval
```python
x = await tool_b(param="value")
response = await ai_eval("Human-readable summary of {data}", data=x.content)
```

E4: multiple tool calls with ai_eval for matching user terms to options
```python
await tool_c()
x = await tool_d(id=123)
y = await tool_e(p1=x.data)
closest_name = await ai_eval("Name in {options} closest to 'the titanic'", options=y.options)
z = await tool_f(item_name=closest_name)
result = await ai_eval("Response about {info}", info=z.info)
```

E5: control flow with loops and conditionals
```python
await tool_c()
x = await tool_e(123)
results = [await tool_f(item_name=item.name) for item in x.items if item.value > 3]
result = await ai_eval("Summary of {results}", results=results)
```

{COMMENTED_OUT_AI_EXEC_EXAMPLE}
"""

# TODO: Uncomment for production. Commented out for benchmarking to prevent LLMs from using ai_exec.
COMMENTED_OUT_AI_EXEC_EXAMPLE = """
E6: ai_exec for sub-tasks with optional structured output
```python
# Without output_schema (returns dict with .items property)
x = await ai_exec("Complete the login form")

# With output_schema (provide JSON schema, NOT a Python dict)
schema = {"type": "object", "properties": {"success": {"type": "boolean"}}, "required": ["success"]}
x = await ai_exec("Login to the application", output_schema=schema)
if x.get("success"):
    result = "Login successful"
```
"""

RULES_BASE = """# - Do not generate comments
# - If using a user-provided term (e.g. a name to filter by/search for/pass to a tool), use ai_eval to determine which option it most closely matches.
# - When calling ai_eval, always specify (1) the desired output format, default: "Markdown/text" unless task says a specific format (2) how descriptive, default: if evaluating intermediate output: "only", if evaluating final output: "very descriptive using full sentences"."""

@dataclass
class CodeCandidate:
    """A generated code candidate with detailed timing breakdown."""
    code: str
    rank: int  # Lower is better
    is_valid: bool
    validation_error: Optional[str] = None
    iterations_used: int = 0  # Number of iterations used to get valid code (0 if never valid)
    
    # Detailed timing breakdown
    total_time: float = 0.0
    llm_time: float = 0.0          # Total time spent in LLM calls
    validation_time: float = 0.0   # Total time spent in validation
    fix_time: float = 0.0          # Total time spent applying fixes
    llm_timings: List[LLMTiming] = field(default_factory=list)  # Per-iteration LLM timings
    
    # Iteration failure tracking
    total_iterations: int = 0      # Total iterations attempted (including failures)
    failed_iterations: int = 0     # Number of iterations that failed validation
    # Per-iteration error tracking (order matches iteration execution)
    iteration_errors: List[str] = field(default_factory=list)
    iteration_error_types: List[str] = field(default_factory=list)  # syntax|types|ordering|pre-tools|none|unknown


class CodeGenerator:
    """
    Generates Python code for agent tasks with definition code and validation.
    
    Includes tool definitions as executable Python code in the generation,
    using XML-based prompting similar to derive_synthesis_agent.
    
    Supports multiple LLM instances for parallel generation with different models.
    """
    
    def __init__(
        self,
        agent,
        llm: Optional[BaseChatModel] = None,
        llms: Optional[List[BaseChatModel]] = None,
        state_aware: bool = False,
        num_candidates: int = 1,
        max_iterations: int = 3,
        accept_cost_threshold: Optional[float] = None,
        min_candidates_for_comparison: int = 1,
        compare_cost_threshold: Optional[float] = None,
        timezone: str = "UTC",
        debug_print_prompt: bool = False
    ):
        """
        Initialize code generator.
        
        Args:
            agent: Agent instance with tools
            llm: Single LLM for code generation (deprecated - use llms instead)
            llms: List of LLM instances for parallel generation. If None, uses single llm.
            state_aware: Whether to include state preconditions/postconditions
            num_candidates: Number of candidates when using single llm (ignored if llms provided)
            max_iterations: Maximum iterations per candidate generation (default 3)
            accept_cost_threshold: If set, immediately accept candidate below this cost (default None)
            min_candidates_for_comparison: Minimum candidates before comparison (default 1)
            compare_cost_threshold: If set, compare and return when we have min_candidates below this threshold (default None)
            timezone: Timezone string in IANA format for current date/time context (default 'UTC')
            debug_print_prompt: If True, print the full codegen prompt for first iteration (default False)
        """
        self.agent = agent
        self.state_aware = state_aware
        self.max_iterations = max_iterations
        self.accept_cost_threshold = accept_cost_threshold
        self.min_candidates_for_comparison = min_candidates_for_comparison
        self.compare_cost_threshold = compare_cost_threshold
        self.timezone = timezone
        self.debug_print_prompt = debug_print_prompt
        
        # Configure LLMs for parallel generation
        if llms:
            # Use provided list of LLMs (each may be different model)
            self.llms = llms
            self.num_candidates = len(llms)
        elif llm:
            # Use single LLM repeated num_candidates times (backwards compatible)
            self.llms = [llm] * num_candidates
            self.num_candidates = num_candidates
        else:
            raise ValueError("Must provide either llm or llms parameter")
        
        # Cache for definition code CFG (built once, reused across iterations)
        self._definition_code_cache: Optional[str] = None
        self._definition_cfg_cache: Optional[tuple] = None  # (start_block, blocks)
    
    def _build_definition_code(self) -> str:
        """
        Build definition code showing tool signatures with precondition/postcondition assertions.
        
        This code is part of the Python code block that gets generated, showing the
        developer what tools are available and their contracts.
        
        Returns definitions as Python code with:
        - STATE dictionary for tracking page, selectedRestaurant, selectedCategory, current_url
        - Pydantic models for input parameters (matching JSON schemas)
        - Proper typed function signatures
        - Assert statements for preconditions (checking STATE)
        - Function implementation placeholder
        - Assert statements for postconditions (checking STATE after update)
        """
        lines = []
        
        # Import Pydantic for type definitions and other required modules
        lines.append("from pydantic import BaseModel, Field")
        lines.append("from typing import Optional, List, Dict, Any, Union, Literal")
        lines.append("import re")
        lines.append("")
        
        # Initialize STATE dictionary dynamically from all tools' pre/post variables
        # Only include STATE if state_aware is True
        if self.state_aware:
            # Collect all state keys referenced in pre/post and URL patterns
            state_keys = set()
            for t in self.agent.tools:
                if t.tool_executor_type == ToolExecutorType.SMCP:
                    if getattr(t, "pre", None) and isinstance(t.pre, dict):
                        state_keys.update(t.pre.keys())
                    if getattr(t, "post", None) and isinstance(t.post, dict):
                        state_keys.update(t.post.keys())
                    # Note: We no longer track current_url in STATE - use get_url() instead

            # Make deterministic ordering
            state_keys_list = sorted(list(state_keys))
            
            # Define STATE dictionary (provided at runtime)
            if state_keys_list:
                state_init = "{" + ", ".join([f'"{k}": None' for k in state_keys_list]) + "}"
                lines.append(f"STATE: Dict[str, Any] = {state_init}")
                lines.append("")
        
        # Track which tools have been called (for pre_tools enforcement) - only if state_aware
        if self.state_aware:
            lines.append("# Track which tools have been called")
            lines.append("TOOLS_CALLED: set[str] = set()")
            lines.append("")
        
        # Add SMCP tool definitions with precondition/postcondition assertions
        for tool in self.agent.tools:
            if tool.tool_executor_type == ToolExecutorType.SMCP:
                smcp_tool = tool
                assert isinstance(smcp_tool, SMCPTool)
                
                # Skip OBSERVE tools - they're called automatically, not by user code
                if smcp_tool.type == SMCPToolType.OBSERVE:
                    continue
                
                # Generate Pydantic model for input parameters if they exist
                has_params = smcp_tool.input_schema and smcp_tool.input_schema.get("properties")
                if has_params:
                    model_name = f"{snake_to_pascal(smcp_tool.name)}Input"
                    lines.append(f"class {model_name}(BaseModel):")
                    
                    properties = smcp_tool.input_schema["properties"]
                    required = smcp_tool.input_schema.get("required", [])
                    
                    for param_name, param_schema in properties.items():
                        param_type = json_type_to_python(param_schema.get("type", "str"))
                        is_required = param_name in required
                        param_desc = param_schema.get("description", "")
                        
                        if is_required:
                            lines.append(f'    {param_name}: {param_type} = Field(..., description="{param_desc}")')
                        else:
                            lines.append(f'    {param_name}: Optional[{param_type}] = Field(None, description="{param_desc}")')
                    
                    lines.append("")
                
                # Generate Pydantic model for output schema if it exists
                has_output = smcp_tool.output_schema and smcp_tool.output_schema.get("properties")
                output_model_name = None
                if has_output:
                    output_model_name = f"{snake_to_pascal(smcp_tool.name)}Output"
                    # Use recursive generation to handle nested schemas
                    # This will generate all nested classes and return the root class name
                    temp_lines = []
                    generate_nested_pydantic_classes(smcp_tool.output_schema, output_model_name, temp_lines)
                    lines.extend(temp_lines)
                
                # Generate function signature with proper types (not **kwargs)
                # Return the Pydantic model for type safety and attribute access
                return_type = output_model_name if output_model_name else "Dict[str, Any]"
                
                if has_params:
                    # Build typed parameters
                    properties = smcp_tool.input_schema["properties"]
                    required = smcp_tool.input_schema.get("required", [])
                    params = []
                    # Ensure required parameters come first in signature to avoid
                    # "non-default argument follows default argument" syntax errors.
                    # To make this robust against arbitrary ordering in the schema
                    # and missing/None `required`, iterate properties in their
                    # defined order and partition them into required vs optional.
                    properties_items = list(properties.items())
                    required_set = set(required or [])

                    # First, add required params in the properties order
                    for param_name, param_schema in properties_items:
                        if param_name in required_set:
                            param_type = json_type_to_python(param_schema.get("type", "str"))
                            params.append(f"{param_name}: {param_type}")

                    # Then add optional params (with defaults) in the properties order
                    for param_name, param_schema in properties_items:
                        if param_name in required_set:
                            continue
                        param_type = json_type_to_python(param_schema.get("type", "str"))
                        params.append(f"{param_name}: Optional[{param_type}] = None")
                    
                    params_str = ", ".join(params)
                    lines.append(f"async def {smcp_tool.name}({params_str}) -> {return_type}:")
                else:
                    lines.append(f"async def {smcp_tool.name}() -> {return_type}:")
                
                lines.append(f'    """')
                lines.append(f'    {smcp_tool.description}')
                lines.append(f'    """')
                
                # PRE_TOOLS ASSERTIONS (check that required tools have been called) - only if state_aware
                if self.state_aware and hasattr(smcp_tool, 'pre_tools') and smcp_tool.pre_tools:
                    # pre_tools is Dict[str, List[str]] mapping param names to required tool names
                    # Flatten to get all required tools
                    required_tools = set()
                    for param_name, tool_names in smcp_tool.pre_tools.items():
                        required_tools.update(tool_names)
                    
                    if required_tools:
                        for required_tool in sorted(required_tools):
                            lines.append(f'    assert "{required_tool}" in TOOLS_CALLED, "Must call {required_tool} before {smcp_tool.name}"')
                
                # PRECONDITIONS (assert statements checking STATE)
                if self.state_aware:
                    # URL pattern precondition - use get_url() to get current URL dynamically
                    if smcp_tool.pre_path and smcp_tool.pre_path != "":
                        url_pattern = smcp_tool.pre_path.replace("*", ".*")
                        lines.append(f'    current_url = await get_url()')
                        lines.append(f'    assert re.match(r"{url_pattern}", current_url)')
                    
                    # State preconditions with pattern support
                    if smcp_tool.pre and isinstance(smcp_tool.pre, dict):
                        for key, value in smcp_tool.pre.items():
                            # Skip None values - they don't impose constraints
                            if value is None:
                                continue
                            assertion = self._generate_assertion(key, value, "precondition", state_check=True)
                            if assertion:
                                lines.append(f'    {assertion}')
                
                # IMPLEMENTATION (stub - runtime provides actual implementation)
                lines.append(f'    ...')
                
                # POSTCONDITIONS (assert statements checking STATE after update)
                if self.state_aware and smcp_tool.post and isinstance(smcp_tool.post, dict):
                    for key, value in smcp_tool.post.items():
                        # Skip None values - they don't impose constraints
                        if value is None:
                            continue
                        
                        # For postconditions, check STATE after update
                        if isinstance(value, str) and value.startswith("$"):
                            # Reference to input parameter
                            param_ref = value[1:]  # Remove $
                            lines.append(f'    assert STATE["{key}"] == {param_ref}')
                        else:
                            assertion = self._generate_assertion(key, value, "postcondition", state_check=True)
                            if assertion:
                                lines.append(f'    {assertion}')
                
                # Track that this tool was called (conceptually - runtime handles this) - only if state_aware
                if self.state_aware:
                    lines.append(f'    TOOLS_CALLED.add("{smcp_tool.name}")')
                    lines.append('')
        
        # Add core tool definitions (simplified signatures)
        for tool in self.agent.tools:
            if tool.tool_executor_type == ToolExecutorType.CORE:
                if tool.name == "update_smcp_tool":
                    lines.append('async def update_smcp_tool(name: str, type: str, execute: str, is_ready: str, is_completed: str, preconditions: List[Dict[str, str]], postconditions: List[Dict[str, str]], input_parameters: List[str]) -> Dict[str, Any]:')
                elif tool.name == "remove_smcp_tool":
                    lines.append('async def remove_smcp_tool(name: str) -> Dict[str, Any]:')
                elif tool.name == "list_smcp_tools":
                    lines.append('async def list_smcp_tools() -> Dict[str, Any]:')
                elif tool.name in ("ask_human", "ask_human_cli"):
                    lines.append(f'async def {tool.name}(question: str) -> str:')
                else:
                    lines.append(f'async def {tool.name}(**kwargs) -> Dict[str, Any]:')
                
                lines.append(f'    """')
                # Include the tool's description and explicit note about STATE handling
                if tool.name in ("ask_human", "ask_human_cli"):
                    # make it clear to generated code authors that STATE is updated by the runtime
                    lines.append(f'    {tool.description} Updates STATE automatically.')
                else:
                    lines.append(f'    {tool.description}')
                lines.append(f'    """')
                lines.append(f'    # ... implementation omitted ...')
                lines.append(f'    pass')
                lines.append('')
        
        # Add utility functions (stub implementations for type checking)
        lines.append('async def get_url() -> str:')
        lines.append('    """Get the current URL from the browser."""')
        lines.append('    raise NotImplementedError("Runtime implementation")')
        lines.append('')
        # NOTE: goto(url) is intentionally NOT exposed to codegen.
        # Using goto(url) bypasses state machine validation since we can't statically verify
        # that the dynamic URL will satisfy the preconditions for subsequent tool calls.
        # LLMs should use the SMCP navigation tools (e.g., goto_restaurants_list) which have
        # well-defined pre/post conditions that can be validated.
        
        # TODO: Uncomment ai_exec for production use. Currently commented out for benchmarking
        # to avoid LLMs using it instead of proper SMCP navigation tools.
        # lines.append('async def ai_exec(subtask: str, output_schema: Optional[Dict[str, Any]] = None) -> Any:')
        # lines.append('    """')
        # lines.append('    Execute an AI agent to complete a given subtask.')
        # lines.append('    ')
        # lines.append('    Args:')
        # lines.append('        subtask: Description of the subtask for the AI agent to complete')
        # lines.append('        output_schema: Optional JSON schema for structured output.')
        # lines.append('                      If omitted, returns string result.')
        # lines.append('    ')
        # lines.append('    Returns:')
        # lines.append('        - Pydantic model (if output_schema provided)')
        # lines.append('        - String (if output_schema omitted)')
        # lines.append('    """')
        # lines.append('    raise NotImplementedError("Runtime implementation")')
        # lines.append('')
        lines.append('async def ai_eval(expr: str, **kwargs) -> str:')
        lines.append('    """')
        lines.append('    Evaluate an expression by asking AI.')
        lines.append('    Template format: ai_eval("Result from {variable}", variable=value)')
        lines.append('    The template string uses {variable} placeholders that get replaced with kwargs.')
        lines.append('    """')
        lines.append('    raise NotImplementedError("Runtime implementation")')
        lines.append('')
        
        # NOTE: The definition code does NOT include initial STATE or URL assertions
        # Those are added separately in the prompt generation (see _generate_candidate)
        # This keeps the definition code reusable across different initial states
        
        # NOTE: core tool stubs, including ask_human, are generated above in the core-tool loop.
        # Do not add duplicate ask_human stubs here.
        
        return '\n'.join(lines)
    
    def _pattern_description(self, value: Any) -> str:
        """Generate human-readable description of a state pattern."""
        if value is None:
            return "must be null"
        if not isinstance(value, str):
            return f"== {json.dumps(value)}"
        if value == "*":
            return "can be any non-null value"
        elif value == "":
            return "must be null/empty"
        elif "|" in value:
            options = value.split("|")
            return f"must be one of: {', '.join(options)}"
        elif value.startswith("$"):
            return f"must match parameter {value[1:]}"
        else:
            return f"== {json.dumps(value)}"
    
    def _generate_assertion(self, var_name: str, pattern: Any, assertion_type: str, state_check: bool = False) -> Optional[str]:
        """
        Generate Python assertion code for a state pattern.
        
        Args:
            var_name: Variable name to check
            pattern: Pattern to match (*, |, $param, "", None, or concrete value)
            assertion_type: "precondition" or "postcondition"
            state_check: If True, check STATE[var_name]; otherwise check var_name directly
        
        Returns:
            Python assertion code or None if no assertion needed
        """
        var_ref = f'STATE["{var_name}"]' if state_check else var_name
        
        # Handle None/null explicitly
        if pattern is None:
            return f'assert {var_ref} is None'
        
        # Handle non-string patterns (numbers, booleans, etc.)
        if not isinstance(pattern, str):
            return f'assert {var_ref} == {json.dumps(pattern)}'
        
        # Handle string patterns
        if pattern == "*":
            # Must be non-null
            return f'assert {var_ref} is not None'
        elif pattern == "":
            # Must be null/empty
            return f'assert not {var_ref}'
        elif "|" in pattern:
            # Must be one of the options
            options = pattern.split("|")
            options_json = json.dumps(options)
            return f'assert {var_ref} in {options_json}'
        elif pattern.startswith("$"):
            # Handled separately in caller (needs context of params)
            return None
        else:
            # Concrete value
            return f'assert {var_ref} == {json.dumps(pattern)}'
    
    def _check_candidate(self, code: str, initial_state: Optional[Dict[str, Any]] = None) -> tuple[bool, Optional[str]]:
        """
        Check if generated code is valid.
        
        Args:
            code: Generated Python code
            initial_state: Optional initial state for CFG validation
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get definition code for mypy type checking
        definition_code = self._build_definition_code()
        return check_code_candidate(
            code, 
            agent=self.agent, 
            initial_state=initial_state,
            definition_code=definition_code
        )
    
    def _compute_candidate_cost(self, code: str) -> float:
        """
        Compute cost/score for a code candidate.
        Lower is better.
        
        For now, uses code length as a simple heuristic.
        
        Args:
            code: Generated Python code
        
        Returns:
            Cost score (lower is better)
        """
        return compute_code_cost(code)
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from markdown code blocks."""
        # Primary: fenced code blocks
        patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```py\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                extracted = matches[0].strip()
                # Guard against model echoing examples without user code marker
                if extracted:
                    return extracted
        # Fallback heuristics: attempt to salvage code when no fences present
        # 1. If response contains lines with 'await ' or assignment to result
        lines = [l.rstrip() for l in response.splitlines()]
        code_like = []
        in_code = False
        for l in lines:
            stripped = l.strip()
            if not stripped:
                continue
            # Start collecting when we see common Python starters
            if not in_code and (
                stripped.startswith(('result', 'await ', 'for ', 'if ', 'while ', 'import ', 'from ', 'def ', 'class ', '#'))
                or 'await ' in stripped
            ):
                in_code = True
            if in_code:
                code_like.append(l)
        fallback = '\n'.join(code_like).strip()
        if fallback:
            # Remove trailing triple backticks if model left them open
            fallback = re.sub(r'```+$', '', fallback).strip()
            return fallback if fallback else None
        return None
    
    def _generate_rules(self, initial_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate dynamic RULES based on current state.
        
        Checks if any tools' preconditions are satisfied by the current state.
        If not, prompts to use ai_exec or ask_human.
        
        Args:
            initial_state: Current abstract state
            
        Returns:
            Formatted RULES string
        """
        rules = [RULES_BASE]
        
        # Check if we have state-aware tools and a current state
        if self.state_aware and initial_state:
            from .codecheck import find_callable_tools
            
            # Find which tools are currently callable
            callable_tools = find_callable_tools(initial_state, self.agent.tools)
            
            # Check if ask_human is available
            has_ask_human = any(
                hasattr(t, 'name') and t.name in ('ask_human', 'ask_human_cli')
                for t in self.agent.tools
            )
            
            # If no tools are callable, mandate use of ai_exec/ask_human
            if not callable_tools:
                fallback_options = []
                fallback_options.append("ai_exec")
                if has_ask_human:
                    fallback_options.append("ask_human")
                rules.append(f"# - You must use {' or '.join(fallback_options)} given no tool's precondition satisfies the current state.")
            else:
                # Tools are available, so standard fallback rule
                fallback_options = []
                if has_ask_human:
                    fallback_options.append("ask_human")
                fallback_options.append("ai_exec")
                if fallback_options:
                    rules.append(f"# - Use {' or '.join(fallback_options)} if no tool's precondition is satisfied.")
        else:
            # Non-state-aware or no initial state - use standard rule
            has_ask_human = any(
                hasattr(t, 'name') and t.name in ('ask_human', 'ask_human_cli')
                for t in self.agent.tools
            )
            fallback_options = []
            if has_ask_human:
                fallback_options.append("ask_human")
            fallback_options.append("ai_exec")
            if fallback_options:
                rules.append(f"# - Use {' or '.join(fallback_options)} if no tool's precondition is satisfied.")
        rules.append("# - Do not access or modify STATE directly. Do not call get_url()")
        
        return '\n'.join(rules)
    
    async def generate_code(
        self,
        task: str,
        history: List[BaseMessage],
        error: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        current_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate Python code to complete the task.
        
        Uses parallel candidate generation with iterative refinement.
        Supports early acceptance and comparison thresholds.
        
        Args:
            task: Task description
            history: Conversation history as list of BaseMessage (UserMessage, AssistantMessage, SystemMessage)
            error: Optional error from previous attempt (for first iteration)
            initial_state: Optional initial STATE values to include in generated code
            current_url: Optional current URL to provide context to code generator
        
        Returns:
            Generated code or None if generation failed
        """
        if self.num_candidates <= 1:
            # Single candidate generation
            candidate = await self._generate_candidate(task, history, error, initial_state, current_url)
            return candidate.code if candidate and candidate.is_valid else None
        
        # Parallel candidate generation
        logger.info(f"Generating {self.num_candidates} candidates in parallel")
        
        # Launch parallel candidate generation tasks
        tasks = [
            self._generate_candidate(task, history, error, initial_state, current_url, candidate_num=i)
            for i in range(self.num_candidates)
        ]
        
        # Collect valid candidates as they complete
        valid_candidates: List[CodeCandidate] = []
        
        # Process candidates as they complete
        for completed_task in asyncio.as_completed(tasks):
            candidate = await completed_task
            
            if candidate and candidate.is_valid:
                # Check for immediate acceptance based on cost threshold
                if self.accept_cost_threshold is not None and candidate.rank <= self.accept_cost_threshold:
                    logger.info(f"Immediately accepting candidate with cost {candidate.rank} (below threshold {self.accept_cost_threshold})")
                    logger.info(f"✓ SELECTED CODE TO EXECUTE:\n```python\n{candidate.code}\n```\n{'='*80}")
                    return candidate.code
                
                # If no cost threshold configured, use first valid candidate (fast convergence)
                if self.accept_cost_threshold is None and self.compare_cost_threshold is None:
                    logger.info(f"Using first valid candidate (cost: {candidate.rank})")
                    logger.info(f"✓ SELECTED CODE TO EXECUTE:\n```python\n{candidate.code}\n```\n{'='*80}")
                    return candidate.code
                
                valid_candidates.append(candidate)
                
                # Check for early comparison
                if self.compare_cost_threshold is not None and len(valid_candidates) >= self.min_candidates_for_comparison:
                    # Check if all collected candidates are below comparison threshold
                    below_threshold = [c for c in valid_candidates if c.rank <= self.compare_cost_threshold]
                    if len(below_threshold) >= self.min_candidates_for_comparison:
                        logger.info(f"Early comparison: {len(below_threshold)} candidates below threshold {self.compare_cost_threshold}")
                        best = min(below_threshold, key=lambda c: c.rank)
                        logger.info(f"✓ SELECTED CODE TO EXECUTE (cost: {best.rank}):\n```python\n{best.code}\n```\n{'='*80}")
                        return best.code
        
        # All candidates completed - select best
        if not valid_candidates:
            logger.warning("No valid candidates generated")
            return None
        
        best = min(valid_candidates, key=lambda c: c.rank)
        logger.info(f"Selected best candidate with cost {best.rank} from {len(valid_candidates)} valid candidates")
        logger.info(f"✓ SELECTED CODE TO EXECUTE:\n```python\n{best.code}\n```\n{'='*80}")
        return best.code

    # Public wrapper to expose a single candidate with iteration metadata
    async def generate_candidate(
        self,
        task: str,
        history: List[BaseMessage],
        error: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        current_url: Optional[str] = None
    ) -> Optional[CodeCandidate]:
        """Generate and return a single CodeCandidate (with retries and metrics)."""
        return await self._generate_candidate(
            task=task,
            history=history,
            initial_error=error,
            initial_state=initial_state,
            current_url=current_url,
        )
    
    async def _generate_candidate(
        self,
        task: str,
        history: List[BaseMessage],
        initial_error: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        current_url: Optional[str] = None,
        candidate_num: int = 0
    ) -> Optional[CodeCandidate]:
        """
        Generate a single code candidate with iterative refinement.
        
        Maintains a list of chat messages and iteratively refines based on errors.
        The history parameter contains the broader conversation context from previous
        code execution attempts. Each iteration of refinement appends to a local
        message list.
        
        Tracks detailed timing for LLM calls, validation, and fixes.
        
        Args:
            task: Task description
            history: Conversation history from AgentExecutor (previous code execution attempts)
            initial_error: Optional error from previous attempt to retry
            initial_state: Optional initial STATE values to include in generated code
            current_url: Optional current URL to provide context to code generator
            candidate_num: Candidate number for logging (0 = log input, others = skip input logging)
        
        Returns:
            CodeCandidate with code, validity, cost, and timing metrics
        """
        candidate_start_time = time.time()
        total_llm_time = 0.0
        total_validation_time = 0.0
        total_fix_time = 0.0
        llm_timings: List[LLMTiming] = []
        total_iterations = 0
        failed_iterations = 0
        iteration_errors: List[str] = []
        iteration_error_types: List[str] = []
        
        # Build definition code (and cache CFG for validation performance)
        definition_code = self._build_definition_code()
        
        # Cache definition code CFG if not already cached
        if self._definition_code_cache != definition_code:
            self._definition_code_cache = definition_code
            self._definition_cfg_cache = None  # Invalidate cache
            logger.debug("Definition code changed, CFG cache invalidated")
        
        # Initialize message list for this candidate
        # Start with the broader conversation history from executor
        messages = []
        
        # Add system message for this code generation session
        messages.append(
            SystemMessage(content=f"""You are an expert in writing code that calls tools and programatically asks AI questions.
<examples>
Here are examples of good code to generate. Use them as reference but never copy them directly.
{EXAMPLE_CODE if not self.state_aware else EXAMPLE_CODE.replace("E4: multiple tool calls", EXAMPLE_LABEL_WITH_STATE + ":")}
</examples>
<instructions>
Generate completion of the given code block to implement the TASK.
If an error is reported, fix the previously generated code accordingly.
</instructions>
""")
        )

        if history:
            messages.extend(history)
        
        # Build initial prompt
        error = initial_error
        
        for iteration in range(self.max_iterations):
            total_iterations += 1
            
            # Build prompt with XML tags
            prompt_parts = []
            
            if error:
                prompt_parts.append(error)
            else:
                # Compute current date/time FIRST (NOT prefixed with #)
                from datetime import datetime
                import zoneinfo
                timestamp_str = None
                try:
                    tz = zoneinfo.ZoneInfo(self.timezone)
                    now = datetime.now(tz)
                    # Format: "Monday, November 4, 2025 3:30:00 PM PST"
                    day_of_week = now.strftime("%A")
                    date_part = now.strftime("%B %d, %Y")
                    time_part = now.strftime("%I:%M:%S %p")  # 12-hour format with AM/PM
                    tz_abbr = now.strftime("%Z")
                    timestamp_str = f"{day_of_week}, {date_part} {time_part} {tz_abbr}"
                except Exception as e:
                    logger.warning(f"Failed to get current time for timezone {self.timezone}: {e}")
                
                # TIMESTAMP AT TOP (before code block, NO # prefix)
                if timestamp_str:
                    prompt_parts.append(f"Now is {timestamp_str}")
                    prompt_parts.append("")
                
                # Add system prompt (agent description) ABOVE code block (NOT inside it)
                # The agent.description comes from build_complete_system_prompt
                # Do NOT comment it out - it's the actual system prompt
                if self.agent.description:
                    prompt_parts.append(self.agent.description)
                    prompt_parts.append("")
                
                prompt_parts.append("```python")
                prompt_parts.append(definition_code)
                prompt_parts.append("")
                
                # Initialize STATE with initial values if provided (only if state_aware)
                if self.state_aware and initial_state:
                    prompt_parts.append("# INITIAL STATE. DO NOT ACCESS OR MODIFY.")
                    state_init = ", ".join([f'"{k}": {repr(v)}' for k, v in initial_state.items()])
                    prompt_parts.append(f"STATE.update({{{state_init}}})")
                    prompt_parts.append("")
                
                # Show that we've already navigated to current URL using an assertion
                # NOTE: We use get_url() assertion instead of goto() to avoid exposing
                # the raw goto function to the LLM, which would generate unverifiable code
                if current_url:
                    prompt_parts.append(f'# Already on the page:')
                    prompt_parts.append(f'assert await get_url() == "{current_url}"')
                    prompt_parts.append("")
                
                prompt_parts.append("# RULES:")
                # Generate dynamic RULES based on current state
                rules = self._generate_rules(initial_state)
                prompt_parts.append(rules)
                
                # Format the user task - each line on separate line, commented
                prompt_parts.append("")
                prompt_parts.append("# TASK: Create a response for the following input from the user:")
                for line in task.split('\n'):
                    if line.strip():
                        prompt_parts.append(f"# {line}")
                prompt_parts.append("#")
                prompt_parts.append("# RESPOND WITH ONLY YOUR CODE AFTER THIS USING ABOVE DEFINITIONS")
                # Note: We deliberately leave the code block open - the LLM will close it
            
            prompt = '\n'.join(prompt_parts)
            
            # Add user message to conversation
            messages.append(UserMessage(content=prompt))
            
            # Debug: Print full prompt on first iteration if enabled
            if self.debug_print_prompt and iteration == 0:
                print("\n" + "="*100)
                print("CODEGEN PROMPT (First Iteration)")
                print("="*100)
                print("\nSYSTEM MESSAGE:")
                print("-"*100)
                print(messages[0].content)
                print("\nUSER MESSAGE:")
                print("-"*100)
                print(prompt)
                print("="*100 + "\n")
            
            # Generate code
            try:
                # Select LLM for this candidate
                llm = self.llms[candidate_num] if candidate_num < len(self.llms) else self.llms[0]
                
                # Only log input for first candidate (candidate_num == 0)
                if candidate_num == 0:
                    logger.info(f"\n{'='*80}\nCODEGEN ITERATION {iteration + 1}/{self.max_iterations}\n{'='*80}")
                    logger.info(f"LLM INPUT (last message):\n{prompt}\n{'-'*80}")
                
                # Call LLM with streaming and detailed timing
                completion, streaming_timing = await stream_llm_call(llm, messages)
                total_llm_time += streaming_timing.total_seconds
                
                # Convert StreamingTiming to LLMTiming for compatibility
                llm_timing = LLMTiming(
                    total_seconds=streaming_timing.total_seconds,
                    time_to_first_token=streaming_timing.time_to_first_token,
                    tokens_per_second=streaming_timing.tokens_per_second,
                    total_tokens=streaming_timing.total_tokens
                )
                llm_timings.append(llm_timing)
                
                # Only log LLM output for first candidate (candidate_num == 0)
                if candidate_num == 0:
                    # Log with detailed timing breakdown
                    # Handle None values gracefully in format strings
                    ttft_str = f"TTFT={streaming_timing.time_to_first_token:.2f}s, " if streaming_timing.time_to_first_token is not None else ""
                    gen_time = streaming_timing.generation_seconds if streaming_timing.generation_seconds is not None else 0.0
                    speed_str = f", {streaming_timing.tokens_per_second:.1f} tok/s" if streaming_timing.tokens_per_second is not None else ""
                    logger.info(f"LLM OUTPUT (took {streaming_timing.total_seconds:.2f}s, {ttft_str}Gen={gen_time:.2f}s{speed_str}):\n{completion}\n{'-'*80}")
                
                # Extract code from response
                code = self._extract_code_from_response(completion)
                
                if not code:
                    logger.error(f"No code block found in response (iteration {iteration + 1})")
                    error = "No code block found in LLM response. Please wrap your code in ```python ... ``` markers."
                    # Add assistant message to maintain conversation
                    messages.append(AssistantMessage(content=f"[Error: {error}]"))
                    continue
                
                # Apply automated fixes before validation
                fix_start = time.time()
                fixed_code, was_fixed = apply_code_fixes(code, tools=self.agent.tools)
                fix_elapsed = time.time() - fix_start
                total_fix_time += fix_elapsed
                
                if was_fixed:
                    logger.info(f"Applied automated fixes to code (took {fix_elapsed:.3f}s)")
                    code = fixed_code
                
                # Check candidate validity with initial_state for CFG validation
                validation_start = time.time()
                is_valid, validation_error = self._check_candidate(code, initial_state)
                validation_elapsed = time.time() - validation_start
                total_validation_time += validation_elapsed
                
                # Enhance validation error with suggestions if it failed
                if not is_valid and validation_error:
                    # Check if ask_human is available
                    has_ask_human = any(
                        hasattr(t, 'name') and t.name in ('ask_human', 'ask_human_cli')
                        for t in self.agent.tools
                    )
                    
                    # Enhance the error message with suggestions
                    validation_error = enhance_validation_error(
                        validation_error,
                        initial_state or {},
                        self.agent.tools,
                        has_ask_human
                    )
                
                if is_valid:
                    # Compute cost
                    cost = self._compute_candidate_cost(code)
                    total_time = time.time() - candidate_start_time
                    # Don't log the valid code here - it will be logged when selected in generate_code()
                    # Only log timing breakdown for successful candidates
                    if candidate_num == 0:
                        logger.info(f"✓ VALID CODE GENERATED (cost: {cost})")
                        logger.info(f"Timing breakdown: Total={total_time:.2f}s, LLM={total_llm_time:.2f}s ({total_llm_time/total_time*100:.1f}%), Validation={total_validation_time:.2f}s ({total_validation_time/total_time*100:.1f}%), Fix={total_fix_time:.2f}s")
                        logger.info(f"Iteration stats: {total_iterations} total, {failed_iterations} failed ({failed_iterations/total_iterations*100:.1f}% failure rate)")
                    return CodeCandidate(
                        code=code,
                        rank=cost,
                        is_valid=True,
                        validation_error=None,
                        iterations_used=iteration + 1,
                        total_time=total_time,
                        llm_time=total_llm_time,
                        validation_time=total_validation_time,
                        fix_time=total_fix_time,
                        llm_timings=llm_timings,
                        total_iterations=total_iterations,
                        failed_iterations=failed_iterations,
                        iteration_errors=iteration_errors,
                        iteration_error_types=iteration_error_types
                    )
                else:
                    # Invalid - prepare for next iteration
                    failed_iterations += 1
                    logger.warning(f"✗ VALIDATION FAILED (took {validation_elapsed:.3f}s): {validation_error}")
                    logger.warning(f"Invalid code:\n```python\n{code}\n```")
                    error = validation_error
                    # Track error details
                    iteration_errors.append(validation_error or "")
                    etype = self._classify_error(validation_error)
                    iteration_error_types.append(etype)
                    # Add assistant message and error feedback to conversation
                    messages.append(AssistantMessage(content=f"```python\n{code}\n```"))
                    # Continue to next iteration with error
                    
            except Exception as e:
                logger.error(f"Code generation failed (iteration {iteration + 1}): {e}")
                error = f"Exception during generation: {str(e)}"
                continue
        
        # Max iterations reached without valid code
        total_time = time.time() - candidate_start_time
        logger.error(f"Failed to generate valid candidate after {self.max_iterations} iterations")
        logger.info(f"Timing breakdown: Total={total_time:.2f}s, LLM={total_llm_time:.2f}s ({total_llm_time/total_time*100:.1f}%), Validation={total_validation_time:.2f}s ({total_validation_time/total_time*100:.1f}%), Fix={total_fix_time:.2f}s")
        logger.info(f"Iteration stats: {total_iterations} total, {failed_iterations} failed ({failed_iterations/total_iterations*100:.1f}% failure rate)")
        # Return the last attempt even though it's invalid
        return CodeCandidate(
            code=code if 'code' in locals() else "",
            rank=0.0,
            is_valid=False,
            validation_error=error if 'error' in locals() else "Max iterations reached",
            iterations_used=0,  # Never became valid
            total_time=total_time,
            llm_time=total_llm_time,
            validation_time=total_validation_time,
            fix_time=total_fix_time,
            llm_timings=llm_timings,
            total_iterations=total_iterations,
            failed_iterations=failed_iterations,
            iteration_errors=iteration_errors,
            iteration_error_types=iteration_error_types
        )

    def _classify_error(self, error: Optional[str]) -> str:
        """Classify a validation error string into a category for fractional pass metrics."""
        if not error:
            return "unknown"
        e = error.lower()
        if "syntax" in e or "invalid syntax" in e:
            return "syntax"
        if "line " in e and (":" in error):
            return "types"
        if "precondition" in e or "tool ordering" in e or "ordering" in e:
            return "ordering"
        if "pre-tools" in e:
            return "pre-tools"
        return "unknown"


__all__ = ["CodeGenerator", "CodeCandidate", "EXAMPLE_CODE"]
