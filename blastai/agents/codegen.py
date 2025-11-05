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

from .models import ToolExecutorType, SMCPTool
from .codecheck import check_code_candidate
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
response = await ai_eval(f"Summary of {x.content}")
```

E4: ordering tool calls based on current STATE and tool preconditions/postconditions, passing previous tools results into next tool call either by direct access or ai_eval, using ai_eval to match a user-provided term to an option in a list
```python
await tool_c()
x = await tool_d(id=123)
y = await tool_e(p1=x.data)
z = await tool_f(ai_eval(f"Name in {y.options} closest to 'the ttanic'"))
result = ai_eval(f"Response telling user about {z.info} since they asked about the titanic")
```

E5: control flow with loops and conditionals
```python
await tool_c()
x = await tool_e(123)
results = [await tool_f(item_name=item.name) for item in x.items if item.value > 3]
result = ai_eval(f"Summary of {results} for whatever specific goal user had")
"""

RULES = """
- Do not generate comments
- If using a user-provided term (e.g. a name to filter by/search for/pass to a tool), use ai_eval to determine which option it most closely matches.
"""

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


class CodeGenerator:
    """
    Generates Python code for agent tasks with definition code and validation.
    
    Includes tool definitions as executable Python code in the generation,
    using XML-based prompting similar to derive_synthesis_agent.
    """
    
    def __init__(
        self,
        agent,
        llm: BaseChatModel,
        state_aware: bool = False,
        num_candidates: int = 1,
        max_iterations: int = 3,
        accept_cost_threshold: Optional[float] = None,
        min_candidates_for_comparison: int = 1,
        compare_cost_threshold: Optional[float] = None,
        timezone: str = "UTC"
    ):
        """
        Initialize code generator.
        
        Args:
            agent: Agent instance with tools
            llm: LLM for code generation
            state_aware: Whether to include state preconditions/postconditions
            num_candidates: Number of candidates to generate in parallel (default 1)
            max_iterations: Maximum iterations per candidate generation (default 3)
            accept_cost_threshold: If set, immediately accept candidate below this cost (default None)
            min_candidates_for_comparison: Minimum candidates before comparison (default 1)
            compare_cost_threshold: If set, compare and return when we have min_candidates below this threshold (default None)
            timezone: Timezone string in IANA format for current date/time context (default 'UTC')
        """
        self.agent = agent
        self.llm = llm
        self.state_aware = state_aware
        self.num_candidates = num_candidates
        self.max_iterations = max_iterations
        self.accept_cost_threshold = accept_cost_threshold
        self.min_candidates_for_comparison = min_candidates_for_comparison
        self.compare_cost_threshold = compare_cost_threshold
        self.timezone = timezone
        
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
        
        # Track which tools have been called (for pre_tools enforcement)
        lines.append("# Track which tools have been called")
        lines.append("TOOLS_CALLED: set[str] = set()")
        lines.append("")
        
        # Add SMCP tool definitions with precondition/postcondition assertions
        for tool in self.agent.tools:
            if tool.tool_executor_type == ToolExecutorType.SMCP:
                smcp_tool = tool
                assert isinstance(smcp_tool, SMCPTool)
                
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
                    for param_name, param_schema in properties.items():
                        param_type = json_type_to_python(param_schema.get("type", "str"))
                        is_required = param_name in required
                        if is_required:
                            params.append(f"{param_name}: {param_type}")
                        else:
                            params.append(f"{param_name}: Optional[{param_type}] = None")
                    
                    params_str = ", ".join(params)
                    lines.append(f"async def {smcp_tool.name}({params_str}) -> {return_type}:")
                else:
                    lines.append(f"async def {smcp_tool.name}() -> {return_type}:")
                
                lines.append(f'    """')
                lines.append(f'    {smcp_tool.description}')
                lines.append(f'    """')
                
                # PRE_TOOLS ASSERTIONS (check that required tools have been called)
                if hasattr(smcp_tool, 'pre_tools') and smcp_tool.pre_tools:
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
                
                # Track that this tool was called (conceptually - runtime handles this)
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
                else:
                    lines.append(f'async def {tool.name}(**kwargs) -> Dict[str, Any]:')
                
                lines.append(f'    """')
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
        lines.append('async def goto(url: str) -> Dict[str, Any]:')
        lines.append('    """')
        lines.append('    Navigate to a URL and update STATE via matching observe tool.')
        lines.append('    Handles login/redirect detection automatically.')
        lines.append('    """')
        lines.append('    raise NotImplementedError("Runtime implementation")')
        lines.append('')
        lines.append('async def ai_exec(subtask: str, output_schema: Optional[Dict[str, Any]] = None) -> Any:')
        lines.append('    """')
        lines.append('    Execute an AI agent to complete a given subtask.')
        lines.append('    Optionally provide output_schema for structured output.')
        lines.append('    """')
        lines.append('    raise NotImplementedError("Runtime implementation")')
        lines.append('')
        lines.append('async def ai_eval(expr: str, **kwargs) -> str:')
        lines.append('    """')
        lines.append('    Evaluate an expression by asking AI.')
        lines.append('    Supports template format: ai_eval("Name in {options} closest to X", options=values)')
        lines.append('    """')
        lines.append('    raise NotImplementedError("Runtime implementation")')
        lines.append('')
        
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
            self._generate_candidate(task, history, error, initial_state, current_url)
            for _ in range(self.num_candidates)
        ]
        
        # Collect valid candidates as they complete
        valid_candidates: List[CodeCandidate] = []
        
        # Process candidates as they complete
        for completed_task in asyncio.as_completed(tasks):
            candidate = await completed_task
            
            if candidate and candidate.is_valid:
                # Check for immediate acceptance
                if self.accept_cost_threshold is not None and candidate.rank <= self.accept_cost_threshold:
                    logger.info(f"Immediately accepting candidate with cost {candidate.rank} (below threshold {self.accept_cost_threshold})")
                    return candidate.code
                
                valid_candidates.append(candidate)
                
                # Check for early comparison
                if self.compare_cost_threshold is not None and len(valid_candidates) >= self.min_candidates_for_comparison:
                    # Check if all collected candidates are below comparison threshold
                    below_threshold = [c for c in valid_candidates if c.rank <= self.compare_cost_threshold]
                    if len(below_threshold) >= self.min_candidates_for_comparison:
                        logger.info(f"Early comparison: {len(below_threshold)} candidates below threshold {self.compare_cost_threshold}")
                        best = min(below_threshold, key=lambda c: c.rank)
                        return best.code
        
        # All candidates completed - select best
        if not valid_candidates:
            logger.warning("No valid candidates generated")
            return None
        
        best = min(valid_candidates, key=lambda c: c.rank)
        logger.info(f"Selected best candidate with cost {best.rank} from {len(valid_candidates)} valid candidates")
        return best.code
    
    async def _generate_candidate(
        self,
        task: str,
        history: List[BaseMessage],
        initial_error: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        current_url: Optional[str] = None
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
            initial_error: Optional error from previous attempt
            initial_state: Optional initial STATE values to include in generated code
            current_url: Optional current URL to provide context to code generator
        
        Returns:
            CodeCandidate with detailed timing breakdown
        """
        candidate_start_time = time.time()
        total_llm_time = 0.0
        total_validation_time = 0.0
        total_fix_time = 0.0
        llm_timings: List[LLMTiming] = []
        total_iterations = 0
        failed_iterations = 0
        
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
{EXAMPLE_CODE}
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
            
            # Add current date/time context at the top
            from datetime import datetime
            import zoneinfo
            try:
                tz = zoneinfo.ZoneInfo(self.timezone)
                now = datetime.now(tz)
                # Format: "Monday, November 4, 2025 3:30:00 PM PST"
                day_of_week = now.strftime("%A")
                date_part = now.strftime("%B %d, %Y")
                time_part = now.strftime("%I:%M:%S %p")  # 12-hour format with AM/PM
                tz_abbr = now.strftime("%Z")
                timestamp_str = f"{day_of_week}, {date_part} {time_part} {tz_abbr}"
                prompt_parts.append(f"Now is {timestamp_str}.")
                prompt_parts.append("")
            except Exception as e:
                logger.warning(f"Failed to get current time for timezone {self.timezone}: {e}")
            
            if error:
                prompt_parts.append(error)
            else:
                prompt_parts.append("```python")
                prompt_parts.append(definition_code)
                prompt_parts.append("")
                
                # Initialize STATE with initial values if provided
                if initial_state:
                    prompt_parts.append("# Initialize STATE with current values")
                    state_init = ", ".join([f'"{k}": {repr(v)}' for k, v in initial_state.items()])
                    prompt_parts.append(f"STATE.update({{{state_init}}})")
                    prompt_parts.append("")
                
                # Show that we've already navigated to current URL
                if current_url:
                    prompt_parts.append(f'await goto("{current_url}")')
                    prompt_parts.append("")
                
                prompt_parts.append("# RULES")
                prompt_parts.append(RULES)
                prompt_parts.append("")
                prompt_parts.append("# TASK: " + task)
                prompt_parts.append("# YOUR CODE HERE")
                # Note: We deliberately leave the code block open - the LLM will close it
            
            prompt = '\n'.join(prompt_parts)
            
            # Add user message to conversation
            messages.append(UserMessage(content=prompt))
            
            # Generate code
            try:
                logger.info(f"\n{'='*80}\nCODEGEN ITERATION {iteration + 1}/{self.max_iterations}\n{'='*80}")
                logger.info(f"LLM INPUT (last message):\n{prompt}\n{'-'*80}")
                
                # Call LLM with streaming and detailed timing
                completion, streaming_timing = await stream_llm_call(self.llm, messages)
                total_llm_time += streaming_timing.total_seconds
                
                # Convert StreamingTiming to LLMTiming for compatibility
                llm_timing = LLMTiming(
                    total_seconds=streaming_timing.total_seconds,
                    time_to_first_token=streaming_timing.time_to_first_token,
                    tokens_per_second=streaming_timing.tokens_per_second,
                    total_tokens=streaming_timing.total_tokens
                )
                llm_timings.append(llm_timing)
                
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
                
                if is_valid:
                    # Compute cost
                    cost = self._compute_candidate_cost(code)
                    total_time = time.time() - candidate_start_time
                    logger.info(f"✓ VALID CODE GENERATED (cost: {cost}):\n```python\n{code}\n```\n{'='*80}")
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
                        failed_iterations=failed_iterations
                    )
                else:
                    # Invalid - prepare for next iteration
                    failed_iterations += 1
                    logger.warning(f"✗ VALIDATION FAILED (took {validation_elapsed:.3f}s): {validation_error}")
                    logger.warning(f"Invalid code:\n```python\n{code}\n```")
                    error = validation_error
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
            failed_iterations=failed_iterations
        )


__all__ = ["CodeGenerator", "CodeCandidate", "EXAMPLE_CODE"]
