"""
Code generation module for agent code execution mode.

Handles code generation with validation and definition code inclusion.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import SystemMessage, AssistantMessage, UserMessage, BaseMessage

from .models import ToolExecutorType, SMCPTool
from .local_python_executor import verify_code

logger = logging.getLogger(__name__)


# Example code showing realistic patterns
EXAMPLE_CODE = """
x = await tool_a()
d = []
for y in x.z:
    if y.p:
        d += y.q
result = await ask(f"Summarize {d}")
"""

@dataclass
class CodeCandidate:
    """A generated code candidate."""
    code: str
    rank: int  # Lower is better
    is_valid: bool
    validation_error: Optional[str] = None


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
        compare_cost_threshold: Optional[float] = None
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
        """
        self.agent = agent
        self.llm = llm
        self.state_aware = state_aware
        self.num_candidates = num_candidates
        self.max_iterations = max_iterations
        self.accept_cost_threshold = accept_cost_threshold
        self.min_candidates_for_comparison = min_candidates_for_comparison
        self.compare_cost_threshold = compare_cost_threshold
    
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
        
        # Import Pydantic for type definitions
        lines.append("from pydantic import BaseModel, Field")
        lines.append("from typing import Optional, List, Dict, Any")
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
                # If tool has a URL pre_path, we use STATE["current_url"] for matching
                if getattr(t, "pre_path", None):
                    state_keys.add("current_url")

        # Make deterministic ordering
        state_keys_list = sorted(list(state_keys))

        # STATE dictionary will be available at runtime (managed by LocalPythonExecutor)
        # Document the state variables for clarity
        if state_keys_list:
            lines.append(f"# STATE variables (tracked at runtime): {', '.join(state_keys_list)}")
        else:
            lines.append("# STATE variables (tracked at runtime): page, current_url")
        lines.append("# Access STATE['key'] to read/write state")
        lines.append("")
        
        # Add SMCP tool definitions with precondition/postcondition assertions
        for tool in self.agent.tools:
            if tool.tool_executor_type == ToolExecutorType.SMCP:
                smcp_tool = tool
                assert isinstance(smcp_tool, SMCPTool)
                
                # Generate Pydantic model for input parameters if they exist
                has_params = smcp_tool.input_schema and smcp_tool.input_schema.get("properties")
                if has_params:
                    model_name = f"{self._snake_to_pascal(smcp_tool.name)}Input"
                    lines.append(f"class {model_name}(BaseModel):")
                    
                    properties = smcp_tool.input_schema["properties"]
                    required = smcp_tool.input_schema.get("required", [])
                    
                    for param_name, param_schema in properties.items():
                        param_type = self._json_type_to_python(param_schema.get("type", "str"))
                        is_required = param_name in required
                        param_desc = param_schema.get("description", "")
                        
                        if is_required:
                            lines.append(f'    {param_name}: {param_type} = Field(..., description="{param_desc}")')
                        else:
                            lines.append(f'    {param_name}: Optional[{param_type}] = Field(None, description="{param_desc}")')
                    
                    lines.append("")
                
                # Generate function signature with proper types (not **kwargs)
                if has_params:
                    # Build typed parameters
                    properties = smcp_tool.input_schema["properties"]
                    required = smcp_tool.input_schema.get("required", [])
                    params = []
                    for param_name, param_schema in properties.items():
                        param_type = self._json_type_to_python(param_schema.get("type", "str"))
                        is_required = param_name in required
                        if is_required:
                            params.append(f"{param_name}: {param_type}")
                        else:
                            params.append(f"{param_name}: Optional[{param_type}] = None")
                    
                    params_str = ", ".join(params)
                    lines.append(f"async def {smcp_tool.name}({params_str}) -> Dict[str, Any]:")
                else:
                    lines.append(f"async def {smcp_tool.name}() -> Dict[str, Any]:")
                
                lines.append(f'    """')
                lines.append(f'    {smcp_tool.description}')
                lines.append(f'    """')
                
                # PRECONDITIONS (assert statements checking STATE)
                if self.state_aware:
                    # URL pattern precondition
                    if smcp_tool.pre_path and smcp_tool.pre_path != "":
                        url_pattern = smcp_tool.pre_path.replace("*", ".*")
                        lines.append(f'    # Precondition: URL must match "{smcp_tool.pre_path}"')
                        lines.append(f'    assert re.match(r"{url_pattern}", STATE["current_url"]), \\')
                        lines.append(f'        f"Expected URL matching {smcp_tool.pre_path}, got {{STATE[\\"current_url\\"]}}"')
                    
                    # State preconditions with pattern support
                    if smcp_tool.pre and isinstance(smcp_tool.pre, dict):
                        for key, value in smcp_tool.pre.items():
                            # Skip None values - they don't impose constraints
                            if value is None:
                                continue
                            assertion = self._generate_assertion(key, value, "precondition", state_check=True)
                            if assertion:
                                lines.append(f'    # Precondition: {key} {self._pattern_description(value)}')
                                lines.append(f'    {assertion}')
                
                # IMPLEMENTATION (placeholder showing it calls internal implementation)
                lines.append(f'    ')
                lines.append(f'    # ... implementation omitted ...')
                
                if has_params:
                    lines.append(f'    # Validate and call')
                    model_name = f"{self._snake_to_pascal(smcp_tool.name)}Input"
                    # Build kwargs from function parameters
                    properties = smcp_tool.input_schema["properties"]
                    param_names = list(properties.keys())
                    kwargs_dict = ", ".join([f'"{p}": {p}' for p in param_names])
                    lines.append(f'    params = {model_name}({kwargs_dict})')
                    lines.append(f'    result = await _call_tool_impl("{smcp_tool.name}", params.model_dump())')
                else:
                    lines.append(f'    result = await _call_tool_impl("{smcp_tool.name}", {{}})')
                lines.append(f'    ')
                
                # Update STATE from result
                lines.append(f'    # Update STATE from result')
                lines.append(f'    STATE.update(result)')
                lines.append(f'    ')
                
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
                            lines.append(f'    # Postcondition: {key} == input parameter {param_ref}')
                            lines.append(f'    assert STATE["{key}"] == {param_ref}, \\')
                            lines.append(f'        f"{key} must match parameter {param_ref}, got {{STATE[\\"{key}\\"]}}"')
                        else:
                            assertion = self._generate_assertion(key, value, "postcondition", state_check=True)
                            if assertion:
                                lines.append(f'    # Postcondition: {key} {self._pattern_description(value)}')
                                lines.append(f'    {assertion}')
                
                lines.append(f'    return result')
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
        
        # Add utility functions
        lines.append('async def run(task: str) -> Any:')
        lines.append('    """')
        lines.append('    Run an AI agent to complete this task. Use f-string to pass in previous results.')
        lines.append('    """')
        lines.append('    # ... implementation omitted ...')
        lines.append('    pass')
        lines.append('')
        lines.append('async def ask(prompt: str) -> str:')
        lines.append('    """Ask AI a question. Use f-string to pass in previous results."""')
        lines.append('    # ... implementation omitted ...')
        lines.append('    pass')
        lines.append('')
        
        return '\n'.join(lines)
    
    def _snake_to_pascal(self, snake_str: str) -> str:
        """Convert snake_case to PascalCase."""
        return ''.join(word.capitalize() for word in snake_str.split('_'))
    
    def _json_type_to_python(self, json_type: str) -> str:
        """Convert JSON schema type to Python type hint."""
        type_map = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]"
        }
        return type_map.get(json_type, "Any")
    
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
            return f'assert {var_ref} is None, f"{var_name} must be None, got {{{var_ref}}}"'
        
        # Handle non-string patterns (numbers, booleans, etc.)
        if not isinstance(pattern, str):
            return f'assert {var_ref} == {json.dumps(pattern)}, f"{var_name} must be {json.dumps(pattern)}, got {{{var_ref}}}"'
        
        # Handle string patterns
        if pattern == "*":
            # Must be non-null
            return f'assert {var_ref} is not None, "{var_name} must be non-null"'
        elif pattern == "":
            # Must be null/empty
            return f'assert not {var_ref}, "{var_name} must be null/empty"'
        elif "|" in pattern:
            # Must be one of the options
            options = pattern.split("|")
            options_json = json.dumps(options)
            return f'assert {var_ref} in {options_json}, f"{var_name} must be one of {options_json}, got {{{var_ref}}}"'
        elif pattern.startswith("$"):
            # Handled separately in caller (needs context of params)
            return None
        else:
            # Concrete value
            return f'assert {var_ref} == {json.dumps(pattern)}, f"{var_name} must be {json.dumps(pattern)}, got {{{var_ref}}}"'
    
    def _check_candidate(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Check if generated code is valid.
        
        Args:
            code: Generated Python code
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return verify_code(code)
    
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
        return float(len(code))
    
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
        error: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate Python code to complete the task.
        
        Uses parallel candidate generation with iterative refinement.
        Supports early acceptance and comparison thresholds.
        
        Args:
            task: Task description
            history: Conversation history as list of BaseMessage (UserMessage, AssistantMessage, SystemMessage)
            error: Optional error from previous attempt (for first iteration)
        
        Returns:
            Generated code or None if generation failed
        """
        if self.num_candidates <= 1:
            # Single candidate generation
            candidate = await self._generate_candidate(task, history, error)
            return candidate.code if candidate and candidate.is_valid else None
        
        # Parallel candidate generation
        logger.info(f"Generating {self.num_candidates} candidates in parallel")
        
        # Launch parallel candidate generation tasks
        tasks = [
            self._generate_candidate(task, history, error)
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
        initial_error: Optional[str] = None
    ) -> Optional[CodeCandidate]:
        """
        Generate a single code candidate with iterative refinement.
        
        Maintains a list of chat messages and iteratively refines based on errors.
        The history parameter contains the broader conversation context from previous
        code execution attempts. Each iteration of refinement appends to a local
        message list.
        
        Args:
            task: Task description
            history: Conversation history from AgentExecutor (previous code execution attempts)
            initial_error: Optional error from previous attempt
        
        Returns:
            CodeCandidate or None if generation failed
        """
        # Build definition code
        definition_code = self._build_definition_code()
        
        # Initialize message list for this candidate
        # Start with the broader conversation history from executor
        messages = []
        
        # Add system message for this code generation session
        messages.append(
            SystemMessage(content="You are an expert in writing code that calls tools and programatically asks AI questions.")
        )

        if history:
            messages.extend(history)
        
        # Build initial prompt
        error = initial_error
        
        for iteration in range(self.max_iterations):
            # Build prompt with XML tags
            prompt_parts = []
            
            prompt_parts.append("<instructions>")
            prompt_parts.append("Generate Python code to complete the TASK. You may use functions defined below.")
            prompt_parts.append("</instructions>")
            prompt_parts.append("")
            
            if error:
                prompt_parts.append("<previous_error>")
                prompt_parts.append(f"The previously generated code failed with error: {error}")
                prompt_parts.append("Fix the error and generate corrected code.")
                prompt_parts.append("</previous_error>")
                prompt_parts.append("")
            else:
                prompt_parts.append("<example>")
                prompt_parts.append(EXAMPLE_CODE.strip())
                prompt_parts.append("</example>")
                prompt_parts.append("")
            
            prompt_parts.append(f"<task>")
            prompt_parts.append(f"TASK: {task}")
            prompt_parts.append("</task>")
            prompt_parts.append("")
            
            prompt_parts.append("```python")
            prompt_parts.append(definition_code)
            prompt_parts.append("")
            prompt_parts.append("# YOUR CODE HERE")
            # Note: We deliberately leave the code block open - the LLM will close it
            
            prompt = '\n'.join(prompt_parts)
            
            # Add user message to conversation
            messages.append(UserMessage(content=prompt))
            
            # Generate code
            try:
                logger.info(f"\n{'='*80}\nCODEGEN ITERATION {iteration + 1}/{self.max_iterations}\n{'='*80}")
                # For the first iteration show full prompt; on subsequent iterations show only the error
                if iteration == 0:
                    logger.info(f"LLM INPUT (last message):\n{prompt}\n{'-'*80}")
                else:
                    err_text = error or "[no error provided]"
                    logger.info(f"LLM INPUT (error only):\n{err_text}\n{'-'*80}")
                
                response = await self.llm.ainvoke(messages)
                
                logger.info(f"LLM OUTPUT:\n{response.completion}\n{'-'*80}")
                
                # Extract code from response
                code = self._extract_code_from_response(response.completion)
                
                if not code:
                    logger.error(f"No code block found in response (iteration {iteration + 1})")
                    error = "No code block found in LLM response. Please wrap your code in ```python ... ``` markers."
                    # Add assistant message to maintain conversation
                    messages.append(AssistantMessage(content=f"[Error: {error}]"))
                    continue
                
                # Check candidate validity
                is_valid, validation_error = self._check_candidate(code)
                
                if is_valid:
                    # Compute cost
                    cost = self._compute_candidate_cost(code)
                    logger.info(f"✓ VALID CODE GENERATED (cost: {cost}):\n```python\n{code}\n```\n{'='*80}")
                    return CodeCandidate(
                        code=code,
                        rank=cost,
                        is_valid=True,
                        validation_error=None
                    )
                else:
                    # Invalid - prepare for next iteration
                    logger.warning(f"✗ VALIDATION FAILED: {validation_error}")
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
        logger.error(f"Failed to generate valid candidate after {self.max_iterations} iterations")
        return None


__all__ = ["CodeGenerator", "CodeCandidate", "EXAMPLE_CODE"]
