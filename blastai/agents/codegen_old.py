"""
Code generation module for agent code execution mode.

Handles code generation with validation and definition code inclusion.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, List

from browser_use.llm.llm import LLM
from browser_use.llm.messages import SystemMessage, UserMessage

from .models import ToolExecutorType, SMCPTool
from .local_python_executor import verify_code

logger = logging.getLogger(__name__)


# Example code showing realistic patterns
EXAMPLE_CODE = """
# Example showing tool usage patterns
categories = await list_restaurant_categories()
for category in categories["items"]:
    if category["category"] == "Japanese":
        await filter_restaurants_by_category(category=category["category"])
        restaurants = await list_restaurants(limit=5)
        if restaurants["items"]:
            answer = await ask(f"Which restaurant should I pick from {restaurants}?")
            break
"""

import asyncio
import logging
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage, UserMessage, AssistantMessage

from .local_python_executor import verify_code
from .models import Agent, Tool, SMCPTool, ToolExecutorType

logger = logging.getLogger(__name__)


# Example code showing patterns for using tools
EXAMPLE_CODE = """
# Example: Using tools with variable passing and control flow
categories = await list_restaurant_categories()
print(f"Found {len(categories['items'])} categories")

results = []
for category in categories['items']:
    if category['name'] == 'Pizza':
        # Filter by this category
        await filter_restaurants_by_category(categoryName=category['name'])
        
        # Get restaurants in this category
        restaurants = await list_restaurants()
        
        # Visit each restaurant and get details
        for restaurant in restaurants['items'][:3]:  # First 3 only
            await goto_restaurant_detail(restaurantName=restaurant['name'])
            details = await get_restaurant_details()
            results.append(details)

# Use ask() to analyze results with f-string
summary = await ask(f"Summarize these restaurant details: {results}")
print(summary)
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
        llm: LLM,
        state_aware: bool = False,
        parallel_codegen: int = 1
    ):
        """
        Initialize code generator.
        
        Args:
            agent: Agent instance with tools
            llm: LLM for code generation
            state_aware: Whether to include state preconditions/postconditions
            parallel_codegen: Number of parallel code generation attempts
        """
        self.agent = agent
        self.llm = llm
        self.state_aware = state_aware
        self.parallel_codegen = parallel_codegen
    
    def _build_definition_code(self) -> str:
        """
        Build definition code showing tool signatures with precondition/postcondition assertions.
        
        This code is part of the Python code block that gets generated, showing the
        developer what tools are available and their contracts.
        
        Returns definitions as Python code with:
        - Pydantic models for input parameters (matching JSON schemas)
        - Assert statements for preconditions
        - Function implementation placeholder
        - Assert statements for postconditions
        """
        lines = []
        
        # Import Pydantic for type definitions
        lines.append("from pydantic import BaseModel, Field")
        lines.append("from typing import Optional, List, Dict, Any")
        lines.append("")
        
        # Add SMCP tool definitions with precondition/postcondition assertions
        for tool in self.agent.tools:
            if tool.tool_executor_type == ToolExecutorType.SMCP:
                smcp_tool = tool
                assert isinstance(smcp_tool, SMCPTool)
                
                # Generate Pydantic model for input parameters if they exist
                if smcp_tool.input_schema and smcp_tool.input_schema.get("properties"):
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
                
                # Generate function signature
                if smcp_tool.input_schema and smcp_tool.input_schema.get("properties"):
                    model_name = f"{self._snake_to_pascal(smcp_tool.name)}Input"
                    lines.append(f"async def {smcp_tool.name}(**kwargs) -> Dict[str, Any]:")
                else:
                    lines.append(f"async def {smcp_tool.name}() -> Dict[str, Any]:")
                
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
                    
                    # State preconditions with pattern support
                    if smcp_tool.pre and isinstance(smcp_tool.pre, dict):
                        for key, value in smcp_tool.pre.items():
                            assertion = self._generate_assertion(key, value, "precondition")
                            if assertion:
                                lines.append(f'    # Precondition: {key} {self._pattern_description(value)}')
                                lines.append(f'    {assertion}')
                
                # IMPLEMENTATION (placeholder showing it calls internal implementation)
                lines.append(f'    ')
                lines.append(f'    # ... implementation omitted ...')
                
                if smcp_tool.input_schema and smcp_tool.input_schema.get("properties"):
                    lines.append(f'    # Validate and call')
                    model_name = f"{self._snake_to_pascal(smcp_tool.name)}Input"
                    lines.append(f'    params = {model_name}(**kwargs)')
                    lines.append(f'    result = await _call_tool_impl("{smcp_tool.name}", params.model_dump())')
                else:
                    lines.append(f'    result = await _call_tool_impl("{smcp_tool.name}", {{}})')
                lines.append(f'    ')
                
                # POSTCONDITIONS (assert statements after result)
                if self.state_aware and smcp_tool.post and isinstance(smcp_tool.post, dict):
                    for key, value in smcp_tool.post.items():
                        # For postconditions, check result or updated state
                        if value.startswith("$"):
                            # Reference to input/output parameter
                            param_ref = value[1:]  # Remove $
                            lines.append(f'    # Postcondition: {key} == input/output parameter {param_ref}')
                            lines.append(f'    assert {key} == kwargs.get("{param_ref}") or {key} == result.get("{param_ref}"), \\')
                            lines.append(f'        f"{key} must match parameter {param_ref}"')
                        else:
                            assertion = self._generate_assertion(f'{key}', value, "postcondition", result_check=True)
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
    
    def _pattern_description(self, value: str) -> str:
        """Generate human-readable description of a state pattern."""
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
    
    def _generate_assertion(self, var_name: str, pattern: str, assertion_type: str, result_check: bool = False) -> Optional[str]:
        """
        Generate Python assertion code for a state pattern.
        
        Args:
            var_name: Variable name to check
            pattern: Pattern to match (*, |, $param, "", or concrete value)
            assertion_type: "precondition" or "postcondition"
            result_check: If True, check in result dict; otherwise check as global variable
        
        Returns:
            Python assertion code or None if no assertion needed
        """
        var_ref = f'result.get("{var_name}")' if result_check else var_name
        
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
    
    async def generate_code(
        self,
        task: str,
        conversation_history: Optional[List[BaseMessage]] = None
    ) -> CodeCandidate:
        """
        Generate code for a task, using parallel generation if configured.
        
        Args:
            task: The task description
            conversation_history: Optional previous messages for iterative refinement
            
        Returns:
            Best CodeCandidate (validated and ranked)
        """
        if self.parallel_generations <= 1:
            # Single generation
            return await self._generate_single(task, conversation_history)
        else:
            # Parallel generation with selection
            return await self._generate_parallel(task, conversation_history)
    
    async def _generate_single(
        self,
        task: str,
        conversation_history: Optional[List[BaseMessage]] = None
    ) -> CodeCandidate:
        """Generate a single code candidate."""
        functions_doc = self._build_functions_doc()
        
        prompt = f"""TASK: {task}

AVAILABLE FUNCTIONS:
{functions_doc}

EXAMPLE PATTERN:
{EXAMPLE_CODE}

Generate Python code to complete the task using the available functions shown above.
Follow the example pattern: use top-level await statements, pass variables between tools, use for/if for control flow.
Do NOT use 'async def' - just write await statements directly.

CODE:
```python
"""
        
        # Build messages
        if conversation_history:
            messages = conversation_history + [UserMessage(content=prompt)]
        else:
            messages = [UserMessage(content=prompt)]
        
        # Get response
        response = await self.llm.ainvoke(messages)
        code = self._extract_code_from_response(response.completion)
        
        if not code:
            return CodeCandidate(
                code="",
                rank=999,
                is_valid=False,
                validation_error="No code found in LLM response"
            )
        
        # Validate
        is_valid, error_msg = verify_code(code)
        
        return CodeCandidate(
            code=code,
            rank=0,
            is_valid=is_valid,
            validation_error=error_msg if not is_valid else None
        )
    
    async def _generate_parallel(
        self,
        task: str,
        conversation_history: Optional[List[BaseMessage]] = None
    ) -> CodeCandidate:
        """
        Generate multiple code candidates in parallel and select the best.
        
        Selection criteria:
        1. Must be valid (passes verify_code)
        2. Prefer shorter code (simpler solutions)
        3. Prefer code without error handling (indicates confidence)
        """
        logger.info(f"Generating {self.parallel_generations} parallel code candidates")
        
        # Generate candidates in parallel
        tasks = [
            self._generate_single(task, conversation_history)
            for _ in range(self.parallel_generations)
        ]
        candidates = await asyncio.gather(*tasks)
        
        # Filter to valid candidates
        valid_candidates = [c for c in candidates if c.is_valid]
        
        if not valid_candidates:
            logger.warning("No valid candidates generated, returning first one")
            return candidates[0]
        
        # Rank valid candidates
        # Simple heuristic: prefer shorter code
        for i, candidate in enumerate(valid_candidates):
            lines = len(candidate.code.split('\n'))
            candidate.rank = lines
        
        # Sort by rank
        valid_candidates.sort(key=lambda c: c.rank)
        
        best = valid_candidates[0]
        logger.info(f"Selected best candidate with {best.rank} lines ({len(valid_candidates)} valid candidates)")
        
        return best
    
    async def refine_code(
        self,
        original_code: str,
        error_message: str,
        task: str,
        conversation_history: Optional[List[BaseMessage]] = None
    ) -> CodeCandidate:
        """
        Refine code after an execution error.
        
        Args:
            original_code: The code that failed
            error_message: The error message
            task: Original task
            conversation_history: Previous conversation
            
        Returns:
            New CodeCandidate
        """
        # Build conversation history with error feedback
        if conversation_history is None:
            conversation_history = []
        
        # Add the failed code and error
        history = conversation_history + [
            AssistantMessage(content=f"```python\n{original_code}\n```"),
            UserMessage(content=f"Error: {error_message}\n\nPlease fix the code. Use top-level await, not async def.")
        ]
        
        # Generate new code with history
        return await self.generate_code(task, conversation_history=history)


__all__ = ["CodeGenerator", "CodeCandidate", "CODEGEN_RULES"]
