"""
Code validation and verification for generated Python code.

Provides functions to check code syntax and safety before execution.
Includes CFG-based validation to ensure tool ordering respects preconditions.
Includes mypy type checking for full type safety.
"""

import ast
import logging
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Set, Any, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
from .cfg_builder import CFGBuilder, BasicBlock

logger = logging.getLogger(__name__)



def matches_pattern(state_value: Any, pattern: Any, params: Dict[str, Any] = None) -> bool:
    """
    Check if a state value matches a pattern.
    
    Patterns:
    - None: state_value must be None
    - "*": state_value must be non-None
    - "": state_value must be None/empty/falsy
    - "a|b|c": state_value must be one of the options
    - "$param": state_value must equal params[param]
    - concrete value: state_value must equal the value
    
    Args:
        state_value: Current value in state
        pattern: Pattern to match against
        params: Parameter values (for $param references)
        
    Returns:
        True if matches, False otherwise
    """
    if params is None:
        params = {}
    
    # Handle None pattern
    if pattern is None:
        return state_value is None
    
    # Handle non-string patterns (numbers, booleans, etc.)
    if not isinstance(pattern, str):
        return state_value == pattern
    
    # Handle string patterns
    if pattern == "*":
        return state_value is not None
    elif pattern == "":
        return not state_value  # Falsy values
    elif "|" in pattern:
        options = pattern.split("|")
        return state_value in options
    elif pattern.startswith("$"):
        param_name = pattern[1:]
        return state_value == params.get(param_name)
    else:
        # Concrete value
        return state_value == pattern


def check_tool_ordering(
    blocks: Dict[int, BasicBlock],
    start_block: BasicBlock,
    initial_state: Dict[str, Any],
    tools_by_name: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate tool ordering via CFG traversal with abstract state tracking.
    
    Args:
        blocks: Dictionary of basic blocks
        start_block: Starting block
        initial_state: Initial abstract state
        tools_by_name: Dictionary mapping tool names to tool objects (with pre/post)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Track paths we've explored (block_id -> visit_count)
    visit_counts: Dict[int, int] = {}
    max_loop_iterations = 2
    
    def traverse(block: BasicBlock, state: Dict[str, Any], path: List[str]) -> Tuple[bool, Optional[str]]:
        """Recursively traverse CFG and validate tool ordering."""
        # Prevent infinite loops
        visit_counts[block.bid] = visit_counts.get(block.bid, 0) + 1
        if visit_counts[block.bid] > max_loop_iterations:
            return True, None  # Stop this path
        
        # Check tool calls in this block
        for func_name, call_node in block.calls:
            # Special handling for ai_exec - resets state
            if func_name == "ai_exec":
                state = {}  # Reset state after ai_exec
                path.append(f"ai_exec(...) [state reset]")
                continue
            
            # Check if this is a known tool
            if func_name not in tools_by_name:
                # Unknown tool - skip validation (might be utility function)
                path.append(f"{func_name}(...) [unknown]")
                continue
            
            tool = tools_by_name[func_name]
            
            # Extract parameter values from call
            params = {}
            try:
                for i, arg in enumerate(call_node.args):
                    if i < len(tool.get('param_names', [])):
                        param_name = tool['param_names'][i]
                        # Try to extract constant values
                        if isinstance(arg, (ast.Constant, ast.Num, ast.Str)):
                            params[param_name] = getattr(arg, 'value', getattr(arg, 'n', getattr(arg, 's', None)))
                
                for keyword in call_node.keywords:
                    if isinstance(keyword.value, (ast.Constant, ast.Num, ast.Str)):
                        params[keyword.arg] = getattr(keyword.value, 'value', getattr(keyword.value, 'n', getattr(keyword.value, 's', None)))
            except Exception:
                pass  # Couldn't extract params, continue anyway
            
            # Check preconditions
            pre = tool.get('pre', {})
            if pre:
                for key, pattern in pre.items():
                    if not matches_pattern(state.get(key), pattern, params):
                        error_msg = (
                            f"Tool ordering error: {func_name} requires {key}={pattern}, "
                            f"but state has {key}={state.get(key)}. "
                            f"Call sequence: {' -> '.join(path + [func_name])}"
                        )
                        return False, error_msg
            
            # Update state with postconditions
            post = tool.get('post', {})
            if post:
                for key, pattern in post.items():
                    if isinstance(pattern, str) and pattern.startswith("$"):
                        # Reference to parameter
                        param_name = pattern[1:]
                        if param_name in params:
                            state[key] = params[param_name]
                        # If param not found, leave state unchanged
                    elif pattern == "":
                        state[key] = None
                    elif pattern != "*":
                        # Concrete value or pattern that determines new state
                        state[key] = pattern
                    # pattern == "*" means leave as is (any non-null value)
            
            # Add to path
            path.append(func_name)
        
        # If no next blocks, path is complete
        if not block.next:
            return True, None
        
        # Explore all outgoing edges
        for next_bid in block.next:
            next_block = blocks[next_bid]
            # Create copy of state for this path
            next_state = deepcopy(state)
            next_path = path.copy()
            
            valid, error = traverse(next_block, next_state, next_path)
            if not valid:
                return False, error
        
        return True, None
    
    # Start traversal
    valid, error = traverse(start_block, deepcopy(initial_state), [])
    
    # Reset visit counts
    visit_counts.clear()
    
    return valid, error


def check_mypy_types(full_code: str) -> Tuple[bool, Optional[str]]:
    """
    Run mypy type checking on the full code (definition + generated).
    
    Creates a temporary file with the code and runs mypy on it.
    
    Args:
        full_code: Complete Python code including definitions and generated code
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_path = f.name
        
        try:
            # Run mypy with balanced type checking flags:
            # Key goals:
            # 1. Catch wrong attribute access (e.g., details.price when price is in details.items[0].menu_items[0])
            # 2. Catch dict access on Pydantic models (e.g., details["items"] instead of details.items)
            # 3. Catch type mismatches (e.g., int assigned to str field)
            # 4. Allow reasonable Optional narrowing patterns in list comprehensions/lambdas
            # 5. Don't require type annotations in user-generated code (LLMs don't always add them)
            #
            # --check-untyped-defs: Type check function bodies even without annotations
            # --disallow-any-unimported: Catch missing imports/typos
            # --no-strict-optional: Don't error on Optional[T] narrowing in comprehensions/lambdas
            #                       (this is overly pedantic for LLM-generated code)
            # --no-error-summary: Clean output format
            # --show-column-numbers: Show column positions
            # --no-pretty: Simpler error format
            # Note: NOT using --warn-return-any or --disallow-untyped-defs for flexibility
            result = subprocess.run(
                ['mypy', 
                 '--check-untyped-defs',
                 '--disallow-any-unimported',
                 '--no-strict-optional',  # Allow Optional narrowing
                 '--no-error-summary', 
                 '--show-column-numbers', 
                 '--no-pretty', 
                 temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return True, None
            else:
                # Parse mypy errors and make them more readable
                errors = result.stdout.strip()
                if not errors:
                    errors = result.stderr.strip()
                
                # Remove temp file path from errors
                errors = errors.replace(temp_path, "<code>")
                
                return False, f"Type checking errors:\n{errors}"
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
    except FileNotFoundError:
        # mypy not installed - skip type checking
        logger.debug("mypy not found, skipping type checking")
        return True, None
    except subprocess.TimeoutExpired:
        return False, "Type checking timed out"
    except Exception as e:
        logger.warning(f"Type checking failed with exception: {e}")
        # Don't fail validation if mypy check fails
        return True, None


def check_code_candidate(
    code: str,
    agent: Any = None,
    initial_state: Dict[str, Any] = None,
    definition_code: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    """
    Check if generated code candidate is valid.
    
    Performs:
    1. Syntax validation
    2. Optional mypy type checking (if definition_code provided)
    3. CFG construction
    4. Tool ordering validation via CFG traversal with precondition/postcondition checking
    
    Args:
        code: Generated Python code
        agent: Optional agent object with tools (for precondition/postcondition checking)
        initial_state: Optional initial state dict
        definition_code: Optional definition code to prepend for full type checking
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    # 2. Optional mypy type checking if we have definition code
    if definition_code is not None:
        # Wrap user code in async function to allow await statements
        # Use Any return type since generated code may return different types
        wrapped_code = f"""
async def _user_generated_code() -> Any:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
"""
        full_code = definition_code + "\n\n" + wrapped_code
        is_valid, type_error = check_mypy_types(full_code)
        if not is_valid:
            return False, type_error
    
    # 3. Build CFG and validate tool ordering if we have agent context
    if agent is not None:
        try:
            cfg_builder = CFGBuilder()
            start_block, blocks = cfg_builder.build(tree)
            
            # Build tools_by_name dictionary
            tools_by_name = {}
            for tool in agent.tools:
                # Get tool metadata
                tool_info = {
                    'pre': getattr(tool, 'pre', {}),
                    'post': getattr(tool, 'post', {}),
                    'param_names': []
                }
                
                # Extract parameter names if available
                if hasattr(tool, 'input_schema') and tool.input_schema:
                    properties = tool.input_schema.get('properties', {})
                    tool_info['param_names'] = list(properties.keys())
                
                tools_by_name[tool.name] = tool_info
            
            # 3. Validate tool ordering
            if initial_state is None:
                initial_state = {}
            
            valid, error = check_tool_ordering(blocks, start_block, initial_state, tools_by_name)
            if not valid:
                return False, error
                
        except Exception as e:
            # CFG building/validation failed - log but don't fail validation
            # This allows the code to execute and fail at runtime if needed
            logger.debug(f"CFG validation skipped due to error: {e}")
    
    return True, ""



# Backward compatibility alias
verify_code = check_code_candidate


__all__ = ["check_code_candidate", "verify_code"]