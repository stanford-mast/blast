"""
Lightweight Python executor using exec() with async support.

Simpler and faster than smolagents executor, with full async/await support.
Does NOT provide sandboxing - only use with trusted code generation.
"""

import ast
import asyncio
import logging
from typing import Any, Dict, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Maximum iterations for loops to prevent infinite loops
MAX_LOOP_ITERATIONS = 10000


@dataclass
class CodeOutput:
    """Result of code execution."""
    output: Any
    logs: str
    is_final_answer: bool = False
    error: Optional[str] = None


def verify_code(code: str) -> tuple[bool, str]:
    """
    Verify code is syntactically valid and meets basic safety requirements.
    
    Currently checks:
    - Valid Python syntax
    - No AsyncFunctionDef nodes (must use top-level await instead)
    - No obvious infinite loops (simple heuristic check)
    
    Args:
        code: Python code to verify
        
    Returns:
        (is_valid, error_message) tuple
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    # Check for disallowed constructs
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            return False, (
                "AsyncFunctionDef not allowed. Use top-level 'await' statements instead. "
                "All provided functions are already async - just call them with 'await function_name(args)' directly."
            )
        
        # Check for suspicious infinite loops (while True without break)
        if isinstance(node, ast.While):
            # Check if condition is constant True
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                # Check if there's any break statement in the body
                has_break = any(
                    isinstance(child, ast.Break)
                    for child in ast.walk(node)
                )
                if not has_break:
                    return False, "Detected 'while True' without break - possible infinite loop"
    
    return True, ""


class LocalPythonExecutor:
    """
    Execute Python code using exec() with async support.
    
    Maintains state between executions and provides access to custom functions.
    Captures print outputs and supports async/await natively.
    
    Args:
        additional_functions: Dictionary of functions to make available to code
        additional_imports: Dictionary of module imports to make available
    """
    
    def __init__(
        self,
        additional_functions: Optional[Dict[str, Callable]] = None,
        additional_imports: Optional[Dict[str, Any]] = None,
    ):
        self.state: Dict[str, Any] = {}
        self.additional_functions = additional_functions or {}
        self.additional_imports = additional_imports or {}
        
        # Initialize state with imports and functions
        self.state.update(self.additional_imports)
        self.state.update(self.additional_functions)
        
        # Track print output
        self.print_buffer = []
    
    def _capture_print(self, *args, **kwargs):
        """Capture print() calls."""
        output = " ".join(str(arg) for arg in args)
        self.print_buffer.append(output)
        # Also print to actual stdout for debugging
        print(output, **kwargs)
    
    async def __call__(self, code: str) -> CodeOutput:
        """
        Execute Python code asynchronously.
        
        Args:
            code: Python code to execute
            
        Returns:
            CodeOutput with result, logs, and any errors
        """
        logger.info(f"\n{'='*80}\nEXECUTING CODE\n{'='*80}\n```python\n{code}\n```\n{'-'*80}")
        
        # Verify code first
        is_valid, error_msg = verify_code(code)
        if not is_valid:
            logger.error(f"✗ CODE VERIFICATION FAILED: {error_msg}")
            return CodeOutput(
                output=None,
                logs="",
                is_final_answer=False,
                error=error_msg
            )
        
        # Clear print buffer
        self.print_buffer = []
        
        # Add print capture to state
        self.state['print'] = self._capture_print
        
        # Compile code - use 'exec' mode but need to support top-level await
        try:
            # Wrap code in an async function to support top-level await
            wrapped_code = f"async def __exec_wrapper():\n"
            # Indent each line of the original code
            for line in code.split('\n'):
                wrapped_code += f"    {line}\n"
            wrapped_code += "    return locals()\n"
            
            compiled = compile(wrapped_code, '<generated>', 'exec')
        except SyntaxError as e:
            return CodeOutput(
                output=None,
                logs="",
                is_final_answer=False,
                error=f"Compilation error: {e}"
            )
        
        # Execute in async context
        try:
            # Create execution namespace
            exec_globals = self.state.copy()
            exec_locals = {}
            
            # Execute the wrapped function definition
            exec(compiled, exec_globals, exec_locals)
            
            # Get the wrapper function and call it
            wrapper_func = exec_locals['__exec_wrapper']
            result_locals = await wrapper_func()
            
            # Update state with results (excluding the wrapper function itself)
            result_locals.pop('__exec_wrapper', None)
            self.state.update(result_locals)
            
            # Find the result - use last assigned variable
            result = None
            if result_locals:
                # Get last value (dict preserves insertion order in Python 3.7+)
                result = list(result_locals.values())[-1]
            
            logs = "\n".join(self.print_buffer)
            
            logger.info(f"✓ CODE EXECUTED SUCCESSFULLY")
            logger.info(f"Output: {result}")
            logger.info(f"Logs:\n{logs}")
            logger.info(f"{'='*80}")
            
            return CodeOutput(
                output=result,
                logs=logs,
                is_final_answer=False
            )
            
        except Exception as e:
            logs = "\n".join(self.print_buffer)
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"✗ CODE EXECUTION ERROR: {error_msg}")
            logger.error(f"Logs:\n{logs}")
            logger.error(f"{'='*80}")
            
            return CodeOutput(
                output=None,
                logs=logs,
                is_final_answer=False,
                error=error_msg
            )
    
    def send_tools(self, tools: Dict[str, Callable]):
        """Update available tools."""
        self.additional_functions.update(tools)
        self.state.update(tools)
    
    def send_variables(self, variables: Dict[str, Any]):
        """Update state variables."""
        self.state.update(variables)


__all__ = ["LocalPythonExecutor", "CodeOutput", "verify_code"]
