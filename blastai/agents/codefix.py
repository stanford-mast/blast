"""
Automated code fixing for common LLM mistakes.

Provides passes to automatically fix simple errors before validation:
- Missing await keywords on async calls
- TODO: Missing tool calls to fix broken ordering
"""

import ast
import logging
import re
from typing import Optional, Set, List, Tuple

logger = logging.getLogger(__name__)


def fix_missing_awaits(code: str, async_functions: Set[str]) -> Tuple[str, bool]:
    """
    Fix missing await keywords on async function calls.
    
    Common LLM mistake: calling async functions without await
    Example: result = ai_eval("...") should be result = await ai_eval("...")
    
    Args:
        code: Python code to fix
        async_functions: Set of function names that are async
        
    Returns:
        Tuple of (fixed_code, was_modified)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False
    
    modified = False
    lines = code.split('\n')
    
    # Find all assignments to async calls that are missing await
    class AwaitFixer(ast.NodeVisitor):
        def __init__(self):
            self.fixes: List[Tuple[int, int, str]] = []  # (line, col, func_name)
        
        def visit_Assign(self, node):
            """Check assignments like: result = ai_eval(...)"""
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    func_name = node.value.func.id
                    if func_name in async_functions:
                        # Check if already awaited
                        if not isinstance(node.value, ast.Await):
                            self.fixes.append((node.lineno - 1, node.col_offset, func_name))
            self.generic_visit(node)
        
        def visit_Expr(self, node):
            """Check standalone calls like: ai_eval(...)"""
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    func_name = node.value.func.id
                    if func_name in async_functions:
                        if not isinstance(node.value, ast.Await):
                            self.fixes.append((node.lineno - 1, node.col_offset, func_name))
            self.generic_visit(node)
    
    fixer = AwaitFixer()
    fixer.visit(tree)
    
    if not fixer.fixes:
        return code, False
    
    # Apply fixes (process in reverse to maintain line numbers)
    for line_idx, col_offset, func_name in reversed(fixer.fixes):
        if line_idx < len(lines):
            line = lines[line_idx]
            # Find the function call in the line
            # Look for the pattern: func_name( where it's not preceded by "await "
            
            # Simple regex approach: find func_name( that's not preceded by await
            pattern = rf'(?<!await\s)({re.escape(func_name)}\s*\()'
            
            if re.search(pattern, line):
                # Insert "await " before the function call
                fixed_line = re.sub(pattern, r'await \1', line, count=1)
                if fixed_line != line:
                    lines[line_idx] = fixed_line
                    modified = True
                    logger.info(f"Fixed missing await on {func_name} at line {line_idx + 1}")
    
    if modified:
        return '\n'.join(lines), True
    else:
        return code, False


def fix_tool_ordering(code: str, tool_info: dict) -> Tuple[str, bool]:
    """
    TODO: Fix broken tool ordering by inserting missing tool calls.
    
    Example issue: goto_restaurant_detail requires page=list, but state has page=detail
    Fix: Insert goto_restaurants_list() before goto_restaurant_detail in loop
    
    This is complex and requires:
    - CFG analysis to detect where ordering breaks
    - Understanding which tool calls would fix the state
    - Inserting the fix at the right location
    
    Args:
        code: Python code to fix
        tool_info: Dictionary with tool preconditions/postconditions
        
    Returns:
        Tuple of (fixed_code, was_modified)
    """
    # TODO: Implement this when we have clear examples of what fixes work
    logger.warning("Tool ordering auto-fix not yet implemented")
    return code, False


def apply_code_fixes(
    code: str,
    tools: Optional[List] = None,
    tool_info: Optional[dict] = None
) -> Tuple[str, bool]:
    """
    Apply all automated code fixes.
    
    Args:
        code: Python code to fix
        tools: List of tool objects (will extract async function names)
        tool_info: Dictionary with tool info (for ordering fixes)
        
    Returns:
        Tuple of (fixed_code, was_modified)
    """
    # Build set of async function names from tools
    async_functions = {'ai_eval', 'ai_exec', 'goto', 'get_url'}
    
    if tools:
        from .models import ToolExecutorType
        for tool in tools:
            async_functions.add(tool.name)
    
    original_code = code
    total_modified = False
    
    # Pass 1: Fix missing awaits
    code, modified = fix_missing_awaits(code, async_functions)
    total_modified = total_modified or modified
    
    # Pass 2: Fix tool ordering (TODO)
    if tool_info:
        code, modified = fix_tool_ordering(code, tool_info)
        total_modified = total_modified or modified
    
    return code, total_modified


__all__ = ["fix_missing_awaits", "fix_tool_ordering", "apply_code_fixes"]
