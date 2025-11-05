"""
Automated code fixing for common LLM mistakes.

Provides passes to automatically fix simple errors before validation:
- Missing await keywords on async calls
- Transform ai_eval f-strings to explicit variable passing
- TODO: Missing tool calls to fix broken ordering
"""

import ast
import logging
import re
from typing import Optional, Set, List, Tuple, Dict

logger = logging.getLogger(__name__)


def extract_variable_name(expr: str) -> str:
    """
    Convert an expression to a variable name.
    
    Examples:
        options.employeeOptions -> options_employeeOptions
        a.b[c]["d"] -> a_b_c_d
        items[0] -> items_0
        foo -> foo
    
    Args:
        expr: Python expression string
        
    Returns:
        Variable name suitable for keyword argument
    """
    # Replace dots with underscores
    name = expr.replace('.', '_')
    # Replace square brackets and quotes with underscores
    name = re.sub(r'[\[\]"\']', '_', name)
    # Remove double underscores
    name = re.sub(r'__+', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    # If empty or starts with digit, use generic name
    if not name or name[0].isdigit():
        return 'var'
    return name


def transform_ai_eval_fstrings(code: str) -> Tuple[str, bool]:
    """
    Transform ai_eval f-string calls to explicit variable passing.
    
    Transforms:
        ai_eval(f"Name in {options.employeeOptions} closest to 'Amber'")
    To:
        ai_eval("Name in {options_employeeOptions} closest to 'Amber'", 
                options_employeeOptions=options.employeeOptions)
    
    Args:
        code: Python code to transform
        
    Returns:
        Tuple of (transformed_code, was_modified)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False
    
    modified = False
    
    class FStringTransformer(ast.NodeTransformer):
        def __init__(self):
            self.modified = False
        
        def visit_Call(self, node):
            """Transform ai_eval(f"...{expr}...") calls"""
            # Check if this is a call to ai_eval
            if isinstance(node.func, ast.Name) and node.func.id == 'ai_eval':
                # Check if first argument is an f-string (JoinedStr)
                if node.args and isinstance(node.args[0], ast.JoinedStr):
                    fstring = node.args[0]
                    
                    # Extract all FormattedValue nodes (the {expr} parts)
                    formatted_values = []
                    for value in fstring.values:
                        if isinstance(value, ast.FormattedValue):
                            formatted_values.append(value)
                    
                    if formatted_values:
                        # Build variable mapping
                        var_mapping: Dict[str, ast.expr] = {}
                        var_counter = {}  # Track duplicates
                        
                        for fv in formatted_values:
                            # Convert the expression to source code
                            expr_str = ast.unparse(fv.value)
                            # Generate variable name
                            var_name = extract_variable_name(expr_str)
                            
                            # Handle duplicates by adding counter
                            if var_name in var_counter:
                                var_counter[var_name] += 1
                                var_name = f"{var_name}_{var_counter[var_name]}"
                            else:
                                var_counter[var_name] = 0
                            
                            var_mapping[var_name] = fv.value
                        
                        # Create new f-string with variable names instead of expressions
                        # IMPORTANT: This should be a REGULAR string, not an f-string!
                        # The variable names are just placeholders for formatting, not Python variables
                        new_values = []
                        var_index = 0
                        for value in fstring.values:
                            if isinstance(value, ast.FormattedValue):
                                # Get corresponding variable name
                                expr_str = ast.unparse(value.value)
                                var_name = extract_variable_name(expr_str)
                                # Handle counter for duplicates
                                if var_name in var_counter and var_counter[var_name] > 0:
                                    # Find which occurrence this is
                                    occurrence = sum(1 for i, fv in enumerate(formatted_values[:var_index]) 
                                                   if extract_variable_name(ast.unparse(fv.value)) == var_name)
                                    if occurrence > 0:
                                        var_name = f"{var_name}_{occurrence}"
                                
                                # Add the placeholder text (e.g., "{var_name}")
                                new_values.append(ast.Constant(value="{" + var_name + "}"))
                                var_index += 1
                            else:
                                # Keep constant string parts as-is
                                new_values.append(value)
                        
                        # Create a regular string by joining all parts
                        # Convert JoinedStr values to a single Constant string
                        string_parts = []
                        for val in new_values:
                            if isinstance(val, ast.Constant):
                                string_parts.append(str(val.value))
                            elif isinstance(val, ast.FormattedValue):
                                # This shouldn't happen with our new logic, but handle it
                                string_parts.append("{...}")
                        
                        template_string = "".join(string_parts)
                        new_arg = ast.Constant(value=template_string)
                        
                        # Create keyword arguments for the variables
                        keywords = [
                            ast.keyword(arg=var_name, value=expr)
                            for var_name, expr in var_mapping.items()
                        ]
                        
                        # Create new call with both the new template string and keyword args
                        new_call = ast.Call(
                            func=node.func,
                            args=[new_arg],
                            keywords=keywords
                        )
                        
                        self.modified = True
                        logger.info(f"Transformed ai_eval f-string with {len(var_mapping)} variables: {list(var_mapping.keys())}")
                        
                        return ast.copy_location(new_call, node)
            
            # Continue traversing
            return self.generic_visit(node)
    
    transformer = FStringTransformer()
    new_tree = transformer.visit(tree)
    
    if transformer.modified:
        # Unparse the modified AST back to code
        new_code = ast.unparse(new_tree)
        return new_code, True
    else:
        return code, False


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
    
    # Pass 1: Transform ai_eval f-strings to explicit variable passing
    code, modified = transform_ai_eval_fstrings(code)
    total_modified = total_modified or modified
    
    # Pass 2: Fix missing awaits
    code, modified = fix_missing_awaits(code, async_functions)
    total_modified = total_modified or modified
    
    # Pass 3: Fix tool ordering (TODO)
    if tool_info:
        code, modified = fix_tool_ordering(code, tool_info)
        total_modified = total_modified or modified
    
    return code, total_modified


__all__ = ["fix_missing_awaits", "fix_tool_ordering", "transform_ai_eval_fstrings", "apply_code_fixes"]
