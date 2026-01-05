"""
Code validation and verification for generated Python code.

Provides functions to check code syntax and safety before execution.
Includes CFG-based validation to ensure tool ordering respects preconditions.
Includes ty type checking for full type safety (Rust-based, extremely fast ~9ms startup).
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


def find_callable_tools(current_state: Dict[str, Any], tools: List[Any]) -> List[str]:
    """
    Find all tools whose preconditions are satisfied by the current state.
    
    Args:
        current_state: Current abstract state
        tools: List of tool objects (Agent.tools)
        
    Returns:
        List of tool names that can be called
    """
    import logging
    from .models import ToolExecutorType, SMCPToolType
    
    logger = logging.getLogger(__name__)
    callable_tools = []
    
    logger.debug(f"Finding callable tools for state: {current_state}")
    
    for tool in tools:
        if not hasattr(tool, 'name'):
            continue
            
        # Skip CORE tools (update_smcp_tool, remove_smcp_tool, etc.)
        if hasattr(tool, 'tool_executor_type') and tool.tool_executor_type == ToolExecutorType.CORE:
            continue
        
        # Skip observe tools
        if hasattr(tool, 'type') and tool.type == SMCPToolType.OBSERVE:
            continue
            
        # Skip special utility tools
        # NOTE: goto is intentionally NOT skipped here. If code uses goto(url),
        # it should fail validation since we can't verify that the dynamic URL
        # will satisfy preconditions for subsequent tool calls.
        if tool.name in ('ask_human', 'ask_human_cli', 'ai_exec', 'ai_eval', 'get_url'):
            continue
        
        # Check if all preconditions are satisfied
        # Use 'pre' attribute (not 'preconditions') to match CFG validation
        pre = getattr(tool, 'pre', None) or {}
        if not pre:
            # No preconditions - always callable
            logger.debug(f"  {tool.name}: NO PRECONDITIONS (always callable)")
            callable_tools.append(tool.name)
            continue
        
        # Check each precondition
        all_satisfied = True
        for key, pattern in pre.items():
            state_value = current_state.get(key)
            matches = matches_pattern(state_value, pattern, {})
            logger.debug(f"  {tool.name}: precondition {key}={pattern}, state has {key}={state_value}, matches={matches}")
            if not matches:
                all_satisfied = False
                break
        
        if all_satisfied:
            logger.debug(f"  {tool.name}: ALL PRECONDITIONS SATISFIED")
            callable_tools.append(tool.name)
    
    logger.debug(f"Callable tools result: {callable_tools}")
    return callable_tools


def enhance_validation_error(
    error_msg: str,
    current_state: Dict[str, Any],
    tools: List[Any],
    has_ask_human: bool = False
) -> str:
    """
    Enhance validation error with suggestions for which tools to call.
    
    Args:
        error_msg: Original validation error message
        current_state: Current abstract state
        tools: List of tool objects (Agent.tools)
        has_ask_human: Whether ask_human tool is available
        
    Returns:
        Enhanced error message with suggestions
    """
    # Find tools that can be called from current state
    callable_tools = find_callable_tools(current_state, tools)
    
    # Build suggestion
    if callable_tools:
        tools_str = ", ".join(callable_tools)
        if has_ask_human:
            suggestion = f"Try calling {tools_str}, ask_human, or ai_exec first"
        else:
            suggestion = f"Try calling {tools_str} or ai_exec first"
    else:
        # No tools satisfy preconditions - ONLY suggest ask_human/ai_exec
        # Do NOT list tools that don't match preconditions
        if has_ask_human:
            suggestion = "No tool's precondition satisfies the current state. Use ask_human or ai_exec"
        else:
            suggestion = "No tool's precondition satisfies the current state. Use ai_exec"
    
    # Debug: log what we found
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Callable tools from state {current_state}: {callable_tools}")
    
    return f"{error_msg}. {suggestion}"


def check_tool_ordering(
    blocks: Dict[int, BasicBlock],
    start_block: BasicBlock,
    initial_state: Dict[str, Any],
    tools_by_name: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate tool ordering via CFG traversal with abstract state tracking.
    
    Also validates pre_tools dependencies - ensures that if a tool requires
    other tools to be called first (to get valid input values), those tools
    have been called earlier in the execution path.
    
    Recursively validates tool ordering inside user-defined functions.
    
    Args:
        blocks: Dictionary of basic blocks
        start_block: Starting block
        initial_state: Initial abstract state
        tools_by_name: Dictionary mapping tool names to tool objects (with pre/post, pre_tools)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Build mapping from function names to their body entry blocks
    func_name_to_body_block: Dict[str, int] = {}
    for bid, block in blocks.items():
        if block.stmts:
            func_idx = 0
            for stmt in block.stmts:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Each function's body is the corresponding successor in block.next
                    if func_idx < len(block.next):
                        func_name_to_body_block[stmt.name] = block.next[func_idx]
                        func_idx += 1
    
    # Track paths we've explored (block_id -> visit_count)
    visit_counts: Dict[int, int] = {}
    max_loop_iterations = 2
    # Sentinel value to indicate state is unknown (after ai_exec/ask_human)
    # This is different from empty state {} which means "no state has been set yet"
    STATE_UNKNOWN = object()
    
    def traverse(block: BasicBlock, state: Dict[str, Any], path: List[str], tools_called: Set[str], state_is_unknown: bool = False) -> Tuple[bool, Optional[str]]:
        """Recursively traverse CFG and validate tool ordering."""
        # Prevent infinite loops
        visit_counts[block.bid] = visit_counts.get(block.bid, 0) + 1
        if visit_counts[block.bid] > max_loop_iterations:
            return True, None  # Stop this path
        
        # Check tool calls in this block
        for func_name, call_node, comp_depth in block.calls:
            # Special handling for ai_exec and ask_human variants - resets state
            if func_name in ("ai_exec", "ask_human", "ask_human_cli"):
                # After an AI-assisted subtask or a human-in-the-loop interaction,
                # the abstract state is effectively reset because inputs/assumptions
                # may change. Allow subsequent tools to run without prior preconditions.
                state = {}  # Reset state after ai_exec/ask_human
                state_is_unknown = True  # Mark that we don't know the state anymore
                path.append(f"{func_name}(... ) [state reset]")
                continue
            
            # Check if this is a user-defined function that needs validation
            if func_name in func_name_to_body_block:
                # Recursively validate the user-defined function's body
                func_body_bid = func_name_to_body_block[func_name]
                func_body_block = blocks.get(func_body_bid)
                if func_body_block:
                    # Validate the function body with current state and tools_called
                    # Pass the ACTUAL state and tools_called (not copies) so that
                    # tool calls inside the function update the outer scope's state
                    func_path = path + [f"{func_name}() [entering]"]
                    # Note: We pass state and tools_called directly (not copies) because
                    # tools inside the function modify the global STATE, so changes should propagate
                    valid, error = traverse(func_body_block, state, func_path, tools_called, state_is_unknown)
                    if not valid:
                        return False, error
                path.append(f"{func_name}() [user-defined]")
                continue
            
            # Check if this is a known tool
            if func_name not in tools_by_name:
                # Unknown tool - skip validation (might be utility function)
                path.append(f"{func_name}(...) [unknown]")
                continue
            
            tool = tools_by_name[func_name]
            
            # Validate pre_tools dependencies (must be called before this tool)
            pre_tools = tool.get('pre_tools', {})
            if pre_tools:
                # pre_tools is Dict[str, List[str]] - param name -> required tool names
                # Check all required tools have been called
                all_required_tools = set()
                for param_name, required_tool_names in pre_tools.items():
                    all_required_tools.update(required_tool_names)
                
                for required_tool in all_required_tools:
                    if required_tool not in tools_called:
                        error_msg = (
                            f"Pre-tools validation error: {func_name} requires {required_tool} to be called first. "
                            f"Call sequence: {' -> '.join(path + [func_name])}"
                        )
                        return False, error_msg
            
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
            
            # Validate pattern constraints for constant string parameters
            param_patterns = tool.get('param_patterns', {})
            if param_patterns:
                for param_name, param_value in params.items():
                    if param_name in param_patterns and isinstance(param_value, str):
                        pattern = param_patterns[param_name]
                        if not re.match(pattern, param_value):
                            error_msg = (
                                f"Pattern validation error: {func_name} parameter '{param_name}' "
                                f"does not match required pattern. Expected pattern='{pattern}', got value='{param_value}'. "
                                f"Call sequence: {' -> '.join(path + [func_name])}"
                            )
                            return False, error_msg
            
            # Check preconditions
            # SPECIAL CASE: If state_is_unknown (after ask_human/ai_exec), skip precondition checks
            # because the state is unknown (wildcard) - human could have changed anything
            pre = tool.get('pre', {})
            if pre and not state_is_unknown:  # Check preconditions unless state is unknown
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
            
            # After a tool sets postconditions, we know the state again
            if post:
                state_is_unknown = False
            
            # Add to path and tools_called tracking
            path.append(func_name)
            tools_called.add(func_name)
        
        # If no next blocks, path is complete
        if not block.next:
            return True, None
        
        # Explore all outgoing edges
        for next_bid in block.next:
            next_block = blocks[next_bid]
            # Create copy of state, path, and tools_called for this path
            next_state = deepcopy(state)
            next_path = path.copy()
            next_tools_called = tools_called.copy()
            
            valid, error = traverse(next_block, next_state, next_path, next_tools_called, state_is_unknown)
            if not valid:
                return False, error
        
        return True, None
    
    # Start traversal with empty tools_called set, state is NOT unknown initially
    valid, error = traverse(start_block, deepcopy(initial_state), [], set(), state_is_unknown=False)
    
    # Reset visit counts
    visit_counts.clear()
    
    return valid, error


def check_basedpyright_types(full_code: str) -> Tuple[bool, Optional[str]]:
    """
    Run basedpyright type checking on the full code (definition + generated).
    
    Creates a temporary file with the code and runs basedpyright on it.
    Basedpyright is 3-5x faster than mypy and has better type checking capabilities.
    
    Args:
        full_code: Complete Python code including definitions and generated code
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Create a temporary directory with both the code file and a config file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            code_path = temp_dir_path / "code.py"
            config_path = temp_dir_path / "pyrightconfig.json"
            
            # Write the code
            code_path.write_text(full_code)
            
            # Write basedpyright config to balance strictness with LLM-generated code
            # Key goals:
            # 1. Catch wrong attribute access (e.g., details.price when price is nested)
            # 2. Catch dict access on Pydantic models (e.g., details["items"] instead of details.items)
            # 3. Catch type mismatches (e.g., int assigned to str field)
            # 4. Allow unannotated functions (LLMs don't always add annotations)
            # 5. Allow classes without __init__ (using dataclass/pydantic patterns)
            config_content = {
                "typeCheckingMode": "basic",  # Basic mode, not strict
                "reportMissingImports": "error",
                "reportUndefinedVariable": "error",
                "reportAttributeAccessIssue": "error",  # Catch wrong attr access
                "reportIndexIssue": "error",  # Catch dict access on non-dict types
                "reportAssignmentType": "error",  # Catch type mismatches
                "reportCallIssue": "none",  # Allow flexible function calls (dataclass constructors)
                "reportGeneralTypeIssues": "error",
                "reportUninitializedInstanceVariable": "none",  # Allow dataclass/pydantic patterns
                "reportUnknownParameterType": "none",  # Allow unannotated parameters
                "reportUnknownArgumentType": "none",  # Allow unannotated arguments
                "reportUnknownVariableType": "none",  # Allow unannotated variables
                "reportUnknownMemberType": "none",  # Allow unannotated class members
                "reportMissingParameterType": "none",  # Don't require param annotations
                "reportMissingTypeArgument": "none",  # Allow generic types without args
                "reportOptionalMemberAccess": "error",  # Catch optional access issues
                "reportOptionalSubscript": "error",
                "reportOptionalCall": "error",
            }
            
            import json
            config_path.write_text(json.dumps(config_content, indent=2))
            
            # Run basedpyright with the config
            result = subprocess.run(
                ['basedpyright',
                 '--outputjson',
                 '--project', str(config_path),
                 str(code_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return True, None
            else:
                # Parse basedpyright JSON output
                try:
                    output = json.loads(result.stdout)
                    errors = []
                    
                    for diag in output.get('generalDiagnostics', []):
                        if diag['severity'] == 'error':
                            line = diag['range']['start']['line'] + 1  # Convert to 1-based
                            col = diag['range']['start']['character'] + 1
                            msg = diag['message']
                            rule = diag.get('rule', '')
                            rule_str = f" [{rule}]" if rule else ""
                            errors.append(f"Line {line}, Col {col}: {msg}{rule_str}")
                    
                    if not errors:
                        # Fallback to raw output if no structured errors found
                        errors_str = result.stdout.strip() or result.stderr.strip()
                        errors_str = errors_str.replace(str(code_path), "<code>")
                        return False, f"Type checking errors:\n{errors_str}"
                    
                    return False, f"Type checking errors:\n" + "\n".join(errors)
                except json.JSONDecodeError:
                    # Fallback to raw output if JSON parsing fails
                    errors_str = result.stdout.strip() or result.stderr.strip()
                    errors_str = errors_str.replace(str(code_path), "<code>")
                    return False, f"Type checking errors:\n{errors_str}"
            
    except FileNotFoundError:
        # basedpyright not installed - skip type checking
        logger.debug("basedpyright not found, skipping type checking")
        return True, None
    except subprocess.TimeoutExpired:
        return False, "Type checking timed out"
    except Exception as e:
        logger.warning(f"Type checking failed with exception: {e}")
        # Don't fail validation if basedpyright check fails
        return True, None


def check_ty_types(full_code: str) -> Tuple[bool, Optional[str]]:
    """
    Run ty type checking on the full code (definition + generated).
    
    Creates a temporary file with the code and runs ty (Astral's Rust-based type checker) on it.
    ty is extremely fast (~9ms startup) and has excellent Pydantic support.
    
    Args:
        full_code: Complete Python code including definitions and generated code
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Create a temporary directory with both the code file and a config file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            code_path = temp_dir_path / "code.py"
            pyproject_path = temp_dir_path / "pyproject.toml"
            
            # Write the code
            code_path.write_text(full_code)
            
            # Write ty config to balance strictness with LLM-generated code
            # Key goals:
            # 1. Catch wrong attribute access (e.g., details.price when price is nested)
            # 2. Catch dict access on Pydantic models (e.g., details["items"] instead of details.items)
            # 3. Catch type mismatches (e.g., int assigned to str field)
            # 4. Allow possibly-unresolved references in LLM patterns (e.g., Optional narrowing in comprehensions)
            # 5. Allow possibly-missing attributes/imports (LLM code often has conditional definitions)
            #
            # Rules configuration:
            # - Error level: unresolved-attribute (wrong attrs), invalid-key (dict access on Pydantic),
            #                invalid-assignment (type mismatches), invalid-argument-type
            # - Ignore level: possibly-* rules (too strict for LLM code with conditionals)
            #                 invalid-return-type (gives false positives on code with early returns)
            config_content = """
[tool.ty.rules]
# Error on definite type errors
unresolved-attribute = "error"
invalid-key = "error"
invalid-assignment = "error"
invalid-argument-type = "error"
non-subscriptable = "error"
call-non-callable = "error"
missing-argument = "error"
unknown-argument = "error"

# Ignore return type errors (false positives with early returns)
invalid-return-type = "ignore"

# Ignore "possibly" errors (too strict for LLM code patterns)
possibly-unresolved-reference = "ignore"
possibly-missing-attribute = "ignore"
possibly-missing-import = "ignore"
possibly-missing-implicit-call = "ignore"
"""
            pyproject_path.write_text(config_content)
            
            # Run ty with concise output for easier parsing
            # Use --project to specify directory (ty will find pyproject.toml)
            result = subprocess.run(
                ["ty", "check", "--project", str(temp_dir_path), "--output-format", "concise", str(code_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # ty exits with code 1 if there are errors
            if result.returncode != 0:
                # Parse concise output (format: file:line:col: [rule] message)
                error_lines = []
                for line in result.stdout.strip().split('\n'):
                    if line and ':' in line:
                        # Extract line number and message from concise format
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            line_num = parts[1]
                            message = parts[3].strip()
                            error_lines.append(f"Line {line_num}: {message}")
                        else:
                            error_lines.append(line)
                
                if error_lines:
                    return False, "\n".join(error_lines)
                else:
                    # Fallback if parsing fails - just return stdout
                    return False, result.stdout.strip() or "Type checking failed"
            
            # No errors found
            return True, None
            
    except subprocess.TimeoutExpired:
        return False, "Type checking timed out (>10s)"
    except FileNotFoundError:
        logger.debug("ty not found, skipping type checking")
        return True, None  # Skip silently if not installed
    except Exception as e:
        logger.warning(f"Type checking failed with exception: {e}")
        return True, None  # Don't fail validation if ty check fails


def check_missing_imports(tree: ast.AST) -> Optional[str]:
    """
    Check for common missing imports by detecting attribute access on known stdlib modules.
    
    This catches cases like:
    - re.search(...) without 'import re'
    - json.dumps(...) without 'import json'
    - asyncio.run(...) without 'import asyncio'
    
    Args:
        tree: Parsed AST of the code
    
    Returns:
        Error message if missing import detected, None otherwise
    """
    # Collect all imported module names
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Handle 'import foo' and 'import foo as bar'
                imported_modules.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split('.')[0])
    
    # Modules automatically provided by the executor namespace (from coderun.py)
    # These are imported in create_python_executor and don't need import statements
    EXECUTOR_PROVIDED = {
        're', 'json', 'asyncio', 'math', 'statistics', 'typing'
    }
    
    # Common Python stdlib modules (lenient - don't flag these)
    COMMON_STDLIB = {
        'os', 'sys', 'random', 'datetime', 'time', 'copy', 'functools',
        'pathlib', 'collections', 'itertools', 'operator', 'string', 
        'decimal', 'pickle', 'sqlite3', 'csv', 'logging', 'warnings',
        'traceback', 'inspect', 'gc', 'weakref', 'urllib', 'http',
        'tempfile', 'shutil', 'glob', 'fnmatch', 'gzip', 'zipfile',
        'hashlib', 'hmac', 'secrets', 'uuid', 'base64', 'binascii'
    }
    
    # All modules we're lenient about (executor-provided + common stdlib)
    ALLOWED_WITHOUT_IMPORT = EXECUTOR_PROVIDED | COMMON_STDLIB
    
    # Find attribute access on potential module names
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            # Check if accessing attribute on a Name node (e.g., re.search)
            if isinstance(node.value, ast.Name):
                module_name = node.value.id
                
                # Heuristic: skip if it looks like a variable name (snake_case with multiple underscores)
                # Module names are typically lowercase without underscores (re, json, os)
                # or single-word (math, sys, etc.)
                if '_' in module_name:
                    # Skip - likely a variable like restaurants_result, filter_result, etc.
                    continue
                
                # Only flag if:
                # 1. Not in allowed list (executor-provided or stdlib)
                # 2. Not explicitly imported
                # This catches things like pandas.DataFrame, requests.get, etc.
                if module_name not in ALLOWED_WITHOUT_IMPORT and module_name not in imported_modules:
                    return f"Missing import: '{module_name}' is used but not imported (found '{module_name}.{node.attr}')"


    
    return None


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
    2. Forbidden STATE.update detection
    3. Optional basedpyright type checking (if definition_code provided)
    4. CFG construction
    5. Tool ordering validation via CFG traversal with precondition/postcondition checking
    
    Note: Import checking is NOT performed because the executor provides common modules
    (re, json, asyncio, math, statistics, typing) in its namespace automatically.
    
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
    
    # 2. Check for forbidden STATE.update calls
    # Users should not manually update STATE - tools do it automatically
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for STATE.update(...) pattern
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'STATE' and node.func.attr == 'update':
                    return False, "Do not update STATE in your code, instead you may call the given functions which update STATE automatically"
    
    # 3. Missing import check DISABLED
    # The executor provides common modules (re, json, asyncio, math, statistics, typing)
    # in its namespace, so code doesn't need explicit imports for these.
    # Type checking below will catch any actual missing imports.
    
    # 4. Optional basedpyright type checking if we have definition code
    if definition_code is not None:
        # Wrap user code in async function to allow await statements
        # Use Any return type since generated code may return different types
        wrapped_code = f"""
async def _user_generated_code() -> Any:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
"""
        full_code = definition_code + "\n\n" + wrapped_code
        is_valid, type_error = check_ty_types(full_code)
        if not is_valid:
            return False, type_error
    
    # 4. Build CFG and validate tool ordering if we have agent context
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
                    'param_names': [],
                    'param_patterns': {},  # Track pattern constraints for parameters
                    'pre_tools': {}  # Track pre_tools dependencies
                }
                
                # Extract parameter names and pattern constraints if available
                if hasattr(tool, 'input_schema') and tool.input_schema:
                    properties = tool.input_schema.get('properties', {})
                    tool_info['param_names'] = list(properties.keys())
                    # Extract pattern constraints
                    for param_name, param_schema in properties.items():
                        if isinstance(param_schema, dict) and 'pattern' in param_schema:
                            tool_info['param_patterns'][param_name] = param_schema['pattern']
                
                # Extract pre_tools if available
                if hasattr(tool, 'pre_tools'):
                    tool_info['pre_tools'] = getattr(tool, 'pre_tools', {})
                
                tools_by_name[tool.name] = tool_info
            
            # 5. Validate tool ordering
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