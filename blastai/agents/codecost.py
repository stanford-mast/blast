"""
Code cost computation for ranking code candidates.

Provides functions to compute cost/quality scores for generated code.
Lower costs are better. Cost is based on estimated latency of tool calls
considering loop nesting levels via CFG traversal.

Loop depth calculation:
- For/while loops: Detected via CFG blocks containing ast.For/ast.While
- Comprehensions: Tracked via comprehension_depth in BasicBlock.calls
- Multi-generator comprehensions: depth = number of generators
- Total depth = loop_depth (from CFG) + comprehension_depth
- Cost multiplier = LOOP_MULTIPLIER ^ total_depth
- Function calls: Inline the called function's cost into the caller's context

Examples:
- await ai_eval("x"): 10s (depth 0)
- for x in items: await ai_eval("x"): 100s (loop_depth 1)
- [await ai_eval(x) for x in items]: 100s (comp_depth 1)
- for x in items: [await ai_eval(y) for y in x.children]: 1000s (depth 1+1=2)
- [await ai_eval(x) for x in items for y in subitems]: 1000s (comp_depth 2)
- async def f(): await tool(); for x in items: await f(): func cost × 10
"""

import ast
import logging
from typing import Dict, Set, Tuple, Optional
from .cfg_builder import CFGBuilder, BasicBlock

logger = logging.getLogger(__name__)


# Tool latency estimates in seconds (by SMCP tool type, not function name)
# NOTE: These are base latencies per tool call. They are multiplied by loop depth
# using LOOP_MULTIPLIER (e.g., a tool in a loop runs 10x per item).
# Function names are mapped to these types via the tool_name_to_type parameter
# in compute_code_cost(), or extracted from SMCP registry at runtime.
TOOL_LATENCIES: Dict[str, float] = {
    # SMCP navigation tools - 0.1s each (fast deterministic operations)
    'gotoItem': 0.1,
    'gotoField': 0.1,
    'setFilter': 0.1,
    'setFields': 0.1,
    'getFields': 0.1,
    'listItems': 0.1,
    
    # AI tools - expensive
    'ai_exec': 100.0,   # Browser-use sub-agent execution
    'ai_eval': 10.0,    # LLM evaluation
    'ask_human': 50.0,  # Human-in-the-loop interaction
    
    # TODO: Future optimization - predicate pushdown detection
    # When a filter tool (setFilter) is called before a list/iteration tool (listItems),
    # the subsequent iteration cost should be reduced since the result set is filtered.
    # Example: setFilter() then listItems() is cheaper than listItems() alone.
    # Currently we don't account for this.
    # Consider: (1) Track which tools are "filters" (setFilter, certain gotoItem patterns)
    # (2) Detect filter-then-list patterns in CFG
    # (3) Apply a discount factor to list tool cost when preceded by filter
    # This would properly reward cost-optimal code that uses predicate pushdown.
}

# Loop multiplier - assume each loop runs 10 times
LOOP_MULTIPLIER = 10


def compute_block_cost(
    block: BasicBlock, 
    loop_depth: int = 0, 
    tool_name_to_type: Dict[str, str] = None,
    user_func_costs: Dict[str, float] = None
) -> float:
    """
    Compute cost for all tool calls in a basic block.
    
    Args:
        block: Basic block with tool calls
        loop_depth: Current loop nesting depth (from for/while loops)
        tool_name_to_type: Mapping from tool function names to SMCP types
        user_func_costs: Mapping from user-defined function names to their base costs
        
    Returns:
        Total cost for this block in seconds
    """
    if tool_name_to_type is None:
        tool_name_to_type = {}
    if user_func_costs is None:
        user_func_costs = {}
    
    total_cost = 0.0
    
    for func_name, call_node, comp_depth in block.calls:
        # Total depth = loop depth from for/while + comprehension depth
        total_depth = loop_depth + comp_depth
        multiplier = LOOP_MULTIPLIER ** total_depth
        
        # Check if this is a user-defined function call
        if func_name in user_func_costs:
            # Inline the function's cost, multiplied by current depth
            cost = user_func_costs[func_name] * multiplier
            total_cost += cost
            continue
        
        # Map function name to SMCP type, then look up latency
        tool_type = tool_name_to_type.get(func_name, func_name)  # Fall back to func_name if not in mapping
        base_latency = TOOL_LATENCIES.get(tool_type, 0.0)
        cost = base_latency * multiplier
        total_cost += cost
    
    return total_cost


def traverse_cfg_for_cost(
    blocks: Dict[int, BasicBlock],
    start_block: BasicBlock,
    tool_name_to_type: Dict[str, str] = None,
    loop_depth: int = 0,
    visited: Set[int] = None,
    loop_headers: Set[int] = None,
    current_loop_header: int = None,
    user_func_costs: Dict[str, float] = None
) -> float:
    """
    Traverse CFG and compute total estimated cost.
    
    Uses depth-first traversal with loop detection. When entering a loop body
    (detected via back-edges), increases loop depth.
    
    Args:
        blocks: All basic blocks
        start_block: Block to start traversal from
        tool_name_to_type: Mapping from tool function names to SMCP types
        loop_depth: Current loop nesting depth
        visited: Set of visited block IDs (to prevent infinite loops)
        loop_headers: Set of block IDs that are loop headers (have back-edges)
        current_loop_header: The loop header we're currently inside (None if not in loop)
        user_func_costs: Mapping from user-defined function names to their base costs
        
    Returns:
        Total estimated latency cost in seconds
    """
    if visited is None:
        visited = set()
    
    if tool_name_to_type is None:
        tool_name_to_type = {}
    
    if user_func_costs is None:
        user_func_costs = {}
    
    if loop_headers is None:
        # Detect loop headers on first call
        # A block is a loop header if there's a path from the block back to itself
        # (true loop), not just any backward edge (which can be if/else merges)
        loop_headers = set()
        
        # Only blocks with for/while statements can be loop headers
        for bid, block in blocks.items():
            # Check if this block contains a For or While statement
            if block.stmts:
                for stmt in block.stmts:
                    if isinstance(stmt, (ast.For, ast.While)):
                        loop_headers.add(bid)
                        break
        
        # Build back-edge set: edges that point back to loop headers
        # These indicate which blocks are loop bodies
        back_edges = set()
        for bid, block in blocks.items():
            for next_bid in block.next:
                if next_bid in loop_headers and next_bid != bid:
                    # This is a back-edge if next_bid < bid (backward in CFG)
                    # or if next_bid is a loop header that we're jumping back to
                    back_edges.add((bid, next_bid))
    
    # Prevent infinite recursion
    if start_block.bid in visited:
        return 0.0
    
    visited.add(start_block.bid)
    
    # Cost for this block
    block_cost = compute_block_cost(start_block, loop_depth, tool_name_to_type, user_func_costs)
    
    # If no successors, return block cost
    if not start_block.next:
        return block_cost
    
    # Check if current block is a loop header
    is_loop_header = start_block.bid in loop_headers
    
    # For loop headers, we need special handling:
    # - First successor (idx=0) is after-loop (executed once after loop exits)
    # - Subsequent successors (idx>0) are loop body (executed LOOP_MULTIPLIER times)
    # - We should SUM these, not take max, because after-loop executes AFTER the loop body
    if is_loop_header:
        after_loop_cost = 0.0
        loop_body_cost = 0.0
        
        for idx, next_bid in enumerate(start_block.next):
            next_block = blocks[next_bid]
            
            if next_bid not in visited:
                if idx == 0:
                    # After-loop edge: keep same depth
                    successor_cost = traverse_cfg_for_cost(blocks, next_block, tool_name_to_type, loop_depth, visited.copy(), loop_headers, current_loop_header, user_func_costs)
                    after_loop_cost = successor_cost
                else:
                    # Loop body edge: increment depth
                    next_depth = loop_depth + 1
                    next_loop_header = start_block.bid
                    successor_cost = traverse_cfg_for_cost(blocks, next_block, tool_name_to_type, next_depth, visited.copy(), loop_headers, next_loop_header, user_func_costs)
                    loop_body_cost = max(loop_body_cost, successor_cost)  # If multiple loop bodies, take max
        
        return block_cost + loop_body_cost + after_loop_cost
    
    # For non-loop blocks, traverse all successor paths and take maximum cost path
    # (conservative estimate - assumes worst-case path for branches)
    max_successor_cost = 0.0
    for idx, next_bid in enumerate(start_block.next):
        next_block = blocks[next_bid]
        
        # If this is a forward edge to a block we haven't visited yet
        if next_bid not in visited:
            successor_cost = traverse_cfg_for_cost(blocks, next_block, tool_name_to_type, loop_depth, visited.copy(), loop_headers, current_loop_header, user_func_costs)
            max_successor_cost = max(max_successor_cost, successor_cost)
    
    return block_cost + max_successor_cost


def compute_code_cost(code: str, tools=None, tool_name_to_type: Dict[str, str] = None) -> float:
    """
    Compute cost/score for a code candidate based on estimated latency.
    Lower is better.
    
    Cost calculation:
    - Builds CFG from AST using cfg_builder
    - Finds all tool calls in each basic block
    - For user-defined functions: compute their cost and inline when called
    - Applies base latency for each tool type:
      - gotoItem, gotoField, setFilter, setFields, goto: 5s
      - ai_exec: 100s
      - ai_eval: 10s
      - Other tools: 0s
    - Multiplies by loop nesting level detected via CFG (10x per loop level)
    - Function calls in loops have their cost multiplied accordingly
    - Sums total estimated latency via CFG traversal
    
    Args:
        code: Generated Python code
        tools: List of Tool objects with name and type attributes for mapping
               function names to SMCP types for latency lookup
        tool_name_to_type: Optional dict mapping tool function names to SMCP types
               (used if tools parameter not provided)
    
    Returns:
        Cost score in seconds (estimated latency, lower is better)
    """
    try:
        tree = ast.parse(code)
        
        # Build CFG (reuses cfg_builder.py logic)
        cfg_builder = CFGBuilder()
        start_block, blocks = cfg_builder.build(tree)
        
        # Build mapping from tool name -> tool type for latency lookup
        # Priority: tool_name_to_type parameter > tools parameter
        name_to_type = tool_name_to_type or {}
        if tools and not tool_name_to_type:
            for tool in tools:
                name_to_type[tool.name] = tool.type
        
        # Find all function definitions and their body entry blocks
        # When multiple functions are defined in the same block, each gets a different
        # successor in the block.next list (in order of definition)
        # The LAST successor after all function defs is the continuation of module-level code
        func_name_to_body_block: Dict[str, int] = {}
        module_level_continuation_bid: Optional[int] = None
        
        for bid, block in blocks.items():
            if block.stmts:
                func_idx = 0
                num_func_defs = sum(1 for stmt in block.stmts if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)))
                for stmt in block.stmts:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Each function's body is the corresponding successor in block.next
                        if func_idx < len(block.next):
                            func_name_to_body_block[stmt.name] = block.next[func_idx]
                            func_idx += 1
                
                # If there are more successors than function defs, the last one is module-level continuation
                # This happens when there's code after function definitions at module level
                if bid == start_block.bid and num_func_defs > 0 and len(block.next) > num_func_defs:
                    module_level_continuation_bid = block.next[num_func_defs]
        
        # Compute cost for each user-defined function (without function inlining first)
        # This handles helper functions that only call tools, not other user functions
        user_func_costs: Dict[str, float] = {}
        
        # First pass: compute costs for functions that only call tools (not other user funcs)
        for func_name, body_bid in func_name_to_body_block.items():
            body_block = blocks.get(body_bid)
            if body_block:
                cost = traverse_cfg_for_cost(
                    blocks, body_block, name_to_type,
                    loop_depth=0, visited=set(), loop_headers=None,
                    current_loop_header=None, user_func_costs={}
                )
                user_func_costs[func_name] = cost
                logger.debug(f"Function '{func_name}' base cost: {cost}s")
        
        # Second pass: re-compute with inlined function costs
        # This handles functions that call other user functions
        for func_name, body_bid in func_name_to_body_block.items():
            body_block = blocks.get(body_bid)
            if body_block:
                cost = traverse_cfg_for_cost(
                    blocks, body_block, name_to_type,
                    loop_depth=0, visited=set(), loop_headers=None,
                    current_loop_header=None, user_func_costs=user_func_costs
                )
                user_func_costs[func_name] = cost
                logger.debug(f"Function '{func_name}' with inlining cost: {cost}s")
        
        # Find the main entry point
        # Priority:
        # 1. If there's module-level code (start_block has tool calls or statements that aren't just function defs), use that
        # 2. If there's module-level code after function definitions, use that continuation
        # 3. Look for a function named 'main' or 'run'
        # 4. Use the last defined async function (for scripts that are just one async function)
        main_func_name = None
        
        # Check if start_block has any non-function statements or tool calls
        has_module_level_code = len(start_block.calls) > 0
        if not has_module_level_code and start_block.stmts:
            # Check if there are statements besides function definitions
            for stmt in start_block.stmts:
                if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    has_module_level_code = True
                    break
        
        # If there's module-level code, traverse from start_block
        if has_module_level_code:
            cost_from_start = traverse_cfg_for_cost(
                blocks, start_block, name_to_type,
                user_func_costs=user_func_costs
            )
            return cost_from_start
        
        # Check for module-level continuation (code after function definitions)
        if module_level_continuation_bid is not None:
            # There's module-level code after function definitions
            # Compute cost from that block, plus any tool calls in start_block itself
            continuation_block = blocks.get(module_level_continuation_bid)
            if continuation_block:
                # First compute cost of tool calls in start_block (before the continuation)
                start_block_cost = compute_block_cost(start_block, 0, name_to_type, user_func_costs)
                # Then compute cost from continuation
                continuation_cost = traverse_cfg_for_cost(
                    blocks, continuation_block, name_to_type,
                    loop_depth=0, visited=set(), loop_headers=None,
                    current_loop_header=None, user_func_costs=user_func_costs
                )
                total_cost = start_block_cost + continuation_cost
                logger.debug(f"Module-level code total cost: {total_cost}s (start_block: {start_block_cost}s, continuation: {continuation_cost}s)")
                return total_cost
        
        # Look for a function named 'main' or 'run' first
        for name in ['main', 'run']:
            if name in func_name_to_body_block:
                main_func_name = name
                break
        
        # If no 'main' or 'run', use the last defined async function
        if main_func_name is None:
            for bid, block in blocks.items():
                if block.stmts:
                    for stmt in block.stmts:
                        if isinstance(stmt, ast.AsyncFunctionDef):
                            main_func_name = stmt.name
        
        # Compute cost from the main entry point with function inlining
        if main_func_name and main_func_name in func_name_to_body_block:
            body_bid = func_name_to_body_block[main_func_name]
            body_block = blocks.get(body_bid)
            if body_block:
                total_cost = traverse_cfg_for_cost(
                    blocks, body_block, name_to_type,
                    loop_depth=0, visited=set(), loop_headers=None,
                    current_loop_header=None, user_func_costs=user_func_costs
                )
                logger.debug(f"Main function '{main_func_name}' total cost with inlining: {total_cost}s")
                return total_cost
        
        # Fallback: traverse from start_block (old behavior)
        cost_from_start = traverse_cfg_for_cost(
            blocks, start_block, name_to_type,
            user_func_costs=user_func_costs
        )
        return cost_from_start
        
    except SyntaxError as e:
        logger.warning(f"Syntax error in code cost analysis: {e}")
        # Fall back to code length as cost
        return float(len(code))
    except Exception as e:
        logger.warning(f"Error in code cost analysis: {e}")
        # Fall back to code length as cost
        return float(len(code))


__all__ = ["compute_code_cost", "TOOL_LATENCIES", "LOOP_MULTIPLIER"]
