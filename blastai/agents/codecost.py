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

Examples:
- await ai_eval("x"): 10s (depth 0)
- for x in items: await ai_eval("x"): 100s (loop_depth 1)
- [await ai_eval(x) for x in items]: 100s (comp_depth 1)
- for x in items: [await ai_eval(y) for y in x.children]: 1000s (depth 1+1=2)
- [await ai_eval(x) for x in items for y in subitems]: 1000s (comp_depth 2)
"""

import ast
import logging
from typing import Dict, Set, Tuple
from .cfg_builder import CFGBuilder, BasicBlock

logger = logging.getLogger(__name__)


# Tool latency estimates in seconds
TOOL_LATENCIES: Dict[str, float] = {
    # SMCP navigation tools - 5s each
    'gotoItem': 5.0,
    'gotoField': 5.0,
    'setFilter': 5.0,
    'setFields': 5.0,
    'goto': 5.0,
    
    # AI tools - expensive
    'ai_exec': 20.0,
    'ai_eval': 10.0,
    
    # Other tools - instant (0s)
    # Any tool not in this dict defaults to 0
}

# Loop multiplier - assume each loop runs 10 times
LOOP_MULTIPLIER = 10


def compute_block_cost(block: BasicBlock, loop_depth: int = 0, tool_name_to_type: Dict[str, str] = None) -> float:
    """
    Compute cost for all tool calls in a basic block.
    
    Args:
        block: Basic block with tool calls
        loop_depth: Current loop nesting depth (from for/while loops)
        tool_name_to_type: Mapping from tool function names to SMCP types
        
    Returns:
        Total cost for this block in seconds
    """
    if tool_name_to_type is None:
        tool_name_to_type = {}
    
    total_cost = 0.0
    
    for func_name, call_node, comp_depth in block.calls:
        # Total depth = loop depth from for/while + comprehension depth
        total_depth = loop_depth + comp_depth
        multiplier = LOOP_MULTIPLIER ** total_depth
        
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
    current_loop_header: int = None
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
        
    Returns:
        Total estimated latency cost in seconds
    """
    if visited is None:
        visited = set()
    
    if tool_name_to_type is None:
        tool_name_to_type = {}
    
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
    block_cost = compute_block_cost(start_block, loop_depth, tool_name_to_type)
    
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
                    successor_cost = traverse_cfg_for_cost(blocks, next_block, tool_name_to_type, loop_depth, visited.copy(), loop_headers, current_loop_header)
                    after_loop_cost = successor_cost
                else:
                    # Loop body edge: increment depth
                    next_depth = loop_depth + 1
                    next_loop_header = start_block.bid
                    successor_cost = traverse_cfg_for_cost(blocks, next_block, tool_name_to_type, next_depth, visited.copy(), loop_headers, next_loop_header)
                    loop_body_cost = max(loop_body_cost, successor_cost)  # If multiple loop bodies, take max
        
        return block_cost + loop_body_cost + after_loop_cost
    
    # For non-loop blocks, traverse all successor paths and take maximum cost path
    # (conservative estimate - assumes worst-case path for branches)
    max_successor_cost = 0.0
    for idx, next_bid in enumerate(start_block.next):
        next_block = blocks[next_bid]
        
        # If this is a forward edge to a block we haven't visited yet
        if next_bid not in visited:
            successor_cost = traverse_cfg_for_cost(blocks, next_block, tool_name_to_type, loop_depth, visited.copy(), loop_headers, current_loop_header)
            max_successor_cost = max(max_successor_cost, successor_cost)
    
    return block_cost + max_successor_cost


def compute_code_cost(code: str, tools=None) -> float:
    """
    Compute cost/score for a code candidate based on estimated latency.
    Lower is better.
    
    Cost calculation:
    - Builds CFG from AST using cfg_builder
    - Finds all tool calls in each basic block
    - Applies base latency for each tool type:
      - gotoItem, gotoField, setFilter, setFields, goto: 5s
      - ai_exec: 20s
      - ai_eval: 10s
      - Other tools: 0s
    - Multiplies by loop nesting level detected via CFG (10x per loop level)
    - Sums total estimated latency via CFG traversal
    
    Args:
        code: Generated Python code
        tools: List of Tool objects with name and type attributes for mapping
               function names to SMCP types for latency lookup
    
    Returns:
        Cost score in seconds (estimated latency, lower is better)
    """
    try:
        tree = ast.parse(code)
        
        # Build CFG (reuses cfg_builder.py logic)
        cfg_builder = CFGBuilder()
        start_block, blocks = cfg_builder.build(tree)
        
        # Build mapping from tool name -> tool type for latency lookup
        tool_name_to_type = {}
        if tools:
            for tool in tools:
                tool_name_to_type[tool.name] = tool.type
        
        # FIX: When there are multiple function definitions, CFG creates disconnected
        # components. The start_block only reaches the first function's CFG.
        # To get accurate cost, we need to traverse ALL blocks or find all entry points.
        # Simple fix: compute cost for ALL blocks (not just reachable from start_block)
        # This handles cases where helper functions come before the main async function.
        
        total_cost = 0.0
        visited_global = set()
        
        # Find all potential entry points (blocks with no predecessors or that are entry points)
        # For now, just traverse from start_block AND any orphaned blocks with tool calls
        cost_from_start = traverse_cfg_for_cost(blocks, start_block, tool_name_to_type)
        total_cost = max(total_cost, cost_from_start)
        
        # Find orphaned blocks (not reachable from start) that have tool calls
        reachable = set()
        def mark_reachable(bid):
            if bid in reachable or bid not in blocks:
                return
            reachable.add(bid)
            for next_bid in blocks[bid].next:
                mark_reachable(next_bid)
        mark_reachable(start_block.bid)
        
        # For any unreachable blocks with tool calls, traverse from them too
        for bid, block in blocks.items():
            if bid not in reachable and block.calls:
                # This block has tool calls but isn't reachable - likely in a separate function
                # Traverse from this block as a separate entry point
                cost_from_orphan = traverse_cfg_for_cost(blocks, block, tool_name_to_type, visited=set())
                total_cost = max(total_cost, cost_from_orphan)
        
        return total_cost
        
    except SyntaxError as e:
        logger.warning(f"Syntax error in code cost analysis: {e}")
        # Fall back to code length as cost
        return float(len(code))
    except Exception as e:
        logger.warning(f"Error in code cost analysis: {e}")
        # Fall back to code length as cost
        return float(len(code))


__all__ = ["compute_code_cost", "TOOL_LATENCIES", "LOOP_MULTIPLIER"]
