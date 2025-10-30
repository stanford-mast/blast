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


def compute_block_cost(block: BasicBlock, loop_depth: int = 0) -> float:
    """
    Compute cost for all tool calls in a basic block.
    
    Args:
        block: Basic block with tool calls
        loop_depth: Current loop nesting depth (from for/while loops)
        
    Returns:
        Total cost for this block in seconds
    """
    total_cost = 0.0
    
    for func_name, call_node, comp_depth in block.calls:
        # Total depth = loop depth from for/while + comprehension depth
        total_depth = loop_depth + comp_depth
        multiplier = LOOP_MULTIPLIER ** total_depth
        
        base_latency = TOOL_LATENCIES.get(func_name, 0.0)
        cost = base_latency * multiplier
        total_cost += cost
    
    return total_cost


def traverse_cfg_for_cost(
    blocks: Dict[int, BasicBlock],
    start_block: BasicBlock,
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
        loop_depth: Current loop nesting depth
        visited: Set of visited block IDs (to prevent infinite loops)
        loop_headers: Set of block IDs that are loop headers (have back-edges)
        current_loop_header: The loop header we're currently inside (None if not in loop)
        
    Returns:
        Total estimated latency cost in seconds
    """
    if visited is None:
        visited = set()
    
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
    
    # Prevent infinite recursion
    if start_block.bid in visited:
        return 0.0
    
    visited.add(start_block.bid)
    
    # Cost for this block
    block_cost = compute_block_cost(start_block, loop_depth)
    
    # If no successors, return block cost
    if not start_block.next:
        return block_cost
    
    # Traverse all successor paths and take maximum cost path
    # (conservative estimate - assumes worst-case path)
    max_successor_cost = 0.0
    for next_bid in start_block.next:
        next_block = blocks[next_bid]
        
        # If this is a forward edge to a block we haven't visited yet
        if next_bid not in visited:
            next_depth = loop_depth
            next_loop_header = current_loop_header
            
            # Increment depth only when entering a NEW loop header
            # (not when re-entering the current loop or continuing in same loop)
            if next_bid in loop_headers and next_bid != current_loop_header:
                next_depth = loop_depth + 1
                next_loop_header = next_bid
            
            successor_cost = traverse_cfg_for_cost(blocks, next_block, next_depth, visited.copy(), loop_headers, next_loop_header)
            max_successor_cost = max(max_successor_cost, successor_cost)
    
    return block_cost + max_successor_cost


def compute_code_cost(code: str) -> float:
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
    
    Returns:
        Cost score in seconds (estimated latency, lower is better)
    """
    try:
        tree = ast.parse(code)
        
        # Build CFG (reuses cfg_builder.py logic)
        cfg_builder = CFGBuilder()
        start_block, blocks = cfg_builder.build(tree)
        
        # Traverse CFG to compute cost
        cost = traverse_cfg_for_cost(blocks, start_block)
        
        return cost
        
    except SyntaxError as e:
        logger.warning(f"Syntax error in code cost analysis: {e}")
        # Fall back to code length as cost
        return float(len(code))
    except Exception as e:
        logger.warning(f"Error in code cost analysis: {e}")
        # Fall back to code length as cost
        return float(len(code))


__all__ = ["compute_code_cost", "TOOL_LATENCIES", "LOOP_MULTIPLIER"]
