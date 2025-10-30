from copy import deepcopy
import ast
from typing import Optional, Dict, List, Set, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class BasicBlock:
    """Represents a basic block in the control flow graph."""
    bid: int
    stmts: List[ast.AST] = field(default_factory=list)
    calls: List[Tuple[str, ast.Call]] = field(default_factory=list)  # (function_name, call_node)
    prev: List[int] = field(default_factory=list)
    next: List[int] = field(default_factory=list)
    
    def is_empty(self) -> bool:
        return len(self.stmts) == 0


class CFGBuilder(ast.NodeVisitor):
    """Build control flow graph from AST, handling all Python constructs."""
    
    def __init__(self):
        self.block_counter = 0
        self.blocks: Dict[int, BasicBlock] = {}
        self.edges: Dict[Tuple[int, int], Optional[ast.AST]] = {}
        self.curr_block: Optional[BasicBlock] = None
        self.start_block: Optional[BasicBlock] = None
        self.loop_stack: List[BasicBlock] = []
        self.in_comprehension = False
        
    def new_block(self) -> BasicBlock:
        """Create a new basic block."""
        self.block_counter += 1
        block = BasicBlock(self.block_counter)
        self.blocks[self.block_counter] = block
        return block
    
    def add_edge(self, from_bid: int, to_bid: int, condition: Optional[ast.AST] = None):
        """Add an edge between blocks."""
        self.blocks[from_bid].next.append(to_bid)
        self.blocks[to_bid].prev.append(from_bid)
        self.edges[(from_bid, to_bid)] = condition
        return self.blocks[to_bid]
    
    def add_stmt(self, block: BasicBlock, stmt: ast.AST):
        """Add statement to block."""
        block.stmts.append(stmt)
    
    def build(self, tree: ast.AST) -> Tuple[BasicBlock, Dict[int, BasicBlock]]:
        """Build CFG from AST."""
        self.curr_block = self.new_block()
        self.start_block = self.curr_block
        self.visit(tree)
        return self.start_block, self.blocks
    
    def extract_calls_from_node(self, node: ast.AST):
        """Recursively extract function calls from any AST node."""
        if isinstance(node, ast.Await):
            if isinstance(node.value, ast.Call):
                func_name = self.get_function_name(node.value.func)
                if func_name:
                    self.curr_block.calls.append((func_name, node.value))
        elif isinstance(node, ast.Call):
            func_name = self.get_function_name(node.func)
            if func_name:
                self.curr_block.calls.append((func_name, node))
            # Check arguments for nested calls
            for arg in node.args:
                self.extract_calls_from_node(arg)
            for keyword in node.keywords:
                self.extract_calls_from_node(keyword.value)
        elif isinstance(node, ast.BinOp):
            self.extract_calls_from_node(node.left)
            self.extract_calls_from_node(node.right)
        elif isinstance(node, ast.UnaryOp):
            self.extract_calls_from_node(node.operand)
        elif isinstance(node, ast.Compare):
            self.extract_calls_from_node(node.left)
            for comp in node.comparators:
                self.extract_calls_from_node(comp)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                self.extract_calls_from_node(value)
        elif isinstance(node, ast.IfExp):
            # Handle ternary expressions
            self.extract_calls_from_node(node.test)
            self.extract_calls_from_node(node.body)
            self.extract_calls_from_node(node.orelse)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                self.extract_calls_from_node(elt)
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k:
                    self.extract_calls_from_node(k)
                self.extract_calls_from_node(v)
        elif isinstance(node, ast.Subscript):
            self.extract_calls_from_node(node.value)
            if isinstance(node.slice, ast.Index):
                self.extract_calls_from_node(node.slice.value)
            else:
                self.extract_calls_from_node(node.slice)
        elif isinstance(node, ast.Attribute):
            self.extract_calls_from_node(node.value)
    
    def visit_Expr(self, node: ast.Expr):
        """Visit expression statement - check for function calls."""
        self.extract_calls_from_node(node.value)
        self.add_stmt(self.curr_block, node)
        
    def visit_Assign(self, node: ast.Assign):
        """Visit assignment - check if RHS is a function call."""
        self.extract_calls_from_node(node.value)
        self.add_stmt(self.curr_block, node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Visit annotated assignment."""
        if node.value:
            self.extract_calls_from_node(node.value)
        self.add_stmt(self.curr_block, node)
    
    def visit_AugAssign(self, node: ast.AugAssign):
        """Visit augmented assignment (+=, -=, etc.)."""
        self.extract_calls_from_node(node.value)
        self.add_stmt(self.curr_block, node)
    
    def visit_Return(self, node: ast.Return):
        """Visit return statement."""
        if node.value:
            self.extract_calls_from_node(node.value)
        self.add_stmt(self.curr_block, node)
        # Create new unreachable block after return
        self.curr_block = self.new_block()
    
    def visit_If(self, node: ast.If):
        """Visit if statement."""
        # Extract calls from condition
        self.extract_calls_from_node(node.test)
        self.add_stmt(self.curr_block, node)
        
        # Create afterif block
        afterif_block = self.new_block()
        
        # Create if body block
        if_block = self.add_edge(self.curr_block.bid, self.new_block().bid, node.test)
        
        # Handle else
        if node.orelse:
            else_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
            self.curr_block = else_block
            for stmt in node.orelse:
                self.visit(stmt)
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, afterif_block.bid)
        else:
            self.add_edge(self.curr_block.bid, afterif_block.bid)
        
        # Handle if body
        self.curr_block = if_block
        for stmt in node.body:
            self.visit(stmt)
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, afterif_block.bid)
        
        self.curr_block = afterif_block
    
    def visit_For(self, node: ast.For):
        """Visit for loop."""
        # Extract calls from iterator
        self.extract_calls_from_node(node.iter)
        
        loop_block = self.new_block()
        self.add_edge(self.curr_block.bid, loop_block.bid)
        self.curr_block = loop_block
        self.add_stmt(self.curr_block, node)
        
        after_loop = self.new_block()
        self.add_edge(self.curr_block.bid, after_loop.bid)
        
        body_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
        self.loop_stack.append(after_loop)
        self.curr_block = body_block
        
        for stmt in node.body:
            self.visit(stmt)
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, loop_block.bid)
        
        # Handle else clause
        if node.orelse:
            else_block = self.new_block()
            self.add_edge(loop_block.bid, else_block.bid)
            self.curr_block = else_block
            for stmt in node.orelse:
                self.visit(stmt)
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, after_loop.bid)
        
        self.loop_stack.pop()
        self.curr_block = after_loop
    
    def visit_While(self, node: ast.While):
        """Visit while loop."""
        # Extract calls from test condition
        self.extract_calls_from_node(node.test)
        
        loop_block = self.new_block()
        self.add_edge(self.curr_block.bid, loop_block.bid)
        self.curr_block = loop_block
        self.add_stmt(self.curr_block, node)
        
        after_loop = self.new_block()
        self.add_edge(self.curr_block.bid, after_loop.bid)
        
        body_block = self.add_edge(self.curr_block.bid, self.new_block().bid, node.test)
        self.loop_stack.append(after_loop)
        self.curr_block = body_block
        
        for stmt in node.body:
            self.visit(stmt)
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, loop_block.bid)
        
        # Handle else clause
        if node.orelse:
            else_block = self.new_block()
            self.add_edge(loop_block.bid, else_block.bid)
            self.curr_block = else_block
            for stmt in node.orelse:
                self.visit(stmt)
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, after_loop.bid)
        
        self.loop_stack.pop()
        self.curr_block = after_loop
    
    def visit_Try(self, node: ast.Try):
        """Visit try/except."""
        self.add_stmt(self.curr_block, node)
        
        after_try = self.new_block()
        
        # Try body
        try_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
        self.curr_block = try_block
        for stmt in node.body:
            self.visit(stmt)
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, after_try.bid)
        
        # Exception handlers
        for handler in node.handlers:
            handler_block = self.add_edge(try_block.bid, self.new_block().bid)
            self.curr_block = handler_block
            for stmt in handler.body:
                self.visit(stmt)
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, after_try.bid)
        
        # Else clause
        if node.orelse:
            else_block = self.add_edge(try_block.bid, self.new_block().bid)
            self.curr_block = else_block
            for stmt in node.orelse:
                self.visit(stmt)
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, after_try.bid)
        
        # Finally clause
        if node.finalbody:
            finally_block = self.add_edge(after_try.bid, self.new_block().bid)
            self.curr_block = finally_block
            for stmt in node.finalbody:
                self.visit(stmt)
            after_finally = self.new_block()
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, after_finally.bid)
            self.curr_block = after_finally
        else:
            self.curr_block = after_try
    
    def visit_With(self, node: ast.With):
        """Visit with statement."""
        for item in node.items:
            self.extract_calls_from_node(item.context_expr)
        self.add_stmt(self.curr_block, node)
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_Break(self, node: ast.Break):
        """Visit break statement."""
        if self.loop_stack:
            self.add_edge(self.curr_block.bid, self.loop_stack[-1].bid)
        self.curr_block = self.new_block()
    
    def visit_Continue(self, node: ast.Continue):
        """Visit continue statement."""
        # Continue jumps back to loop guard (not implemented fully)
        self.curr_block = self.new_block()
    
    def visit_ListComp(self, node: ast.ListComp):
        """Visit list comprehension."""
        old_comp = self.in_comprehension
        self.in_comprehension = True
        # Extract calls from generators and element expression
        self.extract_calls_from_node(node.elt)
        for gen in node.generators:
            self.extract_calls_from_node(gen.iter)
            for if_clause in gen.ifs:
                self.extract_calls_from_node(if_clause)
        self.in_comprehension = old_comp
    
    def visit_SetComp(self, node: ast.SetComp):
        """Visit set comprehension."""
        old_comp = self.in_comprehension
        self.in_comprehension = True
        self.extract_calls_from_node(node.elt)
        for gen in node.generators:
            self.extract_calls_from_node(gen.iter)
            for if_clause in gen.ifs:
                self.extract_calls_from_node(if_clause)
        self.in_comprehension = old_comp
    
    def visit_DictComp(self, node: ast.DictComp):
        """Visit dict comprehension."""
        old_comp = self.in_comprehension
        self.in_comprehension = True
        self.extract_calls_from_node(node.key)
        self.extract_calls_from_node(node.value)
        for gen in node.generators:
            self.extract_calls_from_node(gen.iter)
            for if_clause in gen.ifs:
                self.extract_calls_from_node(if_clause)
        self.in_comprehension = old_comp
    
    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        """Visit generator expression."""
        old_comp = self.in_comprehension
        self.in_comprehension = True
        self.extract_calls_from_node(node.elt)
        for gen in node.generators:
            self.extract_calls_from_node(gen.iter)
            for if_clause in gen.ifs:
                self.extract_calls_from_node(if_clause)
        self.in_comprehension = old_comp
    
    def get_function_name(self, node: ast.AST) -> Optional[str]:
        """Extract function name from call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None