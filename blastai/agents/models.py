"""
Data models for agents, tools, and states.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Literal, Callable
from enum import Enum
import json


class ToolExecutorType(str, Enum):
    """Type of tool executor."""
    SMCP = "smcp"
    CORE = "core"


class SMCPToolType(str, Enum):
    """Type of SMCP tool operation."""
    OBSERVE = "observe"
    GET_FIELDS = "getFields"
    LIST_ITEMS = "listItems"
    SET_FILTER = "setFilter"
    SET_FIELDS = "setFields"
    GOTO_FIELD = "gotoField"
    GOTO_ITEM = "gotoItem"


# AbstractState is a mapping from variable names to concrete values or patterns
# Patterns can be:
# - Concrete value (str, int, bool, etc.)
# - "*" - can be any value
# - "val1|val2|val3" - can be one of these values
# - "$fieldName" - must match input/output field with this name
AbstractState = Dict[str, Any]


@dataclass
class Tool:
    """
    Base tool representation.
    
    This is an easy-to-serialize object that references a tool.
    The actual execution happens elsewhere (in AgentExecutor).
    """
    name: str
    title: str  # Display name
    description: str
    input_schema: Dict[str, Any]  # JSON Schema
    output_schema: Dict[str, Any]  # JSON Schema
    tool_executor_type: ToolExecutorType
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = asdict(self)
        result['tool_executor_type'] = self.tool_executor_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tool':
        """Deserialize from dictionary."""
        # Determine the type based on tool_executor_type
        executor_type = ToolExecutorType(data['tool_executor_type'])
        
        if executor_type == ToolExecutorType.SMCP:
            return SMCPTool.from_dict(data)
        elif executor_type == ToolExecutorType.CORE:
            return CoreTool.from_dict(data)
        else:
            # Base Tool
            data_copy = data.copy()
            data_copy['tool_executor_type'] = executor_type
            return cls(**data_copy)


@dataclass
class SMCPTool(Tool):
    """
    SMCP (State Machine Control Protocol) Tool.
    
    These tools interact with web pages through JavaScript execution,
    with stabilization, execution, and checking phases.
    """
    lang: str = "js"  # Currently must be "js"
    
    # JavaScript code strings for execution phases
    is_ready: str = ""  # Returns true when ready, or [false, "reason"] when not ready
    is_correct: str = ""  # Returns true when valid, or [false, "reason"] when invalid
    execute: str = ""  # Main execution logic matching input/output schema
    
    # Pre-conditions
    pre_path: str = ""  # Glob pattern for URL matching
    pre: AbstractState = field(default_factory=dict)  # Pre-condition state
    
    # Post-conditions
    post: AbstractState = field(default_factory=dict)  # Post-condition state
    
    # Tool type determines behavior
    type: SMCPToolType = SMCPToolType.OBSERVE
    
    def __post_init__(self):
        """Initialize tool_executor_type."""
        if not isinstance(self.tool_executor_type, ToolExecutorType):
            self.tool_executor_type = ToolExecutorType.SMCP
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = asdict(self)
        result['tool_executor_type'] = self.tool_executor_type.value
        result['type'] = self.type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SMCPTool':
        """Deserialize from dictionary."""
        data_copy = data.copy()
        data_copy['tool_executor_type'] = ToolExecutorType(data_copy['tool_executor_type'])
        data_copy['type'] = SMCPToolType(data_copy['type'])
        return cls(**data_copy)


@dataclass
class CoreTool(Tool):
    """
    A core tool provided by the system (not user-defined SMCP).
    
    These tools are implemented in executor.py and provide essential
    capabilities like managing SMCP tools or inspecting page content.
    
    Valid names:
    - update_smcp_tool: Create/update an SMCP tool
    - remove_smcp_tool: Remove an SMCP tool
    - list_smcp_tools: List all SMCP tools
    - ask_html: Query page HTML for selectors/structure guidance
    """
    
    # Override parent fields with defaults (only name is required)
    title: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None
    tool_executor_type: ToolExecutorType = ToolExecutorType.CORE
    
    def __post_init__(self):
        """Validate CoreTool name."""
        valid_names = ["update_smcp_tool", "remove_smcp_tool", "list_smcp_tools", "ask_html"]
        if self.name not in valid_names:
            raise ValueError(f"CoreTool name must be one of {valid_names}, got {self.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = asdict(self)
        result['tool_executor_type'] = self.tool_executor_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoreTool':
        """Deserialize from dictionary."""
        data_copy = data.copy()
        data_copy['tool_executor_type'] = ToolExecutorType(data_copy['tool_executor_type'])
        return cls(**data_copy)


@dataclass
class Agent:
    """
    Agent with description and tools.
    
    Attributes:
        description: Agent's system prompt/description
        tools: List of available tools (SMCP and Core tools)
        is_ready_timeout_ms: Total time (in milliseconds) to spread 30 is_ready retry attempts.
                              Default is 30000ms (30s). Synthesis agents use 5000ms (5s) for faster iteration.
    """
    description: str = ""
    tools: List[Tool] = field(default_factory=list)
    is_ready_timeout_ms: int = 30000  # Time to spread 30 is_ready retries (default 30s)
    
    def add_tool(self, tool: Tool):
        """Add a tool to the agent."""
        self.tools.append(tool)
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name. Returns True if found and removed."""
        for i, tool in enumerate(self.tools):
            if tool.name == name:
                self.tools.pop(i)
                return True
        return False
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "description": self.description,
            "tools": [tool.to_dict() for tool in self.tools],
            "is_ready_timeout_ms": self.is_ready_timeout_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Deserialize from dictionary."""
        tools = [Tool.from_dict(tool_data) for tool_data in data.get("tools", [])]
        return cls(
            description=data.get("description", ""),
            tools=tools,
            is_ready_timeout_ms=data.get("is_ready_timeout_ms", 30000)
        )
    
    def to_json(self, filepath: str):
        """Save agent to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Agent':
        """Load agent from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def derive_synthesis_agent(self) -> 'Agent':
        """
        Derive a synthesis agent that can create SMCP tools.
        
        This creates a new agent with:
        1. Updated description with instructions to create SMCP tools
        2. All existing tools from this agent
        3. Core tools for updating/removing/listing SMCP tools
        
        The synthesis agent can use evaluate() to execute JavaScript for
        is_ready/execute/is_correct phases and create complete SMCP tools.
        """
        synthesis_description = self.description + """

Complete the TASK using ONLY SMCP actions.

<workflow>
1. Call `list_smcp_tools` to see available tools
2. If tool exists: Call it directly
3. If tool missing or exists but fails:
   a. If it exists, call `list_smcp_tools` with `get_code_for` set to the tool's name
   b. If it failed but partially progressed, reset.
   c. Optionally call `ask_html` for just the current tool
   d. Create tool with `update_smcp_tool`
   e. Then call the new tool
4. Loop until TASK complete
</workflow>

<ask_html_usage>
Use `ask_html` to query the current page HTML for specific guidance:

Examples: 
- {{"ask_html": {{"query": "What selector finds all restaurant cards and how to extract their IDs?"}}}}
- {{"ask_html": {{"query": "How to include the name of each restaurant?", "max_length": 200000}}}} (for larger pages)

Default max_length is 100000 chars (~100KB). Increase if page is large or truncation is affecting results.
</ask_html_usage>

<javascript_requirements>
is_ready, execute, and is_correct are FUNCTION BODIES ONLY:

✓ CORRECT:
   is_ready: "return !!document.body;"
   is_ready: "return document.body ? true : [false, 'Body not loaded'];"
   execute: "return {items: []};"
   is_correct: "return Array.isArray(output.items);"
   is_correct: "return output.items ? true : [false, 'Missing items array'];"

✗ WRONG:
   is_ready: "(function(){return !!document.body;})()"
   execute: "async function() { return {items: []}; }"

For async: "const data = await fetch(...); return data;"
</javascript_requirements>

<tool_types>
- observe: Returns state (page, selectedId, etc.) - ONE per domain
- listItems: Returns {items: [...]}
- getFields: Returns field values object
- setFilter: Changes filter/category
- setFields: Fills form fields  
- gotoItem: Navigates to item
- gotoField: Opens/focuses field
</tool_types>

<update_smcp_tool_required>
name: Tool identifier (e.g. "list_restaurants")
type: One of: observe, listItems, getFields, setFilter, setFields, gotoItem, gotoField
is_ready: JS function BODY that validates PRECONDITIONS before execute runs (e.g. for gotoItem: verify the item to click exists) (has access to `inputs` object)
execute: JS function BODY that performs the action (has access to `inputs`)
is_correct: JS function BODY that validates OUTPUT after execute runs (e.g. for gotoItem: verify navigation succeeded) (has access to `output`)
preconditions: AbstractState dict before running (e.g. {"page": "list"})
postconditions: AbstractState dict after running (e.g. {"page": "detail"})
input_parameters: Array of param names (e.g. ["restaurantId", "limit"]) or []
</update_smcp_tool_required>

<abstract_state>
AbstractState is a mapping from state variable names to concrete values or patterns
Patterns can be:
- Concrete value (str, int, bool, etc.)
- "*" - can be any value
- "val1|val2|val3" - can be one of these values
- "$fieldName" - must match input/output field with this name
</abstract_state>

<examples>
Here are examples of tools. Use them as reference but never copy them directly.

observe (no params):
  name: "observe_site"
  type: "observe"
  input_parameters: []
  is_ready: "return !!document.body;"
  execute: "const page = location.pathname === '/' ? 'home' : 'detail'; return {page};"
  is_correct: "return typeof output.page === 'string';"
  preconditions: {}
  postconditions: {}

listItems (no params):
  name: "list_items"
  type: "listItems"
  input_parameters: []
  is_ready: "const items = document.querySelectorAll('.item'); return items.length > 0 ? true : [false, 'No items found on page'];"
  execute: "const items = Array.from(document.querySelectorAll('.item')).map(el => ({id: el.id, name: el.textContent})); return {items};"
  is_correct: "return Array.isArray(output.items) ? true : [false, 'Output missing items array'];"
  preconditions: {}
  postconditions: {}

gotoItem (with params):
  name: "goto_restaurant"
  type: "gotoItem"
  input_parameters: ["restaurantId"]
  preconditions: {"page": "list"}
  postconditions: {"page": "detail", "selectedId": "$restaurantId"}
  is_ready: "const items = document.querySelectorAll('[data-restaurant-id]'); return items.length > 0 ? true : [false, 'Restaurant list not loaded'];"
  execute: "const link = document.querySelector(`[data-restaurant-id='${inputs.restaurantId}']`); if (link) link.click(); return {success: !!link};"
  is_correct: "return output.success === true ? true : [false, 'Navigation did not succeed'];"
</examples>

TASK: """
        
        # Create new agent with synthesis instructions
        synthesis_agent = Agent(
            description=synthesis_description,
            tools=self.tools.copy(),
            is_ready_timeout_ms=5000  # Fast 5s timeout for synthesis agents
        )
        
        # Add core tools for managing SMCP tools
        # These are just markers - the actual implementation is in executor.py
        core_tools = [
            CoreTool(name="update_smcp_tool"),
            CoreTool(name="remove_smcp_tool"),
            CoreTool(name="list_smcp_tools"),
            CoreTool(name="ask_html")
        ]
        
        for core_tool in core_tools:
            synthesis_agent.add_tool(core_tool)
        
        return synthesis_agent
