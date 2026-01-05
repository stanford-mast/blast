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
    is_completed: str = ""  # Returns true when valid, or [false, "reason"] when invalid
    execute: str = ""  # Main execution logic matching input/output schema
    
    # Pre-conditions
    pre_path: str = ""  # Glob pattern for URL matching
    pre: AbstractState = field(default_factory=dict)  # Pre-condition state
    
    # Post-conditions
    post: AbstractState = field(default_factory=dict)  # Post-condition state
    
    # Tool type determines behavior
    type: SMCPToolType = SMCPToolType.OBSERVE
    
    # Pre-tools: Maps input parameter names to lists of tool names that must be called first
    # This is a weak precondition that helps prevent invalid tool usage
    # Example: {"employee": ["get_timeoff_field_options"], "policy": ["get_timeoff_field_options"]}
    # Means employee and policy inputs require get_timeoff_field_options to be called first
    pre_tools: Dict[str, List[str]] = field(default_factory=dict)
    
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
    - ask_human_cli: Ask for human assistance via CLI stdin
    """
    
    # Override parent fields with defaults (only name is required)
    title: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None
    tool_executor_type: ToolExecutorType = ToolExecutorType.CORE
    
    def __post_init__(self):
        """Validate CoreTool name."""
        valid_names = ["update_smcp_tool", "remove_smcp_tool", "list_smcp_tools", "ask_html", "ask_human", "ask_human_cli"]
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
        """Serialize agent to dictionary (includes description, tools, settings)."""
        return {
            "description": self.description,
            "tools": [tool.to_dict() for tool in self.tools],
            "is_ready_timeout_ms": self.is_ready_timeout_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Deserialize agent from dictionary."""
        tools = [Tool.from_dict(tool_data) for tool_data in data.get("tools", [])]
        return cls(
            description=data.get("description", ""),
            tools=tools,
            is_ready_timeout_ms=data.get("is_ready_timeout_ms", 30000)
        )
    
    def to_json(self, filepath: str):
        """Save complete agent to JSON file (description, tools, settings)."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Agent':
        """Load complete agent from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_smcp_registry(self, filepath: str):
        """
        Save only SMCP tools to registry JSON file.
        
        This creates a registry file with empty description and only SMCP tools
        (excludes CoreTools). Use this for saving tool registries that will be
        loaded later via from_smcp_registry().
        """
        smcp_tools = [
            tool for tool in self.tools
            if hasattr(tool, 'tool_executor_type') and tool.tool_executor_type == ToolExecutorType.SMCP
        ]
        registry_data = {
            "description": "",
            "tools": [tool.to_dict() for tool in smcp_tools],
            "is_ready_timeout_ms": 30000
        }
        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    @classmethod
    def from_smcp_registry(cls, filepath: str) -> 'Agent':
        """
        Load agent from SMCP registry JSON file.
        
        Creates an Agent with only the SMCP tools from the registry.
        The agent will have empty description and default settings.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load tools (should only be SMCP tools in a registry)
        tools = [Tool.from_dict(tool_data) for tool_data in data.get("tools", [])]
        
        return cls(
            description="",
            tools=tools,
            is_ready_timeout_ms=data.get("is_ready_timeout_ms", 30000)
        )
    
    def derive_synthesis_agent(self) -> 'Agent':
        """
        Derive a synthesis agent that can create SMCP tools.
        
        This creates a new agent with:
        1. Updated description with instructions to create SMCP tools
        2. All existing tools from this agent
        3. Core tools for updating/removing/listing SMCP tools
        
        The synthesis agent can use evaluate() to execute JavaScript for
        is_ready/execute/is_completed phases and create complete SMCP tools.
        """
        synthesis_description = self.description + """

Complete the TASK using ONLY SMCP actions, unless rewinding or human assistance.

<workflow>
1. Call `list_smcp_tools` to see available tools
2. Observe
    a. If tool exists: Call it directly
    b. If tool missing or exists but fails or returns wrong state:
        i. If it exists, call `list_smcp_tools` with `get_code_for` set to the tool's name
        ii. Optionally call `ask_html` to know how to get AbstractState dict for just the current state (Do NOT attempt to add code to get state for states not visited yet)
        iii. Create tool with `update_smcp_tool`
        iv. Then call the new tool
3. Act
    a. If on a page that needs authentication and observe output correct (e.g. page=login): ask for human assistance.
    b. If tool exists and precondition matches currently observed state: Call it directly
    c. If tool missing or exists but fails:
        i. If it exists, call `list_smcp_tools` with `get_code_for` set to the tool's name
        ii. If it failed but partially progressed, rewind.
        iii. Optionally call `ask_html` for just the current tool
        iv. Create tool with `update_smcp_tool`
        v. Then call the new tool
6. Loop until TASK complete
</workflow>

<ask_html_usage>
Use `ask_html` to query the current page HTML for specific guidance:

Examples: 
- {{"ask_html": {{"query": "What selector finds all restaurant cards and how to extract their names?"}}}}
- {{"ask_html": {{"query": "What selector lets get me all the input fields' values and set them for currently open dialog?"}}}}
</ask_html_usage>

<javascript_requirements>
is_ready, execute, and is_completed are FUNCTION BODIES ONLY:

✓ CORRECT:
   is_ready: "return document.querySelector('#elementRequiredForExecute') ? true : [false, 'Body not loaded'];"
   execute: "return {items: []};"
   is_completed: "return document.querySelector('h3').textContent.includes(inputs.restaurantName) ? true : [false, 'Wrong restaurant loaded'];"

✗ WRONG:
   is_ready: "(function(){ })()"
   execute: "async function() {  }"

For async: "const data = await fetch(...); return data;"
</javascript_requirements>

<tool_types>
- observe: Returns state (page, selectedRestaurant, etc.) - ONE per domain
- listItems: Returns {items: [...]}
- getFields: Returns field values object
- setFilter: Sets a filter of items by something (e.g. search query, category) and applies it (e.g. submit, apply, no-op)
- setFields: Edits (e.g. fill a form) and/or proceeds (e.g. submits, approve/declines, no-op)
- gotoItem: Navigates to item - must work for any item returned by listItems (even if not visible in viewport)
- gotoField: Opens/focuses field
</tool_types>

<update_smcp_tool_required>
name: Tool identifier (e.g. "list_restaurants")
type: One of: observe, listItems, getFields, setFilter, setFields, gotoItem, gotoField
is_ready: JS function BODY that checks DOM to validate PRECONDITIONS before execute runs (e.g. for gotoItem: verify the item to click exists in DOM) (has access to `inputs` object)
execute: JS function BODY that performs the action (has access to `inputs`)
is_completed: JS function BODY that checks DOM to validate OUTPUT after execute runs (e.g. for gotoItem: verify navigation succeeded by checking DOM) (has access to `inputs` and `output`)
preconditions: AbstractState dict before running (e.g. {"page": "list"})
postconditions: AbstractState dict after running (e.g. {"page": "detail"})
input_parameters: Array of param names (e.g. ["restaurantName", "limit"]) or []
</update_smcp_tool_required>

<abstract_state>
AbstractState is a mapping from state variable names to concrete values or patterns
Patterns can be:
- Concrete value (str, int, bool, etc.)
- "*" - can be any non-null value
- "val1|val2|val3" - can be one of these values
- "$parameterName" - must match input/output parameter with this name
- "" - must be null
</abstract_state>

<examples>
Here are examples of tools. Use them as reference but NEVER copy them directly.

observe (no params):
  name: "observe_some_app"
  type: "observe"
  input_parameters: []
  is_ready: "return document.querySelector('#someAppHeader') ? true : [false, 'App header not loaded yet'];"
  execute: "const page = location.pathname === '/' ? 'home' : 'detail'; return {page};"
  is_completed: "return true;"
  preconditions: {}
  postconditions: {}

listItems (no params):
  name: "list_items"
  type: "listItems"
  input_parameters: []
  is_ready: "const items = document.querySelectorAll('.item'); return items.length > 0 ? true : [false, 'No items found on page'];"
  execute: "const items = Array.from(document.querySelectorAll('.item')).map(el => ({name: el.textContent})); return {items};"
  is_completed: "return true;"
  preconditions: {}
  postconditions: {}

gotoItem (with params):
  name: "goto_product"
  type: "gotoItem"
  input_parameters: ["productName"]
  preconditions: {"page": "list"}
  postconditions: {"page": "detail", "selectedProduct": "$productName"}
  is_ready: "const link = document.querySelector(`.product-link:contains('${inputs.productName}')`); return link ? true : [false, 'Product link not found'];"
  execute: "const items = document.querySelectorAll('.product-link'); const link = Array.from(items).find(el => el.textContent.includes(inputs.productName)); if (link) link.click(); return {success: !!link};"
  is_completed: "return document.querySelector('.product-detail h1').textContent.includes(inputs.productName) ? true : [false, 'Detail page not loaded'];"
</examples>

TASK: """
        
        # Create new agent with synthesis instructions
        synthesis_agent = Agent(
            description=synthesis_description,
            tools=self.tools.copy(),
            is_ready_timeout_ms=15000  # Fast 5s timeout for synthesis agents
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
