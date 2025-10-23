# blastai.agents Module

Agent system with SMCP (State Machine Control Protocol) tools for web automation via JavaScript execution in browser-use.

## Components

- **Agent**: Container for tools and description
- **SMCPTool**: JS-based tools with is_ready/execute/is_correct phases, URL preconditions, pre/post AbstractState
- **CoreTool**: Meta-tools (update_smcp_tool, remove_smcp_tool, list_smcp_tools, ask_html)
- **AgentExecutor**: Runs agents in loop mode (browser-use) or code mode (LLM codegen → LocalPythonExecutor)

## Quick Start

```python
from blastai.agents import Agent, AgentExecutor, SMCPTool, SMCPToolType

# Create and execute agent
agent = Agent(description="Web automation agent")
executor = AgentExecutor(agent, state_aware=True)
result = await executor.run("Complete this task", mode="loop", initial_url="https://example.com")

# Generate tools dynamically (adds update_smcp_tool, remove_smcp_tool, list_smcp_tools, ask_html)
synthesis_agent = agent.derive_synthesis_agent()

# Serialize/load
agent.to_json("agent.json")
loaded = Agent.from_json("agent.json")
```

## SMCP Tool Structure

Three phases (3 CDP calls via JS while loops):

1. **is_ready** (optional): Returns true when ready, or [false, "reason"]
2. **execute**: Main logic matching input/output schema  
3. **check** (optional): Returns true when valid, or [false, "reason"]

**Tool Types**: observe, getFields, listItems, setFilter, setFields, gotoField, gotoItem

**AbstractState**: `{"var": "value"}`, `{"var": "*"}`, `{"var": "a|b|c"}`, `{"var": "$fieldName"}`

**URL Preconditions**: `pre_path: "https://example.com/*"` (glob pattern)

## Scripts

Generate tools (writes agent+tools JSON):
```bash
python experiments/generate_tools.py dashdish-1 --output experiments/tools/dashdish.json
python experiments/generate_tools.py --prompt "My custom task" --output experiments/tools/dashdish.json
python experiments/generate_tools.py --prompt "My custom instructions" --output experiments/tools/dashdish.json
```

Evaluate performance (baseline vs. tools, optional --code-mode):
```bash
python experiments/evaluate_tools.py dashdish-1 --tools experiments/tools/dashdish-1.json --runs 3
```

## Execution Modes

- **Loop mode**: browser-use Agent.run() - tools registered as browser-use actions
- **Code mode**: LLM generates Python → LocalPythonExecutor with SMCP tools as functions

## Synthesis Agent

`Agent.derive_synthesis_agent()` returns agent with core tools that can create SMCP tools:

1. Use `evaluate()` to test JavaScript for is_ready/execute/check phases
2. Use `update_smcp_tool()` to create tools with input/output schemas, pre/post states
3. Use `list_smcp_tools()` to check existing tools and avoid duplication
4. Use `ask_html()` to query page HTML for selectors/structure (print_html=True shows filtered HTML)

Code mode prompt includes tool preconditions/postconditions when `state_aware=True` (default).

## Tasks

Pre-imported: `experiments/tasks/agisdk/agisdk.yaml` (558 REAL benchmark tasks)

Re-import:
```bash
cd experiments/tasks/agisdk
git clone https://github.com/agi-inc/agisdk.git
python import_real.py
```

## Environment

```bash
export OPENAI_API_KEY="key"
export OPENAI_MODEL="gpt-4.1"
export HEADLESS="false"  # show browser
```

## Logging

Standalone scripts need logging enabled FIRST (before importing blastai):

```python
from blastai.logging_setup import enable_standalone_mode
enable_standalone_mode()  # or enable_standalone_mode(browser_use_log_level="DEBUG")
```

Scripts already configured: `generate_tools.py`, `evaluate_tools.py`

````
