"""
Generate SMCP tools from agisdk.yaml tasks or custom prompts.

Usage:
    # From task YAML:
    python experiments/generate_tools.py <task_id> [--output <output_path>]
    
    # From custom prompt:
    python experiments/generate_tools.py --prompt "Create observe tool..." --url https://example.com --output tools.json
    
    # Simple test:
    python experiments/generate_tools.py --test

Examples:
    # Task mode:
    python experiments/generate_tools.py dashdish-1 --output experiments/tools/dashdish-1.json
    
    # Prompt mode:
    python experiments/generate_tools.py --prompt "Create observe tool to detect page state and tools to list/filter items" --url https://dashdish.com --output dashdish-tools.json
    
    # Test mode:
    python experiments/generate_tools.py --test
"""

import asyncio
import argparse
import yaml
import os
import sys
import logging
from pathlib import Path

# ⚠️ IMPORTANT: Enable standalone mode FIRST, before importing blastai
# This prevents blastai from capturing/filtering logging output
# Pass 'DEBUG' to see all browser-use logs (or set BLASTAI_LOG_LEVEL=DEBUG)
from blastai.agents.models import CoreTool
from blastai.logging_setup import enable_standalone_mode
enable_standalone_mode(browser_use_log_level="INFO")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from blastai.agents import Agent, AgentExecutor


async def generate_tools_from_prompt(prompt: str, initial_url: str, output_path: str):
    """
    Generate SMCP tools from a custom prompt describing the toolset to create.
    
    This is different from task-based generation:
    - Task mode: "Complete this task using SMCP tools" (may or may not create tools)
    - Prompt mode: "Create these specific tools" (focused on tool creation)
    
    Args:
        prompt: Free-form description of tools to create
        initial_url: URL to navigate to for testing tools
        output_path: Where to save the generated tools
    """
    print(f"\n{'='*60}")
    print(f"Generating tools from custom prompt")
    print(f"Initial URL: {initial_url}")
    print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
    print(f"{'='*60}\n")
    
    # Load existing tools from output file if it exists
    import os
    if os.path.exists(output_path):
        print(f"Loading existing tools from: {output_path}")
        base_agent = Agent.from_smcp_registry(output_path)
        print(f"Loaded {len(base_agent.tools)} existing SMCP tools")
    else:
        print("No existing tools file - starting from scratch")
        base_agent = Agent(
            description="",
            tools=[]
        )
    
    # Derive synthesis agent
    synthesis_agent = base_agent.derive_synthesis_agent()
    print(f"Created synthesis agent with {len(synthesis_agent.tools)} tools (including core tools)")
    
    # Create executor
    executor = AgentExecutor(synthesis_agent)
    
    # The prompt becomes the task - it describes WHAT tools to create, not WHAT task to complete
    task = prompt
    
    try:
        # Run synthesis agent
        print("Running synthesis agent...")
        result = await executor.run(task, mode="loop", initial_url=initial_url)
        
        print(f"\nSynthesis complete!")
        print(f"Result: {result}")
        
        # Get SMCP tools created (exclude core tools)
        smcp_tools = [
            tool for tool in synthesis_agent.tools
            if hasattr(tool, 'tool_executor_type') and tool.tool_executor_type.value == 'smcp'
        ]
        
        print(f"\nGenerated {len(smcp_tools)} SMCP tools:")
        for tool in smcp_tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Save only SMCP tools (not core tools) to output with empty description
        # (synthesis prompt is not needed when loading tools, only during creation)
        synthesis_agent.to_smcp_registry(output_path)
        print(f"\nSaved {len(smcp_tools)} SMCP tools to: {output_path}")
        
        return synthesis_agent
        
    except Exception as e:
        print(f"\nError during tool generation: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await executor.cleanup()


async def generate_tools_for_task(task_id: str, task_data: dict, output_path: str):
    """
    Generate SMCP tools for a specific task.
    
    Args:
        task_id: Task identifier
        task_data: Task data with initial_url and goal
        output_path: Where to save the generated tools
    """
    print(f"\n{'='*60}")
    print(f"Generating tools for task: {task_id}")
    print(f"Initial URL: {task_data.get('initial_url')}")
    print(f"Goal: {task_data.get('goal')}")
    print(f"{'='*60}\n")
    
    # Load existing tools from output file if it exists
    import os
    if os.path.exists(output_path):
        print(f"Loading existing tools from: {output_path}")
        base_agent = Agent.from_smcp_registry(output_path)
        print(f"Loaded {len(base_agent.tools)} existing SMCP tools")
    else:
        print("No existing tools file - starting from scratch")
        base_agent = Agent(
            description="",
            tools=[]
        )
    
    # Derive synthesis agent
    synthesis_agent = base_agent.derive_synthesis_agent()
    print(f"Created synthesis agent with {len(synthesis_agent.tools)} tools (including core tools)")
    
    # Create executor
    executor = AgentExecutor(synthesis_agent)
    
    # Task is just the goal - executor will handle initial_url separately
    task = f"Goal: {task_data.get('goal')}"
    initial_url = task_data.get('initial_url')
    
    try:
        # Run synthesis agent
        print("Running synthesis agent...")
        result = await executor.run(task, mode="loop", initial_url=initial_url)
        
        print(f"\nSynthesis complete!")
        print(f"Result: {result}")
        
        # Get SMCP tools created (exclude core tools)
        smcp_tools = [
            tool for tool in synthesis_agent.tools
            if hasattr(tool, 'tool_executor_type') and tool.tool_executor_type.value == 'smcp'
        ]
        
        print(f"\nGenerated {len(smcp_tools)} SMCP tools:")
        for tool in smcp_tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Save only SMCP tools (not core tools) to output with empty description
        # (synthesis prompt is not needed when loading tools, only during creation)
        synthesis_agent.to_smcp_registry(output_path)
        print(f"\nSaved {len(smcp_tools)} SMCP tools to: {output_path}")
        
        return synthesis_agent
        
    except Exception as e:
        print(f"\nError during tool generation: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await executor.cleanup()


async def test_browser():
    """
    Test ask_html functionality by loading a URL and querying the HTML.
    
    Usage:
        python experiments/generate_tools.py --test --prompt "Your question about the HTML" --url https://example.com
    
    This test will:
    1. Launch a browser and navigate to the URL
    2. Call ask_html with your question and print_html=True
    3. Print both the filtered HTML and LLM response
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Question to ask about the HTML")
    parser.add_argument("--url", required=True, help="URL to load")
    args, _ = parser.parse_known_args()
    
    print("\n" + "="*60)
    print("🧪 ask_html Test Mode")
    print("="*60)
    print(f"\nURL: {args.url}")
    print(f"Question: {args.prompt}")
    print("\nThis will show you:")
    print("  1. The filtered HTML sent to the LLM")
    print("  2. The LLM's response")
    print("="*60 + "\n")
    
    # Create a simple agent
    agent = Agent(description="", tools=[CoreTool(name="ask_html")])
    executor = AgentExecutor(agent)
    
    try:
        # Create a task that MUST call ask_html with print_html=True
        # Make it very explicit that we want to use the ask_html tool
        task = f'Use the ask_html tool (not evaluate!) to query: "{args.prompt}". Set print_html=True parameter.'
        
        print(f"🔄 Running task: {task}\n")
        
        # Run with navigation - this initializes the browser and executes the task
        result = await executor.run(
            task=task,
            mode="loop",
            initial_url=args.url
        )
        
        print("\n" + "="*60)
        print("TASK RESULT:")
        print("="*60)
        print(result)
        print("="*60 + "\n")
        
        print("✅ Test complete!\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await executor.cleanup()
        print("✓ Browser closed\n")
async def main():
    parser = argparse.ArgumentParser(
        description="Generate SMCP tools from tasks or custom prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from task YAML:
  python experiments/generate_tools.py dashdish-1 --output tools.json
  
  # Generate from custom prompt:
  python experiments/generate_tools.py --prompt "Create observe tool..." --url https://example.com --output tools.json
  
  # Test ask_html:
  python experiments/generate_tools.py --test --prompt "What selector finds restaurant links?" --url https://example.com
"""
    )
    
    parser.add_argument(
        "task_id",
        nargs="?",
        help="Task ID from YAML file (e.g., dashdish-1)"
    )
    parser.add_argument(
        "--prompt",
        help="Custom prompt describing tools to create (use with --url) OR question for --test mode"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test ask_html functionality (requires --prompt and --url)"
    )
    parser.add_argument(
        "--url",
        help="Initial URL (required with --prompt or --test)"
    )
    parser.add_argument(
        "--output",
        help="Output path for generated tools JSON",
        default=None
    )
    parser.add_argument(
        "--tasks-file",
        help="Path to tasks YAML file (for task mode)",
        default="experiments/tasks/agisdk/agisdk.yaml"
    )
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        if not args.prompt or not args.url:
            parser.error("--test requires both --prompt and --url")
        await test_browser()
        return
    
    # Handle prompt mode (tool generation)
    if args.prompt:
        if not args.url:
            parser.error("--url is required when using --prompt")
        if not args.output:
            parser.error("--output is required when using --prompt for tool generation")
        
        await generate_tools_from_prompt(
            prompt=args.prompt,
            initial_url=args.url,
            output_path=args.output
        )
        return
    
    # Handle task mode
    if not args.task_id:
        parser.print_help()
        print("\nError: Either task_id or --prompt is required (or use --test)")
        sys.exit(1)
    
    # Load tasks
    tasks_file = Path(args.tasks_file)
    if not tasks_file.exists():
        print(f"Error: Tasks file not found: {tasks_file}")
        sys.exit(1)
    
    with open(tasks_file, 'r') as f:
        tasks = yaml.safe_load(f)
    
    # Find task
    task_data = None
    for task in tasks:
        if task.get('id') == args.task_id:
            task_data = task
            break
    
    if task_data is None:
        print(f"Error: Task ID '{args.task_id}' not found in {tasks_file}")
        print(f"Available tasks: {[t.get('id') for t in tasks[:10]]}...")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default to experiments/tools/<task_id>.json
        output_dir = Path("experiments/tools")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{args.task_id}.json")
    
    # Generate tools
    await generate_tools_for_task(args.task_id, task_data, output_path)


if __name__ == "__main__":
    asyncio.run(main())
