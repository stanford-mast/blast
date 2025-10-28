"""
Agent executor that integrates browser-use with SMCP and core tools.
"""

import asyncio
import os
import re
from typing import Dict, Any, Optional, Callable, List, TYPE_CHECKING
from pydantic import BaseModel, Field, create_model
import json
import logging

from browser_use import Browser, Agent as BrowserUseAgent, Tools, ActionResult
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage, UserMessage, AssistantMessage
from openai import OpenAI

from .local_python_executor import LocalPythonExecutor
from .codegen import CodeGenerator

from .models import Agent, Tool, SMCPTool, CoreTool, ToolExecutorType, SMCPToolType
from .tools_smcp import add_smcp_tool, execute_smcp_tool
from .tools_synthesis import add_core_tool

if TYPE_CHECKING:
    from browser_use.browser import BrowserSession

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Executes an Agent by integrating with browser-use.
    
    Handles:
    - Converting SMCP tools to browser-use actions
    - Managing core tools (update/remove SMCP tools)
    - Running in loop mode (browser-use) or code mode (LocalPythonExecutor)
    """
    
    def __init__(
        self, 
        agent: Agent, 
        llm: Optional[BaseChatModel] = None,
        browser: Optional['BrowserSession'] = None,
        state_aware: bool = True,
        codegen_llm: Optional[BaseChatModel] = None,
        parallel_codegen: int = 1
    ):
        """
        Initialize AgentExecutor.
        
        Args:
            agent: The Agent to execute
            llm: Optional LLM model for loop mode. If None, will create from environment.
            browser: Optional BrowserSession instance. If None, creates new one. 
                     Multiple Agents can share a BrowserSession instance.
            state_aware: Include preconditions/postconditions in generated code (default True)
            codegen_llm: Optional LLM for code generation. If None, uses same as llm.
            parallel_codegen: Number of parallel code generations (default 1)
        """
        self.agent = agent
        self.llm = llm or self._create_llm_from_env()
        self.codegen_llm = codegen_llm or self.llm
        self.browser = browser or self._create_browser()
        self.state_aware = state_aware
        self.parallel_codegen = parallel_codegen
        self.tools = Tools()
        
        # Storage for dynamically created tools
        self._dynamic_tools: Dict[str, Callable] = {}
        
        # Track which tool names are registered to avoid conflicts
        self._registered_tool_names: set[str] = set()
        
        # Browser-use agent will be created per run (can't be reused across tasks)
        self.browser_use_agent: Optional[BrowserUseAgent] = None
        
        # Set up tools
        self._setup_tools()
        
        # Code generator and Python executor for code mode (created lazily)
        self.code_generator: Optional[CodeGenerator] = None
        self.python_executor: Optional[LocalPythonExecutor] = None
    
    def _create_llm_from_env(self) -> BaseChatModel:
        """Create LLM from environment variables."""
        # Import here to avoid circular dependencies
        from browser_use.llm.openai.chat import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4.1")
        
        return ChatOpenAI(
            model=model,
            api_key=api_key
        )
    
    def _create_browser(self):
        """Create browser-use BrowserSession instance."""
        from browser_use import BrowserSession, BrowserProfile
        
        # Get viewport/window size from environment
        width = int(os.getenv("BROWSER_WIDTH", "1280"))
        height = int(os.getenv("BROWSER_HEIGHT", "720"))
        headless = os.getenv("HEADLESS", "false").lower() == "true"
        
        # Build Chrome args for WSL GPU fix
        args = []
        
        # Disable GPU if BROWSER_DISABLE_GPU=1 (for WSL transparency issues)
        if os.getenv("BROWSER_DISABLE_GPU", "").lower() in ("1", "true", "yes"):
            args.extend(["--disable-gpu", "--disable-gpu-sandbox"])
        
        # Create profile with proper viewport and window settings
        profile = BrowserProfile(
            headless=headless,
            # Viewport is what the browser *sees*
            viewport={"width": width, "height": height},
            # Window size is the OS window size (used when not headless)
            window_size={"width": width, "height": height},
            # Chrome command-line arguments
            args=args,
        )
        
        return BrowserSession(browser_profile=profile)
    
    def _setup_tools(self):
        """Set up all tools from the agent."""
        logger.info(f"Setting up {len(self.agent.tools)} tools from agent")
        for tool in self.agent.tools:
            logger.info(f"  Setting up tool: {tool.name} (type: {tool.tool_executor_type})")
            if tool.tool_executor_type == ToolExecutorType.SMCP:
                add_smcp_tool(self, tool)
            elif tool.tool_executor_type == ToolExecutorType.CORE:
                add_core_tool(self, tool)
    
    def _create_python_executor(self) -> LocalPythonExecutor:
        """Create Python executor for code mode."""
        # Create executor first so we can register wrapped tool functions that
        # assert preconditions against STATE and update STATE with postconditions.
        import_modules = {
            "json": __import__("json"),
            "re": __import__("re"),
            "asyncio": __import__("asyncio"),
        }

        executor = LocalPythonExecutor(
            additional_functions={},
            additional_imports=import_modules
        )

        # Initialize STATE dictionary in executor.state with keys from tool contracts
        # Collect all state keys referenced in pre/post and URL patterns
        state_keys = set()
        for t in self.agent.tools:
            if t.tool_executor_type == ToolExecutorType.SMCP:
                if getattr(t, "pre", None) and isinstance(t.pre, dict):
                    state_keys.update(t.pre.keys())
                if getattr(t, "post", None) and isinstance(t.post, dict):
                    state_keys.update(t.post.keys())
                # If tool has a URL pre_path, we use STATE["current_url"] for matching
                if getattr(t, "pre_path", None):
                    state_keys.add("current_url")

        # Initialize STATE as a dict with all keys set to None
        STATE = {key: None for key in sorted(state_keys)} if state_keys else {"page": None, "current_url": None}
        executor.state["STATE"] = STATE

        # Helper for runtime pattern checks
        def _check_pattern(state_val, pattern, param_vals: Dict[str, Any]):
            # None: skip if None means no constraint
            if pattern is None:
                return True
            # Non-string exact match
            if not isinstance(pattern, str):
                return state_val == pattern
            if pattern == "*":
                return state_val is not None
            if pattern == "":
                return not state_val
            if "|" in pattern:
                return state_val in pattern.split("|")
            if pattern.startswith("$"):
                ref = pattern[1:]
                # Compare to provided param value if present
                return param_vals.get(ref) == state_val
            return state_val == pattern

        # Build wrappers for SMCP tools and register them into executor.state
        wrapped_tools: Dict[str, Callable] = {}
        for t in self.agent.tools:
            if t.tool_executor_type != ToolExecutorType.SMCP:
                continue

            def make_tool_wrapper(tool: SMCPTool):
                async def wrapper(**kwargs):
                    """Wrapper that asserts preconditions against STATE and
                    updates STATE with the tool's result (postconditions).
                    """
                    # Access STATE from executor.state
                    STATE = executor.state["STATE"]

                    # Check URL pre_path if present
                    if getattr(tool, "pre_path", None):
                        pattern = tool.pre_path.replace("*", ".*")
                        cur = STATE.get("current_url")
                        if not cur or not re.match(pattern, cur):
                            raise AssertionError(f"Expected URL matching {tool.pre_path}, got {cur}")

                    # Check state preconditions
                    if getattr(tool, "pre", None) and isinstance(tool.pre, dict):
                        for key, pat in tool.pre.items():
                            if pat is None:
                                continue
                            ok = _check_pattern(STATE.get(key), pat, kwargs)
                            if not ok:
                                raise AssertionError(f"Precondition failed for {key}: expected {pat}, got {STATE.get(key)}")

                    # Call the actual SMCP tool implementation
                    result = await execute_smcp_tool(self, tool, kwargs)

                    # Update STATE with result dict contents when possible
                    try:
                        if isinstance(result, dict):
                            STATE.update(result)
                    except Exception:
                        # Best-effort update - don't fail on update
                        pass

                    # Ensure postconditions reflected in STATE (handle $param references)
                    if getattr(tool, "post", None) and isinstance(tool.post, dict):
                        for key, pat in tool.post.items():
                            if pat is None:
                                continue
                            if isinstance(pat, str) and pat.startswith("$"):
                                ref = pat[1:]
                                if ref in kwargs:
                                    STATE[key] = kwargs.get(ref)
                                else:
                                    # fallback to result value
                                    STATE[key] = (result or {}).get(ref, STATE.get(key))

                    return result

                wrapper.__name__ = tool.name
                return wrapper

            wrapped_tools[t.name] = make_tool_wrapper(t)

        # Register wrapped SMCP tools into executor state so generated code can call them
        executor.send_tools(wrapped_tools)

        # Add utility functions (these are available inside generated code)
        async def run(task: str, initial_url: Optional[str] = None):
            """
            Run browser-use agent with the given task.
            Creates a new Agent instance with same browser and tools.
            """
            # Create new agent with same tools but new task
            new_agent = Agent(
                description=self.agent.description,
                tools=self.agent.tools.copy(),
                is_ready_timeout_ms=self.agent.is_ready_timeout_ms
            )
            # Create executor with shared browser
            executor = AgentExecutor(new_agent, llm=self.llm, browser=self.browser)
            result = await executor.run(task, mode="loop", initial_url=initial_url)
            return result

        async def ask(prompt: str):
            """
            Ask the LLM a question.
            """
            messages = [UserMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            return response.completion

        async def goto(url: str):
            """
            Navigate to a given URL.
            """
            page = await self.browser.get_current_page()
            if page:
                await page.goto(url)
                # Wait a bit for page load
                await asyncio.sleep(1)
                return {"success": True, "url": url}
            return {"success": False, "error": "No active page"}

        additional_functions = {}
        additional_functions["run"] = run
        additional_functions["ask"] = ask
        additional_functions["goto"] = goto

        # Register utilities into executor
        executor.send_tools(additional_functions)

        # Return the prepared executor
        return executor
    
    async def run(self, task: str, mode: str = "loop", initial_url: Optional[str] = None) -> Any:
        """
        Run the agent with the given task.
        
        Args:
            task: The task description
            mode: "loop" for browser-use Agent.run, "code" for Python code generation
            initial_url: Optional initial URL to navigate to
            
        Returns:
            The result of execution
        """
        # Prepend agent description to task (with space if description exists)
        if self.agent.description:
            full_task = f"{self.agent.description} {task}"
        else:
            full_task = task
        
        if mode == "loop":
            return await self._run_loop_mode(full_task, initial_url)
        elif mode == "code":
            return await self._run_code_mode(full_task, initial_url)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'loop' or 'code'.")
    
    async def _run_loop_mode(self, task: str, initial_url: Optional[str] = None) -> Any:
        """
        Run in loop mode using browser-use Agent.run.
        
        Creates a new browser-use Agent for each run to ensure clean state.
        Multiple runs can share the same BrowserSession instance.
        
        If initial_url is provided, it's passed as initial_actions to the Agent,
        which automatically navigates before starting the main task.
        """
        try:
            # Build initial actions if URL provided
            initial_actions = None
            if initial_url:
                logger.info(f"Setting initial URL: {initial_url}")
                initial_actions = [{'navigate': {'url': initial_url, 'new_tab': False}}]
            
            # Append SMCP tool names to task
            smcp_tool_names = [t.name for t in self.agent.tools if t.tool_executor_type == ToolExecutorType.SMCP]
            if smcp_tool_names:
                task += " You may use " + ", ".join(smcp_tool_names) + "."
                logger.info(f"Added {len(smcp_tool_names)} SMCP tools to task prompt: {smcp_tool_names}")
            else:
                logger.info("No SMCP tools available to add to task prompt")
            
            logger.info(f"Final task for BrowserUseAgent: {task}")
            
            # Create a fresh browser-use agent for this run
            # Note: We can reuse the same BrowserSession instance across multiple agents
            self.browser_use_agent = BrowserUseAgent(
                task=task,
                llm=self.llm,
                browser=self.browser,
                tools=self.tools,
                initial_actions=initial_actions  # Agent will execute navigation before task
            )
            
            result = await self.browser_use_agent.run()
            return result
        finally:
            # Clean up agent reference (but keep browser)
            self.browser_use_agent = None
    
    async def _run_code_mode(self, task: str, initial_url: Optional[str] = None) -> Any:
        """
        Run in code mode using LLM code generation + LocalPythonExecutor.
        
        This mode:
        1. If initial_url provided, navigates there first
        2. Generates Python code using CodeGenerator (with parallel generation)
        3. Validates code before execution
        4. Executes code with LocalPythonExecutor
        5. If there's an error, refines code and retries
        6. Loops until success
        """
        # First, ensure browser is connected (don't wait for navigation yet)
        if not self.browser._cdp_client_root:
            logger.info("Starting browser connection in background...")
            # Start connection but don't wait - it will be ready when we need it
            connection_task = asyncio.create_task(self.browser.connect())
        else:
            connection_task = None
        
        # Create code generator and python executor lazily
        if self.code_generator is None:
            self.code_generator = CodeGenerator(
                agent=self.agent,
                llm=self.codegen_llm,
                num_candidates=self.parallel_codegen,
                state_aware=self.state_aware
            )
        
        if self.python_executor is None:
            self.python_executor = self._create_python_executor()
        
        # Now navigate to initial URL if provided
        if initial_url:
            logger.info(f"Navigating to initial URL: {initial_url}")
            try:
                # Wait for browser connection if it was just started
                if connection_task:
                    await connection_task
                    logger.debug("Browser connected")
                
                page = await self.browser.get_current_page()
                if page:
                    await page.goto(initial_url)
                    logger.debug(f"Navigation complete: {initial_url}")
                    # Set STATE current_url for code execution
                    if self.python_executor is not None:
                        # Some browser page objects expose .url
                        url = getattr(page, "url", None)
                        if url and "STATE" in self.python_executor.state:
                            self.python_executor.state["STATE"]["current_url"] = url
                else:
                    logger.warning("Could not get current page, using navigate_to instead")
                    await self.browser.navigate_to(initial_url)
            except Exception as e:
                logger.warning(f"Error during initial navigation: {e}")
        
        # Iterative code generation and execution
        # This history tracks the broader conversation: code generated -> execution error -> retry
        # Each code generation may have its own internal refinement iterations for validation errors
        conversation_history: List[BaseMessage] = []
        max_iterations = 10
        
        for iteration in range(max_iterations):
            logger.info(f"Code mode iteration {iteration + 1}/{max_iterations}")
            
            # Generate code with conversation history
            # Note: generate_code returns Optional[str], not CodeCandidate
            code = await self.code_generator.generate_code(
                task=task,
                history=conversation_history,
                error=None  # Error is in conversation history already
            )
            
            if not code:
                logger.error("Failed to generate valid code")
                # Can't proceed without valid code
                raise RuntimeError("Code generation failed - no valid code produced")
            
            logger.info(f"Generated code ({len(code)} chars)")
            
            # Add generated code to conversation history
            conversation_history.append(AssistantMessage(content=f"```python\n{code}\n```"))
            
            # Execute code
            result = await self.python_executor(code)
            
            if result.error:
                logger.error(f"Code execution error: {result.error}")
                
                # Add error to conversation history for next iteration
                error_message = f"Error during execution: {result.error}\n\nPlease fix the code."
                if result.logs:
                    error_message += f"\n\nLogs from execution:\n{result.logs}"
                
                conversation_history.append(UserMessage(content=error_message))
                continue
            
            # Success!
            logger.info(f"Code execution completed successfully")
            if result.logs:
                logger.info(f"Logs:\n{result.logs}")
            logger.info(f"Result: {result.output}")
            
            return result.output
        
        raise RuntimeError(f"Failed to complete task after {max_iterations} iterations")
    
    async def cleanup(self):
        """Clean up resources."""
        # Try to close the Browser if it has a close() coroutine
        if self.browser:
            try:
                close_fn = getattr(self.browser, "close", None)
                if close_fn and asyncio.iscoroutinefunction(close_fn):
                    await close_fn()
                elif close_fn and callable(close_fn):
                    # synchronous close
                    close_fn()
            except AttributeError:
                # Fallback: try to close browser-use agent if available
                pass

        # If a browser-use Agent was created, attempt to close its session
        if self.browser_use_agent:
            try:
                await self.browser_use_agent.close()
            except Exception:
                # Best-effort cleanup; don't raise
                pass
