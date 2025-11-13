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
from .coderun import create_python_executor

from .models import Agent, Tool, SMCPTool, CoreTool, ToolExecutorType, SMCPToolType
from .tools_smcp import add_smcp_tool, execute_smcp_tool
from .tools_synthesis import add_core_tool
from .execution_hooks import ExecutionHooks, StopExecutionError

# Apply browser-use patches early, before any browser-use tools are created
from ..browser_use_patches import apply_all_patches
apply_all_patches()

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
        parallel_codegen: int = 1,
        output_model_schema: Optional[type[BaseModel]] = None,
        user_id: Optional[str] = None,
        timezone: Optional[str] = None,
        stop_if_codegen_fails: bool = False,
        allowed_domains: Optional[List[str]] = None,
        send_message_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        check_stop_callback: Optional[Callable[[], bool]] = None,
        session_id: Optional[str] = None,
        cycle_id: Optional[int] = None,
        summarizer_llm: Optional[BaseChatModel] = None
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
            output_model_schema: Optional Pydantic model for structured output (passed to browser-use Agent)
            user_id: Optional user identifier for persistent browser profiles. If set and browser=None,
                     creates a browser with user_data_dir specific to this user, persisting cookies/sessions.
            timezone: Optional timezone string in IANA format (e.g., 'America/Los_Angeles', 'Europe/London').
                     Used to determine current date/time for code generation context. Defaults to UTC.
            stop_if_codegen_fails: If True, raise error when code generation fails. 
                     If False (default), fall back to loop mode.
            allowed_domains: Optional list of allowed domains for browser navigation. 
                     Auto-derived from initial URL if not specified. Example: ["*.sage.hr"]
            send_message_callback: Optional callback for sending messages (AgentThought, RequestForHuman, etc.)
                     Signature: async def callback(session_id: str, message: Dict[str, Any])
            check_stop_callback: Optional callback to check if execution should stop
                     Signature: def callback() -> bool
            session_id: Optional session ID for message routing
            cycle_id: Optional cycle ID to include in messages
            summarizer_llm: Optional LLM for summarizing agent thoughts. Defaults to codegen_llm.
        """
        self.agent = agent
        self.llm = llm or self._create_llm_from_env()
        self.codegen_llm = codegen_llm or self._create_codegen_llm()
        self.summarizer_llm = summarizer_llm or self.codegen_llm
        self.allowed_domains = allowed_domains
        self.user_id = user_id
        self.timezone = timezone or "UTC"
        self.stop_if_codegen_fails = stop_if_codegen_fails
        self.browser = browser or self._create_browser()
        self.state_aware = state_aware
        self.parallel_codegen = parallel_codegen
        self.output_model_schema = output_model_schema
        self.tools = Tools()
        
        # Callbacks for DBOS workflow integration
        self.send_message_callback = send_message_callback
        self.check_stop_callback = check_stop_callback
        self.session_id = session_id
        self.cycle_id = cycle_id
        
        # Step counter for tracking execution progress
        self.step_count = 0
        
        # History for code mode follow-up requests
        self.history: List[BaseMessage] = []
        
        # Storage for dynamically created tools
        self._dynamic_tools: Dict[str, Callable] = {}
        
        # Track which tool names are registered to avoid conflicts
        self._registered_tool_names: set[str] = set()
        
        # Browser-use agent will be reused for follow-up requests
        self.browser_use_agent: Optional[BrowserUseAgent] = None
        
        # Set up tools
        self._setup_tools()
        
        # Code generator and Python executor for code mode (created lazily)
        self.code_generator: Optional[CodeGenerator] = None
        self.python_executor: Optional[LocalPythonExecutor] = None
    
    def _create_llm_from_env(self) -> BaseChatModel:
        """Create LLM from environment variables."""
        from .llm_factory import LLMFactory
        
        # Use LLMFactory to support multiple providers
        # Looks for BLASTAI_MODEL, BLASTAI_PROVIDER, etc.
        # Falls back to OPENAI_MODEL for backwards compatibility
        model = os.getenv("BLASTAI_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1")
        provider = os.getenv("BLASTAI_PROVIDER")  # Optional explicit provider
        
        return LLMFactory.create_llm(
            model_name=model,
            provider=provider,
            temperature=0.5  # non-zero temperature for variation
        )
    
    def _create_codegen_llm(self) -> BaseChatModel:
        """Create LLM specifically for code generation (uses Llama-4 Maverick by default)."""
        from .llm_factory import LLMFactory
        
        # Use dedicated codegen model if specified, otherwise use Llama-4 Maverick
        model = os.getenv("BLASTAI_CODEGEN_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
        provider = os.getenv("BLASTAI_CODEGEN_PROVIDER")  # Optional provider override
        
        return LLMFactory.create_llm(
            model_name=model,
            provider=provider,
            temperature=0.0  # Zero temperature for deterministic code generation
        )
    
    def _create_browser(self):
        """Create browser-use BrowserSession instance with optional user-specific profile."""
        from browser_use import BrowserSession, BrowserProfile
        from pathlib import Path
        
        # Get viewport/window size from environment
        width = int(os.getenv("BROWSER_WIDTH", "1280"))
        height = int(os.getenv("BROWSER_HEIGHT", "720"))
        headless = os.getenv("HEADLESS", "false").lower() == "true"

        # Build Chrome args ensuring GPU disabled (alignment with page load measurement path)
        args = ["--disable-gpu", "--disable-gpu-sandbox"]
        # Allow additional custom args via env var (comma-separated)
        custom_args = os.getenv("BROWSER_EXTRA_ARGS")
        if custom_args:
            for a in custom_args.split(","):
                clean = a.strip()
                if clean:
                    args.append(clean)
        
        # Determine user_data_dir for persistent profiles
        user_data_dir = None
        if self.user_id:
            # Create user-specific profile directory
            # Format: ./blast-profiles/user-{user_id}
            profiles_root = Path(os.getenv("BLASTAI_PROFILES_DIR", "./blast-profiles"))
            user_data_dir = str(profiles_root / f"user-{self.user_id}")
            logger.info(f"Using persistent browser profile for user_id={self.user_id}: {user_data_dir}")
        
        # Create profile with proper viewport and window settings
        profile = BrowserProfile(
            headless=headless,
            viewport={"width": width, "height": height},
            window_size={"width": width, "height": height},
            args=args,
            user_data_dir=user_data_dir,
            allowed_domains=self.allowed_domains,
            keep_alive=True,
            # Provide small network idle waits to reduce flakiness on first page interactions
            wait_for_network_idle_page_load_time=float(os.getenv("BROWSER_NETWORK_IDLE_WAIT", "1.5")),
            minimum_wait_page_load_time=float(os.getenv("BROWSER_MINIMUM_WAIT", "0.25"))
        )
        
        return BrowserSession(browser_profile=profile)

    async def _ensure_browser_started(self):
        """Robustly ensure browser session is started. Adds timeout, health check, and clearer logging.

        Reasons:
        - Prior implementation relied on internal attribute _cdp_client_root which may change.
        - Occasional freezes observed during code mode initial navigation.
        - This wraps start with a timeout and validates we can get a page instance.
        """
        if getattr(self.browser, "_started", False):
            return
        start_timeout = float(os.getenv("BROWSER_START_TIMEOUT", "20"))
        logger.info(f"Starting browser (timeout={start_timeout}s)...")
        try:
            await asyncio.wait_for(self.browser.start(), timeout=start_timeout)
        except asyncio.TimeoutError:
            logger.error("Browser start timed out")
            raise
        except Exception as e:
            logger.error(f"Browser start failed: {e}")
            raise
        # Health check
        try:
            page = await asyncio.wait_for(self.browser.get_current_page(), timeout=10)
            if not page:
                raise RuntimeError("Browser page object not available after start")
            logger.info("Browser started successfully and page object acquired")
            setattr(self.browser, "_started", True)
        except Exception as e:
            logger.error(f"Browser health check failed after start: {e}")
            raise
    
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
        return create_python_executor(
            agent=self.agent,
            browser=self.browser,
            llm=self.llm,
            agent_executor=self
        )
    
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
        # Auto-derive allowed_domains from initial_url if not specified
        if self.allowed_domains is None and initial_url:
            from urllib.parse import urlparse
            parsed = urlparse(initial_url)
            if parsed.hostname:
                # Extract base domain (e.g., "ourera.sage.hr" -> "sage.hr")
                parts = parsed.hostname.split('.')
                if len(parts) >= 2:
                    base_domain = '.'.join(parts[-2:])  # Last two parts (sage.hr)
                    self.allowed_domains = [f"*.{base_domain}"]
                    logger.info(f"Auto-derived allowed_domains from initial URL: {self.allowed_domains}")
                else:
                    self.allowed_domains = [f"*.{parsed.hostname}"]
        
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
        
        Reuses browser-use Agent for follow-up requests.
        Uses add_new_task() for new tasks on existing agent.
        
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
            
            # Check if ask_human or ask_human_cli is available
            has_ask_human = any(
                hasattr(t, 'name') and t.name in ('ask_human', 'ask_human_cli')
                for t in self.agent.tools
            )
            if has_ask_human:
                task += "\nUse ask_human if stuck or unauthenticated or task turned out to be ambiguous."
                logger.info("Added ask_human prompt injection")
            
            logger.info(f"Final task for BrowserUseAgent: {task}")
            
            # Reuse existing browser-use agent if available, otherwise create new one
            if self.browser_use_agent is None:
                logger.info("Creating new browser-use agent")
                self.browser_use_agent = BrowserUseAgent(
                    task=task,
                    llm=self.llm,
                    browser=self.browser,
                    tools=self.tools,
                    initial_actions=initial_actions,  # Agent will execute navigation before task
                    output_model_schema=self.output_model_schema,  # Pass structured output schema
                    step_timeout=3600  # 1 hour timeout - hooks handle stop checking via check_stop_callback
                )
            else:
                # Reuse existing agent with new task
                logger.info("Reusing existing browser-use agent with add_new_task")
                self.browser_use_agent.add_new_task(task)
            
            # Define thought extractor for hooks
            def get_thought(agent) -> str:
                """Extract current thought from browser-use agent."""
                import random
                try:
                    if hasattr(agent, 'state') and agent.state and hasattr(agent.state, 'last_model_output'):
                        model_output = agent.state.last_model_output
                        
                        if model_output:
                            # Extract fields from browser-use's AgentOutput
                            evaluation_previous_goal = model_output.evaluation_previous_goal
                            memory = model_output.memory
                            
                            # Build candidates list
                            candidates = []
                            
                            if memory:
                                candidates.append(memory)
                            
                            if evaluation_previous_goal:
                                # Ignore if ends with "Verdict: Failure"
                                if not evaluation_previous_goal.strip().endswith("Verdict: Failure"):
                                    # Trim "Verdict: Success" if present
                                    cleaned_eval = evaluation_previous_goal.strip()
                                    cleaned_eval = cleaned_eval.replace("Verdict: Success.", "").replace("Verdict: Success", "").strip()
                                    candidates.append(cleaned_eval)
                            
                            # Randomly choose one if we have candidates
                            if candidates:
                                return random.choice(candidates)
                    
                    # Fallback
                    return f"Completed step {self.step_count}"
                except Exception as e:
                    logger.debug(f"Error extracting thought: {e}")
                    return f"Completed step {self.step_count}"
            
            # Create hooks using ExecutionHooks
            hooks = ExecutionHooks(
                agent_executor=self,
                session_id=self.session_id,
                cycle_id=self.cycle_id
            )
            on_step_start, on_step_end = hooks.create_loop_hooks(get_thought)
            
            # Run agent with hooks
            try:
                result = await self.browser_use_agent.run(
                    on_step_start=on_step_start,
                    on_step_end=on_step_end,
                    max_steps=50  # Reasonable limit
                )
                
                # Send ResponseToHuman on success
                await hooks.send_response_to_human(result)
                return result
                
            except InterruptedError:
                # Agent was stopped - send AgentStopped
                logger.info("Agent was interrupted (stopped or paused)")
                await hooks.send_agent_stopped(reason="interrupted")
                if self.browser_use_agent:
                    return self.browser_use_agent.history
                return None
        except Exception as e:
            # Send AgentStopped on error
            logger.error(f"Agent execution failed: {e}")
            if hasattr(self, 'session_id'):
                hooks = ExecutionHooks(
                    agent_executor=self,
                    session_id=self.session_id,
                    cycle_id=self.cycle_id
                )
                await hooks.send_agent_stopped(reason=f"error: {str(e)}")
            raise
    
    async def _run_code_mode(self, task: str, initial_url: Optional[str] = None) -> Any:
        """
        Run in code mode using LLM code generation + LocalPythonExecutor.
        
        CRITICAL ARCHITECTURE FOR HANDLING LOGIN/REDIRECT ISSUES:
        =========================================================
        This is essential for complex web applications that may redirect to login pages.
        
        The Problem:
        - User provides initial_url matching the app (e.g., "https://app.com/dashboard")
        - Browser may redirect to login page with non-matching URL (e.g., "https://login.com/auth")
        - Traditional codegen assumes goto() means we're on the right page
        - Reality: We may be on login page and need to detect this
        
        The Solution:
        1. Use goto() tool (not direct navigate) for initial_url
           - goto() calls observe matching the REQUESTED URL (not final redirected URL)
           - This handles case (b): login URLs that don't match observe pre_path
           
        2. Pass initial_state to code generation
           - After goto(), get STATE from python executor
           - Codegen sees if page="login" or page=null
           - Handles case (a): STATE indicates login page
           
        3. Codegen generates appropriate response
           - If STATE shows login: generates ai_exec("log in") or ask_human
           - If STATE shows wrong page: generates navigation/login handling
           - Doesn't assume goto() success without checking STATE
        
        Execution Flow:
        1. Creates python executor with tools
        2. If initial_url: calls goto() tool to navigate + observe
        3. If no initial_url: calls observe tool if any matches current URL
        4. Gets current STATE and URL (CRITICAL - this is what codegen sees)
        5. Generates Python code with initial STATE/URL context
        6. Validates and executes code
        7. If error, refines code and retries
        8. Loops until success
        
        Key Requirements for This to Work:
        - Observe tools MUST include login state detection (page="login")
        - This prevents timeouts waiting for elements that don't exist on login page
        - goto() in coderun.py matches observe on INPUT url, not current url
        - get_url() used instead of STATE["current_url"] for dynamic URL access
        """
        # First, ensure browser is started and connected (robust)
        await self._ensure_browser_started()
        
        # Create python executor first (needed for goto/observe tools)
        if self.python_executor is None:
            self.python_executor = self._create_python_executor()
        
        # Create code generator (needs to be created before iteration loop)
        if self.code_generator is None:
            self.code_generator = CodeGenerator(
                agent=self.agent,
                llm=self.codegen_llm,
                num_candidates=self.parallel_codegen,
                state_aware=self.state_aware,
                timezone=self.timezone
            )
        
        # Import helper for observe calls
        from .coderun import find_and_call_observe
        
        # Use self.history for follow-up requests, or start fresh
        # self.history contains UserMessage (TASK/RULES) and AssistantMessage (generated code) pairs
        # Code generator will append to this during retries
        max_iterations = 10
        
        # Create hooks for code mode completion messages (if callbacks provided)
        hooks = None
        if self.send_message_callback:
            hooks = ExecutionHooks(
                agent_executor=self,
                session_id=self.session_id,
                cycle_id=self.cycle_id
            )
        
        try:
            for iteration in range(max_iterations):
                logger.info(f"Code mode iteration {iteration + 1}/{max_iterations}")
                
                # Handle navigation and STATE population for first iteration only
                if iteration == 0 and initial_url:
                    # First iteration with initial_url: use goto() tool to navigate + observe
                    # CRITICAL: goto() matches observe tool to REQUESTED URL (not final URL after redirect)
                    # This allows observe to run even if page redirects to login, detecting the login state
                    logger.info(f"Calling goto() tool for initial URL: {initial_url}")
                    try:
                        goto_result = await self.python_executor.state["goto"](initial_url)
                        logger.debug(f"goto() result: {goto_result}")
                    except Exception as e:
                        logger.warning(f"Error during goto: {e}")
                elif iteration == 0 and not initial_url:
                    # No initial_url: call observe on current URL
                    # This ensures STATE is refreshed with latest page state
                    logger.info(f"Calling observe for current page")
                    try:
                        current_url_temp = await self.python_executor.state["get_url"]()
                        if current_url_temp:
                            await find_and_call_observe(
                                current_url_temp,
                                self.agent.tools,
                                self.python_executor.state["STATE"],
                                self,
                                " (initial observe)"
                            )
                    except Exception as e:
                        logger.warning(f"Error calling observe: {e}")
                
                # Get current STATE and URL to pass to code generation
                # This is CRITICAL - codegen needs to see if we're on login page, etc.
                # Must be done AFTER goto/observe updates STATE
                current_state = dict(self.python_executor.state.get("STATE", {}))
                current_url = ""
                try:
                    current_url = await self.python_executor.state["get_url"]()
                except Exception as e:
                    logger.warning(f"Error getting current URL: {e}")
                
                logger.info(f"STATE for codegen (iteration {iteration + 1}): {current_state}")
                logger.info(f"URL for codegen (iteration {iteration + 1}): {current_url}")
                
                # Generate code with self.history AND current STATE
                # The STATE tells codegen if we're on login page, what page state is, etc.
                # Code generator modifies history internally during retries
                code = await self.code_generator.generate_code(
                    task=task,
                    history=self.history,  # Use self.history for follow-ups!
                    error=None,  # Error is in history already
                    initial_state=current_state,
                    current_url=current_url
                )
                
                if not code:
                    logger.error("Failed to generate valid code")
                    
                    # Check if we should stop or fall back to loop mode
                    if self.stop_if_codegen_fails:
                        # CLI mode: raise error to stop execution
                        raise RuntimeError("Code generation failed - no valid code produced")
                    else:
                        # Interactive mode: fall back to loop mode
                        logger.warning("Code generation failed, falling back to loop mode")
                        return await self._run_loop_mode(task, initial_url=None)
                
                logger.info(f"Generated code ({len(code)} chars)")
                
                # Update self.history with the generated code
                # Code generator already appended UserMessage (TASK/RULES prompt without definitions)
                # and AssistantMessage (generated code) during generate_code()
                # But we need to check if it's a retry iteration (history already has messages)
                # If this is a fresh run or first iteration, history was updated by generate_code
                # If this is a retry, generate_code added error message and regenerated code
                
                # Execute code
                result = await self.python_executor(code)
                
                if result.error:
                    logger.error(f"Code execution error: {result.error}")
                    
                    # Add error to self.history for next iteration
                    error_message = f"Error during execution: {result.error}\n\nPlease fix the code."
                    if result.logs:
                        error_message += f"\n\nLogs from execution:\n{result.logs}"
                    
                    self.history.append(UserMessage(content=error_message))
                    continue
                
                # Success!
                logger.info(f"Code execution completed successfully")
                if result.logs:
                    logger.info(f"Logs:\n{result.logs}")
                logger.info(f"Result: {result.output}")
                
                # Send ResponseToHuman if hooks available
                if hooks:
                    await hooks.send_response_to_human(result=result.output)
                
                return result.output
            
            # Ran out of iterations - send AgentStopped
            error_msg = f"Failed to complete task after {max_iterations} iterations"
            if hooks:
                await hooks.send_agent_stopped(reason=f"max_iterations: {error_msg}")
            raise RuntimeError(error_msg)
            
        except StopExecutionError as e:
            # Code mode was stopped via decorator (check_stop_callback returned True)
            logger.info(f"Code mode stopped via StopExecutionError: {e}")
            if hooks:
                await hooks.send_agent_stopped(reason=f"stop_requested: {str(e)}")
            raise
        except Exception as e:
            # Unexpected error during code mode
            logger.error(f"Code mode error: {e}")
            if hooks:
                await hooks.send_agent_stopped(reason=f"error: {str(e)}")
            raise
    
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
