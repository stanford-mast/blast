"""BLAST Executor for managing browser-based task execution."""

import asyncio
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import List, Optional, Union, AsyncIterator, Dict, Any, cast, Tuple
from datetime import datetime
from urllib.parse import urlparse, quote_plus

from browser_use import Agent, Controller
from browser_use.agent.views import AgentHistoryList
from browser_use.browser import BrowserSession
from browser_use.llm.base import BaseChatModel

from lmnr import Laminar

logger = logging.getLogger(__name__)

from .config import Settings, Constraints
from .response import AgentReasoning, AgentHistoryListResponse
from .utils import estimate_llm_cost, get_base_url_for_provider
from .models import is_openai_model, TokenUsage
from .resource_factory_utils import cleanup_stealth_profile_dir

# Initialize Laminar if available and API key is set
laminar_api_key = os.environ.get("LMNR_PROJECT_API_KEY")
if laminar_api_key:
    try:
        # Check for self-hosting configuration
        base_url = os.environ.get("LMNR_BASE_URL")
        http_port = os.environ.get("LMNR_HTTP_PORT")
        grpc_port = os.environ.get("LMNR_GRPC_PORT")
        
        # Initialize with appropriate parameters
        if base_url and http_port and grpc_port:
            # Self-hosted configuration
            Laminar.initialize(
                project_api_key=laminar_api_key,
                base_url=base_url,
                http_port=int(http_port),
                grpc_port=int(grpc_port)
            )
            logger.info(f"Laminar instrumentation initialized with self-hosted instance at {base_url}")
        else:
            # Default cloud configuration
            Laminar.initialize(project_api_key=laminar_api_key)
            logger.info("Laminar instrumentation initialized with cloud instance")
    except Exception as e:
        logger.warning(f"Failed to initialize Laminar instrumentation: {e}")
else:
    logger.info("LMNR_PROJECT_API_KEY not found in environment, Laminar instrumentation disabled")

class Executor:
    """Wrapper around browser_use Agent for task execution."""
    
    def __init__(self, browser_session: BrowserSession, controller: Controller,
                 llm: BaseChatModel, constraints: Constraints, task_id: str,
                 settings: Settings = None, engine_hash: str = None, scheduler = None,
                 sensitive_data: Optional[Dict[str, str]] = None,
                 user_data_dir: Optional[str] = None,
                 vnc_session: Optional[Any] = None,
                 live_url: Optional[str] = None):
        """Initialize executor with browser session, controller, LLM and constraints."""
        self.browser_session = browser_session
        self.llm = llm
        self.controller = controller
        self.constraints = constraints
        self.task_id = task_id
        self.settings = settings or Settings()
        self.engine_hash = engine_hash
        self.scheduler = scheduler
        self.sensitive_data = sensitive_data
        self.user_data_dir = user_data_dir  # Store user_data_dir for cleanup
        self.vnc_session = vnc_session  # Store VNC session for cleanup
        self.live_url = live_url  # Store live URL for responses
        self.agent: Optional[Agent] = None
        self._paused = False
        self._running = False
        self._task: Optional[str] = None
        self._history: Optional[AgentHistoryList] = None
        self._last_state: Optional[Dict[str, Any]] = None
        self._total_cost = 0.0  # Track total LLM cost
        self._total_token_usage = TokenUsage()  # Track total LLM token usage
        self._cb = None  # Track current callback
        
    def _get_url_or_search(self, input_str: Optional[str]) -> Optional[str]:
        """Convert input to URL or Google search URL.
        
        Args:
            input_str: Input string (URL or search query)
            
        Returns:
            URL to use (either direct URL or Google search URL)
            None if input_str is None
        """
        if not input_str:
            return None
            
        # Check if it's a valid URL
        try:
            result = urlparse(input_str)
            is_url = all([result.scheme, result.netloc])
        except:
            is_url = False
            
        if is_url:
            return input_str
        else:
            # Convert to Google search URL
            search_query = quote_plus(input_str)
            return f'https://www.google.com/search?q={search_query}'
        
    async def _get_cost_from_agent(self, agent):
        """Get cost and token usage information from agent's token cost service."""
        try:
            if hasattr(agent, 'token_cost_service') and agent.token_cost_service:
                # Get usage summary for all models
                usage_summary = await agent.token_cost_service.get_usage_summary()
                
                # Update total cost
                self._total_cost += usage_summary.total_cost
                
                # Create TokenUsage from summary and add to total
                current_usage = TokenUsage(
                    prompt=usage_summary.total_prompt_tokens,
                    prompt_cached=usage_summary.total_prompt_cached_tokens,
                    completion=usage_summary.total_completion_tokens,
                    total=usage_summary.total_tokens
                )
                self._total_token_usage += current_usage
                
                return usage_summary.total_cost, current_usage
            
            return 0.0, TokenUsage()
        except Exception as e:
            logger.debug(f"Error getting cost from agent: {e}")
            return 0.0, TokenUsage()

    async def run(self, task_or_plan: Union[str, AgentHistoryList], initial_url: Optional[str] = None) -> AgentHistoryList:
        """Run a task or reuse a cached plan.
        
        Args:
            task_or_plan: Either a task description string to execute,
                         or a cached AgentHistoryList plan to reuse
            
        Returns:
            AgentHistoryList containing the execution history
        """
        self._running = True
        try:
            if isinstance(task_or_plan, str):
                self._task = task_or_plan
                task = task_or_plan
                
                # Create agent if this is first run, otherwise add new task
                if not self.agent:
                    # Create initial actions if URL/search provided
                    initial_actions = None
                    url = self._get_url_or_search(initial_url)
                    if url:
                        initial_actions = [{'go_to_url': {'url': url, 'new_tab': False}}]
                        
                    logger.debug(f"Creating new agent for task: {task} with sensitive data: {self.sensitive_data}")
                    self.agent = Agent(
                        task=task,
                        browser_session=self.browser_session,
                        controller=self.controller,
                        llm=self.llm,
                        use_vision=self.constraints.allow_vision,
                        initial_actions=initial_actions,
                        sensitive_data=self.sensitive_data,
                        calculate_cost=True,
                    )
                else:
                    # For follow-up tasks, we need to clear any initial_actions
                    # to prevent them from being executed again
                    self.agent.initial_actions = None
                    
                    # Reset agent's stopped state if it was previously stopped
                    # This ensures the agent can run again after being stopped
                    if hasattr(self.agent, 'state') and hasattr(self.agent.state, 'stopped') and self.agent.state.stopped:
                        logger.debug(f"Resetting stopped state for agent in task: {task}")
                        self.agent.state.stopped = False
                    
                    # Get the current page URL from the browser session
                    # This ensures we're working with the tab the user is currently on
                    current_page = await self.agent.browser_session.get_current_page()
                    current_url = current_page.url if current_page else None
                    
                    # Only navigate to URL if explicitly provided for this task
                    # Otherwise, we'll use the current page that the user is on
                    # Only navigate to URL if explicitly provided for this task
                    # Otherwise, we'll use the current page that the user is on
                    url = self._get_url_or_search(initial_url)
                    if url:
                        # Instead of opening a new tab, navigate the current page to the URL
                        # This ensures we stay on the tab the user switched to
                        logger.debug(f"Navigating current page to URL: {url}")
                        await self.agent.browser_session.navigate(url)
                    elif current_url:
                        # Log that we're using the current page that the user switched to
                        logger.debug(f"Using current page URL for task: {current_url}")
                    
                    # Reinitialize the EventBus to ensure it's properly set up for the new task
                    # This prevents the "EventBus._start() must be called before _run_loop_step()" error
                    if hasattr(self.agent, 'eventbus') and self.agent.eventbus:
                        # Create a new EventBus with the same configuration
                        wal_path = self.agent.eventbus.wal_path
                        name = self.agent.eventbus.name
                        parallel_handlers = self.agent.eventbus.parallel_handlers
                        
                        # Create a new EventBus instance
                        from bubus import EventBus
                        try:
                            self.agent.eventbus = EventBus(name=name, wal_path=wal_path, parallel_handlers=parallel_handlers)
                        except Exception as e:
                            logger.error(f"Error creating EventBus: {e}, creating a new one with a unique name")
                            timestamp = str(int(time.time() * 1000))[-8:]
                            unique_name = f"{name}_{timestamp}"
                            self.agent.eventbus = EventBus(name=unique_name, wal_path=wal_path, parallel_handlers=parallel_handlers)
                        
                        # Explicitly call _start() to ensure the EventBus is properly initialized
                        # This ensures event_queue, runloop_lock, and on_idle are set up
                        self.agent.eventbus._start()
                        
                        # Register any existing handlers
                        if hasattr(self.agent, 'cloud_sync') and self.agent.cloud_sync:
                            self.agent.eventbus.on('*', self.agent.cloud_sync.handle_event)
                    
                    # Update the agent's task property directly before calling add_new_task
                    self.agent.task = task
                    self.agent.add_new_task(task)
                # Run task
                try:
                    # Make sure calculate_cost is enabled in the agent settings
                    if hasattr(self.agent, 'settings'):
                        self.agent.settings.calculate_cost = True
                    
                    self._history = await self.agent.run()
                    
                    # Get cost from agent's token cost service
                    await self._get_cost_from_agent(self.agent)
                except Exception as e:
                    logger.error(f"Error running agent: {e}")
                    raise
                return self._history
                
            else:
                # Reuse cached plan
                if not self.agent:
                    self.agent = Agent(
                        task="",  # Plan already contains the task
                        browser_session=self.browser_session,
                        controller=self.controller,
                        llm=self.llm,
                        use_vision=self.constraints.allow_vision,
                        sensitive_data=self.sensitive_data,
                        calculate_cost=True,
                    )
                # Run plan
                try:
                    # Make sure calculate_cost is enabled in the agent settings
                    if hasattr(self.agent, 'settings'):
                        self.agent.settings.calculate_cost = True
                    
                    self._history = await self.agent.rerun_history(task_or_plan)
                    
                    # Get cost from agent's token cost service
                    await self._get_cost_from_agent(self.agent)
                except Exception as e:
                    logger.error(f"Error rerunning history: {e}")
                    raise
                return self._history
            
        except Exception as e:
            # logger.error(f"Task {self.task_id} failed: {str(e)}")
            raise RuntimeError(f"Failed to execute task: {str(e)}")
        finally:
            self._running = False
            
    async def get_reasoning(self) -> List[AgentReasoning]:
        """Get current agent reasoning states.
        
        Returns:
            List of AgentReasoning objects, one each for goal, memory, and screenshot.
            Empty list if agent is not running or has no state.
        """
        if not self.agent or not self.agent.state or not self.agent.state.history:
            return []
            
        thoughts = self.agent.state.history.model_thoughts()
        if not thoughts:
            return []
            
        brain = thoughts[-1]
        reasonings = []
        
        # Create separate reasoning for goal if available
        if hasattr(brain, 'next_goal') and brain.next_goal:
            reasonings.append(AgentReasoning(
                task_id=self.task_id,
                type="thought",
                thought_type="goal",
                content=brain.next_goal,
                live_url=self.live_url
            ))
            
        # Create separate reasoning for memory if available
        if hasattr(brain, 'memory') and brain.memory:
            reasonings.append(AgentReasoning(
                task_id=self.task_id,
                type="thought",
                thought_type="memory",
                content=brain.memory,
                live_url=self.live_url
            ))
        
        # Create separate reasoning for screenshot if available
        try:
            state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=True)
            if state and state.screenshot:
                reasonings.append(AgentReasoning(
                    task_id=self.task_id,
                    type="screenshot",
                    content=state.screenshot,
                    live_url=self.live_url
                ))
        except Exception as e:
            pass
            
        return reasonings
    
    async def pause(self):
        """Pause the executor if running."""
        if self._running and self.agent and not self._paused:
            self.agent.pause()
            self._paused = True
            
    async def resume(self):
        """Resume the executor if paused."""
        if self._running and self.agent and self._paused:
            self.agent.resume()
            self._paused = False
            
    @property
    def is_running(self) -> bool:
        """Check if the executor is currently running."""
        return self._running
        
    def get_plan(self) -> Optional[AgentHistoryList]:
        """Get the current execution plan if available."""
        return self._history
            
    async def cleanup(self):
        """Clean up resources properly.
        
        Cleanup order:
        1. Close agent (which closes browser session)
        2. Clean up VNC session (which also cleans up its browser session)
        3. Clean up user data directory
        4. Clear all references
        
        Each cleanup step is attempted even if previous steps fail.
        All errors are logged but don't prevent other cleanup steps.
        """
        errors = []
        
        # Close agent first (this closes the browser session)
        if self.agent:
            try:
                await self.agent.close()
            except Exception as e:
                errors.append(f"Error closing agent: {e}")
        
        # Clean up VNC session if it exists
        if self.vnc_session:
            try:
                await self.vnc_session.cleanup()
            except Exception as e:
                errors.append(f"Error cleaning up VNC session: {e}")
        
        # Clean up stealth profile directory if it was a temporary one
        if self.user_data_dir:
            try:
                cleanup_stealth_profile_dir(self.user_data_dir)
            except Exception as e:
                errors.append(f"Error cleaning up stealth profile: {e}")
        
        # Clear all references
        self.agent = None
        self.vnc_session = None
        self.browser_session = None
        self.user_data_dir = None
        
        # Log any errors that occurred during cleanup
        if errors:
            logger.error("Errors during executor cleanup:\n" + "\n".join(errors))
        
    def get_total_cost(self) -> float:
        """Get total LLM cost for this executor."""
        # We've been tracking the total cost as we go
        return self._total_cost

    def get_total_token_usage(self) -> TokenUsage:
        """Get total LLM token usage for this executor."""
        return self._total_token_usage
        
    def set_task_id(self, task_id: str, controller: Controller):
        """Update task ID and controller for both executor and agent.
        
        Args:
            task_id: New task ID
            controller: New controller to use
        """
        self.task_id = task_id
        self.controller = controller
        if self.agent:
            self.agent.controller = controller
