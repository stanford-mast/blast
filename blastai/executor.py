"""BLAST Executor for managing browser-based task execution."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Union, AsyncIterator, Dict, Any
from datetime import datetime

from browser_use import Agent, Browser, Controller
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.context import BrowserContext
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

logger = logging.getLogger(__name__)

from .config import Settings, Constraints
from .response import AgentReasoning, AgentHistoryListResponse

class Executor:
    """Wrapper around browser_use Agent for task execution."""
    
    def __init__(self, browser: Browser, browser_context: BrowserContext, controller: Controller,
                 llm: ChatOpenAI, constraints: Constraints, task_id: str,
                 settings: Settings = None, engine_hash: str = None, scheduler = None):
        """Initialize executor with browser, controller, LLM and constraints."""
        self.browser = browser
        self.browser_context = browser_context
        self.llm = llm
        self.controller = controller
        self.constraints = constraints
        self.task_id = task_id
        self.settings = settings or Settings()
        self.engine_hash = engine_hash
        self.scheduler = scheduler
        self.agent: Optional[Agent] = None
        self._paused = False
        self._running = False
        self._task: Optional[str] = None
        self._history: Optional[AgentHistoryList] = None
        self._last_state: Optional[Dict[str, Any]] = None
        self._total_cost = 0.0  # Track total LLM cost
        
    async def run(self, task_or_plan: Union[str, AgentHistoryList], initial_url: Optional[str] = None) -> AgentHistoryList:
        """Run a task or reuse a cached plan.
        
        Args:
            task_or_plan: Either a task description string to execute,
                         or a cached AgentHistoryList plan to reuse
            
        Returns:
            AgentHistoryList containing the execution history
        """
        try:
            self._running = True
            
            if isinstance(task_or_plan, str):
                self._task = task_or_plan
                task = task_or_plan
                
                # Create agent if this is first run, otherwise add new task
                if not self.agent:
                    # Create initial actions if URL provided
                    initial_actions = None
                    if initial_url:
                        initial_actions = [{'open_tab': {'url': initial_url}}]
                        
                    self.agent = Agent(
                        task=task,
                        browser=self.browser,
                        browser_context=self.browser_context,
                        controller=self.controller,
                        llm=self.llm,
                        use_vision=self.constraints.allow_vision,
                        initial_actions=initial_actions
                    )
                else:
                    if initial_url:
                        await self.agent.multi_act([{'open_tab': {'url': initial_url}}])
                    self.agent.add_new_task(task)
                
                # Run task with cost tracking
                with get_openai_callback() as cb:
                    self._history = await self.agent.run()
                    self._total_cost += cb.total_cost
                return self._history
                
            else:
                # Reuse cached plan
                if not self.agent:
                    self.agent = Agent(
                        task="",  # Plan already contains the task
                        browser=self.browser,
                        browser_context=self.browser_context,
                        controller=self.controller,
                        llm=self.llm,
                        use_vision=self.constraints.allow_vision
                    )
                
                # Run plan with cost tracking
                with get_openai_callback() as cb:
                    self._history = await self.agent.rerun_history(task_or_plan)
                    self._total_cost += cb.total_cost
                return self._history
            
        except Exception as e:
            logger.error(f"Task {self.task_id} failed: {str(e)}")
            raise RuntimeError(f"Failed to execute task: {str(e)}")
        finally:
            self._running = False
            
    def get_reasoning(self) -> List[AgentReasoning]:
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
                content=brain.next_goal
            ))
            
            # Create separate reasoning for memory if available
            if hasattr(brain, 'memory') and brain.memory:
                reasonings.append(AgentReasoning(
                    task_id=self.task_id,
                    type="thought",
                    thought_type="memory",
                    content=brain.memory
                ))
            
            # Create separate reasoning for screenshot if available
            if self.browser_context and self.browser_context.current_state.screenshot:
                reasonings.append(AgentReasoning(
                    task_id=self.task_id,
                    type="screenshot",
                    content=self.browser_context.current_state.screenshot
                ))
            
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
        """Clean up resources properly."""
        if self.agent:
            try:
                await self.agent.close()
            except Exception as e:
                logger.error(f"Error closing agent: {e}")
        self.agent = None
        
    def get_total_cost(self) -> float:
        """Get total LLM cost for this executor."""
        return self._total_cost
