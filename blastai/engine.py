"""BLAST Engine for managing browser-based task execution."""

import asyncio
import hashlib
import logging
import threading
import time
from typing import Optional, Union, List, Dict, Any, AsyncIterator, Set, TYPE_CHECKING
from uuid import uuid4

logger = logging.getLogger(__name__)

# Import non-browser_use modules first
from .response import AgentHistoryListResponse, AgentReasoning
from .config import Settings, Constraints
from .resource_manager import ResourceManager
from .scheduler import Scheduler
from .cache import CacheManager
from .planner import Planner

if TYPE_CHECKING:
    from browser_use.agent.views import AgentHistoryList

class Engine:
    """Main BLAST engine for running browser-based tasks."""
    
    # Import browser_use only when needed
    from browser_use.agent.views import AgentHistoryList
    
    def __init__(self, constraints: Optional[Constraints] = None, settings: Optional[Settings] = None):
        """Initialize engine with optional constraints and settings."""
        self.constraints = constraints or Constraints()
        self.settings = settings or Settings()
        
        # Create unique hash for this engine instance
        hash_input = f"{time.time()}-{id(self)}"
        self._instance_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        
        # Initialize components in correct order to handle dependencies
        self.planner = Planner(constraints)
        
        # Create CacheManager first (no scheduler needed yet)
        self.cache_manager = CacheManager(
            instance_hash=self._instance_hash,
            persist=self.settings.persist_cache,
            constraints=self.constraints,
        )
        
        # Create Scheduler with CacheManager
        self.scheduler = Scheduler(
            constraints=self.constraints,
            cache_manager=self.cache_manager,
            planner=self.planner
        )
        
        # Load CacheManager with scheduler
        self.cache_manager.load(self.scheduler)
        
        # Finally create ResourceManager with all dependencies
        self.resource_manager = ResourceManager(
            scheduler=self.scheduler,
            constraints=self.constraints,
            settings=self.settings,
            engine_hash=self._instance_hash,
            cache_manager=self.cache_manager
        )
        self._started = False

    @classmethod
    async def create(cls, **kwargs) -> "Engine":
        """Asynchronously create an engine instance with optional settings."""
        new_engine = cls(**kwargs)
        await new_engine.start()
        return new_engine
        
    async def start(self):
        """Start the engine's resource management."""
        if not self._started:
            await self.resource_manager.start()
            self._started = True
        # Thread for logging metrics (if needed)
        # self._metrics_thread = threading.Thread(target=self._log_metrics, daemon=True)
        # self._metrics_thread.start()

    def _log_metrics(self):
        """Log engine metrics every 5 seconds."""
        while self._started:
            try:
                metrics = asyncio.run(self.get_metrics())
                logger.debug(f"Engine metrics: {metrics}")
            except Exception as e:
                logger.error(f"Error logging metrics: {e}")
            time.sleep(5)
            
    async def stop(self):
        """Stop the engine and cleanup resources."""
        if self._started:
            try:
                # First stop any running tasks
                for task_id, task in list(self.scheduler.tasks.items()):
                    if task.executor:
                        if task.executor_run_task and not task.executor_run_task.done():
                            task.executor_run_task.cancel()
                            try:
                                await asyncio.wait_for(task.executor_run_task, timeout=2.0)
                            except (asyncio.CancelledError, asyncio.TimeoutError):
                                pass
                
                # Then cleanup executors
                for task_id, task in list(self.scheduler.tasks.items()):
                    if task.executor:
                        try:
                            await asyncio.wait_for(task.executor.cleanup(), timeout=2.0)
                        except asyncio.TimeoutError:
                            pass
                
                # Finally stop resource manager and clear caches
                await self.resource_manager.stop()
                self.cache_manager.clear()  # This will respect persist setting
                self._started = False
                
            except Exception as e:
                logger.error(f"Error during engine stop: {e}")
                raise
            
    async def run(self, task_descriptions: Union[str, List[str]],
                 cache_control: Union[str, List[str]] = "",
                 stream: bool = False,
                 previous_response_id: Optional[str] = None) -> Union[AgentHistoryListResponse, AsyncIterator[Union[AgentReasoning, AgentHistoryListResponse]]]:
        """Run one or more tasks and return their results.
        
        Args:
            task_descriptions: Either a single task description string or a list of task descriptions.
                             If a list is provided, each task will be scheduled as a parent of the next task.
            cache_control: Cache control settings for each task
            stream: Whether to stream execution updates
            previous_response_id: Optional ID of previous response for conversation continuity
            
        Returns:
            If stream=False: AgentHistoryListResponse containing the final task execution history
            If stream=True: AsyncIterator yielding AgentReasoning updates and final AgentHistoryListResponse
        """
        # Ensure engine is started
        if not self._started:
            await self.start()
            
        # Extract previous task ID from response ID if provided
        prev_task_id = None
        if previous_response_id:
            # Response ID format: "resp_<task_id>" or "chatcmpl-<task_id>"
            if previous_response_id.startswith("resp_"):
                prev_task_id = previous_response_id.split('_')[1]
            elif previous_response_id.startswith("chatcmpl-"):
                prev_task_id = previous_response_id.split('-')[1]
            if prev_task_id not in self.scheduler.tasks:
                raise RuntimeError(f"Previous task {prev_task_id} not found")

        # Schedule task(s)
        cache_controls = [cache_control] if isinstance(cache_control, str) else cache_control
        if isinstance(task_descriptions, list):
            # For multiple tasks, schedule them in sequence
            task_ids = []
            current_task_id = prev_task_id
            for i, desc in enumerate(task_descriptions):
                task_id = self.scheduler.schedule_task(desc, prerequisite_task_id=current_task_id, cache_control=cache_controls[i])
                task = self.scheduler.tasks[task_id]
                logger.debug(f"Task {task_id} scheduled (prerequisite: {current_task_id}, url: {task.initial_url})")
                task_ids.append(task_id)
                current_task_id = task_id
            final_task_id = task_ids[-1]
        else:
            # For single task, let scheduler handle it directly
            final_task_id = self.scheduler.schedule_task(
                task_descriptions,
                prerequisite_task_id=prev_task_id,
                cache_control=cache_controls[0]
            )
            task = self.scheduler.tasks[final_task_id]
            logger.debug(f"Task {final_task_id} scheduled (prerequisite: {prev_task_id}, url: {task.initial_url})")
        
        try:
            # For non-streaming, wait for all tasks to complete
            if not stream:
                # For non-streaming, wait for final result
                final_history = await self.scheduler.get_task_result(final_task_id)
                if not final_history:
                    logger.error(f"Task {final_task_id} failed in get_task_result")
                    raise RuntimeError(f"Task {final_task_id} failed to complete")
                
                logger.info(f"Task {final_task_id} completed with result: {final_history.final_result()}")
                # Convert to response type with task ID
                response = AgentHistoryListResponse.from_history(
                    history=final_history,
                    task_id=final_task_id
                )
                return response
            
            # For streaming, return scheduler's stream directly
            return self.scheduler.stream_task_events(final_task_id)
        except Exception as e:
            logger.error(f"Task {final_task_id} failed: {str(e)}")
            raise RuntimeError(f"Failed to run task(s): {str(e)}")
            
    async def __aenter__(self):
        """Context manager entry."""
        await self.start()
        return self
        
    async def get_metrics(self):
        """Get current engine metrics."""
        tasks = self.scheduler.tasks
        running_tasks = [t for t in tasks.values() if t.executor and not t.is_completed]
        completed_tasks = [t for t in tasks.values() if t.is_completed]
        scheduled_tasks = [t for t in tasks.values() if not t.is_completed and not t.executor]
        
        # Get total memory usage from resource manager
        total_memory = self.resource_manager._get_total_memory_usage()
        memory_gb = total_memory / (1024 * 1024 * 1024)  # Convert to GB
        
        # Get cost from resource manager
        total_cost = self.resource_manager._get_cost()
        
        return {
            "tasks": {
                "scheduled": len(scheduled_tasks),
                "running": len(running_tasks),
                "completed": len(completed_tasks)
            },
            "concurrent_browsers": len([t for t in tasks.values() if t.executor and t.executor.browser]),
            "memory_usage_gb": round(memory_gb, 2),
            "total_cost": round(total_cost, 2)
        }

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop()