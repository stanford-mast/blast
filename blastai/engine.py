"""BLAST Engine for managing browser-based task execution."""

import asyncio
import hashlib
import logging
import threading
import time
import yaml
from pathlib import Path
from typing import Literal, Optional, Union, List, Dict, Any, AsyncIterator, Set, Tuple, TYPE_CHECKING
from uuid import uuid4

logger = logging.getLogger(__name__)

# Import non-browser_use modules first
from .response import AgentHistoryListResponse, AgentReasoning, HumanRequest, HumanResponse, StopRequest, AgentScheduled
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
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Optional path to config YAML file
            
        Returns:
            Dictionary containing settings and constraints
        """
        # Load default config
        default_config_path = Path(__file__).parent / 'default_config.yaml'
        with open(default_config_path) as f:
            config = yaml.safe_load(f)
            
        # Override with user config if provided
        if config_path:
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                # Update nested dicts
                if 'settings' in user_config:
                    config['settings'].update(user_config['settings'])
                if 'constraints' in user_config:
                    config['constraints'].update(user_config['constraints'])
        
        return config
    
    @classmethod
    async def create(cls, 
                    config_path: Optional[str] = None,
                    settings: Optional[Settings] = None,
                    constraints: Optional[Constraints] = None) -> "Engine":
        """Create an engine instance.
        
        This method handles several initialization cases:
        1. No arguments -> load from default_config.yaml
        2. config_path -> load from specified config file
        3. settings/constraints -> use provided instances
        4. Mix of above -> merge appropriately
        
        Args:
            config_path: Optional path to config YAML file
            settings: Optional Settings instance
            constraints: Optional Constraints instance
            
        Returns:
            Initialized Engine instance
        """
        # Load config if needed
        config = None
        if config_path is not None or (settings is None and constraints is None):
            config = cls.load_config(config_path)
        
        # Create or update settings
        if settings is None:
            if config is None:
                config = cls.load_config()
            settings = Settings.create(**config['settings'])
        
        # Create or update constraints
        if constraints is None:
            if config is None:
                config = cls.load_config()
            constraints = Constraints.create(**config['constraints'])
        
        # Create and start engine
        engine = cls(settings=settings, constraints=constraints)
        await engine.start()
        return engine
    
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
        
    async def start(self):
        """Start the engine's resource management."""
        if not self._started:
            await self.resource_manager.start()
            self._started = True
            
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
                 mode: Literal["block", "stream", "interactive"] = "block",
                 previous_response_id: Optional[str] = None,
                 initial_url: Optional[str] = None) -> Union[
                     AgentHistoryListResponse,  # block mode
                     AsyncIterator[Union[AgentReasoning, AgentHistoryListResponse]],  # stream mode
                     Tuple[asyncio.Queue, asyncio.Queue]  # interactive mode
                 ]:
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
            # Response ID format: "resp_<task_id>" or "chatcmpl-<task_id>" or direct task ID
            if previous_response_id.startswith("resp_"):
                prev_task_id = previous_response_id.split('_')[1]
            elif previous_response_id.startswith("chatcmpl-"):
                prev_task_id = previous_response_id.split('-')[1]
            else:
                # Assume it's a direct task ID
                prev_task_id = previous_response_id
                
            # Check if task exists
            if prev_task_id not in self.scheduler.tasks:
                logger.warning(f"Previous task {prev_task_id} not found")
                prev_task_id = None
            
        # Create queues if interactive mode
        interactive_queues = None
        if mode == "interactive":
            to_client: asyncio.Queue = asyncio.Queue()
            from_client: asyncio.Queue = asyncio.Queue()
            interactive_queues = {
                'to_client': to_client,
                'from_client': from_client
            }

        # Schedule task(s)
        # Disable caching and force new plan in interactive mode
        if mode == "interactive":
            cache_control = "no-cache,no-cache-plan"
        cache_controls = [cache_control] if isinstance(cache_control, str) else cache_control
        if isinstance(task_descriptions, list):
            # For multiple tasks, schedule them in sequence
            task_ids = []
            current_task_id = prev_task_id
                
            for i, desc in enumerate(task_descriptions):
                task_id = self.scheduler.schedule_task(
                    desc,
                    prerequisite_task_id=current_task_id,
                    cache_control=cache_controls[i],
                    interactive_queues=interactive_queues,
                    initial_url=initial_url if i == 0 else None  # Only pass initial_url for the first task
                )
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
                cache_control=cache_controls[0],
                interactive_queues=interactive_queues,
                initial_url=initial_url
            )
            task = self.scheduler.tasks[final_task_id]
            logger.debug(f"Task {final_task_id} scheduled (prerequisite: {prev_task_id}, url: {task.initial_url})")
        
        try:
            if mode == "block":
                # For blocking mode, wait for final result
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
            
            elif mode == "stream":
                # For streaming mode, return scheduler's stream directly
                return self.scheduler.stream_task_events(final_task_id)
                
            else:  # interactive mode
                # Get queues from task state
                task = self.scheduler.tasks[final_task_id]
                queues = task.interactive_queues
                if not queues:
                    raise RuntimeError("Interactive mode requires queues")
                
                # Send immediate task scheduled notification with task ID
                await queues['to_client'].put(
                    AgentScheduled(
                        task_id=final_task_id,
                        description=task.description
                    )
                )
                
                # Start streaming task in background
                async def stream_to_client():
                    try:
                        async for event in self.scheduler.stream_task_events(final_task_id):
                            await queues['to_client'].put(event)
                    except Exception as e:
                        logger.error(f"Error in stream_to_client: {e}")
                        
                # Start monitoring client messages in background
                async def monitor_client_messages():
                    try:
                        while True:
                            # Check if task completed first
                            task = self.scheduler.tasks[final_task_id]
                            if task.is_completed:
                                break
                                
                            # Try to get message with timeout
                            try:
                                msg = await asyncio.wait_for(queues['from_client'].get(), timeout=0.1)
                                if isinstance(msg, StopRequest):
                                    # Get all tasks in dependency chain
                                    dependency_ids = self.scheduler._get_dependency_ids(final_task_id)
                                    
                                    # Process each task in the dependency chain
                                    for task_id in dependency_ids:
                                        task = self.scheduler.tasks.get(task_id)
                                        if not task or task.is_completed:
                                            continue  # Skip if task doesn't exist or is already completed

                                        # End the task - preserve executor if it has running prerequisites
                                        # Otherwise clean up the executor
                                        cleanup_executor = len(self.scheduler._get_prereq_ids(task_id, running_only=True)) > 0
                                        logger.debug(f"Ending task {task_id} with cleanup_executor={cleanup_executor}")
                                        await self.resource_manager.end_task(task_id, cleanup_executor=cleanup_executor)
                                    
                                    break
                            except asyncio.TimeoutError:
                                continue  # No message, check completion again
                    except Exception as e:
                        logger.error(f"Error in monitor_client_messages: {e}")
                
                # Start background tasks
                asyncio.create_task(stream_to_client())
                asyncio.create_task(monitor_client_messages())
                
                # Return queues for bidirectional communication
                return queues['to_client'], queues['from_client']
        except Exception as e:
            # Log full exception details with traceback
            logger.error(f"Task {final_task_id} failed", exc_info=True)
            # Wrap original exception to preserve traceback
            raise RuntimeError(f"Failed to run task(s): {str(e)}") from e
            
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
        
        # Get cost and token usage from resource manager
        total_cost = self.resource_manager._get_cost()
        total_token_usage = self.resource_manager._get_token_usage()
        
        return {
            "tasks": {
                "scheduled": len(scheduled_tasks),
                "running": len(running_tasks),
                "completed": len(completed_tasks)
            },
            "concurrent_browsers": len([t for t in tasks.values() if t.executor and t.executor.browser_session]),
            "memory_usage_gb": round(memory_gb, 2),
            "total_cost": round(total_cost, 2),
            "total_token_usage": total_token_usage.to_json(),
            "total_token_usage_str": total_token_usage.format_detailed(),
        }

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop()