"""BLAST Engine for managing browser-based task execution."""

import asyncio
import hashlib
import logging
import threading
import time
import yaml
import docker
import os
import subprocess
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, AsyncIterator, Tuple, Set, TYPE_CHECKING
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
    def check_requirements(cls, config_path: Optional[str] = None,
                         settings: Optional[Settings] = None,
                         constraints: Optional[Constraints] = None) -> Tuple[bool, str]:
        """Check if all requirements are met for running the engine.
        
        Args:
            config_path: Optional path to config YAML file
            settings: Optional Settings instance
            constraints: Optional Constraints instance
            
        Returns:
            Tuple of (requirements_met, error_message)
        """
        # Load config if needed
        config = None
        if config_path is not None or (settings is None and constraints is None):
            config = cls.load_config(config_path)
        
        # Get constraints
        if constraints is None:
            if config is None:
                config = cls.load_config()
            constraints = Constraints.create(**config['constraints'])
            
        # Check Steel requirements if enabled
        if constraints.require_steel:
            # Import here to avoid circular import
            from .cli_installation import is_wsl
            
            try:
                client = docker.from_env()
                client.ping()
            except docker.errors.DockerException as e:
                if is_wsl() and ("not found" in str(e) or "Connection aborted" in str(e)):
                    return False, "Docker is not available in WSL. Docker Desktop integration needs to be enabled."
                elif "ConnectionError" in str(e):
                    return False, "Docker daemon is not running. Please start Docker and try again."
                elif "not found" in str(e):
                    return False, "Docker is not installed. Please install Docker to use Steel integration."
                else:
                    return False, f"Docker error: {e}"
            except Exception as e:
                return False, f"Failed to connect to Docker: {e}"
                
        return True, ""

    @classmethod
    async def create(cls,
                    config_path: Optional[str] = None,
                    settings: Optional[Settings] = None,
                    constraints: Optional[Constraints] = None,
                    instance_hash: Optional[str] = None) -> "Engine":
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
        engine = cls(settings=settings, constraints=constraints, instance_hash=instance_hash)
        # Start engine and store Steel port if used
        await engine.start()
        return engine
        
    def get_steel_port(self) -> Optional[int]:
        """Get the port number for the Steel API server if running."""
        return getattr(self, '_steel_port', None)
    
    def __init__(self, constraints: Optional[Constraints] = None, settings: Optional[Settings] = None, instance_hash: Optional[str] = None):
        """Initialize engine with optional constraints and settings."""
        self.constraints = constraints or Constraints()
        self.settings = settings or Settings()
        
        # Use provided instance hash or generate one
        self._instance_hash = instance_hash or hashlib.sha256(f"{time.time()}-{id(self)}".encode()).hexdigest()[:8]
        
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
        
        # Store Steel port if using Steel
        self._steel_port = None
        
        # Finally create ResourceManager with all dependencies
        self.resource_manager = ResourceManager(
            scheduler=self.scheduler,
            constraints=self.constraints,
            settings=self.settings,
            engine_hash=self._instance_hash,
            cache_manager=self.cache_manager,
            steel_port=None  # Will be set after Steel server starts
        )
        self._started = False
        
    async def start(self):
        """Start the engine's resource management."""
        if not self._started:
            # Start Steel server if required
            if self.constraints.require_steel:
                # Check requirements first
                requirements_ok, error_msg = self.check_requirements(
                    settings=self.settings,
                    constraints=self.constraints
                )
                if not requirements_ok:
                    raise RuntimeError(error_msg)

                try:
                    client = docker.from_env()
                    # Find available ports
                    from .cli_process import find_available_port
                    api_port = find_available_port(3100)
                    debug_port = find_available_port(9229)
                    ui_port = find_available_port(5174)
                    
                    if not all([api_port, debug_port, ui_port]):
                        raise RuntimeError("Could not find available ports for Steel server")
                    
                    logger.info(f"Starting Steel server on ports - API: {api_port}, Debug: {debug_port}, UI: {ui_port}")
                    self._steel_port = api_port
                    self.resource_manager._steel_port = api_port
                    
                    # Start Steel server using docker compose
                    try:
                        # Get path to steel-docker-compose.yml
                        compose_file = Path(__file__).parent / 'steel-docker-compose.yml'
                        if not compose_file.exists():
                            raise RuntimeError("steel-docker-compose.yml not found")

                        # Set environment variables for ports
                        env = os.environ.copy()
                        env.update({
                            "API_PORT": str(api_port),
                            "DEBUG_PORT": str(debug_port),
                            "UI_PORT": str(ui_port)
                        })

                        # Run docker compose with environment variables
                        compose_cmd = [
                            "docker", "compose",
                            "-f", str(compose_file),
                            "up", "-d"
                        ]
                        subprocess.run(compose_cmd, env=env, check=True)
                        logger.info("Steel server started successfully")
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(f"Failed to start Steel server: {e}")
                        
                except Exception as e:
                    logger.error(f"Failed to start Steel server: {e}")
                    raise RuntimeError(f"Steel server is required but failed to start: {e}")
            
            await self.resource_manager.start()
            self._started = True
            
    async def stop(self):
        """Stop the engine and cleanup resources."""
        if self._started:
            try:
                # Stop Steel server if it was started
                if self.constraints.require_steel:
                    try:
                        compose_file = Path(__file__).parent / 'steel-docker-compose.yml'
                        if compose_file.exists():
                            subprocess.run(["docker", "compose", "-f", str(compose_file), "down"], check=True)
                            logger.info("Stopped Steel server")
                        else:
                            logger.warning("steel-docker-compose.yml not found during shutdown")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to stop Steel server: {e}")

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