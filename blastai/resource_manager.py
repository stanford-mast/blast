"""Resource management for BLAST.

This module implements resource management and constraint enforcement for browser-LLM tasks.
It works in conjunction with the scheduler to coordinate task execution.

Resource Lifecycle:
1. Allocation:
   - Task becomes ready (not completed, no executor)
   - ResourceManager checks cache
   - Tries to reuse executor from completed task
   - Creates new executor if needed and constraints allow
   
2. Monitoring:
   - Tracks total executors, memory usage, and costs
   - Evicts executors from completed tasks when needed
   - Pauses tasks to maintain constraints
   
3. Cleanup:
   - Waits for any running tasks
   - Records final costs
   - Cleans up browser resources
   - Clears references and cache

Resource Constraints:
- Maximum concurrent browser contexts
- Maximum memory usage
- Cost per minute/hour limits
- Minimum running executors (3)

Browser Management:
- Each executor has its own isolated browser context
- When share_browser_process=true, contexts share a single browser process
- When share_browser_process=false, each context gets its own process

Module Responsibilities:
- ResourceManager owns executor lifecycle and resource tracking
- Scheduler owns task states and relationships
- Both coordinate via task state changes
"""

import asyncio
import logging
import os
import psutil
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from browser_use import Agent
from browser_use.browser import BrowserSession, BrowserProfile

from .config import Settings, Constraints
from .executor import Executor
from .tools import Tools
from .secrets import SecretsManager
from .utils import find_local_browser, init_model
from .resource_factory import create_executor, cleanup_stealth_profile_dir
from .models import TokenUsage

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages executor allocation and resource constraints."""
    
    def __init__(self, scheduler, constraints: Constraints,
                 settings: Settings, engine_hash: str,
                 cache_manager):
        """Initialize resource manager."""
        self.scheduler = scheduler
        self.constraints = constraints
        self.settings = settings
        self.engine_hash = engine_hash
        self.cache_manager = cache_manager
        
        # We don't use shared browser process to avoid VNC session management complexity
        
        self._allocate_task = None
        self._monitor_task = None
        self._running = False
        self._total_cost_evicted_executors = 0.0
        self._total_token_usage_evicted_executors = TokenUsage()
        self._cost_history: List[Tuple[float, datetime]] = []  # List of (cost, timestamp) tuples
        self._token_usage_history: List[Tuple[TokenUsage, datetime]] = [] # List of (token_usage, timestamp) tuples
        self._start_time = time.time()  # Track when the resource manager was created
        self._prev_not_allocated = 0  # Track previous not_allocated count
        self._prev_completed_with_executor = 0
        
        # Initialize secrets manager
        self._secrets_manager = SecretsManager()
        self._secrets_manager.load_secrets(self.settings.secrets_file_path)

        # For debugging output
        self._last_constraint_report = 0.0  # Last time we reported constraint violation
        self._reported_constraint_tasks: Set[str] = set()  # Tasks we've already reported constraints for
        
    async def start(self):
        """Start resource management."""
        if not self._running:
            self._running = True
            self._allocate_task = asyncio.create_task(self._allocate_resources())
            self._monitor_task = asyncio.create_task(self._monitor_resources())
            
    async def stop(self):
        """Stop resource management."""
        if self._running:
            self._running = False
            if self._allocate_task:
                self._allocate_task.cancel()
                try:
                    await self._allocate_task
                except asyncio.CancelledError:
                    pass
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
                    
            # No shared browser cleanup needed
                    
    def _get_total_memory_usage(self) -> float:
        """Get total memory usage for all browser processes created since engine start.
        
        Returns:
            Total memory usage in bytes
        """
        total_memory = 0
        
        # Find all headless_shell processes created after engine start
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                proc_name = proc.info['name'].lower()
                if (('headless_shell' in proc_name or
                     'chromium' in proc_name or
                     'playwright' in proc_name) and
                    proc.info['create_time'] >= self._start_time):
                    total_memory += proc.memory_info().rss
                    
                    # Include child processes
                    try:
                        for child in proc.children(recursive=True):
                            if child.create_time() >= self._start_time:
                                total_memory += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return total_memory
        
    def _get_cost(self, time_window: Optional[timedelta] = None) -> float:
        """Get total cost across all executors within time window.
        
        Args:
            time_window: Optional time window to calculate cost for
                        None means get total cost
        """
        now = datetime.now()
        
        # Calculate current total cost
        current_cost = self._total_cost_evicted_executors
        for task in self.scheduler.tasks.values():
            if task.executor:
                current_cost += task.executor.get_total_cost()
                
        # Add to history
        self._cost_history.append((current_cost, now))
        
        if time_window:
            cutoff = now - time_window
            
            # Keep most recent entry before cutoff if it exists
            oldest_entry = None
            for cost, ts in self._cost_history:
                if ts < cutoff:
                    oldest_entry = (cost, ts)
                else:
                    break
            
            # Filter history but keep oldest entry if found
            self._cost_history = (
                ([oldest_entry] if oldest_entry else []) +
                [(c, t) for c, t in self._cost_history if t >= cutoff]
            )
            
            # Calculate cost within window
            if len(self._cost_history) > 0:
                return self._cost_history[-1][0] - self._cost_history[0][0]
            return 0.0
            
        return current_cost
    
    def _get_token_usage(self, time_window: Optional[timedelta] = None) -> TokenUsage:
        """Get total token usage across all executors within time window.
        
        Args:
            time_window: Optional time window to calculate token usage for
                        None means get total token usage
        """
        now = datetime.now()
        
        # Calculate current total token usage
        # Make a copy to avoid modifying the original
        current_token_usage = self._total_token_usage_evicted_executors.copy()
        for task in self.scheduler.tasks.values():
            if task.executor:
                current_token_usage += task.executor.get_total_token_usage()
                
        # Add to history
        self._token_usage_history.append((current_token_usage, now))
        
        if time_window:
            cutoff = now - time_window
            
            # Keep most recent entry before cutoff if it exists
            oldest_entry = None
            for token_usage, ts in self._token_usage_history:
                if ts < cutoff:
                    oldest_entry = (token_usage, ts)
                else:
                    break
            
            # Filter history but keep oldest entry if found
            self._token_usage_history = (
                ([oldest_entry] if oldest_entry else []) +
                [(t, ts) for t, ts in self._token_usage_history if ts >= cutoff]
            )
            
            # Calculate token usage within window
            if len(self._token_usage_history) > 0:
                return self._token_usage_history[-1][0] - self._token_usage_history[0][0]
            return TokenUsage()
            
        return current_token_usage
        
    def check_constraints_sat(self, with_new_executors: int = 0) -> bool:
        """Check if resource constraints are satisfied.
        
        Args:
            with_new_executors: Number of new executors to check for
        """
        # Count running executors
        running_executors = sum(1 for task in self.scheduler.tasks.values()
                               if task.executor)
                               
        # Check max concurrent browsers
        if running_executors + with_new_executors > self.constraints.max_concurrent_browsers:
            return False
            
        # Check memory limit if set
        if self.constraints.max_memory is not None:
            total_memory = self._get_total_memory_usage()
            # Estimate memory for new executors (500MB each)
            total_memory += (500 * 1024 * 1024) * with_new_executors
            if total_memory > self.constraints.max_memory:
                return False
                
        # Check cost limits if set
        if self.constraints.max_cost_per_minute is not None:
            cost_last_minute = self._get_cost(timedelta(minutes=1))
            if cost_last_minute > self.constraints.max_cost_per_minute:
                return False
        if self.constraints.max_cost_per_hour is not None:
            cost_last_hour = self._get_cost(timedelta(hours=1))
            if cost_last_hour > self.constraints.max_cost_per_hour:
                return False
                
        return True
        

    async def _request_executor(self, task_id: str) -> Optional[Executor]:
        """Request a new executor if constraints allow."""
        if not self.check_constraints_sat(with_new_executors=1):
            # Only log constraint violation once per task
            if task_id not in self._reported_constraint_tasks:
                self._reported_constraint_tasks.add(task_id)
                
                # Check which constraint was violated
                running_executors = sum(1 for task in self.scheduler.tasks.values() if task.executor)
                if running_executors + 1 > self.constraints.max_concurrent_browsers:
                    logger.debug(f"Cannot create executor for task {task_id}: would exceed max_concurrent_browsers ({running_executors + 1} > {self.constraints.max_concurrent_browsers})")
                    return None
                    
                total_memory = self._get_total_memory_usage()
                if total_memory + (500 * 1024 * 1024) > self.constraints.max_memory:
                    logger.debug(f"Cannot create executor for task {task_id}: would exceed max_memory ({(total_memory + 500 * 1024 * 1024) / (1024 * 1024):.1f}MB > {self.constraints.max_memory / (1024 * 1024):.1f}MB)")
                    return None
                    
                cost_last_minute = self._get_cost(timedelta(minutes=1))
                if cost_last_minute > self.constraints.max_cost_per_minute:
                    logger.debug(f"Cannot create executor for task {task_id}: exceeded max_cost_per_minute (${cost_last_minute:.2f} > ${self.constraints.max_cost_per_minute:.2f})")
                    return None
                    
                cost_last_hour = self._get_cost(timedelta(hours=1))
                if cost_last_hour > self.constraints.max_cost_per_hour:
                    logger.debug(f"Cannot create executor for task {task_id}: exceeded max_cost_per_hour (${cost_last_hour:.2f} > ${self.constraints.max_cost_per_hour:.2f})")
                    return None
            return None
            
        # Get secrets if available
        secrets = self._secrets_manager.get_secrets()
        sensitive_data = secrets if secrets else None

        # Create new executor (no shared browser process)
        return await create_executor(
            task_id=task_id,
            constraints=self.constraints,
            settings=self.settings,
            scheduler=self.scheduler,
            resource_manager=self,
            engine_hash=self.engine_hash,
            sensitive_data=sensitive_data
        )
        
    async def end_task(self, task_id: str, cleanup_executor: bool = True):
        """Force end a task.
        
        This method:
        1. Stops the agent if present
        2. Cancels any running executor task
        3. Marks task as completed but unsuccessful
        4. Optionally cleans up executor resources
        5. Updates cost tracking
        
        Args:
            task_id: Task ID
            cleanup_executor: Whether to clean up the executor (default: True)
                            Set to False to keep executor alive for task continuity
        """
        task = self.scheduler.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        if task.is_completed:
            return  # Already completed
        
        # Stop the agent first if present
        if task.executor and hasattr(task.executor, 'agent') and task.executor.agent:
            logger.debug(f"Stopping agent for task {task_id}")
            task.executor.agent.stop()
            
        # Cancel executor run task if running
        if task.executor_run_task and not task.executor_run_task.done():
            logger.debug(f"Canceling executor_run_task for task {task_id}")
            task.executor_run_task.cancel()
            try:
                await task.executor_run_task
            except asyncio.CancelledError:
                pass
                
        # Clean up executor if present and cleanup_executor is True
        if task.executor and cleanup_executor:
            logger.debug(f"Cleaning up executor for task {task_id}")
            await self._evict_executor(task_id)
        elif task.executor and not cleanup_executor:
            logger.debug(f"Preserving executor for task {task_id} for future reuse")
            
        # Mark task as completed but unsuccessful
        await self.scheduler.complete_task(task_id, success=False)

    async def _evict_executor(self, task_id: str):
        """Evict an executor and clean up its resources."""
        # Clear interactive queues from task state
        task = self.scheduler.tasks.get(task_id)
        if not task:
            return
        task.interactive_queues = None
        task = self.scheduler.tasks.get(task_id)
        if not task or not task.executor:
            return
        logger.debug(f"Evicted executor for task {task_id}")
            
        # Wait for any running task to complete
        if task.executor_run_task:
            try:
                await task.executor_run_task
            except Exception:
                pass  # Ignore errors during cleanup
            
        # Add cost to evicted total
        self._total_cost_evicted_executors += task.executor.get_total_cost()

        # Add token usage to evicted total
        self._total_token_usage_evicted_executors += task.executor.get_total_token_usage()
        
        # Store a reference to the executor before clearing it
        executor = task.executor
        
        # Clear the executor reference from the task
        task.executor = None
        
        # Clean up executor using its cleanup method which properly handles VNC sessions
        if executor:
            try:
                # This will properly clean up the VNC session, browser session, and stealth profile
                await executor.cleanup()
                logger.debug(f"Cleaned up executor resources for task {task_id}")
            except Exception as e:
                logger.error(f"Error during executor cleanup: {e}")
        
        
    async def _allocate_resources(self):
        """Background task for allocating resources to tasks."""
        
        while self._running:
            try:
                # Get tasks ready for allocation
                ready_tasks = []
                for task in self.scheduler.tasks.values():
                    # Task must not be completed and not have an executor
                    if not task.is_completed and not task.executor:
                        # Check prerequisite if any
                        if task.prerequisite_task_id:
                            prereq = self.scheduler.tasks.get(task.prerequisite_task_id)
                            if not prereq or not prereq.is_completed:
                                continue
                        ready_tasks.append(task.id)
                        
                # Check cache for each task
                for task_id in ready_tasks[:]:
                    task = self.scheduler.tasks[task_id]
                    lineage = self.scheduler.get_lineage(task_id)
                    if self.cache_manager.get_result(lineage, task.cache_options):
                        await self.scheduler.complete_task(task_id)
                        ready_tasks.remove(task_id)
                        
                # Try to reuse executors
                for task_id in ready_tasks[:]:
                    task = self.scheduler.tasks[task_id]
                    task_lineage = self.scheduler.get_lineage(task_id)
                    
                    # Sort completed tasks with executors by priority
                    def task_priority(t):
                        # Skip tasks that don't meet basic criteria
                        if not t.is_completed or not t.executor:
                            return 999  # Low priority
                            
                        # Check if lineage matches
                        other_lineage = self.scheduler.get_lineage(t.id)
                        if not (len(task_lineage) == len(other_lineage) + 1 and task_lineage[:-1] == other_lineage):
                            return 999  # Low priority
                            
                        # Priority: is it the prerequisite task?
                        return 0 if t.id == task.prerequisite_task_id else 1
                    
                    sorted_tasks = sorted(self.scheduler.tasks.values(), key=task_priority)
                    
                    # Use the first suitable task (if any)
                    if sorted_tasks and task_priority(sorted_tasks[0]) < 999:
                        other_task = sorted_tasks[0]
                        
                        # Reuse the executor
                        executor = other_task.executor
                        other_task.executor = None  # Remove reference from old task
                        
                        logger.debug(f"Reusing executor from task {other_task.id} for task {task.id}")
                        
                        # Create new Tools instance with resource manager and queues
                        queues = task.interactive_queues
                        tools = Tools(
                            scheduler=self.scheduler,
                            task_id=task.id,
                            resource_manager=self,
                            human_request_queue=queues['to_client'] if queues else None,
                            human_response_queue=queues['from_client'] if queues else None
                        )
                        executor.set_task_id(task.id, tools.controller)
                        
                        # Start execution with reused executor
                        cached_plan = self.cache_manager.get_plan(
                            task_lineage,
                            task.cache_options
                        )
                        await self.scheduler.start_task_exec(
                            task_id,
                            executor,
                            cached_plan
                        )
                        ready_tasks.remove(task_id)
                                
                # Sort remaining tasks by priority
                priority_groups = self.scheduler.priority_sort(ready_tasks)
                
                # Try to allocate new executors
                tasks_allocated = 0
                tasks_not_allocated = len(ready_tasks)
                for group in priority_groups:
                    for task_id in group.task_ids:
                        # Skip tasks that have cached results (they were handled above)
                        task = self.scheduler.tasks[task_id]
                        lineage = self.scheduler.get_lineage(task_id)
                        if self.cache_manager.get_result(lineage, task.cache_options):
                            continue
                            
                        # Request new executor
                        executor = await self._request_executor(task_id)
                        if not executor:
                            # Stop if constraints would be violated
                            break
                            
                        # Get cached plan if available
                        cached_plan = self.cache_manager.get_plan(
                            lineage,
                            task.cache_options
                        )
                        
                        # Create Tools instance with queues
                        queues = task.interactive_queues
                        tools = Tools(
                            scheduler=self.scheduler,
                            task_id=task_id,
                            resource_manager=self,
                            human_request_queue=queues['to_client'] if queues else None,
                            human_response_queue=queues['from_client'] if queues else None
                        )
                        executor.set_task_id(task.id, tools.controller)
                        
                        # Start execution
                        await self.scheduler.start_task_exec(
                            task_id,
                            executor,
                            cached_plan
                        )
                        tasks_allocated += 1
                        tasks_not_allocated -= 1
                        
                # Get current executor stats
                running_executors = sum(1 for task in self.scheduler.tasks.values()
                                    if task.executor and not task.is_completed)
                completed_with_executor = sum(1 for task in self.scheduler.tasks.values()
                                         if task.is_completed)

                if tasks_allocated > 0 or tasks_not_allocated != self._prev_not_allocated or completed_with_executor != (self._prev_completed_with_executor or completed_with_executor):
                    logger.debug(f"Tasks: {tasks_allocated} allocated, {tasks_not_allocated} pending, {running_executors} running, {completed_with_executor} completed")
                    self._prev_not_allocated = tasks_not_allocated
                    self._completed_with_executor = completed_with_executor
                        
            except Exception as e:
                logger.error(f"Error in resource allocation: {e}")
                
            await asyncio.sleep(0.1)  # Check every second
            
    async def _monitor_resources(self):
        """Background task for monitoring resource usage."""
        while self._running:
            try:
                # Check if constraints are satisfied
                def get_num_unscheduled_tasks():
                    return sum(1 for task in self.scheduler.tasks.values()
                                if not task.is_completed and not task.executor)
                if not self.check_constraints_sat() or not self.check_constraints_sat(with_new_executors=get_num_unscheduled_tasks()):
                    current_time = time.time()
                    # Only log constraint violations every 30 seconds
                    if current_time - self._last_constraint_report >= 30:
                        self._last_constraint_report = current_time
                        
                        # Log which constraint was violated
                        running_executors = sum(1 for task in self.scheduler.tasks.values() if task.executor)
                        if running_executors > self.constraints.max_concurrent_browsers:
                            logger.debug(f"Resource monitor: exceeded max_concurrent_browsers ({running_executors} > {self.constraints.max_concurrent_browsers})")
                        
                        total_memory = self._get_total_memory_usage()
                        if self.constraints.max_memory is not None and total_memory > self.constraints.max_memory:
                            logger.debug(f"Resource monitor: exceeded max_memory ({total_memory / (1024 * 1024):.1f}MB > {self.constraints.max_memory / (1024 * 1024):.1f}MB)")
                        
                        cost_last_minute = self._get_cost(timedelta(minutes=1))
                        if self.constraints.max_cost_per_minute is not None and cost_last_minute > self.constraints.max_cost_per_minute:
                            logger.debug(f"Resource monitor: exceeded max_cost_per_minute (${cost_last_minute:.2f} > ${self.constraints.max_cost_per_minute:.2f})")
                        
                        cost_last_hour = self._get_cost(timedelta(hours=1))
                        if self.constraints.max_cost_per_hour is not None and cost_last_hour > self.constraints.max_cost_per_hour:
                            logger.debug(f"Resource monitor: exceeded max_cost_per_hour (${cost_last_hour:.2f} > ${self.constraints.max_cost_per_hour:.2f})")
                    
                    # First try evicting completed executors
                    for task in self.scheduler.tasks.values():
                        if task.is_completed and task.executor:
                            await self._evict_executor(task.id)
                            if self.check_constraints_sat(with_new_executors=get_num_unscheduled_tasks()):
                                break
                                
                # If still unsat, pause lowest priority tasks
                if not self.check_constraints_sat():
                    # Get tasks that have executors and aren't completed
                    running_tasks = [t.id for t in self.scheduler.tasks.values()
                                    if not t.is_completed and t.executor]
                    priority_groups = self.scheduler.priority_sort(running_tasks)
                    
                    # Pause tasks in reverse priority order
                    paused = 0
                    min_running = 3  # Keep at least 3 running
                    for group in reversed(priority_groups):
                        for task_id in group.task_ids:
                            task = self.scheduler.tasks[task_id]
                            if task.executor:
                                running = sum(1 for t in self.scheduler.tasks.values()
                                            if t.executor)
                                if running > min_running:
                                    await task.executor.pause()
                                    logger.debug(f"Paused task {task_id}")
                                    paused += 1
                                    if self.check_constraints_sat():
                                        break
                        if self.check_constraints_sat():
                            break
                                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                
            await asyncio.sleep(5)  # Check every 5 second