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
- Maximum concurrent browsers
- Maximum memory usage
- Cost per minute/hour limits
- Minimum running executors (3)

Module Responsibilities:
- ResourceManager owns executor lifecycle and resource tracking
- Scheduler owns task states and relationships
- Both coordinate via task state changes
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from browser_use import Browser
from browser_use.browser.context import BrowserContext
from langchain_openai import ChatOpenAI

from .config import Settings, Constraints
from .executor import Executor
from .tools import Tools

logger = logging.getLogger(__name__)

@dataclass
class TaskPriorityGroup:
    """Group of tasks with same priority level.
    
    Used by scheduler.priority_sort() to organize tasks by priority.
    ResourceManager respects this ordering when allocating executors.
    
    Attributes:
        name: Priority group name (e.g. "cached_result", "subtask")
        task_ids: List of task IDs in this group
    """
    name: str
    task_ids: List[str]

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
        
        self._allocate_task = None
        self._monitor_task = None
        self._running = False
        self._total_cost_evicted_executors = 0.0
        self._executor_processes: Dict[str, List[int]] = {}  # task_id -> list of process IDs
        self._cost_history: List[Tuple[float, datetime]] = []  # List of (cost, timestamp) tuples
        self._prev_not_allocated = 0  # Track previous not_allocated count

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
                    
    def _get_memory_usage(self, executor: Executor) -> float:
        """Get memory usage for an executor.
        
        Uses process info to find Playwright/Chromium processes
        and sum their memory usage.
        
        Args:
            executor: Executor to check
            
        Returns:
            Memory usage in bytes
        """
        # Get process IDs for this executor
        if executor.task_id not in self._executor_processes:
            # First time - find all related processes
            pids = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if 'playwright' in proc.info['name'].lower() or 'chromium' in proc.info['name'].lower():
                        pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self._executor_processes[executor.task_id] = pids
            
        # Sum memory for known processes
        total_memory = 0
        for pid in self._executor_processes[executor.task_id][:]:  # Copy list to allow modification
            try:
                proc = psutil.Process(pid)
                total_memory += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process no longer exists - remove from list
                self._executor_processes[executor.task_id].remove(pid)
                
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
        
    def check_constraints_sat(self, with_new_executor: bool = False) -> bool:
        """Check if resource constraints are satisfied."""
        # Count running executors
        running_executors = sum(1 for task in self.scheduler.tasks.values()
                              if task.executor)
                              
        # Check max concurrent browsers
        if with_new_executor:
            if running_executors + 1 > self.constraints.max_concurrent_browsers:
                return False
        elif running_executors > self.constraints.max_concurrent_browsers:
            return False
            
        # Check memory limit if set
        if self.constraints.max_memory:
            total_memory = 0
            for task in self.scheduler.tasks.values():
                if task.executor:
                    total_memory += self._get_memory_usage(task.executor)
            if with_new_executor:
                # Estimate memory for new executor
                total_memory += 500 * 1024 * 1024  # Assume 500MB
            if total_memory > self.constraints.max_memory:
                return False
                
        # Check cost limits if set
        if self.constraints.max_cost_per_minute:
            cost_last_minute = self._get_cost(timedelta(minutes=1))
            if cost_last_minute > self.constraints.max_cost_per_minute:
                return False
        if self.constraints.max_cost_per_hour:
            cost_last_hour = self._get_cost(timedelta(hours=1))
            if cost_last_hour > self.constraints.max_cost_per_hour:
                return False
                
        return True
        
    async def _request_executor(self, task_id: str) -> Optional[Executor]:
        """Request a new executor if constraints allow."""
        if not self.check_constraints_sat(with_new_executor=True):
            # Only log constraint violation once per task
            if task_id not in self._reported_constraint_tasks:
                self._reported_constraint_tasks.add(task_id)
                
                # Check which constraint was violated
                running_executors = sum(1 for task in self.scheduler.tasks.values() if task.executor)
                if running_executors + 1 > self.constraints.max_concurrent_browsers:
                    logger.debug(f"Cannot create executor for task {task_id}: would exceed max_concurrent_browsers ({running_executors + 1} > {self.constraints.max_concurrent_browsers})")
                    return None
                    
                total_memory = sum(self._get_memory_usage(task.executor) for task in self.scheduler.tasks.values() if task.executor)
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
            
        # Create browser components
        browser = Browser(headless=self.constraints.require_headless)
        browser_context = BrowserContext(browser=browser)
        
        # Create fresh Tools instance with scheduler for this executor
        tools = Tools(scheduler=self.scheduler)
        
        # Create LLM
        llm = ChatOpenAI(model=self.constraints.llm_model)
        
        # Create and return executor
        logger.debug(f"Created new executor for task {task_id}")
        return Executor(
            browser=browser,
            browser_context=browser_context,
            controller=tools.controller,
            llm=llm,
            constraints=self.constraints,
            task_id=task_id,
            settings=self.settings,
            engine_hash=self.engine_hash,
            scheduler=self.scheduler
        )
        
    async def _evict_executor(self, task_id: str):
        """Evict an executor and clean up its resources."""
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
        
        # Clean up executor
        await task.executor.cleanup()
        
        # Clear executor reference and process list
        task.executor = None
        self._executor_processes.pop(task_id, None)
        
    async def _allocate_resources(self):
        """Background task for allocating resources to tasks."""
        self._start_time = time.time()
        
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
                    
                    # Look for completed tasks with executors
                    for other_task in self.scheduler.tasks.values():
                        if other_task.is_completed and other_task.executor:
                            other_lineage = self.scheduler.get_lineage(other_task.id)
                            
                            # Check if lineages match except for last task
                            if (len(task_lineage) == len(other_lineage) + 1 and
                                task_lineage[:-1] == other_lineage):
                                # Swap executor and process list
                                logger.debug(f"Reused executor for task {task.id} (previously used for task {other_task.id})")
                                task.executor = other_task.executor
                                other_task.executor = None
                                self._executor_processes[task.id] = self._executor_processes.pop(other_task.id, [])
                                
                                # Start execution
                                cached_plan = self.cache_manager.get_plan(
                                    task_lineage,
                                    task.cache_options
                                )
                                await self.scheduler.start_task_exec(
                                    task_id,
                                    task.executor,
                                    cached_plan
                                )
                                ready_tasks.remove(task_id)
                                break
                                
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
                        
                        # Start execution
                        await self.scheduler.start_task_exec(
                            task_id,
                            executor,
                            cached_plan
                        )
                        tasks_allocated += 1
                        tasks_not_allocated -= 1
                        
                if tasks_allocated > 0 or tasks_not_allocated != self._prev_not_allocated:
                    logger.debug(f"Allocated executors for {tasks_allocated} tasks (not allocated for: {tasks_not_allocated})")
                    self._prev_not_allocated = tasks_not_allocated
                        
            except Exception as e:
                logger.error(f"Error in resource allocation: {e}")
                
            await asyncio.sleep(1)  # Check every second
            
    async def _monitor_resources(self):
        """Background task for monitoring resource usage."""
        while self._running:
            try:
                # Check if constraints are satisfied
                if not self.check_constraints_sat():
                    current_time = time.time()
                    # Only log constraint violations every 30 seconds
                    if current_time - self._last_constraint_report >= 30:
                        self._last_constraint_report = current_time
                        
                        # Log which constraint was violated
                        running_executors = sum(1 for task in self.scheduler.tasks.values() if task.executor)
                        if running_executors > self.constraints.max_concurrent_browsers:
                            logger.debug(f"Resource monitor: exceeded max_concurrent_browsers ({running_executors} > {self.constraints.max_concurrent_browsers})")
                        
                        total_memory = sum(self._get_memory_usage(task.executor) for task in self.scheduler.tasks.values() if task.executor)
                        if total_memory > self.constraints.max_memory:
                            logger.debug(f"Resource monitor: exceeded max_memory ({total_memory / (1024 * 1024):.1f}MB > {self.constraints.max_memory / (1024 * 1024):.1f}MB)")
                        
                        cost_last_minute = self._get_cost(timedelta(minutes=1))
                        if cost_last_minute > self.constraints.max_cost_per_minute:
                            logger.debug(f"Resource monitor: exceeded max_cost_per_minute (${cost_last_minute:.2f} > ${self.constraints.max_cost_per_minute:.2f})")
                        
                        cost_last_hour = self._get_cost(timedelta(hours=1))
                        if cost_last_hour > self.constraints.max_cost_per_hour:
                            logger.debug(f"Resource monitor: exceeded max_cost_per_hour (${cost_last_hour:.2f} > ${self.constraints.max_cost_per_hour:.2f})")
                    
                    # First try evicting completed executors
                    for task in self.scheduler.tasks.values():
                        if task.is_completed and task.executor:
                            await self._evict_executor(task.id)
                            if self.check_constraints_sat():
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
                
            await asyncio.sleep(5)  # Check every 5 seconds