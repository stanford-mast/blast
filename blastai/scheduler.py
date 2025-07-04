"""Task scheduling and management for BLAST.

This module implements task scheduling and state management for browser-LLM tasks.
It works in conjunction with the resource manager to coordinate task execution.

Task Lifecycle:
1. Creation: Task is scheduled via schedule_task() or schedule_subtask()
   - Assigned unique ID
   - Initial state: not completed, no executor
   - Checked for cached result
   
2. Resource Allocation (handled by ResourceManager):
   - Task waits for prerequisite completion if any
   - ResourceManager assigns executor (new or reused)
   - Scheduler starts execution via start_task_exec()
   
3. Execution:
   - Task runs via executor.run()
   - Progress streamed via stream_task_events()
   - Result obtained via get_task_result()
   
4. Completion:
   - Task marked completed via complete_task()
   - Result cached if successful
   - Executor freed for reuse/eviction

Task Relationships:
- Prerequisites: Task A must complete before Task B starts
- Parent/Child: Task B is a subtask of Task A
- Lineage: Chain of parent tasks used for executor reuse

Task Priorities (highest to lowest):
1. Tasks with cached results
2. Tasks with cached execution plans
3. Subtasks of running tasks
4. Tasks with paused executors
5. Remaining tasks (FIFO order)

Module Responsibilities:
- Scheduler owns task states and relationships
- ResourceManager owns executor lifecycle
- Both coordinate via task state changes
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, AsyncIterator, Union
from dataclasses import dataclass
from datetime import datetime

from .response import AgentReasoning, AgentHistoryListResponse
from browser_use.agent.views import AgentHistoryList
from .executor import Executor
from .planner import Planner

logger = logging.getLogger(__name__)

@dataclass
class TaskState:
    """State of a scheduled task.
    
    This class tracks all state for a single task, including:
    - Basic info: ID, description
    - Execution state: executor, run task, result
    - Relationships: prerequisite task, parent task
    - Timing: schedule time, start time, completion time
    - Cache settings and success status
    
    The state transitions through these phases:
    1. Created: Has ID and description
    2. Ready: Prerequisites completed
    3. Running: Has executor and run task
    4. Completed: Has result (success=True) or failed (success=False)
    
    Attributes:
        id: Unique task identifier
        description: Task description/instructions
        executor: Optional executor running this task
        prerequisite_task_id: Optional ID of task that must complete first
        parent_task_id: Optional ID of parent task if this is a subtask
        executor_run_task: Optional asyncio task running the executor
        result: Optional result from execution or cache
        cache_options: Cache control directives
        completed: Whether task is done (success or failure)
        success: Whether task completed successfully
        time_schedule: When task was scheduled
        time_start_exec: When execution started
        time_complete: When task completed
        initial_url: Optional starting URL for browser tasks
    """
    
    id: str
    description: str
    executor: Optional[Executor] = None
    prerequisite_task_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    executor_run_task: Optional[asyncio.Task] = None
    result: Optional[AgentHistoryList] = None
    cache_options: str = ""
    completed: bool = False
    success: bool = False
    time_schedule: Optional[datetime] = None
    time_start_exec: Optional[datetime] = None
    time_complete: Optional[datetime] = None
    initial_url: Optional[str] = None
    interactive_queues: Optional[Dict[str, asyncio.Queue]] = None  # Queues for interactive mode
    # TODO: Add stopped so that any stopped tasks or human-intervened tasks can only
    # permit executor reuse if there is a prerequisite task, can't be lineage based
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.completed
        
    @property
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return bool(self.executor_run_task and not self.executor_run_task.done())
        
    @property
    def lineage(self) -> List[str]:
        """Get task lineage (list of ancestor task descriptions)."""
        return [self.description]  # Will be extended by get_lineage

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

class Scheduler:
    """Manages task scheduling and execution.
    
    This class is responsible for:
    - Maintaining task states and relationships
    - Scheduling new tasks and subtasks
    - Starting task execution
    - Streaming execution events
    - Managing task completion and results
    - Coordinating with cache manager
    
    It works with ResourceManager which handles:
    - Executor allocation and cleanup
    - Resource constraint enforcement
    - Executor reuse across tasks
    
    The scheduler focuses on task state and relationships while
    letting the resource manager handle executor lifecycle.
    """
    
    def _generate_task_id(self) -> str:
        """Generate an alphabetic task ID.
        
        Generates IDs in sequence: A, B, C, ..., Z, AA, AB, AC, ...
        
        Returns:
            Alphabetic task ID
        """
        num = len(self.tasks)
        result = []
        while num >= 0:
            num, remainder = divmod(num, 26)
            result.append(chr(65 + remainder))  # 65 is ASCII for 'A'
            num -= 1
        return ''.join(reversed(result))
        
    def __init__(self, constraints, cache_manager, planner):
        """Initialize scheduler with constraints and cache manager."""
        self.constraints = constraints
        self.cache_manager = cache_manager
        self.planner = planner
        self.tasks: Dict[str, TaskState] = {}
        
    def schedule_task(self, description: str, prerequisite_task_id: Optional[str] = None,
                     parent_task_id: Optional[str] = None, cache_control: str = "",
                     interactive_queues: Optional[Dict[str, asyncio.Queue]] = None,
                     initial_url: Optional[str] = None) -> str:
        """Schedule a new task.
        
        Creates a new task state and checks for cached results.
        Does not allocate resources - that's handled by ResourceManager.
        
        Args:
            description: Task description
            prerequisite_task_id: Optional ID of task that must complete before this one
            parent_task_id: Optional ID of parent task for subtasks
            cache_control: Cache control directives
            
        Returns:
            Task ID
        """
        # Generate unique alphabetic task ID
        task_id = self._generate_task_id()
        
        # Create task state
        task = TaskState(
            id=task_id,
            description=description,
            prerequisite_task_id=prerequisite_task_id,
            parent_task_id=parent_task_id,
            cache_options=cache_control,
            time_schedule=datetime.now(),
            interactive_queues=interactive_queues,
            initial_url=initial_url
        )
        
        # Add task to dictionary first
        self.tasks[task_id] = task
        
        # Then check cache
        lineage = self.get_lineage(task_id)
        cached_result = self.cache_manager.get_result(lineage, cache_control)
        if cached_result:
            logger.debug(f"Reused result from cache for task {task_id}")
            task.result = cached_result
            task.completed = True
            task.success = True
            task.time_complete = datetime.now()
        return task_id
        
    def schedule_subtask(self, description: str, parent_task_id: str,
                        cache_control: str = "",
                        interactive_queues: Optional[Dict[str, asyncio.Queue]] = None,
                        initial_url: Optional[str] = None) -> str:
        """Schedule a subtask of an existing task.
        
        Subtasks have a parent-child relationship but can run in parallel.
        They do not have prerequisite relationships with each other.
        
        Args:
            description: Task description
            parent_task_id: ID of parent task
            cache_control: Cache control directives
            
        Returns:
            Task ID
        """
        if parent_task_id not in self.tasks:
            raise ValueError(f"Parent task {parent_task_id} not found")
            
        # Schedule task with parent but no prerequisite
        task_id = self.schedule_task(
            description=description,
            parent_task_id=parent_task_id,
            prerequisite_task_id=None,  # Explicitly set no prerequisite
            cache_control=cache_control,
            interactive_queues=interactive_queues,
            initial_url=initial_url
        )
        return task_id
        
    async def get_task_result(self, task_id: str) -> Optional[AgentHistoryList]:
        """Get task result, waiting for completion if needed.
        
        This method will:
        1. Wait for prerequisite task if needed
        2. Return cached result if available
        3. Wait for executor to be assigned and task to complete
        4. Return result when available
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result when available
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        # Wait for prerequisite if any
        if task.prerequisite_task_id:
            prereq = self.tasks.get(task.prerequisite_task_id)
            if not prereq:
                raise ValueError(f"Prerequisite task {task.prerequisite_task_id} not found")
            if not prereq.is_completed:
                await self.get_task_result(task.prerequisite_task_id)
                
        # Keep checking until task completes or fails
        while True:
            # Return cached result if available
            if task.is_completed:
                return task.result
            elif task.result:  # Has result but not marked completed
                await self.complete_task(task_id, task.result)
                return task.result
                
            # Wait for executor result if running
            if task.executor_run_task:
                try:
                    # Wait for task to complete
                    result = await task.executor_run_task
                    
                    # Mark task as complete and cache result
                    await self.complete_task(task_id, result)
                    
                    # Return result
                    return result
                    
                except Exception as e:
                    # logger.error(f"Task {task_id} failed: {e}")
                    # Mark task as complete but failed
                    await self.complete_task(task_id, success=False)
                    raise
                    
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
    async def stream_task_events(self, task_id: str) -> AsyncIterator[Union[AgentReasoning, AgentHistoryListResponse]]:
        """Stream task execution events.
        
        This method yields:
        1. Events from uncompleted prerequisite tasks and their subtasks
        2. Events from the main task's subtasks
        3. Events from the main task itself
        4. Final result when complete
        
        Args:
            task_id: Task ID
            
        Yields:
            Task execution events
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        # Track seen reasonings and completed tasks to avoid duplicates
        seen_reasonings = {}
        yielded_completed_tasks = set()
        
        # Get uncompleted prerequisite tasks
        dependency_ids = self._get_dependency_ids(task_id)
        uncompleted_deps = [dep_id for dep_id in dependency_ids
                          if not self.tasks[dep_id].is_completed]
        
        while True:
            # Get all task IDs to process in this iteration
            all_task_ids = set()
            
            # Add subtasks from uncompleted prerequisites
            for dep_id in uncompleted_deps:
                subtask_ids = self._get_subtask_ids(dep_id)
                all_task_ids.update(subtask_ids)
                all_task_ids.add(dep_id)
            
            # Add main task's subtasks
            all_task_ids.update(self._get_subtask_ids(task_id))
            
            # Ensure main task is processed last
            all_task_ids.discard(task_id)
            ordered_task_ids = list(all_task_ids) + [task_id]
            
            # Process each task
            for current_task_id in ordered_task_ids:
                current_task = self.tasks[current_task_id]
                
                # Handle running tasks
                if current_task.executor and not current_task.is_completed:
                    # Get new reasonings
                    reasonings = await current_task.executor.get_reasoning()
                    for reasoning in reasonings:
                        key = (reasoning.type, reasoning.thought_type, reasoning.content)
                        if current_task_id not in seen_reasonings:
                            seen_reasonings[current_task_id] = set()
                        if key not in seen_reasonings[current_task_id]:
                            seen_reasonings[current_task_id].add(key)
                            yield reasoning
                            
                    # Check for executor result
                    if current_task.executor_run_task and current_task.executor_run_task.done():
                        try:
                            result = await current_task.executor_run_task
                            await self.complete_task(current_task_id, result)
                            if result and current_task_id not in yielded_completed_tasks:
                                yielded_completed_tasks.add(current_task_id)
                                yield AgentHistoryListResponse.from_history(
                                    history=result,
                                    task_id=current_task_id
                                )
                        except Exception as e:
                            logger.error(f"Task {current_task_id} failed: {e}", exc_info=True)
                
                # Handle newly completed tasks
                elif current_task.is_completed and current_task.result and current_task_id not in yielded_completed_tasks:
                    yielded_completed_tasks.add(current_task_id)
                    yield AgentHistoryListResponse.from_history(
                        history=current_task.result,
                        task_id=current_task_id
                    )
                        
            # Then check main task
            if task.executor:
                # Get new reasonings
                reasonings = await task.executor.get_reasoning()
                for reasoning in reasonings:
                    key = (reasoning.type, reasoning.thought_type, reasoning.content)
                    if task_id not in seen_reasonings:
                        seen_reasonings[task_id] = set()
                    if key not in seen_reasonings[task_id]:
                        seen_reasonings[task_id].add(key)
                        yield reasoning
                        
            # Break if main task is completed
            if task.is_completed:
                break
                    
            await asyncio.sleep(0.1)  # Prevent tight loop
            
    async def start_task_exec(self, task_id: str, executor: Executor,
                           cached_plan: Optional[AgentHistoryList] = None):
        """Start task execution.
        
        This method:
        1. Assigns executor to task
        2. Records start time
        3. Gets plan from planner
        4. Creates execution coroutine
        5. Starts async execution task
        
        Args:
            task_id: Task ID
            executor: Executor to use
            cached_plan: Optional cached execution plan
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        if task.is_running:
            raise RuntimeError(f"Task {task_id} is already running")
            
        task.executor = executor
        task.time_start_exec = datetime.now()
        
        # Create execution coroutine
        if cached_plan:
            coro = executor.run(cached_plan)
        else:
            # Calculate subtask depth
            depth = 0
            current_id = task.parent_task_id
            while current_id:
                depth += 1
                parent = self.tasks.get(current_id)
                if not parent:
                    break
                current_id = parent.parent_task_id
            
            # Collect previous task information
            previous_tasks = []
            current_id = task.prerequisite_task_id
            
            # Recursively collect all prerequisite tasks and their prerequisites
            def collect_prerequisites(task_id):
                if not task_id:
                    return []
                
                prereq_task = self.tasks.get(task_id)
                if not prereq_task:
                    return []
                
                # Recursively get prerequisites of this prerequisite
                prereq_list = collect_prerequisites(prereq_task.prerequisite_task_id)
                
                # Add this prerequisite to the list if it has a result
                if prereq_task.result:
                    final_response = prereq_task.result.final_result()
                    if final_response:
                        prereq_list.append((prereq_task.description, final_response))
                
                return prereq_list
            
            # Collect all prerequisite tasks
            previous_tasks = collect_prerequisites(task.prerequisite_task_id)
            
            # Get plan from planner
            plan = await self.planner.plan(
                task.description,
                subtask_depth=depth,
                previous_tasks=previous_tasks,
                initial_url=task.initial_url
            )
            logger.debug(f"Planned task {task_id} to '{task.description}' with '{plan}'")
            
            # Start execution coroutine
            logger.debug(f"Running task {task_id} with browser_use")
            coro = executor.run(plan, task.initial_url)
            
        # Create and store task
        task.executor_run_task = asyncio.create_task(coro)

    async def complete_task(self, task_id: str, result: Optional[AgentHistoryList] = None, success: bool = None):
        """Mark task as completed.
        
        This method:
        1. Marks task as completed
        2. Records completion time
        3. Stores result if provided
        4. Updates cache if successful
        
        Args:
            task_id: Task ID
            result: Optional task result
            success: Optional success flag (defaults to True if result provided, False if not)
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if task.is_completed:
            return  # Already completed
            
        task.completed = True
        task.time_complete = datetime.now()
        
        task.executor_run_task = None  # Clear the run task since it's done
        
        # Set success flag based on result or explicit success parameter
        task.success = success if success is not None else bool(result)
        
        if result:
            task.result = result
            logger.debug(f"Completed task {task_id} successfully")
            
            # Cache result and plan only for successful tasks
            if task.success:
                self.cache_manager.update_result(
                    task_lineage=self.get_lineage(task_id),
                    result=result,
                    cache_control=task.cache_options
                )
                
                self.cache_manager.update_plan(
                    task_lineage=self.get_lineage(task_id),
                    plan=result,
                    cache_control=task.cache_options
                )
        else:
            logger.debug(f"Completed task {task_id} without result")
            
    def get_lineage(self, task_id: str) -> List[str]:
        """Get list of task descriptions representing the conversation history.
        
        This method follows prerequisite relationships to build the complete
        conversation context leading up to this task. Used for:
        - Cache key generation
        - Executor reuse decisions
        - Conversation continuity
        
        Args:
            task_id: Task ID
            
        Returns:
            List of task descriptions in conversation order
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        lineage = [task.description]
        current_id = task.prerequisite_task_id
        
        while current_id:
            prereq = self.tasks.get(current_id)
            if not prereq:
                break
            lineage.insert(0, prereq.description)
            current_id = prereq.prerequisite_task_id
            
        return lineage
        
    def _get_dependency_ids(self, task_id: str) -> List[str]:
        """Get IDs of all tasks in dependency chain.
        
        This includes:
        1. All prerequisite tasks recursively
        2. All subtasks of those tasks
        
        Args:
            task_id: Task ID
            
        Returns:
            List of all dependent task IDs in order:
            - First prerequisite tasks (oldest to newest)
            - Then their subtasks (depth-first)
            - Finally the task itself and its subtasks
        """
        visited = set()
        dependency_ids = []
        
        def get_prereqs(tid: str):
            """Get all prerequisite tasks recursively."""
            if tid in visited:
                return
            visited.add(tid)
            
            task = self.tasks.get(tid)
            if not task:
                return
                
            # First get prerequisites recursively
            if task.prerequisite_task_id:
                get_prereqs(task.prerequisite_task_id)
                
            # Then add this task and its subtasks
            dependency_ids.append(tid)
            subtasks = self._get_subtask_ids(tid)
            for subtask_id in subtasks:
                if subtask_id not in visited:
                    dependency_ids.append(subtask_id)
                    visited.add(subtask_id)
                    
        get_prereqs(task_id)
        return dependency_ids

    def _get_subtask_ids(self, task_id: str) -> List[str]:
        """Get IDs of all subtasks of a task recursively.
        
        This method returns all subtasks at any depth in the task tree,
        not just immediate children.
        
        Args:
            task_id: Task ID
            
        Returns:
            List of all subtask IDs in the task tree
        """
        subtask_ids = []
        
        # Get immediate children
        direct_subtasks = [t.id for t in self.tasks.values() if t.parent_task_id == task_id]
        
        # Recursively get their subtasks
        for subtask_id in direct_subtasks:
            subtask_ids.append(subtask_id)
            subtask_ids.extend(self._get_subtask_ids(subtask_id))
            
        return subtask_ids
        
    def _get_prereq_ids(self, task_id: str, running_only: bool = False) -> List[str]:
        """Get IDs of all prerequisite tasks recursively.
        
        This method returns all prerequisite tasks at any depth in the prerequisite chain.
        
        Args:
            task_id: Task ID
            running_only: If True, only return running prerequisites
            
        Returns:
            List of all prerequisite task IDs
        """
        prereq_ids = []
        visited = set()
        
        def collect_prereqs(tid: str):
            if tid in visited:
                return
            visited.add(tid)
            
            task = self.tasks.get(tid)
            if not task:
                return
                
            if task.prerequisite_task_id:
                prereq_task = self.tasks.get(task.prerequisite_task_id)
                if prereq_task:
                    # Only include if not filtering or if running and filtering
                    if not running_only or prereq_task.is_running:
                        prereq_ids.append(task.prerequisite_task_id)
                    # Always recurse regardless of running status
                    collect_prereqs(task.prerequisite_task_id)
        
        collect_prereqs(task_id)
        return prereq_ids
        
    def priority_sort(self, task_ids: List[str]) -> List[TaskPriorityGroup]:
        """Sort tasks by priority.
        
        Priority order (highest to lowest):
        1. Tasks with cached results
        2. Tasks with cached plans
        3. Subtasks of running tasks
        4. Tasks with paused executors
        5. Remaining tasks (FIFO)
        
        Args:
            task_ids: List of task IDs
            
        Returns:
            List of TaskPriorityGroup objects
        """
        # Group tasks by priority
        groups = []

        # Group 0: Tasks with cached results
        cached_result_tasks = []
        for task_id in task_ids:
            task = self.tasks[task_id]
            lineage = self.get_lineage(task_id)
            if self.cache_manager.get_result(lineage, task.cache_options):
                cached_result_tasks.append(task_id)
        if cached_result_tasks:
            groups.append(TaskPriorityGroup("cached_result", cached_result_tasks))
        
        # Group 1: Tasks with cached plans
        cached_plan_tasks = []
        for task_id in task_ids:
            if task_id not in cached_result_tasks:  # Skip if already in higher priority group
                task = self.tasks[task_id]
                lineage = self.get_lineage(task_id)
                if self.cache_manager.get_plan(lineage, task.cache_options):
                    cached_plan_tasks.append(task_id)
        if cached_plan_tasks:
            groups.append(TaskPriorityGroup("cached_plan", cached_plan_tasks))
            
        # Group 2: Subtasks
        subtasks = []
        for task_id in task_ids:
            if task_id not in cached_result_tasks and task_id not in cached_plan_tasks:
                task = self.tasks[task_id]
                if task.parent_task_id:
                    subtasks.append(task_id)
        if subtasks:
            groups.append(TaskPriorityGroup("subtask", subtasks))
            
        # Group 3: Tasks with paused executors
        paused_tasks = []
        for task_id in task_ids:
            if task_id not in cached_result_tasks and task_id not in cached_plan_tasks and task_id not in subtasks:
                task = self.tasks[task_id]
                if task.executor and hasattr(task.executor, '_paused') and task.executor._paused:
                    paused_tasks.append(task_id)
        if paused_tasks:
            groups.append(TaskPriorityGroup("resume", paused_tasks))
            
        # Group 4: Remaining tasks (FIFO)
        remaining = [
            task_id for task_id in task_ids
            if task_id not in cached_result_tasks
            and task_id not in cached_plan_tasks
            and task_id not in subtasks
            and task_id not in paused_tasks
        ]
        if remaining:
            groups.append(TaskPriorityGroup("fifo", remaining))
            
        return groups