"""Custom tools for BlastAI."""

import asyncio
from typing import Optional, Any, List
from browser_use import Controller, ActionResult
from .scheduler import Scheduler

class Tools:
    """Manages a Controller instance with registered tools."""
    
    def __init__(self, scheduler: Optional[Scheduler] = None, task_id: Optional[str] = None):
        """Initialize Tools with optional scheduler and task_id.
        
        Args:
            scheduler: Optional scheduler instance for subtask management
            task_id: Optional task ID for tracking parent-child relationships
        """
        self.controller = Controller()
        self.task_id = task_id
        
        # Register subtask tools if scheduler is provided
        if scheduler:
            self._register_subtask_tools(scheduler)
    
    def _register_subtask_tools(self, scheduler: Scheduler):
        """Register tools that require a scheduler."""
        
        @self.controller.action("Launch a subtask")
        async def launch_subtask(task: str, optional_initial_search_or_url: Optional[str] = None, browser: Optional[Any] = None) -> ActionResult:
            """Launch a new subtask.
            
            Args:
                task: Task description
                optional_initial_search_or_url: Optional initial URL or search query for the task
                browser: Optional browser instance
                
            Returns:
                Result of launching the subtask
            """
            # Use task_id from Tools instance as parent_task_id
            parent_task_id = self.task_id
                
            # Schedule the subtask
            task_id = scheduler.schedule_subtask(
                description=task,
                parent_task_id=parent_task_id,
                cache_control=""  # No special cache control for subtasks
            )
            
            # Set initial URL if provided
            if task_id in scheduler.tasks and optional_initial_search_or_url:
                scheduler.tasks[task_id].initial_url = optional_initial_search_or_url
                
            return ActionResult(
                success=True,
                extracted_content=f"🚀 Launched subtask {task_id} to \"{task}\""
            )
                
        @self.controller.action("Get result(s) of subtask(s)")
        async def get_subtask_results(comma_separated_list_of_task_ids: str, browser: Optional[Any] = None) -> ActionResult:
            """Get the results of multiple subtasks in parallel.
            
            Args:
                comma_separated_list_of_task_ids: Comma-separated list of task IDs
                browser: Optional browser instance
                
            Returns:
                Combined results of all subtasks
            """
            # Parse task IDs
            task_ids = [tid.strip() for tid in comma_separated_list_of_task_ids.split(',')]
            
            # Create tasks for getting results in parallel
            tasks = [scheduler.get_task_result(task_id) for task_id in task_ids]
            
            try:
                # Wait for all results in parallel
                results = await asyncio.gather(*tasks)
                
                # Combine results
                combined_results = []
                for task_id, result in zip(task_ids, results):
                    if result:
                        combined_results.append(f"  Subtask {task_id}: {result.final_result()}")
                    else:
                        combined_results.append(f"  Subtask {task_id}: No result available")
                
                return ActionResult(
                    success=True,
                    extracted_content="📋 Subtask results:\n" + "\n".join(combined_results)
                )
            except Exception as e:
                return ActionResult(
                    success=False,
                    error=f"Failed to get subtask results: {str(e)}"
                )