"""Custom tools for BlastAI."""

from typing import Optional, Any
from browser_use import Controller, ActionResult
from .scheduler import Scheduler

class Tools:
    """Manages a Controller instance with registered tools."""
    
    def __init__(self, scheduler: Optional[Scheduler] = None):
        """Initialize Tools with optional scheduler.
        
        Args:
            scheduler: Optional scheduler instance for subtask management
        """
        self.controller = Controller()
        
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
            # Get current task ID from browser if available
            parent_task_id = None
            if browser and hasattr(browser, 'task_id'):
                parent_task_id = browser.task_id
                
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
                extracted_content=f"🚀 Launched task {task_id} to \"{task}\""
            )
            
        @self.controller.action("Get results of a subtask")
        async def get_subtask_results(task_id: str, browser: Optional[Any] = None) -> ActionResult:
            """Get the results of a previously launched subtask.
            
            Args:
                task_id: ID of the subtask
                browser: Optional browser instance
                
            Returns:
                Results of the subtask
            """            
            # Get task result
            try:
                result = await scheduler.get_task_result(task_id)
                if result:
                    return ActionResult(
                        success=True,
                        extracted_content=f"📋 Result of task {task_id}: {result.final_result()}"
                    )
                else:
                    return ActionResult(
                        success=False,
                        error=f"No result available for task {task_id}"
                    )
            except Exception as e:
                return ActionResult(
                    success=False,
                    error=f"Failed to get result for task {task_id}: {str(e)}"
                )