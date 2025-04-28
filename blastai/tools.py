"""Custom tools for BlastAI."""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Any, List
from browser_use import Controller, ActionResult
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from .scheduler import Scheduler

logger = logging.getLogger(__name__)

class Tools:
    """Manages a Controller instance with registered tools."""
    
    def __init__(self, scheduler: Optional[Scheduler] = None, task_id: Optional[str] = None, resource_manager = None):
        """Initialize Tools with optional scheduler and task_id.
        
        Args:
            scheduler: Optional scheduler instance for subtask management
            task_id: Optional task ID for tracking parent-child relationships
            resource_manager: Optional resource manager for task lifecycle management
        """
        self.controller = Controller()
        self.task_id = task_id
        self.cache_control = ""
        self.resource_manager = resource_manager
        
        # Get parent task's cache control if available
        if scheduler and task_id:
            task = scheduler.tasks.get(task_id)
            if task:
                self.cache_control = task.cache_options
        
        # Register subtask tools if scheduler is provided
        if scheduler:
            self._register_subtask_tools(scheduler)

    async def _get_first_subtask_result(self, scheduler: Scheduler, task_ids: List[str], as_final: bool = False) -> ActionResult:
        """Helper function to get first result from multiple subtasks.
        
        Args:
            task_ids: List of task IDs
            as_final: Whether to return result as final
            
        Returns:
            First available result
        """
        # Create actual Tasks (not just coroutine objects)
        task_map: Dict[asyncio.Task, str] = {}
        for tid in task_ids:
            coro = scheduler.get_task_result(tid)
            t = asyncio.create_task(coro)
            task_map[t] = tid

        try:
            # Wait until the first one finishes
            done, pending = await asyncio.wait(
                task_map.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )

            # Try each completed task until we find a valid result
            while done:
                completed_task = done.pop()
                completed_tid = task_map[completed_task]
                try:
                    result = await completed_task
                    if result:  # Only use if we got a valid result
                        # Cancel any others
                        for p in pending:
                            p.cancel()
                        for t in done:
                            t.cancel()

                        # Clean up the other subtasks in the resource manager
                        for other_tid in task_ids:
                            if other_tid != completed_tid:
                                await self.resource_manager.end_task(other_tid)

                        if as_final:
                            return ActionResult(
                                success=True,
                                extracted_content=result.final_result(),
                                is_done=True  # Mark this as the final result
                            )
                        else:
                            return ActionResult(
                                success=True,
                                extracted_content=(
                                    f"📋 First result from subtask {completed_tid}: "
                                    f"{result.final_result()}"
                                )
                            )
                except Exception as e:
                    logger.error(f"Task {completed_tid} failed: {e}")
                    continue

            # If we get here, no tasks produced a valid result
            return ActionResult(
                success=False,
                error="No valid results available from any subtask"
            )

        except Exception as e:
            return ActionResult(
                success=False,
                error=f"Failed to get first subtask result: {e}"
            )
    
    def _register_subtask_tools(self, scheduler: Scheduler):
        """Register tools that require a scheduler."""
        
        @self.controller.action("Launch a subtask")
        async def launch_subtask(task: str, optional_initial_search_or_url: Optional[str] = None, num_copies: Optional[int] = 1, browser: Optional[Any] = None) -> ActionResult:
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
                
            # Schedule the subtasks
            task_ids = []
            for _ in range(num_copies):
                task_id = scheduler.schedule_subtask(
                    description=task,
                    parent_task_id=parent_task_id,
                    cache_control=self.cache_control  # Inherit cache control from parent task
                )
                
                # Set initial URL if provided
                if task_id in scheduler.tasks and optional_initial_search_or_url:
                    scheduler.tasks[task_id].initial_url = optional_initial_search_or_url
                    
                task_ids.append(task_id)
            
            if num_copies == 1:
                return ActionResult(
                    success=True,
                    extracted_content=f"🚀 Launched subtask {task_ids[0]} to \"{task}\""
                )
            else:
                return ActionResult(
                    success=True,
                    extracted_content=f"🚀 Launched subtasks {','.join(task_ids)} to \"{task}\""
                )

        @self.controller.action("Get first result from subtask(s)")
        async def get_first_subtask_result(
            comma_separated_list_of_task_ids: str,
            browser: Optional[Any] = None
        ) -> ActionResult:
            """Get the first result from multiple subtasks running in parallel.
            
            Args:
                comma_separated_list_of_task_ids: Comma-separated list of task IDs
                browser: Optional browser instance
                
            Returns:
                First available result from any subtask
            """
            task_ids = [tid.strip() for tid in comma_separated_list_of_task_ids.split(',')]
            return await self._get_first_subtask_result(scheduler, task_ids, as_final=False)

        @self.controller.action("Extract content in parallel chunks")
        async def extract_content_parallel(goal: str, should_strip_link_urls: bool = False, browser: Optional[Any] = None, page_extraction_llm: Optional[BaseChatModel] = None) -> ActionResult:
            """Extract content by splitting into large chunks and processing in parallel.
            
            Args:
                goal: Extraction goal/query
                should_strip_link_urls: Whether to strip URLs from links
                browser: Browser instance
                
            Returns:
                Combined results from all chunks
            """
            if not browser:
                return ActionResult(
                    success=False,
                    error="Browser instance required"
                )

            try:
                # Get raw content
                page = await browser.get_current_page()
                import markdownify
                import json

                strip = []
                if should_strip_link_urls:
                    strip = ['a', 'img']

                content = markdownify.markdownify(await page.content(), strip=strip)

                # Add iframe content
                for iframe in page.frames:
                    if iframe.url != page.url and not iframe.url.startswith('data:'):
                        content += f'\n\nIFRAME {iframe.url}:\n'
                        content += markdownify.markdownify(await iframe.content())

                # Calculate chunk size to ensure max 8 chunks while maintaining min 3000 chars per chunk
                total_length = len(content)
                chunk_size = max(total_length // 8, 3000)  # At least 3000 chars per chunk
                chunks = []
                current_chunk = []
                current_size = 0
                
                for line in content.split('\n'):
                    line_size = len(line)
                    if current_size + line_size > chunk_size and current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = [line]
                        current_size = line_size
                    else:
                        current_chunk.append(line)
                        current_size += line_size
                        
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))

                start_time = time.time()

                # Process chunks in parallel
                prompt = 'Your task is to extract the content of this chunk of text. You will be given a chunk and a goal and you should extract all relevant information around this goal from the chunk. If the goal is vague, summarize the chunk. Respond in json format. Extraction goal: {goal}, Chunk: {chunk}'
                template = PromptTemplate(input_variables=['goal', 'chunk'], template=prompt)

                # tasks = []
                # for chunk in chunks:
                #     if not page_extraction_llm:
                #         return ActionResult(
                #             success=False,
                #             error="page_extraction_llm is required for content extraction"
                #         )
                #     tasks.append(page_extraction_llm.ainvoke(template.format(goal=goal, chunk=chunk)))
                # Process all chunks in one batch
                results = await page_extraction_llm.abatch([
                    template.format(goal=goal, chunk=chunk) for chunk in chunks
                ])
                
                end_time = time.time()
                total_time = end_time - start_time
                avg_time = total_time / len(chunks)
                logger.debug(f"Parallel extraction of {len(chunks)} chunks completed in {total_time:.2f}s (avg {avg_time:.2f}s per chunk), Content length: {total_length}, using chunk size: {chunk_size}")
                
                # Combine results
                combined_result = {}
                for result in results:
                    try:
                        chunk_data = json.loads(result.content)
                        # Merge dictionaries recursively
                        def merge_dicts(d1, d2):
                            for k, v in d2.items():
                                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                                    merge_dicts(d1[k], v)
                                elif k in d1 and isinstance(d1[k], list) and isinstance(v, list):
                                    d1[k].extend(v)
                                else:
                                    if k in d1:
                                        if isinstance(d1[k], list):
                                            if isinstance(v, list):
                                                d1[k].extend(v)
                                            else:
                                                d1[k].append(v)
                                        else:
                                            d1[k] = [d1[k], v]
                                    else:
                                        d1[k] = v
                        merge_dicts(combined_result, chunk_data)
                    except json.JSONDecodeError:
                        # If not JSON, treat as text
                        if not combined_result:
                            combined_result = result.content
                        else:
                            combined_result += "\n" + result.content

                msg = f'📄  Extracted from page (parallel):\n{json.dumps(combined_result, indent=2) if isinstance(combined_result, dict) else combined_result}\n'
                return ActionResult(
                    success=True,
                    extracted_content=msg
                )
                
            except Exception as e:
                return ActionResult(
                    success=False,
                    error=f"Failed to extract content in parallel: {str(e)}"
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