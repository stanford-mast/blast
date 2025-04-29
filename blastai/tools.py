"""Custom tools for BlastAI."""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Any, List
from browser_use import Controller, ActionResult
import markdownify
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from .scheduler import Scheduler

logger = logging.getLogger(__name__)

class Tools:
    """Manages a Controller instance with registered tools."""
    
    def __init__(self, scheduler: Optional[Scheduler] = None, task_id: Optional[str] = None,
                 resource_manager = None, llm_model: Optional[BaseChatModel] = None):
        """Initialize Tools with optional scheduler and task_id.
        
        Args:
            scheduler: Optional scheduler instance for subtask management
            task_id: Optional task ID for tracking parent-child relationships
            resource_manager: Optional resource manager for task lifecycle management
            llm_model: Optional LLM model to use for content extraction
        """
        self.llm_model = llm_model
        # Determine which actions to exclude based on constraints
        exclude_actions = []
        if scheduler and scheduler.constraints.allow_parallelism.get("data", False):
            # If data parallelism is enabled, exclude regular extract_content
            exclude_actions.append("extract_content")
            
        self.controller = Controller(exclude_actions=exclude_actions)
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
            # Filter out current task ID from list
            task_ids = [tid.strip() for tid in comma_separated_list_of_task_ids.split(',')
                       if tid.strip() != self.task_id]
            if not task_ids:
                return ActionResult(
                    success=False,
                    error="No valid task IDs provided (filtered out current task)"
                )
            return await self._get_first_subtask_result(scheduler, task_ids, as_final=False)

        @self.controller.action("Extract page content to retrieve specific information from the page, e.g. all company names, a specific description, all information about, links with companies in structured format or simply links")
        async def extract_content_fast(goal: str, should_strip_link_urls: bool = False, browser: Optional[Any] = None, page_extraction_llm: Optional[BaseChatModel] = None) -> ActionResult:
            """Extract content by splitting into chunks and processing in parallel."""
            if not browser:
                return ActionResult(success=False, error="Browser instance required")

            try:
                # Record overall start time
                overall_start = time.time()

                # Get raw content
                page = await browser.get_current_page()
                content_start = time.time()
                
                strip = ['a', 'img'] if should_strip_link_urls else []
                content = markdownify.markdownify(await page.content(), strip=strip)

                # Add iframe content
                for iframe in page.frames:
                    if iframe.url != page.url and not iframe.url.startswith('data:'):
                        content += f'\n\nIFRAME {iframe.url}:\n'
                        content += markdownify.markdownify(await iframe.content())
                
                content_time = time.time() - content_start
                total_chars = len(content)

                # Get LLM instance ID
                llm_id = id(page_extraction_llm)

                # # First try with whole content (commented out for performance)
                # prompt = 'Your task is to extract the content of the page. You will be given a page and a goal and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format. Extraction goal: {goal}, Page: {page}'
                # template = PromptTemplate(input_variables=['goal', 'page'], template=prompt)
                
                # # Time single LLM call
                # single_start = time.time()
                # single_output = await page_extraction_llm.ainvoke(template.format(goal=goal, page=content))
                # single_time = time.time() - single_start

                # Now try with parallel chunks
                chunk_start = time.time()
                chunk_size = max(total_chars // 8, 3000)  # At least 3000 chars per chunk
                # Split content into chunks
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
                chunk_time = time.time() - chunk_start

                # Process chunks in parallel
                parallel_start = time.time()
                chunk_prompt = 'Your task is to extract the content of this chunk of text. You will be given a chunk and a goal and you should extract all relevant information around this goal from the chunk. If the goal is vague, summarize the chunk. Respond in json format. Extraction goal: {goal}, Chunk: {chunk}'
                chunk_template = PromptTemplate(input_variables=['goal', 'chunk'], template=chunk_prompt)

                # Use provided model or fallback to page_extraction_llm
                extraction_model = self.llm_model or page_extraction_llm
                parallel_results = await extraction_model.abatch([
                    chunk_template.format(goal=goal, chunk=chunk) for chunk in chunks
                ])
                parallel_time = time.time() - parallel_start

                # Log timing comparison
                overall_time = time.time() - overall_start
                start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start))
                end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start + overall_time))
                
                # logger.debug(
                #     f"extract_content_fast [{start_time} -> {end_time}] Content ({total_chars} chars): {content_time:.2f}s, "
                #     f"Single LLM ({llm_id}): {single_time:.2f}s, Chunking ({len(chunks)} chunks): {chunk_time:.2f}s, "
                #     f"Parallel LLM: {parallel_time:.2f}s, Total: {overall_time:.2f}s"
                # )

                # Helper function to merge JSON objects
                def merge_json(d1: dict, d2: dict) -> dict:
                    result = d1.copy()
                    for k, v in d2.items():
                        if k in result:
                            if isinstance(result[k], dict) and isinstance(v, dict):
                                result[k] = merge_json(result[k], v)
                            elif isinstance(result[k], list) and isinstance(v, list):
                                result[k].extend(v)
                            elif isinstance(result[k], list):
                                result[k].append(v)
                            elif isinstance(v, list):
                                result[k] = [result[k]] + v
                            else:
                                result[k] = [result[k], v]
                        else:
                            result[k] = v
                    return result

                # Process and merge results
                merged_json = {}
                text_results = []

                # # Try to parse single output as JSON
                # try:
                #     single_json = json.loads(single_output.content)
                #     merged_json = single_json
                # except json.JSONDecodeError:
                #     text_results.append("Full page analysis:")
                #     text_results.append(single_output.content)

                # Process and merge parallel results
                for i, result in enumerate(parallel_results):
                    try:
                        chunk_json = json.loads(result.content)
                        merged_json = merge_json(merged_json, chunk_json)
                    except json.JSONDecodeError:
                        text_results.append(result.content)

                # Build final output
                output_parts = []
                if merged_json:
                    output_parts.append("")
                    output_parts.append(json.dumps(merged_json, indent=2))
                if text_results:
                    if output_parts:
                        output_parts.append("\n")
                    output_parts.extend(text_results)

                msg = f'📄 Extracted from page:\n{"\n".join(output_parts)}\n'
                return ActionResult(extracted_content=msg, include_in_memory=True)
                
            except Exception as e:
                # Log timing info even on failure
                error_time = time.time() - overall_start
                logger.error(
                    f"Failed after {error_time:.2f}s: {str(e)}\n"
                    f"Content extraction: {content_time:.2f}s, "
                    f"LLM ({llm_id}): {single_time if 'single_time' in locals() else 'N/A'}s"
                )
                return ActionResult(
                    success=False,
                    error=f"Failed to extract content: {str(e)}"
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
            # Filter out current task ID from list
            task_ids = [tid.strip() for tid in comma_separated_list_of_task_ids.split(',')
                       if tid.strip() != self.task_id]
            if not task_ids:
                return ActionResult(
                    success=False,
                    error="No valid task IDs provided (filtered out current task)"
                )
            
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