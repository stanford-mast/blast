"""Custom tools for BlastAI."""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Any, List, Tuple, cast
from browser_use import Controller, ActionResult
from browser_use.browser import BrowserSession
import markdownify
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import UserMessage, SystemMessage
from .scheduler import Scheduler
from .response import HumanRequest, HumanResponse

logger = logging.getLogger(__name__)

class Tools:
    """Manages a Controller instance with registered tools."""
    
    def __init__(self, scheduler: Optional[Scheduler] = None, task_id: Optional[str] = None,
                 resource_manager = None, llm_model: Optional[BaseChatModel] = None,
                 human_request_queue: Optional[asyncio.Queue] = None,
                 human_response_queue: Optional[asyncio.Queue] = None):
        """Initialize Tools with optional scheduler and task_id.
        
        Args:
            scheduler: Optional scheduler instance for subtask management
            task_id: Optional task ID for tracking parent-child relationships
            resource_manager: Optional resource manager for task lifecycle management
            llm_model: Optional LLM model to use for content extraction
        """
        self.llm_model = llm_model
        self.human_request_queue = human_request_queue
        self.human_response_queue = human_response_queue
        self.interactive_queues: Optional[Dict[str, asyncio.Queue]] = None
        
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
        
        # Register tools based on available functionality
        if scheduler:
            self._register_subtask_tools(scheduler)
        if human_request_queue and human_response_queue:
            self._register_human_tools()
            self.interactive_queues = {
                "to_client": human_request_queue,  # to_client is for sending TO client (human requests)
                "from_client": human_response_queue  # from_client is for receiving FROM client (human responses)
            }

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
        async def launch_subtask(task: str, optional_initial_search_or_url: Optional[str] = None, num_copies: int = 1) -> ActionResult:
            """Launch a new subtask.
            
            Args:
                task: Task description
                optional_initial_search_or_url: Optional initial URL or search query for the task
                num_copies: Optional number of parallel copies to launch
                
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
                    cache_control=self.cache_control,  # Inherit cache control from parent task
                    interactive_queues=self.interactive_queues
                )
                
                # Set initial URL if provided
                if task_id in scheduler.tasks and optional_initial_search_or_url:
                    scheduler.tasks[task_id].initial_url = optional_initial_search_or_url
                    
                task_ids.append(task_id)
            
            if num_copies == 1:
                return ActionResult(
                    extracted_content=f"🚀 Launched subtask {task_ids[0]} to \"{task}\""
                )
            else:
                return ActionResult(
                    extracted_content=f"🚀 Launched subtasks {','.join(task_ids)} to \"{task}\""
                )

        @self.controller.action("Get first result from subtask(s)")
        async def get_first_subtask_result(
            comma_separated_list_of_task_ids: str
        ) -> ActionResult:
            """Get the first result from multiple subtasks running in parallel.
            
            Args:
                comma_separated_list_of_task_ids: Comma-separated list of task IDs
                
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

        @self.controller.action("""Extract structured, semantic data (e.g. product description, price, all information about XYZ) from the current webpage based on a textual query.
Only use this for extracting info from a single product/article page, not for entire listings or search results pages.
""")
        async def extract_structured_content_fast(query: str, browser_session: BrowserSession = None, page_extraction_llm=None) -> ActionResult:
            """Extract structured content by splitting into chunks and processing in parallel."""
            if not browser_session:
                return ActionResult(success=False, error="Browser session required")

            try:
                # Record overall start time
                overall_start = time.time()
                
                # Get raw content
                page = await browser_session.get_current_page()
                content_start = time.time()
                
                # Determine if we should include links based on query
                strip = []
                include_links = False
                lower_query = query.lower()
                url_keywords = ['url', 'links']
                if any(keyword in lower_query for keyword in url_keywords):
                    include_links = True

                if not include_links:
                    strip = ['a', 'img']
                
                # Get page content
                content = markdownify.markdownify(await page.content(), strip=strip)
                
                # Add iframe content
                for iframe in page.frames:
                    try:
                        await iframe.wait_for_load_state(timeout=5000)  # extra on top of already loaded page
                    except Exception as e:
                        pass

                    if iframe.url != page.url and not iframe.url.startswith('data:'):
                        content += f'\n\nIFRAME {iframe.url}:\n'
                        try:
                            iframe_html = await iframe.content()
                            iframe_markdown = markdownify.markdownify(iframe_html, strip=strip)
                            content += iframe_markdown
                        except Exception as e:
                            logger.debug(f'Error extracting iframe content from within page {page.url}: {type(e).__name__}: {e}')
                
                content_time = time.time() - content_start
                total_chars = len(content)
                
                # Limit content size if needed
                max_chars = 40000
                if len(content) > max_chars:
                    content = (
                        content[: max_chars // 2]
                        + '\n... left out the middle because it was too long ...\n'
                        + content[-max_chars // 2 :]
                    )
                
                # Split content into chunks for parallel processing
                chunk_start = time.time()
                chunk_size = max(total_chars // 8, 3000)  # At least 3000 chars per chunk
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
                
                # Structured data extraction prompt
                chunk_prompt = """You convert websites into structured information. Extract information from this webpage chunk based on the query. Focus only on content relevant to the query. If
1. The query is vague
2. Does not make sense for the page
3. Some/all of the information is not available

Explain the content of the chunk and that the requested information is not available in the chunk. Respond in JSON format.\nQuery: {query}\n Website chunk:\n{chunk}"""
                
                # Use provided model or fallback
                extraction_model = self.llm_model or page_extraction_llm
                if not extraction_model:
                    return ActionResult(success=False, error="No LLM model available for extraction")
                
                # Create user messages for each chunk
                user_messages = []
                for chunk in chunks:
                    formatted_prompt = chunk_prompt.format(query=query, chunk=chunk)
                    user_messages.append([UserMessage(content=formatted_prompt)])
                
                # Process all chunks in parallel using asyncio.gather
                tasks = [extraction_model.ainvoke(message) for message in user_messages]
                parallel_results = await asyncio.gather(*tasks)
                parallel_time = time.time() - parallel_start
                
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
                
                for i, result in enumerate(parallel_results):
                    try:
                        chunk_json = json.loads(result.completion)
                        merged_json = merge_json(merged_json, chunk_json)
                    except json.JSONDecodeError:
                        text_results.append(result.completion)
                
                # Format the final extracted content
                extracted_content = f'Page Link: {page.url}\nQuery: {query}\nExtracted Content:\n'
                
                if merged_json:
                    extracted_content += json.dumps(merged_json, indent=2)
                
                if text_results:
                    extracted_content += "\n\nAdditional extracted content:\n" + "\n".join(text_results)
                
                # Determine memory handling based on content size
                MAX_MEMORY_SIZE = 600
                if len(extracted_content) < MAX_MEMORY_SIZE:
                    memory = extracted_content
                    include_extracted_content_only_once = False
                else:
                    # Find lines until MAX_MEMORY_SIZE
                    lines = extracted_content.splitlines()
                    display = ''
                    display_lines_count = 0
                    for line in lines:
                        if len(display) + len(line) < MAX_MEMORY_SIZE:
                            display += line + '\n'
                            display_lines_count += 1
                        else:
                            break
                    memory = f'Extracted content from {page.url}\n<query>{query}\n</query>\n<extracted_content>\n{display}{len(lines) - display_lines_count} more lines...\n</extracted_content>'
                    include_extracted_content_only_once = True
                
                # Log timing info
                overall_time = time.time() - overall_start
                logger.info(f'📄 Extracted structured content in {overall_time:.2f}s (content: {content_time:.2f}s, chunking: {chunk_time:.2f}s, parallel processing: {parallel_time:.2f}s)')
                
                return ActionResult(
                    extracted_content=extracted_content,
                    include_extracted_content_only_once=include_extracted_content_only_once,
                    long_term_memory=memory,
                )
                
            except Exception as e:
                logger.error(f'Error extracting structured content: {e}')
                return ActionResult(
                    success=False,
                    error=f"Failed to extract structured content: {str(e)}"
                )

        @self.controller.action("Get result(s) of subtask(s)")
        async def get_subtask_results(comma_separated_list_of_task_ids: str) -> ActionResult:
            """Get the results of multiple subtasks in parallel.
            
            Args:
                comma_separated_list_of_task_ids: Comma-separated list of task IDs
                
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
                # Wait for all results in parallel with error handling
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine results
                combined_results = []
                all_successful = True
                failed_tasks = []
                successful_tasks = []
                
                for task_id, result in zip(task_ids, results):
                    if isinstance(result, Exception):
                        combined_results.append(f"  Subtask {task_id}: Failed - {str(result)}")
                        failed_tasks.append(task_id)
                        all_successful = False
                    elif result:
                        result_text = result.final_result()
                        combined_results.append(f"  Subtask {task_id}: {result_text}")
                        successful_tasks.append(task_id)
                    else:
                        combined_results.append(f"  Subtask {task_id}: No result available")
                        failed_tasks.append(task_id)
                        all_successful = False

                if not all_successful:
                    # Log summary to help debug truncation issues
                    summary = f"Completed: {len(successful_tasks)}/{len(task_ids)} tasks. "
                    if successful_tasks:
                        summary += f"Success: {','.join(successful_tasks)}. "
                    if failed_tasks:
                        summary += f"Failed/Incomplete: {','.join(failed_tasks)}"
                    
                    return ActionResult(
                        success=False,
                        error=f"{summary}\n\n" + "\n".join(combined_results)
                    )
                
                return ActionResult(
                    is_done=True,
                    success=True,
                    extracted_content="📋 Subtask results:\n" + "\n".join(combined_results)
                )
            except Exception as e:
                return ActionResult(
                    success=False,
                    error=f"Failed to get subtask results: {str(e)}"
                )

    def _register_human_tools(self):
        """Register human-in-loop tools."""
        
        @self.controller.action("Ask for human assistance with CAPTCHA, 2FA, credentials or other input (✓ enabled)")
        async def ask_human(prompt: str, allow_takeover: bool = False, browser_session: Optional[BrowserSession] = None) -> ActionResult:
            """Ask for human assistance with a task.
            
            Args:
                prompt: Question or request for the human
                allow_takeover: Whether to allow human to take control of browser
                browser_session: Optional browser session for getting live URL
                
            Returns:
                Human's response
            """
            try:
                # Get live URL if browser session available
                live_url = None
                if browser_session:
                    page = await browser_session.get_current_page()
                    if page:
                        live_url = page.url

                # Send request to human
                request = HumanRequest(
                    task_id=self.task_id,
                    prompt=prompt,
                    allow_takeover=allow_takeover,
                    live_url=live_url
                )
                await self.human_request_queue.put(request)

                # Wait for response  
                response = await self.human_response_queue.get()
                if not isinstance(response, HumanResponse) or response.task_id != self.task_id:
                    return ActionResult(
                        success=False,
                        error="Invalid response received from human"
                    )

                return ActionResult(
                    is_done=True,
                    success=True,
                    extracted_content=f"Human responded: {response.response}",
                    include_in_memory=True
                )

            except Exception as e:
                return ActionResult(
                    success=False,
                    error=f"Failed to get human assistance: {str(e)}"
                )