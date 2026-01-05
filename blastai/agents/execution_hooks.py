"""
Execution hooks for both loop mode and code mode.

Provides:
- Loop mode hooks (on_step_start, on_step_end) for browser-use Agent
- Code mode decorator for wrapping tool execution with hooks
- Shared logic for stop checking, state capture, and message sending
"""

import time
import logging
import random
from typing import Callable, Any, Dict, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class StopExecutionError(Exception):
    """Raised when execution should stop (code mode)."""
    pass


class ExecutionHooks:
    """
    Manages execution hooks for AgentExecutor.
    
    Provides unified hook system for both loop mode (browser-use Agent)
    and code mode (LocalPythonExecutor with wrapped tools).
    """
    
    def __init__(
        self,
        agent_executor,
        session_id: str,
        cycle_id: Optional[int] = None,
        timeout_seconds: int = 10 * 60
    ):
        """
        Initialize execution hooks.
        
        Args:
            agent_executor: AgentExecutor instance with callbacks and browser
            session_id: Session ID for message routing
            cycle_id: Optional cycle ID to include in messages
            timeout_seconds: Timeout in seconds (default 10 minutes)
        """
        self.agent_executor = agent_executor
        self.session_id = session_id
        self.cycle_id = cycle_id
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()
        self.terminal_message_sent = False
    
    async def _summarize_thought(self, raw_thought: str) -> str:
        """Summarize raw thought using LLM.
        
        Falls back to raw thought if summarization fails (e.g., auth errors).
        This keeps the system resilient - thought summarization is optional.
        """
        if not raw_thought or not raw_thought.strip():
            return ""
        
        # If raw thought is already short, don't bother summarizing
        if len(raw_thought) <= 80:
            return raw_thought
        
        try:
            from browser_use.llm.messages import UserMessage, SystemMessage
            
            # Check that summarizer_llm exists before attempting to use it
            if not self.agent_executor.summarizer_llm:
                logger.debug("No summarizer_llm available, returning raw thought")
                return raw_thought[:100]
            
            prompt = f"""Rewrite this update as if you're my colleague talking out loud to provide a real-time update on a computer task you're performing, coming across as thoughtful, easygoing, professional, reliable, concise (1 short sentence).

Do NOT say anything technical - describe what you are trying to accomplish instead.

Update: {raw_thought}

Rewritten:"""
            
            messages = [SystemMessage(content="You are a helpful assistant. Do NOT generate code or reference names of tools."), UserMessage(content=prompt)]
            
            # Use timeout to prevent hanging
            import asyncio
            response = await asyncio.wait_for(
                self.agent_executor.summarizer_llm.ainvoke(messages),
                timeout=3.0  # 3 second timeout for summarization
            )
            
            # Remove surrounding quotation marks from the response
            result = response.completion.strip()
            # Strip leading/trailing quotes if present
            if (result.startswith('"') and result.endswith('"')) or (result.startswith("'") and result.endswith("'")):
                result = result[1:-1]
            
            # Ensure we got a reasonable response
            if result and len(result) > 10:
                return result
            else:
                # Response was empty or too short, use raw thought
                return raw_thought[:100]
                
        except asyncio.TimeoutError:
            logger.debug("Thought summarization timed out (3s), returning raw thought")
            return raw_thought[:100]
        except Exception as e:
            # Don't log as warning - summarization failures are expected/tolerable
            logger.debug(f"Skipping thought summarization (not critical): {type(e).__name__}")
            return raw_thought[:100]  # Fallback to truncated raw thought
    
    def create_loop_hooks(self, get_thought: Callable[[Any], str]):
        """
        Create on_step_start and on_step_end hooks for loop mode.
        
        Args:
            get_thought: Callable that extracts current thought string from agent
            
        Returns:
            Tuple of (on_step_start, on_step_end) hook functions
        """
        
        async def on_step_start(agent):
            """Check for stop request BEFORE executing next step, then increment step count."""
            try:
                # CRITICAL: Check stop flag FIRST, before executing the next step
                # This ensures the agent stops immediately when StopRequest is received
                if self.agent_executor.check_stop_callback and self.agent_executor.check_stop_callback():
                    logger.info("🛑 StopRequest detected at step start, raising InterruptedError to halt agent")
                    raise InterruptedError("Agent stopped by user request (StopRequest received)")
                
                self.agent_executor.step_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                logger.info(f"Step {self.agent_executor.step_count} starting (elapsed: {elapsed_time:.1f}s)")
                    
            except InterruptedError:
                # Re-raise InterruptedError to stop the agent loop
                raise
            except Exception as e:
                logger.error(f"Error in step start hook: {e}")
        
        async def on_step_end(agent):
            """Send AgentThought message with summarized thought after each step."""
            try:
                # Check if stop was requested during step execution
                if self.agent_executor.check_stop_callback and self.agent_executor.check_stop_callback():
                    logger.info(f"Skipping step_end (StopRequest detected)")
                    return
                
                # Skip if terminal message already sent
                if self.terminal_message_sent:
                    logger.debug("Skipping step_end (terminal message already sent)")
                    return
                
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                logger.info(f"Step {self.agent_executor.step_count} completed (elapsed: {elapsed_time:.1f}s)")
                
                # Get raw thought from agent
                raw_thought = get_thought(agent)
                
                # Summarize thought with LLM (await directly, no fire-and-forget)
                thought_text = await self._summarize_thought(raw_thought)
                
                # Skip sending if thought is too generic or raw
                if not thought_text or thought_text.startswith("Calling "):
                    logger.debug(f"Skipping step_end thought send (no meaningful content)")
                    return
                
                # Get current browser state
                current_url = None
                page_title = None
                try:
                    current_url = await self.agent_executor.browser.get_current_page_url()
                    page = await self.agent_executor.browser.get_current_page()
                    if page:
                        # Use browser-use API: page.get_title() is the correct async method
                        try:
                            page_title = await page.get_title()
                        except Exception as title_error:
                            logger.debug(f"Could not get page title: {title_error}")
                            page_title = None
                except Exception as e:
                    logger.debug(f"Failed to get browser state: {e}")
                
                # Send AgentThought message (await directly to maintain workflow context)
                if self.agent_executor.send_message_callback:
                    try:
                        await self.agent_executor.send_message_callback(self.session_id, {
                            "message": {
                                "messageType": "AgentThought",
                                "cycleId": self.cycle_id,
                                "payload": {
                                    "step": self.agent_executor.step_count,
                                    "elapsed_seconds": elapsed_time,
                                    "current_url": current_url,
                                    "page_title": page_title,
                                    "thought": thought_text
                                },
                                "messageId": f"step-end-{self.session_id}-{self.agent_executor.step_count}-{int(current_time)}"
                            }
                        })
                    except Exception as send_error:
                        logger.debug(f"Failed to send AgentThought for step {self.agent_executor.step_count} (non-critical): {send_error}")
                    
            except Exception as e:
                logger.error(f"Error in step end hook: {e}")
        
        return on_step_start, on_step_end
    
    async def send_agent_stopped(self, reason: str = "completed"):
        """Send AgentStopped message."""
        if self.agent_executor.send_message_callback:
            self.terminal_message_sent = True
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            try:
                await self.agent_executor.send_message_callback(self.session_id, {
                    "message": {
                        "messageType": "AgentStopped",
                        "cycleId": self.cycle_id,
                        "payload": {
                            "reason": reason,
                            "step_count": self.agent_executor.step_count,
                            "elapsed_seconds": elapsed_time
                        },
                        "messageId": f"stopped-{self.session_id}-{int(current_time)}"
                    }
                })
            except Exception as send_error:
                logger.warning(f"Failed to send AgentStopped message: {send_error}")
    
    async def send_response_to_human(self, result: Any = None):
        """Send ResponseToHuman message (successful completion)."""
        if self.agent_executor.send_message_callback:
            self.terminal_message_sent = True
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Extract final result text from browser-use AgentHistoryList
            result_text = "Task completed"
            if result is not None:
                try:
                    # Check if it's an AgentHistoryList from browser-use
                    if hasattr(result, 'action_results'):
                        action_results = result.action_results()
                        # Find the last done action
                        for action in reversed(action_results):
                            if hasattr(action, 'is_done') and action.is_done:
                                # Use extracted_content if available, otherwise fall back
                                if hasattr(action, 'extracted_content') and action.extracted_content:
                                    result_text = action.extracted_content
                                    break
                    else:
                        # Fall back to string representation for non-browser-use results
                        result_text = str(result)
                except Exception as e:
                    logger.warning(f"Failed to extract final result text: {e}, using fallback")
                    result_text = "Task completed"
            
            try:
                await self.agent_executor.send_message_callback(self.session_id, {
                    "message": {
                        "messageType": "ResponseToHuman",
                        "cycleId": self.cycle_id,
                        "payload": {
                            "text": result_text,
                            "executionTime": elapsed_time,
                            "stepCount": self.agent_executor.step_count
                        },
                        "messageId": f"response-{self.session_id}-{int(current_time)}"
                    }
                })
            except Exception as send_error:
                logger.warning(f"Failed to send ResponseToHuman message: {send_error}")
    
    def create_code_mode_decorator(self):
        """
        Create decorator for code mode tool execution.
        
        Wraps tool functions to:
        1. Check stop flag before execution
        2. Send AgentThought before tool call
        3. Execute tool
        4. Send AgentThought after tool call (with result)
        
        Returns:
            Decorator function
        """
        
        def tool_execution_hook(func):
            """Decorator that wraps tool execution with hooks."""
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Skip if terminal message already sent
                if self.terminal_message_sent:
                    logger.debug(f"Skipping {func.__name__} thought (terminal message already sent)")
                    # Still execute the function, just don't send thought
                    return await func(*args, **kwargs)
                
                # Increment step count
                self.agent_executor.step_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                # Check for stop BEFORE execution
                if self.agent_executor.check_stop_callback and self.agent_executor.check_stop_callback():
                    logger.info(f"Stop requested before {func.__name__}")
                    raise StopExecutionError(f"Stopped before {func.__name__}")
                
                # Check timeout
                if elapsed_time > self.timeout_seconds:
                    logger.warning(f"Timeout before {func.__name__}")
                    raise StopExecutionError(f"Timeout before {func.__name__}")
                
                # Build thought for function call
                func_name = func.__name__
                
                # Skip sending AgentThought for non-consequential utility functions
                # These are frequently called helpers that don't represent meaningful actions
                non_consequential_funcs = {'get_url', 'find_and_call_observe'}
                should_send_thought = func_name not in non_consequential_funcs
                
                # Format args/kwargs for display
                arg_strs = [repr(a) for a in args[:3]]  # First 3 args
                kwarg_strs = [f"{k}={repr(v)[:50]}" for k, v in list(kwargs.items())[:3]]  # First 3 kwargs
                all_args = arg_strs + kwarg_strs
                args_display = ", ".join(all_args)
                if len(args) > 3 or len(kwargs) > 3:
                    args_display += ", ..."
                
                raw_thought = f"Calling {func.__name__}({args_display})"
                
                # Only summarize and send thought for consequential functions
                if should_send_thought:
                    # Summarize the thought
                    thought_text = await self._summarize_thought(raw_thought)
                    
                    # Skip sending if thought is just the raw function call signature
                    # (means summarization failed and we only have raw thought)
                    if thought_text and thought_text.startswith("Calling "):
                        logger.debug(f"Skipping thought send for {func_name} (summarization failed, raw thought only)")
                    elif thought_text:  # Only send if we have a meaningful thought
                        if self.agent_executor.send_message_callback:
                            # Get current URL
                            current_url = None
                            try:
                                current_url = await self.agent_executor.browser.get_current_page_url()
                            except Exception:
                                pass
                            
                            # Send message directly (await to maintain DBOS workflow context)
                            try:
                                await self.agent_executor.send_message_callback(self.session_id, {
                                    "message": {
                                        "messageType": "AgentThought",
                                        "cycleId": self.cycle_id,
                                        "payload": {
                                            "step": self.agent_executor.step_count,
                                            "elapsed_seconds": elapsed_time,
                                            "current_url": current_url,
                                            "thought": thought_text
                                        },
                                        "messageId": f"code-tool-{self.session_id}-{self.agent_executor.step_count}-{int(current_time)}"
                                    }
                                })
                            except Exception as send_error:
                                logger.debug(f"Failed to send tool AgentThought (non-critical): {send_error}")
                
                # Execute the actual function
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Log error and re-raise
                    logger.error(f"Error in {func_name}: {e}")
                    raise
            
            return wrapper
        
        return tool_execution_hook


__all__ = ["ExecutionHooks", "StopExecutionError"]
