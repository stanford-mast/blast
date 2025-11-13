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
    
    async def _summarize_thought(self, raw_thought: str) -> str:
        """Summarize raw thought using LLM."""
        if not raw_thought or not raw_thought.strip():
            return ""
        
        try:
            from browser_use.llm.messages import UserMessage
            
            prompt = f"""Rewrite this update as if you're my colleague talking out loud to provide a real-time update on a computer task they're performing, coming across as thoughtful, easygoing, professional, reliable, concise (1 short sentence).

Update: {raw_thought}

Rewritten:"""
            
            messages = [UserMessage(content=prompt)]
            response = await self.agent_executor.summarizer_llm.ainvoke(messages)
            return response.completion.strip()
        except Exception as e:
            logger.warning(f"Failed to summarize thought: {e}")
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
            """Increment step count and log progress."""
            try:
                self.agent_executor.step_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                logger.info(f"Step {self.agent_executor.step_count} starting (elapsed: {elapsed_time:.1f}s)")
                    
            except Exception as e:
                logger.error(f"Error in step start hook: {e}")
        
        async def on_step_end(agent):
            """Send AgentThought message with summarized thought after each step."""
            try:
                # Check if stop was requested during step execution
                if self.agent_executor.check_stop_callback and self.agent_executor.check_stop_callback():
                    logger.info(f"Skipping step_end (StopRequest detected)")
                    return
                
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                logger.info(f"Step {self.agent_executor.step_count} completed (elapsed: {elapsed_time:.1f}s)")
                
                # Get raw thought from agent
                raw_thought = get_thought(agent)
                
                # Summarize thought with LLM
                thought_text = await self._summarize_thought(raw_thought)
                
                # Get current browser state
                current_url = None
                page_title = None
                try:
                    current_url = await self.agent_executor.browser.get_current_page_url()
                    page = await self.agent_executor.browser.get_current_page()
                    if page:
                        page_title = await page.title()
                except Exception as e:
                    logger.debug(f"Failed to get browser state: {e}")
                
                # Send AgentThought message
                if self.agent_executor.send_message_callback:
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
                    
            except Exception as e:
                logger.error(f"Error in step end hook: {e}")
        
        return on_step_start, on_step_end
    
    async def send_agent_stopped(self, reason: str = "completed"):
        """Send AgentStopped message."""
        if self.agent_executor.send_message_callback:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
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
    
    async def send_response_to_human(self, result: Any = None):
        """Send ResponseToHuman message (successful completion)."""
        if self.agent_executor.send_message_callback:
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
                # Format args/kwargs for display
                arg_strs = [repr(a) for a in args[:3]]  # First 3 args
                kwarg_strs = [f"{k}={repr(v)[:50]}" for k, v in list(kwargs.items())[:3]]  # First 3 kwargs
                all_args = arg_strs + kwarg_strs
                args_display = ", ".join(all_args)
                if len(args) > 3 or len(kwargs) > 3:
                    args_display += ", ..."
                
                raw_thought = f"Calling {func_name}({args_display})"
                
                # Summarize and send AgentThought BEFORE execution
                thought_text = await self._summarize_thought(raw_thought)
                
                if self.agent_executor.send_message_callback:
                    # Get current URL
                    current_url = None
                    try:
                        current_url = await self.agent_executor.browser.get_current_page_url()
                    except Exception:
                        pass
                    
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
