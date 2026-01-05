"""
Human-in-the-loop tools for CLI and workflow contexts.

This module provides:
- ask_human_cli: CLI tool that blocks on stdin  
- create_ask_human_tool: Factory for creating DBOS workflow-compatible ask_human tool
"""

import asyncio
from typing import Any, Optional, Callable


async def ask_human_cli(prompt: str) -> str:
    """
    Ask for human assistance via CLI stdin.
    
    This is a CoreTool designed for CLI context. It blocks on stdin
    to get human input, unlike the server-based ask_human which uses queues.
    
    Args:
        prompt: Question or request for the human
        
    Returns:
        Human's response as a string
        
    Example:
        response = await ask_human_cli("What is your password?")
    """
    print(f"\n{'='*60}")
    print(f"🤖 AGENT NEEDS HELP")
    print(f"{'='*60}")
    print(f"\n{prompt}\n")
    print(f"{'='*60}")
    print(f"Your response: ", end="", flush=True)

    # Run input() in executor to avoid blocking event loop
    loop = asyncio.get_event_loop()

    def _read_input() -> str:
        try:
            return input()
        except EOFError as exc:
            raise RuntimeError("stdin is not available for ask_human_cli") from exc

    response = await loop.run_in_executor(None, _read_input)
    
    print(f"{'='*60}\n")
    
    return response.strip()


def create_ask_human_tool(
    ask_human_callback: Callable[[str], Any]
):
    """
    Create ask_human tool for DBOS workflows using callback.
    
    CRITICAL: This function creates an ask_human tool that uses a callback
    created in the workflow context. The callback has full DBOS.recv_async access.
    
    When generated code calls ask_human("question"):
    1. ask_human calls the callback with the question
    2. Callback (defined in agent_workflow) handles all DBOS send/recv
    3. Response is returned to generated code
    
    Args:
        ask_human_callback: Async callback to handle ask_human requests
            Signature: async def callback(question: str) -> str
            The callback has cycle_id and user_email in its closure from agent_workflow.
        
    Returns:
        Async function that can be used as a tool
        
    Example:
        ask_human = create_ask_human_tool(ask_human_callback)
        response = await ask_human("What is the password?")
    """
    
    async def ask_human(question: str) -> str:
        """
        Ask human for help, delegating to workflow-context callback.
        
        The callback handles:
        1. Sending RequestForHuman via DBOS.send_async
        2. Waiting for ResponseToAgent via DBOS.recv_async
        3. Handling StopRequest
        
        Args:
            question: Question or request for the human
            
        Returns:
            Human's response string
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Agent asking human: {question}")
        
        try:
            # Call the workflow-context callback
            # It has DBOS access and can handle recv/send
            # The callback has cycle_id and user_email in its closure
            response = await ask_human_callback(question)
            logger.info(f"Received human response: {response}")
            return response
        except Exception as e:
            logger.error(f"ask_human failed: {e}")
            return f"Error: {e}"
    
    return ask_human


__all__ = ["ask_human_cli", "create_ask_human_tool"]
