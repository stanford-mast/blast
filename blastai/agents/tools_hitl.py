"""
Human-in-the-loop tools for CLI and workflow contexts.

This module provides:
- ask_human_cli: CLI tool that blocks on stdin  
- create_ask_human_tool: Factory for creating DBOS workflow-compatible ask_human tool
"""

import asyncio
from typing import Any, Dict, Optional, Callable


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
    print("Your response: ", end="", flush=True)
    
    # Run input() in executor to avoid blocking event loop
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, input)
    
    print(f"{'='*60}\n")
    
    return response.strip()


def create_ask_human_tool(
    send_message_callback: Callable[[str, Dict[str, Any]], Any],
    session_id: str,
    user_email: Optional[str] = None,
    cycle_id: Optional[int] = None
):
    """
    Create ask_human tool for DBOS workflows.
    
    Works in both loop mode (browser-use action) and code mode (Python function).
    Sends RequestForHuman message, then blocks waiting for ResponseToAgent or StopRequest.
    
    Args:
        send_message_callback: Async callback to send messages
            Signature: async def callback(session_id: str, message: Dict[str, Any])
        session_id: Session ID for message routing
        user_email: Optional user email for notifications
        cycle_id: Optional cycle ID to include in messages
        
    Returns:
        Async function that can be used as a tool
        
    Example:
        ask_human = create_ask_human_tool(send_callback, "session-123", cycle_id=1)
        response = await ask_human("What is the password?")
    """
    
    async def ask_human(question: str) -> str:
        """
        Ask human for help, wait for ResponseToAgent or StopRequest.
        
        Blocks execution until human responds (7-day timeout).
        """
        import asyncio
        import time
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"Agent asking human: {question}")
        
        # Get the current event loop
        loop = asyncio.get_event_loop()
        
        # Send RequestForHuman message
        loop.run_until_complete(send_message_callback(session_id, {
            "message": {
                "messageType": "RequestForHuman",
                "cycleId": cycle_id,
                "payload": {
                    "question": question,
                    "timestamp": str(int(time.time()))
                },
                "messageId": f"human-request-{session_id}-{int(time.time())}"
            }
        }))
        
        # TODO: Send email notification if user_email is available
        # Would require notification service integration
        
        logger.info(f"⏸️  Waiting for human response (7-day timeout)...")
        
        # Wait for ResponseToAgent or StopRequest messages
        while True:
            try:
                from dbos import DBOS
            except ImportError:
                logger.error("DBOS not available - ask_human only works in DBOS workflows")
                return "Error: DBOS not available"
            
            # Use recv_async to wait for response
            response_msg = loop.run_until_complete(DBOS.recv_async(
                topic=f"AgentResponses:{session_id}",
                timeout_seconds=7 * 24 * 60 * 60  # 7 days
            ))
            
            if not response_msg:
                logger.warning("Received null message from DBOS.recv_async")
                return "No response received from human (null message)"
            
            if isinstance(response_msg, dict):
                message = response_msg.get("message", {})
                message_type = message.get("messageType")
                
                if message_type == "ResponseToAgent":
                    answer = message.get("payload", {}).get("answer", "No answer provided")
                    logger.info(f"▶️  Received human response: {answer}")
                    return answer
                
                elif message_type == "StopRequest":
                    logger.info("🛑 Received StopRequest in ask_human")
                    # Define StopExecutionError if needed
                    class StopExecutionError(Exception):
                        pass
                    raise StopExecutionError("Agent stopped by user request during ask_human")
                
                else:
                    logger.info(f"Ignoring message type: {message_type}, continuing to wait")
                    # Continue waiting for the right message type
    
    return ask_human

