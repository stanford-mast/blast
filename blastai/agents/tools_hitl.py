"""
Human-in-the-loop tools for CLI context.

This module provides ask_human_cli which is a CoreTool that blocks
on stdin to get human input. Unlike the server-based ask_human tool
which uses queues, this is designed for CLI usage.
"""

import asyncio
from typing import Any, Dict


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
