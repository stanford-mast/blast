"""Utility functions for BlastAI."""

import os
import sys
from pathlib import Path
import json
from typing import Dict, Optional

def estimate_llm_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: Optional[int] = 0
) -> float:
    """Estimate LLM cost based on token counts and model pricing.
    
    Args:
        model_name: Name of the LLM model
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens (default: 0)
        
    Returns:
        Estimated cost in USD
    """
    # Load pricing config
    pricing_path = os.path.join(os.path.dirname(__file__), 'pricing_openai_api.json')
    with open(pricing_path) as f:
        pricing_config = json.load(f)
        
    if model_name not in pricing_config["models"]:
        return 0.0
        
    pricing = pricing_config["models"][model_name]
    
    # Calculate costs
    input_cost = (prompt_tokens - cached_tokens) * pricing["input"] / 1_000_000
    cached_cost = cached_tokens * pricing.get("cachedInput", pricing["input"]) / 1_000_000
    output_cost = completion_tokens * pricing["output"] / 1_000_000
    
    return input_cost + cached_cost + output_cost

def get_appdata_dir() -> Path:
    """Get the appropriate app data directory for the current platform.
    
    Returns:
        Path to the BlastAI app data directory
    """
    if sys.platform == "win32":
        # Windows: %LOCALAPPDATA%\blastai
        base_dir = os.environ.get("LOCALAPPDATA")
        if not base_dir:
            base_dir = os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":
        # macOS: ~/Library/Application Support/blastai
        base_dir = os.path.expanduser("~/Library/Application Support")
    else:
        # Linux/Unix: ~/.local/share/blastai
        base_dir = os.path.expanduser("~/.local/share")
        
    app_dir = Path(base_dir) / "blastai"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir