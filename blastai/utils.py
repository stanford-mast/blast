"""Utility functions for BlastAI."""

import os
import sys
from pathlib import Path

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