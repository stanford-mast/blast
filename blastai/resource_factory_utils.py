"""Utilities for resource factory operations."""

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

def get_stealth_profile_dir(task_id: str) -> str:
    """Get a unique stealth profile directory path for a task.
    
    Args:
        task_id: Task ID to create profile for
        
    Returns:
        Path to stealth profile directory (will be created if doesn't exist)
    """
    # Get base and task-specific paths
    base_stealth_dir = os.path.expanduser('~/.config/browseruse/profiles/stealth')
    stealth_dir = os.path.expanduser(f'~/.config/browseruse/profiles/stealth_{task_id}')
    
    base_stealth_path = Path(base_stealth_dir)
    stealth_dir_path = Path(stealth_dir)
    
    # Ensure base directory exists
    if not base_stealth_path.exists():
        return base_stealth_dir
    
    # Create task-specific directory
    if stealth_dir_path.exists():
        shutil.rmtree(stealth_dir)
    if base_stealth_path.exists():
        shutil.copytree(base_stealth_dir, stealth_dir)
    else:
        stealth_dir_path.mkdir(parents=True, exist_ok=True)
    
    return stealth_dir

def cleanup_stealth_profile_dir(profile_dir: str) -> None:
    """Safely clean up a stealth profile directory.
    
    Args:
        profile_dir: Path to profile directory to clean up
    """
    try:
        if not profile_dir or 'stealth_' not in profile_dir:
            return
            
        profile_path = Path(os.path.expanduser(profile_dir))
        if profile_path.exists():
            shutil.rmtree(profile_path)
            logger.debug(f"Cleaned up stealth profile: {profile_path}")
    except Exception as e:
        logger.error(f"Error cleaning up stealth profile: {e}")