"""Factory for creating BLAST resources."""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

from browser_use import Agent
from browser_use.browser import BrowserSession
from patchright.async_api import async_playwright as async_patchright

# Import moved to line 27

from .config import Settings, Constraints
from .tools import Tools
from .utils import find_local_browser, init_model
from .resource_factory_utils import (
    get_stealth_profile_dir,
    cleanup_stealth_profile_dir,
    launch_vnc_session
)

logger = logging.getLogger(__name__)

# Import but don't apply patches yet
from .browser_session_patch import apply_all_patches
_patches_applied = False

async def create_executor(
    task_id: str,
    constraints: Constraints,
    settings: Settings,
    scheduler,
    resource_manager,
    engine_hash: str = None,
    sensitive_data: Optional[Dict[str, str]] = None
) -> Optional['Executor']:
    """Create a new executor with browser session.
    
    Args:
        task_id: Task ID
        constraints: Constraints configuration
        settings: Settings configuration
        scheduler: Scheduler instance
        resource_manager: ResourceManager instance
        engine_hash: Optional engine hash
        sensitive_data: Optional sensitive data dict
        
    Returns:
        New Executor instance or None if creation fails
    """
    from .executor import Executor  # Import here to avoid circular import
    
    # Apply patches if not already applied
    global _patches_applied
    if not _patches_applied:
        apply_all_patches()
        _patches_applied = True
    
    try:
        # Create Tools instance first since we need it for both paths
        llm = init_model(constraints.llm_model)
        llm_mini = init_model(constraints.llm_model_mini) if constraints.llm_model_mini else llm
        tools = Tools(
            scheduler=scheduler,
            task_id=task_id,
            resource_manager=resource_manager,
            llm_model=llm_mini
        )

        # If human-in-loop is required, use VNC session
        if constraints.require_human_in_loop:
            vnc_session = None
            try:
                # Launch VNC session with appropriate configuration
                target_url = "about:blank"  # Initial blank page
                vnc_session = await launch_vnc_session(
                    target_url=target_url,
                    stealth=constraints.require_patchright
                )
                
                # Get browser session and live URL
                browser_session = await vnc_session.get_browser_session()
                live_url = vnc_session.get_novnc_url()
                logger.info(f"Started VNC session with live URL: {live_url}")
                
                # Create executor with VNC session
                executor = Executor(
                    browser_session=browser_session,
                    controller=tools.controller,
                    llm=llm,
                    constraints=constraints,
                    task_id=task_id,
                    settings=settings,
                    engine_hash=engine_hash,
                    scheduler=scheduler,
                    sensitive_data=sensitive_data,
                    vnc_session=vnc_session,
                    live_url=live_url,
                    user_data_dir=vnc_session.stealth_dir if constraints.require_patchright else None
                )
                
                # Success - don't cleanup resources
                vnc_session = None
                return executor
                
            except Exception as e:
                logger.error(f"Failed to create VNC executor: {e}")
                # Clean up VNC session if it exists
                if vnc_session:
                    try:
                        await vnc_session.cleanup()
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up VNC session: {cleanup_error}")
                return None

        # Otherwise use regular browser session
        # Configure regular browser session
        browser_args = {
            'headless': constraints.require_headless,
            'user_data_dir': None,  # Use ephemeral profile for security
            'keep_alive': False,  # Set to False so agent.close() can clean up
            'highlight_elements': False,  # Disable element highlighting
        }
        
        # Add allowed domains if configured
        if constraints.allowed_domains is not None:
            browser_args['allowed_domains'] = constraints.allowed_domains

        # Handle local browser path if not "none"
        if settings.local_browser_path != "none":
            if settings.local_browser_path == "auto":
                browser_path = find_local_browser()
                if browser_path:
                    browser_args['executable_path'] = browser_path
                    logger.debug(f"Using auto-detected browser at: {browser_path}")
            else:
                browser_path = settings.local_browser_path
                if not os.path.exists(browser_path):
                    logger.error(f"Specified local browser path does not exist: {browser_path}")
                    return None
                browser_args['executable_path'] = browser_path

        # Initialize patchright if required
        if constraints.require_patchright:
            playwright = await async_patchright().start()
            browser_args['playwright'] = playwright
            
            # Get stealth profile directory path
            stealth_dir = get_stealth_profile_dir(task_id)
            browser_args['user_data_dir'] = stealth_dir
            browser_args['disable_security'] = False
            browser_args['deterministic_rendering'] = False
            logger.debug(f"Using patchright for browser automation with profile: {stealth_dir}")

        # Create and start browser session
        browser_session = BrowserSession(**browser_args)
        await browser_session.start()
        
        # Create and return regular executor
        logger.debug(f"Created new executor for task {task_id}")
        return Executor(
            browser_session=browser_session,
            controller=tools.controller,
            llm=llm,
            constraints=constraints,
            task_id=task_id,
            settings=settings,
            engine_hash=engine_hash,
            scheduler=scheduler,
            sensitive_data=sensitive_data,
            user_data_dir=browser_args.get('user_data_dir')
        )
        
    except Exception as e:
        logger.error(f"Failed to create executor: {e}")
        # Clean up resources on failure (non-VNC path only)
        if constraints.require_patchright and not constraints.require_human_in_loop:
            try:
                cleanup_stealth_profile_dir(stealth_dir)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up stealth profile: {cleanup_error}")
        return None