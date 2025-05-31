"""Factory for creating BLAST resources."""

import logging
import os
from pathlib import Path
from typing import Optional, Dict

from browser_use import Agent
from browser_use.browser import BrowserSession
from patchright.async_api import async_playwright as async_patchright
from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings, Constraints
from .tools import Tools
from .utils import find_local_browser, init_model
from .resource_factory_utils import get_stealth_profile_dir, cleanup_stealth_profile_dir

logger = logging.getLogger(__name__)

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
    
    # Configure browser session
    browser_args = {
        'headless': constraints.require_headless,
        'user_data_dir': None,  # Use ephemeral profile for security
        'keep_alive': True,  # Keep browser alive between tasks
    }
    
    # Add allowed domains if configured
    if constraints.allowed_domains is not None:
        browser_args['allowed_domains'] = constraints.allowed_domains

    # Handle local browser path if not "none"
    if settings.local_browser_path != "none":
        if settings.local_browser_path == "auto":
            # Auto-detect browser path
            browser_path = find_local_browser()
            if browser_path:
                browser_args['executable_path'] = browser_path
                logger.debug(f"Using auto-detected browser at: {browser_path}")
        else:
            # Use specified path directly
            browser_path = settings.local_browser_path
            if not os.path.exists(browser_path):
                logger.error(f"Specified local browser path does not exist: {browser_path}")
                return None
            browser_args['executable_path'] = browser_path

    try:
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

        # Create browser session
        browser_session = BrowserSession(**browser_args)
        await browser_session.start()
        
        # Create LLMs
        llm = init_model(constraints.llm_model)
        llm_mini = init_model(constraints.llm_model_mini) if constraints.llm_model_mini else llm
        
        # Create Tools instance
        tools = Tools(
            scheduler=scheduler,
            task_id=task_id,
            resource_manager=resource_manager,
            llm_model=llm_mini
        )
        
        # Create and return executor
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
            user_data_dir=browser_args.get('user_data_dir')  # Pass user_data_dir to executor
        )
        
    except Exception as e:
        logger.error(f"Failed to create executor: {e}")
        # Clean up profile directory on failure
        if constraints.require_patchright:
            cleanup_stealth_profile_dir(stealth_dir)
        return None