"""Logging configuration for BLAST."""

import logging
import os
import sys
import warnings
from typing import Optional

from .config import Settings

def setup_logging(settings: Optional[Settings] = None):
    """Configure logging for BLAST.
    
    This function:
    1. Configures root logger and all third-party loggers to ERROR by default
    2. Sets up blastai logger based on config
    3. Sets up browser-use logger via environment variable
    4. Silences deprecation warnings
    
    Args:
        settings: Optional Settings instance with logging configuration
    """
    # Use default settings if none provided
    if not settings:
        settings = Settings()
        
    # Silence deprecation warnings (e.g. from uvicorn)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Configure root logger to error by default
    logging.basicConfig(
        level=logging.ERROR,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Configure all loggers to error level
    logging.getLogger().setLevel(logging.ERROR)
    
    # Configure blastai logger
    blastai_logger = logging.getLogger('blastai')
    blastai_level = getattr(logging, settings.blastai_log_level.upper())
    blastai_logger.setLevel(blastai_level)
    
    # Configure browser-use logger via environment variable
    # Set this before any browser-use imports
    os.environ["BROWSER_USE_LOGGING_LEVEL"] = settings.browser_use_log_level.lower()
    
    # Silence third-party loggers
    for logger_name in [
        'uvicorn',
        'uvicorn.access',
        'uvicorn.error',
        'asyncio',
        'httpx',
        'httpcore',
        'playwright',
        'PIL',
    ]:
        third_party = logging.getLogger(logger_name)
        third_party.setLevel(logging.ERROR)
        third_party.propagate = False

def should_show_metrics(settings: Settings) -> bool:
    """Determine if metrics should be shown based on log levels.
    
    Args:
        settings: Settings instance with logging configuration
        
    Returns:
        True if metrics should be shown, False otherwise
    """
    # Only show metrics if both loggers are at error or critical level
    blastai_level = settings.blastai_log_level.upper()
    browser_use_level = settings.browser_use_log_level.upper()
    
    allowed_levels = {'ERROR', 'CRITICAL'}
    return blastai_level in allowed_levels and browser_use_level in allowed_levels