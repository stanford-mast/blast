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
    
    # Get log levels
    blastai_level = getattr(logging, settings.blastai_log_level.upper())
    browser_level = getattr(logging, settings.browser_use_log_level.upper())
    
    # Show timestamps for all levels except ERROR and CRITICAL
    def should_show_timestamp(level):
        return level not in {logging.ERROR, logging.CRITICAL}
    
    # Set format based on log levels
    show_blastai_timestamp = should_show_timestamp(blastai_level)
    show_browser_timestamp = should_show_timestamp(browser_level)
    use_detailed_format = True # show_blastai_timestamp or show_browser_timestamp
    
    # Use consistent timestamp format without milliseconds
    log_format = '%(asctime)s [%(name)s] %(message)s' if use_detailed_format else '%(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format, date_format))
    
    logging.basicConfig(
        level=logging.ERROR,
        handlers=[handler]
    )
    
    # Configure all loggers to error level
    logging.getLogger().setLevel(logging.ERROR)
    
    # Configure blastai logger
    blastai_logger = logging.getLogger('blastai')
    blastai_logger.setLevel(blastai_level)
    
    # Configure browser-use logger via environment variable and direct configuration
    # Set env var for browser-use's own setup
    os.environ["BROWSER_USE_LOGGING_LEVEL"] = settings.browser_use_log_level.lower()
    
    # Also configure browser-use logger directly after its setup
    browser_level = getattr(logging, settings.browser_use_log_level.upper())
    browser_use_logger = logging.getLogger('browser_use')
    browser_use_logger.setLevel(browser_level)
    
    # And ensure playwright logger matches the level
    playwright_logger = logging.getLogger('playwright')
    playwright_logger.setLevel(browser_level)
    
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
    
    Only show metrics if both blastai and browser-use loggers are at ERROR
    or CRITICAL level, since there will be too many logs otherwise.
    
    Args:
        settings: Settings instance with logging configuration
        
    Returns:
        True if metrics should be shown, False otherwise
    """
    blastai_level = settings.blastai_log_level.upper()
    browser_level = settings.browser_use_log_level.upper()
    
    # Only show metrics if both loggers are at ERROR or CRITICAL
    allowed_levels = {'ERROR', 'CRITICAL'}
    return blastai_level in allowed_levels and browser_level in allowed_levels