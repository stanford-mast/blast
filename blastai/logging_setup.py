"""Logging configuration for BLAST."""

import logging
import logging.config
import os
from pathlib import Path
import sys
import warnings
from typing import Optional, Dict

from .config import Settings

def create_log_config(settings: Settings, logs_dir: Path, engine_hash: str) -> Dict:
    """Create logging configuration dictionary.
    
    Args:
        settings: Settings instance with logging configuration
        logs_dir: Path to logs directory
        engine_hash: Hash of the engine instance for unique log files
        
    Returns:
        Logging configuration dictionary
    """
    # Use consistent timestamp format without milliseconds
    log_format = '%(asctime)s [%(name)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": log_format,
                "datefmt": date_format,
            },
        },
        "handlers": {
            "engine_file": {
                "class": "logging.FileHandler",
                "filename": str(logs_dir / f"{engine_hash}.engine.log"),
                "formatter": "default",
            },
            "web_file": {
                "class": "logging.FileHandler",
                "filename": str(logs_dir / f"{engine_hash}.web.log"),
                "formatter": "default",
            },
            "warnings_file": {
                "class": "logging.FileHandler",
                "filename": str(logs_dir / f"{engine_hash}.warnings.log"),
                "formatter": "default",
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["engine_file"],
                "level": "ERROR",
                "propagate": True,
            },
            "blastai": {
                "handlers": ["engine_file"],
                "level": settings.blastai_log_level.upper(),
                "propagate": False,
            },
            "browser_use": {
                "handlers": ["engine_file"],
                "level": settings.browser_use_log_level.upper(),
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["engine_file"],
                "level": settings.blastai_log_level.upper(),
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["engine_file"],
                "level": settings.blastai_log_level.upper(),
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["engine_file"],
                "level": settings.blastai_log_level.upper(),
                "propagate": False,
            },
            "web": {
                "handlers": ["web_file"],
                "level": settings.browser_use_log_level.upper(),
                "propagate": False,
            },
            "py.warnings": {
                "handlers": ["warnings_file"],
                "level": "WARNING",
                "propagate": False,
            }
        }
    }

def setup_logging(settings: Optional[Settings] = None, engine_hash: Optional[str] = None):
    """Configure logging for BLAST.
    
    This function:
    1. Configures root logger and all third-party loggers to ERROR by default
    2. Sets up blastai logger based on config
    3. Sets up browser-use logger via environment variable
    4. Configures warning handling through logging system
    
    Args:
        settings: Optional Settings instance with logging configuration
        engine_hash: Optional hash of the engine instance for unique log files
    """
    # Use default settings if none provided
    if not settings:
        settings = Settings()
    
    # Remove any existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up warning logging
    logging.captureWarnings(True)
    warnings.filterwarnings("default")  # Enable warnings
    
    # Configure logging based on settings
    if settings.logs_dir:
        # Create logs directory if needed
        logs_dir = Path(settings.logs_dir)
        if not logs_dir.exists():
            logs_dir.mkdir(parents=True)
            
        # Create and apply logging config
        log_config = create_log_config(settings, logs_dir, engine_hash or "default")
        logging.config.dictConfig(log_config)
        
        # Configure warning formatting to log to file
        def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
            logger = logging.getLogger('py.warnings')
            logger.warning(f'{category.__name__}: {message}')
            return ''  # Suppress terminal output
        warnings.formatwarning = warning_on_one_line
        
    else:
        # Configure console logging
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter('%(asctime)s [%(name)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        )
        
        # Configure root logger
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.ERROR)
        
        # Configure specific loggers
        blastai_logger = logging.getLogger('blastai')
        blastai_logger.setLevel(getattr(logging, settings.blastai_log_level.upper()))
        
        # Set browser-use log level via environment variable
        os.environ["BROWSER_USE_LOGGING_LEVEL"] = settings.browser_use_log_level.lower()
        
        # Configure browser-use logger
        browser_level = getattr(logging, settings.browser_use_log_level.upper())
        browser_use_logger = logging.getLogger('browser_use')
        browser_use_logger.setLevel(browser_level)
        
        # Configure playwright logger
        playwright_logger = logging.getLogger('playwright')
        playwright_logger.setLevel(browser_level)
    
    # Silence third-party loggers
    for logger_name in [
        'asyncio',
        'httpx',
        'httpcore',
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