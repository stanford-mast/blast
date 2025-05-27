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
                "filters": ["stdout_filter"],
                "mode": "a",  # Append mode
                "delay": True,  # Only open file when needed
            },
            "web_file": {
                "class": "logging.FileHandler",
                "filename": str(logs_dir / f"{engine_hash}.web.log"),
                "formatter": "default",
                "filters": ["stdout_filter"],
                "mode": "a",
                "delay": True,
            },
            "warnings_file": {
                "class": "logging.FileHandler",
                "filename": str(logs_dir / f"{engine_hash}.warnings.log"),
                "formatter": "default",
                "filters": ["stdout_filter"],
                "mode": "a",
                "delay": True,
            },
            "null_handler": {
                "class": "logging.NullHandler",  # Discard any output not caught by other handlers
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["engine_file"],
                "level": "ERROR",
                "propagate": False,  # Don't propagate to avoid double logging
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
            },
            "asyncio": {
                "handlers": ["engine_file"],
                "level": "ERROR",  # Only log asyncio errors
                "propagate": False,
            },
            "playwright": {
                "handlers": ["engine_file"],
                "level": "ERROR",  # Only log playwright errors
                "propagate": False,
            },
            "httpx": {
                "handlers": ["engine_file"],
                "level": "ERROR",  # Only log httpx errors
                "propagate": False,
            },
            "fastapi": {
                "handlers": ["engine_file"],
                "level": "ERROR",  # Only log FastAPI errors
                "propagate": False,
            },
            "starlette": {
                "handlers": ["engine_file"],
                "level": "ERROR",  # Only log Starlette errors
                "propagate": False,
            },
            "httpcore": {
                "handlers": ["engine_file"],
                "level": "ERROR",  # Only log httpcore errors
                "propagate": False,
            }
        },
        "disable_existing_loggers": True,  # Disable any existing loggers
        "filters": {
            "stdout_filter": {
                "()": "logging.Filter",
                "name": ""  # Empty string means root logger
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
        # If no engine_hash, use console logging temporarily
        if not engine_hash:
            # Add null handler to root logger to prevent stdout output
            root_logger.addHandler(logging.NullHandler())
            root_logger.setLevel(logging.ERROR)
            
            # Configure specific loggers with null handlers
            loggers = {
                'blastai': settings.blastai_log_level.upper(),
                'browser_use': settings.browser_use_log_level.upper(),
                'playwright': 'ERROR',
                'asyncio': 'ERROR',
                'httpx': 'ERROR',
                'fastapi': 'ERROR',
                'starlette': 'ERROR',
                'httpcore': 'ERROR',
            }
            
            for name, level in loggers.items():
                logger = logging.getLogger(name)
                logger.addHandler(logging.NullHandler())
                logger.setLevel(getattr(logging, level))
                logger.propagate = False
        else:
            # If we have engine_hash but no logs_dir specified, use blast-logs/
            logs_dir = Path("blast-logs")
            if not logs_dir.exists():
                logs_dir.mkdir(parents=True)
                
            # Create and apply logging config
            log_config = create_log_config(settings, logs_dir, engine_hash)
            logging.config.dictConfig(log_config)
    
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