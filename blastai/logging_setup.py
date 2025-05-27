"""Logging configuration for BLAST."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, TextIO

from .config import Settings

# Import metrics display from cli_process
from .cli_process import get_metrics_display

class LogRedirect:
    """Redirect stdout/stderr to log files while allowing specific prints through."""
    def __init__(self, log_file: Path, original_stream: TextIO):
        self.log_file = log_file
        self.original_stream = original_stream
        
    def write(self, text: str):
        # Handle ANSI escape sequences and metrics display
        is_ansi = text.startswith('\033')
        is_metrics = text.strip().startswith('Tasks:')  # Only match start of metrics block
        is_metrics_content = (
            'Scheduled:' in text or
            'Running:' in text or
            'Completed:' in text or
            'Resources:' in text or
            'Memory usage:' in text or
            'Total cost:' in text
        )
        is_panel = any(x in text for x in ['Engine: http://', 'Web: http://'])
        is_shutdown = 'Shutting down' in text

        display = get_metrics_display()
        # Allow through:
        # 1. Initial panel
        # 2. Metrics updates (both header and content) and their ANSI control sequences
        # 3. Shutdown message
        if (is_panel or is_shutdown or
            is_metrics or  # Metrics header
            (is_metrics_content and display.initialized) or  # Metrics content after initialization
            (is_ansi and display.initialized)):  # ANSI codes for metrics updates
            self.original_stream.write(text)
        elif text.strip():  # Skip empty lines for log file
            with open(self.log_file, 'a') as f:
                f.write(f"{text}\n")
                    
    def flush(self):
        self.original_stream.flush()

def setup_logging(settings: Optional[Settings] = None, engine_hash: Optional[str] = None):
    """Set up logging - redirect everything to files except metrics and shutdown."""
    if not settings:
        settings = Settings()
        
    # Create logs directory
    logs_dir = Path(settings.logs_dir or 'blast-logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up log files
    engine_log = logs_dir / f"{engine_hash or 'default'}.engine.log"
    web_log = logs_dir / f"{engine_hash or 'default'}.web.log"
    
    # Redirect stdout/stderr to engine.log
    sys.stdout = LogRedirect(engine_log, sys.__stdout__)
    sys.stderr = LogRedirect(engine_log, sys.__stderr__)
    
    # Get log levels from settings
    blastai_level = getattr(logging, settings.blastai_log_level.upper(), logging.DEBUG)
    browser_use_level = getattr(logging, settings.browser_use_log_level.upper(), logging.INFO)
    
    # Configure root logger to WARNING (for all third-party libraries)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up engine log handler for all logs
    engine_handler = logging.FileHandler(engine_log)
    engine_handler.setFormatter(formatter)
    engine_handler.setLevel(logging.WARNING)  # Base level for third-party libs
    root_logger.addHandler(engine_handler)
    
    # Set up web log handler for web-specific logs
    web_handler = logging.FileHandler(web_log)
    web_handler.setFormatter(formatter)
    web_handler.addFilter(lambda record: record.name == 'web')
    web_handler.setLevel(logging.INFO)  # Web UI logs at INFO level
    root_logger.addHandler(web_handler)
    
    # Configure loggers with their own handlers to ensure proper levels
    def setup_logger(name: str, level: int):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Add handler specifically for this logger
        handler = logging.FileHandler(engine_log)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)
        
        # Don't propagate to root logger since we have our own handler
        logger.propagate = False
        
        return logger
    
    # Set up specific loggers
    blastai_logger = setup_logger('blastai', blastai_level)
    browser_use_logger = setup_logger('browser_use', browser_use_level)
    web_logger = setup_logger('web', logging.INFO)
    
    # Also set child loggers
    for name, level in [('blastai', blastai_level), ('browser_use', browser_use_level)]:
        for child in ['server', 'engine', 'scheduler', 'executor']:
            child_logger = logging.getLogger(f"{name}.{child}")
            child_logger.setLevel(level)

def should_show_metrics(settings: Settings) -> bool:
    """Always return True since metrics should always be shown."""
    return True