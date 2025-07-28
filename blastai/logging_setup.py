"""Logging configuration for BLAST."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, TextIO, List, Dict, Any, Tuple

from .config import Settings

# We'll use a lazy import for metrics display to avoid circular imports
_metrics_display = None

def get_metrics_display():
    """Lazy import of get_metrics_display to avoid circular imports."""
    global _metrics_display
    if _metrics_display is None:
        # Only import when needed
        from .cli_process import get_metrics_display as _get_metrics_display
        _metrics_display = _get_metrics_display()
    return _metrics_display

# Global buffer to store early logs before we know the engine hash
_early_logs: List[Tuple[str, str, int, str]] = []  # (logger_name, level, timestamp, message)
_early_logging_configured = False

class EarlyLogHandler(logging.Handler):
    """Handler that captures logs before proper logging is set up."""
    def emit(self, record):
        global _early_logs
        _early_logs.append((
            record.name,
            record.levelname,
            record.created,
            self.format(record)
        ))

def capture_early_logs():
    """Set up minimal logging to capture logs before proper logging is set up."""
    global _early_logging_configured
    if _early_logging_configured:
        return
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Add handler to capture logs
    handler = EarlyLogHandler()
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    _early_logging_configured = True
    
    # Set up stdout/stderr redirection to capture prints
    sys.stdout = EarlyStdoutRedirect(sys.__stdout__)
    sys.stderr = EarlyStdoutRedirect(sys.__stderr__)

class EarlyStdoutRedirect:
    """Capture stdout/stderr before proper logging is set up."""
    def __init__(self, original_stream: TextIO):
        self.original_stream = original_stream
        
    def write(self, text: str):
        # Always write to console for early logs
        self.original_stream.write(text)
        
        # Also capture for later writing to log file
        if text.strip():
            global _early_logs
            _early_logs.append((
                "stdout" if self.original_stream is sys.__stdout__ else "stderr",
                "INFO",
                0,  # No timestamp
                text.rstrip()
            ))
    
    def flush(self):
        self.original_stream.flush()

class LogRedirect:
    """Redirect stdout/stderr to log files while allowing specific prints through."""
    def __init__(self, log_file: Path, original_stream: TextIO):
        self.log_file = log_file
        self.original_stream = original_stream
        self.server_started = False
        
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
        is_cli_output = (
            'Usage:' in text or
            'Commands:' in text or
            'Options:' in text or
            'Error:' in text or
            '🚀  Browser-LLM' in text or
            'Support' in text or
            'Version:' in text
        )
        
        # Detect warning messages
        is_warning = (
            'Warning:' in text or
            'DeprecationWarning:' in text or
            text.lstrip().startswith('Warning:') or
            text.lstrip().startswith('DeprecationWarning:')
        )

        # Mark server as started when we see the panel
        if is_panel:
            self.server_started = True

        # Before server starts, allow CLI output through
        if not self.server_started:
            # Before server starts, allow CLI output but redirect warnings
            if is_warning:
                with open(self.log_file, 'a') as f:
                    f.write(f"{text}\n")
            else:
                self.original_stream.write(text)
            return
            
        # After server starts, allow through:
        # 1. Initial panel
        # 2. Metrics updates (both header and content) and their ANSI control sequences
        # 3. Shutdown message
        # Check for Pydantic deprecation warnings
        is_pydantic_warning = (
            'PydanticDeprecatedSince20' in text
        )
        
        if is_pydantic_warning:
            # Always redirect Pydantic warnings to log file
            with open(self.log_file, 'a') as f:
                f.write(f"{text}\n")
        elif (is_panel or is_shutdown or
              is_metrics or  # Metrics header
              (is_metrics_content and get_metrics_display() and get_metrics_display().initialized) or  # Metrics content after initialization
              (is_ansi and get_metrics_display() and get_metrics_display().initialized)):  # ANSI codes for metrics updates
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
    
    # Process any early logs if we have an engine hash
    global _early_logs
    if engine_hash and _early_logs:
        with open(engine_log, 'a') as f:
            for logger_name, level, timestamp, message in _early_logs:
                f.write(f"{message}\n")
        # Clear early logs after processing
        _early_logs = []
    
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
        
        # Clear any existing handlers
        logger.handlers.clear()
        
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
            # Clear any existing handlers
            child_logger.handlers.clear()

def should_show_metrics(settings: Settings) -> bool:
    """Always return True since metrics should always be shown."""
    return True