"""BLAST - Browser-LLM Auto-Scaling Technology"""

import warnings
import logging
import sys

# Create a custom warning filter that logs to file
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    # Get the logger
    logger = logging.getLogger('blastai')
    # Log the warning
    logger.warning(f'{category.__name__}: {message}')
    # Return empty string to suppress terminal output
    return ''

# Only set up warning formatting if not running tests
if not sys.warnoptions:
    # Capture warnings in logs
    logging.captureWarnings(True)
    # Use our custom formatter
    warnings.formatwarning = warning_on_one_line

from .logging_setup import setup_logging
from .config import Settings

# Set up logging with default settings
setup_logging()

# Import everything else
from .engine import Engine
from .server import app, init_app_state
from .config import Settings, Constraints

__all__ = [
    'Engine',
    'app',
    'init_app_state',
    'Settings',
    'Constraints',
]