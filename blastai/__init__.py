"""BLAST - Browser Language Agent System for Tasks."""

# Configure logging before any imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from .logging_setup import setup_logging
from .config import Settings

# Set up logging with default settings
setup_logging()

# Now import everything else
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