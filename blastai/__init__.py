"""BLAST - Browser-LLM Auto-Scaling Technology"""

import os
from typing import TYPE_CHECKING

from .config import Constraints, Settings
from .logging_setup import capture_early_logs

# Capture early logs before proper logging is set up
# Skip if we're in a standalone script mode (e.g., generate_tools.py, evaluate_tools.py)
if not os.environ.get('BLASTAI_STANDALONE_MODE', '').lower() in ('1', 'true', 'yes'):
    capture_early_logs()

# Lazy import of Engine to avoid heavy dependencies during package import
if TYPE_CHECKING:
    from .engine import Engine

__all__ = [
    "Engine",
    "Settings",
    "Constraints",
]


def __getattr__(name):
    """Lazy import for heavy dependencies."""
    if name == "Engine":
        from .engine import Engine

        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
