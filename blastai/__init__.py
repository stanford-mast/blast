"""BLAST - Browser-LLM Auto-Scaling Technology"""

from typing import TYPE_CHECKING

from .config import Constraints, Settings
from .logging_setup import capture_early_logs

# Capture early logs before proper logging is set up
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
