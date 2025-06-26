"""BLAST - Browser-LLM Auto-Scaling Technology"""
from .config import Settings
from .logging_setup import capture_early_logs

# Capture early logs before proper logging is set up
capture_early_logs()

# Import everything else after logging is configured
from .engine import Engine

__all__ = [
    'Engine',
    'Settings',
]