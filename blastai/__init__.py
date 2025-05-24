"""BLAST - Browser-LLM Auto-Scaling Technology"""
from .config import Settings
from .logging_setup import setup_logging

# Set up logging with default settings
setup_logging(Settings())

# Import everything else after logging is configured
from .engine import Engine

__all__ = [
    'Engine',
    'Settings',
]