"""BLAST - Browser-LLM Auto-Scaling Technology"""
import os
from .config import Settings
from .logging_setup import capture_early_logs

# Capture early logs before proper logging is set up
# Skip if we're in a standalone script mode (e.g., generate_tools.py, evaluate_tools.py)
if not os.environ.get('BLASTAI_STANDALONE_MODE', '').lower() in ('1', 'true', 'yes'):
    capture_early_logs()

# Import everything else after logging is configured
from .engine import Engine

__all__ = [
    'Engine',
    'Settings',
]