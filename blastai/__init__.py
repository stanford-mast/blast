"""BLAST - Browser-LLM Auto-Scaling Technology"""
import os
from .config import Settings
from .logging_setup import capture_early_logs

# Capture early logs before proper logging is set up
# Skip if we're in a standalone script mode (e.g., generate_tools.py, evaluate_tools.py)
if not os.environ.get('BLASTAI_STANDALONE_MODE', '').lower() in ('1', 'true', 'yes'):
    capture_early_logs()

# Only import Engine if explicitly needed (CLI/server context)
# Skip in DBOS/workflow context to avoid patchright dependency
# The Engine can still be imported explicitly with: from blastai.engine import Engine
__all__ = [
    'Settings',
]

# Lazy import Engine only if accessed
def __getattr__(name):
    if name == 'Engine':
        from .engine import Engine
        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")