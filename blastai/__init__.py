"""BLAST - Browser-LLM Auto-Scaling Technology"""

# Configure logging and warnings before any imports
import os
import sys
import warnings
import builtins

# Create a warning interceptor
class WarningInterceptor:
    def __init__(self):
        self._orig_warnings = sys.modules['warnings']
        self._orig_warn = warnings.warn
        self._orig_showwarning = warnings.showwarning
        self._orig_formatwarning = warnings.formatwarning
        self._orig_filterwarnings = warnings.filterwarnings
        
    def warn(self, *args, **kwargs):
        message = str(args[0]) if args else ""
        if "numpy.core._multiarray_umath" in message or "faiss.loader" in str(kwargs.get('module', '')):
            return
        return self._orig_warn(*args, **kwargs)
        
    def showwarning(self, *args, **kwargs):
        message = str(args[0]) if args else ""
        if "numpy.core._multiarray_umath" in message or "faiss.loader" in str(kwargs.get('module', '')):
            return
        return self._orig_showwarning(*args, **kwargs)
        
    def formatwarning(self, *args, **kwargs):
        message = str(args[0]) if args else ""
        if "numpy.core._multiarray_umath" in message:
            return ""
        return self._orig_formatwarning(*args, **kwargs)
        
    def filterwarnings(self, *args, **kwargs):
        return self._orig_filterwarnings(*args, **kwargs)
        
    def __getattr__(self, name):
        return getattr(self._orig_warnings, name)

# Install the interceptor
_warning_interceptor = WarningInterceptor()
sys.modules['warnings'] = _warning_interceptor
builtins.Warning = type('Warning', (), {'__str__': lambda s: ''})

# Set warning filters as backup
warnings.simplefilter("ignore", DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

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