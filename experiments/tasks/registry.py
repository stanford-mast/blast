"""
Task validator registry.

Maps task IDs to their corresponding validator implementations.
Each task must have a corresponding validator that implements the TaskValidator interface.
"""
from typing import Dict, Optional
from pathlib import Path
import importlib.util
import logging

from .base import TaskValidator

logger = logging.getLogger(__name__)


class TaskValidatorRegistry:
    """Registry for mapping task IDs to validators."""

    def __init__(self):
        self._validators: Dict[str, TaskValidator] = {}
        self._load_validators()

    def _load_validators(self):
        """
        Auto-discover validators in the tasks directory.

        Looks for Python files that export a 'validator' instance.
        The task ID is derived from the filename (e.g., dashdish_deepresearch1.py -> dashdish-deepresearch1).
        """
        tasks_dir = Path(__file__).parent

        for py_file in tasks_dir.glob("*.py"):
            # Skip __init__, base, registry, and other infrastructure files
            if py_file.stem in ["__init__", "base", "registry", "validation", "accuracy"]:
                continue

            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Check if it has a 'validator' attribute
                if hasattr(module, 'validator') and isinstance(module.validator, TaskValidator):
                    # Convert filename to task ID (underscore to hyphen)
                    task_id = py_file.stem.replace('_', '-')
                    self._validators[task_id] = module.validator
                    logger.info(f"Registered validator for task: {task_id}")

            except Exception as e:
                logger.warning(f"Failed to load validator from {py_file}: {e}")

    def get_validator(self, task_id: str) -> Optional[TaskValidator]:
        """
        Get validator for a task ID.

        Args:
            task_id: Task identifier (e.g., "dashdish-deepresearch1")

        Returns:
            TaskValidator instance or None if not found
        """
        return self._validators.get(task_id)

    def has_validator(self, task_id: str) -> bool:
        """Check if a validator exists for the given task ID."""
        return task_id in self._validators

    def list_tasks(self) -> list[str]:
        """Return list of all registered task IDs."""
        return sorted(self._validators.keys())


# Global registry instance
_registry = TaskValidatorRegistry()


def get_validator(task_id: str) -> Optional[TaskValidator]:
    """Get validator for a task ID."""
    return _registry.get_validator(task_id)


def has_validator(task_id: str) -> bool:
    """Check if a validator exists for the given task ID."""
    return _registry.has_validator(task_id)


def list_tasks() -> list[str]:
    """Return list of all registered task IDs."""
    return _registry.list_tasks()
