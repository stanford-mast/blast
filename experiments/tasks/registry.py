"""
Task validator registry with unified validation interface.

Maps task IDs to their corresponding validator implementations.
Supports both script-based (state + string matching) and LLM-based (structured output) validation.
"""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import yaml

from .base import TaskValidator

logger = logging.getLogger(__name__)


# Type alias for script evaluator function
ScriptEvaluator = Callable[[Dict[str, Any], str], Tuple[bool, float, Dict[str, Any]]]


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
            if py_file.stem in [
                "__init__",
                "base",
                "registry",
                "validation",
                "accuracy",
            ]:
                continue

            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Check if it has a 'validator' attribute
                if hasattr(module, "validator") and isinstance(
                    module.validator, TaskValidator
                ):
                    # Convert filename to task ID (underscore to hyphen)
                    task_id = py_file.stem.replace("_", "-")
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


class UnifiedValidator:
    """
    Unified validator supporting both script-based and LLM-based validation.

    - Script-based: Uses eval scripts that check browser state + string matching.
                    Requires final_state from /finish endpoint.
    - LLM-based: Uses LLM to parse structured output and validate against ground truth.
    """

    def __init__(self):
        self._task_registry = TaskValidatorRegistry()
        self._eval_types: Dict[str, str] = {}
        self._script_evaluators: Dict[str, ScriptEvaluator] = {}
        self._load_task_configs()
        self._load_script_evaluators()

    def _load_task_configs(self):
        """Load eval_type from task YAML files."""
        tasks_dir = Path(__file__).parent

        for yaml_file in tasks_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)

                tasks = data.get("tasks", []) if isinstance(data, dict) else data
                if not isinstance(tasks, list):
                    continue

                for task in tasks:
                    task_id = task.get("id")
                    eval_type = task.get("eval_type", "llm")  # Default to llm
                    if task_id:
                        self._eval_types[task_id] = eval_type
                        logger.debug(f"Loaded eval_type for {task_id}: {eval_type}")

            except Exception as e:
                logger.warning(f"Failed to load task config from {yaml_file}: {e}")

    def _load_script_evaluators(self):
        """Load script evaluators from eval_scripts directory."""
        eval_scripts_dir = Path(__file__).parent / "eval_scripts"

        if not eval_scripts_dir.exists():
            logger.warning(f"Eval scripts directory not found: {eval_scripts_dir}")
            return

        for py_file in eval_scripts_dir.glob("eval_*.py"):
            try:
                # Extract task ID from filename: eval_dashdish_custom_1.py -> dashdish-custom-1
                stem = py_file.stem  # e.g., "eval_dashdish_custom_1"
                if stem.startswith("eval_"):
                    # Remove "eval_" prefix and convert underscores to hyphens
                    task_id = stem[5:].replace("_", "-")

                    # Import the module
                    module_name = f"experiments.tasks.eval_scripts.{py_file.stem}"
                    module = importlib.import_module(module_name)

                    # Check if it has an 'evaluate' function
                    if hasattr(module, "evaluate") and callable(module.evaluate):
                        self._script_evaluators[task_id] = module.evaluate
                        logger.info(f"Loaded script evaluator for task: {task_id}")

            except Exception as e:
                logger.warning(f"Failed to load script evaluator from {py_file}: {e}")

    def get_eval_type(self, task_id: str) -> str:
        """Get the evaluation type for a task."""
        return self._eval_types.get(task_id, "llm")

    def has_script_evaluator(self, task_id: str) -> bool:
        """Check if a script evaluator exists for the task."""
        return task_id in self._script_evaluators

    def has_llm_validator(self, task_id: str) -> bool:
        """Check if an LLM validator exists for the task."""
        return self._task_registry.has_validator(task_id)

    async def validate(
        self,
        task_id: str,
        final_result: str,
        final_state: Optional[Dict[str, Any]] = None,
        return_pct: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate task output using the appropriate method.

        Args:
            task_id: Task identifier
            final_result: The agent's final output/answer
            final_state: Browser state from /finish endpoint (required for script validation)
            return_pct: If True, include 'correctness_pct' field

        Returns:
            dict with:
                - success: bool (whether parsing/validation succeeded)
                - correct: bool (whether output is 100% correct)
                - correctness_pct: float (0.0-1.0, only if return_pct=True)
                - eval_type: str ("script" or "llm")
                - details: dict with additional info
        """
        eval_type = self.get_eval_type(task_id)

        if eval_type == "script":
            if final_state is None:
                # Script eval requires final_state - fail explicitly rather than
                # silently falling back to LLM eval (which would be incorrect for
                # action-based tasks like placing orders)
                logger.error(
                    f"Task {task_id} requires script eval but final_state is None. "
                    "Cannot validate without browser state from /finish endpoint."
                )
                result = {
                    "success": False,
                    "correct": False,
                    "eval_type": "script",
                    "details": {
                        "error": "Script eval requires final_state from /finish endpoint, but it was not provided"
                    },
                }
                if return_pct:
                    result["correctness_pct"] = 0.0
                return result
            return self._run_script_eval(task_id, final_state, final_result, return_pct)
        else:
            return await self._run_llm_eval(task_id, final_result, return_pct)

    def _run_script_eval(
        self,
        task_id: str,
        final_state: Dict[str, Any],
        final_result: str,
        return_pct: bool = False,
    ) -> Dict[str, Any]:
        """Run script-based evaluation."""
        if task_id not in self._script_evaluators:
            logger.warning(f"No script evaluator found for task: {task_id}")
            result = {
                "success": False,
                "correct": False,
                "eval_type": "script",
                "details": {"error": f"No script evaluator for task {task_id}"},
            }
            if return_pct:
                result["correctness_pct"] = 0.0
            return result

        try:
            # Debug: Log what we're passing to the evaluator
            logger.debug(
                f"Script eval for {task_id}: final_state keys = {list(final_state.keys())}"
            )
            logger.debug(
                f"Script eval for {task_id}: final_result = {final_result[:200] if final_result else 'None'}..."
            )

            evaluator = self._script_evaluators[task_id]
            success, percentage, details = evaluator(final_state, final_result)

            # Debug: Log evaluation result details
            logger.info(
                f"Script eval for {task_id}: success={success}, pct={percentage:.0%}, details={details}"
            )

            result = {
                "success": True,
                "correct": success,
                "eval_type": "script",
                "details": details,
            }
            if return_pct:
                result["correctness_pct"] = percentage

            return result

        except Exception as e:
            logger.error(f"Script evaluation failed for {task_id}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            result = {
                "success": False,
                "correct": False,
                "eval_type": "script",
                "details": {"error": str(e)},
            }
            if return_pct:
                result["correctness_pct"] = 0.0
            return result

    async def _run_llm_eval(
        self,
        task_id: str,
        final_result: str,
        return_pct: bool = False,
    ) -> Dict[str, Any]:
        """Run LLM-based evaluation."""
        validator = self._task_registry.get_validator(task_id)

        if validator is None:
            logger.warning(f"No LLM validator found for task: {task_id}")
            result = {
                "success": False,
                "correct": False,
                "eval_type": "llm",
                "details": {"error": f"No LLM validator for task {task_id}"},
            }
            if return_pct:
                result["correctness_pct"] = 0.0
            return result

        try:
            validation = await validator.validate(final_result, return_pct=return_pct)
            result = {
                "success": validation.get("success", False),
                "correct": validation.get("correct", False),
                "eval_type": "llm",
                "details": {"parsed_output": validation.get("parsed_output")},
            }
            if return_pct:
                result["correctness_pct"] = validation.get("correctness_pct", 0.0)

            return result

        except Exception as e:
            logger.error(f"LLM evaluation failed for {task_id}: {e}")
            result = {
                "success": False,
                "correct": False,
                "eval_type": "llm",
                "details": {"error": str(e)},
            }
            if return_pct:
                result["correctness_pct"] = 0.0
            return result


# Global registry instances
_registry = TaskValidatorRegistry()
_unified_validator: Optional[UnifiedValidator] = None


def get_validator(task_id: str) -> Optional[TaskValidator]:
    """Get validator for a task ID (LLM-based validators only)."""
    return _registry.get_validator(task_id)


def has_validator(task_id: str) -> bool:
    """Check if a validator exists for the given task ID."""
    return _registry.has_validator(task_id)


def list_tasks() -> list[str]:
    """Return list of all registered task IDs."""
    return _registry.list_tasks()


def get_unified_validator() -> UnifiedValidator:
    """Get the unified validator singleton."""
    global _unified_validator
    if _unified_validator is None:
        _unified_validator = UnifiedValidator()
    return _unified_validator
