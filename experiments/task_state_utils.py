"""Utilities for managing and merging task states in experiments."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from blastai.scheduler import TaskState

AGISDK_TIMEOUT = 10  # seconds


def get_successful_task(
    parallelism_config: Dict[str, Any],
    task_states: Dict[str, TaskState],
    logger: logging.Logger,
) -> Optional[TaskState]:
    """
    Get the successful task according to the parallelism config.

    For first-of-n, we need to find the specific subtask that succeeded.
    Otherwise, we can use the main task.
    """
    if parallelism_config.get("first_of_n", False):
        subtask = get_successful_subtask(task_states, logger)
        if subtask:
            return subtask
        # Fall back to main task if no subtasks found (single browser case)
        logger.info(
            "No subtasks found in first-of-n mode, falling back to main task", indent=6
        )
        return get_successful_main_task(task_states, logger)
    return get_successful_main_task(task_states, logger)


def get_task_for_evaluation(
    parallelism_config: Dict[str, Any],
    task_states: Dict[str, TaskState],
    logger: logging.Logger,
) -> Optional[TaskState]:
    """
    Get a task for evaluation, preferring successful tasks but falling back to any completed task.

    This ensures we can still evaluate even when the task reported failure,
    which is important for checking if actions were actually performed correctly.
    """
    # First try to get a successful task
    successful = get_successful_task(parallelism_config, task_states, logger)
    if successful:
        return successful

    # Fall back to any completed task for evaluation
    logger.info("No successful task found, falling back to any completed task", indent=6)

    if parallelism_config.get("first_of_n", False):
        subtask = get_any_completed_subtask(task_states, logger)
        if subtask:
            return subtask
        return get_any_completed_main_task(task_states, logger)

    return get_any_completed_main_task(task_states, logger)


def get_successful_main_task(
    task_states: Dict[str, TaskState], logger: logging.Logger
) -> Optional[TaskState]:
    """Get the successful main task from the task states."""
    for task_state in task_states.values():
        # Main task is completed successfully, has no parent, has an executor with browser session
        if (
            task_state.is_completed
            and task_state.success
            and task_state.parent_task_id is None
            and task_state.executor
            and task_state.executor.browser_session
        ):
            return task_state
    logger.warning("No successful main task found", indent=6)
    return None


def get_successful_subtask(
    task_states: Dict[str, TaskState], logger: logging.Logger
) -> Optional[TaskState]:
    """Get the successful subtask from the task states."""
    for task_state in task_states.values():
        # Subtask is completed successfully, has a parent, has an executor with browser session
        if (
            task_state.is_completed
            and task_state.success
            and task_state.executor
            and task_state.executor.browser_session
            and task_state.parent_task_id
        ):
            logger.info(f"Found successful subtask: {task_state.id}", indent=6)
            return task_state

    logger.warning("No successful subtask found", indent=6)
    return None


def get_any_completed_subtask(
    task_states: Dict[str, TaskState], logger: logging.Logger
) -> Optional[TaskState]:
    """Get any completed subtask from the task states, regardless of success."""
    for task_state in task_states.values():
        # Subtask is completed, has a parent, has an executor with browser session
        if (
            task_state.is_completed
            and task_state.executor
            and task_state.executor.browser_session
            and task_state.parent_task_id
        ):
            logger.info(f"Found completed subtask: {task_state.id}", indent=6)
            return task_state

    logger.warning("No completed subtask found", indent=6)
    return None


def get_any_completed_main_task(
    task_states: Dict[str, TaskState], logger: logging.Logger
) -> Optional[TaskState]:
    """Get any completed main task from the task states, regardless of success."""
    for task_state in task_states.values():
        if (
            task_state.is_completed
            and task_state.parent_task_id is None
            and task_state.executor
            and task_state.executor.browser_session
        ):
            return task_state
    logger.warning("No completed main task found", indent=6)
    return None


def get_all_completed_tasks(
    task_states: Dict[str, TaskState], logger: logging.Logger
) -> List[TaskState]:
    """Get all completed tasks that have executors with browser sessions."""
    completed_tasks = []
    for task_state in task_states.values():
        if (
            task_state.is_completed
            and task_state.executor
            and task_state.executor.browser_session
        ):
            completed_tasks.append(task_state)
    logger.info(f"Found {len(completed_tasks)} completed tasks", indent=6)
    return completed_tasks


async def fetch_final_state(
    task_state: TaskState, initial_url: str, logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Fetch the final state from a single task by navigating to the finish page."""
    try:
        if not task_state.executor or not task_state.executor.browser_session:
            logger.warning(
                f"Task {task_state.id} has no executor or browser session", indent=8
            )
            return None

        page = await task_state.executor.browser_session.get_current_page()
        if not page:
            logger.warning(f"Task {task_state.id} has no current page", indent=8)
            return None

        finish_url = urljoin(initial_url, "finish")
        logger.info(
            f"Fetching final state from task {task_state.id}: {finish_url}", indent=8
        )

        await page.goto(finish_url)
        await asyncio.sleep(AGISDK_TIMEOUT)

        env_state = await page.evaluate(
            "() => document.querySelector('pre')?.textContent || ''"
        )
        if env_state:
            return json.loads(env_state)
        else:
            logger.warning(f"Task {task_state.id} returned empty state", indent=8)
            return None

    except Exception as e:
        logger.error(
            f"Failed to fetch final state from task {task_state.id}: {e}", indent=8
        )
        return None


def _merge_differences(
    final_states: Dict[str, Dict[str, Any]], logger: logging.Logger
) -> Dict[str, Any]:
    """
    Merge the 'differences' section from multiple task states.

    Each task may have its own differences (added/deleted/updated emails, etc.).
    We need to combine all of them to get the full picture of what changed.
    """
    merged_differences: Dict[str, Any] = {}
    # Track seen IDs per category/change_type to deduplicate
    seen_ids: Dict[str, Dict[str, set]] = {}

    for task_id, state in final_states.items():
        task_differences = state.get("differences", {})
        if not task_differences:
            continue

        for category, changes in task_differences.items():
            # category is e.g., "emails", and changes is {"added": [...], "deleted": [...], "updated": [...]}
            if category not in merged_differences:
                merged_differences[category] = {}
                seen_ids[category] = {}

            if not isinstance(changes, dict):
                continue

            for change_type, items in changes.items():
                # change_type is e.g., "added", "deleted", "updated"
                if change_type not in merged_differences[category]:
                    merged_differences[category][change_type] = []
                    seen_ids[category][change_type] = set()

                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            item_id = item.get("id")
                            # Deduplicate by ID
                            if item_id is not None:
                                if item_id in seen_ids[category][change_type]:
                                    continue  # Skip duplicate
                                seen_ids[category][change_type].add(item_id)
                            item["_source_task_id"] = task_id
                            merged_differences[category][change_type].append(item)
                elif items is not None:
                    # Handle non-list values (shouldn't happen often, but be safe)
                    merged_differences[category][change_type].append(items)

    return merged_differences


def merge_parallel_final_states(
    final_states: Dict[str, Any], logger: logging.Logger
) -> Dict[str, Any]:
    """
    Merge final states from multiple tasks.

    Combines:
    - Action histories (sorted by timestamp)
    - Differences (added/deleted/updated items from all tasks)

    Other fields are taken from the last task's state.

    This is only used if parallelism is enabled.
    """
    if not final_states:
        logger.warning("No final states to merge", indent=8)
        return {}

    if len(final_states) == 1:
        logger.info("Only one final state, no merging needed", indent=8)
        return list(final_states.values())[0]

    logger.info(f"Merging {len(final_states)} final states", indent=8)

    # Combine all action histories and sort by timestamp
    all_actions = []
    for task_id, state in final_states.items():
        actions = state.get("actionhistory", [])
        for action in actions:
            action["_source_task_id"] = (
                task_id  # Add metadata about which task this action came from
            )
        all_actions.extend(actions)

    all_actions.sort(key=lambda x: x.get("timestamp", 0))  # Sort by timestamp
    for idx, action in enumerate(all_actions):
        action["index"] = idx

    # Merge differences from all tasks
    merged_differences = _merge_differences(final_states, logger)

    # Other fields are from the last state
    merged_state = final_states[list(final_states.keys())[-1]].copy()
    merged_state["actionhistory"] = all_actions
    merged_state["differences"] = merged_differences
    merged_state["_metadata"] = {
        "num_tasks_merged": len(final_states),
        "task_ids": list(final_states.keys()),
        "total_actions": len(all_actions),
    }

    # Log merge statistics
    diff_stats = []
    for category, changes in merged_differences.items():
        for change_type, items in changes.items():
            if items:
                diff_stats.append(f"{category}.{change_type}: {len(items)}")

    logger.info(
        f"Merged state has {len(all_actions)} total actions from {len(final_states)} tasks",
        indent=8,
    )
    if diff_stats:
        logger.info(f"Merged differences: {', '.join(diff_stats)}", indent=8)

    return merged_state
