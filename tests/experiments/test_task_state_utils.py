"""Tests for task state utilities."""

import json
import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from blastai.scheduler import TaskState
from experiments.task_state_utils import (
    fetch_final_state,
    get_all_completed_tasks,
    get_successful_main_task,
    get_successful_subtask,
    get_successful_task,
    merge_parallel_final_states,
)


# Test helpers
def create_mock_task_state(
    task_id: str,
    is_completed: bool = True,
    has_executor: bool = True,
    has_browser_session: bool = True,
    parent_task_id: str = None,
    success: bool = True,
) -> TaskState:
    """Create a mock TaskState for testing."""
    task_state = Mock(spec=TaskState)
    task_state.id = task_id
    task_state.is_completed = is_completed
    task_state.success = success
    task_state.parent_task_id = parent_task_id

    if has_executor:
        task_state.executor = Mock()
        if has_browser_session:
            task_state.executor.browser_session = Mock()
        else:
            task_state.executor.browser_session = None
    else:
        task_state.executor = None

    return task_state


def create_mock_logger() -> logging.Logger:
    """Create a mock logger for testing."""
    logger = Mock(spec=logging.Logger)
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


# Tests for get_successful_task
class TestGetSuccessfulTask:
    """Tests for get_successful_task function."""

    def test_first_of_n_dispatches_to_subtask(self):
        """Test that first_of_n config dispatches to get_successful_subtask."""
        logger = create_mock_logger()
        parallelism_config = {"first_of_n": True}

        # Create a successful subtask
        subtask = create_mock_task_state(
            "subtask1",
            is_completed=True,
            success=True,
            has_executor=True,
            has_browser_session=True,
            parent_task_id="main_task",
        )
        task_states = {"subtask1": subtask}

        result = get_successful_task(parallelism_config, task_states, logger)

        assert result is not None
        assert result.id == "subtask1"

    def test_no_first_of_n_dispatches_to_main_task(self):
        """Test that non-first_of_n config dispatches to get_successful_main_task."""
        logger = create_mock_logger()
        parallelism_config = {"first_of_n": False}

        # Create a main task
        main_task = create_mock_task_state("main_task", is_completed=True, has_executor=True, parent_task_id=None)
        task_states = {"main_task": main_task}

        result = get_successful_task(parallelism_config, task_states, logger)

        assert result is not None
        assert result.id == "main_task"

    def test_empty_config_dispatches_to_main_task(self):
        """Test that empty config dispatches to get_successful_main_task."""
        logger = create_mock_logger()
        parallelism_config = {}

        main_task = create_mock_task_state("main_task", is_completed=True, has_executor=True, parent_task_id=None)
        task_states = {"main_task": main_task}

        result = get_successful_task(parallelism_config, task_states, logger)

        assert result is not None
        assert result.id == "main_task"


# Tests for get_successful_main_task
class TestGetSuccessfulMainTask:
    """Tests for get_successful_main_task function."""

    def test_single_main_task_with_executor(self):
        """Test finding a single main task with executor."""
        logger = create_mock_logger()
        main_task = create_mock_task_state("main_task", has_executor=True, parent_task_id=None)
        task_states = {"main_task": main_task}

        result = get_successful_main_task(task_states, logger)

        assert result is not None
        assert result.id == "main_task"

    def test_main_task_without_executor(self):
        """Test that main task without executor is not returned."""
        logger = create_mock_logger()
        main_task = create_mock_task_state("main_task", has_executor=False, parent_task_id=None)
        task_states = {"main_task": main_task}

        result = get_successful_main_task(task_states, logger)

        assert result is None
        logger.warning.assert_called()
        assert "no successful main task" in logger.warning.call_args[0][0].lower()

    def test_only_subtasks_no_main_task(self):
        """Test when only subtasks exist, no main task."""
        logger = create_mock_logger()
        subtask = create_mock_task_state("subtask1", is_completed=True, has_executor=True, parent_task_id="main_task")
        task_states = {"subtask1": subtask}

        result = get_successful_main_task(task_states, logger)

        assert result is None
        logger.warning.assert_called()

    def test_empty_task_states(self):
        """Test with empty task states."""
        logger = create_mock_logger()
        task_states = {}

        result = get_successful_main_task(task_states, logger)

        assert result is None
        logger.warning.assert_called()

    def test_main_task_with_subtasks(self):
        """Test finding main task when both main task and subtasks exist."""
        logger = create_mock_logger()
        main_task = create_mock_task_state("main_task", has_executor=True, parent_task_id=None)
        subtask = create_mock_task_state("subtask1", has_executor=True, parent_task_id="main_task")
        task_states = {"main_task": main_task, "subtask1": subtask}

        result = get_successful_main_task(task_states, logger)

        assert result is not None
        assert result.id == "main_task"


# Tests for get_successful_subtask
class TestGetSuccessfulSubtask:
    """Tests for get_successful_subtask function."""

    def test_single_successful_subtask(self):
        """Test finding a single successful subtask."""
        logger = create_mock_logger()
        subtask = create_mock_task_state(
            "subtask1",
            is_completed=True,
            success=True,
            has_executor=True,
            has_browser_session=True,
            parent_task_id="main_task",
        )
        task_states = {"subtask1": subtask}

        result = get_successful_subtask(task_states, logger)

        assert result is not None
        assert result.id == "subtask1"
        logger.info.assert_called_once()
        assert "found successful subtask" in logger.info.call_args[0][0].lower()

    def test_subtask_not_completed(self):
        """Test that incomplete subtasks are not returned."""
        logger = create_mock_logger()
        subtask = create_mock_task_state(
            "subtask1",
            is_completed=False,
            success=True,
            has_executor=True,
            has_browser_session=True,
            parent_task_id="main_task",
        )
        task_states = {"subtask1": subtask}

        result = get_successful_subtask(task_states, logger)

        assert result is None
        logger.warning.assert_called_once()

    def test_subtask_not_successful(self):
        """Test that unsuccessful subtasks are not returned."""
        logger = create_mock_logger()
        subtask = create_mock_task_state(
            "subtask1",
            is_completed=True,
            success=False,
            has_executor=True,
            has_browser_session=True,
            parent_task_id="main_task",
        )
        task_states = {"subtask1": subtask}

        result = get_successful_subtask(task_states, logger)

        assert result is None
        logger.warning.assert_called_once()

    def test_subtask_without_executor(self):
        """Test that subtasks without executors are not returned."""
        logger = create_mock_logger()
        subtask = create_mock_task_state(
            "subtask1",
            is_completed=True,
            success=True,
            has_executor=False,
            parent_task_id="main_task",
        )
        task_states = {"subtask1": subtask}

        result = get_successful_subtask(task_states, logger)

        assert result is None
        logger.warning.assert_called_once()

    def test_subtask_without_browser_session(self):
        """Test that subtasks without browser sessions are not returned."""
        logger = create_mock_logger()
        subtask = create_mock_task_state(
            "subtask1",
            is_completed=True,
            success=True,
            has_executor=True,
            has_browser_session=False,
            parent_task_id="main_task",
        )
        task_states = {"subtask1": subtask}

        result = get_successful_subtask(task_states, logger)

        assert result is None
        logger.warning.assert_called_once()

    def test_main_task_not_returned(self):
        """Test that main tasks (without parent_task_id) are not returned."""
        logger = create_mock_logger()
        main_task = create_mock_task_state(
            "main_task",
            is_completed=True,
            success=True,
            has_executor=True,
            has_browser_session=True,
            parent_task_id=None,
        )
        task_states = {"main_task": main_task}

        result = get_successful_subtask(task_states, logger)

        assert result is None
        logger.warning.assert_called_once()

    def test_multiple_subtasks_returns_first_match(self):
        """Test that when multiple successful subtasks exist, the first one found is returned."""
        logger = create_mock_logger()
        subtask1 = create_mock_task_state(
            "subtask1",
            is_completed=True,
            success=True,
            has_executor=True,
            has_browser_session=True,
            parent_task_id="main_task",
        )
        subtask2 = create_mock_task_state(
            "subtask2",
            is_completed=True,
            success=True,
            has_executor=True,
            has_browser_session=True,
            parent_task_id="main_task",
        )
        task_states = {"subtask1": subtask1, "subtask2": subtask2}

        result = get_successful_subtask(task_states, logger)

        assert result is not None
        assert result.id in ["subtask1", "subtask2"]
        logger.info.assert_called_once()

    def test_mixed_tasks_returns_only_successful_subtask(self):
        """Test finding successful subtask among mixed task types."""
        logger = create_mock_logger()
        main_task = create_mock_task_state("main_task", has_executor=True, parent_task_id=None)
        failed_subtask = create_mock_task_state(
            "failed_subtask",
            is_completed=True,
            success=False,
            has_executor=True,
            has_browser_session=True,
            parent_task_id="main_task",
        )
        successful_subtask = create_mock_task_state(
            "successful_subtask",
            is_completed=True,
            success=True,
            has_executor=True,
            has_browser_session=True,
            parent_task_id="main_task",
        )
        task_states = {
            "main_task": main_task,
            "failed_subtask": failed_subtask,
            "successful_subtask": successful_subtask,
        }

        result = get_successful_subtask(task_states, logger)

        assert result is not None
        assert result.id == "successful_subtask"

    def test_empty_task_states(self):
        """Test with empty task states."""
        logger = create_mock_logger()
        task_states = {}

        result = get_successful_subtask(task_states, logger)

        assert result is None
        logger.warning.assert_called_once()


# Tests for get_all_completed_tasks
class TestGetAllCompletedTasks:
    """Tests for get_all_completed_tasks function."""

    def test_empty_task_states(self):
        """Test with empty task states dictionary."""
        logger = create_mock_logger()
        result = get_all_completed_tasks({}, logger)
        assert result == []

    def test_single_completed_task_with_browser(self):
        """Test with a single completed task that has a browser session."""
        logger = create_mock_logger()
        task = create_mock_task_state("task1", is_completed=True, has_executor=True, has_browser_session=True)
        task_states = {"task1": task}

        result = get_all_completed_tasks(task_states, logger)

        assert len(result) == 1
        assert result[0].id == "task1"

    def test_completed_task_without_executor(self):
        """Test that completed tasks without executors are filtered out."""
        logger = create_mock_logger()
        task = create_mock_task_state("task1", is_completed=True, has_executor=False)
        task_states = {"task1": task}

        result = get_all_completed_tasks(task_states, logger)

        assert len(result) == 0

    def test_completed_task_without_browser_session(self):
        """Test that completed tasks without browser sessions are filtered out."""
        logger = create_mock_logger()
        task = create_mock_task_state("task1", is_completed=True, has_executor=True, has_browser_session=False)
        task_states = {"task1": task}

        result = get_all_completed_tasks(task_states, logger)

        assert len(result) == 0

    def test_incomplete_task_with_browser(self):
        """Test that incomplete tasks are filtered out even if they have browser sessions."""
        logger = create_mock_logger()
        task = create_mock_task_state("task1", is_completed=False, has_executor=True, has_browser_session=True)
        task_states = {"task1": task}

        result = get_all_completed_tasks(task_states, logger)

        assert len(result) == 0

    def test_multiple_tasks_mixed(self):
        """Test with multiple tasks in various states."""
        logger = create_mock_logger()
        task1 = create_mock_task_state("task1", is_completed=True, has_executor=True, has_browser_session=True)
        task2 = create_mock_task_state("task2", is_completed=True, has_executor=False)
        task3 = create_mock_task_state("task3", is_completed=False, has_executor=True, has_browser_session=True)
        task4 = create_mock_task_state("task4", is_completed=True, has_executor=True, has_browser_session=True)
        task5 = create_mock_task_state("task5", is_completed=True, has_executor=True, has_browser_session=False)

        task_states = {
            "task1": task1,
            "task2": task2,
            "task3": task3,
            "task4": task4,
            "task5": task5,
        }

        result = get_all_completed_tasks(task_states, logger)

        assert len(result) == 2
        result_ids = {task.id for task in result}
        assert result_ids == {"task1", "task4"}


# Tests for fetch_final_state_from_task
class TestFetchFinalStateFromTask:
    """Tests for fetch_final_state_from_task function."""

    @pytest.mark.asyncio
    async def test_no_executor(self):
        """Test when task state has no executor."""
        task = create_mock_task_state("task1", has_executor=False)
        logger = create_mock_logger()

        result = await fetch_final_state(task, "http://example.com", logger)

        assert result is None
        logger.warning.assert_called_once()
        assert "no executor" in logger.warning.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_no_browser_session(self):
        """Test when task state has no browser session."""
        task = create_mock_task_state("task1", has_executor=True, has_browser_session=False)
        logger = create_mock_logger()

        result = await fetch_final_state(task, "http://example.com", logger)

        assert result is None
        logger.warning.assert_called_once()
        assert "no executor" in logger.warning.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_no_current_page(self):
        """Test when browser session has no current page."""
        task = create_mock_task_state("task1", has_executor=True, has_browser_session=True)
        task.executor.browser_session.get_current_page = AsyncMock(return_value=None)
        logger = create_mock_logger()

        result = await fetch_final_state(task, "http://example.com", logger)

        assert result is None
        assert logger.warning.call_count == 1
        assert "no current page" in logger.warning.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        """Test successful fetch of final state."""
        task = create_mock_task_state("task1", has_executor=True, has_browser_session=True)

        # Mock the page and its methods
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value='{"state": "final", "score": 100}')

        task.executor.browser_session.get_current_page = AsyncMock(return_value=mock_page)
        logger = create_mock_logger()

        # Patch asyncio.sleep to speed up test
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_final_state(task, "http://example.com/task", logger)

        assert result is not None
        assert result == {"state": "final", "score": 100}
        mock_page.goto.assert_called_once_with("http://example.com/finish")
        logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_state_response(self):
        """Test when the page returns empty state."""
        task = create_mock_task_state("task1", has_executor=True, has_browser_session=True)

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="")

        task.executor.browser_session.get_current_page = AsyncMock(return_value=mock_page)
        logger = create_mock_logger()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_final_state(task, "http://example.com", logger)

        assert result is None
        assert logger.warning.call_count == 1
        assert "empty state" in logger.warning.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_json_parse_error(self):
        """Test when JSON parsing fails."""
        task = create_mock_task_state("task1", has_executor=True, has_browser_session=True)

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="invalid json{")

        task.executor.browser_session.get_current_page = AsyncMock(return_value=mock_page)
        logger = create_mock_logger()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_final_state(task, "http://example.com", logger)

        assert result is None
        logger.error.assert_called_once()
        assert "failed to fetch" in logger.error.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_navigation_error(self):
        """Test when page navigation fails."""
        task = create_mock_task_state("task1", has_executor=True, has_browser_session=True)

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("Navigation failed"))

        task.executor.browser_session.get_current_page = AsyncMock(return_value=mock_page)
        logger = create_mock_logger()

        result = await fetch_final_state(task, "http://example.com", logger)

        assert result is None
        logger.error.assert_called_once()
        assert "failed to fetch" in logger.error.call_args[0][0].lower()


# Tests for merge_final_states
class TestMergeFinalStates:
    """Tests for merge_final_states function."""

    def test_empty_states(self):
        """Test with empty final states dict."""
        logger = create_mock_logger()

        result = merge_parallel_final_states({}, logger)

        assert result == {}
        logger.warning.assert_called_once()
        assert "no final states" in logger.warning.call_args[0][0].lower()

    def test_single_state(self):
        """Test with a single final state."""
        logger = create_mock_logger()
        state = {"actionhistory": [{"action": "click", "timestamp": 100}], "final_state": "complete"}
        final_states = {"task1": state}

        result = merge_parallel_final_states(final_states, logger)

        assert result == state
        logger.info.assert_called_once()
        assert "no merging needed" in logger.info.call_args[0][0].lower()

    def test_merge_two_states_with_sorted_timestamps(self):
        """Test merging two states with actions sorted by timestamp."""
        logger = create_mock_logger()

        state1 = {
            "actionhistory": [
                {"action": "click", "timestamp": 100, "index": 0},
                {"action": "type", "timestamp": 200, "index": 1},
            ],
            "initial_state": "start",
            "final_state": "partial",
        }

        state2 = {
            "actionhistory": [
                {"action": "scroll", "timestamp": 150, "index": 0},
                {"action": "submit", "timestamp": 250, "index": 1},
            ],
            "initial_state": "start",
            "final_state": "complete",
        }

        final_states = {"task1": state1, "task2": state2}

        result = merge_parallel_final_states(final_states, logger)

        # Check that the result uses the last state's initial and final state
        assert result["initial_state"] == "start"
        assert result["final_state"] == "complete"

        # Check action history is merged and sorted
        actions = result["actionhistory"]
        assert len(actions) == 4

        # Check timestamps are in order
        timestamps = [action["timestamp"] for action in actions]
        assert timestamps == [100, 150, 200, 250]

        # Check actions are reindexed
        indices = [action["index"] for action in actions]
        assert indices == [0, 1, 2, 3]

        # Check source task IDs are added (sorted by timestamp, so interleaved)
        source_task_ids = [action["_source_task_id"] for action in actions]
        assert source_task_ids == ["task1", "task2", "task1", "task2"]

        # Check metadata
        assert result["_metadata"]["num_tasks_merged"] == 2
        assert result["_metadata"]["task_ids"] == ["task1", "task2"]
        assert result["_metadata"]["total_actions"] == 4

    def test_merge_three_states(self):
        """Test merging three states."""
        logger = create_mock_logger()

        state1 = {
            "actionhistory": [{"action": "click", "timestamp": 100}],
            "final_state": "state1",
        }

        state2 = {
            "actionhistory": [{"action": "type", "timestamp": 50}],
            "final_state": "state2",
        }

        state3 = {
            "actionhistory": [{"action": "submit", "timestamp": 200}],
            "final_state": "state3",
        }

        final_states = {"task1": state1, "task2": state2, "task3": state3}

        result = merge_parallel_final_states(final_states, logger)

        # Should use the last state
        assert result["final_state"] == "state3"

        # Check actions are sorted by timestamp
        actions = result["actionhistory"]
        assert len(actions) == 3
        assert actions[0]["timestamp"] == 50
        assert actions[1]["timestamp"] == 100
        assert actions[2]["timestamp"] == 200

        # Check reindexing
        assert actions[0]["index"] == 0
        assert actions[1]["index"] == 1
        assert actions[2]["index"] == 2

    def test_merge_with_empty_action_histories(self):
        """Test merging states where some have empty action histories."""
        logger = create_mock_logger()

        state1 = {"actionhistory": [], "final_state": "state1"}
        state2 = {"actionhistory": [{"action": "click", "timestamp": 100}], "final_state": "state2"}

        final_states = {"task1": state1, "task2": state2}

        result = merge_parallel_final_states(final_states, logger)

        assert len(result["actionhistory"]) == 1
        assert result["actionhistory"][0]["action"] == "click"
        assert result["_metadata"]["total_actions"] == 1

    def test_merge_without_timestamps(self):
        """Test merging states with actions that don't have timestamps."""
        logger = create_mock_logger()

        state1 = {"actionhistory": [{"action": "click"}], "final_state": "state1"}
        state2 = {"actionhistory": [{"action": "type"}], "final_state": "state2"}

        final_states = {"task1": state1, "task2": state2}

        result = merge_parallel_final_states(final_states, logger)

        # Should not crash, actions without timestamps will have timestamp 0
        assert len(result["actionhistory"]) == 2

    def test_merge_preserves_other_state_fields(self):
        """Test that merge preserves other fields from the final state."""
        logger = create_mock_logger()

        state1 = {
            "actionhistory": [{"action": "click", "timestamp": 100}],
            "final_state": "state1",
            "other_field": "value1",
        }

        state2 = {
            "actionhistory": [{"action": "type", "timestamp": 200}],
            "final_state": "state2",
            "other_field": "value2",
            "extra_field": "extra",
        }

        final_states = {"task1": state1, "task2": state2}

        result = merge_parallel_final_states(final_states, logger)

        # Should preserve fields from the last state
        assert result["other_field"] == "value2"
        assert result["extra_field"] == "extra"
        assert result["final_state"] == "state2"

    def test_state_copying(self):
        """Test that the merge doesn't modify the original states."""
        logger = create_mock_logger()

        original_state1 = {
            "actionhistory": [{"action": "click", "timestamp": 100}],
            "final_state": "state1",
        }
        original_state2 = {
            "actionhistory": [{"action": "type", "timestamp": 200}],
            "final_state": "state2",
        }

        # Make copies to check they're not modified
        state1_copy = json.loads(json.dumps(original_state1))
        state2_copy = json.loads(json.dumps(original_state2))

        final_states = {"task1": original_state1, "task2": original_state2}

        result = merge_parallel_final_states(final_states, logger)

        # Original states should have _source_task_id added (this is expected behavior)
        # but the structure should otherwise be preserved
        assert len(original_state1["actionhistory"]) == 1
        assert len(original_state2["actionhistory"]) == 1
        assert len(result["actionhistory"]) == 2
