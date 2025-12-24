"""Tests for scheduler task completion behavior."""

from unittest.mock import MagicMock

import pytest

from blastai.scheduler import Scheduler


def get_mock_result(is_successful: bool):
    """Create a mock AgentHistoryList with specified success status."""
    result = MagicMock()
    result.is_successful.return_value = is_successful
    return result


@pytest.fixture
def scheduler():
    """Create a scheduler with mocked dependencies."""
    mock_constraints = MagicMock()
    mock_cache_manager = MagicMock()
    # Ensure cache always returns None (no cached result)
    mock_cache_manager.get_result.return_value = None
    mock_planner = MagicMock()
    return Scheduler(
        constraints=mock_constraints,
        cache_manager=mock_cache_manager,
        planner=mock_planner,
    )


@pytest.mark.asyncio
async def test_complete_task_with_successful_result(scheduler: Scheduler):
    """Test that task is marked successful when result.is_successful() is True."""
    # Create a task with a successful result
    task_id = scheduler.schedule_task("Test successful task")
    result = get_mock_result(is_successful=True)
    await scheduler.complete_task(task_id, result)
    
    # Verify task is marked as successful
    task = scheduler.tasks[task_id]
    assert task.is_completed is True
    assert task.success is True, "Task should be successful when result.is_successful() is True"


@pytest.mark.asyncio
async def test_complete_task_with_failed_result(scheduler: Scheduler):
    """Test that task is marked as failed when result.is_successful() is False."""
    # Create a task with a failed result
    task_id = scheduler.schedule_task("Test failed task")
    result = get_mock_result(is_successful=False)
    await scheduler.complete_task(task_id, result)
    
    # Verify task is marked as failed
    task = scheduler.tasks[task_id]
    assert task.is_completed is True
    assert task.success is False, "Task should NOT be successful when result.is_successful() is False"


@pytest.mark.asyncio
async def test_complete_task_with_explicit_success_true(scheduler: Scheduler):
    """Test that explicit success=True overrides result.is_successful()."""
    # Create a task with a failed result but pass success=True explicitly
    task_id = scheduler.schedule_task("Test explicit success")
    result = get_mock_result(is_successful=False)
    await scheduler.complete_task(task_id, result, success=True)

    # Verify task is marked as successful
    task = scheduler.tasks[task_id]
    assert task.success is True, "Explicit success=True should override result"


@pytest.mark.asyncio
async def test_complete_task_with_explicit_success_false(scheduler: Scheduler):
    """Test that explicit success=False overrides result.is_successful()."""
    # Create a task with a successful result but pass success=False explicitly
    task_id = scheduler.schedule_task("Test explicit failure")
    result = get_mock_result(is_successful=True)
    await scheduler.complete_task(task_id, result, success=False)
    
    # Verify task is marked as failed
    task = scheduler.tasks[task_id]
    assert task.success is False, "Explicit success=False should override result"


@pytest.mark.asyncio
async def test_complete_task_without_result(scheduler: Scheduler):
    """Test that task without result is marked as failed."""
    # Create a task without a result
    task_id = scheduler.schedule_task("Test no result")
    await scheduler.complete_task(task_id, result=None)
    
    # Verify task is marked as failed
    task = scheduler.tasks[task_id]
    assert task.is_completed is True
    assert task.success is False, "Task without result should be marked as failed"

