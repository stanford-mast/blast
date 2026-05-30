"""Tests for error logging improvements."""

import logging
import pytest
from unittest.mock import patch, MagicMock
from blastai.scheduler import Scheduler
from blastai.executor import Executor
from blastai.config import Constraints, Settings
from blastai.planner import Planner
from blastai.cache import CacheManager


@pytest.mark.asyncio
async def test_scheduler_exception_logging():
    """Test that scheduler logs exceptions with stack traces."""
    # Create a mock logger
    with patch('blastai.scheduler.logger') as mock_logger:
        # Create scheduler with minimal setup
        constraints = Constraints()
        planner = Planner(constraints)
        cache_manager = CacheManager(instance_hash="test", persist=False, constraints=constraints)
        scheduler = Scheduler(constraints=constraints, cache_manager=cache_manager, planner=planner)
        cache_manager.load(scheduler)
        
        # Schedule a task
        task_id = scheduler.schedule_task("test task", cache_control="")
        task = scheduler.tasks[task_id]
        
        # Create a mock executor that raises an exception
        import asyncio
        
        async def failing_run(*args, **kwargs):
            raise RuntimeError("Test LLM error")
        
        mock_executor = MagicMock()
        mock_executor.run = failing_run
        task.executor = mock_executor
        
        # Create a mock task
        task.executor_run_task = asyncio.create_task(mock_executor.run())
        
        # Try to get result - should raise and log with exc_info
        with pytest.raises(RuntimeError):
            await scheduler.get_task_result(task_id)
        
        # Verify that logger.error was called with exc_info=True
        mock_logger.error.assert_called()
        # Check that exc_info=True was passed
        call_args = mock_logger.error.call_args
        assert call_args is not None
        assert call_args.kwargs.get('exc_info') is True, "exc_info=True should be passed to logger.error"


@pytest.mark.asyncio 
async def test_executor_exception_logging():
    """Test that executor logs exceptions with stack traces."""
    with patch('blastai.executor.logger') as mock_logger:
        # Create minimal executor setup
        from browser_use.browser import BrowserSession
        from browser_use import Controller
        
        # Mock the browser session and LLM
        mock_browser = MagicMock(spec=BrowserSession)
        mock_llm = MagicMock()
        mock_controller = MagicMock(spec=Controller)
        
        constraints = Constraints()
        settings = Settings()
        
        executor = Executor(
            browser_session=mock_browser,
            controller=mock_controller,
            llm=mock_llm,
            constraints=constraints,
            task_id="test",
            settings=settings
        )
        
        # Create an async function that raises an exception
        async def failing_run(*args, **kwargs):
            raise RuntimeError("Test LLM connection error")
        
        # Mock agent.run() to raise an exception
        executor.agent = MagicMock()
        executor.agent.run = failing_run
        
        # Try to run - should raise and log with exc_info
        with pytest.raises(RuntimeError):
            await executor.run("test task")
        
        # Verify that logger.error was called with exc_info=True
        mock_logger.error.assert_called()
        # Check that at least one call has exc_info=True
        exc_info_calls = [call for call in mock_logger.error.call_args_list 
                          if call.kwargs.get('exc_info') is True]
        assert len(exc_info_calls) > 0, "At least one logger.error call should have exc_info=True"
