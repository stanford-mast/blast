"""Tests for price and token usage tracking."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from blastai import Engine
from blastai.config import Constraints
from blastai.executor import Executor
from blastai.scheduler import TaskState


def get_mock_executor(task_id: str = "test-task"):
    return Executor(
        browser_session=MagicMock(),
        controller=MagicMock(),
        llm=MagicMock(),
        constraints=Constraints(),
        task_id=task_id,
    )


def get_mock_usage_summary(
    total_cost: float,
    total_prompt_tokens: int,
    total_prompt_cached_tokens: int,
    total_completion_tokens: int,
    total_tokens: int,
) -> MagicMock:
    summary = MagicMock()
    summary.total_cost = total_cost
    summary.total_prompt_tokens = total_prompt_tokens
    summary.total_prompt_cached_tokens = total_prompt_cached_tokens
    summary.total_completion_tokens = total_completion_tokens
    summary.total_tokens = total_tokens
    return summary


@pytest.mark.asyncio
async def test_single_round_cost_tracking():
    """Test that costs are correctly tracked for a single round."""
    executor = get_mock_executor()
    mock_token_service = MagicMock()
    mock_token_service.get_usage_summary = AsyncMock(
        side_effect=[
            get_mock_usage_summary(0.01, 800, 0, 200, 1000),
        ]
    )
    executor.agent = MagicMock()
    executor.agent.token_cost_service = mock_token_service

    await executor._get_cost_from_agent(executor.agent)
    cost_after_round1 = executor.get_total_cost()
    tokens_after_round1 = executor.get_total_token_usage()

    print(f"Round 1 - Cost: ${cost_after_round1:.4f}, Tokens: {tokens_after_round1.total}")
    assert cost_after_round1 == 0.01, f"Round 1 cost should be $0.01, got ${cost_after_round1}"
    assert tokens_after_round1.total == 1000, f"Round 1 tokens should be 1000, got {tokens_after_round1.total}"
    assert tokens_after_round1.prompt == 800, f"Round 1 prompt tokens should be 800, got {tokens_after_round1.prompt}"
    assert tokens_after_round1.prompt_cached == 0, (
        f"Round 1 prompt cached tokens should be 0, got {tokens_after_round1.prompt_cached}"
    )
    assert tokens_after_round1.completion == 200, (
        f"Round 1 completion tokens should be 200, got {tokens_after_round1.completion}"
    )


@pytest.mark.asyncio
async def test_multiround_cost_tracking():
    """Test that costs are correctly tracked across multiple rounds."""
    executor = get_mock_executor()
    mock_token_service = MagicMock()

    mock_token_service.get_usage_summary = AsyncMock(
        side_effect=[
            get_mock_usage_summary(0.01, 800, 0, 200, 1000),
            get_mock_usage_summary(0.025, 2000, 0, 500, 2500),
            get_mock_usage_summary(0.04, 3200, 0, 800, 4000),
        ]
    )

    # Mock the agent
    executor.agent = MagicMock()
    executor.agent.token_cost_service = mock_token_service

    # Simulate round 1
    await executor._get_cost_from_agent(executor.agent)
    cost_after_round1 = executor.get_total_cost()
    tokens_after_round1 = executor.get_total_token_usage()

    print(f"Round 1 - Cost: ${cost_after_round1:.4f}, Tokens: {tokens_after_round1.total}")
    assert cost_after_round1 == 0.01, f"Round 1 cost should be $0.01, got ${cost_after_round1}"
    assert tokens_after_round1.total == 1000, f"Round 1 tokens should be 1000, got {tokens_after_round1.total}"

    # Simulate round 2
    await executor._get_cost_from_agent(executor.agent)
    cost_after_round2 = executor.get_total_cost()
    tokens_after_round2 = executor.get_total_token_usage()

    print(f"Round 2 - Cost: ${cost_after_round2:.4f}, Tokens: {tokens_after_round2.total}")
    assert cost_after_round2 == 0.025, f"Round 2 cost should be $0.025, got ${cost_after_round2:.4f}"
    assert tokens_after_round2.total == 2500, f"Round 2 tokens should be 2500, got {tokens_after_round2.total}"

    # Simulate round 3
    await executor._get_cost_from_agent(executor.agent)
    cost_after_round3 = executor.get_total_cost()
    tokens_after_round3 = executor.get_total_token_usage()

    print(f"Round 3 - Cost: ${cost_after_round3:.4f}, Tokens: {tokens_after_round3.total}")
    assert cost_after_round3 == 0.04, f"Round 3 cost should be $0.04, got ${cost_after_round3:.4f}"
    assert tokens_after_round3.total == 4000, f"Round 3 tokens should be 4000, got {tokens_after_round3.total}"


@pytest.mark.asyncio
async def test_multiple_rounds_with_different_costs():
    """Test executor tracking across rounds with varying costs."""
    executor = get_mock_executor()
    mock_token_service = MagicMock()

    # Simulate 5 rounds with different incremental costs
    summaries = []
    cumulative_cost = 0.0
    cumulative_tokens = 0

    for i in range(1, 6):
        increment = i * 0.005  # $0.005, $0.010, $0.015, $0.020, $0.025
        cumulative_cost += increment
        cumulative_tokens += i * 500

        summary = get_mock_usage_summary(
            cumulative_cost, cumulative_tokens // 2, 0, cumulative_tokens // 2, cumulative_tokens
        )
        summaries.append(summary)

    mock_token_service.get_usage_summary = AsyncMock(side_effect=summaries)

    executor.agent = MagicMock()
    executor.agent.token_cost_service = mock_token_service

    # Run through all rounds
    expected_costs = [0.005, 0.015, 0.030, 0.050, 0.075]
    expected_tokens = [500, 1500, 3000, 5000, 7500]

    for round_num, (expected_cost, expected_total_tokens) in enumerate(zip(expected_costs, expected_tokens), 1):
        await executor._get_cost_from_agent(executor.agent)
        actual_cost = executor.get_total_cost()
        actual_tokens = executor.get_total_token_usage().total

        print(f"Round {round_num} - Cost: ${actual_cost:.4f}, Tokens: {actual_tokens}")

        assert abs(actual_cost - expected_cost) < 0.0001, (
            f"Round {round_num}: cost should be ${expected_cost:.4f}, got ${actual_cost:.4f}"
        )
        assert actual_tokens == expected_total_tokens, (
            f"Round {round_num}: tokens should be {expected_total_tokens}, got {actual_tokens}"
        )


@pytest.mark.asyncio
async def test_engine_get_metrics_cost_aggregation():
    """Test that engine.get_metrics() correctly aggregates costs from executors."""
    engine = await Engine.create()

    try:
        executor1 = get_mock_executor(task_id="test-task-1")

        executor2 = get_mock_executor(task_id="test-task-2")

        # Mock token services for both executors
        # Executor 1: $0.01, 1000 tokens
        mock_service1 = MagicMock()
        mock_service1.get_usage_summary = AsyncMock(return_value=get_mock_usage_summary(0.01, 800, 0, 200, 1000))

        executor1.agent = MagicMock()
        executor1.agent.token_cost_service = mock_service1
        await executor1._get_cost_from_agent(executor1.agent)

        # Executor 2: $0.02, 2000 tokens
        mock_service2 = MagicMock()
        mock_service2.get_usage_summary = AsyncMock(return_value=get_mock_usage_summary(0.02, 1600, 0, 400, 2000))

        executor2.agent = MagicMock()
        executor2.agent.token_cost_service = mock_service2
        await executor2._get_cost_from_agent(executor2.agent)

        # Add executors to scheduler tasks
        task1 = TaskState(id=executor1.task_id, description="test task 1")
        task1.executor = executor1
        task1.completed = False

        task2 = TaskState(id=executor2.task_id, description="test task 2")
        task2.executor = executor2
        task2.completed = False

        engine.scheduler.tasks[executor1.task_id] = task1
        engine.scheduler.tasks[executor2.task_id] = task2

        # Test 1: Get metrics should aggregate both executors
        metrics = await engine.get_metrics()

        print(f"Metrics - Total cost: ${metrics['total_cost']:.2f}")
        print(f"Metrics - Total tokens: {metrics['total_token_usage']['total']}")

        # Should be sum of both executors (0.01 + 0.02 = 0.03)
        assert metrics["total_cost"] == 0.03, f"Total cost should be $0.03, got ${metrics['total_cost']}"
        assert metrics["total_token_usage"]["total"] == 3000, (
            f"Total tokens should be 3000, got {metrics['total_token_usage']['total']}"
        )

        # Test 2: Simulate executor1 having a second round, total cost should be 0.03 + 0.02 = 0.05
        mock_service1.get_usage_summary = AsyncMock(return_value=get_mock_usage_summary(0.03, 3800, 0, 700, 4500))

        await executor1._get_cost_from_agent(executor1.agent)

        # Get metrics again
        metrics = await engine.get_metrics()

        print(f"After round 2 - Total cost: ${metrics['total_cost']:.2f}")
        print(f"After round 2 - Total tokens: {metrics['total_token_usage']['total']}")

        expected_cost_rounded = 0.05
        assert metrics["total_cost"] == expected_cost_rounded, (
            f"Total cost should be ${expected_cost_rounded:.2f} (0.045 rounded), got ${metrics['total_cost']}"
        )

        # Tokens should be 4500 (executor1) + 2000 (executor2) = 6500
        assert metrics["total_token_usage"]["total"] == 6500, (
            f"Total tokens should be 6500, got {metrics['total_token_usage']['total']}"
        )

    finally:
        await engine.stop()
