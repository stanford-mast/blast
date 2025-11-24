"""Benchmark for multiple executor creation.

This benchmark launches multiple browser sessions to measure actual performance.

Run with:
    pytest tests/benchmarks/test_benchmark_multi_executors.py -m benchmark -v -s

Benchmarks are skipped by default in regular test runs.
"""

import asyncio
import time

import pytest

from blastai.cache import CacheManager
from blastai.config import Constraints, Settings
from blastai.planner import Planner
from blastai.resource_manager import ResourceManager
from blastai.scheduler import Scheduler


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_browser_session_creation_benchmark():
    """Benchmark real browser session creation.

    This test launches N real browser sessions and measures the time.
    """
    NUM_BROWSERS = 10

    # Setup components
    constraints = Constraints()
    settings = Settings()
    cache_manager = CacheManager(instance_hash="benchmark_test")
    planner = Planner(constraints=constraints)
    scheduler = Scheduler(constraints=constraints, cache_manager=cache_manager, planner=planner)

    resource_manager = ResourceManager(
        scheduler=scheduler,
        constraints=constraints,
        settings=settings,
        engine_hash="benchmark_test",
        cache_manager=cache_manager,
    )

    await resource_manager.start()

    # Give the background allocation loop time to start
    await asyncio.sleep(0.2)

    try:
        # Schedule tasks
        task_ids = []
        for i in range(NUM_BROWSERS):
            task_id = scheduler.schedule_task(
                description=f"Benchmark browser {i}",
                cache_control="no-cache",
                initial_url="about:blank",
            )
            task_ids.append(task_id)

        print(f"\n{'=' * 70}")
        print(f"Benchmarking Real Browser Session Creation")
        print(f"{'=' * 70}")
        print(f"Number of browsers: {NUM_BROWSERS}")
        print(f"Starting timer...")
        print(f"{'=' * 70}\n")

        start_time = time.time()

        # Wait for all tasks to get executors
        max_wait_time = 120  # 2 minutes timeout
        poll_interval = 0.2
        elapsed = 0
        last_status_print = 0

        while elapsed < max_wait_time:
            executors_assigned = sum(1 for tid in task_ids if scheduler.tasks[tid].executor is not None)

            # Print status every second
            if elapsed - last_status_print >= 1.0:
                print(f"  [{elapsed:.1f}s] Executors assigned: {executors_assigned}/{NUM_BROWSERS}")
                last_status_print = elapsed

            if executors_assigned == NUM_BROWSERS:
                break

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        creation_time = time.time() - start_time

        # Verify all created
        successful = sum(1 for tid in task_ids if scheduler.tasks[tid].executor is not None)

        print(f"\n{'=' * 70}")
        print(f"Results:")
        print(f"{'=' * 70}")
        print(f"  Browsers created: {successful}/{NUM_BROWSERS}")
        print(f"  Total time: {creation_time:.2f}s")
        print(f"  Average per browser: {creation_time / successful:.2f}s")
        print(f"")
        print(f"{'=' * 70}\n")

        assert successful == NUM_BROWSERS, f"Only {successful}/{NUM_BROWSERS} browser sessions created"

    finally:
        print("Cleaning up browser sessions...")
        for task_id in task_ids:
            task = scheduler.tasks.get(task_id)
            if task and task.executor:
                try:
                    await task.executor.cleanup()
                except Exception as e:
                    print(f"Error cleaning up task {task_id}: {e}")

        await resource_manager.stop()
        print("Cleanup complete.\n")
