"""
Runner for experiments.

Usage:
    python -m experiments.runner --config configs/testing-experiment-config.yaml [--evaluate]

Options:
    --config: Path to the experiment config file
    --evaluate: Whether to evaluate the result using the agisdk evaluator. This is only supported when running tasks in the REAL Bench format.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from blastai import Engine
from blastai.logging_setup import setup_logging
from blastai.response import AgentHistoryListResponse

from .logger import ExperimentLogger
from .task_state_utils import (
    fetch_final_state,
    get_all_completed_tasks,
    get_successful_task,
    merge_parallel_final_states,
)
from .utils import ensure_parent_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/testing-experiment-config.yaml",
        help="Path to the experiment config file",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to evaluate the result using the agisdk evaluator. This is only supported when running one of the REAL tasks: https://github.com/agi-inc/agisdk/tree/main/src/agisdk/REAL/browsergym/webclones/tasks",
    )
    return parser.parse_args()


@dataclass
class ExperimentResult:
    """ExperimentResult stores the results from a single experiment run."""

    experiment_id: str
    engine_id: str
    task_goal: str
    llm_model: str
    llm_model_mini: str
    stage_name: str
    run_number: int
    reported_success: bool
    evaluated_success: Optional[bool]
    error: Optional[str]
    total_time: float
    metrics: Dict[str, Any]
    final_result: Optional[str]
    final_state_path: Optional[str]


class ExperimentRunner:
    def __init__(
        self,
        config_path: str = "experiments/experiment-config.yaml",
        evaluate: bool = False,
    ):
        self.config_path = config_path
        self.evaluate = evaluate
        self.results: List[ExperimentResult] = []
        self.load_config()

        self.logger = ExperimentLogger(self.config["settings"]["logs_dir"])
        self.expected_results_count = 0
        self.actual_results_count = 0

    def load_config(self):
        """Load the experiment config."""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def _get_experiment_hash(self, task_id: str, stage_name: str) -> str:
        """Get a unique hash for the experiment (this is shared across all runs)."""
        hash_input = f"{task_id}-{stage_name}-{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:8]

    def _create_experiment_folder(
        self,
        task_id: str,
        stage_name: str,
        run_number: int,
        experiment_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Create and return the experiment folder path."""
        if experiment_id is None:
            hash_id = self._get_experiment_hash(task_id, stage_name)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_id = f"{hash_id}_{timestamp}"

        experiment_folder = (
            Path(self.config["settings"]["logs_dir"]) / task_id / stage_name / experiment_id / f"run_{run_number}"
        )
        ensure_parent_dir(experiment_folder)
        return str(experiment_folder), experiment_id

    def _create_engine_config(self, stage_config: Dict[str, Any], experiment_folder: str) -> Dict[str, Any]:
        """Create engine configuration for the experiment."""
        return {
            "settings": {
                "local_browser_path": "none",
                "persist_cache": False,
                "logs_dir": experiment_folder,
                "secrets_file_path": "secrets.env",
                "blastai_log_level": "debug",
                "browser_use_log_level": "debug",
                "server_port": 8000,
                "web_port": 3000,
            },
            "constraints": {
                "max_memory": None,
                "max_concurrent_browsers": 20,
                "require_headless": True,
                "require_patchright": True,
                "require_human_in_loop": False,
                "share_browser_process": False,
                "allowed_domains": None,  # TODO: Support adding allowed domains for the task
                "allow_vision": False,
                **stage_config,  # Override with stage-specific parallelism config
            },
        }

    def _try_create_evaluator(self, task_id: str, version: str):
        """Try to create an evaluator for the task."""
        try:
            from agisdk.REAL.browsergym.webclones.evaluate import (
                WebCloneEvaluator,
            )
            from agisdk.REAL.browsergym.webclones.task_config import TaskConfig

            task_config = TaskConfig(task_id, version)
            evaluator = WebCloneEvaluator(task_config)
            return evaluator
        except Exception as e:
            self.logger.error(f"Failed to create evaluator: {e}", indent=6)
            raise

    def _save_config(self, config: Dict[str, Any], experiment_folder: str) -> str:
        """Write the engine configuration to a file."""
        config_path = Path(experiment_folder) / "engine-config.yaml"
        ensure_parent_dir(config_path)

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return str(config_path)

    def _save_final_state(
        self, experiment_folder: str, final_result: Optional[str] = None, final_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save the final result and/or final state to a file.

        Args:
            experiment_folder: The folder to save the final state to.
            final_result: The final result from the agent.
            final_state: The final state from the environment.
        """
        final_state_path = Path(experiment_folder) / "final_state.json"

        data = {}
        if final_result is not None:
            data["final_result"] = final_result
        if final_state is not None:
            data["final_state"] = final_state

        with open(final_state_path, "w") as f:
            json.dump(data, f, indent=2)
        return str(final_state_path)

    async def run_single_experiment(
        self,
        task: Dict[str, str],
        stage_config: Dict[str, Any],
        stage_name: str,
        run_number: int,
        shared_experiment_id: Optional[str] = None,
        evaluate: bool = False,
    ) -> Optional[ExperimentResult]:
        """Run a single experiment with given configuration."""
        experiment_folder, experiment_id = self._create_experiment_folder(
            task["id"], stage_name, run_number, shared_experiment_id
        )

        self.logger.info(f"Running {stage_name} - Run {run_number}", indent=4)
        self.logger.info(f"Experiment ID: {experiment_id}", indent=4)

        # Create and save engine configuration
        engine_config = self._create_engine_config(stage_config, experiment_folder)
        config_path = self._save_config(engine_config, experiment_folder)
        parallelism_config = stage_config["allow_parallelism"]

        # Initialize result object
        start_time = time.time()

        engine = None
        result = None
        evaluator = None

        try:
            if self.evaluate:
                evaluator = self._try_create_evaluator(task["id"], "custom")

            engine = await Engine.create(config_path=config_path)
            setup_logging(engine.settings, engine._instance_hash)

            result = ExperimentResult(
                experiment_id=experiment_id,
                engine_id=engine._instance_hash,
                task_goal=task["goal"],
                llm_model=engine_config["constraints"]["llm_model"],
                llm_model_mini=engine_config["constraints"]["llm_model_mini"],
                stage_name=stage_name,
                run_number=run_number,
                reported_success=False,
                evaluated_success=None,
                error=None,
                total_time=0.0,
                metrics={},
                final_result=None,
                final_state_path=None,
            )

            # Run the task
            task_result = await engine.run(task["goal"], initial_url=task["initial_url"], mode="block")
            assert isinstance(task_result, AgentHistoryListResponse), "Task result is not an AgentHistoryListResponse"
            assert task_result.is_done(), "Task is not done"

            finish_time = time.time()
            result.total_time = finish_time - start_time
            self.logger.info(f"Finished running task in {result.total_time:.2f} seconds", indent=6)

            metrics = await engine.get_metrics()
            result.metrics = metrics
            result.reported_success = bool(task_result.is_successful())
            result.final_result = task_result.final_result()
            result.final_state_path = self._save_final_state(experiment_folder, final_result=result.final_result)

            if not evaluate:  # If not evaluating, return the result
                return result

            if not "initial_url" in task:
                self.logger.error("Initial URL not found in task config. Unable to evaluate result.", indent=6)
                return result

            task_states = engine.scheduler.tasks
            is_single_task_mode = not parallelism_config.get("task", False)

            if is_single_task_mode:  # For sequential or first-of-n mode, only get the state from the successful task
                self.logger.info("Single task mode: fetching state from one successful task", indent=6)

                successful_task = get_successful_task(parallelism_config, task_states, self.logger)
                if not successful_task:
                    self.logger.error("No successful task found for evaluation", indent=6)
                    return result

                final_state = await fetch_final_state(successful_task, task["initial_url"], self.logger)
                if final_state is None:
                    self.logger.error("Failed to fetch final state for evaluation", indent=6)
                    return result

                final_state_path = self._save_final_state(
                    experiment_folder, final_result=result.final_result, final_state=final_state
                )
                self.logger.info(f"Saved final state to {final_state_path}", indent=6)

            else:  # For task parallelism, merge states from all completed tasks
                self.logger.info(
                    "Task parallelism mode: fetching and merging states from all completed tasks", indent=6
                )
                completed_tasks = get_all_completed_tasks(task_states, self.logger)

                final_states = await asyncio.gather(
                    *[fetch_final_state(task_state, task["initial_url"], self.logger) for task_state in completed_tasks]
                )
                valid_states_dict = {
                    task_state.id: state
                    for task_state, state in zip(completed_tasks, final_states)
                    if state is not None
                }
                final_state = merge_parallel_final_states(valid_states_dict, self.logger)
                if not final_state:
                    self.logger.error("Failed to merge final states for evaluation", indent=6)
                    return result

                final_state_path = self._save_final_state(
                    experiment_folder, final_result=result.final_result, final_state=final_state
                )
                self.logger.info(f"Saved final state to {final_state_path}", indent=6)

            try:
                # Wrap the state in the expected format for the evaluator
                env_state = {"final_state": final_state, "final_result": result.final_result}
                reward, _, message, info = evaluator.evaluate(env_state, result.final_result)
                self.logger.info(
                    f"Evaluation result: {message}, Reward: {reward}",
                    indent=6,
                )
                result.evaluated_success = all(result[0] for result in info["results"])
            except Exception as e:
                self.logger.error(f"Failed to evaluate result: {e}", indent=6)
                result.evaluated_success = False
                result.error = str(e)

        except Exception as e:
            if result is not None:
                result.error = str(e)
            else:
                result = None

        finally:
            if engine:
                await engine.stop()
        self.logger.info(f"Run {run_number} completed in {result.total_time if result else 0:.2f} seconds", indent=4)
        self.logger.info("--------------------------------", indent=2)
        return result

    async def run_experiment(self, experiment: Dict[str, Any]):
        """Run a complete experiment across all stages."""
        tasks = experiment["tasks"]
        stages = experiment["stages"]
        settings = experiment["settings"]

        output_dir = Path(self.config["settings"]["output_dir"])
        ensure_parent_dir(output_dir)

        self.logger.info("--------------------------------")
        self.logger.info(
            f"Running {len(tasks)} tasks across {len(stages)} stages, {settings['runs_per_stage']} runs per stage"
        )
        self.expected_results_count = len(tasks) * len(stages) * settings["runs_per_stage"]
        self.logger.info(f"Expected {self.expected_results_count} results")
        self.logger.info(f"Tasks: {tasks}")
        self.logger.info(f"Stages: {stages}")
        self.logger.info(f"Settings: {settings}")
        self.logger.info("--------------------------------")

        for task in tasks:
            # Run each task across different stages
            self.results_path = (
                Path(self.config["settings"]["output_dir"])
                / f"{task['id']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            )
            self.logger.info("Running task:", indent=2)
            self.logger.info(f"{task}", indent=4)
            self.logger.info(f"Results will be saved to {self.results_path}", indent=4)
            self.logger.info("--------------------------------", indent=2)

            for stage in stages:
                stage_name = stage["name"]
                stage_config = stage["config"]

                # Each experiment is run multiple times to track variance
                # We create a shared experiment ID for all runs of this task and stage
                shared_experiment_id = None
                for run_num in range(1, settings["runs_per_stage"] + 1):
                    result = await self.run_single_experiment(
                        task=task,
                        stage_config=stage_config,
                        stage_name=stage_name,
                        run_number=run_num,
                        shared_experiment_id=shared_experiment_id,
                        evaluate=self.evaluate,
                    )

                    if result is None:
                        self.logger.error(f"Run {run_num} failed: result is None", indent=6)
                        continue

                    if result.error:
                        self.logger.error(f"Run {run_num} failed with error: {result.error}", indent=6)
                        continue

                    # Use the experiment ID from the first run for subsequent runs
                    if shared_experiment_id is None:
                        shared_experiment_id = result.experiment_id

                    self.results.append(result)
                    self.save_results(self.results_path)
                    await asyncio.sleep(2)

            # Clear results after each task
            self.actual_results_count += len(self.results)
            self.results.clear()

        self.logger.info(
            f"Experiment completed with {self.actual_results_count} results out of {self.expected_results_count} expected"
        )
        self.logger.info("--------------------------------")

    def save_results(self, results_path: Path):
        """Save experiment results to JSON."""
        ensure_parent_dir(results_path)
        results_data = [asdict(result) for result in self.results]

        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)


async def main():
    args = parse_args()
    config_path = args.config
    evaluate = args.evaluate
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    runner = ExperimentRunner(config_path, evaluate)
    try:
        await runner.run_experiment(runner.config)
    finally:
        print("Experiment completed. Exiting.")


if __name__ == "__main__":
    asyncio.run(main())
