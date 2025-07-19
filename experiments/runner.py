import asyncio
import json
import logging
import os
import hashlib
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from blastai import Engine
from blastai.logging_setup import setup_logging
from blastai.response import AgentHistoryListResponse


def ensure_parent_dir(file_path: str | Path) -> None:
    """Ensure the parent directory of a file path exists."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentResult:
    """ExperimentResult stores the results from a single experiment run."""

    experiment_id: str
    engine_id: str
    stage_name: str
    run_number: int
    success: bool
    error: Optional[str]
    total_time: float
    metrics: Dict[str, Any]
    final_result: Optional[str]


class ExperimentLogger:
    """ExperimentLogger sets up logging for experiments. This is separate from the engine logging."""

    def __init__(self, experiment_folder: str):
        self.experiment_folder = experiment_folder
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging for a single experiment run."""
        logger = logging.getLogger("blastai-experiment-runner")
        logger.setLevel(logging.DEBUG)

        logger.handlers.clear()

        log_file = Path(self.experiment_folder) / f"{logger.name}.log"
        ensure_parent_dir(log_file)
        file_handler = logging.FileHandler(log_file, mode="w")
        logger.addHandler(file_handler)

        return logger

    def info(self, message: str, indent: int = 0):
        """Log info message."""
        self.logger.info(f"{' ' * indent}{message}")

    def error(self, message: str, indent: int = 0):
        """Log error message."""
        self.logger.error(f"{' ' * indent}{message}")


class ExperimentRunner:
    def __init__(self, config_path: str = "experiments/experiment-config.yaml"):
        self.config_path = config_path
        self.results: List[ExperimentResult] = []
        self.load_config()

        self.logger = ExperimentLogger(self.config["settings"]["logs_dir"])
        self.results_count = 0

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
            Path(self.config["settings"]["logs_dir"])
            / task_id
            / stage_name
            / experiment_id
            / f"run_{run_number}"
        )
        ensure_parent_dir(experiment_folder)
        return str(experiment_folder), experiment_id

    def _create_engine_config(
        self, stage_config: Dict[str, Any], experiment_folder: str
    ) -> Dict[str, Any]:
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
                "allowed_domains": None,
                "allow_vision": False,
                **stage_config,  # Override with stage-specific parallelism config
            },
        }

    def _save_config(self, config: Dict[str, Any], experiment_folder: str) -> str:
        """Write the engine configuration to a file."""
        config_path = Path(experiment_folder) / "engine-config.yaml"
        ensure_parent_dir(config_path)

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return str(config_path)

    async def run_single_experiment(
        self,
        task: Dict[str, str],
        stage_config: Dict[str, Any],
        stage_name: str,
        run_number: int,
        shared_experiment_id: Optional[str] = None,
    ) -> ExperimentResult:
        """Run a single experiment with given configuration."""
        experiment_folder, experiment_id = self._create_experiment_folder(
            task["id"], stage_name, run_number, shared_experiment_id
        )

        self.logger.info(f"Running {stage_name} - Run {run_number}", indent=4)
        self.logger.info(f"Experiment ID: {experiment_id}", indent=4)

        # Create and save engine configuration
        engine_config = self._create_engine_config(stage_config, experiment_folder)
        config_path = self._save_config(engine_config, experiment_folder)

        # Initialize result object
        start_time = time.time()

        engine = None
        try:
            engine = await Engine.create(config_path=config_path)
            setup_logging(engine.settings, engine._instance_hash)

            result = ExperimentResult(
                experiment_id=experiment_id,
                engine_id=engine._instance_hash,
                stage_name=stage_name,
                run_number=run_number,
                success=False,
                error=None,
                total_time=0.0,
                metrics={},
                final_result=None,
            )

            task_result = await engine.run(
                task["goal"], initial_url=task["initial_url"], mode="block"
            )

            metrics = await engine.get_metrics()

            if isinstance(task_result, AgentHistoryListResponse):
                result.success = (
                    bool(task_result.is_successful())
                    if task_result.is_done()
                    else False
                )
                result.metrics = metrics
                result.final_result = (
                    task_result.final_result() if task_result.is_done() else None
                )
            else:
                self.logger.error(f"Unexpected result type: {type(task_result)}")
                result.error = f"Unexpected result type: {type(task_result)}"

        except Exception as e:
            self.logger.error(f"Run {run_number} failed with error: {e}")
            result.error = str(e)

        finally:
            result.total_time = time.time() - start_time

            if engine:
                await engine.stop()
        self.logger.info(f"Run {run_number} completed in {result.total_time:.2f} seconds", indent=4)
        self.logger.info(f"Result: {result}", indent=4)
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
        self.logger.info(f"Running {len(tasks)} tasks across {len(stages)} stages, {settings['runs_per_stage']} runs per stage")
        self.logger.info(f"Expected {len(tasks) * len(stages) * settings['runs_per_stage']} results")
        self.logger.info(f"Tasks: {tasks}")
        self.logger.info(f"Stages: {stages}")
        self.logger.info(f"Settings: {settings}")
        self.logger.info("--------------------------------")

        for task in tasks:
            # Run each task across different stages
            self.results_path = Path(self.config["settings"]["output_dir"]) / f"{task['id']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            self.logger.info("Running task:", indent=2)
            self.logger.info(f"{task}", indent=4)
            self.logger.info(f"Results will be saved to {self.results_path}")
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
                    )

                    # Use the experiment ID from the first run for subsequent runs
                    if shared_experiment_id is None:
                        shared_experiment_id = result.experiment_id

                    self.results.append(result)
                    self.save_results(self.results_path)
                    await asyncio.sleep(2)

            # Clear results after each task
            self.results_count += len(self.results)
            self.results.clear()
        
        self.logger.info(f"Experiment completed with {self.results_count} results")
        self.logger.info("--------------------------------")

    def save_results(self, results_path: Path):
        """Save experiment results to JSON."""
        ensure_parent_dir(results_path)
        results_data = [asdict(result) for result in self.results]

        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)


async def main():
    config_path = "experiments/configs/test_first_of_n.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    runner = ExperimentRunner(config_path)
    try:
        await runner.run_experiment(runner.config)
    finally:
        print("Experiment completed. Exiting.")


if __name__ == "__main__":
    asyncio.run(main())
