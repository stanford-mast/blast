"""Logger for experiments."""

import logging
from pathlib import Path

from .utils import ensure_parent_dir


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

    def warning(self, message: str, indent: int = 0):
        """Log warning message."""
        self.logger.warning(f"{' ' * indent}{message}")

    def error(self, message: str, indent: int = 0):
        """Log error message."""
        self.logger.error(f"{' ' * indent}{message}")
