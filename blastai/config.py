"""Configuration classes for BLAST."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Settings(BaseModel):
    """Settings for BLAST configuration."""

    local_browser_path: str = Field(
        default="none",
        description="Path to local Chrome/Chromium browser binary. Set to 'auto' to auto-detect, 'none' to use default, or provide specific path",
    )

    browser_cdp_url: Optional[str] = Field(
        default=None,
        description="HTTP(S) or WebSocket CDP URL for an existing local or remote Chromium browser",
    )

    persist_cache: bool = Field(default=False, description="Whether to persist cache between runs")

    browser_use_log_level: str = Field(default="error", description="Logging level for browser-use module")

    blastai_log_level: str = Field(default="error", description="Logging level for blastai module")

    secrets_file_path: Optional[str] = Field(
        default="secrets.env", description="Path to secrets file containing sensitive data"
    )

    logs_dir: str = Field(
        default="blast-logs", description="Directory for log files (defaults to blast-logs if not specified)"
    )

    server_port: int = Field(default=8000, description="Port number for the BLAST server")

    web_port: int = Field(default=3000, description="Port number for the web UI")

    @field_validator("browser_cdp_url")
    @classmethod
    def validate_browser_cdp_url(cls, value: Optional[str]) -> Optional[str]:
        """Only accept URL forms supported by Browser Use's CDP connector."""
        if value is None:
            return None

        value = value.strip()
        if not value.startswith(("http://", "https://", "ws://", "wss://")):
            raise ValueError("browser_cdp_url must start with http://, https://, ws://, or wss://")
        return value

    @model_validator(mode="after")
    def validate_browser_source(self):
        """A session cannot launch a local executable and connect over CDP."""
        if self.browser_cdp_url and self.local_browser_path != "none":
            raise ValueError("browser_cdp_url and local_browser_path cannot be set together")
        return self

    @classmethod
    def create(cls, **kwargs):
        """Create Settings from keyword arguments."""
        return cls(**kwargs)


class Constraints(BaseModel):
    """Constraints for BLAST execution."""

    max_memory: Optional[int] = Field(default=None, description="Maximum memory usage in bytes")

    max_concurrent_browsers: int = Field(default=20, description="Maximum number of concurrent browser contexts")

    max_cost_per_minute: Optional[float] = Field(default=None, description="Maximum cost per minute in USD")

    max_cost_per_hour: Optional[float] = Field(default=None, description="Maximum cost per hour in USD")

    allow_parallelism: dict = Field(
        default={
            "task": True,  # Enable task-level parallelism for subtasks
            "data": True,  # Enable data-level parallelism for content extraction
            "first_of_n": False,  # Default disable first-result parallelism
        },
        description="Types of parallelism to allow: task (subtasks), data (content extraction), first_of_n (first result)",
    )

    first_of_n_num_copies: int = Field(
        default=3, description="Number of copies to launch for first_of_n parallelism strategy"
    )

    max_parallelism_nesting_depth: int = Field(default=1, description="Maximum depth of nested parallel tasks")

    llm_model: str = Field(default="gpt-4.1", description="Primary LLM model to use")

    llm_model_mini: str = Field(default="gpt-4.1-mini", description="Smaller LLM model to use for parallel processing")

    allow_vision: bool = Field(default=True, description="Whether to allow vision capabilities")

    require_headless: bool = Field(default=True, description="Whether to require headless browser mode")

    require_patchright: bool = Field(
        default=True, description="Whether to use patchright for stealth browser automation"
    )

    require_human_in_loop: bool = Field(
        default=False,
        description="Whether to enable human-in-loop capabilities including VNC access and human assistance (requires vncserver installed)",
    )

    share_browser_process: bool = Field(default=True, description="Whether to share browser process between requests")

    allowed_domains: Optional[list[str]] = Field(
        default=None, description="List of allowed domains for browser navigation. None means all domains are allowed."
    )

    @classmethod
    def create(cls, max_memory: Optional[str] = None, **kwargs):
        """Create Constraints from keyword arguments.

        Args:
            max_memory: Maximum memory as string (e.g. "4GB")
            **kwargs: Additional keyword arguments matching field names

        Returns:
            Constraints instance
        """
        # Convert memory string to bytes if provided
        memory_bytes = None
        if max_memory:
            units = {"B": 1, "KB": 1024, "MB": 1024 * 1024, "GB": 1024 * 1024 * 1024, "TB": 1024 * 1024 * 1024 * 1024}

            # Extract number and unit
            import re

            match = re.match(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", max_memory)
            if match:
                number = float(match.group(1))
                unit = match.group(2).upper()
                if unit in units:
                    memory_bytes = int(number * units[unit])
                else:
                    raise ValueError(f"Invalid memory unit: {unit}")
            else:
                raise ValueError(f"Invalid memory format: {max_memory}")

        # Update kwargs with converted memory
        if memory_bytes is not None:
            kwargs["max_memory"] = memory_bytes

        return cls(**kwargs)
