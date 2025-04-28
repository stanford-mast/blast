"""Configuration classes for BLAST."""

from typing import Optional
from pydantic import BaseModel, Field

class Settings(BaseModel):
    """Settings for BLAST configuration."""
    
    persist_cache: bool = Field(
        default=False,
        description="Whether to persist cache between runs"
    )
    
    browser_use_log_level: str = Field(
        default="error",
        description="Logging level for browser-use module"
    )
    
    blastai_log_level: str = Field(
        default="error",
        description="Logging level for blastai module"
    )

class Constraints(BaseModel):
    """Constraints for BLAST execution."""
    
    max_memory: Optional[int] = Field(
        default=None,
        description="Maximum memory usage in bytes"
    )
    
    max_concurrent_browsers: int = Field(
        default=20,
        description="Maximum number of concurrent browser instances"
    )
    
    max_cost_per_minute: Optional[float] = Field(
        default=None,
        description="Maximum cost per minute in USD"
    )
    
    max_cost_per_hour: Optional[float] = Field(
        default=None,
        description="Maximum cost per hour in USD"
    )
    
    allow_parallelism: dict = Field(
        default={
            "task": False,
            "data": False,
            "first_of_n": True
        },
        description="Types of parallelism to allow: task (subtasks), data (content extraction), first_of_n (first result)"
    )
    
    max_parallelism_nesting_depth: int = Field(
        default=1,
        description="Maximum depth of nested parallel tasks"
    )
    
    llm_model: str = Field(
        default="gpt-4.1",
        description="LLM model to use"
    )
    
    allow_vision: bool = Field(
        default=True,
        description="Whether to allow vision capabilities"
    )
    
    require_headless: bool = Field(
        default=True,
        description="Whether to require headless browser mode"
    )
    
    @classmethod
    def create(cls, max_memory: Optional[str] = None,
                   max_concurrent_browsers: int = 20,
                   max_cost_per_minute: Optional[float] = None,
                   max_cost_per_hour: Optional[float] = None,
                   allow_parallelism: dict = {"task": False, "data": False, "first_of_n": True},
                   max_parallelism_nesting_depth: int = 1,
                   llm_model: str = "gpt-4.1",
                   allow_vision: bool = True,
                   require_headless: bool = True) -> "Constraints":
        """Create Constraints from string values.
        
        Args:
            max_memory: Maximum memory as string (e.g. "4GB")
            max_concurrent_browsers: Maximum concurrent browsers
            max_cost_per_minute: Maximum cost per minute in USD
            max_cost_per_hour: Maximum cost per hour in USD
            allow_parallelism: Whether to allow parallel execution
            max_parallelism_nesting_depth: Maximum depth of nested parallel tasks
            llm_model: LLM model identifier
            allow_vision: Whether to allow vision capabilities
            require_headless: Whether to require headless mode
            
        Returns:
            Constraints instance
        """
        # Convert memory string to bytes if provided
        memory_bytes = None
        if max_memory:
            units = {
                'B': 1,
                'KB': 1024,
                'MB': 1024 * 1024,
                'GB': 1024 * 1024 * 1024,
                'TB': 1024 * 1024 * 1024 * 1024
            }
            
            # Extract number and unit
            import re
            match = re.match(r'(\d+(?:\.\d+)?)\s*([A-Za-z]+)', max_memory)
            if match:
                number = float(match.group(1))
                unit = match.group(2).upper()
                if unit in units:
                    memory_bytes = int(number * units[unit])
                else:
                    raise ValueError(f"Invalid memory unit: {unit}")
            else:
                raise ValueError(f"Invalid memory format: {max_memory}")
                
        return cls(
            max_memory=memory_bytes,
            max_concurrent_browsers=max_concurrent_browsers,
            max_cost_per_minute=max_cost_per_minute,
            max_cost_per_hour=max_cost_per_hour,
            allow_parallelism=allow_parallelism,
            max_parallelism_nesting_depth=max_parallelism_nesting_depth,
            llm_model=llm_model,
            allow_vision=allow_vision,
            require_headless=require_headless
        )