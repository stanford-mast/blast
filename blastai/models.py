"""Model-related utilities for BLAST."""

from dataclasses import dataclass

def is_openai_model(model_name: str) -> bool:
    """Check if a model name is from OpenAI.
    
    Args:
        model_name: Name of the model (with optional provider prefix)
        
    Returns:
        True if model is from OpenAI, False otherwise
    """
    model_name = model_name.lower()
    
    # Handle provider prefix (e.g., "openai:gpt-4")
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
        if provider == "openai":
            return True
        return False
        
    # Check for OpenAI model patterns
    return any(prefix in model_name for prefix in ['gpt-3', 'gpt-4', 'o1', 'o1-mini', 'o3', 'o3-mini', 'o4', 'o4-mini'])

@dataclass
class TokenUsage:
    """Dataclass for detailed token usage tracking."""
    prompt: int = 0
    prompt_cached: int = 0
    completion: int = 0
    total: int = 0
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add two TokenUsage instances."""
        return TokenUsage(
            prompt=self.prompt + other.prompt,
            prompt_cached=self.prompt_cached + other.prompt_cached,
            completion=self.completion + other.completion,
            total=self.total + other.total
        )
    
    def __iadd__(self, other: 'TokenUsage') -> 'TokenUsage':
        """In-place addition of TokenUsage instances."""
        self.prompt += other.prompt
        self.prompt_cached += other.prompt_cached
        self.completion += other.completion
        self.total += other.total
        return self
    
    def __sub__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Subtract two TokenUsage instances."""
        return TokenUsage(
            prompt=self.prompt - other.prompt,
            prompt_cached=self.prompt_cached - other.prompt_cached,
            completion=self.completion - other.completion,
            total=self.total - other.total
        )
    
    def __str__(self) -> str:
        """String representation using compact format."""
        return self.format_compact()
    
    def __repr__(self) -> str:
        """Repr showing all fields."""
        return f"TokenUsage(prompt={self.prompt}, prompt_cached={self.prompt_cached}, completion={self.completion}, total={self.total})"
    
    def copy(self) -> 'TokenUsage':
        """Create a copy of this TokenUsage instance."""
        return TokenUsage(
            prompt=self.prompt,
            prompt_cached=self.prompt_cached,
            completion=self.completion,
            total=self.total
        )
    
    @classmethod
    def from_tuple(cls, usage_tuple: tuple) -> 'TokenUsage':
        """Create TokenUsage from a (prompt, prompt_cached, completion, total) tuple."""
        if len(usage_tuple) != 4:
            raise ValueError(f"Expected 4-element tuple, got {len(usage_tuple)}")
        return cls(
            prompt=usage_tuple[0],
            prompt_cached=usage_tuple[1], 
            completion=usage_tuple[2],
            total=usage_tuple[3]
        )
    
    def to_tuple(self) -> tuple:
        """Convert to a (prompt, prompt_cached, completion, total) tuple."""
        return (self.prompt, self.prompt_cached, self.completion, self.total)

    @classmethod
    def from_json(cls, json_data: dict) -> 'TokenUsage':
        """Create TokenUsage from JSON."""
        return cls(
            prompt=json_data["prompt"],
            prompt_cached=json_data["prompt_cached"],
            completion=json_data["completion"],
            total=json_data["total"]
        )
    
    def to_json(self) -> dict:
        """Convert to JSON."""
        return {
            "prompt": self.prompt,
            "prompt_cached": self.prompt_cached,
            "completion": self.completion,
            "total": self.total
        }
    
    def format_detailed(self) -> str:
        """Format with detailed breakdown (showing Total, Prompt, Cached, Output)."""
        return f"{self.total:,} (Prompt: {self.prompt:,}, Cached: {self.prompt_cached:,}, Output: {self.completion:,})"
    
    def format_compact(self) -> str:
        """Format compactly (showing Total, Prompt, Cached, Output)."""
        if self.prompt_cached > 0:
            return f"{self.total:,} ({self.prompt:,}+{self.prompt_cached:,}c+{self.completion:,})"
        else:
            return f"{self.total:,} ({self.prompt:,}+{self.completion:,})"

    def format_minimal(self) -> str:
        """Format minimally (only showing Total)."""
        return f"{self.total:,}"
