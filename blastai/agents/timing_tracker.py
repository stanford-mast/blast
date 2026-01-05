"""
Comprehensive timing tracker for agent execution.

Tracks:
- Planning time (code generation)
- Execution time (running the plan)
- LLM inference breakdown (prefill vs decode)
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional
from threading import Lock


@dataclass
class LLMCallTiming:
    """Timing for a single LLM call."""
    total_seconds: float
    prefill_seconds: Optional[float] = None  # time_to_first_token
    decode_seconds: Optional[float] = None   # generation_seconds
    tokens: Optional[int] = None


@dataclass
class ExecutionTiming:
    """Comprehensive timing for an execution."""
    planning_seconds: float = 0.0  # Time spent generating code (if code mode)
    execution_seconds: float = 0.0  # Time spent executing (code or loop)
    
    # LLM timing breakdown during execution (not planning)
    llm_calls: List[LLMCallTiming] = field(default_factory=list)
    
    @property
    def total_llm_seconds(self) -> float:
        """Total LLM time during execution."""
        return sum(call.total_seconds for call in self.llm_calls)
    
    @property
    def total_prefill_seconds(self) -> float:
        """Total prefill time during execution."""
        return sum(call.prefill_seconds or 0.0 for call in self.llm_calls)
    
    @property
    def total_decode_seconds(self) -> float:
        """Total decode time during execution."""
        return sum(call.decode_seconds or 0.0 for call in self.llm_calls)
    
    @property
    def total_seconds(self) -> float:
        """Total time (planning + execution)."""
        return self.planning_seconds + self.execution_seconds
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'planning_seconds': self.planning_seconds,
            'execution_seconds': self.execution_seconds,
            'total_seconds': self.total_seconds,
            'llm_total_seconds': self.total_llm_seconds,
            'llm_prefill_seconds': self.total_prefill_seconds,
            'llm_decode_seconds': self.total_decode_seconds,
            'num_llm_calls': len(self.llm_calls),
        }


class TimingTracker:
    """Thread-safe tracker for execution timing."""
    
    def __init__(self):
        self._timing = ExecutionTiming()
        self._lock = Lock()
        self._planning_start: Optional[float] = None
        self._execution_start: Optional[float] = None
    
    def start_planning(self):
        """Mark start of planning phase."""
        with self._lock:
            self._planning_start = time.time()
    
    def end_planning(self):
        """Mark end of planning phase."""
        with self._lock:
            if self._planning_start is not None:
                self._timing.planning_seconds = time.time() - self._planning_start
                self._planning_start = None
    
    def start_execution(self):
        """Mark start of execution phase."""
        with self._lock:
            self._execution_start = time.time()
    
    def end_execution(self):
        """Mark end of execution phase."""
        with self._lock:
            if self._execution_start is not None:
                self._timing.execution_seconds = time.time() - self._execution_start
                self._execution_start = None
    
    def record_llm_call(self, total_seconds: float, prefill_seconds: Optional[float] = None,
                       decode_seconds: Optional[float] = None, tokens: Optional[int] = None):
        """Record an LLM call during execution."""
        with self._lock:
            call = LLMCallTiming(
                total_seconds=total_seconds,
                prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,
                tokens=tokens
            )
            self._timing.llm_calls.append(call)
    
    def get_timing(self) -> ExecutionTiming:
        """Get current timing snapshot."""
        with self._lock:
            return ExecutionTiming(
                planning_seconds=self._timing.planning_seconds,
                execution_seconds=self._timing.execution_seconds,
                llm_calls=self._timing.llm_calls.copy()
            )
    
    def reset(self):
        """Reset all timing."""
        with self._lock:
            self._timing = ExecutionTiming()
            self._planning_start = None
            self._execution_start = None


__all__ = ['TimingTracker', 'ExecutionTiming', 'LLMCallTiming']

# Module-level current tracker (used by lower-level LLM streaming helpers)
_CURRENT_TRACKER: Optional[TimingTracker] = None


def set_current_tracker(tracker: Optional[TimingTracker]):
    """Set the module-level current TimingTracker instance (or None to clear)."""
    global _CURRENT_TRACKER
    _CURRENT_TRACKER = tracker


def get_current_tracker() -> Optional[TimingTracker]:
    """Return the current TimingTracker instance, or None if not set."""
    return _CURRENT_TRACKER
