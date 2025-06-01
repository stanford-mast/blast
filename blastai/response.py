from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from browser_use.agent.views import AgentHistoryList


class AgentHistoryListResponse(AgentHistoryList):
    """List of agent history items with task ID"""
    
    task_id: str

    @classmethod
    def from_history(cls, history: AgentHistoryList, task_id: str) -> AgentHistoryListResponse:
        """Convert an AgentHistoryList to AgentHistoryListResponse by adding task_id"""
        # Create new instance with all fields from history plus task_id
        return cls(history=history.history, task_id=task_id)


class AgentReasoning(BaseModel):
    """Agent reasoning information including task ID, type, and content"""
    
    task_id: str
    type: Literal["screenshot", "thought"]
    thought_type: Literal["memory", "goal"] | None = None  # Only required when type is "thought"
    content: str
    live_url: Optional[str] = None


class HumanRequest(BaseModel):
    """Request for human assistance"""
    task_id: str
    prompt: str
    allow_takeover: bool = False
    live_url: Optional[str] = None


class HumanResponse(BaseModel):
    """Response from human assistance"""
    task_id: str
    response: str


class StopRequest(BaseModel):
    """Request to stop a task and all its dependencies"""
    type: Literal["stop"] = "stop"