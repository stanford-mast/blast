from __future__ import annotations

from typing import List, Literal, Optional, Union

from browser_use.agent.views import ActionResult, AgentHistory, AgentHistoryList
from browser_use.browser.views import BrowserStateHistory
from pydantic import BaseModel, Field


class AgentHistoryListResponse(AgentHistoryList):
    """List of agent history items with task ID"""

    task_id: str

    @classmethod
    def from_history(cls, history: Union[AgentHistoryList, List], task_id: str) -> AgentHistoryListResponse:
        """Convert an AgentHistoryList to AgentHistoryListResponse by adding task_id"""
        # Create new instance with all fields from history plus task_id
        if isinstance(history, list):
            # Check if it's a list of ActionResult objects
            if history and isinstance(history[0], ActionResult):
                # Convert ActionResult list to proper format
                agent_history = AgentHistory(
                    model_output=None,
                    result=history,
                    state=BrowserStateHistory(
                        url=None, title=None, tabs=[], screenshot=None, interacted_element=[None]
                    ),
                )
                return cls(history=[agent_history], task_id=task_id)
            # Handle case where history is a list instead of AgentHistoryList
            return cls(history=history, task_id=task_id)
        else:
            # Normal case where history is an AgentHistoryList object
            return cls(history=history.history, task_id=task_id)


class AgentReasoning(BaseModel):
    """Agent reasoning information including task ID, type, and content"""

    task_id: str
    type: Literal["screenshot", "thought"]
    thought_type: Literal["memory", "goal"] | None = None  # Only required when type is "thought"
    content: str
    live_url: Optional[str] = None


class AgentScheduled(BaseModel):
    """Notification that an agent task has been scheduled"""

    task_id: str
    description: str


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
