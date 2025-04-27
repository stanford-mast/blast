"""Task planning for BLAST."""

import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import Constraints

logger = logging.getLogger(__name__)

class Planner:
    """Plans task execution with subtask management."""
    
    def __init__(self, constraints: Optional[Constraints] = None):
        """Initialize planner.
        
        Args:
            constraints: Optional constraints for LLM model selection
        """
        self.constraints = constraints or Constraints()
        self.llm = ChatOpenAI(model=self.constraints.llm_model)
        
        # System prompt for planning
        self.system_prompt = """You are a task planner that generates concise 1-2 sentence plans for web browser tasks.
Focus only on if/how/when to launch subtasks and get their results.
Be specific about URLs or search queries to pass to subtasks.
Keep plans brief and actionable."""
        
    async def plan(self, task_description: str) -> str:
        """Generate a plan for task execution.
        
        Args:
            task_description: Description of the task to plan
            
        Returns:
            Brief plan focusing on subtask management
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Plan this task: {task_description}")
        ]
        
        response = await self.llm.ainvoke(messages)
        plan = response.content.strip()
        
        # Ensure plan is concise
        if len(plan.split('\n')) > 2:
            plan = ' '.join(plan.split('\n')[:2])
            
        logger.debug(f"Generated plan for task: {plan}")
        return plan