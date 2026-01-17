"""
Base module for task validation with helpers.

This module provides:
1. LLM-based parsing to convert any text output to structured format
2. Abstract base class for task-specific validation
"""

import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, TypeVar

from pydantic import BaseModel

# Load .env file at module import time
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                # Remove quotes if present
                value = value.strip('"').strip("'")
                os.environ[key] = value

T = TypeVar("T", bound=BaseModel)


async def parse_output_with_llm(output: str, output_schema: type[T]) -> Optional[T]:
    """
    Parse any text output into structured format using LLM.

    Args:
        output: Raw text output from code execution
        output_schema: Pydantic model defining the expected structure

    Returns:
        Parsed output as instance of output_schema, or None if parsing failed
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        from browser_use.llm.messages import UserMessage

        from blastai.agents.llm_factory import LLMFactory

        llm = LLMFactory.create_llm("claude-sonnet-4-20250514", temperature=0.0)

        prompt = f"""Convert the following text to the requested structured JSON format.

Text:
{output}"""

        logger.info(f"Parsing output with LLM (schema: {output_schema.__name__})")
        logger.debug(f"Output to parse (first 500 chars): {output[:500]}")

        messages = [UserMessage(content=prompt)]
        result = await llm.ainvoke(messages, output_format=output_schema)

        restaurants = getattr(result.completion, "restaurants", None)
        logger.info(
            f"LLM parsing succeeded, parsed {len(restaurants) if restaurants else '?'} items"
        )
        return result.completion

    except Exception as e:
        logger.error(f"LLM parsing failed: {e}")
        import traceback

        traceback.print_exc()
        return None


class TaskValidator(ABC):
    """
    Abstract base class for task-specific validation.

    Each task should subclass this and implement:
    1. output_schema: Pydantic model defining expected output structure
    2. check_correctness: Function to validate parsed output against ground truth (boolean)
    3. check_correctness_pct: (Optional) Function to return partial correctness as 0.0-1.0
    """

    @property
    @abstractmethod
    def output_schema(self) -> type[BaseModel]:
        """Return the Pydantic model for this task's expected output."""
        pass

    @abstractmethod
    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """
        Check if the parsed output is correct.

        Args:
            parsed_output: Output parsed into the task's schema

        Returns:
            True if output is completely correct, False otherwise
        """
        pass

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """
        Check partial correctness as a percentage (0.0 to 1.0).

        Default implementation: returns 1.0 if check_correctness is True, else 0.0.
        Subclasses can override to provide more granular scoring.

        Args:
            parsed_output: Output parsed into the task's schema

        Returns:
            Percentage correct (0.0 to 1.0)
        """
        return 1.0 if self.check_correctness(parsed_output) else 0.0

    async def validate(self, output: str, return_pct: bool = False) -> dict:
        """
        Validate output by parsing and checking correctness.

        Args:
            output: Raw text output from code execution
            return_pct: If True, include 'correctness_pct' field

        Returns:
            dict with:
                - success: bool (whether parsing succeeded)
                - correct: bool (whether output is 100% correct)
                - correctness_pct: float (0.0-1.0, only if return_pct=True)
                - parsed_output: parsed data or None
        """
        parsed = await parse_output_with_llm(output, self.output_schema)

        if parsed is None:
            result = {
                "success": False,
                "correct": False,
                "parsed_output": None,
            }
            if return_pct:
                result["correctness_pct"] = 0.0
            return result

        is_correct = self.check_correctness(parsed)
        result = {
            "success": True,
            "correct": is_correct,
            "parsed_output": parsed,
        }

        if return_pct:
            result["correctness_pct"] = self.check_correctness_pct(parsed)

        return result
