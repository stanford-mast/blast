"""
Omnizon Custom Task 6: Coffee Maker Touchscreen Comparison

Task: Which of the following coffee makers has touchscreen: Gevi 10, Gevi 12,
Simply Good Coffee, and BELLA Single Serve? Make sure you visit the product
pages of each of them.

Correct Answer: Gevi 12
"""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import TaskValidator


class OmnizonCustom6Output(BaseModel):
    """Expected output schema for omnizon-custom-6 task"""

    coffee_maker_with_touchscreen: str = Field(
        description="Name of the coffee maker that has a touchscreen"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Optional explanation of the findings"
    )


# Ground truth - Gevi 12 is the one with touchscreen
EXPECTED_ANSWER = "gevi 12"


class OmnizonCustom6Validator(TaskValidator):
    """Validator for Omnizon Custom Task 6"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return OmnizonCustom6Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """Check if the correct coffee maker is identified."""
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, OmnizonCustom6Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        answer = parsed_output.coffee_maker_with_touchscreen.lower().strip()

        # Check if answer contains "gevi 12" or "gevi12"
        is_correct = "gevi 12" in answer or "gevi12" in answer

        logger.info(
            f"check_correctness_pct: expected='{EXPECTED_ANSWER}', "
            f"got='{answer}', correct={is_correct}"
        )

        return 1.0 if is_correct else 0.0

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """Check if the correct coffee maker is identified."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = OmnizonCustom6Validator()
