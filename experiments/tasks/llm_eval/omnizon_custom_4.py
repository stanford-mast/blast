"""
Omnizon Custom Task 4: Order Status Count

Task: How many orders have not yet shipped? How many cancelled?

Correct Answer:
- Not yet shipped: 8
- Cancelled: 2
"""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import TaskValidator


class OmnizonCustom4Output(BaseModel):
    """Expected output schema for omnizon-custom-4 task"""

    not_yet_shipped_count: int = Field(
        description="Number of orders that have not yet shipped"
    )
    cancelled_count: int = Field(
        description="Number of cancelled orders"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Optional explanation of the findings"
    )


# Ground truth
EXPECTED_NOT_SHIPPED = 8
EXPECTED_CANCELLED = 2


class OmnizonCustom4Validator(TaskValidator):
    """Validator for Omnizon Custom Task 4"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return OmnizonCustom4Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """Check if the order counts are correct. Partial credit for each."""
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, OmnizonCustom4Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        score = 0.0
        
        # Check not yet shipped count (50%)
        if parsed_output.not_yet_shipped_count == EXPECTED_NOT_SHIPPED:
            score += 0.5
            logger.info(f"✓ Not yet shipped: {parsed_output.not_yet_shipped_count}")
        else:
            logger.info(
                f"✗ Not yet shipped: got {parsed_output.not_yet_shipped_count}, "
                f"expected {EXPECTED_NOT_SHIPPED}"
            )

        # Check cancelled count (50%)
        if parsed_output.cancelled_count == EXPECTED_CANCELLED:
            score += 0.5
            logger.info(f"✓ Cancelled: {parsed_output.cancelled_count}")
        else:
            logger.info(
                f"✗ Cancelled: got {parsed_output.cancelled_count}, "
                f"expected {EXPECTED_CANCELLED}"
            )

        return score

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """Check if both counts are correct."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = OmnizonCustom4Validator()
