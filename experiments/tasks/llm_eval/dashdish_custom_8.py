"""
DashDish Custom Task 8: Light & Fresh Delivery Count

Task: How many restaurants in the "Light & fresh" category offer delivery?

Expected Answer: 5
"""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import TaskValidator


class DashDishCustom8Output(BaseModel):
    """Expected output schema for dashdish-custom-8 task"""

    restaurant_count: int = Field(
        description="Number of restaurants in the 'Light & fresh' category that offer delivery"
    )
    restaurant_names: Optional[list[str]] = Field(
        default=None, description="Optional list of restaurant names"
    )


# Ground truth
EXPECTED_COUNT = 10


class DashDishCustom8Validator(TaskValidator):
    """Validator for DashDish Custom Task 8"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return DashDishCustom8Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """Check if the count is correct."""
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, DashDishCustom8Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        is_correct = parsed_output.restaurant_count == EXPECTED_COUNT
        logger.info(
            f"check_correctness_pct: expected={EXPECTED_COUNT}, "
            f"got={parsed_output.restaurant_count}, correct={is_correct}"
        )

        return 1.0 if is_correct else 0.0

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """Check if the count is correct."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = DashDishCustom8Validator()
