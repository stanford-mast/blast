"""
DashDish Custom Task 6: Best Near You Review Count

Task: Visit the first five restaurants in "Best near you" and report how many
of them have under 20 reviews.

Expected Answer: 4 restaurants have under 20 reviews
"""

from typing import Optional

from pydantic import BaseModel, Field

from experiments.tasks.base import TaskValidator


class DashDishCustom6Output(BaseModel):
    """Expected output schema for dashdish-custom-6 task"""

    count_under_20_reviews: int = Field(
        description="Number of restaurants with under 20 reviews"
    )
    restaurant_details: Optional[list[dict]] = Field(
        default=None, description="Optional details about each restaurant visited"
    )


# Ground truth
EXPECTED_COUNT = 4


class DashDishCustom6Validator(TaskValidator):
    """Validator for DashDish Custom Task 6"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return DashDishCustom6Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """Check if the count is correct."""
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, DashDishCustom6Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        is_correct = parsed_output.count_under_20_reviews == EXPECTED_COUNT
        logger.info(
            f"check_correctness_pct: expected={EXPECTED_COUNT}, "
            f"got={parsed_output.count_under_20_reviews}, correct={is_correct}"
        )

        return 1.0 if is_correct else 0.0

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """Check if the count is correct."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = DashDishCustom6Validator()
