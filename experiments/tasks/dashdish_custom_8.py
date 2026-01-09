"""
DashDish Custom Task 8: RT Rotisserie Pickup Time

Task: How long does it take to pick up an order from RT Rotisserie?

Expected Answer: 17 minutes
"""

from typing import Optional

from pydantic import BaseModel, Field

from experiments.tasks.base import TaskValidator


class DashDishCustom8Output(BaseModel):
    """Expected output schema for dashdish-custom-8 task"""

    pickup_time_minutes: int = Field(
        description="Pickup time in minutes"
    )
    time_range: Optional[str] = Field(
        default=None, description="Optional time range if provided (e.g., '15-20 minutes')"
    )


# Ground truth
EXPECTED_TIME_MINUTES = 17


class DashDishCustom8Validator(TaskValidator):
    """Validator for DashDish Custom Task 8"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return DashDishCustom8Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """Check if the pickup time is correct."""
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, DashDishCustom8Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        is_correct = parsed_output.pickup_time_minutes == EXPECTED_TIME_MINUTES
        logger.info(
            f"check_correctness_pct: expected={EXPECTED_TIME_MINUTES}, "
            f"got={parsed_output.pickup_time_minutes}, correct={is_correct}"
        )

        return 1.0 if is_correct else 0.0

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """Check if the time is correct."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = DashDishCustom8Validator()
