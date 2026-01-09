"""
DashDish Custom Task 4: Honey BBQ Wings Customization

Task: Can Honey BBQ Wings from Wingstop be ordered with sauce on the side?

Expected Answer: Yes
"""

from typing import Optional

from pydantic import BaseModel, Field

from experiments.tasks.base import TaskValidator


class DashDishCustom4Output(BaseModel):
    """Expected output schema for dashdish-custom-4 task"""

    can_order_sauce_on_side: bool = Field(
        description="Whether Honey BBQ Wings can be ordered with sauce on the side"
    )
    explanation: Optional[str] = Field(
        default=None, description="Optional explanation of the answer"
    )


# Ground truth
EXPECTED_ANSWER = True  # Yes, you can order sauce on the side


class DashDishCustom4Validator(TaskValidator):
    """Validator for DashDish Custom Task 4"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return DashDishCustom4Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """Check if the boolean answer is correct."""
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, DashDishCustom4Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        is_correct = parsed_output.can_order_sauce_on_side == EXPECTED_ANSWER
        logger.info(
            f"check_correctness_pct: expected={EXPECTED_ANSWER}, "
            f"got={parsed_output.can_order_sauce_on_side}, correct={is_correct}"
        )

        return 1.0 if is_correct else 0.0

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """Check if the answer is correct."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = DashDishCustom4Validator()
