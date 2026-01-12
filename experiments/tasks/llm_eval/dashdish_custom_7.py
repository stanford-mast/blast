"""
DashDish Custom Task 7: Petco Most Expensive Item

Task: What's the most expensive item from Petco?

Expected Answer: Fish Tank Starter Kit at $89.99
"""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import TaskValidator


class DashDishCustom7Output(BaseModel):
    """Expected output schema for dashdish-custom-7 task"""

    item_name: str = Field(description="Name of the most expensive item")
    price: Optional[float] = Field(
        default=None, description="Price of the item in dollars"
    )


# Ground truth
EXPECTED_ITEM_NAME = "Fish Tank Starter Kit"
EXPECTED_PRICE = 89.99


def normalize_name(name: str) -> str:
    """Normalize item name for comparison."""
    return name.lower().strip()


class DashDishCustom7Validator(TaskValidator):
    """Validator for DashDish Custom Task 7"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return DashDishCustom7Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """
        Check correctness as percentage.

        50% for correct item name, 50% for correct price.
        """
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, DashDishCustom7Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        score = 0.0

        # Check item name (fuzzy match)
        name_correct = (
            normalize_name(EXPECTED_ITEM_NAME) in normalize_name(parsed_output.item_name)
            or normalize_name(parsed_output.item_name) in normalize_name(EXPECTED_ITEM_NAME)
        )
        if name_correct:
            score += 0.5
            logger.debug(f"Item name correct: {parsed_output.item_name}")
        else:
            logger.debug(
                f"Item name mismatch: expected '{EXPECTED_ITEM_NAME}', got '{parsed_output.item_name}'"
            )

        # Check price (with tolerance)
        if parsed_output.price is not None:
            price_correct = abs(parsed_output.price - EXPECTED_PRICE) < 0.01
            if price_correct:
                score += 0.5
                logger.debug(f"Price correct: ${parsed_output.price}")
            else:
                logger.debug(
                    f"Price mismatch: expected ${EXPECTED_PRICE}, got ${parsed_output.price}"
                )

        logger.info(f"check_correctness_pct: {score:.0%}")
        return score

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """Check if both name and price are correct."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = DashDishCustom7Validator()
