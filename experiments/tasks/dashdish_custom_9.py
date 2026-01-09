"""
DashDish Custom Task 9: Cheapest Fries Comparison

Task: Tell me which of Wingstop, Man vs. Fries, McDonald's, Popeyes, and Ike's
has the cheapest fries item and its price.

Expected Answer: Popeyes with fries at $3.49
"""

from typing import Optional

from pydantic import BaseModel, Field

from experiments.tasks.base import TaskValidator


class DashDishCustom9Output(BaseModel):
    """Expected output schema for dashdish-custom-9 task"""

    restaurant_name: str = Field(
        description="Name of the restaurant with the cheapest fries"
    )
    fries_item_name: Optional[str] = Field(
        default=None, description="Name of the fries item"
    )
    price: float = Field(description="Price of the fries item in dollars")


# Ground truth
EXPECTED_RESTAURANT = "Popeyes"
EXPECTED_PRICE = 3.49


def normalize_name(name: str) -> str:
    """Normalize name for comparison."""
    name = name.lower().strip()
    # Handle variations
    if "popeye" in name:
        return "popeyes"
    return name


class DashDishCustom9Validator(TaskValidator):
    """Validator for DashDish Custom Task 9"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return DashDishCustom9Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """
        Check correctness as percentage.

        50% for correct restaurant, 50% for correct price.
        """
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, DashDishCustom9Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        score = 0.0

        # Check restaurant name
        restaurant_correct = (
            normalize_name(parsed_output.restaurant_name) == normalize_name(EXPECTED_RESTAURANT)
        )
        if restaurant_correct:
            score += 0.5
            logger.debug(f"Restaurant correct: {parsed_output.restaurant_name}")
        else:
            logger.debug(
                f"Restaurant mismatch: expected '{EXPECTED_RESTAURANT}', got '{parsed_output.restaurant_name}'"
            )

        # Check price (with tolerance)
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
        """Check if both restaurant and price are correct."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = DashDishCustom9Validator()
