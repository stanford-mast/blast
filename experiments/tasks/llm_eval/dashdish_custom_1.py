"""
DashDish Custom Task 1: Stonemill Matcha Price Lookup

Task: How much is a matcha croissant and a regular iced matcha latte from
stonemill matcha? What about a ceremonial grade matcha latte and a mochi set?

Expected Answer:
- Individual prices: Matcha Croissant: $4.49, Regular Iced Matcha Latte: $6.99,
  Ceremonial Grade Matcha Latte: $18.08, Mochi Set (6 pieces): $17.80
- OR Bundle totals: First (Matcha Croissant + Regular Iced Matcha Latte): $11.48,
  Second (Ceremonial Grade Matcha Latte + Mochi Set): $35.88
"""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import TaskValidator


class ItemPrice(BaseModel):
    """Price information for a single menu item"""

    item_name: str = Field(description="Name of the menu item")
    price: float = Field(description="Price of the item in dollars")


class DashDishCustom1Output(BaseModel):
    """Expected output schema for dashdish-custom1 task"""

    matcha_croissant_price: Optional[float] = Field(
        default=None, description="Price of the Matcha Croissant in dollars"
    )
    iced_matcha_latte_price: Optional[float] = Field(
        default=None, description="Price of the Regular Iced Matcha Latte in dollars"
    )
    ceremonial_matcha_latte_price: Optional[float] = Field(
        default=None,
        description="Price of the Ceremonial Grade Matcha Latte in dollars",
    )
    mochi_set_price: Optional[float] = Field(
        default=None, description="Price of the Mochi Set in dollars"
    )
    first_bundle_total: Optional[float] = Field(
        default=None,
        description="Total price for Matcha Croissant + Regular Iced Matcha Latte",
    )
    second_bundle_total: Optional[float] = Field(
        default=None,
        description="Total price for Ceremonial Grade Matcha Latte + Mochi Set",
    )


# Ground truth prices from eval script
INDIVIDUAL_PRICES = {
    "matcha_croissant": 4.49,
    "iced_matcha_latte": 6.99,
    "ceremonial_matcha_latte": 18.08,
    "mochi_set": 17.80,
}

BUNDLE_PRICES = {
    "first_bundle": 11.48,  # Matcha Croissant + Regular Iced Matcha Latte
    "second_bundle": 35.88,  # Ceremonial Grade Matcha Latte + Mochi Set
}


def prices_match(
    actual: Optional[float], expected: float, tolerance: float = 0.01
) -> bool:
    """Check if two prices match within tolerance."""
    if actual is None:
        return False
    return abs(actual - expected) <= tolerance


class DashDishCustom1Validator(TaskValidator):
    """Validator for DashDish Custom Task 1"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return DashDishCustom1Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """
        Check correctness as a percentage (0.0 to 1.0).

        Two valid answer formats:
        1. All 4 individual prices are correct
        2. Both bundle totals are correct

        Returns percentage of the best matching format.
        """
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, DashDishCustom1Output):
            logger.warning(
                f"check_correctness_pct: parsed_output is not DashDishCustom1Output (got {type(parsed_output)})"
            )
            return 0.0

        # Check individual prices (4 fields)
        individual_correct = 0
        if prices_match(
            parsed_output.matcha_croissant_price, INDIVIDUAL_PRICES["matcha_croissant"]
        ):
            individual_correct += 1
        else:
            logger.debug(
                f"Matcha Croissant price mismatch: expected {INDIVIDUAL_PRICES['matcha_croissant']}, got {parsed_output.matcha_croissant_price}"
            )

        if prices_match(
            parsed_output.iced_matcha_latte_price,
            INDIVIDUAL_PRICES["iced_matcha_latte"],
        ):
            individual_correct += 1
        else:
            logger.debug(
                f"Iced Matcha Latte price mismatch: expected {INDIVIDUAL_PRICES['iced_matcha_latte']}, got {parsed_output.iced_matcha_latte_price}"
            )

        if prices_match(
            parsed_output.ceremonial_matcha_latte_price,
            INDIVIDUAL_PRICES["ceremonial_matcha_latte"],
        ):
            individual_correct += 1
        else:
            logger.debug(
                f"Ceremonial Matcha Latte price mismatch: expected {INDIVIDUAL_PRICES['ceremonial_matcha_latte']}, got {parsed_output.ceremonial_matcha_latte_price}"
            )

        if prices_match(parsed_output.mochi_set_price, INDIVIDUAL_PRICES["mochi_set"]):
            individual_correct += 1
        else:
            logger.debug(
                f"Mochi Set price mismatch: expected {INDIVIDUAL_PRICES['mochi_set']}, got {parsed_output.mochi_set_price}"
            )

        individual_pct = individual_correct / 4.0

        # Check bundle prices (2 fields)
        bundle_correct = 0
        if prices_match(
            parsed_output.first_bundle_total, BUNDLE_PRICES["first_bundle"]
        ):
            bundle_correct += 1
        else:
            logger.debug(
                f"First bundle total mismatch: expected {BUNDLE_PRICES['first_bundle']}, got {parsed_output.first_bundle_total}"
            )

        if prices_match(
            parsed_output.second_bundle_total, BUNDLE_PRICES["second_bundle"]
        ):
            bundle_correct += 1
        else:
            logger.debug(
                f"Second bundle total mismatch: expected {BUNDLE_PRICES['second_bundle']}, got {parsed_output.second_bundle_total}"
            )

        bundle_pct = bundle_correct / 2.0

        # Return the better of the two formats
        best_pct = max(individual_pct, bundle_pct)
        logger.info(
            f"check_correctness_pct: individual={individual_correct}/4 ({individual_pct * 100:.1f}%), bundle={bundle_correct}/2 ({bundle_pct * 100:.1f}%), best={best_pct * 100:.1f}%"
        )

        return best_pct

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """
        Check if parsed output matches ground truth.

        Two valid answer formats (either one is acceptable):
        1. All 4 individual prices are correct
        2. Both bundle totals are correct

        Returns True if either format is 100% correct.
        """
        if not isinstance(parsed_output, DashDishCustom1Output):
            return False

        # Check if all individual prices are correct
        individual_correct = (
            prices_match(
                parsed_output.matcha_croissant_price,
                INDIVIDUAL_PRICES["matcha_croissant"],
            )
            and prices_match(
                parsed_output.iced_matcha_latte_price,
                INDIVIDUAL_PRICES["iced_matcha_latte"],
            )
            and prices_match(
                parsed_output.ceremonial_matcha_latte_price,
                INDIVIDUAL_PRICES["ceremonial_matcha_latte"],
            )
            and prices_match(
                parsed_output.mochi_set_price, INDIVIDUAL_PRICES["mochi_set"]
            )
        )

        # Check if bundle totals are correct
        bundle_correct = prices_match(
            parsed_output.first_bundle_total, BUNDLE_PRICES["first_bundle"]
        ) and prices_match(
            parsed_output.second_bundle_total, BUNDLE_PRICES["second_bundle"]
        )

        return individual_correct or bundle_correct


# Create singleton validator instance
validator = DashDishCustom1Validator()
