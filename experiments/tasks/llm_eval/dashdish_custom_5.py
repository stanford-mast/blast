"""
DashDish Custom Task 5: First Three Restaurants on Homepage

Task: What are the first three restaurants listed on the homepage?

Expected Answer: Gambinos New York Subs, Wingstop, Man vs. Fries
"""

import re
from typing import Optional

from pydantic import BaseModel, Field

from ..base import TaskValidator


class DashDishCustom5Output(BaseModel):
    """Expected output schema for dashdish-custom-5 task"""

    first_restaurant: Optional[str] = Field(
        default=None, description="Name of the first restaurant listed"
    )
    second_restaurant: Optional[str] = Field(
        default=None, description="Name of the second restaurant listed"
    )
    third_restaurant: Optional[str] = Field(
        default=None, description="Name of the third restaurant listed"
    )
    restaurants: Optional[list[str]] = Field(
        default=None, description="List of the first three restaurants in order"
    )


# Ground truth - expected restaurant names in order
EXPECTED_RESTAURANTS = [
    "Gambinos New York Subs",
    "Wingstop",
    "Man vs. Fries",
]


def normalize_name(name: str) -> str:
    """Normalize a restaurant name for comparison.

    Removes punctuation, extra whitespace, and converts to lowercase.
    """
    if not name:
        return ""
    # Remove punctuation and convert to lowercase
    normalized = re.sub(r"[^\w\s]", "", name.lower())
    # Collapse multiple spaces into one and strip
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def names_match(actual: Optional[str], expected: str) -> bool:
    """Check if two restaurant names match (ignoring case, punctuation, etc.)."""
    if actual is None:
        return False
    return normalize_name(actual) == normalize_name(expected)


class DashDishCustom5Validator(TaskValidator):
    """Validator for DashDish Custom Task 5"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return DashDishCustom5Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """
        Check correctness as a percentage (0.0 to 1.0).

        Accepts two formats:
        1. Individual restaurant fields (first_restaurant, second_restaurant, third_restaurant)
        2. A list of restaurants in the 'restaurants' field

        Returns percentage of correct answers.
        """
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, DashDishCustom5Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        # Check individual fields
        individual_correct = 0
        individual_names = [
            parsed_output.first_restaurant,
            parsed_output.second_restaurant,
            parsed_output.third_restaurant,
        ]

        for i, (actual, expected) in enumerate(
            zip(individual_names, EXPECTED_RESTAURANTS)
        ):
            if names_match(actual, expected):
                individual_correct += 1
            else:
                logger.debug(
                    f"Restaurant {i + 1} mismatch: expected '{expected}', got '{actual}'"
                )

        individual_pct = individual_correct / 3.0

        # Check list field
        list_correct = 0
        if parsed_output.restaurants:
            for i, expected in enumerate(EXPECTED_RESTAURANTS):
                if i < len(parsed_output.restaurants):
                    actual = parsed_output.restaurants[i]
                    if names_match(actual, expected):
                        list_correct += 1
                    else:
                        logger.debug(
                            f"Restaurant list[{i}] mismatch: expected '{expected}', got '{actual}'"
                        )

        list_pct = list_correct / 3.0

        # Return the better of the two formats
        best_pct = max(individual_pct, list_pct)
        logger.info(
            f"check_correctness_pct: individual={individual_correct}/3 ({individual_pct * 100:.1f}%), "
            f"list={list_correct}/3 ({list_pct * 100:.1f}%), best={best_pct * 100:.1f}%"
        )

        return best_pct

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """Check if all three restaurants are correct."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = DashDishCustom5Validator()
