"""
Omnizon Custom Task 7: Category Product Colors

Task: Visit "Gift", "Toy", "Gaming", and "Cosmetic" categories on the menu bar.
For each, click on the first item and report the product color.

Expected colors:
- Gift: Yellow
- Toy: Purple
- Gaming: Black
- Cosmetic: Multicolor
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field

from ..base import TaskValidator


class CategoryColor(BaseModel):
    """Color for a category's first product"""

    category: Optional[str] = Field(default=None, description="Category name")
    color: Optional[str] = Field(default=None, description="Product color")


class OmnizonCustom7Output(BaseModel):
    """Expected output schema for omnizon-custom-7 task"""

    gift_color: Optional[str] = Field(default=None, description="Color of first Gift item")
    toy_color: Optional[str] = Field(default=None, description="Color of first Toy item")
    gaming_color: Optional[str] = Field(
        default=None, description="Color of first Gaming item"
    )
    cosmetic_color: Optional[str] = Field(
        default=None, description="Color of first Cosmetic item"
    )
    reasoning: Optional[str] = Field(default=None, description="Optional explanation")


# Ground truth - expected colors for each category
EXPECTED_COLORS: Dict[str, str] = {
    "gift": "yellow",
    "toy": "purple",
    "gaming": "black",
    "cosmetic": "multicolor",
}


def normalize_color(color: Optional[str]) -> str:
    """Normalize color string for comparison."""
    if color is None:
        return ""
    return color.lower().strip()


class OmnizonCustom7Validator(TaskValidator):
    """Validator for Omnizon Custom Task 7"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return OmnizonCustom7Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """Check how many colors are correct. Partial credit for each."""
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, OmnizonCustom7Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        # Map output fields to categories
        found_colors = {
            "gift": normalize_color(parsed_output.gift_color),
            "toy": normalize_color(parsed_output.toy_color),
            "gaming": normalize_color(parsed_output.gaming_color),
            "cosmetic": normalize_color(parsed_output.cosmetic_color),
        }

        logger.info(f"Found colors: {found_colors}")
        logger.info(f"Expected colors: {EXPECTED_COLORS}")

        # Count correct matches
        matched = 0
        for category, expected in EXPECTED_COLORS.items():
            found = found_colors.get(category, "")
            if found and expected in found or found == expected:
                matched += 1
                logger.info(f"✓ {category}: found '{found}' matches expected '{expected}'")
            else:
                logger.info(f"✗ {category}: found '{found}' != expected '{expected}'")

        score = matched / len(EXPECTED_COLORS)
        logger.info(f"Score: {matched}/{len(EXPECTED_COLORS)} = {score:.0%}")

        return score

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """Check if all colors are correct."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = OmnizonCustom7Validator()
