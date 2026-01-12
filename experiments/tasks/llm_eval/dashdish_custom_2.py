"""
DashDish Custom Task 2: Costume Party Ready Reviews

Task: For each vendor under the 'Costume party ready' category, visit the store
page and get the first three reviews.

Expected stores and reviewers:
- Walgreens: Noah, Olivia, Ethan
- CVS: Mike, Isabella, Lucas
- Office Depot OfficeMax: Alex, Maria, John
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from ..base import TaskValidator


class StoreReviews(BaseModel):
    """Reviews for a single store"""

    store_name: str = Field(description="Name of the store")
    reviewer_names: List[str] = Field(
        description="List of reviewer names (first 3 reviews)"
    )


class DashDishCustom2Output(BaseModel):
    """Expected output schema for dashdish-custom-2 task"""

    stores: List[StoreReviews] = Field(
        description="List of stores with their reviewer names"
    )


# Ground truth data
GROUND_TRUTH = {
    "walgreens": ["Noah", "Olivia", "Ethan"],
    "cvs": ["Mike", "Isabella", "Lucas"],
    "office depot officemax": ["Alex", "Maria", "John"],
}


def normalize_name(name: str) -> str:
    """Normalize name for comparison."""
    return name.lower().strip().split()[0]


def normalize_store_name(name: str) -> str:
    """Normalize store name for comparison."""
    name = name.lower().strip()
    # Handle variations
    if "office depot" in name or "officemax" in name:
        return "office depot officemax"
    return name


class DashDishCustom2Validator(TaskValidator):
    """Validator for DashDish Custom Task 2"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return DashDishCustom2Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """
        Check correctness as percentage.

        3 stores × 3 reviewers = 9 total names to check.
        """
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, DashDishCustom2Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        total_names = 9  # 3 stores × 3 reviewers
        correct_names = 0

        # Build lookup by normalized store name
        parsed_stores = {
            normalize_store_name(s.store_name): s for s in parsed_output.stores
        }

        for gt_store, gt_names in GROUND_TRUTH.items():
            if gt_store not in parsed_stores:
                logger.debug(f"Store '{gt_store}' not found in output")
                continue

            parsed_names = [
                normalize_name(n) for n in parsed_stores[gt_store].reviewer_names
            ]

            for gt_name in gt_names:
                if normalize_name(gt_name) in parsed_names:
                    correct_names += 1
                else:
                    logger.debug(
                        f"Reviewer '{gt_name}' not found in {gt_store} reviews"
                    )

        pct = correct_names / total_names
        logger.info(f"check_correctness_pct: {correct_names}/{total_names} = {pct:.0%}")
        return pct

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """Check if all reviewer names are correct."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = DashDishCustom2Validator()
