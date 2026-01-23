"""
Omnizon Custom Task 3: Headphones Category Product Retrieval

Task: Scroll through the homepage, click on the "Headphones" category card,
and retrieve all product listings within that category.

This is an LLM-based verifier that checks if the agent's response contains
multiple headphone products with their names and basic details.

Rubric (from omnizon-3.json):
"Does the answer list multiple headphone products from the Headphones category,
including their names and basic details?"
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from ..base import TaskValidator


class HeadphoneProduct(BaseModel):
    """A single headphone product listing"""

    brand: Optional[str] = Field(
        default=None, description="Brand name (e.g., Sony, Apple)"
    )
    name: str = Field(description="Product name or title")
    price: Optional[str] = Field(default=None, description="Price of the product")
    features: Optional[str] = Field(
        default=None, description="Product features or description"
    )


class OmnizonCustom3Output(BaseModel):
    """Expected output schema for omnizon-custom-3 task"""

    products: List[HeadphoneProduct] = Field(
        description="List of headphone products retrieved from the category"
    )


def has_basic_details(product: HeadphoneProduct) -> bool:
    """Check if a product has basic details (brand, features, or descriptive name)."""
    # Has brand
    if product.brand and len(product.brand.strip()) > 0:
        return True
    # Has features/description
    if product.features and len(product.features.strip()) > 0:
        return True
    # Name is descriptive (more than just a simple name, includes specs/details)
    if product.name and len(product.name.strip()) > 20:
        return True
    return False


class OmnizonCustom3Validator(TaskValidator):
    """
    LLM-based validator for Omnizon Custom Task 3.

    Validates based on the rubric:
    "Does the answer list multiple headphone products from the Headphones category,
    including their names and basic details?"
    """

    @property
    def output_schema(self) -> type[BaseModel]:
        return OmnizonCustom3Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """
        Check correctness as percentage.

        Rubric: "Does the answer list multiple headphone products from the
        Headphones category, including their names and basic details?"

        Scoring:
        - 50% for having 2+ products with names (multiple products)
        - 25% for having 3+ products (comprehensive retrieval)
        - 25% for products having basic details (brand, features, or descriptive names)
        """
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, OmnizonCustom3Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        products = parsed_output.products
        score = 0.0

        # Check number of products with names
        products_with_names = [
            p for p in products if p.name and len(p.name.strip()) > 0
        ]
        num_products = len(products_with_names)

        logger.info(f"Found {num_products} products with names")

        if num_products >= 2:
            score += 0.5  # 50% for having at least 2 products (multiple)
            logger.debug("Score +0.5: At least 2 products listed (multiple)")

        if num_products >= 3:
            score += 0.25  # Additional 25% for 3+ products (comprehensive)
            logger.debug("Score +0.25: At least 3 products listed (comprehensive)")

        # Check if products have basic details (brand, features, or descriptive names)
        products_with_details = [p for p in products_with_names if has_basic_details(p)]
        num_with_details = len(products_with_details)

        logger.info(f"Products with basic details: {num_with_details}/{num_products}")

        # Award 25% if at least half of products have basic details
        if num_products > 0 and num_with_details >= num_products / 2:
            score += 0.25
            logger.debug("Score +0.25: Products include basic details")

        logger.info(f"check_correctness_pct: {score:.0%}")
        return min(score, 1.0)

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """
        Check if the output is correct.

        Correct if score >= 75% (multiple products with basic details)
        """
        return self.check_correctness_pct(parsed_output) >= 0.75


# Create singleton validator instance
validator = OmnizonCustom3Validator()
