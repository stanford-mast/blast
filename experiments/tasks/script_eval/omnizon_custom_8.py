"""
Omnizon Custom Task 8: Espresso Machine Purchase

Task: Search for "Automatic Espresso Machine," click on the cheapest one,
change the quantity to 5, use "buy now" to purchase them and complete the checkout.

This script evaluator checks:
- Quantity of 5: orderDetailsDiff.added."0".items[0].quantity == 5
- Order placed: orderDetailsDiff.added."0" exists with items
- 1 product type in order: length(orderDetailsDiff.added."0".items) == 1

Note: The original omnizon-8.json used "order.cart" paths but the actual state
uses "items" directly. Also, we don't check specific product ID since the
cheapest espresso machine may vary across site versions.

Reference: experiments/agisdk/src/agisdk/REAL/browsergym/webclones/v1/tasks/omnizon-8.json
"""

import json
import sys
from typing import Any, Dict, Tuple

import jmespath

# Expected values
EXPECTED_QUANTITY = 5
EXPECTED_ITEMS_LENGTH = 1


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output using jmespath queries.

    Criteria:
    1. Order placed: orderDetailsDiff.added."0" exists
    2. Quantity of 5: orderDetailsDiff.added."0".items[0].quantity == 5
    3. 1 product type in order: length(orderDetailsDiff.added."0".items) == 1

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    # Debug: Log the structure of final_state
    logger.debug(f"final_state keys: {list(final_state.keys())}")

    details = {
        "order_placed": False,
        "quantity_correct": False,
        "items_length_correct": False,
        "actual_quantity": None,
        "actual_product_id": None,
        "actual_items_length": None,
    }

    score = 0.0
    checks_passed = 0
    total_checks = 3

    # Query 1: Check order was placed (orderDetailsDiff.added."0" exists)
    try:
        order = jmespath.search('orderDetailsDiff.added."0"', final_state)
        if order is not None:
            details["order_placed"] = True
            checks_passed += 1
            logger.info("✓ Order placed check passed")
        else:
            logger.info("✗ Order placed check failed: no order found")
    except Exception as e:
        logger.warning(f"Failed to query order: {e}")

    # Query 2: Check quantity == 5
    try:
        quantity = jmespath.search(
            'orderDetailsDiff.added."0".items[0].quantity', final_state
        )
        details["actual_quantity"] = quantity
        if quantity == EXPECTED_QUANTITY:
            details["quantity_correct"] = True
            checks_passed += 1
            logger.info(f"✓ Quantity check passed: {quantity}")
        else:
            logger.info(
                f"✗ Quantity check failed: expected {EXPECTED_QUANTITY}, got {quantity}"
            )
    except Exception as e:
        logger.warning(f"Failed to query quantity: {e}")

    # Also capture product ID for debugging (not used in scoring)
    try:
        product_id = jmespath.search(
            'orderDetailsDiff.added."0".items[0].id', final_state
        )
        details["actual_product_id"] = product_id
        logger.info(f"Product ID purchased: {product_id}")
    except Exception as e:
        logger.warning(f"Failed to query product id: {e}")

    # Query 3: Check items length == 1
    try:
        items = jmespath.search('orderDetailsDiff.added."0".items', final_state)
        items_length = len(items) if items else 0
        details["actual_items_length"] = items_length
        if items_length == EXPECTED_ITEMS_LENGTH:
            details["items_length_correct"] = True
            checks_passed += 1
            logger.info(f"✓ Items length check passed: {items_length}")
        else:
            logger.info(
                f"✗ Items length check failed: expected {EXPECTED_ITEMS_LENGTH}, got {items_length}"
            )
    except Exception as e:
        logger.warning(f"Failed to query items length: {e}")

    # Calculate score
    score = checks_passed / total_checks
    success = checks_passed == total_checks

    logger.info(
        f"Evaluation result: {checks_passed}/{total_checks} checks passed ({score:.0%})"
    )

    return success, score, details


def main():
    """CLI entry point for subprocess execution."""
    state_path = sys.argv[1]
    with open(state_path) as f:
        state = json.load(f)
        final_state = state.get("final_state", {})
        final_result = state.get("final_result", "")

    success, pct, details = evaluate(final_state, final_result)

    print(f"{'SUCCESS' if success else 'FAILURE'} ({pct * 100:.0f}%)")
    print(f"Details: {json.dumps(details, indent=2)}")


if __name__ == "__main__":
    main()
