"""
Omnizon Custom Task 5: Buy Now with Max Quantity and Latest Delivery

Task: Click on "buy now" on any product, increase its quantity to the maximum allowed,
update the delivery date to the last available, and place the order.

Reference: experiments/agisdk/src/agisdk/REAL/browsergym/webclones/v1/tasks/omnizon-10.json

Criteria:
1. Quantity is 30 (maximum allowed) - 50%
2. Delivery date is "Monday, Jul 21" (latest available) - 50%
"""

import json
import sys
from typing import Any, Dict, Tuple

import jmespath


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Criteria:
    1. Quantity is 30 (max) - 50%
    2. Delivery date is latest available - 50%

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    # Expected values from reference
    EXPECTED_QUANTITY = 30
    EXPECTED_DELIVERY = "Monday, Jul 21"

    details = {
        "quantity_correct": False,
        "delivery_correct": False,
        "actual_quantity": None,
        "actual_delivery": None,
        "order_placed": False,
    }

    score = 0.0

    # Query 1: Check quantity is 30 (max)
    try:
        # Try the exact path from reference
        quantity = jmespath.search(
            'orderDetailsDiff.added."0".order.cart[0].quantity', final_state
        )

        # Fallback: try alternative paths
        if quantity is None:
            quantity = jmespath.search(
                'orderDetailsDiff.added."0".items[0].quantity', final_state
            )
        if quantity is None:
            # Try looking in initialfinaldiff
            quantity = jmespath.search(
                'initialfinaldiff.added.order.orders."0".items[0].quantity', final_state
            )

        details["actual_quantity"] = quantity

        if quantity is not None:
            details["order_placed"] = True
            if quantity == EXPECTED_QUANTITY:
                details["quantity_correct"] = True
                score += 0.5
                logger.info(f"✓ Quantity is {quantity} (max)")
            else:
                logger.info(f"✗ Quantity is {quantity}, expected {EXPECTED_QUANTITY}")
        else:
            logger.info("✗ No order quantity found")
    except Exception as e:
        logger.warning(f"Failed to query quantity: {e}")

    # Query 2: Check delivery date is latest
    try:
        # Try the exact path from reference
        delivery = jmespath.search(
            'orderDetailsDiff.added."0".order.selected_delivery', final_state
        )

        # Fallback: try alternative paths
        if delivery is None:
            delivery = jmespath.search(
                'orderDetailsDiff.added."0".selected_delivery', final_state
            )
        if delivery is None:
            delivery = jmespath.search(
                'initialfinaldiff.added.order.orders."0".selected_delivery', final_state
            )

        details["actual_delivery"] = delivery

        if delivery is not None:
            # Check if it contains the expected date (may have different formats)
            # Accept: "Monday, Jul 21", "Jul 21, 2024", "Jul 21", etc.
            delivery_str = str(delivery).lower()
            if "jul 21" in delivery_str or "july 21" in delivery_str:
                details["delivery_correct"] = True
                score += 0.5
                logger.info(f"✓ Delivery date is '{delivery}' (latest)")
            else:
                logger.info(f"✗ Delivery date is '{delivery}', expected 'Jul 21'")
        else:
            logger.info("✗ No delivery date found")
    except Exception as e:
        logger.warning(f"Failed to query delivery: {e}")

    # Success requires both criteria
    success = details["quantity_correct"] and details["delivery_correct"]

    logger.info(
        f"Evaluation result: {'SUCCESS' if success else 'FAILURE'} ({score:.0%})"
    )

    return success, score, details


def main():
    """CLI entry point for subprocess execution."""
    path = sys.argv[1]
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"FAILURE: Could not load state file: {e}")
        return

    final_state = data.get("final_state", data)
    final_result = data.get("final_result", "")

    success, pct, details = evaluate(final_state, final_result)

    print(f"{'SUCCESS' if success else 'FAILURE'} ({pct * 100:.0f}%)")
    print(f"Details: {json.dumps(details, indent=2)}")


if __name__ == "__main__":
    main()
