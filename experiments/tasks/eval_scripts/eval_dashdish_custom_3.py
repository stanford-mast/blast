"""
Task: Buy a taco from any store and make sure you place the order.

Verifier:
- Check that exactly one order matching "taco" is placed (100% weight)
"""

import json
import sys
from typing import Any, Dict, Tuple

from .eval_utils import count_orders_with_keyword

TACO_KEYWORD = "taco"


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Returns:
        (success, percentage, details)
    """
    import logging

    from .eval_utils import extract_orders

    logger = logging.getLogger(__name__)

    # Debug: Log the structure of final_state
    logger.debug(f"final_state keys: {list(final_state.keys())}")

    # Extract and log orders for debugging
    orders = extract_orders(final_state)
    logger.info(f"Found {len(orders)} total orders in final_state")
    for i, order in enumerate(orders):
        cart_items = order.get("cartItems", [])
        item_names = [
            item.get("name", "unknown") for item in cart_items if isinstance(item, dict)
        ]
        logger.info(f"  Order {i}: items = {item_names}")

    # Check if exactly one taco order is placed
    taco_order_count = count_orders_with_keyword(final_state, TACO_KEYWORD)
    order_ok = taco_order_count == 1

    # Binary: taco must be in an order, no partial credit for cart
    total_pct = 1.0 if order_ok else 0.0

    success = order_ok

    details = {
        "taco_orders_found": taco_order_count,
        "taco_orders_expected": 1,
        "total_orders_found": len(orders),
    }

    return success, total_pct, details


def main():
    """CLI entry point for subprocess execution."""
    state_path = sys.argv[1]
    with open(state_path) as f:
        state = json.load(f)
        final_state = state.get("final_state", {})
        final_result = state.get("final_result", "")

    success, pct, details = evaluate(final_state, final_result)

    if success:
        print(f"SUCCESS ({pct * 100:.0f}%)")
    else:
        print(f"FAILURE ({pct * 100:.0f}%)")


if __name__ == "__main__":
    main()
