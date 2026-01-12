"""
Task: Place an order for any type of sub-sandwich, keep the total under $30

Verifier:
- Check that at least one order contains a sub-sandwich
- Check that the order total is under $30
"""

import json
import sys
from typing import Any, Dict, Tuple

from .eval_utils import extract_orders

# Keywords to identify sub-sandwich items
SUB_KEYWORDS = ["sub", "sandwich", "hoagie", "hero", "grinder"]
MAX_TOTAL = 30.0


def to_float(val: Any) -> float | None:
    """Convert a value to float, handling currency strings."""
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            s = val.strip().replace("$", "").replace(",", "")
            return float(s)
    except Exception:
        return None
    return None


def contains_sub(text: str) -> bool:
    """Check if text contains any sub-sandwich keyword (case-insensitive)."""
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in SUB_KEYWORDS)


def order_has_sub(order: Dict[str, Any]) -> bool:
    """Check if an order contains a sub-sandwich in any cart item."""
    items = order.get("cartItems", [])
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        desc = item.get("description", "")
        if contains_sub(name) or contains_sub(desc):
            return True
    return False


def get_order_total(order: Dict[str, Any]) -> float | None:
    """Get the total amount from an order's checkout details."""
    charges = order.get("checkoutDetails", {}).get("charges", {})
    return to_float(charges.get("totalAmount"))


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    # Extract orders
    orders = extract_orders(final_state)
    logger.info(f"Found {len(orders)} total orders in final_state")

    # Find orders with sub-sandwiches and check total
    sub_orders = []
    valid_orders = []  # Orders with sub AND total under $30

    for i, order in enumerate(orders):
        has_sub = order_has_sub(order)
        total = get_order_total(order)

        cart_items = order.get("cartItems", [])
        item_names = [
            item.get("name", "unknown") for item in cart_items if isinstance(item, dict)
        ]
        logger.info(
            f"  Order {i}: items={item_names}, total=${total}, has_sub={has_sub}"
        )

        if has_sub:
            sub_orders.append(order)
            if total is not None and total <= MAX_TOTAL:
                valid_orders.append(order)
                logger.info(
                    f"    -> Valid order: sub found and total ${total} <= ${MAX_TOTAL}"
                )
            elif total is not None:
                logger.info(f"    -> Invalid: total ${total} exceeds ${MAX_TOTAL}")

    # Partial credit:
    # - 50% for placing any order
    # - 50% for order being correct (sub-sandwich AND total under $30)
    order_placed = len(orders) >= 1
    order_correct = len(valid_orders) >= 1

    total_pct = 0.0
    if order_placed:
        total_pct += 0.5
    if order_correct:
        total_pct += 0.5

    success = order_placed and order_correct

    details = {
        "total_orders_found": len(orders),
        "sub_orders_found": len(sub_orders),
        "valid_orders_found": len(valid_orders),
        "max_total_allowed": MAX_TOTAL,
        "order_placed": order_placed,
        "order_correct": order_correct,
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
