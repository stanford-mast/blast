"""
Omnizon Custom Task 1: Cancel Most Expensive Order

Task: Find the most expensive order on Returns & Orders page and cancel it.

Expected: The most expensive order ($2448.88) should be cancelled.

Criteria:
1. An order with total $2448.88 appears in cancelledOrders
"""

import json
import sys
from typing import Any, Dict, Tuple

# Expected order total to be cancelled (the most expensive one)
# Order number is randomized, but total is consistent
EXPECTED_ORDER_TOTAL = 2448.88


def safe_get(d: Dict, *keys) -> Any:
    """Safely traverse nested dictionary keys."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Criteria:
    - An order with the expected total ($2448.88) appears in cancelledOrders

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    details = {
        "expected_total": EXPECTED_ORDER_TOTAL,
        "order_cancelled": False,
        "cancelled_orders": [],
        "matched_order": None,
    }

    # Check for cancelled orders in finalstate.order.cancelledOrders
    cancelled_orders = safe_get(final_state, "finalstate", "order", "cancelledOrders")

    # Also check in state.order.cancelledOrders as fallback
    if not cancelled_orders:
        cancelled_orders = safe_get(final_state, "state", "order", "cancelledOrders")

    if not cancelled_orders:
        logger.info("No cancelledOrders found in state")
        return False, 0.0, details

    # Check if cancelled_orders is a dict (indexed by "0", "1", etc.)
    if isinstance(cancelled_orders, dict):
        cancelled_list = list(cancelled_orders.values())
    elif isinstance(cancelled_orders, list):
        cancelled_list = cancelled_orders
    else:
        logger.warning(f"Unexpected cancelledOrders type: {type(cancelled_orders)}")
        return False, 0.0, details

    # Extract order info from cancelled orders
    for order in cancelled_list:
        if isinstance(order, dict):
            order_num = order.get("orderNumber", order.get("orderId", ""))
            total = order.get("total", 0)
            details["cancelled_orders"].append(
                {
                    "orderNumber": order_num,
                    "total": total,
                }
            )

            # Check if this order has the expected total (within small tolerance for float comparison)
            if (
                isinstance(total, (int, float))
                and abs(total - EXPECTED_ORDER_TOTAL) < 0.01
            ):
                details["order_cancelled"] = True
                details["matched_order"] = order_num
                logger.info(
                    f"✓ Order with total ${EXPECTED_ORDER_TOTAL} was cancelled (order: {order_num})"
                )

    if not details["order_cancelled"]:
        logger.info(
            f"✗ No order with total ${EXPECTED_ORDER_TOTAL} was found in cancelled orders"
        )
        logger.info(
            f"  Cancelled orders found: {[(o['orderNumber'], o['total']) for o in details['cancelled_orders']]}"
        )

    # All-or-nothing scoring
    success = details["order_cancelled"]
    score = 1.0 if success else 0.0

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
