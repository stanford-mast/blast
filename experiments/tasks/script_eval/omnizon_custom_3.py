"""
Omnizon Custom Task 3: Cancel July Unshipped Orders

Task: Cancel the three orders placed in July that have not been shipped yet
on Returns & Orders page.

July orders (NOT YET SHIPPED - should be cancelled):
1. $332.89 (7/18/2024)
2. $119.99 (7/17/2024)
3. $177.89 (7/16/2024)

Other orders (should NOT be cancelled):
4. $33.97 (12/20/2023)
5. $119.95 (12/15/2023)
6. $642.96 (12/1/2023)
7. $98.97 (11/25/2023)
8. $2448.88 (11/18/2023)

Criteria:
1. All 3 July orders must be cancelled
2. No other orders should be cancelled
"""

import json
import sys
from typing import Any, Dict, List, Set, Tuple

# July order totals (should be cancelled)
UNSHIPPED_ORDER_TOTALS: Set[float] = {
    332.89,  # Order 1 (7/18/2024)
    119.99,  # Order 2 (7/17/2024)
    177.89,  # Order 3 (7/16/2024)
}

# Other order totals (should NOT be cancelled)
SHIPPED_ORDER_TOTALS: Set[float] = {
    33.97,  # Order 4
    119.95,  # Order 5
    642.96,  # Order 6
    98.97,  # Order 7
    2448.88,  # Order 8
}


def safe_get(d: Dict, *keys) -> Any:
    """Safely traverse nested dictionary keys."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def get_cancelled_orders(final_state: Dict[str, Any]) -> List[Dict]:
    """Get all cancelled orders from various possible locations in state."""
    # Check for cancelled orders in finalstate.order.cancelledOrders
    cancelled_orders = safe_get(final_state, "finalstate", "order", "cancelledOrders")

    # Also check in state.order.cancelledOrders as fallback
    if not cancelled_orders:
        cancelled_orders = safe_get(final_state, "state", "order", "cancelledOrders")

    # Also check initialfinaldiff for newly added cancelled orders
    if not cancelled_orders:
        diff_cancelled = safe_get(
            final_state, "initialfinaldiff", "added", "order", "cancelledOrders"
        )
        if diff_cancelled:
            cancelled_orders = diff_cancelled

    if not cancelled_orders:
        return []

    # Handle dict format with numeric keys
    if isinstance(cancelled_orders, dict):
        return list(cancelled_orders.values())
    elif isinstance(cancelled_orders, list):
        return cancelled_orders

    return []


def match_total(actual: float, expected: float, tolerance: float = 0.02) -> bool:
    """Check if two totals match within tolerance."""
    return abs(actual - expected) < tolerance


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Criteria:
    1. All 3 July orders must be cancelled
    2. No other orders should be incorrectly cancelled

    Scoring:
    - Each correctly cancelled July order: 33.3%
    - Penalty for cancelling other orders: -10% each

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    details = {
        "unshipped_cancelled": [],
        "unshipped_not_cancelled": list(UNSHIPPED_ORDER_TOTALS),
        "shipped_incorrectly_cancelled": [],
        "cancelled_orders_found": [],
        "expected_unshipped_count": len(UNSHIPPED_ORDER_TOTALS),
        "actual_unshipped_cancelled_count": 0,
    }

    # Get all cancelled orders
    cancelled_orders = get_cancelled_orders(final_state)

    unshipped_cancelled = set()
    shipped_incorrectly_cancelled = []

    for order in cancelled_orders:
        if isinstance(order, dict):
            order_num = order.get("orderNumber", order.get("orderId", ""))
            total = order.get("total", 0)
            order_info = {
                "orderNumber": order_num,
                "total": total,
            }
            details["cancelled_orders_found"].append(order_info)

            # Check if this is an unshipped order (should be cancelled)
            matched_unshipped = False
            for expected_total in UNSHIPPED_ORDER_TOTALS:
                if match_total(total, expected_total):
                    unshipped_cancelled.add(expected_total)
                    matched_unshipped = True
                    logger.info(f"✓ Unshipped order ${expected_total} was cancelled")
                    break

            # Check if this is a shipped order (should NOT be cancelled)
            if not matched_unshipped:
                for expected_total in SHIPPED_ORDER_TOTALS:
                    if match_total(total, expected_total):
                        shipped_incorrectly_cancelled.append(expected_total)
                        logger.warning(
                            f"✗ Shipped order ${expected_total} was incorrectly cancelled"
                        )
                        break

    # Update details
    details["unshipped_cancelled"] = list(unshipped_cancelled)
    details["unshipped_not_cancelled"] = list(
        UNSHIPPED_ORDER_TOTALS - unshipped_cancelled
    )
    details["shipped_incorrectly_cancelled"] = shipped_incorrectly_cancelled
    details["actual_unshipped_cancelled_count"] = len(unshipped_cancelled)

    # Log missing cancellations
    for missing in details["unshipped_not_cancelled"]:
        logger.info(f"✗ Unshipped order ${missing} was NOT cancelled")

    # Calculate score
    # Base score: percentage of unshipped orders correctly cancelled
    base_score = len(unshipped_cancelled) / len(UNSHIPPED_ORDER_TOTALS)

    # Penalty for incorrectly cancelling shipped orders (-10% each, min 0)
    penalty = len(shipped_incorrectly_cancelled) * 0.10

    score = max(0.0, base_score - penalty)

    # Success requires ALL unshipped orders cancelled and NO shipped orders cancelled
    all_unshipped_cancelled = len(unshipped_cancelled) == len(UNSHIPPED_ORDER_TOTALS)
    no_shipped_cancelled = len(shipped_incorrectly_cancelled) == 0
    success = all_unshipped_cancelled and no_shipped_cancelled

    logger.info(
        f"Evaluation result: {'SUCCESS' if success else 'FAILURE'} ({score:.0%}) - "
        f"{len(unshipped_cancelled)}/{len(UNSHIPPED_ORDER_TOTALS)} unshipped orders cancelled, "
        f"{len(shipped_incorrectly_cancelled)} shipped orders incorrectly cancelled"
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
