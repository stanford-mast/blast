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
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

# July order totals (should be cancelled) - keyed by (total, month, year)
UNSHIPPED_ORDERS: Set[Tuple[float, int, int]] = {
    (332.89, 7, 2024),
    (119.99, 7, 2024),
    (177.89, 7, 2024),
}

# Just totals for backwards compatibility
UNSHIPPED_ORDER_TOTALS: Set[float] = {t[0] for t in UNSHIPPED_ORDERS}

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


def parse_order_date(date_str: str) -> Tuple[int, int]:
    """Parse order date string and return (month, year) tuple."""
    if not date_str:
        return (0, 0)
    try:
        # Handle ISO format: "2024-07-17T14:20:00.000Z"
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return (dt.month, dt.year)
    except (ValueError, AttributeError):
        return (0, 0)


def is_july_2024_order(order: Dict) -> bool:
    """Check if an order was placed in July 2024."""
    date_str = order.get("date", "")
    month, year = parse_order_date(date_str)
    return month == 7 and year == 2024


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
            order_date = order.get("date", "")
            month, year = parse_order_date(order_date)
            order_info = {
                "orderNumber": order_num,
                "total": total,
                "date": order_date,
                "month": month,
                "year": year,
            }
            details["cancelled_orders_found"].append(order_info)

            # Check if this is a July 2024 unshipped order (should be cancelled)
            # Must match BOTH total AND be from July 2024
            matched_unshipped = False
            if is_july_2024_order(order):
                for expected_total in UNSHIPPED_ORDER_TOTALS:
                    if match_total(total, expected_total):
                        unshipped_cancelled.add(expected_total)
                        matched_unshipped = True
                        logger.info(
                            f"✓ Unshipped order ${expected_total} (July 2024) was cancelled"
                        )
                        break

            # Check if this is a shipped order (should NOT be cancelled)
            # Only count as incorrectly cancelled if it's not a pre-existing cancelled order
            # Pre-existing cancelled orders have dates before July 2024
            if not matched_unshipped and is_july_2024_order(order):
                # This is a non-July-target order from July 2024 that was cancelled
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
