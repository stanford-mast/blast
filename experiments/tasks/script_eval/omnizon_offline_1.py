"""
Omnizon Offline Task 1: Write Review and Cancel First Order

Task: Go to Returns & Orders page, write a product review for the first item
in the first order, and then cancel the entire order.

The first order contains:
- Item 1: Product ID "1" (Owala FreeSip Water Bottle) - $32.99
- Item 2: Product ID "3" (Cuisinart Air Fryer Oven) x2 - $149.95 each
- Total: $332.89

Criteria:
1. A review was created for product ID "1" (first item)
2. The first order ($332.89) was cancelled
"""

import json
import sys
from typing import Any, Dict, List, Tuple

# Expected values for the first order
EXPECTED_PRODUCT_ID = "1"
EXPECTED_ORDER_TOTAL = 332.89


def safe_get(d: Dict, *keys) -> Any:
    """Safely traverse nested dictionary keys."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def get_reviews(final_state: Dict[str, Any]) -> List[Dict]:
    """Get all user-created reviews from various possible locations in state."""
    # Check for reviews in finalstate.review.userCreatedReviews
    user_reviews = safe_get(final_state, "finalstate", "review", "userCreatedReviews")

    # Also check in initialfinaldiff for added reviews
    diff_reviews = safe_get(
        final_state, "initialfinaldiff", "added", "review", "userCreatedReviews"
    )

    # Also check state.review.userCreatedReviews as fallback
    state_reviews = safe_get(final_state, "state", "review", "userCreatedReviews")

    # Use whichever has reviews
    if isinstance(user_reviews, list) and len(user_reviews) > 0:
        return user_reviews
    elif isinstance(diff_reviews, list) and len(diff_reviews) > 0:
        return diff_reviews
    elif isinstance(diff_reviews, dict) and len(diff_reviews) > 0:
        # Handle dict format with numeric keys
        return list(diff_reviews.values())
    elif isinstance(state_reviews, list) and len(state_reviews) > 0:
        return state_reviews

    return []


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


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Criteria:
    1. A review was created for product ID "1" (first item in first order)
    2. The first order ($332.89) was cancelled

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    details = {
        "review_created": False,
        "order_cancelled": False,
        "expected_product_id": EXPECTED_PRODUCT_ID,
        "expected_order_total": EXPECTED_ORDER_TOTAL,
        "reviews_found": [],
        "cancelled_orders_found": [],
    }

    # Check for reviews
    reviews = get_reviews(final_state)
    for review in reviews:
        if isinstance(review, dict):
            product_id = str(review.get("productId", ""))
            review_info = {
                "productId": product_id,
                "rating": review.get("rating"),
                "title": review.get("title", "")[:50] if review.get("title") else "",
            }
            details["reviews_found"].append(review_info)

            if product_id == EXPECTED_PRODUCT_ID:
                details["review_created"] = True
                logger.info(f"✓ Review found for product ID {EXPECTED_PRODUCT_ID}")

    if not details["review_created"]:
        logger.info(f"✗ No review found for product ID {EXPECTED_PRODUCT_ID}")

    # Check for cancelled orders
    cancelled_orders = get_cancelled_orders(final_state)
    for order in cancelled_orders:
        if isinstance(order, dict):
            order_num = order.get("orderNumber", order.get("orderId", ""))
            total = order.get("total", 0)
            order_info = {
                "orderNumber": order_num,
                "total": total,
            }
            details["cancelled_orders_found"].append(order_info)

            # Check if this is the expected order (within small tolerance for float comparison)
            if isinstance(total, (int, float)) and abs(total - EXPECTED_ORDER_TOTAL) < 0.01:
                details["order_cancelled"] = True
                logger.info(
                    f"✓ Order with total ${EXPECTED_ORDER_TOTAL} was cancelled (order: {order_num})"
                )

    if not details["order_cancelled"]:
        logger.info(f"✗ No order with total ${EXPECTED_ORDER_TOTAL} found in cancelled orders")

    # Both criteria must be met for success
    success = details["review_created"] and details["order_cancelled"]

    # Partial scoring: 50% for each criterion
    score = 0.0
    if details["review_created"]:
        score += 0.5
    if details["order_cancelled"]:
        score += 0.5

    if success:
        logger.info("✓ Both criteria met: review created and order cancelled")
    else:
        missing = []
        if not details["review_created"]:
            missing.append("review for product 1")
        if not details["order_cancelled"]:
            missing.append(f"cancellation of ${EXPECTED_ORDER_TOTAL} order")
        logger.info(f"✗ Missing: {', '.join(missing)}")

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
