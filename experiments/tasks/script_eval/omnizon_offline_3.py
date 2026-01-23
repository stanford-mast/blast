"""
Omnizon Offline Task 3: Purchase Playstation Controller from Gaming Page

Task: Go to the "Gaming" page, and click on the Playstation wireless controller
to view the product details and add three to the cart. Make sure you buy it.

Criteria:
1. An order was placed (orderDetailsDiff.added contains an order)
2. The order contains an item with quantity 3
3. Gaming page was visited
"""

import json
import sys
from typing import Any, Dict, List, Tuple


def safe_get(d: Dict, *keys) -> Any:
    """Safely traverse nested dictionary keys."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def get_all_visited_paths(final_state: Dict) -> List[str]:
    """Get all visited paths from action history."""
    action_history = final_state.get("actionhistory", [])
    paths = []
    for action in action_history:
        if action.get("type") == "router/locationChange":
            payload = action.get("payload", {})
            location = payload.get("location", {})
            pathname = location.get("pathname", "")
            if pathname:
                paths.append(pathname.lower())
    return paths


def extract_new_orders(final_state: Dict) -> List[Dict]:
    """Extract newly placed orders from orderDetailsDiff or initialfinaldiff."""
    orders = []

    # Primary: orderDetailsDiff.added
    order_diff = final_state.get("orderDetailsDiff", {})
    added = order_diff.get("added", {})
    if isinstance(added, dict):
        for key, order in added.items():
            if isinstance(order, dict) and "items" in order:
                orders.append(order)

    # Fallback: initialfinaldiff.added.order.orders
    if not orders:
        ifd = final_state.get("initialfinaldiff", {})
        added_section = ifd.get("added", {})
        order_obj = added_section.get("order", {})
        orders_dict = order_obj.get("orders", {})
        if isinstance(orders_dict, dict):
            for key, order in orders_dict.items():
                if isinstance(order, dict) and "items" in order:
                    orders.append(order)

    return orders


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Criteria:
    1. Order placed with quantity 3 (50%)
    2. Gaming page visited (50%)

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    details = {
        "order_placed": False,
        "has_qty_3": False,
        "gaming_visited": False,
        "items_found": [],
        "paths_visited": [],
    }

    score = 0.0
    checks_passed = 0
    total_checks = 2

    # Check 1: Order was placed with quantity 3
    orders = extract_new_orders(final_state)
    if orders:
        details["order_placed"] = True
        logger.info(f"✓ Order placed: found {len(orders)} new order(s)")

        # Analyze items in the orders
        all_items = []
        for order in orders:
            items = order.get("items", [])
            if isinstance(items, list):
                all_items.extend(items)

        for item in all_items:
            if isinstance(item, dict):
                item_info = {
                    "id": item.get("id"),
                    "quantity": item.get("quantity"),
                    "price": item.get("price"),
                    "name": item.get("name", item.get("title", "")),
                }
                details["items_found"].append(item_info)

        # Check for quantity 3
        quantities = [item.get("quantity") for item in all_items if isinstance(item, dict)]
        if 3 in quantities:
            details["has_qty_3"] = True
            checks_passed += 1
            logger.info("✓ Found item with quantity 3 (Playstation controller)")
        else:
            logger.info(f"✗ No item with quantity 3 found. Quantities: {quantities}")
    else:
        logger.info("✗ No order placed")

    # Check 2: Gaming page was visited
    paths = get_all_visited_paths(final_state)
    details["paths_visited"] = paths[:20]  # Limit for readability
    gaming_visited = any("gaming" in path for path in paths)
    if gaming_visited:
        details["gaming_visited"] = True
        checks_passed += 1
        logger.info("✓ Gaming page visited")
    else:
        # Fallback: check if agent's result mentions gaming page
        if final_result and "gaming" in final_result.lower():
            details["gaming_visited"] = True
            checks_passed += 1
            logger.info("✓ Gaming page mentioned in agent result")
        else:
            logger.info("✗ Gaming page not visited")

    # Calculate final score
    score = checks_passed / total_checks

    # Success requires both: order with qty 3 AND gaming page visited
    success = details["has_qty_3"] and details["gaming_visited"]

    logger.info(
        f"Evaluation result: {'SUCCESS' if success else 'FAILURE'} ({score:.0%}) - "
        f"{checks_passed}/{total_checks} checks passed"
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
