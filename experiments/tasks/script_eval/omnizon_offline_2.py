"""
Omnizon Offline Task 2: Search and Purchase Sony Headphones

Task: Search for "Sony WH-1000XM5" and click into the first result.
Add it to the cart and buy it.

Criteria:
1. An order was placed (orderDetailsDiff.added contains an order)
2. The order contains at least 1 item with quantity 1
3. Search was performed for Sony WH-1000XM5
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


def get_search_terms(final_state: Dict) -> List[str]:
    """Extract search terms from action history and state diffs."""
    terms = []

    # Check initialfinaldiff for search terms
    ifd = final_state.get("initialfinaldiff", {})
    for section in ("updated", "added"):
        sec = ifd.get(section)
        if not isinstance(sec, dict):
            continue
        filt = sec.get("filter")
        if isinstance(filt, dict):
            for key in ("searchQuery", "searchInputValue"):
                val = filt.get(key)
                if isinstance(val, str) and val:
                    terms.append(val.lower())

    # Check action history for search actions
    action_history = final_state.get("actionhistory", [])
    for action in action_history:
        action_type = action.get("type", "")
        if "search" in action_type.lower() or "filter" in action_type.lower():
            payload = action.get("payload", {})
            if isinstance(payload, str) and payload:
                terms.append(payload.lower())
            elif isinstance(payload, dict):
                for key in ("query", "searchQuery", "searchInputValue", "term"):
                    val = payload.get(key)
                    if isinstance(val, str) and val:
                        terms.append(val.lower())

    # Also check URL search params in visited paths
    for action in action_history:
        if action.get("type") == "router/locationChange":
            payload = action.get("payload", {})
            location = payload.get("location", {})
            search = location.get("search", "")
            if search and "search" in search.lower():
                terms.append(search.lower())

    return terms


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
    1. Order placed with items (50%)
    2. Search performed for Sony WH-1000XM5 (50%)

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    details = {
        "order_placed": False,
        "has_item": False,
        "search_performed": False,
        "items_found": [],
        "search_terms_found": [],
    }

    score = 0.0
    checks_passed = 0
    total_checks = 2

    # Check 1: Order was placed with at least one item
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

        # Check if there's at least one item
        if all_items:
            details["has_item"] = True
            checks_passed += 1
            logger.info(f"✓ Order has {len(all_items)} item(s)")
        else:
            logger.info("✗ Order has no items")
    else:
        logger.info("✗ No order placed")

    # Check 2: Search was performed for Sony WH-1000XM5
    search_terms = get_search_terms(final_state)
    details["search_terms_found"] = search_terms
    sony_searched = any(
        "sony" in term or "wh-1000xm5" in term or "xm5" in term for term in search_terms
    )
    if sony_searched:
        details["search_performed"] = True
        checks_passed += 1
        logger.info("✓ Search for Sony WH-1000XM5 detected")
    else:
        # Fallback: check if agent's result mentions the search
        if final_result and (
            "sony" in final_result.lower() or "wh-1000xm5" in final_result.lower()
        ):
            details["search_performed"] = True
            checks_passed += 1
            logger.info("✓ Sony WH-1000XM5 mentioned in agent result")
        else:
            logger.info("✗ No search for Sony WH-1000XM5 detected")

    # Calculate final score
    score = checks_passed / total_checks

    # Success requires both: order placed with item AND search performed
    success = details["has_item"] and details["search_performed"]

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
