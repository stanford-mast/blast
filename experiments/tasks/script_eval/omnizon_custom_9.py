"""
Omnizon Custom Task 9: Gaming Collection Purchase

Task: I want to create a gaming collection - buy any gaming device under $100.

This script evaluator is based on:
experiments/agisdk/src/agisdk/REAL/browsergym/webclones/v2/eval_scripts/eval_omnizon_9.py

Criteria:
1. Has a purchase (order with items)
2. Price under $100
3. Gaming device (via search terms, known IDs, product name, or agent description if purchase exists)
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


def extract_orders(ifd: Dict) -> List[Dict]:
    """Extract orders from initialfinaldiff."""
    orders = []
    if not isinstance(ifd, dict):
        return orders
    for section in ("added", "updated"):
        sec = ifd.get(section)
        if not isinstance(sec, dict):
            continue
        order_obj = sec.get("order")
        if not isinstance(order_obj, dict):
            continue
        orders_dict = order_obj.get("orders")
        if isinstance(orders_dict, dict):
            for v in orders_dict.values():
                if isinstance(v, dict):
                    orders.append(v)
    return orders


def extract_search_terms(ifd: Dict) -> List[str]:
    """Extract search terms from initialfinaldiff."""
    terms = []
    if not isinstance(ifd, dict):
        return terms
    for section in ("updated", "added"):
        sec = ifd.get(section)
        if not isinstance(sec, dict):
            continue
        filt = sec.get("filter")
        if not isinstance(filt, dict):
            continue
        for key in ("searchQuery", "searchInputValue"):
            val = filt.get(key)
            if isinstance(val, str):
                terms.append(val)
    return terms


# Known gaming device IDs observed in training data
KNOWN_GAMING_IDS = {"16", "192", "193", 16, 192, 193}


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Criteria:
    1. Has a purchase (order with items) - 34%
    2. Price under $100 - 33%
    3. Gaming device (via search terms or known IDs) - 33%

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    details = {
        "has_purchase": False,
        "price_ok": False,
        "is_gaming": False,
        "gaming_intent": False,
        "gaming_by_id": False,
        "gaming_by_name": False,
        "gaming_in_result": False,
        "search_terms": [],
        "order_count": 0,
        "items_found": [],
    }

    ifd = final_state.get("initialfinaldiff")
    orders = extract_orders(ifd)
    details["order_count"] = len(orders)

    # Determine gaming intent via search terms
    terms = extract_search_terms(ifd)
    details["search_terms"] = terms
    gaming_intent = any(isinstance(t, str) and ("gaming" in t.lower()) for t in terms)
    details["gaming_intent"] = gaming_intent

    has_purchase = False
    price_ok = False
    gaming_by_id = False
    gaming_by_name = False

    for order in orders:
        items = order.get("items")
        if isinstance(items, list) and len(items) > 0:
            has_purchase = True
            for it in items:
                if isinstance(it, dict):
                    item_name = it.get("name", it.get("title", ""))
                    item_info = {
                        "id": it.get("id"),
                        "name": item_name if item_name else "unknown",
                        "price": it.get("price"),
                    }
                    details["items_found"].append(item_info)

                    # Check ID against known gaming IDs
                    item_id = it.get("id")
                    if item_id in KNOWN_GAMING_IDS:
                        gaming_by_id = True
                        logger.info(f"✓ Found known gaming ID: {item_id}")

                    # Check if product name contains "gaming"
                    if item_name and "gaming" in item_name.lower():
                        gaming_by_name = True
                        logger.info(f"✓ Found 'gaming' in product name: {item_name}")

                    # Check price < 100
                    price = it.get("price")
                    if isinstance(price, (int, float)) and price < 100:
                        price_ok = True
                        logger.info(f"✓ Price under $100: ${price}")

        # Fallback to order total if item prices absent
        if not price_ok:
            total = order.get("total")
            if isinstance(total, (int, float)) and total < 100:
                price_ok = True
                logger.info(f"✓ Order total under $100: ${total}")

    # Check agent's final result for "gaming" - but ONLY if there's a purchase
    # This handles cases where product name isn't in order data but agent bought a gaming product
    gaming_in_result = False
    if has_purchase and final_result and "gaming" in final_result.lower():
        gaming_in_result = True
        logger.info("✓ Found 'gaming' in agent's result (with valid purchase)")

    details["has_purchase"] = has_purchase
    details["price_ok"] = price_ok
    details["gaming_by_id"] = gaming_by_id
    details["gaming_by_name"] = gaming_by_name
    details["gaming_in_result"] = gaming_in_result
    details["is_gaming"] = (
        gaming_intent or gaming_by_id or gaming_by_name or gaming_in_result
    )

    # Log results
    logger.info(f"Has purchase: {has_purchase}")
    logger.info(f"Price OK (<$100): {price_ok}")
    logger.info(f"Gaming intent (search): {gaming_intent}")
    logger.info(f"Gaming by ID: {gaming_by_id}")
    logger.info(f"Gaming by name: {gaming_by_name}")
    logger.info(f"Gaming in result: {gaming_in_result}")

    # Calculate score
    score = 0.0
    if has_purchase:
        score += 0.34
    if price_ok:
        score += 0.33
    if gaming_intent or gaming_by_id or gaming_by_name or gaming_in_result:
        score += 0.33

    success = (
        has_purchase
        and price_ok
        and (gaming_intent or gaming_by_id or gaming_by_name or gaming_in_result)
    )

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
