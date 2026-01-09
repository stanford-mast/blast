"""
Common utility functions for eval scripts.
"""

from typing import Any, Dict, List, Optional, Tuple


class ActionType:
    """Action type constants organized by category."""

    class Router:
        """Router-related actions."""

        PREFIX = "router"
        LOCATION_CHANGE = "router/locationChange"

    class UI:
        """UI-related actions."""

        PREFIX = "ui"
        OPEN_PRODUCT_MODAL = "ui/openProductModal"
        SET_SEARCH_TERM = "ui/setSearchTerm"

    class Cart:
        """Cart-related actions."""

        PREFIX = "cart"


def get_actions(state: dict, action_type: Optional[type] = None) -> List[dict]:
    """
    Return all actions from the action history, optionally filtered by type prefix.

    Args:
        state: The state dictionary containing actionhistory
        action_type: Optional ActionType class (e.g., ActionType.Router, ActionType.UI, ActionType.Cart).
                If None, returns all actions. Filtering uses startswith on the action type PREFIX.

    Returns:
        List of action dictionaries matching the filter

    Examples:
        get_actions(state)  # All actions
        get_actions(state, ActionType.Router)  # Only router/* actions
        get_actions(state, ActionType.UI)  # Only ui/* actions
        get_actions(state, ActionType.Cart)  # Only cart/* actions
    """
    actions = state.get("actionhistory", [])
    if action_type is None:
        return actions
    return [
        action
        for action in actions
        if action.get("type", "").startswith(f"{action_type.PREFIX}/")
    ]


def get_all_visited_paths(state: dict) -> List[str]:
    """
    Return a list of all visited paths from the action history.

    This is a convenience wrapper around get_actions() for router/locationChange actions.
    """
    router_actions = get_actions(state, ActionType.Router)
    paths = []
    for action in router_actions:
        if action["type"] == ActionType.Router.LOCATION_CHANGE:
            paths.append(action["payload"]["location"]["pathname"])
    return paths


def check_path(state: dict, paths: List[str], verbose: bool = False) -> bool:
    """
    Check that each given path is visited.

    Args:
        state: The final state dictionary containing actionhistory
        paths: List of paths to verify were visited
        verbose: If True, prints debug information about missing paths

    Returns:
        True if all paths were visited, False otherwise
    """
    all_visited_paths = get_all_visited_paths(state)
    all_present = True
    for path in paths:
        if path not in all_visited_paths:
            if verbose:
                print(f"Path {path} not visited")
            all_present = False
    return all_present


def count_visited_paths(state: dict, paths: List[str]) -> int:
    """
    Count how many of the given paths were visited.

    Args:
        state: The final state dictionary containing actionhistory
        paths: List of paths to check

    Returns:
        Number of paths that were visited
    """
    all_visited_paths = get_all_visited_paths(state)
    return sum(1 for path in paths if path in all_visited_paths)


def extract_orders(data: dict) -> List[dict]:
    """
    Extract all orders from the data.

    This helper extracts orders from multiple possible locations in the state:
    - differences.foodOrders.added
    - initialfinaldiff.added.cart.foodOrders
    - Recursive scan of initialfinaldiff.added for order-like objects

    Orders are identified by having 'orderId', 'cartItems', AND 'checkoutDetails'.
    The 'orderId' requirement distinguishes actual placed orders from the cart object itself
    (which has cartItems and checkoutDetails but no orderId until the order is placed).
    Duplicates are removed based on orderId.
    """
    orders = []
    # Primary: differences.foodOrders.added
    dif_orders = data.get("differences", {}).get("foodOrders", {}).get("added", {})
    if isinstance(dif_orders, dict):
        for v in dif_orders.values():
            if isinstance(v, dict) and "cartItems" in v and "checkoutDetails" in v:
                orders.append(v)
    # Secondary: initialfinaldiff.added.cart.foodOrders
    init_cart_orders = (
        data.get("initialfinaldiff", {})
        .get("added", {})
        .get("cart", {})
        .get("foodOrders", {})
    )
    if isinstance(init_cart_orders, dict):
        for v in init_cart_orders.values():
            if isinstance(v, dict) and "cartItems" in v and "checkoutDetails" in v:
                orders.append(v)

    # Fallback: scan recursively for any object that looks like an order under 'added'
    # An actual placed order must have an 'orderId'.
    def rec(obj):
        if isinstance(obj, dict):
            # A placed order must have orderId, cartItems, and checkoutDetails
            # The cart object itself has cartItems/checkoutDetails but no orderId
            if "orderId" in obj and "cartItems" in obj and "checkoutDetails" in obj:
                orders.append(obj)
            for vv in obj.values():
                rec(vv)
        elif isinstance(obj, list):
            for vv in obj:
                rec(vv)

    rec(data.get("initialfinaldiff", {}).get("added", {}))

    # Deduplicate by orderId if present to avoid duplicates from multiple paths
    seen = set()
    unique_orders = []
    for o in orders:
        oid = o.get("orderId")
        key = ("OID", oid) if oid is not None else ("OBJ", id(o))
        if key not in seen:
            seen.add(key)
            unique_orders.append(o)
    return unique_orders


def check_orders(final_state: dict, keyword: str) -> bool:
    """Check that exactly one order matching the keyword is placed."""
    match_count = 0
    for order in extract_orders(final_state):
        if not isinstance(order["cartItems"], list) or len(order["cartItems"]) == 0:
            continue
        for item in order["cartItems"]:
            if not isinstance(item, dict) or "name" not in item:
                continue
            if keyword.lower() in item["name"].lower():
                match_count += 1
    return match_count == 1


def count_orders_with_keyword(final_state: dict, keyword: str) -> int:
    """Count orders containing items matching the keyword."""
    match_count = 0
    for order in extract_orders(final_state):
        if not isinstance(order["cartItems"], list) or len(order["cartItems"]) == 0:
            continue
        for item in order["cartItems"]:
            if not isinstance(item, dict) or "name" not in item:
                continue
            if keyword.lower() in item["name"].lower():
                match_count += 1
    return match_count


def get_cart_items(final_state: dict) -> List[dict]:
    """Get all items from the cart."""
    return final_state.get("finalstate", {}).get("cart", {}).get("cartItems", [])


def get_cart_checkout_details(final_state: dict) -> dict:
    """Get the checkout details from the cart."""
    return final_state.get("finalstate", {}).get("cart", {}).get("checkoutDetails", {})


def check_cart_item(final_state: dict, keyword: str) -> bool:
    """Check that the cart contains an item matching the keyword."""
    match_count = 0
    for item in get_cart_items(final_state):
        if not isinstance(item, dict):
            continue
        item_name = item.get("name", "")
        if keyword.lower() in item_name.lower():
            match_count += 1
    return match_count == 1


# Type alias for eval result
EvalResult = Tuple[bool, float, Dict[str, Any]]
