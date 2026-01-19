"""
Zilloft Test v2-10: Find House and Book Tour

Task: Find me a house under $500,000 with at least 2 bedrooms and book a tour

Verifier:
- Check for valid tour requests
- Validate search filters (price < $500,000, bedrooms >= 2)
- Check property data meets constraints
- Scoring: All or nothing

Reference: experiments/agisdk/src/agisdk/REAL/browsergym/webclones/v2/eval_scripts/eval_zilloft_10.py
"""

import re
from typing import Any, Dict, List, Tuple


def has_valid_tour_request(container: Any) -> List[Dict]:
    """Check if container has valid tour requests and return them."""
    valid_requests = []
    if not isinstance(container, dict):
        return valid_requests

    for _, entry in container.items():
        if not isinstance(entry, dict):
            continue
        rtd = entry.get("requestTourData")
        if isinstance(rtd, dict):
            form = rtd.get("formValues")
            msg = form.get("message") if isinstance(form, dict) else None
            options = rtd.get("options")
            if (isinstance(msg, str) and msg.strip()) or (
                isinstance(options, list) and len(options) > 0
            ):
                valid_requests.append(entry)
    return valid_requests


def extract_property_from_message(message: Any) -> Dict | None:
    """Extract property address from tour request message."""
    if not isinstance(message, str):
        return None

    patterns = [
        r"interested in (.+?)(?:\.|$)",
        r"tour (?:of |for )?(.+?)(?:\.|$)",
        r"viewing (.+?)(?:\.|$)",
        r"schedule.*?(?:for|at) (.+?)(?:\.|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            address = match.group(1).strip()
            if len(address) > 10 and any(char.isdigit() for char in address):
                return {"address": address, "from_message": True}

    return None


def check_search_filters(data: Dict) -> Dict[str, Any]:
    """Check if proper search filters were applied."""
    filters_found = {
        "price_set": False,
        "price_under_500k": False,
        "bedrooms_set": False,
        "bedrooms_min_2": False,
        "max_price_value": None,
        "min_bedrooms_value": None,
    }

    def search_filters(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                k_lower = str(k).lower()

                # Check for price filters
                if any(
                    term in k_lower
                    for term in [
                        "maxprice",
                        "max_price",
                        "pricemax",
                        "price_max",
                        "upperprice",
                    ]
                ):
                    filters_found["price_set"] = True
                    try:
                        if isinstance(v, (int, float)):
                            filters_found["max_price_value"] = v
                            if v <= 500000:
                                filters_found["price_under_500k"] = True
                        elif isinstance(v, str):
                            nums = re.findall(r"\d+", v.replace(",", ""))
                            if nums:
                                price_val = int(nums[0])
                                filters_found["max_price_value"] = price_val
                                if price_val <= 500000:
                                    filters_found["price_under_500k"] = True
                    except Exception:
                        pass

                # Check for bedroom filters
                if any(
                    term in k_lower
                    for term in [
                        "bedroom",
                        "bed",
                        "minbedroom",
                        "min_bedroom",
                        "bedrooms_min",
                        "minbeds",
                    ]
                ):
                    filters_found["bedrooms_set"] = True
                    try:
                        if isinstance(v, (int, float)):
                            filters_found["min_bedrooms_value"] = v
                            if v >= 2:
                                filters_found["bedrooms_min_2"] = True
                        elif isinstance(v, str):
                            nums = re.findall(r"\d+", v)
                            if nums:
                                bed_val = int(nums[0])
                                filters_found["min_bedrooms_value"] = bed_val
                                if bed_val >= 2:
                                    filters_found["bedrooms_min_2"] = True
                    except Exception:
                        pass

                try:
                    search_filters(v)
                except Exception:
                    pass
        elif isinstance(obj, list):
            for item in obj:
                try:
                    search_filters(item)
                except Exception:
                    pass

    try:
        search_filters(data)
    except Exception:
        pass

    return filters_found


def find_homes_data(data: Dict) -> List[Dict]:
    """Search for home/property data in various locations in the JSON."""
    homes = []

    def search_homes(obj):
        try:
            if isinstance(obj, dict):
                has_price = "price" in obj or "listPrice" in obj
                has_beds = "bedrooms" in obj or "beds" in obj
                has_address = "address" in obj

                if has_price or has_beds or has_address:
                    homes.append(obj)

                for _k, v in obj.items():
                    search_homes(v)
            elif isinstance(obj, list):
                for item in obj:
                    search_homes(item)
        except Exception:
            pass

    try:
        search_homes(data)
    except Exception:
        pass

    return homes


def validate_property_constraints(
    homes: List[Dict], filters: Dict, tour_requests: List[Dict]
) -> bool:
    """Validate that tour requests are for properties meeting the constraints."""
    # If we have explicit filter violations, fail
    if filters["price_set"] and not filters["price_under_500k"]:
        return False
    if filters["bedrooms_set"] and not filters["bedrooms_min_2"]:
        return False

    # If filters are properly set, we can pass
    if filters["price_under_500k"] and filters["bedrooms_min_2"]:
        return True

    # Check if tour requests contain specific property addresses
    has_specific_properties = False
    for tour_req in tour_requests:
        try:
            form_values = tour_req.get("requestTourData", {}).get("formValues", {})
            message = form_values.get("message", "")

            property_info = extract_property_from_message(message)
            if property_info:
                has_specific_properties = True
                break
        except Exception:
            pass

    if has_specific_properties and tour_requests:
        # Check if home data violates constraints
        for home in homes:
            price = home.get("price") or home.get("listPrice")
            bedrooms = home.get("bedrooms") or home.get("beds")

            if price is not None:
                try:
                    if isinstance(price, (int, float)):
                        price_val = price
                    elif isinstance(price, str):
                        nums = re.findall(
                            r"\d+", price.replace(",", "").replace("$", "")
                        )
                        if nums:
                            price_val = int(nums[0])
                        else:
                            continue

                    if price_val > 500000:
                        return False
                except Exception:
                    pass

            if bedrooms is not None:
                try:
                    if isinstance(bedrooms, (int, float)):
                        bed_val = bedrooms
                    elif isinstance(bedrooms, str):
                        nums = re.findall(r"\d+", bedrooms)
                        if nums:
                            bed_val = int(nums[0])
                        else:
                            continue

                    if bed_val < 2:
                        return False
                except Exception:
                    pass

        return True

    # Check home data for violations
    for home in homes:
        price = home.get("price") or home.get("listPrice")
        bedrooms = home.get("bedrooms") or home.get("beds")

        if price is not None:
            try:
                if isinstance(price, (int, float)):
                    price_val = price
                elif isinstance(price, str):
                    nums = re.findall(r"\d+", price.replace(",", "").replace("$", ""))
                    if nums:
                        price_val = int(nums[0])
                    else:
                        continue

                if price_val > 500000:
                    return False
            except Exception:
                pass

        if bedrooms is not None:
            try:
                if isinstance(bedrooms, (int, float)):
                    bed_val = bedrooms
                elif isinstance(bedrooms, str):
                    nums = re.findall(r"\d+", bedrooms)
                    if nums:
                        bed_val = int(nums[0])
                    else:
                        continue

                if bed_val < 2:
                    return False
            except Exception:
                pass

    # If we couldn't find definitive evidence either way,
    # require that filters were at least set properly
    if not filters["price_under_500k"] or not filters["bedrooms_min_2"]:
        return False

    return True


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Checks for:
    - Valid tour requests
    - Search filters (price < $500,000, bedrooms >= 2)
    - Property data meets constraints

    Returns:
        (success, percentage, details)
    """
    # Check for tour requests
    tour_requests = []

    # Check initialfinaldiff.added.tourRequests.requestTourList
    try:
        init_added = (
            final_state.get("initialfinaldiff", {})
            .get("added", {})
            .get("tourRequests", {})
            .get("requestTourList")
        )
        if isinstance(init_added, dict):
            tour_requests.extend(has_valid_tour_request(init_added))
    except Exception:
        pass

    # Check differences.requestTours.added
    try:
        diff_added = (
            final_state.get("differences", {}).get("requestTours", {}).get("added")
        )
        if isinstance(diff_added, dict):
            tour_requests.extend(has_valid_tour_request(diff_added))
    except Exception:
        pass

    has_tour_request = len(tour_requests) > 0

    # Check search filters
    filters = check_search_filters(final_state)

    # Find any home/property data
    homes = find_homes_data(final_state)

    # Validate that the property meets constraints
    try:
        meets_constraints = validate_property_constraints(homes, filters, tour_requests)
    except Exception:
        meets_constraints = False

    success = has_tour_request and meets_constraints
    percentage = 1.0 if success else 0.0

    details = {
        "has_tour_request": has_tour_request,
        "tour_requests_count": len(tour_requests),
        "filters": filters,
        "homes_found": len(homes),
        "meets_constraints": meets_constraints,
    }

    return success, percentage, details
