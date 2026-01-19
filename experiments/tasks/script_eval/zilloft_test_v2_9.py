"""
Zilloft Test v2-9: Find a house over 1000 square feet

Task: Find me a house over 1000 square feet

Verifier:
- Check for valid tour request entries
- A valid tour request has:
  - An id string
  - requestTourData containing non-empty options list, formValues dict, or shareInfoDetails dict
- Checks both:
  - differences.requestTours.added
  - initialfinaldiff.added.tourRequests.requestTourList
- Scoring: All or nothing based on valid tour request

Reference: experiments/agisdk/src/agisdk/REAL/browsergym/webclones/v2/eval_scripts/eval_zilloft_9.py
"""

from typing import Any, Dict, List, Optional, Tuple


def get_nested(d: Any, path: List[str], default: Any = None) -> Any:
    """Safely get a nested value from a dict."""
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def is_valid_tour_request(entry: Any) -> Tuple[bool, Dict[str, Any]]:
    """
    Determine if an entry looks like a real tour request.
    
    We require: an id string and requestTourData containing either 
    non-empty options list or non-empty formValues dict or shareInfoDetails dict.
    
    Returns:
        (is_valid, details)
    """
    details = {
        "has_id": False,
        "has_options": False,
        "has_form": False,
        "has_share": False,
    }
    
    if not isinstance(entry, dict):
        return False, details
    
    entry_id = entry.get("id")
    if not isinstance(entry_id, str) or not entry_id.strip():
        return False, details
    details["has_id"] = True
    
    rtd = entry.get("requestTourData")
    if not isinstance(rtd, dict):
        return False, details
    
    options = rtd.get("options")
    has_options = isinstance(options, list) and len(options) > 0
    details["has_options"] = has_options
    
    form_vals = rtd.get("formValues")
    has_form = isinstance(form_vals, dict) and len(form_vals) > 0
    details["has_form"] = has_form
    
    share = rtd.get("shareInfoDetails")
    has_share = isinstance(share, dict) and len(share) > 0
    details["has_share"] = has_share
    
    is_valid = has_options or has_form or has_share
    return is_valid, details


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Checks for valid tour requests in:
    - differences.requestTours.added
    - initialfinaldiff.added.tourRequests.requestTourList

    Returns:
        (success, percentage, details)
    """
    # Collect potential tour requests from both observed locations
    candidates = []
    candidate_sources = []

    # Check differences.requestTours.added
    added_req_tours = get_nested(final_state, ["differences", "requestTours", "added"], {})
    if isinstance(added_req_tours, dict):
        for k, v in added_req_tours.items():
            candidates.append(v)
            candidate_sources.append(f"differences.requestTours.added.{k}")

    # Check initialfinaldiff.added.tourRequests.requestTourList
    initial_req_list = get_nested(
        final_state, ["initialfinaldiff", "added", "tourRequests", "requestTourList"], {}
    )
    if isinstance(initial_req_list, dict):
        for k, v in initial_req_list.items():
            candidates.append(v)
            candidate_sources.append(f"initialfinaldiff.added.tourRequests.requestTourList.{k}")

    # Validate candidates
    valid_requests = []
    checked_candidates = []
    
    for i, candidate in enumerate(candidates):
        is_valid, candidate_details = is_valid_tour_request(candidate)
        entry_info = {
            "source": candidate_sources[i] if i < len(candidate_sources) else "unknown",
            "is_valid": is_valid,
            **candidate_details,
        }
        checked_candidates.append(entry_info)
        if is_valid:
            valid_requests.append(entry_info)

    success = len(valid_requests) > 0
    percentage = 1.0 if success else 0.0

    details = {
        "success": success,
        "total_candidates": len(candidates),
        "valid_requests": len(valid_requests),
        "checked_candidates": checked_candidates,
    }

    return success, percentage, details
