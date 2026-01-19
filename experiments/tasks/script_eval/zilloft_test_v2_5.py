"""
Zilloft Test v2-5: Contact an agent in San Jose, CA to hire

Task: Contact an agent in San Jose, CA to hire.

Verifier:
- Check for contact agent action with valid form submission
- Looks for entries in:
  - initialfinaldiff.added.tourRequests.contactAgentList
  - differences.contactAgents.added
- Valid if formValues contains non-empty email, phone, or name
- Scoring: All or nothing

Reference: experiments/agisdk/src/agisdk/REAL/browsergym/webclones/v2/eval_scripts/eval_zilloft_5.py
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple


def safe_get(d: Any, path: List[str], default: Any = None) -> Any:
    """Safely get a nested value from a dict."""
    cur = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def extract_form_values(container: Any) -> Iterator[Dict]:
    """
    Given a dict container that maps arbitrary keys to entries, each possibly containing
    contactAgentData.formValues, yield the formValues dicts found.
    """
    if not isinstance(container, dict):
        return
    for _k, v in container.items():
        if isinstance(v, dict):
            # Typical path
            fv = safe_get(v, ["contactAgentData", "formValues"], None)
            if isinstance(fv, dict):
                yield fv
            else:
                # In case formValues sits at root of v (defensive)
                if isinstance(v.get("formValues"), dict):
                    yield v.get("formValues")


def fv_has_identity(fv: Any) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if formValues has valid identity fields.
    
    Returns:
        (is_valid, details)
    """
    details = {
        "has_email": False,
        "has_phone": False,
        "has_name": False,
        "email": None,
        "phone": None,
        "name": None,
    }
    
    if not isinstance(fv, dict):
        return False, details
    
    # Consider valid if at least one of email/phone/name is a non-empty string
    for field in ("email", "phone", "name"):
        val = fv.get(field)
        if isinstance(val, str) and val.strip() != "":
            details[f"has_{field}"] = True
            details[field] = val.strip()
    
    is_valid = details["has_email"] or details["has_phone"] or details["has_name"]
    return is_valid, details


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Checks for valid contact agent submission in:
    - initialfinaldiff.added.tourRequests.contactAgentList
    - differences.contactAgents.added

    Returns:
        (success, percentage, details)
    """
    # Paths to check
    contact_list = safe_get(
        final_state, ["initialfinaldiff", "added", "tourRequests", "contactAgentList"], {}
    )
    contact_added = safe_get(final_state, ["differences", "contactAgents", "added"], {})

    valid_submissions = []
    checked_entries = []

    # Check both containers for at least one valid formValues
    containers = [
        ("initialfinaldiff.added.tourRequests.contactAgentList", contact_list),
        ("differences.contactAgents.added", contact_added),
    ]

    for source_name, container in containers:
        if isinstance(container, dict) and container:
            for fv in extract_form_values(container):
                is_valid, fv_details = fv_has_identity(fv)
                entry_info = {
                    "source": source_name,
                    "is_valid": is_valid,
                    **fv_details,
                }
                checked_entries.append(entry_info)
                if is_valid:
                    valid_submissions.append(entry_info)

    success = len(valid_submissions) > 0
    percentage = 1.0 if success else 0.0

    details = {
        "success": success,
        "valid_submissions": len(valid_submissions),
        "checked_entries": checked_entries,
        "first_valid": valid_submissions[0] if valid_submissions else None,
    }

    return success, percentage, details
