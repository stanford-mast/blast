"""
Zilloft Test v2-6: Contact agent for Sacramento house tour

Task: Contact an agent for a house in Sacramento that is in the price range 500k - 1.1M.
      It needs to be 4+ bedrooms and 2+ bathrooms. Have the house tour on July 19th around 1pmish.

Verifier:
- Check for contact/tour request entries with selectedDate
- Date must be July 19
- Time must be around 1 PM (12:00 PM to 2:00 PM inclusive)
- Scoring: All or nothing based on date/time match

Reference: experiments/agisdk/src/agisdk/REAL/browsergym/webclones/v2/eval_scripts/eval_zilloft_6.py
"""

from typing import Any, Dict, Iterator, Optional, Tuple


def parse_time_to_minutes(t: Any) -> Optional[int]:
    """Parse time string to minutes since midnight."""
    if not isinstance(t, str):
        return None
    s = t.strip()
    if not s:
        return None
    # Normalize spacing before AM/PM
    s_up = s.upper().replace(".", "")
    ampm = None
    # Ensure there's a space before AM/PM if missing (e.g., 1:00PM)
    if s_up.endswith("AM"):
        ampm = "AM"
        core = s_up[:-2].strip()
    elif s_up.endswith("PM"):
        ampm = "PM"
        core = s_up[:-2].strip()
    else:
        # If no AM/PM marker, cannot confidently parse in this context
        return None
    # Split hours and minutes
    if ":" in core:
        hh_str, mm_str = core.split(":", 1)
        # If minutes include extra, trim non-digits
        mm = ""
        for ch in mm_str:
            if ch.isdigit():
                mm += ch
            else:
                break
        if mm == "":
            mm = "0"
        try:
            hh = int(hh_str.strip())
            mi = int(mm)
        except Exception:
            return None
    else:
        try:
            hh = int(core)
            mi = 0
        except Exception:
            return None
    if not (1 <= hh <= 12) or not (0 <= mi < 60):
        return None
    # Convert to 24h minutes
    if ampm == "AM":
        if hh == 12:
            hh = 0
    else:  # PM
        if hh != 12:
            hh += 12
    return hh * 60 + mi


def is_july_19(date_str: Any) -> bool:
    """Check if date string represents July 19."""
    if not isinstance(date_str, str):
        return False
    ds = date_str.strip()
    if not ds:
        return False
    # Prefer ISO-like: YYYY-MM-DD[...]
    # Extract date part before 'T' if present
    if "T" in ds:
        ds_part = ds.split("T", 1)[0]
    else:
        ds_part = ds
    # Try YYYY-MM-DD
    if "-" in ds_part:
        parts = ds_part.split("-")
        if len(parts) >= 3:
            _y, m, d = parts[0], parts[1], parts[2]
            try:
                m_i = int(m)
                d_i = int(d)
                return m_i == 7 and d_i == 19
            except Exception:
                pass
    # Try MM/DD/YYYY
    if "/" in ds_part:
        parts = ds_part.split("/")
        if len(parts) >= 3:
            try:
                m_i = int(parts[0])
                d_i = int(parts[1])
                return m_i == 7 and d_i == 19
            except Exception:
                pass
    # Fallback: substring heuristics
    if "07-19" in ds_part or "7-19" in ds_part or "07/19" in ds_part or "7/19" in ds_part:
        return True
    # Could be named month
    ds_upper = ds_part.upper()
    if "JUL" in ds_upper and "19" in ds_upper:
        return True
    return False


def iter_dicts(obj: Any) -> Iterator[Dict]:
    """Recursively yield all dicts within obj."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            for d in iter_dicts(v):
                yield d
    elif isinstance(obj, list):
        for item in obj:
            for d in iter_dicts(item):
                yield d


def find_selected_date_entries(data: Dict) -> list:
    """Find all selectedDate entries in the data."""
    entries = []
    # Look for structures with 'selectedDate' nested or as field
    for d in iter_dicts(data):
        # Case 1: contactAgentData contains selectedDate
        if "contactAgentData" in d and isinstance(d["contactAgentData"], dict):
            cad = d["contactAgentData"]
            if isinstance(cad.get("selectedDate"), dict):
                entries.append(cad["selectedDate"])
        # Case 2: selectedDate directly in dict
        if isinstance(d.get("selectedDate"), dict):
            entries.append(d["selectedDate"])
    return entries


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Checks for:
    - Contact/tour request with selectedDate on July 19
    - Time around 1 PM (12:00 PM to 2:00 PM inclusive = 720-840 minutes)

    Returns:
        (success, percentage, details)
    """
    # Collect all selectedDate dicts
    selected_dates = find_selected_date_entries(final_state)

    success_found = False
    matching_entry = None
    checked_entries = []

    for sd in selected_dates:
        date_val = sd.get("date")
        time_val = sd.get("time")

        entry_info = {
            "date": date_val,
            "time": time_val,
            "is_july_19": False,
            "time_in_range": False,
            "time_minutes": None,
        }

        if not date_val or not time_val:
            checked_entries.append(entry_info)
            continue

        entry_info["is_july_19"] = is_july_19(date_val)
        if not entry_info["is_july_19"]:
            checked_entries.append(entry_info)
            continue

        mins = parse_time_to_minutes(time_val)
        entry_info["time_minutes"] = mins
        if mins is None:
            checked_entries.append(entry_info)
            continue

        # Around 1 PM: accept 12:00 PM (720) to 2:00 PM (840) inclusive
        entry_info["time_in_range"] = 720 <= mins <= 840
        if entry_info["time_in_range"]:
            success_found = True
            matching_entry = entry_info

        checked_entries.append(entry_info)

    percentage = 1.0 if success_found else 0.0

    details = {
        "success": success_found,
        "selected_dates_found": len(selected_dates),
        "checked_entries": checked_entries,
        "matching_entry": matching_entry,
        "expected_date": "July 19",
        "expected_time_range": "12:00 PM - 2:00 PM (720-840 minutes)",
    }

    return success_found, percentage, details
