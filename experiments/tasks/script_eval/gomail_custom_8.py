"""
GoMail Custom Task 8: Create Labels

Task: Create three new labels: 'Work', 'Personal', 'Urgent'.

Verifier:
- Check that 3 labels were created with exact names (case-insensitive)
- Expected labels: "Work", "Personal", "Urgent"
- Partial credit: labels_created / 3

NOTE: Label creation may be stored in different locations:
- initialfinaldiff.added.email.labels
- initialfinaldiff.added.ui.labels
- differences.labels.added
- Or via action history with label creation actions
"""

import json
import sys
from typing import Any, Dict, List, Set, Tuple


def normalize_str(s: Any) -> str:
    """Normalize string for comparison (strip whitespace, lowercase)."""
    return (s or "").strip().lower()


# Expected labels (case-insensitive matching)
EXPECTED_LABELS = {"work", "personal", "urgent"}


def find_created_labels(data: Dict[str, Any]) -> Set[str]:
    """
    Find all labels that were created.

    Searches in multiple possible locations:
    - initialfinaldiff.added.settings.customLabels (primary location in GoMail)
    - initialfinaldiff.added.email.labels
    - initialfinaldiff.added.ui.labels
    - differences.labels.added
    - Action history for settings/addCustomLabel
    """
    created_labels: Set[str] = set()

    # Search initialfinaldiff
    initdiff = data.get("initialfinaldiff")
    if isinstance(initdiff, dict):
        added = initdiff.get("added", {})

        # Check settings.customLabels (primary location for GoMail labels)
        settings_labels = added.get("settings", {}).get("customLabels")
        if isinstance(settings_labels, dict):
            for key, val in settings_labels.items():
                if isinstance(val, dict):
                    name = val.get("name") or val.get("label") or key
                    created_labels.add(normalize_str(name))
                elif isinstance(val, str):
                    created_labels.add(normalize_str(val))
        elif isinstance(settings_labels, list):
            for item in settings_labels:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("label")
                    if name:
                        created_labels.add(normalize_str(name))
                elif isinstance(item, str):
                    created_labels.add(normalize_str(item))

        # Check email.labels
        email_labels = added.get("email", {}).get("labels")
        if isinstance(email_labels, dict):
            for key, val in email_labels.items():
                if isinstance(val, dict):
                    name = val.get("name") or val.get("label") or key
                    created_labels.add(normalize_str(name))
                elif isinstance(val, str):
                    created_labels.add(normalize_str(val))
                else:
                    # Key itself might be the label name
                    created_labels.add(normalize_str(key))
        elif isinstance(email_labels, list):
            for item in email_labels:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("label")
                    if name:
                        created_labels.add(normalize_str(name))
                elif isinstance(item, str):
                    created_labels.add(normalize_str(item))

        # Check ui.labels
        ui_labels = added.get("ui", {}).get("labels")
        if isinstance(ui_labels, dict):
            for key, val in ui_labels.items():
                if isinstance(val, dict):
                    name = val.get("name") or val.get("label") or key
                    created_labels.add(normalize_str(name))
                elif isinstance(val, str):
                    created_labels.add(normalize_str(val))
                else:
                    created_labels.add(normalize_str(key))
        elif isinstance(ui_labels, list):
            for item in ui_labels:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("label")
                    if name:
                        created_labels.add(normalize_str(name))
                elif isinstance(item, str):
                    created_labels.add(normalize_str(item))

        # Check direct labels under added
        direct_labels = added.get("labels")
        if isinstance(direct_labels, dict):
            for key, val in direct_labels.items():
                if isinstance(val, dict):
                    name = val.get("name") or val.get("label") or key
                    created_labels.add(normalize_str(name))
                elif isinstance(val, str):
                    created_labels.add(normalize_str(val))
                else:
                    created_labels.add(normalize_str(key))
        elif isinstance(direct_labels, list):
            for item in direct_labels:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("label")
                    if name:
                        created_labels.add(normalize_str(name))
                elif isinstance(item, str):
                    created_labels.add(normalize_str(item))

    # Search differences.labels
    differences = data.get("differences", {})
    diff_labels = differences.get("labels", {})

    if isinstance(diff_labels, dict):
        added_labels = diff_labels.get("added")
        if isinstance(added_labels, dict):
            for key, val in added_labels.items():
                if isinstance(val, dict):
                    name = val.get("name") or val.get("label") or key
                    created_labels.add(normalize_str(name))
                elif isinstance(val, str):
                    created_labels.add(normalize_str(val))
                else:
                    created_labels.add(normalize_str(key))
        elif isinstance(added_labels, list):
            for item in added_labels:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("label")
                    if name:
                        created_labels.add(normalize_str(name))
                elif isinstance(item, str):
                    created_labels.add(normalize_str(item))

    # Search action history for label creation actions
    action_history = data.get("actionhistory", [])
    for action in action_history:
        if not isinstance(action, dict):
            continue
        action_type = action.get("type", "")
        # Look for settings/addCustomLabel actions (GoMail's label creation)
        if action_type == "settings/addCustomLabel":
            payload = action.get("payload", {})
            if isinstance(payload, dict):
                label_obj = payload.get("label", {})
                if isinstance(label_obj, dict):
                    name = label_obj.get("label") or label_obj.get("name")
                    if name:
                        created_labels.add(normalize_str(name))
        # Also check generic label creation patterns
        elif "label" in action_type.lower() and ("create" in action_type.lower() or "add" in action_type.lower()):
            payload = action.get("payload", {})
            if isinstance(payload, dict):
                name = payload.get("name") or payload.get("label") or payload.get("labelName")
                if name:
                    created_labels.add(normalize_str(name))

    return created_labels


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    created_labels = find_created_labels(final_state)
    logger.info(f"Labels found in state: {created_labels}")

    # Check which expected labels were created
    matched_labels = created_labels & EXPECTED_LABELS
    missing_labels = EXPECTED_LABELS - matched_labels
    extra_labels = created_labels - EXPECTED_LABELS

    matched_count = len(matched_labels)
    expected_count = len(EXPECTED_LABELS)

    logger.info(f"Matched labels: {matched_labels}")
    logger.info(f"Missing labels: {missing_labels}")

    # Calculate partial credit
    total_pct = matched_count / expected_count
    success = matched_count >= expected_count

    details = {
        "expected_labels": list(EXPECTED_LABELS),
        "created_labels": list(created_labels),
        "matched_labels": list(matched_labels),
        "missing_labels": list(missing_labels),
        "extra_labels": list(extra_labels),
        "matched_count": matched_count,
        "expected_count": expected_count,
    }

    return success, total_pct, details


def main():
    """CLI entry point for subprocess execution."""
    state_path = sys.argv[1]
    with open(state_path) as f:
        data = json.load(f)
        final_state = data.get("final_state", {})
        final_result = data.get("final_result", "")

    success, pct, details = evaluate(final_state, final_result)

    if success:
        print(f"SUCCESS ({pct * 100:.0f}%)")
    else:
        print(f"FAILURE ({pct * 100:.0f}%)")


if __name__ == "__main__":
    main()
