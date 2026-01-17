"""
GoMail Custom Task 4: Clear Visible Emails

Task: Clear all my emails that are visible please!

Verifier:
- Check that approximately 50 emails were moved to trash (typical visible page size)
- Success if trash count is between 1 and 50 (inclusive)
"""

import json
import sys
from typing import Any, Dict, Tuple

# Typical visible page size
MAX_VISIBLE_EMAILS = 50


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

    # Extract updated email changes
    differences = final_state.get("differences", {})
    emails = differences.get("emails", {})
    updated_list = emails.get("updated", [])

    # Count emails that were moved to trash
    trash_updates = [
        u for u in updated_list if isinstance(u, dict) and u.get("trash") is True
    ]
    trash_count = len(trash_updates)
    logger.info(f"Trash count: {trash_count}")

    details = {
        "trash_count": trash_count,
        "max_visible_emails": MAX_VISIBLE_EMAILS,
    }
    print(details)

    # Full credit for exactly 50 emails
    if trash_count == MAX_VISIBLE_EMAILS:
        return True, 1.0, details

    # Partial credit for any non-zero count that isn't exactly 50
    if trash_count >= 1:
        details["partial_credit"] = True
        details["note"] = (
            f"Partial credit: trashed {trash_count}/{MAX_VISIBLE_EMAILS} emails"
        )
        return True, 0.5, details

    # No credit for 0
    details["failure_reason"] = f"Trashed {trash_count} emails, expected at least 1"
    return False, 0.0, details


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
