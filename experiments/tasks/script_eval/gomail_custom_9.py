"""
GoMail Custom Task 9: Send Email to 4 Different People

Task: Draft a new email and send it to any four different people.

Verifier:
- Check for emails with sent: true
- Verify at least 4 unique recipients
- Any content is acceptable
- Partial credit: unique_recipients / 4
"""

import json
import sys
from typing import Any, Dict, List, Set, Tuple


def normalize_str(s: Any) -> str:
    """Normalize string for comparison (strip whitespace, lowercase)."""
    return (s or "").strip().lower()


def get_added_emails(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all added emails from the state.

    Checks both:
    - differences.emails.added (list)
    - initialfinaldiff.added.email.emails (dict)
    """
    added_emails: List[Dict[str, Any]] = []

    # Check differences.emails.added
    differences = data.get("differences") or {}
    emails_diff = differences.get("emails") or {}
    added = emails_diff.get("added") or []

    if isinstance(added, list):
        for msg in added:
            if isinstance(msg, dict):
                added_emails.append(msg)

    # Also check initialfinaldiff.added.email.emails
    init = data.get("initialfinaldiff")
    if isinstance(init, dict):
        added_dict = init.get("added", {}).get("email", {}).get("emails")
        if isinstance(added_dict, dict):
            for v in added_dict.values():
                if isinstance(v, dict):
                    added_emails.append(v)

    return added_emails


def get_sent_email_recipients(emails: List[Dict[str, Any]]) -> Set[str]:
    """
    Get all unique recipients from sent emails.

    Only counts emails with sent: true.
    Handles single recipient (string) and multiple recipients (list).
    """
    recipients: Set[str] = set()

    for email in emails:
        if not isinstance(email, dict):
            continue

        # Must be sent
        if not email.get("sent"):
            continue

        to_field = email.get("to")
        if isinstance(to_field, str):
            recipient = normalize_str(to_field)
            if recipient and "@" in recipient:
                recipients.add(recipient)
        elif isinstance(to_field, list):
            for r in to_field:
                if isinstance(r, str):
                    recipient = normalize_str(r)
                    if recipient and "@" in recipient:
                        recipients.add(recipient)

    return recipients


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

    expected_recipients = 4

    added_emails = get_added_emails(final_state)
    logger.info(f"Found {len(added_emails)} added emails")

    # Get all unique recipients from sent emails
    recipients = get_sent_email_recipients(added_emails)
    recipient_count = len(recipients)

    logger.info(f"Unique recipients from sent emails: {recipient_count}")
    logger.info(f"Recipients: {recipients}")

    # Count sent emails
    sent_count = sum(1 for e in added_emails if isinstance(e, dict) and e.get("sent"))

    # Calculate partial credit
    total_pct = min(recipient_count / expected_recipients, 1.0)
    success = recipient_count >= expected_recipients

    details = {
        "expected_unique_recipients": expected_recipients,
        "unique_recipients_found": recipient_count,
        "recipients": list(recipients),
        "sent_email_count": sent_count,
        "total_emails_added": len(added_emails),
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
