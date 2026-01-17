"""
GoMail Custom Task 6: Reply with Unsubscribe

Task: Reply to each of the first four emails in my inbox with "unsubscribe"

Verifier:
- Check for 4 reply emails with isReply: true and sent: true
- Content should contain "unsubscribe"
- Partial credit: replies_found / 4
"""

import json
import re
import sys
from typing import Any, Dict, List, Tuple


def normalize_str(s: Any) -> str:
    """Normalize string for comparison (strip whitespace, lowercase)."""
    return (s or "").strip().lower()


def strip_html_tags(s: str) -> str:
    """Strip HTML tags from a string and normalize whitespace."""
    if not s:
        return ""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', s)
    # Normalize whitespace
    clean = ' '.join(clean.split())
    return clean.strip()


def get_added_emails(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all added emails from the state.

    Checks both:
    - differences.emails.added (list)
    - initialfinaldiff.added.email.emails (dict)
    - Also checks for nested replies in email threads
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
                # Check for nested replies
                replies = msg.get("replies")
                if isinstance(replies, list):
                    for rep in replies:
                        if isinstance(rep, dict):
                            added_emails.append(rep)

    # Also check initialfinaldiff.added.email.emails
    init = data.get("initialfinaldiff")
    if isinstance(init, dict):
        added_dict = init.get("added", {}).get("email", {}).get("emails")
        if isinstance(added_dict, dict):
            for v in added_dict.values():
                if isinstance(v, dict):
                    added_emails.append(v)
                    # Check for nested replies
                    replies = v.get("replies")
                    if isinstance(replies, list):
                        for rep in replies:
                            if isinstance(rep, dict):
                                added_emails.append(rep)

    return added_emails


def is_unsubscribe_reply(email: Dict[str, Any]) -> bool:
    """
    Check if an email is a reply containing "unsubscribe".

    Criteria:
    - sent: true
    - isReply: true
    - content contains "unsubscribe"
    """
    if not isinstance(email, dict):
        return False

    # Must be sent
    if not email.get("sent"):
        return False

    # Must be a reply
    if not email.get("isReply"):
        return False

    # Content must contain "unsubscribe"
    content = email.get("content") or ""
    clean_content = strip_html_tags(content).lower()

    return "unsubscribe" in clean_content


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

    expected_replies = 4

    added_emails = get_added_emails(final_state)
    logger.info(f"Found {len(added_emails)} added emails")

    # Count unsubscribe replies
    unsubscribe_replies = 0
    reply_details: List[Dict[str, Any]] = []

    for email in added_emails:
        if is_unsubscribe_reply(email):
            unsubscribe_replies += 1
            reply_details.append({
                "to": email.get("to"),
                "subject": email.get("subject"),
                "sent": email.get("sent"),
                "isReply": email.get("isReply"),
            })

    logger.info(f"Unsubscribe replies found: {unsubscribe_replies}/{expected_replies}")

    # Calculate partial credit
    total_pct = min(unsubscribe_replies / expected_replies, 1.0)
    success = unsubscribe_replies >= expected_replies

    details = {
        "expected_replies": expected_replies,
        "unsubscribe_replies_found": unsubscribe_replies,
        "reply_details": reply_details,
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
