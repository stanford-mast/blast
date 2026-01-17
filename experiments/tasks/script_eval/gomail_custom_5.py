"""
GoMail Custom Task 5: Send Christmas Emails to 4 People

Task: Write an email to each of 'alice', 'bob', 'charlie', 'david'@example.com
with personalized greetings and a simple Merry Christmas message.

Verifier:
- Check that 4 emails were sent (sent: true)
- Each to a unique recipient from the expected list
- Content contains "christmas" or "merry christmas"
- Partial credit: successful_emails / 4
"""

import json
import re
import sys
from typing import Any, Dict, List, Set, Tuple


def normalize_str(s: Any) -> str:
    """Normalize string for comparison (strip whitespace, lowercase)."""
    return (s or "").strip().lower()


def strip_html_tags(s: str) -> str:
    """Strip HTML tags from a string and normalize whitespace."""
    if not s:
        return ""
    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", "", s)
    # Normalize whitespace
    clean = " ".join(clean.split())
    return clean.strip()


# Expected recipients
EXPECTED_RECIPIENTS = {
    "alice@example.com",
    "bob@example.com",
    "charlie@example.com",
    "david@example.com",
}


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


def get_email_recipients(email: Dict[str, Any]) -> Set[str]:
    """Get normalized recipient emails from an email."""
    to_list = email.get("to") or []
    if isinstance(to_list, str):
        to_list = [to_list]

    return {normalize_str(t) for t in to_list if isinstance(t, str)}


def has_christmas_content(email: Dict[str, Any]) -> bool:
    """Check if email content mentions Christmas."""
    content = email.get("content") or ""
    subject = email.get("subject") or ""

    # Check both content and subject
    combined = strip_html_tags(content).lower() + " " + subject.lower()

    return "christmas" in combined or "merry" in combined


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

    added_emails = get_added_emails(final_state)
    logger.info(f"Found {len(added_emails)} added emails")

    # Track successful emails
    successful_recipients: Set[str] = set()
    christmas_emails: List[Dict[str, Any]] = []
    missing_christmas_content: List[str] = []

    for email in added_emails:
        # Must be sent
        if not email.get("sent"):
            continue

        recipients = get_email_recipients(email)

        # Check if any recipient is in our expected list
        matching_recipients = recipients & EXPECTED_RECIPIENTS

        for recipient in matching_recipients:
            if recipient in successful_recipients:
                continue  # Already counted this recipient

            # Check for Christmas content
            if has_christmas_content(email):
                successful_recipients.add(recipient)
                christmas_emails.append(
                    {
                        "to": list(recipients),
                        "subject": email.get("subject"),
                        "has_christmas_content": True,
                    }
                )
            else:
                missing_christmas_content.append(recipient)

    successful_count = len(successful_recipients)
    expected_count = len(EXPECTED_RECIPIENTS)

    logger.info(f"Successful Christmas emails: {successful_count}/{expected_count}")
    logger.info(f"Recipients covered: {successful_recipients}")

    # Calculate partial credit
    total_pct = successful_count / expected_count
    success = successful_count >= expected_count

    # Identify missing recipients
    missing_recipients = EXPECTED_RECIPIENTS - successful_recipients

    details = {
        "expected_recipients": list(EXPECTED_RECIPIENTS),
        "successful_recipients": list(successful_recipients),
        "missing_recipients": list(missing_recipients),
        "missing_christmas_content": missing_christmas_content,
        "successful_count": successful_count,
        "expected_count": expected_count,
        "christmas_emails": christmas_emails,
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
