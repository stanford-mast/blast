"""
GoMail Custom Task 3: Compose Email with Specific Content

Task: Compose a new email to jonathan.smith@example.com with the subject
"Meeting Notes" and body "Please find the meeting notes attached."

Verifier (based on v1/gomail-3):
1. Check recipient is jonathan.smith@example.com
2. Check subject is "Meeting Notes"
3. Check body content contains "Please find the meeting notes attached."
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
    clean = re.sub(r"<[^>]+>", "", s)
    # Normalize whitespace
    clean = " ".join(clean.split())
    return clean.strip()


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


def check_email_recipient(email: Dict[str, Any], expected: str) -> bool:
    """Check if email was sent to the expected recipient."""
    to_list = email.get("to") or []
    if isinstance(to_list, str):
        to_list = [to_list]

    for recipient in to_list:
        if normalize_str(recipient) == expected.lower():
            return True
    return False


def check_email_subject(email: Dict[str, Any], expected: str) -> bool:
    """Check if email has the expected subject."""
    subject = email.get("subject") or ""
    # Case-insensitive comparison, trimmed
    return normalize_str(subject) == expected.lower()


def check_email_content(email: Dict[str, Any], expected: str) -> bool:
    """
    Check if email has the expected content.

    The content may be wrapped in HTML tags (e.g., <p>content</p>),
    so we strip HTML and compare the text content.
    """
    content = email.get("content") or ""

    # Strip HTML and normalize
    clean_content = strip_html_tags(content).lower()
    expected_clean = expected.lower().strip()

    return clean_content == expected_clean


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

    expected_recipient = "jonathan.smith@example.com"
    expected_subject = "Meeting Notes"
    expected_body = "Please find the meeting notes attached."

    added_emails = get_added_emails(final_state)
    logger.info(f"Found {len(added_emails)} added emails")

    # Check each added email for the expected values
    recipient_ok = False
    subject_ok = False
    content_ok = False
    matching_email = None

    for email in added_emails:
        # Only consider sent emails or drafts that could be the composed email
        r_ok = check_email_recipient(email, expected_recipient)
        s_ok = check_email_subject(email, expected_subject)
        c_ok = check_email_content(email, expected_body)

        # Track individual criteria matches
        if r_ok:
            recipient_ok = True
        if s_ok:
            subject_ok = True
        if c_ok:
            content_ok = True

        # If all criteria match for this email, we have a match
        if r_ok and s_ok and c_ok:
            matching_email = email
            break

    # Calculate partial credit
    # - 1/3 for correct recipient
    # - 1/3 for correct subject
    # - 1/3 for correct content
    total_pct = 0.0
    if recipient_ok:
        total_pct += 1 / 3
    if subject_ok:
        total_pct += 1 / 3
    if content_ok:
        total_pct += 1 / 3

    success = matching_email is not None
    logger.info(
        f"Email composition check: recipient={recipient_ok}, subject={subject_ok}, content={content_ok}"
    )

    details = {
        "recipient_correct": recipient_ok,
        "subject_correct": subject_ok,
        "content_correct": content_ok,
        "expected_recipient": expected_recipient,
        "expected_subject": expected_subject,
        "expected_body": expected_body,
        "emails_found": len(added_emails),
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
