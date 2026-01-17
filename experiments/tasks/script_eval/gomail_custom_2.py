"""
GoMail Custom Task 2 Test: Forward and Delete Emails

Task: Forward the first two emails in my inbox to me@example.com,
      each with a short note "Forwarded from my inbox",
      and then delete the two original emails.

Verifier:
- Check that both emails were forwarded to the recipient with the note
- Check that both original emails were deleted (moved to trash)
- Recipient: me@example.com
- First 2 emails identified by subject keywords:
  1. "your ride receipt" (Uber)
  2. "your payment was successful" (Stripe)
- Each forward should have isForward: true OR subject containing "forward/fwd/fw"
- Each forward must contain the note "Forwarded from my inbox" in content
- Scoring: 50% for forwarding (per email), 50% for deleting (per email)
  - Total = (forwarded_count + deleted_count) / 4
"""

import json
import sys
from typing import Any, Dict, List, Set, Tuple

TARGET_RECIPIENTS = ["me@example.com"]
REQUIRED_NOTE = "forwarded from my inbox"

EXPECTED_EMAIL_KEYWORDS = [
    "your ride receipt",
    "your payment was successful",
]


def normalize_str(s: Any) -> str:
    """Normalize string for comparison (strip whitespace, lowercase)."""
    return (s or "").strip().lower()


def get_forward_emails(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all forward emails from the state.

    Searches:
    - state.email.emails (main email list)
    - differences.emails.added
    - initialfinaldiff.added.email.emails
    """
    found_emails: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    def check_and_add(email: Dict[str, Any]):
        """Check if email is a forward and add if not duplicate."""
        if not isinstance(email, dict):
            return

        # Check if it's a forward
        if not is_forward_email(email):
            return

        # Avoid duplicates by id
        email_id = str(email.get("id", ""))
        if email_id and email_id in seen_ids:
            return
        if email_id:
            seen_ids.add(email_id)

        found_emails.append(email)

    # Check state.email.emails (main email list)
    state = data.get("state") or {}
    email_section = state.get("email") or {}
    emails_list = email_section.get("emails") or []
    if isinstance(emails_list, list):
        for email in emails_list:
            check_and_add(email)

    # Check differences.emails.added
    differences = data.get("differences") or {}
    emails_diff = differences.get("emails") or {}
    added = emails_diff.get("added") or []
    if isinstance(added, list):
        for email in added:
            check_and_add(email)

    # Check initialfinaldiff.added.email.emails
    init = data.get("initialfinaldiff")
    if isinstance(init, dict):
        added_dict = init.get("added", {}).get("email", {}).get("emails")
        if isinstance(added_dict, dict):
            for email in added_dict.values():
                check_and_add(email)

    return found_emails


def get_trashed_emails(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all emails that were moved to trash.

    Searches differences.emails.updated for emails with trash: true.
    Enriches with subject from initialstate if available.
    """
    trashed = []

    differences = data.get("differences") or {}
    emails_diff = differences.get("emails") or {}
    updated = emails_diff.get("updated") or []

    # Build a lookup of email ID -> subject from initialstate
    id_to_subject = {}
    initialstate = data.get("initialstate") or {}
    init_email = initialstate.get("email") or {}
    init_emails = init_email.get("emails") or []
    for email in init_emails:
        if isinstance(email, dict):
            email_id = str(email.get("id", ""))
            subject = email.get("subject", "")
            if email_id:
                id_to_subject[email_id] = subject

    for email in updated:
        if isinstance(email, dict) and email.get("trash") is True:
            # Enrich with subject from initialstate if missing
            email_id = str(email.get("id", ""))
            if not email.get("subject") and email_id in id_to_subject:
                email = dict(email)  # Copy to avoid mutation
                email["subject"] = id_to_subject[email_id]
            trashed.append(email)

    return trashed


def is_forward_email(email: Dict[str, Any]) -> bool:
    """Check if an email is a forward."""
    is_forward_flag = email.get("isForward") is True
    subject_lower = normalize_str(email.get("subject", ""))
    has_forward_in_subject = (
        "forward" in subject_lower or "fwd" in subject_lower or "fw:" in subject_lower
    )
    return is_forward_flag or has_forward_in_subject


def has_required_note(email: Dict[str, Any]) -> bool:
    """Check if the email content contains the required note."""
    content = email.get("content") or email.get("body") or ""
    return REQUIRED_NOTE.lower() in content.lower()


def get_matching_keyword(subject: str) -> str | None:
    """Check if subject contains one of the expected keywords. Returns matched keyword or None."""
    subject_lower = normalize_str(subject)
    for keyword in EXPECTED_EMAIL_KEYWORDS:
        if keyword.lower() in subject_lower:
            return keyword
    return None


def get_recipients(email: Dict[str, Any]) -> List[str]:
    """Get normalized list of recipients from an email.

    Handles cases where recipients may be:
    - A list of individual emails: ["a@example.com", "b@example.com"]
    - A list with comma-separated string: ["a@example.com, b@example.com"]
    - A single string: "a@example.com"
    """
    to_list = email.get("to") or []
    if isinstance(to_list, str):
        to_list = [to_list]

    recipients = []
    for item in to_list:
        if isinstance(item, str):
            # Split by comma in case multiple addresses are in one string
            for addr in item.split(","):
                normalized = normalize_str(addr)
                if normalized:
                    recipients.append(normalized)
    return recipients


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Scoring:
    - 2 emails to forward + 2 emails to delete = 4 total actions
    - Score = (forwarded_with_note + deleted) / 4

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    # Get all forward emails
    forward_emails = get_forward_emails(final_state)
    logger.info(f"Found {len(forward_emails)} forward emails")

    # Get all trashed emails
    trashed_emails = get_trashed_emails(final_state)
    logger.info(f"Found {len(trashed_emails)} trashed emails")

    # Track which expected emails were forwarded to the recipient with note
    forwarded_keywords: Set[str] = set()
    forward_details = []

    target_set = {normalize_str(r) for r in TARGET_RECIPIENTS}

    for email in forward_emails:
        subject = email.get("subject", "")
        recipients = get_recipients(email)
        matched_keyword = get_matching_keyword(subject)
        has_note = has_required_note(email)

        # Check if target recipient is in this email's recipients
        matching_recipients = [r for r in recipients if r in target_set]

        forward_details.append(
            {
                "to": email.get("to"),
                "subject": subject,
                "matched_keyword": matched_keyword,
                "matching_recipients": matching_recipients,
                "has_note": has_note,
            }
        )

        # Count if the forward is valid (matches expected email, sent to target, has note)
        if matched_keyword and matching_recipients and has_note:
            forwarded_keywords.add(matched_keyword)

    # Track which expected emails were deleted
    deleted_keywords: Set[str] = set()
    delete_details = []

    for email in trashed_emails:
        subject = email.get("subject", "")
        matched_keyword = get_matching_keyword(subject)

        delete_details.append(
            {
                "id": email.get("id"),
                "subject": subject,
                "matched_keyword": matched_keyword,
            }
        )

        if matched_keyword:
            deleted_keywords.add(matched_keyword)

    # Calculate results
    forwarded_list = [kw for kw in EXPECTED_EMAIL_KEYWORDS if kw in forwarded_keywords]
    not_forwarded_list = [
        kw for kw in EXPECTED_EMAIL_KEYWORDS if kw not in forwarded_keywords
    ]
    deleted_list = [kw for kw in EXPECTED_EMAIL_KEYWORDS if kw in deleted_keywords]
    not_deleted_list = [
        kw for kw in EXPECTED_EMAIL_KEYWORDS if kw not in deleted_keywords
    ]

    logger.info(f"Forwarded: {len(forwarded_list)}/{len(EXPECTED_EMAIL_KEYWORDS)}")
    logger.info(f"Deleted: {len(deleted_list)}/{len(EXPECTED_EMAIL_KEYWORDS)}")

    # Score: 4 total actions (2 forwards + 2 deletes)
    total_actions = len(EXPECTED_EMAIL_KEYWORDS) * 2
    completed_actions = len(forwarded_list) + len(deleted_list)
    total_pct = completed_actions / total_actions

    success = len(forwarded_list) == len(EXPECTED_EMAIL_KEYWORDS) and len(
        deleted_list
    ) == len(EXPECTED_EMAIL_KEYWORDS)

    details = {
        "target_recipients": TARGET_RECIPIENTS,
        "required_note": REQUIRED_NOTE,
        "expected_emails": EXPECTED_EMAIL_KEYWORDS,
        "forwarded_emails": forwarded_list,
        "not_forwarded_emails": not_forwarded_list,
        "deleted_emails": deleted_list,
        "not_deleted_emails": not_deleted_list,
        "forwarded_count": len(forwarded_list),
        "deleted_count": len(deleted_list),
        "total_expected": len(EXPECTED_EMAIL_KEYWORDS),
        "forward_details": forward_details,
        "delete_details": delete_details,
        "total_forwards_found": len(forward_emails),
        "total_trashed_found": len(trashed_emails),
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
    print(details)

    if success:
        print(f"SUCCESS ({pct * 100:.0f}%)")
    else:
        print(f"FAILURE ({pct * 100:.0f}%)")


if __name__ == "__main__":
    main()
