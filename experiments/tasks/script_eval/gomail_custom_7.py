"""
GoMail Custom Task 7 Test: Delete Emails by Sender Category

Task: Delete all emails from 'Alerts', 'Newsletter', or 'Notifications'.

Verifier:
- Check that all emails from these sender categories were deleted (moved to trash)
- Sender categories are matched by the "from" display name (not email address)
- Scoring: (emails_deleted) / (total_expected)

Expected emails to delete:

Alerts (2):
- "Your Ride Receipt: Thank You for Riding with Uber" (alerts@uber.com)
- "Suspicious Activity Detected on Your Account" (alerts@bankofamerica.com)

Newsletter (7):
- "New Leadership Articles You Can't Miss"
- "Breaking News: Election Results Are In"
- "Latest in Tech: August Edition"
- "Today's Headlines: Top Stories from Around the World"
- "Breaking News: Major Event Happening Now"
- "Weekly Tech News Roundup"
- "Top Tracks for Your Week"

Notifications (7):
- "Your LinkedIn Profile Views are Up This Week!" (appears twice, count once)
- "New Job Matches for You!"
- "Your Friend Request Was Accepted!"
- "New Article Recommendation: How to Improve Your Coding Skills"
- "New Shows and Movies for September!"
- "Reminder: Team Sync-Up at 10:00 AM" (Calendar Notifications)
"""

import json
import sys
from typing import Any, Dict, List, Tuple

TARGET_SENDERS = ["alerts", "newsletter", "notifications"]

EXPECTED_EMAILS = [
    # Alerts
    "your ride receipt",
    "suspicious activity detected",
    # Newsletter
    "leadership articles",
    "election results",
    "latest in tech",
    "top stories from around the world",
    "major event happening",
    "weekly tech news",
    "top tracks for your week",
    # Notifications
    "linkedin profile views",
    "new job matches",
    "friend request was accepted",
    "how to improve your coding skills",
    "new shows and movies",
    "team sync-up",
]


def normalize_str(s: Any) -> str:
    """Normalize string for comparison (strip whitespace, lowercase)."""
    return (s or "").strip().lower()


def is_target_sender(from_field: str) -> bool:
    """Check if the sender matches one of the target categories."""
    from_lower = normalize_str(from_field)
    for target in TARGET_SENDERS:
        if target.lower() in from_lower:
            return True
    return False


def get_matching_keyword(subject: str) -> str | None:
    """Check if subject contains one of the expected keywords."""
    subject_lower = normalize_str(subject)
    for keyword in EXPECTED_EMAILS:
        if keyword.lower() in subject_lower:
            return keyword
    return None


def get_trashed_emails(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all emails that were moved to trash.
    Enriches with subject/from from initialstate if available.
    """
    trashed = []

    differences = data.get("differences") or {}
    emails_diff = differences.get("emails") or {}
    updated = emails_diff.get("updated") or []

    # Build a lookup of email ID -> email data from initialstate
    id_to_email = {}
    initialstate = data.get("initialstate") or {}
    init_email = initialstate.get("email") or {}
    init_emails = init_email.get("emails") or []
    for email in init_emails:
        if isinstance(email, dict):
            email_id = str(email.get("id", ""))
            if email_id:
                id_to_email[email_id] = email

    for email in updated:
        if isinstance(email, dict) and email.get("trash") is True:
            email_id = str(email.get("id", ""))
            # Enrich with data from initialstate if missing
            if email_id in id_to_email:
                orig = id_to_email[email_id]
                enriched = dict(email)
                if not enriched.get("subject"):
                    enriched["subject"] = orig.get("subject", "")
                if not enriched.get("from"):
                    enriched["from"] = orig.get("from", "")
                trashed.append(enriched)
            else:
                trashed.append(email)

    return trashed


def get_all_target_emails(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all emails from target senders in the initial state.
    """
    target_emails = []

    initialstate = data.get("initialstate") or {}
    init_email = initialstate.get("email") or {}
    init_emails = init_email.get("emails") or []

    for email in init_emails:
        if isinstance(email, dict):
            from_field = email.get("from", "")
            if is_target_sender(from_field):
                target_emails.append(email)

    return target_emails


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Scoring:
    - Count emails from target senders that were deleted
    - Score = (deleted_target_emails) / (total_target_emails)

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    # Get all target emails from initial state
    target_emails = get_all_target_emails(final_state)
    target_ids = {str(e.get("id", "")) for e in target_emails if e.get("id")}
    logger.info(f"Found {len(target_emails)} target emails to delete")

    # Get all trashed emails
    trashed_emails = get_trashed_emails(final_state)
    trashed_ids = {str(e.get("id", "")) for e in trashed_emails if e.get("id")}
    logger.info(f"Found {len(trashed_emails)} trashed emails")

    # Check which target emails were deleted
    deleted_target_ids = target_ids & trashed_ids
    not_deleted_target_ids = target_ids - trashed_ids

    # Get details
    deleted_emails_info = []
    not_deleted_emails_info = []

    for email in target_emails:
        email_id = str(email.get("id", ""))
        info = {
            "id": email_id,
            "from": email.get("from", ""),
            "subject": email.get("subject", ""),
        }
        if email_id in deleted_target_ids:
            deleted_emails_info.append(info)
        else:
            not_deleted_emails_info.append(info)

    # Check for wrongly deleted emails (not from target senders)
    wrongly_deleted = []
    for email in trashed_emails:
        email_id = str(email.get("id", ""))
        if email_id not in target_ids:
            wrongly_deleted.append(
                {
                    "id": email_id,
                    "from": email.get("from", ""),
                    "subject": email.get("subject", ""),
                }
            )

    logger.info(f"Deleted: {len(deleted_target_ids)}/{len(target_ids)} target emails")
    if wrongly_deleted:
        logger.warning(f"Wrongly deleted: {len(wrongly_deleted)} non-target emails")

    # Calculate score
    # Full credit (1.0) if all target emails deleted
    # Partial credit (0.5) if some but not all deleted
    # No credit (0.0) if none deleted
    if len(target_ids) == 0:
        total_pct = 1.0 if len(trashed_emails) == 0 else 0.0
    elif len(deleted_target_ids) == len(target_ids):
        total_pct = 1.0
    elif len(deleted_target_ids) > 0:
        total_pct = 0.5
    else:
        total_pct = 0.0

    success = len(deleted_target_ids) == len(target_ids) and len(wrongly_deleted) == 0

    details = {
        "target_senders": TARGET_SENDERS,
        "total_target_emails": len(target_ids),
        "deleted_target_count": len(deleted_target_ids),
        "not_deleted_count": len(not_deleted_target_ids),
        "wrongly_deleted_count": len(wrongly_deleted),
        "deleted_emails": deleted_emails_info,
        "not_deleted_emails": not_deleted_emails_info,
        "wrongly_deleted_emails": wrongly_deleted,
        "total_trashed": len(trashed_emails),
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
    print(f"Target emails: {details['total_target_emails']}")
    print(f"Deleted: {details['deleted_target_count']}")
    print(f"Not deleted: {details['not_deleted_count']}")
    print(f"Wrongly deleted: {details['wrongly_deleted_count']}")

    if details["not_deleted_emails"]:
        print(f"\nNot deleted emails:")
        for e in details["not_deleted_emails"][:5]:
            print(f"  - {e['from']}: {e['subject']}")

    if success:
        print(f"\nSUCCESS ({pct * 100:.0f}%)")
    else:
        print(f"\nFAILURE ({pct * 100:.0f}%)")


if __name__ == "__main__":
    main()
