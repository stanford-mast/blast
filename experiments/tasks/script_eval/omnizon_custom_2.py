"""
Omnizon Custom Task 2: Write Product Reviews for First Three Orders

Task: Go to Returns & Orders page and write a product review for each item
in the first three orders.

The first three orders contain 5 items:
- Order 1: Owala FreeSip Water Bottle, Cuisinart Air Fryer Oven (2 items)
- Order 2: Ninja Air Fryer Pro (1 item)
- Order 3: HIWARE Silverware Set, JoyJolt Glass Containers (2 items)

Criteria:
1. Reviews were created (userCreatedReviews is not empty)
2. At least 5 reviews created (one for each item in first 3 orders)
"""

import json
import sys
from typing import Any, Dict, Tuple

# Expected number of items in first 3 orders
EXPECTED_REVIEW_COUNT = 5


def safe_get(d: Dict, *keys) -> Any:
    """Safely traverse nested dictionary keys."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Criteria:
    - Reviews were created in userCreatedReviews
    - At least 5 reviews (one per item in first 3 orders)

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    details = {
        "reviews_created": 0,
        "expected_reviews": EXPECTED_REVIEW_COUNT,
        "review_details": [],
    }

    # Check for reviews in finalstate.review.userCreatedReviews
    user_reviews = safe_get(final_state, "finalstate", "review", "userCreatedReviews")

    # Also check in initialfinaldiff for added reviews
    diff_reviews = safe_get(
        final_state, "initialfinaldiff", "added", "review", "userCreatedReviews"
    )

    # Also check state.review.userCreatedReviews as fallback
    state_reviews = safe_get(final_state, "state", "review", "userCreatedReviews")

    # Use whichever has reviews
    reviews = []
    if isinstance(user_reviews, list) and len(user_reviews) > 0:
        reviews = user_reviews
        logger.info(
            f"Found {len(reviews)} reviews in finalstate.review.userCreatedReviews"
        )
    elif isinstance(diff_reviews, list) and len(diff_reviews) > 0:
        reviews = diff_reviews
        logger.info(
            f"Found {len(reviews)} reviews in initialfinaldiff.added.review.userCreatedReviews"
        )
    elif isinstance(state_reviews, list) and len(state_reviews) > 0:
        reviews = state_reviews
        logger.info(f"Found {len(reviews)} reviews in state.review.userCreatedReviews")
    else:
        logger.info("No reviews found in any location")

    details["reviews_created"] = len(reviews)

    # Extract review details for logging
    for review in reviews:
        if isinstance(review, dict):
            review_info = {
                "productId": review.get("productId"),
                "rating": review.get("rating"),
                "title": review.get("title", "")[:50] if review.get("title") else "",
                "comment": review.get("comment", "")[:50]
                if review.get("comment")
                else "",
            }
            details["review_details"].append(review_info)
            logger.info(
                f"Review found: productId={review_info['productId']}, rating={review_info['rating']}"
            )

    # Scoring: partial credit based on reviews created
    score = min(len(reviews) / EXPECTED_REVIEW_COUNT, 1.0)
    success = len(reviews) >= EXPECTED_REVIEW_COUNT

    if success:
        logger.info(
            f"✓ Success: {len(reviews)} reviews created (expected {EXPECTED_REVIEW_COUNT})"
        )
    else:
        logger.info(
            f"✗ Partial: {len(reviews)}/{EXPECTED_REVIEW_COUNT} reviews created ({score:.0%})"
        )

    return success, score, details


def main():
    """CLI entry point for subprocess execution."""
    path = sys.argv[1]
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"FAILURE: Could not load state file: {e}")
        return

    final_state = data.get("final_state", data)
    final_result = data.get("final_result", "")

    success, pct, details = evaluate(final_state, final_result)

    print(f"{'SUCCESS' if success else 'FAILURE'} ({pct * 100:.0f}%)")
    print(f"Details: {json.dumps(details, indent=2)}")


if __name__ == "__main__":
    main()
