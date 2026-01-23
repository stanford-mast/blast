"""
Omnizon Custom Task 7: Category Product Colors

Task: Visit "Gift", "Toy", "Gaming", and "Cosmetic" categories on the menu bar.
For each, click on the first item and report the product color.

Expected colors:
- Gift: Yellow
- Toy: Purple
- Gaming: Black
- Cosmetic: Multicolor
"""

import json
import re
import sys
from typing import Any, Dict, Tuple

# Expected colors for each category
EXPECTED_COLORS: Dict[str, str] = {
    "gift": "yellow",
    "toy": "purple",
    "gaming": "black",
    "cosmetic": "multicolor",
}


def normalize_color(color: str) -> str:
    """Normalize color string for comparison."""
    return color.lower().strip()


def extract_colors_from_result(result: str) -> Dict[str, str]:
    """Extract category-color pairs from the final result text."""
    colors = {}
    result_lower = result.lower()

    # Try to find patterns like "Gift: Yellow" or "Gift - Yellow" or "Gift color: Yellow"
    for category in EXPECTED_COLORS.keys():
        # Various patterns to match
        patterns = [
            rf"{category}\s*[:\-]\s*(\w+)",  # Gift: Yellow or Gift - Yellow
            rf"{category}\s+color\s*[:\-]?\s*(\w+)",  # Gift color: Yellow
            rf"{category}[^:]*?:\s*(\w+)",  # Gift product: Yellow
        ]
        for pattern in patterns:
            match = re.search(pattern, result_lower)
            if match:
                colors[category] = normalize_color(match.group(1))
                break

    return colors


def evaluate(
    final_state: Dict[str, Any], final_result: str
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate the task output.

    Criteria:
    - Correct colors reported for each category
    - Partial credit for each correct color

    Returns:
        (success, percentage, details)
    """
    import logging

    logger = logging.getLogger(__name__)

    details = {
        "expected_colors": EXPECTED_COLORS,
        "found_colors": {},
        "matches": {},
    }

    # Extract colors from the final result
    found_colors = extract_colors_from_result(final_result)
    details["found_colors"] = found_colors

    logger.info(f"Final result: {final_result[:500]}...")
    logger.info(f"Extracted colors: {found_colors}")
    logger.info(f"Expected colors: {EXPECTED_COLORS}")

    # Count correct matches
    matched = 0
    for category, expected in EXPECTED_COLORS.items():
        found = found_colors.get(category, "")
        is_match = found and (expected in found or found == expected)
        details["matches"][category] = is_match

        if is_match:
            matched += 1
            logger.info(f"✓ {category}: found '{found}' matches expected '{expected}'")
        else:
            logger.info(f"✗ {category}: found '{found}' != expected '{expected}'")

    # Scoring
    score = matched / len(EXPECTED_COLORS)
    success = matched == len(EXPECTED_COLORS)

    if success:
        logger.info(f"✓ Success: all {len(EXPECTED_COLORS)} colors correct")
    else:
        logger.info(f"✗ Partial: {matched}/{len(EXPECTED_COLORS)} colors correct ({score:.0%})")

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
