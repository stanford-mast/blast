"""
Accuracy validation for task outputs.

For each task, defines:
1. Expected output schema (Pydantic model)
2. Output parser (extracts structured data from LLM response)
3. Correctness checker (validates against ground truth)
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class RestaurantInfo(BaseModel):
    """Schema for dashdish-deepresearch1 task output."""
    name: str
    num_reviews: int
    cheapest_item: str
    most_expensive_item: str


class DashdishDeepResearch1Output(BaseModel):
    """Expected output for dashdish-deepresearch1 task."""
    restaurants: List[RestaurantInfo]


# Ground truth for dashdish-deepresearch1
DASHDISH_DEEPRESEARCH1_GROUND_TRUTH = [
    RestaurantInfo(
        name="The Cheesecake Factory",
        num_reviews=25,
        cheapest_item="Pasta Carbonara",
        most_expensive_item="Chicken Madeira"
    ),
    RestaurantInfo(
        name="McDonald's",
        num_reviews=129,
        cheapest_item="Apple Pie",
        most_expensive_item="Chicken McNuggets (10 pc)"
    ),
    RestaurantInfo(
        name="Chipotle Mexican Grill",
        num_reviews=20,
        cheapest_item="Guacamole & Chips",
        most_expensive_item="Barbacoa Quesadilla"
    ),
    RestaurantInfo(
        name="Taco Bell",
        num_reviews=19,
        cheapest_item="Soft Taco",
        most_expensive_item="Crunchy Taco Supreme"
    ),
    RestaurantInfo(
        name="Papa Johns Pizza",
        num_reviews=23,
        cheapest_item="Garlic Knots (8 pc)",
        most_expensive_item="Supreme Pizza"
    ),
    RestaurantInfo(
        name="Little Caesars",
        num_reviews=20,
        cheapest_item="Crazy Bread",
        most_expensive_item="Caesar Wings (10 pc)"
    ),
    RestaurantInfo(
        name="KFC",
        num_reviews=21,
        cheapest_item="Biscuits (2 pc)",
        most_expensive_item="Chicken Tenders (5 pc)"
    ),
    RestaurantInfo(
        name="Popeyes Louisiana Kitchen",
        num_reviews=25,
        cheapest_item="Buttermilk Biscuit",
        most_expensive_item="Spicy Chicken Sandwich"
    ),
    RestaurantInfo(
        name="Charleys Cheesesteaks and Wings",
        num_reviews=15,
        cheapest_item="Ultimate Fries",
        most_expensive_item="Buffalo Wings (10 pc)"
    ),
    RestaurantInfo(
        name="California Pizza Kitchen",
        num_reviews=20,
        cheapest_item="Avocado Club Egg Rolls",
        most_expensive_item="California Club Pizza"
    ),
]


def parse_output_to_json(raw_output: str, schema: type[BaseModel], use_llm: bool = True) -> Optional[BaseModel]:
    """
    Parse raw output to structured JSON.
    
    Args:
        raw_output: Raw string output from code execution
        schema: Pydantic model class for the expected schema
        use_llm: If True and direct parsing fails, use LLM to convert to JSON
    
    Returns:
        Parsed Pydantic model instance, or None if parsing failed
    """
    
    # Try to extract JSON directly
    try:
        # Look for JSON object in the output
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return schema(**data)
    except (json.JSONDecodeError, Exception):
        pass
    
    # Try to parse as JSON array
    try:
        json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if schema == DashdishDeepResearch1Output:
                return DashdishDeepResearch1Output(restaurants=data)
    except (json.JSONDecodeError, Exception):
        pass
    
    # If direct parsing failed and use_llm is True, prompt LLM to convert
    if use_llm:
        # TODO: Implement LLM-based conversion
        # For now, return None
        pass
    
    return None


def normalize_item_name(name: str) -> str:
    """Normalize item names for fuzzy matching."""
    # Remove parentheses content, strip, lowercase
    name = re.sub(r'\([^)]*\)', '', name)
    name = name.strip().lower()
    # Remove common words
    name = re.sub(r'\b(the|and|with|&)\b', '', name)
    name = ' '.join(name.split())  # Normalize whitespace
    return name


def fuzzy_match_item(actual: str, expected: str, threshold: float = 0.6) -> bool:
    """Check if two item names match with fuzzy logic."""
    actual_norm = normalize_item_name(actual)
    expected_norm = normalize_item_name(expected)
    
    # Exact match after normalization
    if actual_norm == expected_norm:
        return True
    
    # Check if one contains the other
    if actual_norm in expected_norm or expected_norm in actual_norm:
        return True
    
    # Check word overlap
    actual_words = set(actual_norm.split())
    expected_words = set(expected_norm.split())
    
    if not actual_words or not expected_words:
        return False
    
    overlap = len(actual_words & expected_words)
    total = len(actual_words | expected_words)
    
    return (overlap / total) >= threshold


def check_dashdish_deepresearch1_correctness(
    parsed_output: DashdishDeepResearch1Output,
    strict: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check correctness for dashdish-deepresearch1 task.
    
    Args:
        parsed_output: Parsed output from the code execution
        strict: If True, require exact matches; if False, use fuzzy matching
    
    Returns:
        (is_correct, details_dict)
    """
    
    ground_truth = DASHDISH_DEEPRESEARCH1_GROUND_TRUTH
    
    # Build lookup by restaurant name
    gt_lookup = {normalize_item_name(r.name): r for r in ground_truth}
    
    results = {
        "total_restaurants_expected": len(ground_truth),
        "total_restaurants_found": len(parsed_output.restaurants),
        "matches": [],
        "missing": [],
        "incorrect": [],
    }
    
    for actual in parsed_output.restaurants:
        actual_name_norm = normalize_item_name(actual.name)
        
        # Find matching ground truth restaurant
        gt_match = gt_lookup.get(actual_name_norm)
        
        if not gt_match:
            # Try fuzzy match on name
            for gt_name_norm, gt in gt_lookup.items():
                if fuzzy_match_item(actual.name, gt.name):
                    gt_match = gt
                    break
        
        if not gt_match:
            results["incorrect"].append({
                "name": actual.name,
                "reason": "Restaurant not in expected list"
            })
            continue
        
        # Check fields
        errors = []
        
        # Check num_reviews (allow ±2 tolerance for dynamic data)
        if abs(actual.num_reviews - gt_match.num_reviews) > 2:
            errors.append(f"num_reviews: {actual.num_reviews} != {gt_match.num_reviews}")
        
        # Check cheapest item
        if strict:
            if actual.cheapest_item != gt_match.cheapest_item:
                errors.append(f"cheapest: {actual.cheapest_item} != {gt_match.cheapest_item}")
        else:
            if not fuzzy_match_item(actual.cheapest_item, gt_match.cheapest_item):
                errors.append(f"cheapest: {actual.cheapest_item} != {gt_match.cheapest_item}")
        
        # Check most expensive item
        if strict:
            if actual.most_expensive_item != gt_match.most_expensive_item:
                errors.append(f"most_expensive: {actual.most_expensive_item} != {gt_match.most_expensive_item}")
        else:
            if not fuzzy_match_item(actual.most_expensive_item, gt_match.most_expensive_item):
                errors.append(f"most_expensive: {actual.most_expensive_item} != {gt_match.most_expensive_item}")
        
        if errors:
            results["incorrect"].append({
                "name": actual.name,
                "errors": errors
            })
        else:
            results["matches"].append(actual.name)
    
    # Check for missing restaurants
    found_names = {normalize_item_name(r.name) for r in parsed_output.restaurants}
    for gt in ground_truth:
        if normalize_item_name(gt.name) not in found_names:
            # Try fuzzy match
            found = False
            for actual in parsed_output.restaurants:
                if fuzzy_match_item(actual.name, gt.name):
                    found = True
                    break
            if not found:
                results["missing"].append(gt.name)
    
    # Overall correctness
    is_correct = (
        len(results["matches"]) == len(ground_truth) and
        len(results["incorrect"]) == 0 and
        len(results["missing"]) == 0
    )
    
    results["accuracy"] = len(results["matches"]) / len(ground_truth) if ground_truth else 0
    
    return is_correct, results


def validate_task_output(task_id: str, raw_output: str, strict: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate output for any task.
    
    Args:
        task_id: Task identifier
        raw_output: Raw string output from code execution
        strict: Whether to use strict matching
    
    Returns:
        (is_correct, details_dict)
    """
    
    if task_id == "dashdish-deepresearch1":
        parsed = parse_output_to_json(raw_output, DashdishDeepResearch1Output, use_llm=False)
        
        if parsed is None:
            return False, {"error": "Failed to parse output to expected schema"}
        
        return check_dashdish_deepresearch1_correctness(parsed, strict=strict)
    
    else:
        return False, {"error": f"Unknown task_id: {task_id}"}
