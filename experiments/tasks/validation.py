"""
Validation schemas and correctness checkers for evaluation tasks.

This module provides:
- Pydantic models for expected output schemas per task
- Harness runner to parse outputs into structured JSON
- Correctness checkers for each task
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass


# =============================================================================
# Schema for dashdish-deepresearch1
# =============================================================================
class RestaurantInfo(BaseModel):
    """Information about a single restaurant."""
    name: str = Field(description="Restaurant name")
    num_ratings: int = Field(description="Number of ratings the restaurant has")
    cheapest_item: str = Field(description="Name of the cheapest menu item")
    most_expensive_item: str = Field(description="Name of the most expensive menu item")


class DashDishDeepResearch1Output(BaseModel):
    """Expected output schema for dashdish-deepresearch1 task."""
    restaurants: List[RestaurantInfo] = Field(description="List of restaurant information")
    
    @field_validator('restaurants', mode='before')
    @classmethod
    def validate_restaurants(cls, v):
        if isinstance(v, list):
            return v
        return []


# =============================================================================
# Ground truth data
# =============================================================================
GROUND_TRUTH = {
    "dashdish-deepresearch1": DashDishDeepResearch1Output(
        restaurants=[
            RestaurantInfo(name="The Cheesecake Factory", num_ratings=25, cheapest_item="Pasta Carbonara", most_expensive_item="Chicken Madeira"),
            RestaurantInfo(name="McDonald's", num_ratings=129, cheapest_item="Apple Pie", most_expensive_item="Chicken McNuggets (10 pc)"),
            RestaurantInfo(name="Chipotle Mexican Grill", num_ratings=20, cheapest_item="Guacamole & Chips", most_expensive_item="Barbacoa Quesadilla"),
            RestaurantInfo(name="Taco Bell", num_ratings=19, cheapest_item="Soft Taco", most_expensive_item="Crunchy Taco Supreme"),
            RestaurantInfo(name="Papa Johns Pizza", num_ratings=23, cheapest_item="Garlic Knots (8 pc)", most_expensive_item="Supreme Pizza"),
            RestaurantInfo(name="Little Caesars", num_ratings=20, cheapest_item="Crazy Bread", most_expensive_item="Caesar Wings (10 pc)"),
            RestaurantInfo(name="KFC", num_ratings=21, cheapest_item="Biscuits (2 pc)", most_expensive_item="Chicken Tenders (5 pc)"),
            RestaurantInfo(name="Popeyes Louisiana Kitchen", num_ratings=25, cheapest_item="Buttermilk Biscuit", most_expensive_item="Spicy Chicken Sandwich"),
            RestaurantInfo(name="Charleys Cheesesteaks and Wings", num_ratings=15, cheapest_item="Ultimate Fries", most_expensive_item="Buffalo Wings (10 pc)"),
            RestaurantInfo(name="California Pizza Kitchen", num_ratings=20, cheapest_item="Avocado Club Egg Rolls", most_expensive_item="California Club Pizza"),
        ]
    )
}


# =============================================================================
# Output parsing utilities
# =============================================================================
def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract JSON from text that may contain other content."""
    if not text:
        return None
    
    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}',
        r'\[[\s\S]*\]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue
    
    return None


async def llm_parse_output(output: str, schema: type[BaseModel], task_id: str) -> Optional[Dict[str, Any]]:
    """Use LLM to parse unstructured output into the expected schema."""
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from blastai.agents.llm_factory import LLMFactory
    
    # Get schema JSON representation
    schema_json = schema.model_json_schema()
    
    prompt = f"""Parse the following output into JSON that matches this schema:

Schema:
{json.dumps(schema_json, indent=2)}

Output to parse:
{output}

Return ONLY valid JSON that matches the schema. No explanation or markdown formatting.
"""
    
    # Use a small model for parsing
    llm = LLMFactory.create_model("gpt-4.1-mini")
    
    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        content = response.content if hasattr(response, 'content') else str(response)
        return extract_json_from_text(content)
    except Exception as e:
        print(f"LLM parse error: {e}")
        return None


# =============================================================================
# Harness runner
# =============================================================================
@dataclass
class ValidationResult:
    """Result of validating task output."""
    is_correct: bool
    parsed_output: Optional[BaseModel]
    accuracy_score: float  # 0.0 to 1.0
    details: Dict[str, Any]


async def parse_output(output: Any, task_id: str) -> Optional[BaseModel]:
    """Parse output into the expected schema for a task."""
    schema_map = {
        "dashdish-deepresearch1": DashDishDeepResearch1Output,
    }
    
    if task_id not in schema_map:
        raise ValueError(f"Unknown task_id: {task_id}")
    
    schema = schema_map[task_id]
    
    # If output is already a dict or the right type
    if isinstance(output, schema):
        return output
    
    # Convert to string if needed
    output_str = str(output) if not isinstance(output, str) else output
    
    # Try to extract JSON directly
    json_data = extract_json_from_text(output_str)
    
    if json_data:
        try:
            # Handle both dict with 'restaurants' key and list format
            if isinstance(json_data, list):
                json_data = {"restaurants": json_data}
            return schema.model_validate(json_data)
        except Exception:
            pass
    
    # Fall back to LLM parsing
    json_data = await llm_parse_output(output_str, schema, task_id)
    if json_data:
        try:
            if isinstance(json_data, list):
                json_data = {"restaurants": json_data}
            return schema.model_validate(json_data)
        except Exception:
            pass
    
    return None


def normalize_name(name: str) -> str:
    """Normalize restaurant/item name for comparison."""
    # Lowercase, remove extra spaces, common variations
    name = name.lower().strip()
    # Remove common suffixes/prefixes
    name = re.sub(r'\s+', ' ', name)
    # Remove apostrophes that might vary (McDonald's vs McDonalds)
    name = name.replace("'", "").replace("'", "")
    return name


def check_correctness_dashdish_deepresearch1(
    parsed: DashDishDeepResearch1Output,
    ground_truth: DashDishDeepResearch1Output
) -> ValidationResult:
    """Check correctness for dashdish-deepresearch1 task."""
    details = {
        "restaurants_found": len(parsed.restaurants),
        "restaurants_expected": len(ground_truth.restaurants),
        "correct_restaurants": [],
        "missing_restaurants": [],
        "incorrect_details": [],
    }
    
    # Build lookup by normalized name
    gt_by_name = {normalize_name(r.name): r for r in ground_truth.restaurants}
    parsed_by_name = {normalize_name(r.name): r for r in parsed.restaurants}
    
    # Check each ground truth restaurant
    correct_count = 0
    total_fields = 0  # name, num_ratings, cheapest, most_expensive per restaurant
    correct_fields = 0
    
    for gt_name_norm, gt_r in gt_by_name.items():
        # Find matching restaurant in parsed output (fuzzy match)
        matched = None
        for p_name_norm, p_r in parsed_by_name.items():
            # Check if names match reasonably
            if gt_name_norm == p_name_norm or gt_name_norm in p_name_norm or p_name_norm in gt_name_norm:
                matched = p_r
                break
        
        if matched:
            total_fields += 3  # ratings, cheapest, most_expensive
            
            # Check ratings (allow ±5 tolerance for potential site updates)
            ratings_correct = abs(matched.num_ratings - gt_r.num_ratings) <= 5
            if ratings_correct:
                correct_fields += 1
            
            # Check cheapest item (fuzzy string match)
            cheapest_correct = (
                normalize_name(matched.cheapest_item) == normalize_name(gt_r.cheapest_item) or
                normalize_name(gt_r.cheapest_item) in normalize_name(matched.cheapest_item) or
                normalize_name(matched.cheapest_item) in normalize_name(gt_r.cheapest_item)
            )
            if cheapest_correct:
                correct_fields += 1
            
            # Check most expensive item
            expensive_correct = (
                normalize_name(matched.most_expensive_item) == normalize_name(gt_r.most_expensive_item) or
                normalize_name(gt_r.most_expensive_item) in normalize_name(matched.most_expensive_item) or
                normalize_name(matched.most_expensive_item) in normalize_name(gt_r.most_expensive_item)
            )
            if expensive_correct:
                correct_fields += 1
            
            if ratings_correct and cheapest_correct and expensive_correct:
                correct_count += 1
                details["correct_restaurants"].append(gt_r.name)
            else:
                details["incorrect_details"].append({
                    "name": gt_r.name,
                    "ratings_correct": ratings_correct,
                    "cheapest_correct": cheapest_correct,
                    "expensive_correct": expensive_correct,
                    "expected": {"ratings": gt_r.num_ratings, "cheapest": gt_r.cheapest_item, "expensive": gt_r.most_expensive_item},
                    "got": {"ratings": matched.num_ratings, "cheapest": matched.cheapest_item, "expensive": matched.most_expensive_item}
                })
        else:
            details["missing_restaurants"].append(gt_r.name)
            total_fields += 3
    
    # Calculate accuracy
    accuracy = correct_fields / total_fields if total_fields > 0 else 0.0
    is_correct = accuracy >= 0.8  # 80% threshold for "correct"
    
    details["correct_count"] = correct_count
    details["field_accuracy"] = accuracy
    
    return ValidationResult(
        is_correct=is_correct,
        parsed_output=parsed,
        accuracy_score=accuracy,
        details=details
    )


# =============================================================================
# Main validation function
# =============================================================================
async def validate_task_output(output: Any, task_id: str) -> ValidationResult:
    """
    Validate task output against ground truth.
    
    Args:
        output: The raw output from task execution
        task_id: The task identifier
    
    Returns:
        ValidationResult with correctness info
    """
    # Parse output
    parsed = await parse_output(output, task_id)
    
    if parsed is None:
        return ValidationResult(
            is_correct=False,
            parsed_output=None,
            accuracy_score=0.0,
            details={"error": "Failed to parse output into expected schema"}
        )
    
    # Get ground truth
    ground_truth = GROUND_TRUTH.get(task_id)
    if ground_truth is None:
        return ValidationResult(
            is_correct=False,
            parsed_output=parsed,
            accuracy_score=0.0,
            details={"error": f"No ground truth available for task {task_id}"}
        )
    
    # Check correctness based on task type
    if task_id == "dashdish-deepresearch1":
        return check_correctness_dashdish_deepresearch1(parsed, ground_truth)
    
    return ValidationResult(
        is_correct=False,
        parsed_output=parsed,
        accuracy_score=0.0,
        details={"error": f"No correctness checker for task {task_id}"}
    )


# =============================================================================
# Convenience functions
# =============================================================================
def get_task_schema(task_id: str) -> Optional[type[BaseModel]]:
    """Get the Pydantic schema for a task."""
    schema_map = {
        "dashdish-deepresearch1": DashDishDeepResearch1Output,
    }
    return schema_map.get(task_id)


def get_ground_truth(task_id: str) -> Optional[BaseModel]:
    """Get the ground truth for a task."""
    return GROUND_TRUTH.get(task_id)
