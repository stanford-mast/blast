"""
DashDish Deep Research Task 1: National Favorites Analysis

Task: Go through each restaurant in "National favorites" category and extract:
- Restaurant name
- Number of reviews/ratings  
- Cheapest menu item
- Most expensive menu item
"""
from typing import List
import re
from pydantic import BaseModel, Field
from experiments.tasks.base import TaskValidator


class RestaurantData(BaseModel):
    """Data for a single restaurant"""
    name: str = Field(description="Restaurant name exactly as shown")
    num_reviews: int = Field(description="Total number of reviews/ratings for this restaurant")
    cheapest_item: str = Field(description="Name of the cheapest menu item (item name only, no price or quantity)")
    most_expensive_item: str = Field(description="Name of the most expensive menu item (item name only, no price or quantity)")


class DashDishDeepResearch1Output(BaseModel):
    """Expected output schema for dashdish-deepresearch1 task"""
    restaurants: List[RestaurantData]


# Ground truth data (from actual model output - Run 195)
GROUND_TRUTH = DashDishDeepResearch1Output(
    restaurants=[
        RestaurantData(
            name="The Cheesecake Factory",
            num_reviews=25,
            cheapest_item="Strawberry Cheesecake",
            most_expensive_item="Chicken Madeira"
        ),
        RestaurantData(
            name="McDonald's",
            num_reviews=129,
            cheapest_item="Apple Pie",
            most_expensive_item="10-piece Chicken McNuggets"
        ),
        RestaurantData(
            name="Chipotle Mexican Grill",
            num_reviews=20,
            cheapest_item="Guacamole & Chips",
            most_expensive_item="Chicken Burrito Bowl"
        ),
        RestaurantData(
            name="Taco Bell",
            num_reviews=19,
            cheapest_item="Soft Taco",
            most_expensive_item="Crunchy Taco Supreme"
        ),
        RestaurantData(
            name="Papa Johns Pizza",
            num_reviews=23,
            cheapest_item="Garlic Knots (8 pieces)",
            most_expensive_item="Supreme Pizza"
        ),
        RestaurantData(
            name="Little Caesars",
            num_reviews=20,
            cheapest_item="Crazy Bread",
            most_expensive_item="Caesar Wings (10 pieces)"
        ),
        RestaurantData(
            name="KFC",
            num_reviews=21,
            cheapest_item="Biscuits (2 pieces)",
            most_expensive_item="Chicken Tenders (5 pieces)"
        ),
        RestaurantData(
            name="Popeyes Louisiana Kitchen",
            num_reviews=25,
            cheapest_item="Buttermilk Biscuit",
            most_expensive_item="Spicy Chicken Sandwich"
        ),
        RestaurantData(
            name="Charleys Cheesesteaks and Wings",
            num_reviews=15,
            cheapest_item="Ultimate Fries",
            most_expensive_item="Buffalo Wings (10 pieces)"
        ),
        RestaurantData(
            name="California Pizza Kitchen",
            num_reviews=20,
            cheapest_item="Avocado Club Egg Rolls",
            most_expensive_item="California Club Pizza"
        ),
    ]
)


def normalize_name(name: str) -> str:
    """Normalize restaurant/item name for comparison"""
    name = name.lower().strip()
    # Remove possessives
    name = name.replace("'s", "s")
    # Remove prices (e.g., "- $8.99" or "$8.99")
    name = re.sub(r'\s*-?\s*\$[0-9.]+', '', name)
    # Remove quantity/size info in parentheses (e.g., "(10 pc)", "(8 pieces)")
    name = re.sub(r'\s*\([^)]*\)', '', name)
    # Remove ALL standalone quantity formats including numbers:
    # "10-piece X", "10 piece X", "10 pc X", "X - 10 pieces", etc.
    name = re.sub(r'\d+\s*-?\s*(piece|pieces|pc|pcs|count|ct)\s*', '', name)
    name = re.sub(r'\s*-?\s*\d+\s*(piece|pieces|pc|pcs|count|ct)', '', name)
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


class DashDishDeepResearch1Validator(TaskValidator):
    """Validator for DashDish Deep Research Task 1"""
    
    @property
    def output_schema(self) -> type[BaseModel]:
        return DashDishDeepResearch1Output
    
    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """
        Check correctness as a percentage (0.0 to 1.0) of fields that match.
        
        Counts: 10 restaurants × 4 fields = 40 total fields to check.
        Returns percentage of fields that are correct.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not isinstance(parsed_output, DashDishDeepResearch1Output):
            logger.warning(f"check_correctness_pct: parsed_output is not DashDishDeepResearch1Output (got {type(parsed_output)})")
            return 0.0
        
        total_fields = 40  # 10 restaurants × 4 fields each
        correct_fields = 0
        
        # Create lookup by normalized name
        parsed_map = {normalize_name(r.name): r for r in parsed_output.restaurants}
        
        # Check each ground truth restaurant
        for gt in GROUND_TRUTH.restaurants:
            gt_name_norm = normalize_name(gt.name)
            
            # Check if restaurant exists (counts as 1 field - the name)
            if gt_name_norm not in parsed_map:
                logger.debug(f"check_correctness_pct: Restaurant '{gt.name}' not found in parsed output")
                continue  # Skip this restaurant, lose all 4 fields
            
            parsed_restaurant = parsed_map[gt_name_norm]
            correct_fields += 1  # Restaurant name match
            
            # Check num_reviews (exact match)
            if parsed_restaurant.num_reviews == gt.num_reviews:
                correct_fields += 1
            else:
                logger.debug(f"check_correctness_pct: {gt.name} review count mismatch: expected {gt.num_reviews}, got {parsed_restaurant.num_reviews}")
            
            # Check cheapest item (fuzzy match)
            if normalize_name(parsed_restaurant.cheapest_item) == normalize_name(gt.cheapest_item):
                correct_fields += 1
            else:
                logger.debug(f"check_correctness_pct: {gt.name} cheapest item mismatch: expected '{gt.cheapest_item}', got '{parsed_restaurant.cheapest_item}'")
            
            # Check most expensive item (fuzzy match)
            if normalize_name(parsed_restaurant.most_expensive_item) == normalize_name(gt.most_expensive_item):
                correct_fields += 1
            else:
                logger.debug(f"check_correctness_pct: {gt.name} most expensive item mismatch: expected '{gt.most_expensive_item}', got '{parsed_restaurant.most_expensive_item}'")
        
        percentage = correct_fields / total_fields
        logger.info(f"check_correctness_pct: {correct_fields}/{total_fields} fields correct ({percentage*100:.1f}%)")
        return percentage
    
    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """
        Check if parsed output matches ground truth exactly.
        
        Uses fuzzy matching for restaurant/item names and exact matching for review counts.
        Returns True only if 100% of fields are correct.
        """
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = DashDishDeepResearch1Validator()
