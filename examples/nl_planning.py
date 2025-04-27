"""Examples of natural language planning with different parallelism patterns."""

import os
import asyncio
from dotenv import load_dotenv
from blastai.planner import Planner
from blastai.config import Constraints
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

async def test_planning():
    """Test planner on various task descriptions."""
    
    # Initialize planner with custom LLM using API key
    constraints = Constraints()
    llm = ChatOpenAI(
        api_key=api_key,
        model=constraints.llm_model
    )
    planner = Planner(constraints)
    planner.llm = llm  # Override default LLM with authenticated one
    
    # Test cases with different parallelism patterns
    test_cases = [
        # No parallelism opportunity
        "Search for 'python programming' on Google and click the first result",
        
        # Simple parallelism
        "What are people talking about on r/securityguards, r/AskLE, and r/protectandserve.",
        
        # Delayed parallelism with dependencies
        "Look up the 3 most popular programming languages, then tell me who created each one",
        
        # Aggregation
        "Search for top 5 restaurants in San Francisco, open each one's google reviews, and collect their ratings",
        
        # Many attributes
        "Kyrie: tell me his points, assists, rebounds, steals, blocks, turnovers, and field goal percentage in the last 5 games",
        
        # Specific website with many parallel actions
        "On GitHub: search for top repositories in Python, JavaScript, and Go",
        
        # No specific details
        "Research the latest news about artificial intelligence",
        
        # Many actions with same data
        "Compare the latest iPhone, Samsung Galaxy, and Google Pixel models",
        
        # Mixed sequential and parallel
        "Compare the last 3 iPhone models, then check their prices on Amazon and eBay",
        
        # Mixed sequential and parallel
        "Compare the last 3 iPhone models, then check their prices on Amazon and eBay",

        # Complex task with multiple dependencies
        "Find the top 10 AI startups and for each of their CEOs find the high school they studied at"
    ]
    
    # Run planner on each test case
    for i, task in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Task: {task}")
        plan = await planner.plan(task)
        print(f"Plan: {plan}")

async def main():
    """Run the planning tests."""
    await test_planning()

if __name__ == "__main__":
    asyncio.run(main())