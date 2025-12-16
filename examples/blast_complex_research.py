"""Compare BLAST and browser-use for complex research task about space company CEOs."""

import asyncio
import os
import shutil
import time
from pathlib import Path

from browser_use import Agent, Browser, BrowserProfile
from langchain_openai import ChatOpenAI
from openai import OpenAI

# Complex research task
TASK = "where did the CEOs of the 10 largest space stocks right now go to college and what specific college experiences most impacted them?"
TASK = "Go to example.com"
TASK = "Compare UW, Berkeley, and Stanford biomedical data science programs"


async def run_with_browser_use() -> tuple[float, str, float]:
    """Run task with browser-use directly."""
    # Create browser with settings matching BLAST's default_config.yaml
    browser_profile = BrowserProfile(
        headless=True  # From require_headless=false
    )
    browser = Browser(browser_profile=browser_profile)

    # Create agent with same model and vision settings as BLAST
    agent = Agent(
        task=TASK,
        llm=ChatOpenAI(model="gpt-4.1"),  # From llm_model
        use_vision=False,  # From allow_vision=false
        browser=browser,  # Pass browser instance directly
        calculate_cost=True,  # Enable automatic cost tracking
    )

    start_time = time.time()
    try:
        history = await agent.run()
        end_time = time.time()

        # Get cost from agent's built-in token cost service
        total_cost = 0.0
        if history.usage and history.usage.total_cost:
            total_cost = history.usage.total_cost

        await browser.close()
        return end_time - start_time, history.final_result(), total_cost
    except Exception as e:
        await browser.close()
        raise e


def run_with_blast() -> tuple[float, str]:
    """Run task with BLAST."""
    # Clear cache first
    cache_dir = Path.home() / ".cache" / "blast"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    client = OpenAI(api_key="not-needed", base_url="http://127.0.0.1:8000")

    start_time = time.time()
    response = client.responses.create(model="gpt-4.1-mini", input=TASK, stream=False)
    end_time = time.time()

    # Extract final result from response
    result = response.choices[0].message.content
    return end_time - start_time, result


async def main():
    """Run the comparison and display results."""
    print("Starting complex research task comparison...")
    print(f"Task: {TASK}\n")

    # # Run BLAST first
    # print("Running with BLAST...")
    # blast_time, blast_result = run_with_blast()
    # print(f"BLAST Time: {blast_time:.2f}s")
    # print("BLAST Result:")
    # print(blast_result)
    # print("\n" + "="*80 + "\n")

    # Then run browser-use
    print("Running with browser-use...")
    browser_use_time, browser_use_result, total_cost = await run_with_browser_use()
    print(f"browser-use Time: {browser_use_time:.2f}s")
    print(f"browser-use Cost: ${total_cost:.4f}")
    print("browser-use Result:")
    print(browser_use_result)

    # Print comparison
    print("\nTime Comparison:")
    # print(f"BLAST:       {blast_time:.2f}s")
    print(f"browser-use: {browser_use_time:.2f}s")
    # print(f"Difference:  {abs(blast_time - browser_use_time):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
