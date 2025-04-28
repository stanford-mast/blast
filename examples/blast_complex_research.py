"""Compare BLAST and browser-use for complex research task about space company CEOs."""

import os
import time
import asyncio
from pathlib import Path
import shutil
from openai import OpenAI
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from blastai.utils import estimate_llm_cost

# Complex research task
TASK = "where did the CEOs of the 10 largest space stocks right now go to college and what specific college experiences most impacted them?"
TASK = "Go to example.com"
TASK = "Compare UW, Berkeley, and Stanford biomedical data science programs"

async def run_with_browser_use() -> tuple[float, str, float]:
    """Run task with browser-use directly."""
    # Create browser with settings matching BLAST's default_config.yaml
    config = BrowserConfig(
        headless=True  # From require_headless=false
    )
    browser = Browser(config=config)
    
    # Create agent with same model and vision settings as BLAST
    agent = Agent(
        task=TASK,
        llm=ChatOpenAI(model="gpt-4.1"),  # From llm_model
        use_vision=False,  # From allow_vision=false
        browser=browser  # Pass browser instance directly
    )
    
    start_time = time.time()
    total_cost = 0.0
    try:
        # Run with cost tracking and estimation
        with get_openai_callback() as cb:
            history = await agent.run()
            cost = cb.total_cost
            if cost == 0 and cb.total_tokens > 0:
                cached_tokens = getattr(cb, "prompt_tokens_cached", 0)
                cost = estimate_llm_cost(
                    model_name="gpt-4.1",
                    prompt_tokens=cb.prompt_tokens,
                    completion_tokens=cb.completion_tokens,
                    cached_tokens=cached_tokens
                )
            total_cost = cost
            
        end_time = time.time()
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
    
    client = OpenAI(
        api_key="not-needed",
        base_url="http://127.0.0.1:8000"
    )
    
    start_time = time.time()
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=TASK,
        stream=False
    )
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