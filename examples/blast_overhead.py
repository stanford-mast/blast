"""Compare latency between BLAST and direct browser-use for a simple task."""

import logging
from blastai.logging_setup import setup_logging
from blastai.config import Settings

# Set up logging with timestamps
settings = Settings(
    blastai_log_level="INFO",
    browser_use_log_level="INFO"
)
setup_logging(settings)
logger = logging.getLogger(__name__)

import os
import time
import asyncio
import statistics
from typing import List, Tuple
from pathlib import Path
import shutil
from openai import OpenAI
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI

# Task to run
TASK = "read what example.com says"

# Number of runs per approach
NUM_RUNS = 3

async def run_with_browser_use() -> float:
    """Run task with browser-use directly."""
    # Create browser with settings matching BLAST's
    config = BrowserConfig(
        headless=True  # From default_config.yaml require_headless=true
    )
    browser = Browser(config=config)
    
    # Create agent with same model settings as BLAST
    agent = Agent(
        task=TASK,
        llm=ChatOpenAI(model="gpt-4.1"),  # From default_config.yaml llm_model
        use_vision=False,  # From default_config.yaml allow_vision=true
        browser=browser  # Pass browser instance directly
    )
    
    start_time = time.time()
    try:
        history = await agent.run()
        end_time = time.time()
        await browser.close()
        return end_time - start_time
    except Exception as e:
        await browser.close()
        raise e

def run_with_blast() -> float:
    """Run task with BLAST."""
    # Clear cache first
    cache_dir = Path.home() / ".cache" / "blast"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    
    client = OpenAI(
        api_key="not-needed",
        base_url="http://127.0.0.1:8000"
    )
    
    # Delete task from cache first
    client.responses.delete(TASK)
    
    start_time = time.time()
    response = client.responses.create(
        model="gpt-4.1",
        input=TASK,
        stream=False
    )
    end_time = time.time()
    
    # Extract result from response
    result = response.output[0].content[0].text
    print(f"BLAST result: {result}")
    
    return end_time - start_time

async def run_comparison() -> Tuple[List[float], List[float]]:
    """Run both approaches multiple times, alternating between them."""
    blast_times = []
    browser_use_times = []
    
    for i in range(NUM_RUNS):
        # Run BLAST first
        blast_time = run_with_blast()
        blast_times.append(blast_time)
        print(f"Run {i+1} BLAST: {blast_time:.2f}s")
        
        # Then browser-use
        browser_use_time = await run_with_browser_use()
        browser_use_times.append(browser_use_time)
        print(f"Run {i+1} browser-use: {browser_use_time:.2f}s")
        
        # Small delay between runs
        await asyncio.sleep(1)
    
    return blast_times, browser_use_times

def print_stats(name: str, times: List[float]):
    """Print statistics for a set of timing measurements."""
    print(f"\n{name} Statistics:")
    print(f"  Mean: {statistics.mean(times):.2f}s")
    print(f"  Std Dev: {statistics.stdev(times):.2f}s")
    print(f"  Min: {min(times):.2f}s")
    print(f"  Max: {max(times):.2f}s")

async def main():
    """Run the performance comparison."""
    print("Starting performance comparison...")
    print(f"Task: {TASK}")
    print(f"Running {NUM_RUNS} times each\n")
    
    blast_times, browser_use_times = await run_comparison()
    
    print("\nResults:")
    print_stats("BLAST", blast_times)
    print_stats("browser-use", browser_use_times)

if __name__ == "__main__":
    asyncio.run(main())

# BLAST Statistics:
#   Mean: 15.24s
#   Std Dev: 0.77s
#   Min: 14.47s
#   Max: 16.01s

# browser-use Statistics:
#   Mean: 15.18s
#   Std Dev: 3.53s
#   Min: 11.56s
#   Max: 18.62s