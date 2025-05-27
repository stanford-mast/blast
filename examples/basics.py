"""Basic example of using BLAST.

Make sure to run `blastai serve` first before running this script!
"""

import asyncio
from openai import OpenAI

# Example task - you can modify this to try different tasks
TASK = "Go to example.com and extract the main heading and first paragraph"

async def main():
    """Run a task using BLAST.
    
    First make sure to run `blastai serve` in a terminal!
    """
    # Connect to local BLAST server
    client = OpenAI(
        api_key="not-needed",  # Not used when connecting to local BLAST server
        base_url="http://127.0.0.1:8001"
    )
    
    print(f"Running task: {TASK}\n")
    
    # Send task to BLAST with streaming
    stream = client.responses.create(
        model="gpt-4.1-mini",
        input=TASK,
        stream=True
    )
    
    # Track and print thoughts as they come
    current_thought = ""
    for event in stream:
        if event.type == "response.output_text.delta":
            # Accumulate the thought
            if ' ' in event.delta:  # Skip screenshots
                current_thought += event.delta
        elif event.type == "response.output_text.done":
            # Print complete thought and reset
            if current_thought:
                print(current_thought)
                current_thought = ""

if __name__ == "__main__":
    asyncio.run(main())