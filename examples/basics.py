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
        base_url="http://127.0.0.1:8002"
    )
    
    print(f"Running task: {TASK}\n")
    
    # Send task to BLAST
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=TASK,
        stream=False
    )
    
    # Print result
    result = response.choices[0].message.content
    print("Result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())