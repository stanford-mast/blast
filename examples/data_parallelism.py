import asyncio
import time
import math
import os
import argparse
from dotenv import load_dotenv
from faker import Faker

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test data parallelism with different LLM models')
    parser.add_argument('--model', type=str, default='gpt-4.1',
                      help='Model to use for testing (e.g., gpt-4.1, gpt-4.1-mini)')
    parser.add_argument('--compare', action='store_true',
                      help='Compare with baseline model (gpt-4.1)')
    return parser.parse_args()

# Load environment variables from .env
load_dotenv()
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import json
import statistics
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt
from blastai.utils import estimate_llm_cost

@dataclass
class TestResult:
    """Store results for a single test configuration."""
    size: int
    config: str
    avg_time: float
    times: List[float]
    prompt_tokens: int
    completion_tokens: int
    cost: float
    num_chunks: int  # Track how many chunks were processed

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens processed per second."""
        return (self.prompt_tokens + self.completion_tokens) / self.avg_time

    @property
    def tokens_per_chunk(self) -> float:
        """Calculate average tokens per chunk."""
        return (self.prompt_tokens + self.completion_tokens) / self.num_chunks

def plot_results(results: List[TestResult], output_file: str = "parallelism_results.png"):
    """Create visualization of chunk size vs performance."""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Get unique text sizes
    sizes = sorted(set(r.size for r in results))
    
    # Plot tokens/second vs chunk size
    for size in sizes:
        size_results = [r for r in results if r.size == size]
        if not args.compare:  # If not comparing, use all results
            chunk_sizes = []
            tokens_per_sec = []
            
            for r in size_results:
                if "chars/chunk" in r.config:
                    chunk_size = int(r.config.split('(')[1].split()[0])
                    chunk_sizes.append(chunk_size)
                    tokens_per_sec.append(r.tokens_per_second)
            
            if chunk_sizes:  # Only plot if we have chunked results
                ax1.plot(chunk_sizes, tokens_per_sec, marker='o', label=f'{size} chars')
        else:  # If comparing, plot each model separately
            for model in ["GPT-4", model_name]:
                model_results = [r for r in size_results if model in r.config]
                chunk_sizes = []
                tokens_per_sec = []
                
                for r in model_results:
                    if "chars/chunk" in r.config:
                        chunk_size = int(r.config.split('(')[1].split()[0])
                        chunk_sizes.append(chunk_size)
                        tokens_per_sec.append(r.tokens_per_second)
                
                if chunk_sizes:  # Only plot if we have chunked results
                    ax1.plot(chunk_sizes, tokens_per_sec, marker='o',
                            label=f'{size} chars ({model})')
            
    ax1.set_xlabel('Chunk Size (characters)')
    ax1.set_ylabel('Tokens/Second')
    ax1.set_title('Processing Speed vs Chunk Size')
    ax1.grid(True)
    ax1.legend()
    
    # Plot speedup vs input size
    baseline_speeds = {}
    best_speeds = {}
    for size in sizes:
        size_results = [r for r in results if r.size == size and "GPT-4-mini" not in r.config]
        baseline = next((r for r in size_results if r.config == "GPT-4 Baseline"), None)
        if baseline:
            baseline_speeds[size] = baseline.tokens_per_second
            best = max((r for r in size_results if "chars/chunk" in r.config),
                      key=lambda r: r.tokens_per_second)
            best_speeds[size] = best.tokens_per_second
    
    sizes_list = sorted(baseline_speeds.keys())
    speedups = [best_speeds[size]/baseline_speeds[size] for size in sizes_list]
    
    ax2.plot(sizes_list, speedups, marker='o')
    ax2.set_xlabel('Input Size (characters)')
    ax2.set_ylabel('Speedup vs Baseline')
    ax2.set_title('Parallelization Speedup vs Input Size')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Initialize Faker for text generation
fake = Faker()

def generate_text(num_chars: int) -> str:
    """Generate realistic English text of approximately num_chars length."""
    text = ""
    while len(text) < num_chars:
        text += fake.paragraph() + "\n\n"
    return text[:num_chars]

async def extract_content_baseline(text: str, llm) -> tuple[float, dict, Tuple[int, int]]:
    """Baseline: Process entire text at once."""
    start_time = time.time()
    
    prompt = 'Extract and summarize the key points from this text. Respond in JSON format with "topics" and "summary" fields. Text: {text}'
    template = PromptTemplate(input_variables=['text'], template=prompt)
    
    result = await llm.ainvoke(template.format(text=text))
    prompt_tokens = result.token_usage.prompt_tokens if hasattr(result, 'token_usage') else 0
    completion_tokens = result.token_usage.completion_tokens if hasattr(result, 'token_usage') else 0
    
    try:
        parsed = json.loads(result.content)
    except:
        parsed = {"error": "Failed to parse JSON"}
    
    total_time = time.time() - start_time
    return total_time, parsed, (prompt_tokens, completion_tokens)

async def extract_content_chunked(text: str, llm, chunk_size: int) -> tuple[float, dict, Tuple[int, int], int]:
    """Process text in parallel chunks of specified size."""
    start_time = time.time()
    
    # Split into chunks of approximately chunk_size characters
    chunks = []
    start = 0
    while start < len(text):
        # Find the nearest paragraph break after chunk_size characters
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Look for next paragraph break
            next_break = text.find("\n\n", end)
            if next_break != -1:
                end = next_break + 2
        chunks.append(text[start:end])
        start = end
    
    num_chunks = len(chunks)
    
    # Process chunks in parallel
    chunk_prompt = 'Extract key points from this text chunk. Respond in JSON format with "topics" and "summary" fields. Text: {text}'
    chunk_template = PromptTemplate(input_variables=['text'], template=chunk_prompt)
    
    parallel_results = await llm.abatch([
        chunk_template.format(text=chunk) for chunk in chunks
    ])
    
    # Track total tokens
    total_prompt_tokens = sum(r.token_usage.prompt_tokens for r in parallel_results if hasattr(r, 'token_usage'))
    total_completion_tokens = sum(r.token_usage.completion_tokens for r in parallel_results if hasattr(r, 'token_usage'))
    
    # Merge results
    merged = {
        "topics": [],
        "summary": []
    }
    
    for result in parallel_results:
        try:
            chunk_data = json.loads(result.content)
            if isinstance(chunk_data.get("topics"), list):
                merged["topics"].extend(chunk_data["topics"])
            if isinstance(chunk_data.get("summary"), list):
                merged["summary"].extend(chunk_data["summary"])
            elif isinstance(chunk_data.get("summary"), str):
                merged["summary"].append(chunk_data["summary"])
        except:
            continue
    
    total_time = time.time() - start_time
    return total_time, merged, (total_prompt_tokens, total_completion_tokens), num_chunks

def find_best_config(results: List[TestResult], size: int, model_name: str) -> Optional[TestResult]:
    """Find the best configuration for a given text size based on tokens/second."""
    size_results = [r for r in results if r.size == size and model_name in r.config]
    if not size_results:
        return None
    return max(size_results, key=lambda r: r.tokens_per_second)

async def run_tests():
    args = parse_args()
    
    # Test configurations
    text_sizes = [1000, 5000, 10000, 20000, 30000]
    chunk_sizes = [
        2000,    # Small chunks
        5000,    # Medium chunks (recommended)
        8000,    # Large chunks
        12000,   # Extra large chunks
    ]
    
    # Initialize model(s)
    main_model = ChatOpenAI(model=args.model)
    baseline_model = ChatOpenAI(model="gpt-4.1") if args.compare else None
    
    model_name = args.model.replace(".", "_")  # For config string
    
    start_time = datetime.now()
    print(f"\nStarting parallelization tests at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    results: List[TestResult] = []
    
    for size in text_sizes:
        text = generate_text(size)
        
        # Test baseline if comparing
        if args.compare:
            times = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            for i in range(3):  # Run 3 times for average
                time_taken, _, (prompt_tokens, completion_tokens) = await extract_content_baseline(text, baseline_model)
                times.append(time_taken)
                total_prompt_tokens = max(total_prompt_tokens, prompt_tokens)  # Use max in case of caching
                total_completion_tokens = max(total_completion_tokens, completion_tokens)
                
            avg_time = statistics.mean(times)
            cost = estimate_llm_cost(
                model_name="gpt-4.1",
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens
            )
            results.append(TestResult(
                size=size,
                config="GPT-4 Baseline",
                avg_time=avg_time,
                times=times,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cost=cost,
                num_chunks=1
            ))
        
        # Test different chunk sizes with main model
        for chunk_size in chunk_sizes:
            times = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            num_chunks = 0
            for i in range(3):  # Run 3 times for average
                time_taken, _, (prompt_tokens, completion_tokens), chunks = await extract_content_chunked(text, main_model, chunk_size)
                times.append(time_taken)
                total_prompt_tokens = max(total_prompt_tokens, prompt_tokens)
                total_completion_tokens = max(total_completion_tokens, completion_tokens)
                num_chunks = chunks  # Should be same each time
                
            avg_time = statistics.mean(times)
            cost = estimate_llm_cost(
                model_name=args.model,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens
            )
            results.append(TestResult(
                size=size,
                config=f"{model_name} ({chunk_size} chars/chunk)",
                avg_time=avg_time,
                times=times,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cost=cost,
                num_chunks=num_chunks
            ))
        
        # Find best configuration for this size if comparing
        if args.compare:
            best_config = find_best_config(results, size)
            chunk_size = int(best_config.config.split('(')[1].split()[0]) if "chars/chunk" in best_config.config else 0
        
        # Print results for this size
        size_results = [r for r in results if r.size == size]
        headers = ["Config", "Chunks", "Avg Time (s)", "Tokens/Chunk", "Tokens/s", "Cost ($)"]
        table = [[
            r.config,
            r.num_chunks,
            f"{r.avg_time:.2f}",
            f"{r.tokens_per_chunk:.1f}",
            f"{r.tokens_per_second:.1f}",
            f"{r.cost:.4f}"
        ] for r in size_results]
        
        print(f"\nResults for {size} characters:")
        print(tabulate(table, headers=headers, tablefmt="grid"))
        print(f"Best configuration: {best_config.config} ({best_config.tokens_per_second:.1f} tokens/s)")
        print()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print("=" * 80)
    print(f"Tests completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration.total_seconds():.1f}s")
    
    # Print overall summary
    print("\nBest configurations by text size:")
    if args.compare:
        headers = ["Text Size", "Best Config", "Chunks", "Tokens/Chunk", "Speedup vs GPT-4"]
        table = []
        for size in text_sizes:
            size_results = [r for r in results if r.size == size]
            model_results = [r for r in size_results if model_name in r.config]
            best = max(model_results, key=lambda r: r.tokens_per_second)
            baseline = next(r for r in size_results if r.config == "GPT-4 Baseline")
            speedup = best.tokens_per_second / baseline.tokens_per_second
            table.append([
                size,
                best.config,
                best.num_chunks,
                f"{best.tokens_per_chunk:.1f}",
                f"{speedup:.1f}x"
            ])
        print(tabulate(table, headers=headers, tablefmt="grid"))
    else:
        headers = ["Text Size", "Best Config", "Chunks", "Tokens/Chunk", "Tokens/s"]
        table = []
        for size in text_sizes:
            size_results = [r for r in results if r.size == size]
            best = max(size_results, key=lambda r: r.tokens_per_second)
            table.append([
                size,
                best.config,
                best.num_chunks,
                f"{best.tokens_per_chunk:.1f}",
                f"{best.tokens_per_second:.1f}"
            ])
        print(tabulate(table, headers=headers, tablefmt="grid"))
    
    # Create visualization
    plot_results(results)
    print(f"\nVisualization saved as parallelism_results.png")

if __name__ == "__main__":
    asyncio.run(run_tests())