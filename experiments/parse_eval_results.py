"""
Parse evaluation results from markdown file to extract specific code candidates.
"""

import json
import re
from typing import List, Dict, Any


def parse_markdown_results(md_path: str) -> List[Dict[str, Any]]:
    """Parse the markdown file and extract all runs with their metadata and code."""
    
    with open(md_path, 'r') as f:
        content = f.read()
    
    # Split by run headers
    run_sections = re.split(r'^## Run (\d+)', content, flags=re.MULTILINE)
    
    runs = []
    
    # Skip the first element (content before first run)
    for i in range(1, len(run_sections), 2):
        run_id = int(run_sections[i])
        run_content = run_sections[i + 1]
        
        # Extract config
        model_match = re.search(r'- Model: `([^`]+)`', run_content)
        protocol_match = re.search(r'- With Protocol: `([^`]+)`', run_content)
        
        # Extract metrics
        gen_lat_match = re.search(r'- Generation Latency: ([\d.]+)s', run_content)
        est_cost_match = re.search(r'- Estimated Cost: ([\d.]+)s', run_content)
        pass_match = re.search(r'- Overall Pass: ([✓✗])', run_content)
        failure_types_match = re.search(r'- Failure Types: (.+)', run_content)
        
        # Extract code
        code_match = re.search(r'```python\n(.*?)\n```', run_content, re.DOTALL)
        
        run_data = {
            'run_id': run_id,
            'model': model_match.group(1) if model_match else None,
            'with_protocol': protocol_match.group(1) == 'True' if protocol_match else None,
            'generation_latency': float(gen_lat_match.group(1)) if gen_lat_match else None,
            'estimated_cost': float(est_cost_match.group(1)) if est_cost_match else None,
            'passed': pass_match.group(1) == '✓' if pass_match else False,
            'failure_types': failure_types_match.group(1).split(', ') if failure_types_match else [],
            'code': code_match.group(1) if code_match else None,
        }
        
        runs.append(run_data)
    
    return runs


def filter_runs(runs: List[Dict[str, Any]], 
                model: str = None, 
                with_protocol: bool = None,
                passed: bool = None) -> List[Dict[str, Any]]:
    """Filter runs by criteria."""
    
    filtered = runs
    
    if model is not None:
        filtered = [r for r in filtered if r['model'] == model]
    
    if with_protocol is not None:
        filtered = [r for r in filtered if r['with_protocol'] == with_protocol]
    
    if passed is not None:
        filtered = [r for r in filtered if r['passed'] == passed]
    
    return filtered


def get_cost_extremes(runs: List[Dict[str, Any]], n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Get the n lowest and highest cost runs."""
    
    sorted_runs = sorted(runs, key=lambda x: x['estimated_cost'] or 0)
    
    return {
        'lowest': sorted_runs[:n],
        'highest': sorted_runs[-n:],
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parse_eval_results.py <markdown_file>")
        sys.exit(1)
    
    md_path = sys.argv[1]
    
    print(f"Parsing {md_path}...")
    runs = parse_markdown_results(md_path)
    print(f"Found {len(runs)} runs\n")
    
    # Example: Get gemini-2.0-flash-lite WITH protocol, passing runs
    flash_lite_passing = filter_runs(
        runs, 
        model='gemini-2.0-flash-lite',
        with_protocol=True,
        passed=True
    )
    
    print(f"gemini-2.0-flash-lite WITH protocol, PASSING: {len(flash_lite_passing)} runs")
    
    if flash_lite_passing:
        extremes = get_cost_extremes(flash_lite_passing, n=3)
        
        print("\n" + "="*80)
        print("LOWEST COST (passing):")
        print("="*80)
        for r in extremes['lowest']:
            print(f"\nRun {r['run_id']}: Cost={r['estimated_cost']}s, GenLat={r['generation_latency']:.2f}s")
            print(f"Code ({len(r['code'])} chars):\n")
            print(r['code'])
            print("\n" + "-"*80)
        
        print("\n" + "="*80)
        print("HIGHEST COST (passing):")
        print("="*80)
        for r in extremes['highest']:
            print(f"\nRun {r['run_id']}: Cost={r['estimated_cost']}s, GenLat={r['generation_latency']:.2f}s")
            print(f"Code ({len(r['code'])} chars):\n")
            print(r['code'])
            print("\n" + "-"*80)
