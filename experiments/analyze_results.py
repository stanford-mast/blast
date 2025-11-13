"""
Simple analysis script to visualize evaluation results.

Reads the summary JSON and prints useful tables and insights.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_results(results_file: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def print_comparison_table(results: Dict[str, Any]):
    """Print comparison table of all configurations."""
    print("\n" + "="*100)
    print("CODE GENERATION EVALUATION RESULTS")
    print("="*100 + "\n")
    
    # Extract configs (exclude page_load)
    configs = {k: v for k, v in results.items() if k != "page_load"}
    
    # Print header
    print(f"{'Config':<60} {'Runs':>6} {'Gen Lat':>10} {'Est Cost':>10} {'Pass Rate':>10}")
    print("-" * 100)
    
    # Print each config
    for config_name, metrics in sorted(configs.items()):
        model = metrics['model'].split('/')[-1][:30]  # Truncate long model names
        protocol = "✓" if metrics['with_protocol'] else "✗"
        
        gen_lat = f"{metrics['avg_generation_latency']:.2f}s"
        est_cost = f"{metrics['avg_estimated_cost']:.1f}s"
        pass_rate = f"{metrics['overall_pass_rate']:.1%}"
        
        config_display = f"{model} (proto={protocol})"
        
        print(f"{config_display:<60} {metrics['num_runs']:>6} {gen_lat:>10} {est_cost:>10} {pass_rate:>10}")
    
    print("-" * 100)


def print_variance_analysis(results: Dict[str, Any]):
    """Analyze variance to answer RQ1 and RQ3."""
    print("\n" + "="*100)
    print("VARIANCE ANALYSIS (RQ1 & RQ3)")
    print("="*100 + "\n")
    
    configs = {k: v for k, v in results.items() if k != "page_load"}
    
    print(f"{'Config':<60} {'Gen Lat Var':>15} {'Est Cost Var':>15}")
    print("-" * 100)
    
    for config_name, metrics in sorted(configs.items()):
        model = metrics['model'].split('/')[-1][:30]
        protocol = "✓" if metrics['with_protocol'] else "✗"
        config_display = f"{model} (proto={protocol})"
        
        gen_var = f"{metrics['var_generation_latency']:.4f}"
        cost_var = f"{metrics['var_estimated_cost']:.2f}"
        
        print(f"{config_display:<60} {gen_var:>15} {cost_var:>15}")
    
    print("-" * 100)
    print("\nInsights:")
    print("  • High 'Gen Lat Var' → parallel generation helps (RQ3)")
    print("  • High 'Est Cost Var' → cost-based ranking valuable (RQ1)")


def print_protocol_comparison(results: Dict[str, Any]):
    """Compare with_protocol=True vs False (RQ4)."""
    print("\n" + "="*100)
    print("PROTOCOL IMPACT ANALYSIS (RQ4)")
    print("="*100 + "\n")
    
    # Group by model
    by_model = {}
    for config_name, metrics in results.items():
        if config_name == "page_load":
            continue
        
        model = metrics['model']
        if model not in by_model:
            by_model[model] = {}
        
        with_proto = metrics['with_protocol']
        by_model[model][with_proto] = metrics
    
    for model, configs in by_model.items():
        print(f"\nModel: {model}")
        print("-" * 100)
        
        if True in configs and False in configs:
            with_proto = configs[True]
            without_proto = configs[False]
            
            print(f"{'Metric':<30} {'With Protocol':>20} {'Without Protocol':>20} {'Difference':>20}")
            print("-" * 100)
            
            # Pass rate
            diff_pass = with_proto['overall_pass_rate'] - without_proto['overall_pass_rate']
            print(f"{'Pass Rate':<30} {with_proto['overall_pass_rate']:>19.1%} {without_proto['overall_pass_rate']:>19.1%} {diff_pass:>+19.1%}")
            
            # Generation latency
            diff_gen = with_proto['avg_generation_latency'] - without_proto['avg_generation_latency']
            print(f"{'Avg Gen Latency (s)':<30} {with_proto['avg_generation_latency']:>19.2f} {without_proto['avg_generation_latency']:>19.2f} {diff_gen:>+19.2f}")
            
            # Estimated cost
            diff_cost = with_proto['avg_estimated_cost'] - without_proto['avg_estimated_cost']
            print(f"{'Avg Est Cost (s)':<30} {with_proto['avg_estimated_cost']:>19.1f} {without_proto['avg_estimated_cost']:>19.1f} {diff_cost:>+19.1f}")
            
            # Individual pass rates
            print(f"\n{'Pass Rate Breakdown':<30}")
            for check_type in ['syntax', 'safety', 'ordering', 'types']:
                key = f'{check_type}_pass_rate'
                if key in with_proto:
                    diff = with_proto[key] - without_proto[key]
                    print(f"  {check_type.capitalize():<28} {with_proto[key]:>19.1%} {without_proto[key]:>19.1%} {diff:>+19.1%}")


def print_page_load_comparison(results: Dict[str, Any]):
    """Compare generation time vs page load time (RQ5)."""
    print("\n" + "="*100)
    print("PAGE LOAD COMPARISON (RQ5)")
    print("="*100 + "\n")
    
    if "page_load" not in results:
        print("No page load data available.")
        return
    
    page_load = results["page_load"]
    avg_load = page_load["avg_load_latency"]
    
    print(f"Average Page Load Time: {avg_load:.2f}s\n")
    print(f"{'Config':<60} {'Avg Gen Lat':>15} {'vs Page Load':>15} {'Status':>15}")
    print("-" * 100)
    
    configs = {k: v for k, v in results.items() if k != "page_load"}
    
    for config_name, metrics in sorted(configs.items()):
        model = metrics['model'].split('/')[-1][:30]
        protocol = "✓" if metrics['with_protocol'] else "✗"
        config_display = f"{model} (proto={protocol})"
        
        gen_lat = metrics['avg_generation_latency']
        diff = gen_lat - avg_load
        
        if gen_lat < avg_load:
            status = "✓ FREE"
        else:
            status = "✗ OVERHEAD"
        
        print(f"{config_display:<60} {gen_lat:>14.2f}s {diff:>+14.2f}s {status:>15}")
    
    print("-" * 100)
    print("\nInsights:")
    print("  • 'FREE' = code generation completes before page loads (no added latency)")
    print("  • 'OVERHEAD' = code generation adds latency beyond page load")


def print_summary(results: Dict[str, Any]):
    """Print overall summary and recommendations."""
    print("\n" + "="*100)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*100 + "\n")
    
    configs = {k: v for k, v in results.items() if k != "page_load"}
    
    # Find best config overall
    best_config = max(configs.items(), key=lambda x: x[1]['overall_pass_rate'])
    
    print("Best Overall Config (by pass rate):")
    print(f"  • {best_config[0]}")
    print(f"  • Pass Rate: {best_config[1]['overall_pass_rate']:.1%}")
    print(f"  • Avg Generation Latency: {best_config[1]['avg_generation_latency']:.2f}s")
    print(f"  • Avg Estimated Cost: {best_config[1]['avg_estimated_cost']:.1f}s")
    
    # Find fastest generation
    fastest_config = min(configs.items(), key=lambda x: x[1]['avg_generation_latency'])
    
    print("\nFastest Generation Config:")
    print(f"  • {fastest_config[0]}")
    print(f"  • Avg Generation Latency: {fastest_config[1]['avg_generation_latency']:.2f}s")
    print(f"  • Pass Rate: {fastest_config[1]['overall_pass_rate']:.1%}")
    
    # Find lowest estimated cost
    lowest_cost_config = min(configs.items(), key=lambda x: x[1]['avg_estimated_cost'])
    
    print("\nLowest Estimated Cost Config:")
    print(f"  • {lowest_cost_config[0]}")
    print(f"  • Avg Estimated Cost: {lowest_cost_config[1]['avg_estimated_cost']:.1f}s")
    print(f"  • Pass Rate: {lowest_cost_config[1]['overall_pass_rate']:.1%}")
    
    print("\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_json_file>")
        print("Example: python analyze_results.py experiments/results/rq_results.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)
    
    results = load_results(results_file)
    
    # Print all analyses
    print_comparison_table(results)
    print_variance_analysis(results)
    print_protocol_comparison(results)
    print_page_load_comparison(results)
    print_summary(results)


if __name__ == '__main__':
    main()
