"""
Results Comparison Script

Compares results between baseline RAG and SimRAG systems.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.evaluation.metrics import compute_improvement_metrics, format_metrics_report


def load_results(results_dir: Path) -> Dict[str, Any]:
    """
    Load results from a results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary containing loaded results
    """
    results = {}
    
    # Load metrics
    metrics_file = results_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            results["metrics"] = json.load(f)
    
    # Load final metrics (for SimRAG)
    final_metrics_file = results_dir / "final_metrics.json"
    if final_metrics_file.exists():
        with open(final_metrics_file, 'r') as f:
            results["final_metrics"] = json.load(f)
    
    # Load improvement history (for SimRAG)
    improvement_file = results_dir / "improvement_history.json"
    if improvement_file.exists():
        with open(improvement_file, 'r') as f:
            results["improvement_history"] = json.load(f)
    
    # Load detailed results
    results_file = results_dir / "results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results["details"] = json.load(f)
    
    return results


def compare_systems(baseline_dir: Path, simrag_dir: Path, output_dir: Path):
    """
    Compare baseline and SimRAG results.
    
    Args:
        baseline_dir: Path to baseline results
        simrag_dir: Path to SimRAG results
        output_dir: Path to save comparison results
    """
    logger.info("Loading results for comparison")
    
    # Load results
    baseline_results = load_results(baseline_dir)
    simrag_results = load_results(simrag_dir)
    
    # Get metrics for comparison
    baseline_metrics = baseline_results.get("metrics", {})
    simrag_metrics = simrag_results.get("final_metrics", simrag_results.get("metrics", {}))
    
    # Compute improvement metrics
    improvement_metrics = compute_improvement_metrics(baseline_metrics, simrag_metrics)
    
    # Create comparison report
    comparison_report = create_comparison_report(
        baseline_metrics, simrag_metrics, improvement_metrics
    )
    
    # Save comparison results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics comparison
    comparison_data = {
        "baseline_metrics": baseline_metrics,
        "simrag_metrics": simrag_metrics,
        "improvement_metrics": improvement_metrics
    }
    
    with open(output_dir / "comparison_metrics.json", "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    # Save formatted report
    with open(output_dir / "comparison_report.txt", "w") as f:
        f.write(comparison_report)
    
    # Create visualizations
    create_comparison_plots(baseline_metrics, simrag_metrics, output_dir)
    
    # Create improvement history plot (if available)
    if "improvement_history" in simrag_results:
        create_improvement_history_plot(simrag_results["improvement_history"], output_dir)
    
    logger.info(f"Comparison results saved to {output_dir}")
    
    # Print summary
    print(comparison_report)


def create_comparison_report(baseline_metrics: Dict[str, float], 
                           simrag_metrics: Dict[str, float],
                           improvement_metrics: Dict[str, float]) -> str:
    """
    Create a formatted comparison report.
    
    Args:
        baseline_metrics: Baseline system metrics
        simrag_metrics: SimRAG system metrics
        improvement_metrics: Improvement metrics
        
    Returns:
        Formatted comparison report
    """
    report = "SimRAG vs Baseline RAG Comparison Report\n"
    report += "=" * 50 + "\n\n"
    
    # Overall improvement summary
    report += "Overall Improvement Summary:\n"
    report += "-" * 30 + "\n"
    
    key_metrics = ["em", "f1", "recall_at_5", "recall_at_10", "ndcg_at_10"]
    for metric in key_metrics:
        if metric in improvement_metrics:
            improvement = improvement_metrics[metric]
            relative_improvement = improvement_metrics.get(f"{metric}_relative_improvement", 0)
            
            report += f"{metric.upper()}: {improvement:+.4f} ({relative_improvement:+.2f}%)\n"
    
    report += "\n"
    
    # Detailed metrics comparison
    report += "Detailed Metrics Comparison:\n"
    report += "-" * 30 + "\n"
    report += f"{'Metric':<20} {'Baseline':<12} {'SimRAG':<12} {'Improvement':<12}\n"
    report += "-" * 60 + "\n"
    
    all_metrics = set(baseline_metrics.keys()) | set(simrag_metrics.keys())
    for metric in sorted(all_metrics):
        baseline_val = baseline_metrics.get(metric, 0.0)
        simrag_val = simrag_metrics.get(metric, 0.0)
        improvement = simrag_val - baseline_val
        
        report += f"{metric:<20} {baseline_val:<12.4f} {simrag_val:<12.4f} {improvement:<+12.4f}\n"
    
    report += "\n"
    
    # Success criteria evaluation
    report += "Success Criteria Evaluation:\n"
    report += "-" * 30 + "\n"
    
    # Check mid-term criteria (+2-3% Recall@k or EM/F1 gain)
    em_improvement = improvement_metrics.get("em", 0)
    f1_improvement = improvement_metrics.get("f1", 0)
    recall_5_improvement = improvement_metrics.get("recall_at_5", 0)
    
    mid_term_success = (em_improvement >= 0.02 or f1_improvement >= 0.02 or 
                       recall_5_improvement >= 0.02)
    
    report += f"Mid-term criteria (≥2% gain): {'✓ PASS' if mid_term_success else '✗ FAIL'}\n"
    report += f"  - EM improvement: {em_improvement:.4f}\n"
    report += f"  - F1 improvement: {f1_improvement:.4f}\n"
    report += f"  - Recall@5 improvement: {recall_5_improvement:.4f}\n"
    
    # Check final criteria (≥5% EM/F1 and Recall@10, ≤5% latency overhead)
    final_em_success = em_improvement >= 0.05
    final_f1_success = f1_improvement >= 0.05
    recall_10_improvement = improvement_metrics.get("recall_at_10", 0)
    final_recall_success = recall_10_improvement >= 0.05
    
    # Check latency overhead (if available)
    latency_improvement = improvement_metrics.get("avg_time_per_query", 0)
    latency_success = latency_improvement <= 0.05  # 5% overhead threshold
    
    final_success = (final_em_success and final_f1_success and 
                    final_recall_success and latency_success)
    
    report += f"\nFinal criteria (≥5% gain, ≤5% latency): {'✓ PASS' if final_success else '✗ FAIL'}\n"
    report += f"  - EM ≥5%: {'✓' if final_em_success else '✗'} ({em_improvement:.4f})\n"
    report += f"  - F1 ≥5%: {'✓' if final_f1_success else '✗'} ({f1_improvement:.4f})\n"
    report += f"  - Recall@10 ≥5%: {'✓' if final_recall_success else '✗'} ({recall_10_improvement:.4f})\n"
    report += f"  - Latency ≤5%: {'✓' if latency_success else '✗'} ({latency_improvement:.4f})\n"
    
    return report


def create_comparison_plots(baseline_metrics: Dict[str, float], 
                          simrag_metrics: Dict[str, float], 
                          output_dir: Path):
    """
    Create comparison plots.
    
    Args:
        baseline_metrics: Baseline system metrics
        simrag_metrics: SimRAG system metrics
        output_dir: Output directory for plots
    """
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create metrics comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SimRAG vs Baseline RAG Comparison', fontsize=16)
    
    # Answer quality metrics
    answer_metrics = ["em", "f1", "semantic_similarity"]
    answer_baseline = [baseline_metrics.get(m, 0) for m in answer_metrics]
    answer_simrag = [simrag_metrics.get(m, 0) for m in answer_metrics]
    
    axes[0, 0].bar(range(len(answer_metrics)), answer_baseline, alpha=0.7, label='Baseline')
    axes[0, 0].bar(range(len(answer_metrics)), answer_simrag, alpha=0.7, label='SimRAG')
    axes[0, 0].set_title('Answer Quality Metrics')
    axes[0, 0].set_xticks(range(len(answer_metrics)))
    axes[0, 0].set_xticklabels([m.upper() for m in answer_metrics])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Retrieval metrics
    retrieval_metrics = ["recall_at_1", "recall_at_5", "recall_at_10"]
    retrieval_baseline = [baseline_metrics.get(m, 0) for m in retrieval_metrics]
    retrieval_simrag = [simrag_metrics.get(m, 0) for m in retrieval_metrics]
    
    axes[0, 1].bar(range(len(retrieval_metrics)), retrieval_baseline, alpha=0.7, label='Baseline')
    axes[0, 1].bar(range(len(retrieval_metrics)), retrieval_simrag, alpha=0.7, label='SimRAG')
    axes[0, 1].set_title('Retrieval Metrics')
    axes[0, 1].set_xticks(range(len(retrieval_metrics)))
    axes[0, 1].set_xticklabels([m.replace('_at_', '@') for m in retrieval_metrics])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # nDCG metrics
    ndcg_metrics = ["ndcg_at_1", "ndcg_at_5", "ndcg_at_10"]
    ndcg_baseline = [baseline_metrics.get(m, 0) for m in ndcg_metrics]
    ndcg_simrag = [simrag_metrics.get(m, 0) for m in ndcg_metrics]
    
    axes[1, 0].bar(range(len(ndcg_metrics)), ndcg_baseline, alpha=0.7, label='Baseline')
    axes[1, 0].bar(range(len(ndcg_metrics)), ndcg_simrag, alpha=0.7, label='SimRAG')
    axes[1, 0].set_title('nDCG Metrics')
    axes[1, 0].set_xticks(range(len(ndcg_metrics)))
    axes[1, 0].set_xticklabels([m.replace('_at_', '@') for m in ndcg_metrics])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance metrics
    perf_metrics = ["avg_time_per_query", "gpu_memory_used_mb"]
    perf_baseline = [baseline_metrics.get(m, 0) for m in perf_metrics]
    perf_simrag = [simrag_metrics.get(m, 0) for m in perf_metrics]
    
    axes[1, 1].bar(range(len(perf_metrics)), perf_baseline, alpha=0.7, label='Baseline')
    axes[1, 1].bar(range(len(perf_metrics)), perf_simrag, alpha=0.7, label='SimRAG')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_xticks(range(len(perf_metrics)))
    axes[1, 1].set_xticklabels([m.replace('_', ' ').title() for m in perf_metrics])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create improvement bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    improvement_metrics = ["em", "f1", "recall_at_5", "recall_at_10", "ndcg_at_10"]
    improvements = [simrag_metrics.get(m, 0) - baseline_metrics.get(m, 0) for m in improvement_metrics]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(range(len(improvement_metrics)), improvements, color=colors, alpha=0.7)
    
    ax.set_title('SimRAG Improvement Over Baseline', fontsize=14)
    ax.set_xticks(range(len(improvement_metrics)))
    ax.set_xticklabels([m.replace('_at_', '@').upper() for m in improvement_metrics])
    ax.set_ylabel('Improvement')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                f'{imp:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_chart.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_improvement_history_plot(improvement_history: List[Dict[str, float]], output_dir: Path):
    """
    Create improvement history plot.
    
    Args:
        improvement_history: List of improvement metrics for each iteration
        output_dir: Output directory for plots
    """
    if not improvement_history:
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(improvement_history)
    df['iteration'] = range(1, len(df) + 1)
    
    # Create improvement history plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SimRAG Self-Improvement History', fontsize=16)
    
    # Plot key metrics over iterations
    key_metrics = ["em", "f1", "recall_at_5", "recall_at_10"]
    
    for i, metric in enumerate(key_metrics):
        if metric in df.columns:
            row, col = i // 2, i % 2
            axes[row, col].plot(df['iteration'], df[metric], marker='o', linewidth=2, markersize=6)
            axes[row, col].set_title(f'{metric.upper()} Improvement')
            axes[row, col].set_xlabel('Iteration')
            axes[row, col].set_ylabel('Improvement')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_history.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(description="Compare baseline and SimRAG results")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline results")
    parser.add_argument("--simrag", type=str, required=True, help="Path to SimRAG results")
    parser.add_argument("--output", type=str, default="./results/comparisons", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run comparison
    baseline_dir = Path(args.baseline)
    simrag_dir = Path(args.simrag)
    output_dir = Path(args.output)
    
    compare_systems(baseline_dir, simrag_dir, output_dir)


if __name__ == "__main__":
    main()
