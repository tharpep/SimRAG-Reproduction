"""
Compare Baseline vs SimRAG Results
Loads results from baseline and SimRAG experiments and generates comparison
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from ...simrag.domain_adaptation import DomainAdaptation
from ...config import get_tuning_config
from ...logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def load_results_file(file_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file
    
    Args:
        file_path: Path to results JSON file
        
    Returns:
        Dictionary with results data
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def compare_results(
    baseline_file: str,
    simrag_file: str,
    output_file: str = None,
    use_timestamp: bool = True
) -> Dict[str, Any]:
    """
    Compare baseline and SimRAG results
    
    Args:
        baseline_file: Path to baseline results JSON
        simrag_file: Path to SimRAG results JSON
        output_file: Path to save comparison JSON (optional)
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("=== Comparing Baseline vs SimRAG Results ===")
    
    # Load results
    logger.info(f"Loading baseline results from {baseline_file}...")
    baseline_results = load_results_file(baseline_file)
    
    logger.info(f"Loading SimRAG results from {simrag_file}...")
    simrag_results = load_results_file(simrag_file)
    
    # Extract performance metrics
    # Baseline structure: summary.avg_context_score or context_scores list
    baseline_scores = baseline_results.get("context_scores", [])
    baseline_avg = baseline_results.get("summary", {}).get("avg_context_score", 0.0)
    if not baseline_avg and baseline_scores:
        all_baseline = [score for scores in baseline_scores if scores for score in scores]
        baseline_avg = sum(all_baseline) / len(all_baseline) if all_baseline else 0.0
    
    # SimRAG structure: testing.avg_context_score and testing.context_scores
    simrag_testing = simrag_results.get("testing", {})
    simrag_scores = simrag_testing.get("context_scores", [])
    simrag_avg = simrag_testing.get("avg_context_score", 0.0)
    if not simrag_avg and simrag_scores:
        all_simrag = [score for scores in simrag_scores if scores for score in scores]
        simrag_avg = sum(all_simrag) / len(all_simrag) if all_simrag else 0.0
    
    # Calculate improvement
    improvement_percent = ((simrag_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0.0
    
    # Response times
    baseline_time = baseline_results.get("summary", {}).get("avg_response_time", 0.0)
    simrag_time = simrag_testing.get("avg_response_time", 0.0)
    
    # Create comparison
    comparison = {
        "comparison_type": "baseline_vs_simrag",
        "baseline": {
            "avg_context_score": baseline_avg,
            "avg_response_time": baseline_time,
            "num_questions": len(baseline_results.get("questions", [])),
            "source_file": baseline_file
        },
        "simrag": {
            "avg_context_score": simrag_avg,
            "avg_response_time": simrag_time,
            "num_questions": len(simrag_results.get("questions", [])),
            "source_file": simrag_file,
            "stage1_version": simrag_results.get("stage1", {}).get("version", "N/A"),
            "stage2_version": simrag_results.get("stage2", {}).get("version", "N/A")
        },
        "improvement": {
            "context_score_improvement": simrag_avg - baseline_avg,
            "context_score_improvement_percent": improvement_percent,
            "response_time_change": simrag_time - baseline_time,
            "response_time_change_percent": ((simrag_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0.0
        }
    }
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"Baseline:")
    logger.info(f"  Avg context score: {baseline_avg:.3f}")
    logger.info(f"  Avg response time: {baseline_time:.2f}s")
    logger.info(f"\nSimRAG:")
    logger.info(f"  Avg context score: {simrag_avg:.3f}")
    logger.info(f"  Avg response time: {simrag_time:.2f}s")
    logger.info(f"\nImprovement:")
    logger.info(f"  Context score: {improvement_percent:+.1f}%")
    logger.info(f"  Response time: {comparison['improvement']['response_time_change_percent']:+.1f}%")
    
    # Save comparison
    if output_file:
        # Add timestamp if not already present and use_timestamp is True
        if use_timestamp:
            from experiments.utils import get_timestamped_filename, has_timestamp
            if not has_timestamp(output_file):
                base_name = Path(output_file).stem
                output_file = get_timestamped_filename(base_name, "json")
        
        output_path = Path(__file__).parent / "results" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"\nComparison saved to {output_path}")
        
        # Store the actual filename used
        comparison["_saved_filename"] = str(output_path)
    
    return comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare baseline vs SimRAG results")
    parser.add_argument("--baseline", type=str, required=True,
                       help="Path to baseline results JSON file")
    parser.add_argument("--simrag", type=str, required=True,
                       help="Path to SimRAG results JSON file")
    parser.add_argument("--output", type=str, default="comparison_results.json",
                       help="Output filename (default: comparison_results.json, auto-timestamped)")
    parser.add_argument("--no-timestamp", action="store_true",
                       help="Don't add timestamp to filename (may overwrite existing)")
    
    args = parser.parse_args()
    
    try:
        comparison = compare_results(
            baseline_file=args.baseline,
            simrag_file=args.simrag,
            output_file=args.output,
            use_timestamp=not args.no_timestamp
        )
        print("\n=== Comparison Complete ===")
        saved_file = comparison.get("_saved_filename", args.output)
        print(f"Results saved to: {saved_file}")
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        sys.exit(1)

