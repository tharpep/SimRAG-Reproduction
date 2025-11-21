"""
Comparison Logic
Statistical comparison of baseline vs fine-tuned results
Matches Colab notebook exactly
"""

import math
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from ...logging_config import get_logger
from ..utils import has_timestamp

logger = get_logger(__name__)


def calculate_stats(scores_list):
    """
    Calculate statistical measures for context scores (matches Colab)
    
    Returns: mean, std, 95% confidence interval
    Used for determining statistical significance of improvements
    """
    all_scores = []
    for scores in scores_list:
        if isinstance(scores, list):
            all_scores.extend(scores)
        elif isinstance(scores, (int, float)):
            all_scores.append(scores)
    
    if not all_scores:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
    
    n = len(all_scores)
    mean = sum(all_scores) / n
    variance = sum((x - mean) ** 2 for x in all_scores) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    se = std / math.sqrt(n) if n > 0 else 0.0
    margin = 1.96 * se  # 95% confidence interval
    ci_lower = mean - margin
    ci_upper = mean + margin
    
    return {"mean": mean, "std": std, "ci_lower": ci_lower, "ci_upper": ci_upper, "n": n}


def compare_results(
    baseline_results: Dict[str, Any],
    finetuned_results: Dict[str, Any],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare baseline vs fine-tuned results (matches Colab comparison)
    
    Args:
        baseline_results: Baseline test results
        finetuned_results: Fine-tuned test results
        output_file: Optional output filename
        
    Returns:
        Dictionary with comparison results (matches Colab format)
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    
    baseline_avg = baseline_results["summary"]["avg_context_score"]
    simrag_avg = finetuned_results["summary"]["avg_context_score"]
    improvement_percent = ((simrag_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0.0
    
    baseline_stats = calculate_stats(baseline_results["context_scores"])
    simrag_stats = calculate_stats(finetuned_results["context_scores"])
    
    baseline_time = baseline_results["summary"]["avg_response_time"]
    simrag_time = finetuned_results["summary"]["avg_response_time"]
    
    # Create comparison dictionary (matches Colab format)
    comparison = {
        "comparison_type": "baseline_vs_simrag",
        "timestamp": datetime.now().isoformat(),
        "validation": {
            "is_valid": True,
            "baseline_questions": len(baseline_results["questions"]),
            "simrag_questions": len(finetuned_results["questions"]),
            "same_questions": baseline_results["questions"] == finetuned_results["questions"],
            "baseline_docs": baseline_results["dataset"]["num_documents"],
            "simrag_docs": finetuned_results["dataset"]["num_documents"],
            "same_docs": baseline_results["dataset"]["num_documents"] == finetuned_results["dataset"]["num_documents"]
        },
        "baseline": {
            "avg_context_score": baseline_avg,
            "context_score_stats": baseline_stats,
            "avg_response_time": baseline_time,
            "avg_answer_quality": baseline_results["summary"].get("avg_overall_score", 0.0),
            "num_questions": len(baseline_results["questions"])
        },
        "simrag": {
            "avg_context_score": simrag_avg,
            "context_score_stats": simrag_stats,
            "avg_response_time": simrag_time,
            "avg_answer_quality": finetuned_results["summary"].get("avg_overall_score", 0.0),
            "num_questions": len(finetuned_results["questions"]),
            "model": finetuned_results.get("model", {})
        },
        "improvement": {
            "context_score_improvement": simrag_avg - baseline_avg,
            "context_score_improvement_percent": improvement_percent,
            "answer_quality_improvement": finetuned_results["summary"].get("avg_overall_score", 0.0) - baseline_results["summary"].get("avg_overall_score", 0.0),
            "response_time_change": simrag_time - baseline_time,
            "response_time_change_percent": ((simrag_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0.0,
            "statistical_significance": {
                "baseline_ci": f"[{baseline_stats['ci_lower']:.3f}, {baseline_stats['ci_upper']:.3f}]",
                "simrag_ci": f"[{simrag_stats['ci_lower']:.3f}, {simrag_stats['ci_upper']:.3f}]",
                "overlap": not (baseline_stats['ci_upper'] < simrag_stats['ci_lower'] or simrag_stats['ci_upper'] < baseline_stats['ci_lower']),
                "note": "If CIs don't overlap, difference is statistically significant at p<0.05"
            }
        }
    }
    
    # Save comparison results
    if output_file:
        import json
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not has_timestamp(output_file):
            base_name = Path(output_file).stem
            output_file = f"{base_name}_{timestamp}.json"
        
        # Save to root comparison_results folder (matches Colab notebook location)
        # Path: __file__ is at simrag_reproduction/experiments/local_testing/comparison.py
        # Root is 3 levels up: local_testing -> experiments -> simrag_reproduction -> root
        project_root = Path(__file__).parent.parent.parent.parent
        output_path = project_root / "comparison_results" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"\n✓ Comparison saved to {output_path}")
        comparison["_saved_filename"] = str(output_path)
    
    logger.info("\n=== Comparison Complete ===")
    logger.info(f"✓ Raw statistics calculated")
    logger.info(f"\n📊 Quick Stats:")
    logger.info(f"  Baseline context score: {baseline_avg:.3f}")
    logger.info(f"  SimRAG context score:   {simrag_avg:.3f}")
    logger.info(f"  Improvement:            {improvement_percent:+.1f}%")
    
    return comparison

