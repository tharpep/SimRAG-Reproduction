"""
Main Experiment Orchestrator
Run complete experiment pipeline: Baseline -> SimRAG -> Comparison
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_config import setup_logging, get_logger

# Import experiment modules
from experiments.baseline.run_baseline import run_baseline_test
from experiments.simrag.run_stage1 import run_stage1_training
from experiments.simrag.run_stage2 import run_stage2_training
from experiments.comparison.compare_results import compare_results

# Setup logging
setup_logging()
logger = get_logger(__name__)


def run_complete_experiment(
    documents_folder: str = "../HTML_DOCS",
    use_real_datasets: bool = True,
    skip_baseline: bool = False,
    skip_simrag: bool = False,
    baseline_file: str = None,
    simrag_file: str = None
):
    """
    Run complete experiment pipeline
    
    Args:
        documents_folder: Path to documents folder
        use_real_datasets: Use Alpaca dataset for Stage 1
        skip_baseline: Skip baseline experiment (use existing results)
        skip_simrag: Skip SimRAG training (use existing results)
        baseline_file: Path to existing baseline results (if skip_baseline)
        simrag_file: Path to existing SimRAG results (if skip_simrag)
    """
    logger.info("="*60)
    logger.info("COMPLETE EXPERIMENT PIPELINE")
    logger.info("="*60)
    
    results_dir = Path(__file__).parent
    
    # Step 1: Baseline
    if not skip_baseline:
        logger.info("\n[1/3] Running Baseline Experiment...")
        baseline_results = run_baseline_test(
            documents_folder=documents_folder,
            output_file="baseline_results.json",
            use_timestamp=True
        )
        # Get the actual filename that was saved
        baseline_file = baseline_results.get("_saved_filename") or str(results_dir / "baseline" / "results" / "baseline_results.json")
    else:
        if not baseline_file:
            baseline_file = str(results_dir / "baseline" / "results" / "baseline_results.json")
        logger.info(f"\n[1/3] Skipping Baseline (using: {baseline_file})")
    
    # Step 2: SimRAG
    if not skip_simrag:
        logger.info("\n[2/3] Running SimRAG Pipeline...")
        
        # Stage 1
        logger.info("  Stage 1: Instruction Following...")
        stage1_results = run_stage1_training(
            use_real_datasets=use_real_datasets,
            output_file="stage1_results.json"
        )
        stage1_model_path = stage1_results["training"]["model_path"]
        
        # Stage 2
        logger.info("  Stage 2: Domain Adaptation...")
        stage2_results = run_stage2_training(
            documents_folder=documents_folder,
            stage_1_model_path=stage1_model_path,
            output_file="stage2_results.json"
        )
        
        # For comparison, we need full pipeline results
        # Create a combined SimRAG results file
        from experiments.simrag.run_full_pipeline import run_full_pipeline
        simrag_results = run_full_pipeline(
            documents_folder=documents_folder,
            use_real_datasets=use_real_datasets,
            output_file="full_pipeline_results.json",
            use_timestamp=True
        )
        # Get the actual filename that was saved
        simrag_file = simrag_results.get("_saved_filename") or str(results_dir / "simrag" / "results" / "full_pipeline_results.json")
    else:
        if not simrag_file:
            simrag_file = str(results_dir / "simrag" / "results" / "full_pipeline_results.json")
        logger.info(f"\n[2/3] Skipping SimRAG (using: {simrag_file})")
    
    # Step 3: Comparison
    logger.info("\n[3/3] Comparing Results...")
    comparison = compare_results(
        baseline_file=baseline_file,
        simrag_file=simrag_file,
        output_file="comparison_results.json"
    )
    
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("="*60)
    logger.info(f"Baseline results: {baseline_file}")
    logger.info(f"SimRAG results: {simrag_file}")
    logger.info(f"Comparison results: experiments/comparison/results/comparison_results.json")
    logger.info(f"\nImprovement: {comparison['improvement']['context_score_improvement_percent']:+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete SimRAG experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_experiment.py
  
  # Use test data for Stage 1 (faster)
  python run_experiment.py --test-data
  
  # Skip baseline (use existing)
  python run_experiment.py --skip-baseline
  
  # Skip SimRAG (use existing)
  python run_experiment.py --skip-simrag
        """
    )
    
    parser.add_argument("--documents", type=str, default="../HTML_DOCS",
                       help="Path to documents folder (default: ../HTML_DOCS)")
    parser.add_argument("--test-data", action="store_true",
                       help="Use test data for Stage 1 instead of Alpaca")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline experiment")
    parser.add_argument("--skip-simrag", action="store_true",
                       help="Skip SimRAG training")
    parser.add_argument("--baseline-file", type=str, default=None,
                       help="Path to existing baseline results (if skipping)")
    parser.add_argument("--simrag-file", type=str, default=None,
                       help="Path to existing SimRAG results (if skipping)")
    
    args = parser.parse_args()
    
    try:
        run_complete_experiment(
            documents_folder=args.documents,
            use_real_datasets=not args.test_data,
            skip_baseline=args.skip_baseline,
            skip_simrag=args.skip_simrag,
            baseline_file=args.baseline_file,
            simrag_file=args.simrag_file
        )
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

