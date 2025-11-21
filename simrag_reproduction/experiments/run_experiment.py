"""
Main Experiment Orchestrator
Run SimRAG training pipeline (Stage 1 -> Stage 2)
"""

import argparse
import sys
from pathlib import Path

from ..logging_config import setup_logging, get_logger
from ..config import get_tuning_config
from .utils import set_random_seeds

setup_logging()
logger = get_logger(__name__)


def run_complete_experiment(
    documents_folder: str = "../../data/documents",
    use_real_datasets: bool = True
):
    """
    Run SimRAG training pipeline (Stage 1 -> Stage 2)
    
    Args:
        documents_folder: Path to documents folder
        use_real_datasets: Use Alpaca dataset for Stage 1
    """
    logger.info("="*60)
    logger.info("SIMRAG TRAINING PIPELINE")
    logger.info("="*60)
    
    config = get_tuning_config()
    seed = config.random_seed
    logger.info(f"Setting random seed: {seed} (for reproducibility)")
    set_random_seeds(seed)
    
    logger.info("\nRunning SimRAG Training Pipeline...")
    
    from .simrag.run_full_pipeline import run_full_pipeline
    simrag_results = run_full_pipeline(
        documents_folder=documents_folder,
        use_real_datasets=use_real_datasets,
        output_file="full_pipeline_results.json",
        use_timestamp=True
    )
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"✓ Models trained successfully")
    logger.info(f"✓ Test models locally: simrag experiment test")
    logger.info(f"✓ Export models: simrag experiment export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SimRAG training pipeline (Stage 1 -> Stage 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full training pipeline
  python run_experiment.py
  
  # Use test data for Stage 1 (faster)
  python run_experiment.py --test-data
  
Testing can be done locally with 'simrag experiment test' or exported for external testing.
        """
    )
    
    parser.add_argument("--documents", type=str, default="../../data/documents",
                       help="Path to documents folder (default: ../../data/documents)")
    parser.add_argument("--test-data", action="store_true",
                       help="Use test data for Stage 1 instead of Alpaca")
    
    args = parser.parse_args()
    
    try:
        run_complete_experiment(
            documents_folder=args.documents,
            use_real_datasets=not args.test_data
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

