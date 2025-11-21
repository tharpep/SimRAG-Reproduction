"""
Main Experiment Orchestrator
Run SimRAG training pipeline (Stage 1 -> Stage 2)
Note: Testing and comparison are done separately in the Colab notebook
"""

import argparse
import sys
from pathlib import Path

from ..logging_config import setup_logging, get_logger
from ..config import get_tuning_config
from .utils import set_random_seeds

# Setup logging
setup_logging()
logger = get_logger(__name__)


def run_complete_experiment(
    documents_folder: str = "../../data/documents",
    use_real_datasets: bool = True
):
    """
    Run SimRAG training pipeline (Stage 1 -> Stage 2)
    
    Note: This function ONLY trains models. Testing and comparison
    are done separately in the Colab notebook.
    
    Args:
        documents_folder: Path to documents folder
        use_real_datasets: Use Alpaca dataset for Stage 1
    """
    logger.info("="*60)
    logger.info("SIMRAG TRAINING PIPELINE")
    logger.info("="*60)
    logger.info("Note: Testing and comparison are done in Colab notebook")
    
    # Set random seeds for reproducibility
    config = get_tuning_config()
    seed = config.random_seed
    logger.info(f"Setting random seed: {seed} (for reproducibility)")
    set_random_seeds(seed)
    
    # Run SimRAG training pipeline (Stage 1 -> Stage 2)
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
    logger.info(f"✓ Export models using: simrag experiment export")
    logger.info(f"✓ Test models in Colab notebook: test_model_colab.ipynb")


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
  
Note: Testing and comparison are done separately in the Colab notebook.
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

