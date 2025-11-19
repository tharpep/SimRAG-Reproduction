"""
Run SimRAG Stage 2 Training
Fine-tune model on synthetic QA pairs from domain documents
"""

import json
from pathlib import Path
from datetime import datetime

from simrag_reproduction.simrag.domain_adaptation import DomainAdaptation
from simrag_reproduction.config import get_tuning_config
from simrag_reproduction.experiments.utils import load_documents_from_folder
from simrag_reproduction.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def run_stage2_training(
    documents_folder: str = "../HTML_DOCS",
    stage_1_model_path: str = None,
    output_file: str = None,
    use_timestamp: bool = True
) -> dict:
    """
    Run SimRAG Stage 2 training
    
    Args:
        documents_folder: Path to folder containing domain documents
        stage_1_model_path: Path to Stage 1 model (optional, uses latest if None)
        output_file: Path to save results JSON (optional)
        
    Returns:
        Dictionary with training results
    """
    logger.info("=== SimRAG Stage 2 Training ===")
    
    # Get config
    config = get_tuning_config()
    
    # Load documents
    logger.info(f"Loading documents from {documents_folder}...")
    documents = load_documents_from_folder(documents_folder, include_html=True)
    
    if not documents:
        raise ValueError(f"No documents found in {documents_folder}")
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Initialize Stage 2 trainer
    logger.info(f"Initializing Stage 2 trainer with model: {config.model_name}")
    stage2 = DomainAdaptation(
        model_name=config.model_name,
        config=config,
        stage_1_model_path=stage_1_model_path
    )
    
    # Train Stage 2
    logger.info("Starting Stage 2 training...")
    try:
        version = stage2.train_stage_2(
            documents=documents,
            questions_per_doc=config.simrag_questions_per_doc,
            min_context_score=config.simrag_min_context_score,
            notes="SimRAG Stage 2 - Domain Adaptation"
        )
        
        if not version:
            raise Exception("Training completed but no version returned")
        
        # Prepare results
        results = {
            "experiment_type": "simrag_stage2",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model_name": config.model_name,
                "device": config.device,
                "batch_size": config.optimized_batch_size,
                "epochs": config.simrag_stage_2_epochs,
                "questions_per_doc": config.simrag_questions_per_doc,
                "min_context_score": config.simrag_min_context_score
            },
            "dataset": {
                "documents_folder": documents_folder,
                "num_documents": len(documents)
            },
            "training": {
                "version": version.version,
                "training_time_seconds": version.training_time_seconds,
                "final_loss": version.final_loss,
                "model_path": stage2.get_model_from_registry(version.version)
            }
        }
        
        logger.info(f"Stage 2 training completed successfully!")
        logger.info(f"  Version: {version.version}")
        logger.info(f"  Training time: {version.training_time_seconds:.1f}s")
        logger.info(f"  Final loss: {version.final_loss:.4f if version.final_loss else 'N/A'}")
        
        # Save results
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
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
            
            # Store the actual filename used
            results["_saved_filename"] = str(output_path)
        
        return results
        
    except Exception as e:
        logger.error(f"Stage 2 training failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SimRAG Stage 2 training")
    parser.add_argument("--documents", type=str, default="../HTML_DOCS",
                       help="Path to documents folder (default: ../HTML_DOCS)")
    parser.add_argument("--stage1-model", type=str, default=None,
                       help="Path to Stage 1 model (uses latest if not specified)")
    parser.add_argument("--output", type=str, default="stage2_results.json",
                       help="Output filename (default: stage2_results.json, auto-timestamped)")
    parser.add_argument("--no-timestamp", action="store_true",
                       help="Don't add timestamp to filename (may overwrite existing)")
    
    args = parser.parse_args()
    
    try:
        results = run_stage2_training(
            documents_folder=args.documents,
            stage_1_model_path=args.stage1_model,
            output_file=args.output,
            use_timestamp=not args.no_timestamp
        )
        print("\n=== Stage 2 Training Complete ===")
        saved_file = results.get("_saved_filename", args.output)
        print(f"Results saved to: {saved_file}")
    except Exception as e:
        logger.error(f"Stage 2 training failed: {e}")
        sys.exit(1)

