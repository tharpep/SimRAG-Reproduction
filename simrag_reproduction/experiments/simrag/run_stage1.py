"""
Run SimRAG Stage 1 Training
Fine-tune model on instruction-following dataset (Alpaca)
"""

import json
import sys
from pathlib import Path
from datetime import datetime

from ...simrag.instruction_following import InstructionFollowing
from ...config import get_tuning_config
from ...logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def run_stage1_training(
    use_real_datasets: bool = True,
    output_file: str = None,
    use_timestamp: bool = True
) -> dict:
    """
    Run SimRAG Stage 1 training
    
    Args:
        use_real_datasets: Use Alpaca dataset (True) or test data (False)
        output_file: Path to save results JSON (optional)
        
    Returns:
        Dictionary with training results
    """
    logger.info("=== SimRAG Stage 1 Training ===")
    
    # Get config
    config = get_tuning_config()
    
    # Initialize Stage 1 trainer
    logger.info(f"Initializing Stage 1 trainer with model: {config.model_name}")
    stage1 = InstructionFollowing(model_name=config.model_name, config=config)
    
    # Train Stage 1
    logger.info(f"Starting Stage 1 training (use_real_datasets={use_real_datasets})...")
    try:
        version = stage1.train_stage_1(
            use_real_datasets=use_real_datasets,
            notes="SimRAG Stage 1 - Instruction Following"
        )
        
        if not version:
            raise Exception("Training completed but no version returned")
        
        # Prepare results
        results = {
            "experiment_type": "simrag_stage1",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model_name": config.model_name,
                "device": config.device,
                "batch_size": config.optimized_batch_size,
                "epochs": config.simrag_stage_1_epochs,
                "use_real_datasets": use_real_datasets
            },
            "training": {
                "version": version.version,
                "training_time_seconds": version.training_time_seconds,
                "final_loss": version.final_loss,
                "model_path": stage1.get_model_from_registry(version.version, stage="stage_1")
            }
        }
        
        logger.info(f"Stage 1 training completed successfully!")
        logger.info(f"  Version: {version.version}")
        logger.info(f"  Training time: {version.training_time_seconds:.1f}s")
        loss_str = f"{version.final_loss:.4f}" if version.final_loss is not None else "N/A"
        logger.info(f"  Final loss: {loss_str}")
        
        # Save results
        if output_file:
            # Add timestamp if not already present and use_timestamp is True
            if use_timestamp:
                from ..utils import get_timestamped_filename, has_timestamp
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
        logger.error(f"Stage 1 training failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SimRAG Stage 1 training")
    parser.add_argument("--test-data", action="store_true",
                       help="Use test data instead of Alpaca dataset")
    parser.add_argument("--output", type=str, default="stage1_results.json",
                       help="Output filename (default: stage1_results.json, auto-timestamped)")
    parser.add_argument("--no-timestamp", action="store_true",
                       help="Don't add timestamp to filename (may overwrite existing)")
    
    args = parser.parse_args()
    
    try:
        results = run_stage1_training(
            use_real_datasets=not args.test_data,
            output_file=args.output,
            use_timestamp=not args.no_timestamp
        )
        print("\n=== Stage 1 Training Complete ===")
        saved_file = results.get("_saved_filename", args.output)
        print(f"Results saved to: {saved_file}")
    except Exception as e:
        logger.error(f"Stage 1 training failed: {e}")
        sys.exit(1)

