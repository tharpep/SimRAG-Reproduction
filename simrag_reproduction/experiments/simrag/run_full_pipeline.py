"""
Run Full SimRAG Pipeline
Orchestrates Stage 1 -> Stage 2 -> Testing
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simrag.instruction_following import InstructionFollowing
from simrag.domain_adaptation import DomainAdaptation
from rag.rag_setup import BasicRAG
from config import get_tuning_config, get_rag_config
from experiments.utils import load_documents_from_folder, get_test_questions
from logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def run_full_pipeline(
    documents_folder: str = "../HTML_DOCS",
    use_real_datasets: bool = True,
    test_questions: list = None,
    output_file: str = None,
    use_timestamp: bool = True
) -> dict:
    """
    Run full SimRAG pipeline: Stage 1 -> Stage 2 -> Testing
    
    Args:
        documents_folder: Path to folder containing domain documents
        use_real_datasets: Use Alpaca dataset for Stage 1 (True) or test data (False)
        test_questions: List of test questions (uses default if None)
        output_file: Path to save results JSON (optional)
        
    Returns:
        Dictionary with complete pipeline results
    """
    logger.info("=== Full SimRAG Pipeline ===")
    
    # Get test questions
    if test_questions is None:
        test_questions = get_test_questions()
    
    # Get configs
    tuning_config = get_tuning_config()
    rag_config = get_rag_config()
    
    results = {
        "experiment_type": "simrag_full_pipeline",
        "timestamp": datetime.now().isoformat(),
        "stage1": {},
        "stage2": {},
        "testing": {}
    }
    
    # Stage 1: Instruction Following
    logger.info("\n" + "="*60)
    logger.info("STAGE 1: Instruction Following Training")
    logger.info("="*60)
    
    stage1 = InstructionFollowing(model_name=tuning_config.model_name, config=tuning_config)
    version1 = stage1.train_stage_1(use_real_datasets=use_real_datasets)
    
    if not version1:
        raise Exception("Stage 1 training failed")
    
    stage1_model_path = stage1.get_model_from_registry(version1.version)
    results["stage1"] = {
        "version": version1.version,
        "training_time": version1.training_time_seconds,
        "final_loss": version1.final_loss,
        "model_path": stage1_model_path
    }
    
    # Stage 2: Domain Adaptation
    logger.info("\n" + "="*60)
    logger.info("STAGE 2: Domain Adaptation Training")
    logger.info("="*60)
    
    # Load documents
    documents = load_documents_from_folder(documents_folder, include_html=True)
    logger.info(f"Loaded {len(documents)} documents for Stage 2")
    
    stage2 = DomainAdaptation(
        model_name=tuning_config.model_name,
        config=tuning_config,
        stage_1_model_path=stage1_model_path
    )
    
    version2 = stage2.train_stage_2(
        documents=documents,
        questions_per_doc=tuning_config.simrag_questions_per_doc,
        min_context_score=tuning_config.simrag_min_context_score
    )
    
    if not version2:
        raise Exception("Stage 2 training failed")
    
    results["stage2"] = {
        "version": version2.version,
        "training_time": version2.training_time_seconds,
        "final_loss": version2.final_loss,
        "model_path": stage2.get_model_from_registry(version2.version),
        "num_documents": len(documents)
    }
    
    # Testing: Evaluate on test questions
    logger.info("\n" + "="*60)
    logger.info("TESTING: Evaluating SimRAG Performance")
    logger.info("="*60)
    
    # Initialize RAG with documents - use local model (Ollama) for testing
    rag = BasicRAG(collection_name="simrag_test", use_persistent=False, force_provider="ollama")
    rag.add_documents(documents)
    
    # Test performance
    test_results = stage2.test_stage_2_performance(rag, test_questions)
    results["testing"] = test_results
    
    logger.info("\n" + "="*60)
    logger.info("Full Pipeline Complete!")
    logger.info("="*60)
    logger.info(f"Stage 1: {results['stage1']['version']} ({results['stage1']['training_time']:.1f}s)")
    logger.info(f"Stage 2: {results['stage2']['version']} ({results['stage2']['training_time']:.1f}s)")
    logger.info(f"Testing: Avg context score = {test_results.get('avg_context_score', 0):.3f}")
    
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full SimRAG pipeline")
    parser.add_argument("--documents", type=str, default="../HTML_DOCS",
                       help="Path to documents folder (default: ../HTML_DOCS)")
    parser.add_argument("--test-data", action="store_true",
                       help="Use test data for Stage 1 instead of Alpaca")
    parser.add_argument("--output", type=str, default="full_pipeline_results.json",
                       help="Output filename (default: full_pipeline_results.json, auto-timestamped)")
    parser.add_argument("--no-timestamp", action="store_true",
                       help="Don't add timestamp to filename (may overwrite existing)")
    
    args = parser.parse_args()
    
    try:
        results = run_full_pipeline(
            documents_folder=args.documents,
            use_real_datasets=not args.test_data,
            output_file=args.output,
            use_timestamp=not args.no_timestamp
        )
        print("\n=== Full Pipeline Complete ===")
        saved_file = results.get("_saved_filename", args.output)
        print(f"Results saved to: {saved_file}")
    except Exception as e:
        logger.error(f"Full pipeline failed: {e}")
        sys.exit(1)

