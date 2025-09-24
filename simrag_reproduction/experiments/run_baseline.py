"""
Baseline RAG Experiment Runner

Runs experiments with the baseline RAG system and saves results.
"""

import argparse
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import torch
from loguru import logger

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from baseline.rag_system import BaselineRAGSystem
from evaluation.metrics import compute_metrics, format_metrics_report
from data.dataset import load_dataset, prepare_qa_data


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging():
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )


def run_baseline_experiment(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Run baseline RAG experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info("Starting baseline RAG experiment")
    
    # Initialize system
    rag_system = BaselineRAGSystem(config)
    
    # Load and prepare data
    logger.info("Loading dataset")
    documents = load_dataset(config["data"]["corpus_size"])
    
    # Add documents to system
    rag_system.add_documents(documents)
    
    # Prepare QA data
    qa_data = prepare_qa_data(documents, config["evaluation"]["test_size"])
    
    # Run evaluation
    logger.info("Running evaluation")
    start_time = time.time()
    
    predictions = []
    references = []
    retrieved_docs = []
    relevant_docs = []
    
    for item in qa_data:
        # Get answer from RAG system
        result = rag_system.answer(item["question"])
        predictions.append(result["answer"])
        references.append(item["answer"])
        retrieved_docs.append(result["retrieved_documents"])
        
        # For this demo, we'll use a simple relevance scoring
        # In practice, you'd have ground truth relevant documents
        relevant_docs.append([doc["index"] for doc in result["retrieved_documents"][:3]])
    
    evaluation_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_metrics(
        predictions, 
        references, 
        retrieved_docs, 
        relevant_docs,
        k_values=config["evaluation"]["k_values"]
    )
    
    # Add timing information
    metrics["evaluation_time"] = evaluation_time
    metrics["avg_time_per_query"] = evaluation_time / len(qa_data)
    
    # Get hardware info
    if torch.cuda.is_available():
        metrics["gpu_memory_used_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        metrics["gpu_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024 / 1024
    
    logger.info("Baseline experiment completed")
    return {
        "config": config,
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
        "num_queries": len(qa_data),
        "num_documents": len(documents)
    }


def save_results(results: Dict[str, Any], output_dir: Path):
    """Save experiment results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=2)
    
    # Save detailed results
    detailed_results = {
        "config": results["config"],
        "metrics": results["metrics"],
        "num_queries": results["num_queries"],
        "num_documents": results["num_documents"]
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save predictions and references
    qa_results = []
    for pred, ref in zip(results["predictions"], results["references"]):
        qa_results.append({
            "prediction": pred,
            "reference": ref
        })
    
    with open(output_dir / "qa_results.json", "w") as f:
        json.dump(qa_results, f, indent=2)
    
    # Save formatted report
    report = format_metrics_report(results["metrics"], "Baseline RAG Results")
    with open(output_dir / "report.txt", "w") as f:
        f.write(report)
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run baseline RAG experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output", type=str, default="./results/baseline", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    if args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    
    # Run experiment
    try:
        results = run_baseline_experiment(config, output_dir)
        save_results(results, output_dir)
        
        # Print summary
        print(format_metrics_report(results["metrics"], "Baseline RAG Results"))
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
