"""
SimRAG Experiment Runner

Runs experiments with the SimRAG system and saves results.
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

from simrag.simrag_system import SimRAGSystem
from evaluation.metrics import compute_metrics, compute_improvement_metrics, format_metrics_report
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


def run_simrag_experiment(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Run SimRAG experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info("Starting SimRAG experiment")
    
    # Initialize system
    simrag_system = SimRAGSystem(config)
    
    # Load and prepare data
    logger.info("Loading dataset")
    documents = load_dataset(config["data"]["corpus_size"])
    
    # Add documents to system
    simrag_system.add_documents(documents)
    
    # Prepare QA data
    qa_data = prepare_qa_data(documents, config["evaluation"]["test_size"])
    
    # Split data for training and evaluation
    split_idx = int(len(qa_data) * 0.8)
    train_data = qa_data[:split_idx]
    eval_data = qa_data[split_idx:]
    
    # Run self-improvement loop
    logger.info("Starting self-improvement loop")
    start_time = time.time()
    
    improvement_history = simrag_system.run_self_improvement_loop(
        train_data, 
        max_iterations=config["self_improvement"]["num_iterations"]
    )
    
    improvement_time = time.time() - start_time
    
    # Final evaluation
    logger.info("Running final evaluation")
    eval_start_time = time.time()
    
    predictions = []
    references = []
    retrieved_docs = []
    relevant_docs = []
    
    for item in eval_data:
        # Get answer from SimRAG system
        result = simrag_system.answer(item["question"])
        predictions.append(result["answer"])
        references.append(item["answer"])
        retrieved_docs.append(result["retrieved_documents"])
        
        # For this demo, we'll use a simple relevance scoring
        relevant_docs.append([doc["index"] for doc in result["retrieved_documents"][:3]])
    
    evaluation_time = time.time() - eval_start_time
    
    # Compute final metrics
    final_metrics = compute_metrics(
        predictions, 
        references, 
        retrieved_docs, 
        relevant_docs,
        k_values=config["evaluation"]["k_values"]
    )
    
    # Add timing information
    final_metrics["total_improvement_time"] = improvement_time
    final_metrics["evaluation_time"] = evaluation_time
    final_metrics["avg_time_per_query"] = evaluation_time / len(eval_data)
    
    # Get hardware info
    if torch.cuda.is_available():
        final_metrics["gpu_memory_used_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        final_metrics["gpu_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024 / 1024
    
    logger.info("SimRAG experiment completed")
    return {
        "config": config,
        "final_metrics": final_metrics,
        "improvement_history": improvement_history,
        "predictions": predictions,
        "references": references,
        "num_queries": len(eval_data),
        "num_documents": len(documents),
        "num_training_pairs": len(train_data)
    }


def save_results(results: Dict[str, Any], output_dir: Path):
    """Save experiment results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final metrics
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(results["final_metrics"], f, indent=2)
    
    # Save improvement history
    with open(output_dir / "improvement_history.json", "w") as f:
        json.dump(results["improvement_history"], f, indent=2)
    
    # Save detailed results
    detailed_results = {
        "config": results["config"],
        "final_metrics": results["final_metrics"],
        "num_queries": results["num_queries"],
        "num_documents": results["num_documents"],
        "num_training_pairs": results["num_training_pairs"]
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
    report = format_metrics_report(results["final_metrics"], "SimRAG Results")
    with open(output_dir / "report.txt", "w") as f:
        f.write(report)
    
    # Save improvement summary
    improvement_summary = "SimRAG Improvement Summary\n"
    improvement_summary += "=" * 30 + "\n\n"
    
    for i, improvement in enumerate(results["improvement_history"]):
        improvement_summary += f"Iteration {i + 1}:\n"
        for metric, value in improvement.items():
            improvement_summary += f"  {metric}: {value:.4f}\n"
        improvement_summary += "\n"
    
    with open(output_dir / "improvement_summary.txt", "w") as f:
        f.write(improvement_summary)
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run SimRAG experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output", type=str, default="./results/simrag", help="Output directory")
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
        results = run_simrag_experiment(config, output_dir)
        save_results(results, output_dir)
        
        # Print summary
        print(format_metrics_report(results["final_metrics"], "SimRAG Results"))
        
        # Print improvement summary
        print("\nImprovement History:")
        for i, improvement in enumerate(results["improvement_history"]):
            print(f"Iteration {i + 1}: {improvement}")
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
